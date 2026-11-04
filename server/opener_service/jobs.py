"""Job queue of the opener service: at most `max_concurrent` analyses run at once, at most
`max_queue` wait, and each owner has at most `per_owner` jobs queued or running. Everything past
that is refused at once (the caller gets a retry hint) instead of piling up: the server's CPU is
shared with the rest of the platform, so the cap is the point.

Each job runs as a subprocess (`lab_analyze.py`: the opener chain, the facts, and the match transcript when
the service has it, see `capabilities`), niced, pinned to `cpus` when
given (Linux: the pin is inherited by ffmpeg and the pipeline too) and limited to `threads`
threads, and is killed past `timeout_s`. Job state is kept in memory and in `<workdir>/<id>/job.json`,
so finished results survive a restart; jobs that were queued or running at a restart fail.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
ACTIVE = (QUEUED, RUNNING)

# lab_analyze.py prints `stage: <name>`; the older recognize.py markers are still understood
STAGE_PREFIX = "stage: "
STAGE_NAMES = ("hud", "detect", "moves", "transcript", "facts", "done")
STAGES = [("running the CV pipeline", "hud"), ("combo starts found", "moves"), ("openers ->", "done")]
RESULT_FILES = ("lab_result.json", "openers.json")      # what a job keeps; the first that exists is the result


MAX_DURATION_S = int(os.environ.get("OPENERS_MAX_DURATION_S", 900))
MAX_UPLOAD = int(float(os.environ.get("OPENERS_MAX_UPLOAD_MB", 600)) * 2**20)


class Refused(Exception):
    """The job can't be taken now; `status` is the HTTP status to answer, `retry_after` seconds."""
    def __init__(self, status: int, reason: str, retry_after: int | None = None):
        super().__init__(reason)
        self.status, self.reason, self.retry_after = status, reason, retry_after


@dataclass
class Limits:
    max_concurrent: int = 1
    max_queue: int = 8
    per_owner: int = 1
    threads: int = 2
    cpus: list[int] | None = None
    timeout_s: int = 1800
    ttl_s: int = 24 * 3600
    light: bool = True
    transcript: bool = False        # OPENERS_TRANSCRIPT=1: run the transcript when there is a GPU

    @classmethod
    def from_env(cls) -> "Limits":
        e = os.environ.get
        cpus = e("OPENERS_CPUS")
        return cls(max_concurrent=int(e("OPENERS_MAX_CONCURRENT", 1)), max_queue=int(e("OPENERS_MAX_QUEUE", 8)),
                   per_owner=int(e("OPENERS_PER_OWNER", 1)), threads=int(e("OPENERS_THREADS", 2)),
                   cpus=[int(c) for c in cpus.split(",")] if cpus else None,
                   timeout_s=int(e("OPENERS_TIMEOUT_S", 1800)), ttl_s=int(float(e("OPENERS_TTL_H", 24)) * 3600),
                   light=e("OPENERS_LIGHT", "1") != "0", transcript=e("OPENERS_TRANSCRIPT", "0") == "1")


@dataclass
class Job:
    id: str
    owner: str
    video: str
    created: float
    status: str = QUEUED
    stage: str | None = None
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)
    url: str | None = None          # fetched when the job starts (a presigned link: never shown)

    def public(self, position: int | None = None) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("video", "url")}
        if position is not None:
            d["position"] = position
        return d


MODEL_DIR = Path(__file__).resolve().parents[2] / "tools" / "ggst_logger" / "model"
TRANSCRIPT_WEIGHTS = ("video_transcript_exp_ens5grid.pt", "video_players_test.pt", "state_embed_exp_se_clip_sup.pt",
                      "transcript_grid_pca.npz")
_GPU: bool | None = None


def gpu_available() -> bool:
    """A CUDA GPU torch can use (checked once, in a subprocess: the service itself never imports torch)."""
    global _GPU
    if _GPU is None:
        import subprocess
        try:
            r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                               capture_output=True, text=True, timeout=120)
            _GPU = r.stdout.strip().endswith("True")
        except Exception:
            _GPU = False
    return _GPU


def transcript_weights() -> bool:
    """The transcript's weights are in GGST_MODEL_DIR. Without GGST_MODEL_DIR (the development PC's data
    folder) lab_analyze checks for itself and falls back to no transcript."""
    d = os.environ.get("GGST_MODEL_DIR")
    return not d or all((Path(d) / f).exists() for f in TRANSCRIPT_WEIGHTS)


def capabilities(limits: Limits) -> dict:
    """What a job of this service produces (lab_result.json `capabilities`): the transcript only with
    OPENERS_TRANSCRIPT=1, its weights and a GPU."""
    return dict(openers=True, transcript=bool(limits.transcript and transcript_weights() and gpu_available()),
                facts=True)


def chain_command(video: Path, out: Path, limits: Limits, transcript: bool = False) -> list[str]:
    """lab_analyze for one video: openers (light unless OPENERS_LIGHT=0) and facts, on the CPU; with the
    transcript, everything on the GPU (GGST_DEVICE overrides)."""
    dev = os.environ.get("GGST_DEVICE") or ("cuda" if transcript else "cpu")
    cmd = [sys.executable, str(MODEL_DIR / "lab_analyze.py"), str(video), "--out", str(out), "--device", dev,
           "--threads", str(limits.threads), "--transcript", "on" if transcript else "off"]
    return cmd + (["--light"] if limits.light else [])


class Queue:
    def __init__(self, workdir: Path, limits: Limits, command=chain_command, caps: dict | None = None):
        """command(video, out, limits, transcript) -> argv; caps: the capabilities (default: probed once)."""
        self.workdir, self.limits, self.command = Path(workdir), limits, command
        self._caps = caps
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.waiting: asyncio.Queue[str] = asyncio.Queue()
        self.procs: dict[str, asyncio.subprocess.Process] = {}
        self.workers: list[asyncio.Task] = []
        self._load()

    # ---- state -----------------------------------------------------------------------------

    def _load(self):
        """Jobs from an earlier run: finished ones keep their result, unfinished ones failed."""
        for f in self.workdir.glob("*/job.json"):
            try:
                job = Job(**json.loads(f.read_text()))
            except Exception:
                continue
            if job.status in ACTIVE:
                job.status, job.error, job.finished = FAILED, "the service restarted", time.time()
                self._save(job)
            self.jobs[job.id] = job

    def _save(self, job: Job):
        d = self.workdir / job.id
        d.mkdir(exist_ok=True)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(job)))
        tmp.replace(d / "job.json")

    def queued_ids(self) -> list[str]:
        return [j.id for j in sorted(self.jobs.values(), key=lambda j: j.created) if j.status == QUEUED]

    def stats(self) -> dict:
        n = {s: 0 for s in (QUEUED, RUNNING)}
        for j in self.jobs.values():
            if j.status in n:
                n[j.status] += 1
        return dict(running=n[RUNNING], queued=n[QUEUED], max_concurrent=self.limits.max_concurrent,
                    max_queue=self.limits.max_queue)

    def capabilities(self) -> dict:
        if self._caps is None:
            self._caps = capabilities(self.limits)
        return dict(self._caps)

    def view(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        q = self.queued_ids()
        return job.public(q.index(job.id) + 1 if job.id in q else None)

    def result_path(self, job_id: str) -> Path | None:
        """lab_result.json of a finished job (openers.json for a job finished before lab_analyze)."""
        job = self.jobs.get(job_id)
        if not job or job.status != DONE:
            return None
        return next((p for n in RESULT_FILES if (p := self.workdir / job_id / "out" / n).exists()), None)

    # ---- admission ---------------------------------------------------------------------------

    def check(self, owner: str):
        """Raise Refused when a new job of this owner can't be taken now."""
        mine = sum(1 for j in self.jobs.values() if j.owner == owner and j.status in ACTIVE)
        if mine >= self.limits.per_owner:
            raise Refused(429, "you already have an analysis queued or running", 60)
        if self.stats()["queued"] >= self.limits.max_queue:
            raise Refused(503, "the analysis queue is full, try again later", 300)

    def new_dir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex
        d = self.workdir / job_id
        d.mkdir()
        return job_id, d

    def submit(self, job_id: str, owner: str, video: Path, meta: dict | None = None, url: str | None = None) -> dict:
        """Queue an uploaded video (already under workdir/<job_id>/). Re-checks the limits: an
        upload can take long and others may have got in meanwhile."""
        try:
            self.check(owner)
        except Refused:
            shutil.rmtree(self.workdir / job_id, ignore_errors=True)
            raise
        job = Job(id=job_id, owner=owner, video=str(video), created=time.time(), meta=meta or {}, url=url)
        self.jobs[job_id] = job
        self._save(job)
        self.waiting.put_nowait(job_id)
        return self.view(job_id)

    async def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status not in ACTIVE:
            return False
        if job.status == RUNNING and job_id in self.procs:
            _kill(self.procs[job_id])
        job.status, job.finished, job.error = CANCELLED, time.time(), None
        self._save(job)
        self._drop_video(job)
        return True

    # ---- running -----------------------------------------------------------------------------

    def start(self):
        self.workers = [asyncio.create_task(self._worker()) for _ in range(self.limits.max_concurrent)]
        self.workers.append(asyncio.create_task(self._sweeper()))

    async def stop(self):
        for t in self.workers:
            t.cancel()
        for p in list(self.procs.values()):
            _kill(p)

    async def _worker(self):
        while True:
            job_id = await self.waiting.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != QUEUED:        # cancelled while waiting
                continue
            try:
                await self._run(job)
            except Exception as e:                          # never let one job stop the worker
                job.status, job.error = FAILED, f"internal error: {type(e).__name__}"
            job.finished = time.time()
            self._save(job)
            self._drop_video(job)

    async def _run(self, job: Job):
        job.status, job.started, job.stage = RUNNING, time.time(), "starting"
        self._save(job)
        if job.url:
            job.stage = "download"
            try:
                await asyncio.to_thread(fetch, job.url, Path(job.video))
                job.meta.update(await asyncio.to_thread(probe, Path(job.video)))
            except Refused as e:
                job.status, job.error = FAILED, e.reason
                return
            except Exception:
                job.status, job.error = FAILED, "could not download the video"
                return
            if job.status == CANCELLED:
                return
        out = self.workdir / job.id / "out"
        transcript = (await asyncio.to_thread(self.capabilities))["transcript"]
        log = open(self.workdir / job.id / "chain.log", "wb")
        env = dict(os.environ, OMP_NUM_THREADS=str(self.limits.threads), MKL_NUM_THREADS=str(self.limits.threads),
                   OPENBLAS_NUM_THREADS=str(self.limits.threads), VM_WORKERS="0", PYTHONUNBUFFERED="1",
                   PYTHONIOENCODING="utf-8")
        proc = await asyncio.create_subprocess_exec(
            *self.command(Path(job.video), out, self.limits, transcript), stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, env=env, preexec_fn=_limit_fn(self.limits) if os.name != "nt" else None,
            start_new_session=os.name != "nt")
        self.procs[job.id] = proc
        try:
            await asyncio.wait_for(self._follow(job, proc, log), self.limits.timeout_s)
            rc = await proc.wait()
        except asyncio.TimeoutError:
            _kill(proc)
            await proc.wait()
            job.status, job.error = FAILED, f"took longer than {self.limits.timeout_s // 60} min"
            return
        finally:
            self.procs.pop(job.id, None)
            log.close()
        if job.status == CANCELLED:
            return
        res = next((p for n in RESULT_FILES if (p := out / n).exists()), None)
        if rc == 0 and res is not None:
            job.status, job.stage = DONE, "done"
            try:
                doc = json.loads(res.read_text(encoding="utf-8"))
                job.meta.update(openers=len(doc.get("openers", [])), characters=doc.get("characters"))
                if "capabilities" in doc:
                    job.meta.update(capabilities=doc["capabilities"],
                                    events=len((doc.get("transcript") or {}).get("events", [])))
            except Exception:
                pass
        else:
            job.status, job.error = FAILED, "the analysis failed (is it a Guilty Gear Strive match video?)"

    async def _follow(self, job: Job, proc, log):
        async for line in proc.stdout:
            log.write(line)
            text = line.decode("utf-8", "replace").strip()
            if text.startswith(STAGE_PREFIX):
                stage = text[len(STAGE_PREFIX):].strip()
                if stage in STAGE_NAMES and stage != job.stage:
                    job.stage = stage
                    self._save(job)
                continue
            for marker, stage in STAGES:
                if marker in text:
                    job.stage = stage
                    self._save(job)

    def _drop_video(self, job: Job):
        """The upload and the intermediate crops go as soon as a job ends; only the result stays."""
        d = self.workdir / job.id
        for p in d.iterdir() if d.exists() else []:
            if p.name in ("job.json", "chain.log"):
                continue
            if p.name == "out":
                for q in p.iterdir():
                    if q.name not in RESULT_FILES:
                        shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
            else:
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)

    async def _sweeper(self, every_s: int = 600):
        while True:
            self.sweep()
            await asyncio.sleep(every_s)

    def sweep(self, now: float | None = None):
        """Forget finished jobs older than the TTL, and stray directories."""
        now = now or time.time()
        for job in list(self.jobs.values()):
            if job.status not in ACTIVE and job.finished and now - job.finished > self.limits.ttl_s:
                shutil.rmtree(self.workdir / job.id, ignore_errors=True)
                del self.jobs[job.id]
        for d in self.workdir.iterdir():
            if d.is_dir() and d.name not in self.jobs and now - d.stat().st_mtime > 3600:
                shutil.rmtree(d, ignore_errors=True)


def allowed_url(url: str) -> bool:
    """Only https links to the hosts in OPENERS_URL_HOSTS (e.g. fra1.digitaloceanspaces.com) are
    fetched, so the service can't be pointed at the local network."""
    from urllib.parse import urlparse
    hosts = [h.strip().lower() for h in os.environ.get("OPENERS_URL_HOSTS", "").split(",") if h.strip()]
    u = urlparse(url)
    host = (u.hostname or "").lower()
    return u.scheme == "https" and any(host == h or host.endswith("." + h) for h in hosts)


def fetch(url: str, dest: Path, max_bytes: int | None = None):
    """Download to dest, refusing more than max_bytes (default OPENERS_MAX_UPLOAD_MB)."""
    import urllib.request
    max_bytes = max_bytes or MAX_UPLOAD
    if not allowed_url(url):
        raise Refused(400, "the url's host is not allowed")
    class NoRedirect(urllib.request.HTTPRedirectHandler):     # a redirect could leave the allowed hosts
        def redirect_request(self, *args, **kwargs):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(url, timeout=60) as r, open(dest, "wb") as f:
        if int(r.headers.get("Content-Length") or 0) > max_bytes:
            raise Refused(413, f"the video is over {max_bytes // 2**20} MB")
        size = 0
        while chunk := r.read(2**20):
            size += len(chunk)
            if size > max_bytes:
                raise Refused(413, f"the video is over {max_bytes // 2**20} MB")
            f.write(chunk)


def probe(path: Path) -> dict:
    """Duration and size of a video; refuses what isn't a readable video, is too long or too small."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not fps or not n or not w:
        raise Refused(422, "not a readable video")
    duration = n / fps
    if duration > MAX_DURATION_S:
        raise Refused(413, f"the video is over {MAX_DURATION_S // 60} min: cut it to one match or set")
    if h < 360:
        raise Refused(422, "the video is below 360p")
    return {"duration_s": round(duration, 1), "fps": round(fps, 2), "size": [w, h]}


def _limit_fn(limits: Limits):
    def fn():
        os.nice(10)
        if limits.cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, limits.cpus)
    return fn


def _kill(proc):
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)       # the chain's ffmpeg / pipeline children too
    except (ProcessLookupError, PermissionError):
        pass
