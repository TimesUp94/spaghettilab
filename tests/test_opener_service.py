"""Opener service: the CPU cap (concurrency, queue length, per-owner limit), cancel, restart, TTL,
the HTTP layer's auth / ownership, stages, capabilities and the lab_result passthrough. The chain is
replaced by a stub script, so no model runs."""
import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from server.opener_service import jobs as jobs_mod  # noqa: E402
from server.opener_service.jobs import CANCELLED, DONE, FAILED, QUEUED, RUNNING, Limits, Queue, Refused  # noqa: E402

# lab_analyze's output: stage lines, openers.json and lab_result.json (sys.argv[5] = transcript on/off)
STUB = """
import json, sys, time, pathlib
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
print("stage: hud", flush=True)
print("running the CV pipeline ...", flush=True)
time.sleep(float(sys.argv[3]))
if sys.argv[4] == "fail":
    sys.exit(1)
print("stage: detect", flush=True)
print("3 combo starts found", flush=True)
print("stage: moves", flush=True)
ops = {"characters": {"P1": "SO", "P2": "KY"}, "openers": [{}, {}]}
(out / "openers.json").write_text(json.dumps(ops))
tr = sys.argv[5] == "on"
if tr:
    print("stage: transcript", flush=True)
print("stage: facts", flush=True)
lab = dict(schema="spaghetti-lab-result/1", characters=ops["characters"], openers=ops["openers"],
           capabilities={"openers": True, "transcript": tr, "facts": True},
           transcript={"events": [{}, {}, {}]} if tr else None, states=None, facts={"md": {"it": "", "en": ""}},
           prompts={"it": "", "en": ""}, timings_s={})
(out / "lab_result.json").write_text(json.dumps(lab))
print("stage: done", flush=True)
"""


def stub_command(tmp_path, seconds=0.3, outcome="ok", seen=None):
    script = tmp_path / "stub.py"
    script.write_text(STUB)

    def cmd(video, out, limits, transcript=False, title=None):
        if seen is not None:
            seen.append((transcript, title))
        return [sys.executable, str(script), str(video), str(out), str(seconds), outcome, "on" if transcript else "off"]
    return cmd


CPU = {"openers": True, "transcript": False, "facts": True}
GPU = {"openers": True, "transcript": True, "facts": True}


def new_job(q, owner):
    job_id, d = q.new_dir()
    (d / "video.mp4").write_bytes(b"x")
    return q.submit(job_id, owner, d / "video.mp4")


async def wait_until(pred, timeout=10):
    t = time.time()
    while not pred():
        assert time.time() - t < timeout, "timed out"
        await asyncio.sleep(0.05)


def test_concurrency_cap_and_queue(tmp_path):
    async def go():
        q = Queue(tmp_path / "w", Limits(max_concurrent=1, max_queue=2, per_owner=1), stub_command(tmp_path, 0.5), caps=CPU)
        q.start()
        a = new_job(q, "a")
        await wait_until(lambda: q.jobs[a["id"]].status == RUNNING)
        b, c = new_job(q, "b"), new_job(q, "c")
        assert q.stats() == dict(running=1, queued=2, max_concurrent=1, max_queue=2)
        assert q.view(b["id"])["position"] == 1 and q.view(c["id"])["position"] == 2
        with pytest.raises(Refused) as e:              # queue full
            q.check("d")
        assert e.value.status == 503
        with pytest.raises(Refused) as e:              # one active job per owner
            q.check("a")
        assert e.value.status == 429
        # never more than one running at a time
        peak = 0
        while any(j.status in (QUEUED, RUNNING) for j in q.jobs.values()):
            peak = max(peak, q.stats()["running"])
            await asyncio.sleep(0.02)
        assert peak == 1
        assert all(j.status == DONE for j in q.jobs.values())
        assert q.jobs[a["id"]].meta["openers"] == 2
        assert q.result_path(a["id"]).exists()
        assert not (tmp_path / "w" / a["id"] / "video.mp4").exists()     # upload dropped
        await q.stop()
    asyncio.run(go())


def test_failure_cancel_and_timeout(tmp_path):
    async def go():
        q = Queue(tmp_path / "w", Limits(max_concurrent=1, per_owner=5, timeout_s=1), stub_command(tmp_path, 3), caps=CPU)
        q.start()
        slow = new_job(q, "a")
        queued = new_job(q, "a")
        assert await q.cancel(queued["id"])
        await wait_until(lambda: q.jobs[slow["id"]].status == FAILED)
        assert "longer" in q.jobs[slow["id"]].error
        assert q.jobs[queued["id"]].status == CANCELLED
        q.command = stub_command(tmp_path, 0.1, "fail")
        bad = new_job(q, "a")
        await wait_until(lambda: q.jobs[bad["id"]].status == FAILED)
        assert q.result_path(bad["id"]) is None
        await q.stop()
    asyncio.run(go())


def test_restart_and_ttl(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    for jid, status in (("done1", DONE), ("run1", RUNNING)):
        (w / jid).mkdir()
        (w / jid / "job.json").write_text(json.dumps(dict(id=jid, owner="a", video="v", created=1.0, status=status,
                                                          finished=time.time() if status == DONE else None)))
    q = Queue(w, Limits(ttl_s=60))
    assert q.jobs["done1"].status == DONE
    assert q.jobs["run1"].status == FAILED and "restarted" in q.jobs["run1"].error
    q.sweep(now=time.time() + 120)
    assert not q.jobs and not (w / "done1").exists()


def test_http_auth_and_ownership(tmp_path, monkeypatch):
    from server.opener_service import app as app_mod
    monkeypatch.setattr(app_mod, "probe", lambda p: {"duration_s": 1.0})
    q = Queue(tmp_path / "w", Limits(max_concurrent=1, per_owner=1), stub_command(tmp_path, 0.2), caps=CPU)
    with TestClient(app_mod.make_app(q, secret="s3cret")) as client:
        h = {"X-Openers-Secret": "s3cret", "X-Owner-Id": "alice"}
        files = {"video": ("match.mp4", b"0" * 10, "video/mp4")}
        assert client.post("/jobs", files=files, headers={**h, "X-Openers-Secret": "nope"}).status_code == 401
        assert client.post("/jobs", files={"video": ("x.txt", b"0", "text/plain")}, headers=h).status_code == 415
        r = client.post("/jobs", files=files, headers=h)
        assert r.status_code == 202, r.text
        jid = r.json()["id"]
        r2 = client.post("/jobs", files=files, headers=h)
        assert r2.status_code == 429 and r2.headers["Retry-After"]
        assert client.get(f"/jobs/{jid}", headers={**h, "X-Owner-Id": "bob"}).status_code == 404
        t = time.time()
        while client.get(f"/jobs/{jid}", headers=h).json()["status"] != DONE:
            assert time.time() - t < 10
            time.sleep(0.05)
        doc = client.get(f"/jobs/{jid}/result", headers=h).json()
        assert len(doc["openers"]) == 2                          # the old field is still there
        assert doc["schema"] == "spaghetti-lab-result/1" and doc["transcript"] is None
        assert doc["capabilities"] == CPU
        health = client.get("/health").json()
        assert health["running"] == 0 and health["capabilities"] == CPU


def test_url_jobs(tmp_path, monkeypatch):
    from server.opener_service import jobs
    monkeypatch.setenv("OPENERS_URL_HOSTS", "fra1.digitaloceanspaces.com")
    assert jobs.allowed_url("https://spaghetti.fra1.digitaloceanspaces.com/u/x.mp4?X-Amz-Signature=1")
    assert not jobs.allowed_url("http://spaghetti.fra1.digitaloceanspaces.com/x.mp4")      # not https
    assert not jobs.allowed_url("https://127.0.0.1/x.mp4")
    assert not jobs.allowed_url("https://evil-fra1.digitaloceanspaces.com.example.org/x.mp4")
    fetched = []
    monkeypatch.setattr(jobs, "fetch", lambda url, dest: (fetched.append(url), dest.write_bytes(b"v")))
    monkeypatch.setattr(jobs, "probe", lambda p: {"duration_s": 2.0})

    async def go():
        q = Queue(tmp_path / "w", Limits(), stub_command(tmp_path, 0.1), caps=CPU)
        q.start()
        job_id, d = q.new_dir()
        v = q.submit(job_id, "a", d / "video.mp4", url="https://x.fra1.digitaloceanspaces.com/v.mp4")
        assert "url" not in v                                        # the presigned link is never shown
        await wait_until(lambda: q.jobs[job_id].status == DONE)
        assert fetched and q.jobs[job_id].meta["duration_s"] == 2.0
        await q.stop()
    asyncio.run(go())


def test_stages_and_transcript_capability(tmp_path):
    """The transcript runs only when the service has it; the stage follows lab_analyze's `stage:` lines."""
    async def go():
        seen = []
        q = Queue(tmp_path / "w", Limits(per_owner=5), stub_command(tmp_path, 0.6, seen=seen), caps=GPU)
        q.start()
        job_id, d = q.new_dir()
        (d / "video.mp4").write_bytes(b"x")
        a = q.submit(job_id, "a", d / "video.mp4", {"filename": "Sol vs Ky.mp4"})
        await wait_until(lambda: q.jobs[a["id"]].stage == "hud")
        await wait_until(lambda: q.jobs[a["id"]].status == DONE)
        job = q.jobs[a["id"]]
        assert job.stage == "done" and seen == [(True, "Sol vs Ky.mp4")]       # the user's name, not the upload path
        assert job.meta["capabilities"] == GPU and job.meta["events"] == 3 and job.meta["openers"] == 2
        assert q.result_path(a["id"]).name == "lab_result.json"
        out = tmp_path / "w" / a["id"] / "out"
        assert sorted(p.name for p in out.iterdir()) == ["lab_result.json", "openers.json"]
        await q.stop()
    asyncio.run(go())


def test_old_results_still_served(tmp_path):
    """A job finished before lab_analyze has only openers.json: that is its result."""
    w = tmp_path / "w"
    (w / "old1" / "out").mkdir(parents=True)
    (w / "old1" / "out" / "openers.json").write_text(json.dumps({"openers": []}))
    (w / "old1" / "job.json").write_text(json.dumps(dict(id="old1", owner="a", video="v", created=1.0, status=DONE,
                                                         finished=time.time())))
    q = Queue(w, Limits(), caps=CPU)
    assert q.result_path("old1").name == "openers.json"


def test_capabilities_and_command(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_mod, "gpu_available", lambda: True)
    monkeypatch.delenv("GGST_DEVICE", raising=False)
    monkeypatch.setenv("GGST_MODEL_DIR", str(tmp_path))
    assert jobs_mod.capabilities(Limits(transcript=False)) == CPU          # OPENERS_TRANSCRIPT unset: never
    assert jobs_mod.capabilities(Limits(transcript=True))["transcript"] is False   # no weights
    for f in jobs_mod.TRANSCRIPT_WEIGHTS:
        (tmp_path / f).write_bytes(b"")
    assert jobs_mod.capabilities(Limits(transcript=True)) == GPU
    monkeypatch.setattr(jobs_mod, "gpu_available", lambda: False)
    assert jobs_mod.capabilities(Limits(transcript=True)) == CPU            # no GPU
    monkeypatch.setenv("OPENERS_TRANSCRIPT", "1")
    assert Limits.from_env().transcript
    cpu = jobs_mod.chain_command(Path("v.mp4"), Path("o"), Limits(light=True))
    assert cpu[1].endswith("lab_analyze.py") and cpu[cpu.index("--transcript") + 1] == "off"
    assert cpu[cpu.index("--device") + 1] == "cpu" and "--light" in cpu
    gpu = jobs_mod.chain_command(Path("v.mp4"), Path("o"), Limits(light=False), transcript=True)
    assert gpu[gpu.index("--transcript") + 1] == "on" and gpu[gpu.index("--device") + 1] == "cuda" and "--light" not in gpu
    assert "--title" not in gpu
    named = jobs_mod.chain_command(Path("v.mp4"), Path("o"), Limits(), title="match 1.mp4")
    assert named[named.index("--title") + 1] == "match 1.mp4"
