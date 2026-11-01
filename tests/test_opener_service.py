"""Opener service: the CPU cap (concurrency, queue length, per-owner limit), cancel, restart, TTL,
and the HTTP layer's auth / ownership. The chain is replaced by a stub script, so no model runs."""
import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from server.opener_service.jobs import CANCELLED, DONE, FAILED, QUEUED, RUNNING, Limits, Queue, Refused  # noqa: E402

STUB = """
import json, sys, time, pathlib
out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
print("running the CV pipeline ...", flush=True)
time.sleep(float(sys.argv[3]))
if sys.argv[4] == "fail":
    sys.exit(1)
print("3 combo starts found", flush=True)
(out / "openers.json").write_text(json.dumps({"characters": {"P1": "SO", "P2": "KY"}, "openers": [{}, {}]}))
print("2 openers -> x", flush=True)
"""


def stub_command(tmp_path, seconds=0.3, outcome="ok"):
    script = tmp_path / "stub.py"
    script.write_text(STUB)

    def cmd(video, out, limits):
        return [sys.executable, str(script), str(video), str(out), str(seconds), outcome]
    return cmd


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
        q = Queue(tmp_path / "w", Limits(max_concurrent=1, max_queue=2, per_owner=1), stub_command(tmp_path, 0.5))
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
        q = Queue(tmp_path / "w", Limits(max_concurrent=1, per_owner=5, timeout_s=1), stub_command(tmp_path, 3))
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
    q = Queue(tmp_path / "w", Limits(max_concurrent=1, per_owner=1), stub_command(tmp_path, 0.2))
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
        assert len(doc["openers"]) == 2
        assert client.get("/health").json()["running"] == 0


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
        q = Queue(tmp_path / "w", Limits(), stub_command(tmp_path, 0.1))
        q.start()
        job_id, d = q.new_dir()
        v = q.submit(job_id, "a", d / "video.mp4", url="https://x.fra1.digitaloceanspaces.com/v.mp4")
        assert "url" not in v                                        # the presigned link is never shown
        await wait_until(lambda: q.jobs[job_id].status == DONE)
        assert fetched and q.jobs[job_id].meta["duration_s"] == 2.0
        await q.stop()
    asyncio.run(go())
