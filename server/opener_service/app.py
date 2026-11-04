"""Opener service: the Spaghetti Lab analysis of a Guilty Gear Strive match video (lab_analyze.py: combo
openers, match facts, and the move transcript when the server has a GPU and OPENERS_TRANSCRIPT=1), on a CPU
server.

Not public: it trusts the caller (FGCBackend) to have authenticated the user. The backend sends
the shared secret in X-Openers-Secret and the user's id in X-Owner-Id; it also keeps the
per-user daily quota. This service keeps the CPU cap (see jobs.py).

    OPENERS_SECRET=... uvicorn --factory server.opener_service.app:make_app --host 127.0.0.1 --port 10490

POST   /jobs              multipart `video` -> 202 {id, status, position}; 429 / 503 + Retry-After
POST   /jobs/url          {"url", "filename"}: a presigned link (OPENERS_URL_HOSTS), fetched when the job starts
GET    /jobs/{id}         status, stage, position in the queue, error
GET    /jobs/{id}/result  lab_result.json of a finished job (spaghetti-lab-result/1, tools/ggst_logger/model/LAB_RESULT.md)
DELETE /jobs/{id}         cancel a queued or running job
GET    /health            queue load and capabilities (no secret needed)
"""
from __future__ import annotations

import asyncio
import hmac
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from pydantic import BaseModel

from .jobs import MAX_UPLOAD, Limits, Queue, Refused, allowed_url, probe

SECRET = os.environ.get("OPENERS_SECRET", "")
WORKDIR = Path(os.environ.get("OPENERS_WORKDIR", Path(__file__).resolve().parent / "work"))
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov"}


class UrlJob(BaseModel):
    url: str
    filename: str = "video.mp4"


def make_app(queue: Queue | None = None, secret: str | None = None) -> FastAPI:
    secret = SECRET if secret is None else secret
    if not secret:
        raise RuntimeError("OPENERS_SECRET is not set")

    @asynccontextmanager
    async def lifespan(app):
        app.state.queue = queue or Queue(WORKDIR, Limits.from_env())
        app.state.queue.start()
        asyncio.get_running_loop().run_in_executor(None, app.state.queue.capabilities)   # probe the GPU once, off the loop
        yield
        await app.state.queue.stop()

    app = FastAPI(title="Spaghetti Lab opener service", lifespan=lifespan, docs_url=None, redoc_url=None)

    def caller(x_openers_secret: str = Header(""), x_owner_id: str = Header("")) -> str:
        if not hmac.compare_digest(x_openers_secret.encode(), secret.encode()):
            raise HTTPException(401, "bad secret")
        if not x_owner_id or len(x_owner_id) > 64:
            raise HTTPException(400, "X-Owner-Id missing")
        return x_owner_id

    def owned(request: Request, job_id: str, owner: str) -> dict:
        view = request.app.state.queue.view(job_id)
        if view is None or view["owner"] != owner:
            raise HTTPException(404, "no such job")
        return view

    @app.exception_handler(Refused)
    async def refused(_, e: Refused):
        headers = {"Retry-After": str(e.retry_after)} if e.retry_after else None
        return JSONResponse({"detail": e.reason}, e.status, headers=headers)

    @app.get("/health")
    async def health(request: Request):
        q: Queue = request.app.state.queue
        return dict(q.stats(), capabilities=await asyncio.to_thread(q.capabilities))

    @app.post("/jobs", status_code=202)
    async def create(request: Request, video: UploadFile, owner: str = Depends(caller)):
        q: Queue = request.app.state.queue
        q.check(owner)                                       # refuse before taking the upload
        ext = Path(video.filename or "").suffix.lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(415, f"expected a video file ({', '.join(sorted(VIDEO_EXT))})")
        job_id, d = q.new_dir()
        path = d / f"video{ext}"
        size = 0
        try:
            with open(path, "wb") as f:
                while chunk := await video.read(2**20):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, f"the video is over {MAX_UPLOAD // 2**20} MB")
                    f.write(chunk)
            meta = probe(path)
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        meta["filename"] = (video.filename or "")[:200]
        return q.submit(job_id, owner, path, meta)

    @app.post("/jobs/url", status_code=202)
    async def create_from_url(request: Request, body: UrlJob, owner: str = Depends(caller)):
        q: Queue = request.app.state.queue
        if not allowed_url(body.url):
            raise HTTPException(400, "the url's host is not allowed")
        ext = Path(body.filename).suffix.lower() or ".mp4"
        if ext not in VIDEO_EXT:
            raise HTTPException(415, f"expected a video file ({', '.join(sorted(VIDEO_EXT))})")
        q.check(owner)
        job_id, d = q.new_dir()
        return q.submit(job_id, owner, d / f"video{ext}", {"filename": body.filename[:200]}, url=body.url)

    @app.get("/jobs/{job_id}")
    async def status(request: Request, job_id: str, owner: str = Depends(caller)):
        return owned(request, job_id, owner)

    @app.get("/jobs/{job_id}/result")
    async def result(request: Request, job_id: str, owner: str = Depends(caller)):
        owned(request, job_id, owner)
        p = request.app.state.queue.result_path(job_id)
        if p is None:
            raise HTTPException(409, "the job has no result (yet)")
        return FileResponse(p, media_type="application/json")

    @app.delete("/jobs/{job_id}")
    async def cancel(request: Request, job_id: str, owner: str = Depends(caller)):
        owned(request, job_id, owner)
        if not await request.app.state.queue.cancel(job_id):
            raise HTTPException(409, "the job has already ended")
        return {"cancelled": True}

    return app
