# Opener service

Runs the opener chain (`tools/ggst_logger/model/recognize.py`: combo starts, characters, the
opening move, counter/punish) on uploaded Guilty Gear Strive videos, on a server **without a GPU**.
It sits behind FGCBackend, which authenticates the user and keeps the daily quota; the service
itself is bound to localhost and keeps the CPU cap.

```
web / app ──JWT──▶ FGCBackend /api/spaghetti-lab/openers ──secret──▶ opener service :10490 ──▶ recognize.py (subprocess)
```

## The CPU cap

| Setting (env) | Default | Meaning |
|---|---|---|
| `OPENERS_MAX_CONCURRENT` | 1 | analyses running at once; the rest wait |
| `OPENERS_MAX_QUEUE` | 8 | analyses waiting; past it a new request gets 503 + `Retry-After` |
| `OPENERS_PER_OWNER` | 1 | queued or running analyses per user (429) |
| `OPENERS_THREADS` | 2 | threads per analysis (torch, OpenMP, BLAS) |
| `OPENERS_CPUS` | all | e.g. `2,3`: cores an analysis is pinned to (Linux), with ffmpeg and the HUD pipeline |
| `OPENERS_TIMEOUT_S` | 1800 | an analysis is killed past this |
| `OPENERS_LIGHT` | 1 | one move net, no frame-shift averaging (~1/3 of the CPU, ~1–2 points of accuracy) |
| `OPENERS_MAX_UPLOAD_MB` | 600 | upload size |
| `OPENERS_MAX_DURATION_S` | 900 | video length (one set, not a whole VOD) |
| `OPENERS_TTL_H` | 24 | results are kept this long |
| `OPENERS_URL_HOSTS` | none | hosts `/jobs/url` may fetch from (https only, no redirects), e.g. `fra1.digitaloceanspaces.com` |

Analyses run niced (+10). With the defaults the service uses at most 2 cores and about 2–3 GB of
RAM at any time, whatever the number of requests. The upload and the intermediate crops are
deleted as soon as an analysis ends; only `openers.json` stays until the TTL.

Measured on the development PC (CPU only, networks alone): about 1.1 s per opener light and 3 s
full, so a 3-minute match (~30 openers) takes about 30 s to 1.5 min plus the HUD pipeline and
decoding. A 4-core VPS core is slower: count on 1–3 min per match.

## Install (Linux VPS)

```bash
git clone <replanal repo> /opt/replanal && cd /opt/replanal
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[server]"
sudo apt install ffmpeg
# the weights are not in the repo: copy them from the training PC
mkdir -p /opt/replanal-models
scp <pc>:G:/Projects/replanal-data/ggst_dataset/model/{video_openers_final.pt,video_players_final.pt,video_moves_final.pt,banner_reader_final.pkl} /opt/replanal-models/
```

Run it with PM2 next to the other services (`ecosystem.config.js` here; set `OPENERS_SECRET` to a
long random string, the same as FGCBackend's `HERC_OPENERS_SECRET`):

```bash
OPENERS_SECRET=... pm2 start server/opener_service/ecosystem.config.js && pm2 save
curl http://127.0.0.1:10490/health
```

Do not expose port 10490 through nginx: only FGCBackend talks to it.

## API

All but `/health` need `X-Openers-Secret` and `X-Owner-Id` (the user's id; jobs are visible only
to their owner).

- `POST /jobs`, multipart field `video` (.mp4/.mkv/.webm/.mov) → `202 {id, status: "queued", position, meta}`;
  `429` (the user already has one), `503` (queue full), `413` (too big/long), `415`, `422` (not a video, under 360p).
- `POST /jobs/url`, JSON `{url, filename}`: a presigned link (FGCBackend sends the user's Spaghetti Lab file in Spaces). The link is fetched when the job's turn comes, so queued jobs take no disk; it is never echoed back.
- `GET /jobs/{id}` → `{status: queued|running|done|failed|cancelled, stage: download|hud|detect|moves|done, position, error, meta}`.
- `GET /jobs/{id}/result` → `openers.json`: `characters`, and per opener `time_s`, `attacker`,
  `attacker_char`, `category`, `move` (+ `move_top3`), `normal`, `hit_kind` (counter/punish/none), and the probabilities.
- `DELETE /jobs/{id}` → cancel.
- `GET /health` → `{running, queued, max_concurrent, max_queue}`.

Tests: `pytest tests/test_opener_service.py` (the chain is stubbed; no model runs).
