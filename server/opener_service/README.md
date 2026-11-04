# Opener service

Runs the Spaghetti Lab analysis (`tools/ggst_logger/model/lab_analyze.py`) on uploaded Guilty Gear Strive
videos, on a server **without a GPU**:
- the opener chain (`recognize.py`: combo starts, characters, the opening move, counter/punish);
- the match facts and the LLM prompts in Italian and English (`lang_summary.py`, pure Python; no LLM is called);
- the match transcript (every move of both players, with outcomes) and the per-tubelet state classes, only when
  the server has a GPU **and** `OPENERS_TRANSCRIPT=1` (off on the CPU VPS).

The result is `lab_result.json`, schema `spaghetti-lab-result/1`, described in
`tools/ggst_logger/model/LAB_RESULT.md` (sample: `tools/ggst_logger/model/fixtures/lab_result_sample.json`).
It sits behind FGCBackend, which authenticates the user and keeps the daily quota; the service
itself is bound to localhost and keeps the CPU cap.

```
web / app ──JWT──▶ FGCBackend /api/spaghetti-lab/openers ──secret──▶ opener service :10490 ──▶ lab_analyze.py (subprocess)
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
| `OPENERS_LIGHT` | 1 | one move net, no frame-shift averaging: ~1/3 of the move nets' CPU; whole chain 0.878 vs 0.888 on the 720p30 test copies (`degrade_eval.py eval --light`) |
| `OPENERS_MAX_UPLOAD_MB` | 600 | upload size |
| `OPENERS_MAX_DURATION_S` | 900 | video length (one set, not a whole VOD) |
| `OPENERS_TTL_H` | 24 | results are kept this long |
| `OPENERS_URL_HOSTS` | none | hosts `/jobs/url` may fetch from (https only, no redirects), e.g. `fra1.digitaloceanspaces.com` |
| `GGST_LANG_PROVIDER` | none | `claude`: an AI summary + coaching notes per match (Claude Opus 5.5, Italian and English) from the facts document only, with `ANTHROPIC_API_KEY` set. Every answer is checked against the facts and withheld if it cites anything that isn't there (`ai_summary` in the result). About 2.5k input + 2–4k output tokens (thinking included) per language: roughly $0.05–0.09 per language at $4/$20 per MTok |
| `OPENERS_TRANSCRIPT` | 0 | `1`: run the transcript too, when torch sees a CUDA GPU and the transcript weights are in `GGST_MODEL_DIR` (checked once at start, in a subprocess; `/health` shows the result). Then the whole analysis runs on the GPU |

Analyses run niced (+10). With the defaults the service uses at most 2 cores and up to about
4.5 GB of RAM at any time, whatever the number of requests (each further concurrent analysis
adds as much). The upload, the intermediate crops and the transcript features are
deleted as soon as an analysis ends; only `lab_result.json` and `openers.json` stay until the TTL.
The facts add well under a second.

Measured end to end through the service on the development PC, CPU only, light, 4 threads, on a
3:16 match at 720p30 (51 openers): 4 min 15 s in all (HUD pipeline 1 min 50 s, the rest crops,
move nets and banners), peak RAM 4.4 GB for the whole process tree. Slower VPS cores and 2 threads
make it roughly 1.5–3× longer: count on 5–12 min per match, i.e. a queue of 8 clears in about
an hour. The networks alone are about 1.1 s per opener light and 3 s full (OPENERS_LIGHT=0).

With the transcript (GPU box, RTX 3090): about 77 s more per match (V-JEPA 2 ViT-L feature pass, bound by the
video decode, plus a second pass for the grid tokens), then a few seconds for the ens5grid head and the state
tower, ~1.6 GB of GPU memory. Do not enable it on a CPU: the feature pass there is about an hour per minute of video.

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

Only on a GPU server with `OPENERS_TRANSCRIPT=1`, the transcript's weights too (pip `transformers` for V-JEPA 2,
which downloads `facebook/vjepa2-vitl-fpc64-256` on first use; CUDA torch instead of the CPU wheel):

```bash
scp <pc>:G:/Projects/replanal-data/ggst_dataset/model/{video_transcript_exp_ens5grid.pt,video_players_test.pt,state_embed_exp_se_clip_sup.pt} /opt/replanal-models/
scp <pc>:G:/Projects/replanal-data/ggst_dataset/transcript/feats_grid/pca.npz /opt/replanal-models/transcript_grid_pca.npz
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
- `GET /jobs/{id}` → `{status: queued|running|done|failed|cancelled, stage: starting|download|hud|detect|moves|transcript|facts|done, position, error, meta}`.
  `stage` follows lab_analyze's `stage: <name>` lines (`transcript` only when it runs). A finished job's `meta`
  has `openers` (count), `characters`, `capabilities` and `events` (transcript events, 0 without it).
- `GET /jobs/{id}/result` → `lab_result.json` (`spaghetti-lab-result/1`, see LAB_RESULT.md): `characters`,
  `capabilities`, `openers` (unchanged: per opener `time_s`, `attacker`, `attacker_char`, `category`, `move`
  (+ `move_top3`), `normal`, `hit_kind` (counter/punish/none), and the probabilities), `transcript`, `states`,
  `facts` (+ `md` per language), `prompts`, `timings_s`. Readers of the old `openers.json` keep working: its
  `characters` and `openers` are the same fields. Jobs finished before this change still return their `openers.json`.
- `DELETE /jobs/{id}` → cancel.
- `GET /health` → `{running, queued, max_concurrent, max_queue, capabilities: {openers, transcript, facts}}`.

Tests: `pytest tests/test_opener_service.py` (the analysis is stubbed; no model runs) and
`pytest tests/test_lab_analyze.py` (lab_analyze's mapping, facts assembly, the sample against the contract, the
search presets against `search_index.query`; no models).
