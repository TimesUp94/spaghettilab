# Desktop login and hosted Lab analysis

The Lab tab's spaghettishowdown.it login and hosted analysis: API reference and how the desktop side is
built (`lab_api.rs`, `lab_auth.rs`, `lab_remote.rs`, `LabPanel.tsx`, `LabHosted.tsx`).

The login and a real analysis are live on the dev server (`https://dev.spaghettishowdown.it`,
`https://dev.backend.spaghettishowdown.it`); production's defaults answer "not available" until they ship there.
`node app/e2e/live/dev-auth-check.mjs` runs the app's token and upload code against dev with the test account,
without starting an analysis. On 2026-11-04 it confirmed: exchange, rotation, 401 `invalid_token`, 400 `invalid_grant`,
403 `scope`, 404 on an unknown job or file, revoke, a 39 MB streamed upload, the file delete, and no charge for an
upload alone. Still unchecked for real: the website's Discord click, and a full analysis through the app.

Checked on 2026-11-04 by reading the code, and with read-only GETs:

| Source | Revision | Covers |
|---|---|---|
| FGCBackend `gitea/master` | `6b8877b` and the deployed `0373bae` (the local checkout is 2 commits behind Gitea) | routes, credits, upload, desktop tokens |
| FGCScoreboard `gitea/master` | `0d14034a` and `b7ae6bc5` | `OpenersPanel.jsx`, `/desktop-login`, `/terms`, `/privacy` |
| replanal `server/opener_service` | this repo | job bodies, stages, limits (FastAPI errors use `detail`) |

Production on that date:
- `GET /api/spaghetti-lab/openers/status` answers 503 `{"error":"Opener analysis is not available"}`.
- `https://spaghettishowdown.it/desktop-login` answers 404. The login code is on Gitea but not deployed.
- `/terms`, `/privacy` and `/support` answer 200.

## Verified API

Base URL: `https://backend.spaghettishowdown.it` (FGCAndroid `BASE_URL_PROD`, FGCScoreboard
`NEXT_PUBLIC_HERCULES`). Website: `https://spaghettishowdown.it`.

### Rules (`support-logic.js`, `support-service.js`)

- `FREE_PER_WEEK = 2`. The week starts Monday 00:00 **Europe/Rome**, and `week_resets_at` is that instant in ISO form.
- EUR 1 = 2 credits (`CENTS_PER_CREDIT = 50`), minimum EUR 2, maximum EUR 500. A guest contribution earns no credits.
- Free analyses are used first, then credits (`chooseSource`). The server decides.
- Given back: a failed job, a job cancelled while `queued`, a service refusal, or a start error. A cancelled `running` job is not given back, and `running` includes stages `starting` and `download`.

### Answers that apply to every route

| Answer | Source | Meaning |
|---|---|---|
| 401 `{"error":"invalid_token"}` | `desktop-scope.js` | desktop access token expired or revoked. Checked before any route, so it can replace the 503 below |
| 403 `{"error":"scope"}` | `desktop-scope.js`, `DELETE /api/user-files/:id` | outside the allow-list, or a file this token didn't upload: a bug to report, not a sign-out |
| 503 `{"error":"unavailable"}` | `desktop-scope.js` | token lookup failed: not a sign-out |
| 401 `{"error":"Authentication required","code":"credentials_required",...}` | global handler | no token, **including unknown `/api/...` paths** (GET on a made-up path was checked in production). A missing route therefore looks like an auth failure, not a 404 |
| 401 `{"error":"Invalid or expired token","code":...}` | global handler | bad player JWT (not used by desktop tokens) |
| 429 plain text `Too many requests from this IP...` | global limiter, 3000 per 15 min per IP | not JSON: always parse bodies defensively |
| 413 / 502 / 504 HTML | nginx | possible on any route. Its limits aren't in the repos (see "Open items") |

CORS: a request with no `Origin` is allowed, so Rust needs no whitelist entry.

### `/api/spaghetti-lab/openers`

Mounted as `iomiddleware, auth.playeroptional, router`. The first router middleware answers
503 `{"error":"Opener analysis is not available"}` on **every** route when `HERC_OPENERS_SECRET` is unset,
including `/quota`, before `auth.playerrequired`.

**`GET /status`** (public; call it without `Authorization`, or a stale token gets 401):
- 200: `{"available":true,"running":0,"queued":0,"max_concurrent":1,"max_queue":8,"capabilities":{"openers":true,"transcript":false,"facts":true,"ai_summary":true},"free_per_week":2}` (seen on the dev backend).
- 503 has two shapes: the `{"error":...}` above, or `{"available":false}` when the service is down.

**`GET /quota`**:
- 200: `{"free_remaining","free_per_week","credits","week_resets_at"}`.
- 500 `{"error":"Opener analysis is not available right now"}`, plus the shared answers.

**`POST /` `{"fileId"}`** (JSON key is camelCase):

| Status | Body | Handling |
|---|---|---|
| 202 | job view + `analysis_source` (`free` or `credit`) + quota fields | started, charged |
| 200 | job view + `reused:true` + quota fields, **no `analysis_source`**; `status` may already be `done` | not charged |
| 429 | `{"error":"quota_exhausted", quota fields}`, `Retry-After` = seconds to the week reset | offer `/support`; never auto-retry |
| 409 | `{"error":"analysis_starting", quota fields}`, `Retry-After: 5` | same POST again (web: 6 tries, 5 s apart) |
| 429 | service: `{"detail":"you already have an analysis queued or running"}`, `Retry-After: 60` | **no `error` field**; analysis given back |
| 503 | service: `{"detail":"the analysis queue is full, try again later"}`, `Retry-After: 300` | **no `error` field**; analysis given back |
| 413 | `{"error":"The video is over 15 min: cut it to one match or set"}` | only when `duration_seconds` is known and non-zero, see below |
| 400 | `{"error":"fileId is required"}` or `{"error":"Not a video file"}`; service: `{"detail":"the url's host is not allowed"}` | |
| 404 | `{"error":"File not found"}` | |
| 415 | service: `{"detail":"expected a video file (.mkv, .mov, .mp4, .webm)"}`, from the upload's file name | analysis given back |
| 502 | `{"error":"Opener analysis is not available right now"}` | the web shows its own fallback text on 502 |

Duration check: `file.duration_seconds` is 0 when the backend's ffprobe failed, and then the check is skipped.
The service probes when the job starts, so the job fails with that message and the analysis is given back.
It also fails jobs that aren't a readable video (`not a readable video`) or are under 360p
(`the video is below 360p`). The app checks all of this before uploading.

A second POST on the same `fileId` never charges twice: a live or finished job is reused, and a charge in the
last 2 minutes without a job answers `analysis_starting`. So a POST that timed out can be retried safely.

**Job view** (`GET /:jobId`, `Job.public`):
- Fields: `{id, owner, created, status, stage, started, finished, error, meta}`, plus `position` (1-based) **only while queued**.
- `status`: `queued|running|done|failed|cancelled`.
- `stage`: `null|starting|download|hud|detect|moves|transcript|facts|summary|done`.
- A failed job's answer adds `analysis_refunded`.
- 404 `{"error":"Analysis not found"}`, or `{"detail":"no such job"}` past the service's TTL (24 h).
- 502 `{"error":"Opener analysis is not available right now"}`.

`analysis_refunded` is **true only for the one call that gave it back** (`refundAnalysis` is idempotent).
A later poll answers false, and so does the first poll if the POST route or the sweep got there first (the sweep
handles jobs started over 10 minutes ago). False does not mean "not given back": re-read `/quota`.

**`GET /:jobId/result`**: 200 `lab_result.json`. 409 `{"detail":"the job has no result (yet)"}`. A job finished
before `lab_analyze` returns the old `openers.json`, which has no `schema` field; the hosted service now writes
`lab_result.json`, so `parse_lab_result` refuses anything else.

**`DELETE /:jobId`**: 200 `{"cancelled":true}`, with **nothing about a refund**, so re-read `/quota`.
409 `{"detail":"the job has already ended"}`: poll once to see how it ended.

### `/api/user-files`

**`POST /upload`** (multipart, field `file`):
- The part's **Content-Type must start with `video/`**, otherwise 400 `{"error":"invalid_file_type","message":...}`.
- A part with no extension is stored with `.mp4`. The storage key is always `.../video.mp4`.
- `file_type` is set to `'video'` by the route, which is what `POST /openers` checks.
- The part's `filename` becomes `original_name`, and the POST passes it to the service. Its extension must be one of `.mp4 .mkv .webm .mov`, and it becomes the **`--title` of the facts document**, which is what the AI summary is written from.
- 200: `{"success":true,"data":{"file_id","annotation_file_id","original_name","file_size","duration_seconds","compression_status"}}`.

| Error | Body |
|---|---|
| 413 | `{"error":"file_too_large","message":"File exceeds the 500MB limit."}` (`UPLOAD_MAX_SIZE_MB`, default 500, 500 x 1024 x 1024 bytes) |
| 400 | `{"error":"upload_failed","message"}` |
| 400 | `{"error":"No file uploaded"}` |
| 400 | `{"error":"Your storage limit reached (5GB). Delete some files first."}` (plain sentence, no code) |
| 400 | `{"error":"Global storage limit reached (250GB). Please contact support."}` |
| 403 | `{"error":"banned_global","message"}` (read the `error` field: this isn't the scope bug) |
| 500 | `{"error":"Failed to upload video"}` |

The 200 only comes after the backend has copied the file to Spaces, so it can arrive long after the last byte
was sent. The web client allows 10 minutes in total.

**Compression.** At the deployed `0373bae`, every upload starts a fire-and-forget libx264 CRF 23 re-encode (no
scaling) that **overwrites the same storage key**, without any lock against the analysis. `6b8877b` skips it for
desktop tokens (`compression_status:"skipped"` in the row and the answer), so **a desktop upload reaches the
analysis byte for byte**. The app relies on `6b8877b`; both ship together, since desktop tokens don't exist before it.

**`DELETE /:fileId`**:
- 200 `{"success":true}`. Removes the video and its paired annotation file.
- 404 `{"error":"File not found"}`: treat as done.
- 403 `{"error":"Not your file"}`, or `{"error":"scope"}` for a file this token didn't upload.
- **Works right after the analysis.** `lab_analysis_usage.file_id` has no foreign key, and `ownJob` looks the job up by job id, so status and result keep working with the file gone.
- **Must wait for the job to end.** The service fetches the file when the job's turn comes, so deleting it earlier fails the job (`could not download the video`).

`GET /api/user-files/:fileId` is outside the desktop scope (403), so the app can't read `compression_status`.

## Login contract

**Proposed, server agent builds it. Code exists on Gitea (FGCBackend `6b8877b`, FGCScoreboard `0d14034a`) but is
not deployed and may change.** Keep every URL and shape in `lab_api.rs`.

**Website page:** `GET {web}/desktop-login?redirect_port=&state=&code_challenge=&code_challenge_method=S256`
- `redirect_port`: 4-5 digits, 1024-65535. `state`: 1-512 characters. `code_challenge`: exactly 43 base64url characters.
- Client-rendered: an invalid link shows a message on the page and sends **no redirect**, so the app can only time out. Build the link correctly.
- A signed-out user logs in on the site. If `sessionStorage` is blocked, the login lands on the home page, so the app needs a way to **open the page again**.
- Allow goes to `http://127.0.0.1:<port>/callback?code=&state=`. Cancel goes to `...?error=access_denied&state=`.
- The same page lists connected PCs and can disconnect them. The app sees that as `invalid_token`, then `invalid_grant`.

**`POST {api}/api/desktop-auth/token`** (30 per minute per IP):
- Body: `{"grant_type":"authorization_code","code","code_verifier"}` (verifier: `[A-Za-z0-9._~-]{43,128}`), or `{"grant_type":"refresh_token","refresh_token"}`.
- 200, `Cache-Control: no-store`: `{"access_token","token_type":"Bearer","expires_in":3600,"refresh_token","scope":"lab","player":{"player_user_id","nickname","avatar"}}`.
- 400 `{"error":"invalid_grant"}` = signed out. 400 `{"error":"unsupported_grant_type"}`.
- 429 `{"error":"too_many_requests"}`. 500 `{"error":"unavailable"}` (not a sign-out).

**`POST {api}/api/desktop-auth/revoke` `{"refresh_token"}`**: 200 `{"ok":true}` even if already gone. 429 and 500 as above.

**On 401:** refresh once, retry once, then signed out. **403 `scope`:** report as a bug, don't sign out.

Details from the code that shape the client:
- Tokens are opaque: `sslab_at_` / `sslab_rt_` + 43 characters, about 52 in all. Only their hashes are stored.
- The code lives 120 s, works once, and a wrong verifier burns it.
- Access lasts 1 h. Refresh lasts 60 days and is extended on each use.
- **Each refresh replaces both tokens at once: the old access token stops working immediately.**
- **Presenting an already-rotated refresh token revokes the session.** A refresh whose response was lost, then retried, signs the user out, as can two app instances refreshing with the same token.
- `avatar` is a Discord CDN URL or null.
- `device_label` comes from `User-Agent` (first 120 characters). The website doesn't show it yet.

Scope allow-list (`isAllowedForLabScope`): `GET openers/status|quota`, `POST openers`, `GET openers/:id`,
`GET openers/:id/result`, `DELETE openers/:id`, `POST user-files/upload`, `DELETE user-files/:id`.

## Mismatches with the posted description

| # | Posted (message 25, task brief) | Real |
|---|---|---|
| 1 | Local FGC checkouts are current with `gitea/master` | Backend 2 commits and Scoreboard 1 commit behind; the login was only on Gitea |
| 2 | Tell the two 429s apart by the `error` field | Only `quota_exhausted` has one. The service's 429 and 503 have `detail`. The global limiter's 429 is plain text |
| 3 | `/status` 503 = not available | Right, but two body shapes, and production's has no `available` key. All routes, `/quota` included, get the 503 while analysis is off |
| 4 | `/status` is public | Only without a token: a stale desktop token gets 401 first |
| 5 | Failed job's answer includes `analysis_refunded` | True only for the call that gave it back; later or lost races answer false although it was given back |
| 6 | 200 `reused` / reopening costs nothing | Per `fileId`, and only while the service has the job (24 h). The app deletes each upload, so **re-analysing the same video always spends an analysis** |
| 7 | 413 over 15 minutes | Skipped when the backend's probe gives 0; then a failed job, given back. Under 360p and unreadable videos fail the same way |
| 8 | 202 carries `analysis_source` | Yes, but 200 `reused` doesn't. `running` for the refund rule includes `starting` and `download` |
| 9 | `DELETE` cancels | The answer doesn't say whether it refunded; 409 when already ended |
| 10 | Upload is multipart field `file`, returns `data.file_id` | Also needs a `video/*` part type and an accepted file name extension. Limit errors have mixed shapes. Compression, unmentioned, overwrites the file at the deployed revision, and is skipped for desktop tokens at `6b8877b` |
| 11 | 5 GB per user | A 400 with a plain sentence, no code |
| 12 | Week from Monday | Monday 00:00 Europe/Rome specifically |
| 13 | Login contract | Matches, with these additions: 401 `invalid_token`, 503 `unavailable`, `unsupported_grant_type`, 500 `unavailable`, reuse detection, access replaced on refresh, `GET /user-files/:id` not allowed, delete limited to the token's own uploads |
| 14 | A 404 tells a missing route | Unknown `/api` paths answer 401 `credentials_required` |
| 15 | "No player names are sent" to the AI summary (message 23) | The upload's file name becomes the facts' title, so a name like `Jason_vs_Enero.mp4` reaches the facts and the summary prompt. The website sends the user's real file name; the app avoids this (decision 7) |
| 16 | Failed and queued-cancelled analyses come back; a cancelled running one doesn't | True in code, but `/terms` section 13 only mentions analyses that fail. The in-app warning is the only place that says a cancelled running one is spent |

For the server agent (not desktop work):
- With a player JWT, a file deleted while compression runs can be written back to Spaces after the delete, leaving an object with no row. This affects website uploads only.
- `recordUpload` failing after the row is created answers 500, while the video stays on Spaces.

## Web client behaviour to match (`OpenersPanel.jsx`)

- Read the quota from any answer that carries all four quota fields, and re-read `/quota` after a failure or cancel, because the server decides any give-back.
- Polling: on 404 forget the job; on any other error keep polling, slower (web 4 s, then 12 s). There is no client timeout.
- A failed job shows `error`, else "The analysis failed".
- `analysis_starting`: same POST, 6 times, 5 s apart, then "still starting: try again in a moment".
- Other errors: on 502 use the fallback text, otherwise `error`, else `detail`, else the fallback.
- Cancel: ignore errors, because "the next poll shows how".
- `quota_exhausted`: show the free count and the reset date in Europe/Rome, with a contribute link.
- AI summary: show `text` only when `grounded`, otherwise the withheld note. Without a transcript, disable the transcript search presets (the app's `LabPanel` already handles both).

One deliberate difference: the website keeps Start enabled with nothing left, since reopening is free. The app re-reads
`/quota` before uploading and stops at 0 free and 0 credits, because a new upload can never be reused. This only saves
the upload; the server still decides and `quota_exhausted` is still handled.

## Desktop side

### Modules (`app/src-tauri/src/`)

- `lab_api.rs`: base URLs, every path, request and response types, and body parsing that tolerates non-JSON.
- `lab_auth.rs`: loopback listener, PKCE, exchange, refresh, revoke, keyring.
- `lab_remote.rs`: transcode, upload, start, poll, cleanup.
- `lab.rs` keeps the local mode, the cache and `parse_lab_result`.

Dependencies: `reqwest` (async; 0.13 is already in the lockfile through the updater; JSON, multipart, stream, rustls),
`keyring` 3 (native Windows and macOS stores, Secret Service on Linux, which needs `libdbus` in the release workflow),
`sha2`, `base64`, `rand`, `tokio-util`. Dev: `tiny_http`.

### Configuration

| Variable | Default | Effect |
|---|---|---|
| `SPAGHETTI_LAB_API_URL` | `https://backend.spaghettishowdown.it` | https only, except loopback |
| `SPAGHETTI_LAB_WEB_URL` | `https://spaghettishowdown.it` | same rule |
| `SPAGHETTI_LAB_KEYRING_SERVICE` | `SpaghettiLab` | test service name |
| `SPAGHETTI_LAB_NO_PROD` | unset | any non-loopback base URL is refused; set by every test run |
| `SPAGHETTI_LAB_TEST_BROWSER` | unset | Rust only requests a website link instead of opening a browser (no redirect is followed; the test calls the loopback callback); works only when both base URLs are loopback |
| `SPAGHETTI_LAB_TEST_POLL_MS` | unset | the job poll's interval, only in that test mode |
| `SPAGHETTI_LAB_MODELS` | unset | shows the local mode, labelled "local models" |

### Rust commands and TS wrappers (`api.ts`)

| Command | Wrapper | Returns |
|---|---|---|
| `lab_config` | `labConfig()` | `{local_models: bool, web_url}` |
| `lab_account` | `labAccount()` | `{signed_in, nickname, avatar, session_only}`; no tokens, ever |
| `lab_sign_in` | `labSignIn()` | the account; resolves when the flow ends |
| `lab_sign_in_reopen` | `labSignInReopen()` | opens the same login page again |
| `lab_sign_in_cancel` | `labSignInCancel()` | closes the listener |
| `lab_sign_out` | `labSignOut()` | refused during a hosted run; deletes pending uploads, revokes, clears the keyring |
| `lab_service_status` | `labServiceStatus()` | `{available, queued, running, max_queue, capabilities, free_per_week}` |
| `lab_quota` | `labQuota()` | quota fields, or null when not available |
| `lab_open_web` | `labOpenWeb("support"\|"terms"\|"privacy")` | opens the system browser |
| `run_lab_analysis_remote` | `runLabAnalysisRemote(dbPath, replayId, {again?, video_path?})` | the result |
| `cancel_lab_analysis_remote` | `cancelLabAnalysisRemote(replayId)` | `{phase: "before_charge"\|"queued"\|"running"}` |
| `lab_remote_running` | `labRemoteRunning()` | `{replay_id, stage, position, phase}` or null |
| `lab_remote_take_error` | `labRemoteTakeError(replayId)` | how the last run of this replay failed, once, for a screen that was closed meanwhile |
| `lab_remote_resume` | `labRemoteResume()` | picks up saved jobs at startup |

`get_lab_result` and `delete_lab_result` are unchanged. `lab_analysis_running` also reports a hosted run.

Errors from the remote commands are objects, `{code, message, retry_after?, refunded?, quota?}`, not strings:

| `code` | When | UI |
|---|---|---|
| `signed_out` | `invalid_grant`, or 401 after the retry | sign-in panel; the job stays saved |
| `login_unavailable` | `/desktop-login` 404, or `/token` 401 or 404 | "Sign-in with spaghettishowdown.it isn't available yet. The rest of the app works as usual." |
| `login_denied`, `login_timeout`, `login_state` | cancel in the browser, timeout, state mismatch | message, retry |
| `already_cached` | a cached result and no `again` | confirm: spends an analysis |
| `busy` | another Lab run is going, or sign-out during a run | message |
| `credential_store` | the OS store can't be read | message; the session stays |
| `rate_limited` | the global limiter's plain 429, or `/token` 429 | message, `retry_after` |
| `unavailable` | 503 or 502 | "Not available right now"; sign-in unaffected |
| `queue_full` | `/status` full, or service 503 | retry later, with `retry_after` |
| `one_at_a_time` | service 429 | "You already have an analysis queued or running (on the website or another PC)." |
| `quota_exhausted` | pre-check or POST | reset date and the `/support` button |
| `too_long`, `too_small`, `too_big` | over 15 min, under 360p, over 500 MB | "A whole VOD: cut it to one set." |
| `storage_full`, `banned` | upload errors | server's text |
| `job_failed` | job failed | `error`, `refunded`, fresh quota |
| `expired` | job 404 | "The server no longer has this analysis." |
| `bad_result` | schema check failed | previous result stays |
| `scope` | 403 scope | "Please report this", not a sign-out |
| `network` | no connection | retry |
| `cancelled` | user cancelled | nothing |

### Progress stages

Event `lab-progress` keeps `{replay_id, stage, line}` and adds optional `percent`, `position`, `sent`, `total`, `source`.

| `stage` | Label | Extras |
|---|---|---|
| `check` | Checking the video | |
| `transcode` | Preparing a 720p copy | `percent` |
| `upload` | Uploading | `percent`, `sent`, `total`; at 100 %: "Finishing the upload" |
| `start` | Starting the analysis | then `source` (`free`, `credit`, or null when reused) |
| `queued` | Waiting in the queue | `position` |
| `starting`, `download` | Fetching the video | |
| `hud`, `detect`, `moves`, `transcript`, `facts`, `summary` | as today (`lib/lab.ts`) | an unknown stage shows "Analysing" |
| `result` | Getting the result | |
| `done` | | |

### Hosted run, in order

1. **Refuse early**: another run going (one at a time), or a cached result without `again`.
2. **`check`**: `/status` without a token (503: `unavailable`; `queued >= max_queue`: `queue_full`). Then `/quota`, and stop if nothing is left. Read the video's duration, size and height with `ffmpeg -i` (there may be no ffprobe; the DB's `duration_ms` can be 0).
3. **Limits**: over 900 s, under 360p: stop before any work.
4. **`transcode`** to a temp file, always removed on every path:
   - Over 720p or too large: `-vf scale=1280:720,fps=30 -c:v libx264 -preset veryfast -b:v 3M -maxrate 3M -bufsize 6M -pix_fmt yuv420p -an -movflags +faststart`.
   - Otherwise, skip the encode and remux with `-c:v copy -an` (re-encode if that fails).
   - Then check the result against 500 x 1024 x 1024 bytes.
5. **`upload`**:
   - Stream from disk with a known length, never held in memory: file name `replay.mp4`, type `video/mp4`, no total timeout, a stalled-read timeout, and a refresh first if the token has under 60 s left.
   - Save `{file_id, player_user_id}` to `lab/<id>/remote_job.json` as soon as the answer arrives.
6. **`start`**: POST. Retry on `analysis_starting` (Retry-After, 6 times) and on network errors (3 times; safe, see above). Save `job_id`. Never abort a POST in flight: if cancel was pressed, wait for the answer and then cancel the job.
7. **Poll** every 15 s, 30 s after an error, with no overall timeout (a job can wait in the queue for hours). A 404 is `expired`.
8. **`result`**: run `parse_lab_result`, then write to `lab/<id>/lab_result.json`. The previous result stays until then.
9. **Clean up** on every path: `DELETE /user-files/:id`, re-read `/quota`, remove `remote_job.json`.
   A result lost to the network or a 503 keeps the job (without the file) so it can be collected later: it's paid for.

**Cleanup that can't finish.** A failed file delete goes to `lab/pending_deletes.json` (`{file_id, player_user_id}`,
never tokens), retried at startup and at sign-in as the same player. 404 and `scope` remove the entry.

**Resume.** `remote_job.json` with a `job_id`: keep polling. Without one: delete the file and don't POST,
because the user didn't ask to spend again. If an earlier POST did reach the server, that job fails and the analysis is given back.

**Cancel by phase.**
- Before the charge: no question. Stop, kill ffmpeg and wait for it (Windows locks the output), abort the upload. Once every byte is sent the server may already hold the file, so the cancel waits for its id and deletes it. Nothing spent.
- The phase is `queued` from the moment the POST goes out, because the server may charge from then on.
- `queued`: the UI says it will be given back.
- `running`, any stage: the UI warns it won't be refunded and asks first.
- After `DELETE /:jobId`: on 409, poll once and, if `done`, keep the result, since it was already paid for. Then delete the file and re-read `/quota`.

**Failed job.** `analysis_refunded: true`: "Your analysis was given back". False: no claim either way, just the fresh quota line.

### Login flow

1. Pre-check `GET {web}/desktop-login`: 404 gives `login_unavailable` without opening a browser; a connection failure gives `network`.
2. Bind `127.0.0.1:0`. Generate `state` (32 random bytes) and a verifier (32 random bytes, base64url), challenge = S256. Open the system browser.
3. Wait **10 minutes** for `GET /callback`. Other paths get 404 and don't end the wait, so a stray `favicon` or local probe can't use up the single callback. Then close the listener. A wrong `state` aborts.
4. Reply with a small static page (no script, `Referrer-Policy: no-referrer`), then exchange the code.

**Tokens.**
- The refresh token lives in the OS store (service as above, account `refresh_token`); nickname and avatar in account `player`. At about 52 characters it fits Windows' 2560-byte limit.
- The access token lives **in memory only**, with its expiry.
- Tokens never reach the webview, SQLite, files or logs, and error strings go through a filter that removes `sslab_` values and `code=`.
- If the OS store fails: the session lives in memory (`session_only: true`, with a note); there is never a file fallback.

**Refresh.**
- One at a time, behind a mutex, reading the store's current token first so a second instance's rotation is picked up.
- The new token is written immediately.
- Failures that never reached the server (connect errors) are retried.
- A lost response isn't retried automatically; the next attempt then gets `invalid_grant` and the user signs in again.

`User-Agent`: `Spaghetti Lab <version> (<OS>)`, no hostname.

### UI (`LabPanel.tsx`)

- **Cached results stay readable signed out.** Nothing outside the Lab tab needs an account.
- **Signed out**: the sign-in panel with the text below, and while waiting "Open the page again" and "Cancel". When the service is down, it says so without touching sign-in state.
- **Signed in**:
  - Nickname, avatar (initial if null), sign out.
  - "N free this week, M credits", refreshed at startup, after each analysis and on window focus (at most every 30 s), and from any answer with quota fields. Reset date in Europe/Rome.
  - Before starting: "This will use a free analysis" or "This will use a credit", worded as what's expected. After `start`, the real `analysis_source`.
  - A cached result: "Analyse again" with a confirmation that it spends an analysis, even for the same video.
- **Exhausted**: reset date and a button opening `{web}/support` in the system browser. No payments in the app. (The site's own link adds `?from=lab` for its source tracking; add it if the server agent wants it counted.)
- **Local mode** only with `SPAGHETTI_LAB_MODELS` set: selected by default, labelled "local models" next to "server", existing options, no login, no spending. Normal users never see `NOT_INSTALLED`.
- Hints that only hold locally (the GPU note on a missing transcript, "nothing is sent anywhere", the API key hint) show only with the local mode.
- **Upload notice** before the first upload, remembered in `localStorage` by text version, so a material change asks again.

**Notice text**, matched to the 2026-11-04 `/terms` (section 13) and `/privacy` (sections 2.1, 4, 6):

> To analyse a match, Spaghetti Lab sends a 720p copy of the video, without audio, to the Spaghetti Showdown servers
> (hosted in the EU). The app deletes it when the analysis ends, whether it finished, failed or was cancelled. The
> result stays on the server for up to 24 hours so the app can fetch it, then only on this PC. If AI summaries are on,
> Anthropic receives game data extracted from the video, never the video. Results are generated automatically and can
> contain errors. You have 2 free analyses a week, then credits, which you can get on the website. An analysis that
> fails or is cancelled while waiting is given back; cancelling one that's running is not.
> [Terms](/terms) [Privacy](/privacy)

An Italian version is needed too, as those pages are in Italian. Things this text must not overstate:
- Neither page promises deletion, or gives a retention period for Lab videos. Deletion is **this app's** behaviour, so the text says "the app deletes it". If a delete fails it's retried later.
- `/privacy` section 4 says only game data reaches Anthropic, which holds only with decision 7.
- Credits (non-transferable, no expiry), contributions and the 14-day withdrawal are website matters under `/terms` section 13, and the app only links there.

### Decisions

1. **Async `reqwest`**, not blocking: dropping the future aborts a stalled upload at once, and a read timeout can detect stalls.
2. **Access token in memory only**: the store holds only what must survive a restart.
3. **The status call is always tokenless**, so a stale token can't turn "not available" into a sign-out.
4. **Audio is removed.** `lab_analyze` and `recognize` don't use it, the file gets smaller, and voices stay on the PC.
5. **Stop before uploading when the quota is 0**, as described above.
6. **Jobs are saved per replay** (the web uses `sessionStorage`), so closing the app during a 15-minute run loses nothing, and a job already paid for is collected on the next start.
7. **The upload's file name is always `replay.mp4`.** This keeps player names away from the facts and the AI prompt, as `/privacy` and message 23 say. The cost is a plain heading in the facts document.
8. **`/callback` is the one request that ends the wait.** The brief says "one request then closed"; other paths get 404 so they can't use it up.
9. **A 10-minute login timeout**, because the first sign-in may include a Discord login.

### Open items

- nginx's `client_max_body_size` and proxy read timeout aren't in any repo. The website's comment says 1 GB and it allows 10 minutes, but a 500 MB upload through production is unverified. A test needs a real account, so it's left to the server agent or the user.
- `UPLOAD_MAX_SIZE_MB`'s production value isn't in the repos. The app uses the 500 MB default and handles 413 either way.
- There's no endpoint for "my active job", so a job running on the website or another PC is only found after the upload (service 429). An endpoint would save that upload.
- `/terms` doesn't state that cancelling a running analysis is not refunded (mismatch 16).

## Tests and the mock backend

Every test run sets `SPAGHETTI_LAB_API_URL` and `SPAGHETTI_LAB_WEB_URL` to the mock, `SPAGHETTI_LAB_NO_PROD=1`,
and `SPAGHETTI_LAB_KEYRING_SERVICE=SpaghettiLabE2E` (entries deleted at teardown). The existing `SPAGHETTI_LAB_NO_MODELS`,
`SPAGHETTI_LAB_NO_GAME` and `SPAGHETTI_LAB_DATA_DIR` stay. Nothing talks to production.

**`app/e2e/mock-lab-backend.ts`**: Node `http`, no dependencies, started by `global-setup.ts` on a random loopback port,
in the runner's process (`E2E_MOCK_URL` tells the workers and the app).

| Part | What it does |
|---|---|
| Web | `GET /desktop-login` applies `desktopLogin.js`'s rules and remembers the link with a code ready; the spec calls the loopback callback itself (a good code, `access_denied`, a wrong `state`), as the redirect would. 404 when the login is "missing" |
| Token | `/api/desktop-auth/token` and `/revoke` with PKCE really checked, one-use codes, rotation, reuse detection, access replaced on refresh |
| Scope | the `isAllowedForLabScope` list, 401 `invalid_token`, 403 `scope`, and 401 `credentials_required` on unknown `/api` paths |
| Upload | counts bytes while **discarding the stream**, reads only the part headers, applies the `video/*` and extension rules, the size limit and the error shapes; records the peak |
| Openers | the exact status codes and bodies from this document. Jobs advance one stage per poll (`queued` with `position`, then `download` to `done`), and `analysis_refunded` is true only once |
| Result | serves `e2e/fixtures/lab_result.json`, plus a no-transcript variant, one with the AI summary withheld, and a bad schema |
| Control | `POST /__mock/scenario` sets knobs, `/__mock/quota` the quota, `/__mock/expire-access` and `/__mock/revoke-all` end sessions; `/__mock/log` returns calls with tokens reduced to a kind; `/__mock/state` counts rotated tokens presented and uploads deleted under a live job, which `15-lab-hosted` requires to stay 0 after every test |

Knobs (`Scenario` in the file): `service` (on, off = production's 503, down), `login` (ok, missing), `startQueue` (the next POST
answers: `analysis_starting`, `quota_exhausted`, `one_at_a_time`, `queue_full`, `too_long`, `plain_429`), `upload` (ok, `too_big`,
`html_413`, `storage_full`, `banned`), `uploadDelayMs`, `holdAt` (a job stops at a stage), `failAt`, `refundLost`, `job404`,
`result` (ok, bad, no_transcript), `fileDelete`, `tokenOutage`.

**Rust** (`cargo test lab_`; these need no DB, unlike the round harnesses):
- Real sockets for the listener: encoded `state`, wrong `state`, cancel, an unrelated request, timeout, and the port closed afterwards.
- `tiny_http` tests: PKCE vector, refresh-once-retry-once, concurrent 401s giving one refresh, rotation written to the store before use, 403 not a sign-out, non-JSON bodies, tokens absent from errors and logs, base-URL rule, NO_PROD refusal.
- Upload: multipart headers and a streamed known length.
- Transcode: 2 s of the small e2e clip through libx264, skipped when the clip isn't there; never a GPU.
- Temp files removed on every path.

**Playwright** (`15-lab-hosted`, real Rust against the mock; the upload is the small clip, stream-copied):
- Signed-out panel and links, sign-in, sign-out, each login failure, the quota line and focus refresh.
- Notice shown once, stages and upload progress, cancel warnings by phase, refund notes, the error rows.
- Re-analyse confirmation, cached result signed out, resume after a restart, cleanup calls in `/__mock/log`, and local mode shown only with the variable.

`14-lab` restarts the app with `SPAGHETTI_LAB_MODELS` set (`connection.restart(env)`) to reach the local form, and back afterwards.
