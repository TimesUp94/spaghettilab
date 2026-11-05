# Desktop login and hosted Lab analysis

The Lab tab's spaghettishowdown.it login and hosted analysis: API reference and how the desktop side is
built (`lab_api.rs`, `lab_auth.rs`, `lab_remote.rs`, `LabPanel.tsx`, `LabHosted.tsx`).

## State on 2026-11-04 (evening)

The desktop side is finished and pushed (`7e5408c`, `1088cfa`, `5517675`; Gitea CI green on all three). It isn't usable on
production yet, because production has neither the login nor analysis on. The app's default URLs are production's, and they
match FGCAndroid `BASE_URL_PROD` and FGCScoreboard `NEXT_PUBLIC_HERCULES`, so no client change is expected when it ships.

| Where | Login | Analysis | What the app shows |
|---|---|---|---|
| Production (`spaghettishowdown.it`, `backend.spaghettishowdown.it`) | `/desktop-login` answers 404 | `/openers/status` answers 503 `{"error":"Opener analysis is not available"}` | The readable "isn't available yet" sign-in message; cached results still open |
| Dev (`dev.spaghettishowdown.it`, `dev.backend.spaghettishowdown.it`) | Live | Live, with the real service and a paid AI summary | The whole flow, with `SPAGHETTI_LAB_API_URL` and `SPAGHETTI_LAB_WEB_URL` |

**Confirmed against dev with the app's real Rust client:** exchange, quota, status, rotation, 8 concurrent callers sharing one
refresh, 401 refresh and retry, 403 `scope`, 400 `invalid_grant`, revoke, a stale refresh token, a streamed upload, the file
delete, three full runs (result validated and cached, upload deleted), free before credits, `reused`, cancel while running (not
given back), and a failed job given back (`analysis_refunded`).

**Confirmed only against the mock:** `quota_exhausted` and its `/support` offer, cancel while queued, the token lookup 503, 403
`scope` on a file delete.

**Not tested anywhere real:** the website's Discord click and its redirect to the app's listener (needs a person), an upload near
500 MB through nginx, a real 1080p match through the encode, a `.spag` session or database-resolved video through a run, a long
match against the 30-minute service timeout, whether a cancelled upload leaves a partial file on the server, the native close
dialog, and killing ffmpeg on exit. Anything on production.

**Not recommended for production yet**, because of the two backend problems under "Open on the server side": a non-game video can
be charged, and the AI summary is usually withheld after it has been paid for. The full list is in "For the server agent".

`node app/e2e/live/dev-auth-check.mjs` runs the app's token and upload code against dev with the test account, without starting an
analysis. `app/src-tauri/src/lab_live_dev.rs` runs full analyses through the app's pipeline on dev (see "Live tests on dev").

Checked on 2026-11-04 by reading the code, and with read-only GETs:

| Source | Revision | Covers |
|---|---|---|
| FGCBackend `gitea/master` | `6b8877b` and the deployed `0373bae` (the local checkout is 2 commits behind Gitea) | routes, credits, upload, desktop tokens |
| FGCScoreboard `gitea/master` | `0d14034a` and `b7ae6bc5` | `OpenersPanel.jsx`, `/desktop-login`, `/terms`, `/privacy` |
| replanal `server/opener_service` | this repo | job bodies, stages, limits (FastAPI errors use `detail`) |

Production, last read that evening: the `/status` and `/desktop-login` answers in the table above; `/terms`, `/privacy` and
`/support` answer 200. The login code is on Gitea but not deployed there.

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
| 502 | `{"error":"Opener analysis is not available right now"}` | the web shows its own fallback text on 502. The app treats every 5xx except the route's 503 as a lost answer, because nginx can give up while Node carries on |

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

**Built by the server agent and live on dev (FGCBackend `6b8877b`, FGCScoreboard `0d14034a`), not on production.**
Keep every URL and shape in `lab_api.rs`.

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

For the server agent (not desktop work). The first group is needed before production, the rest can follow:

- **Deploy the login to production:** `/desktop-login` and `/api/desktop-auth/*`.
- **Turn analysis on in production** after the fixes below, with `/status` answering 200.
- **Fail a run that isn't a match.** No HUD or no openers should end `failed`, given back, with no AI summary (see "Open on the server side").
- **Stop paying for withheld AI summaries.** Fix the fact check and prompt, or switch the summary off until then. The code is `lang_summary.check_answer`, in replanal.
- **The fact check misreads Slayer's moves (2026-11-05, 10:12 set on dev, `8a51971`):** both languages were withheld after their repair call, and all 16 errors were `move_not_in_facts` for `214P`. `NOTATION_RE` in `lang_summary.py` stops at the `/` of the facts' own `214P/K~H`, so that notation can't pass even when quoted exactly.
- **Timing on the same run:** 2,386 s on the service for 612 s of video (3.90x: hud 1,081 s, detect 76 s, moves 1,136 s, summary 85 s). A 15-minute match at that ratio takes about 3,510 s against `OPENERS_TIMEOUT_S` 3600.
- **Check long matches against `OPENERS_TIMEOUT_S`.** Runs took about 3.7x the length of a 90 s clip, against about 1.2x for a 12.8-minute video. At 3.7x, a 15-minute match would hit the 30-minute limit.
- **Confirm a 500 MB upload through production:** nginx `client_max_body_size` and read timeout, and `UPLOAD_MAX_SIZE_MB`.
- **Set the dev test account to 0 free and 0 credits** (asked in project.fgc #36, not answered), so `lab_dev_7` can test `quota_exhausted`. Then reset it.
- **Keep the dev service busy for a moment,** so cancel while queued can be tested with the test account.
- **State the cancel rule in `/terms` section 13:** cancelling a running analysis isn't given back. Only the app says so today (still absent on production on 2026-11-04).

Can follow:
- **`strList` in `labResult.js`:** the AI summary's `errors` and `warnings` are `{kind, text}` objects, so the website probably shows "[object Object]" (read, not run).
- **Website uploads' file names:** the file name can reach the facts' title and so the AI prompt (mismatch 15). The app always sends `replay.mp4`.
- **Partial uploads:** check whether a cancelled desktop upload leaves anything on the server. The app never gets a file id for it.
- **An endpoint for the user's active job,** so a job running on the website or another PC is found before the upload, not after it.
- **Sweep desktop uploads.** Nothing removes a `desktop_auth_upload` file except the app's own DELETE or account deletion. Orphans: a closed app or lost answer during "Finishing the upload" (the app never learned the id, and a desktop token can't list files), and a player who never returns after a started job. They also count against the 5 GB. Wanted: remove such a file once its job has ended, and any with no job after a few hours. The upload notice can then promise more.
- **Look up a job by file.** `GET /api/spaghetti-lab/openers?fileId=`, read-only and charging nothing. Today the only way to learn about a POST whose answer was lost is the same POST, which charges if the first never arrived. With a lookup, a cancel in that window could reach the job too.
- **A start that timed out.** If `POST /jobs/url` times out at axios's 30 s after the service created the job, the route refunds and answers 502 while the job runs on, with no record. Rare, and the app's retry then gets `one_at_a_time`.
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
- `lab.rs` keeps the cache and `parse_lab_result`.
- `lab_local.rs`: the local mode, compiled only with the `local-models` feature (see "Local build").

Dependencies: `reqwest` (async; 0.13 is already in the lockfile through the updater; JSON, multipart, stream, rustls),
`keyring` 3 (native Windows and macOS stores, Secret Service on Linux, which needs `libdbus` in the release workflow),
`sha2`, `base64`, `rand`, `tokio-util`. Dev: `tiny_http`.

### Configuration

| Variable | Default | Effect |
|---|---|---|
| `SPAGHETTI_LAB_API_URL` | `https://backend.spaghettishowdown.it` | https only, except loopback |
| `SPAGHETTI_LAB_WEB_URL` | `https://spaghettishowdown.it` | same rule |
| `SPAGHETTI_LAB_KEYRING_SERVICE` | `SpaghettiLab`; `SpaghettiLabIsolated` with `SPAGHETTI_LAB_DATA_DIR` set | test service name |
| `SPAGHETTI_LAB_SINGLE_INSTANCE` | unset | keeps the single-instance lock on although `SPAGHETTI_LAB_DATA_DIR` is set (a spec tests it this way) |
| `SPAGHETTI_LAB_NO_PROD` | unset | any non-loopback base URL is refused; set by every test run |
| `SPAGHETTI_LAB_TEST_BROWSER` | unset | Rust only requests a website link instead of opening a browser (no redirect is followed; the test calls the loopback callback); works only when both base URLs are loopback |
| `SPAGHETTI_LAB_TEST_POLL_MS` | unset | the job poll's interval, only in that test mode |
| `SPAGHETTI_LAB_MODELS` | unset | the models folder, in the local-models build only; ignored by published builds |

### Rust commands and TS wrappers (`api.ts`)

| Command | Wrapper | Returns |
|---|---|---|
| `lab_config` | `labConfig()` | `{local_models: bool, models_dir: string\|null, web_url}`; `local_models` is true only in the `local-models` build, `models_dir` is the folder it finds by itself |
| `lab_account` | `labAccount()` | `{signed_in, nickname, avatar, session_only, store_unavailable}`; no tokens, ever |
| `lab_sign_in` | `labSignIn()` | the account; resolves when the flow ends |
| `lab_sign_in_reopen` | `labSignInReopen()` | opens the same login page again |
| `lab_sign_in_cancel` | `labSignInCancel()` | closes the listener |
| `lab_sign_out` | `labSignOut()` | refused during a hosted run; deletes pending uploads, revokes, clears the keyring |
| `lab_service_status` | `labServiceStatus()` | `{available, queued, running, max_queue, capabilities, free_per_week}` |
| `lab_quota` | `labQuota()` | quota fields, or null when not available |
| `lab_open_web` | `labOpenWeb("support"\|"terms"\|"privacy")` | opens the system browser |
| `run_lab_analysis_remote` | `runLabAnalysisRemote(dbPath, replayId, {again?, video_path?})` | the result |
| `cancel_lab_analysis_remote` | `cancelLabAnalysisRemote(replayId)` | `{phase}`, the phase as last known |
| `lab_remote_cancel_phase` | `labRemoteCancelPhase(replayId)` | `{phase: "before_charge"\|"starting"\|"queued"\|"running"\|"unknown"}`, with a fresh GET of the job when it has an id; the warning is built from this |
| `lab_remote_running` | `labRemoteRunning()` | `{replay_id, stage, position, phase}` or null |
| `lab_remote_take_error` | `labRemoteTakeError(replayId)` | how the last run of this replay ended if it failed or was cancelled, once. Kept per replay until a screen of that replay has shown it; a new run clears it |
| `lab_remote_resume` | `labRemoteResume()` | picks up saved jobs at startup |

`get_lab_result` and `delete_lab_result` are unchanged. `lab_analysis_running` also reports a hosted run.

Errors from the remote commands are objects, `{code, message, retry_after?, refunded?, quota?}`, not strings:

| `code` | When | UI |
|---|---|---|
| `signed_out` | `invalid_grant`, or 401 after the retry | sign-in panel; the job stays saved |
| `login_unavailable` | `/desktop-login` 404, or `/token` 401 or 404 | "Sign-in with spaghettishowdown.it isn't available yet. The rest of the app works as usual." |
| `login_denied`, `login_timeout` | cancel in the browser, timeout | message, retry |
| `already_cached` | a cached result and no `again` | confirm: spends an analysis |
| `busy` | another Lab run is going, or sign-out during a run | message |
| `credential_store` | the OS store can't be read, or won't delete a token on sign-out | message; the session stays |
| `rate_limited` | the global limiter's plain 429, or `/token` 429 | message, `retry_after` |
| `unavailable` | 503 or 502 | "Not available right now"; sign-in unaffected |
| `queue_full` | `/status` full, or service 503 | retry later, with `retry_after` |
| `one_at_a_time` | service 429 | "You already have an analysis queued or running (on the website or another PC)." |
| `quota_exhausted` | pre-check or POST | reset date and the `/support` button |
| `too_long`, `too_small`, `too_big` | over 15 min, under 360p, over 500 MB | "A whole VOD: cut it to one set." |
| `storage_full`, `banned` | upload errors | server's text |
| `job_failed` | job failed | `error`, `refunded`, fresh quota |
| `expired` | job 404 | "The server no longer has this analysis." |
| `bad_result` | schema check failed | previous result stays; the text is kept as `lab_result.rejected.json`, and an app version that can read it picks it up |
| `result_pending` | job done, but a result answer other than 200 or 404 after 5 tries, or a cancel during them | job kept without the file; "Try again", or the next start, collects it for free |
| `result_write` | the result couldn't be written on this PC | same |
| `start_unconfirmed` | no answer to the start POST after retries | file and record kept (`post_sent`); "Try again", or the next start, asks with the same POST |
| `cancel_unconfirmed` | no DELETE confirmed after 3 tries | file and record kept (`cancel_requested`); "Try again", or the next start, cancels first |
| `scope` | 403 scope | "Please report this", not a sign-out |
| `network` | no connection | retry |
| `cancelled` | user cancelled, confirmed by the server | a note: `refunded` from the quota before and after the DELETE, or `phase: "unconfirmed"` when the start was never answered |

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

While a job is `running`, its `stage` can read `done` (the chain's last stage, before the AI summary). The app shows nothing new for that; only `status: done` ends it.

### Hosted run, in order

1. **Refuse early**: another run going (one at a time), or a cached result without `again`.
2. **`check`**: `/status` without a token (503: `unavailable`; `queued >= max_queue`: `queue_full`). Then `/quota`, and stop if nothing is left. Read the video's duration, size and height with `ffmpeg -i` (there may be no ffprobe; the DB's `duration_ms` can be 0).
3. **Limits**: over 900 s, under 360p: stop before any work.
4. **`transcode`** to a temp file, always removed on every path:
   - Over 720p or too large: `-vf scale=1280:720,fps=30 -c:v libx264 -preset veryfast -b:v 3M -maxrate 3M -bufsize 6M -pix_fmt yuv420p -an -movflags +faststart`.
   - Otherwise, skip the encode and remux with `-c:v copy -an` (re-encode if that fails).
   - Then check the result against 500 x 1024 x 1024 bytes.
5. **`upload`**:
   - Stream from disk with a known length, never held in memory: file name `replay.mp4`, type `video/mp4`, no total timeout, a stalled-read timeout, and a refresh first if the token has under 30 minutes left (nginx may hold the whole body before the backend checks the token).
   - Save `{file_id, player_user_id}` to `lab/<id>/remote_job.json` as soon as the answer arrives.
6. **`start`**: save `post_sent` first, then POST. Retry on `analysis_starting` (Retry-After, 6 times) and on a lost answer (network error or any 5xx but the route's 503; 3 times; safe, see above). Save `job_id`. A POST in flight is never aborted by a cancel: wait for the answer and then cancel the job. A cancel does stop the next POST. Closing the app can abort one, so the record is what makes it safe.
7. **Poll** every 15 s, 30 s after an error, with no overall timeout (a job can wait in the queue for hours). A 404 is `expired`.
8. **`result`**: run `parse_lab_result`, then write to `lab/<id>/lab_result.json`. The previous result stays until then.
9. **Clean up**: `DELETE /user-files/:id`, re-read `/quota`, remove `remote_job.json`. The record goes only after a stored
   result, a job known to have ended without one, or a definite 404. Any other answer about a finished job keeps it (without the file),
   because it's paid for and the service keeps the result for 24 h. `signed_out`, `credential_store`, `scope` and an unconfirmed cancel change nothing: a job that may be alive keeps its file.

**Cleanup that can't finish.** A failed file delete goes to `lab/pending_deletes.json` (`{file_id, player_user_id}`,
never tokens), retried at startup and at sign-in as the same player. 404 and `scope` remove the entry.

**Resume.** `remote_job.json` with a `job_id`: keep polling (or cancel first, with `cancel_requested`). With `post_sent` and no
`job_id`: the same POST again. A job on that file answers 200 `reused` for free, and otherwise the POST starts the analysis the user
asked for. Neither: no POST went out, so delete the file and don't POST, because the user didn't ask to spend again.
A refusal by the route (`quota_exhausted`, `one_at_a_time`, `queue_full`, 413, 400, 404, 415) is an answer: no job, file deleted, even after a lost POST.

**Second launch.** `tauri-plugin-single-instance`: a second launch hands its `.spag`/`.spagz` to the running window and exits, so two
processes never share the session, `remote_job.json` or `.tmp`. An isolated instance (`SPAGHETTI_LAB_DATA_DIR`) runs next to the real app with
its own data and credential service. So `npx tauri dev` without a data dir exits at once while an installed app is running.

**Closing the app.** While a run is before the charge (`check`, transcode, upload) the window asks first, in a native dialog; once the POST
is out or a job is saved nothing is lost and it closes as usual. `RunEvent::Exit` kills the ffmpeg of the temp copy (process exit runs no
drops, so `kill_on_drop` isn't enough) and removes `.tmp`. A crash can still leave that ffmpeg to finish alone; the next start removes `.tmp`.

**Cancel by phase.**
The refund depends on the job's status when the DELETE arrives, so pressing Cancel first asks Rust (`lab_remote_cancel_phase`), which GETs the job.
- Before the charge: no question. Stop, kill ffmpeg and wait for it (Windows locks the output), abort the upload. Once every byte is sent the server may already hold the file, so the cancel waits for its id and deletes it. Nothing spent.
- `starting`, from the moment the POST goes out: with an empty queue the job can be running at once, so the warning covers both cases. Same for `unknown` (the job couldn't be read).
- `queued`: the UI says it will be given back unless its turn comes first.
- `running`, any stage: the UI warns it won't be refunded and asks first.
- After `DELETE /:jobId`, tried up to 3 times:
  - 200: delete the file. Comparing `/quota` before and after gives the note: given back or not. The DELETE itself doesn't say.
  - 404: gone.
  - 409: poll once. `done` keeps the result, since it was already paid for; `failed` is a failed job.
  - Anything else: `cancel_unconfirmed`, and the job and file stay.
- With a lost POST and no job id, a cancel stops further POSTs (one could only charge) and deletes the file, so a job still waiting fails and is given back. A job already past its download can't be reached without a lookup by file (see "For the server agent"), so the note says so.

**Failed job.** `analysis_refunded: true`: "Your analysis was given back". False: no claim either way, just the fresh quota line.

### Login flow

1. Pre-check `GET {web}/desktop-login`: 404 gives `login_unavailable` without opening a browser; a connection failure gives `network`.
2. Bind `127.0.0.1:0`. Generate `state` (32 random bytes) and a verifier (32 random bytes, base64url), challenge = S256. Open the system browser.
3. Wait **10 minutes** for `GET /callback`. Other paths get 404 and a wrong `state` a 400, and neither ends the wait, because anything on this PC can reach the port. Connections are served side by side, each with one 5 s deadline for its whole request, so a silent or trickling one can't hold up the browser's. Then close the listener.
4. Reply with a small static page (no script, `Referrer-Policy: no-referrer`), then exchange the code.

**Tokens.**
- The refresh token lives in the OS store (service as above, account `refresh_token`); nickname and avatar in account `player`. At about 52 characters it fits Windows' 2560-byte limit.
- The access token lives **in memory only**, with its expiry.
- Tokens never reach the webview, SQLite, files or logs, and error strings go through a filter that removes `sslab_` values and `code=`.
- If the OS store fails on a write: the session lives in memory (`session_only: true`, with a note); there is never a file fallback.
- If it fails on a read: `store_unavailable: true`, with a note. That isn't signed out, and saved jobs aren't touched. A sign-in that replaces an older one revokes it where it can be read.

**Refresh.**
- One at a time, behind a mutex, reading the store's current token first. Only one process uses a session (see "Second launch").
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
- **Local mode** only in the `local-models` build: selected by default, labelled "local models" next to "server", existing options, no login, no spending. Normal users never see `NOT_INSTALLED`.
- Hints that only hold locally (the GPU note on a missing transcript, "nothing is sent anywhere", the API key hint) show only with the local mode.
- **Upload notice** before the first upload, remembered in `localStorage` by text version, so a material change asks again.

**Notice text**, matched to the 2026-11-04 `/terms` (section 13) and `/privacy` (sections 2.1, 4, 6):

> To analyse a match, Spaghetti Lab sends a 720p copy of the video, without audio, to the Spaghetti Showdown servers
> (hosted in the EU). The app deletes it when the analysis ends, whether it finished, failed or was cancelled; if the app
> is closed or offline then, it does so the next time it runs while you're signed in. If the connection drops just as an
> upload finishes, the app can't know the copy arrived: you can remove it from your files on the website. The
> result stays on the server for up to 24 hours so the app can fetch it, then only on this PC. If AI summaries are on,
> Anthropic receives game data extracted from the video, never the video. Results are generated automatically and can
> contain errors. You have 2 free analyses a week, then credits, which you can get on the website. An analysis that
> fails or is cancelled while waiting is given back; cancelling one that's running is not.
> [Terms](/terms) [Privacy](/privacy)

An Italian version is needed too, as those pages are in Italian. Things this text must not overstate:
- Neither page promises deletion, or gives a retention period for Lab videos. Deletion is **this app's** behaviour, so the text says "the app deletes it", and says when it can't: nothing on the server sweeps desktop uploads yet (see "For the server agent"). If a delete fails it's retried later as the same player.
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

## Local build

Our own build runs the research models on this PC: no sign-in, nothing spent. Builds we publish can't.

```powershell
cd app
npm run tauri:local         # dev: npx tauri dev --features local-models (or .\build-local.ps1)
npm run tauri:local:build   # a standalone exe, target\release\spaghetti-lab.exe (or .\build-local.ps1 -Build)
```

- **The switch** is the cargo feature `local-models`, default off. It compiles in `lab_local.rs`, the only code that starts
  `tools/ggst_logger/model/lab_analyze.py`, and registers `run_lab_analysis` and `cancel_lab_analysis`. `lab_config.local_models`
  tells the frontend, which hides the form otherwise. The form's code stays in the published frontend bundle, unreachable.
- **No variable is needed** when `G:\Projects\replanal-data\ggst_dataset\model` exists. The order: the Lab panel's folder,
  `SPAGHETTI_LAB_MODELS`, `GGST_MODEL_DIR`, `GGST_DATA`'s `ggst_dataset\model`, that default. The script and Python come from the checkout (`find_project_root`, `require_python`), so run it from there.
- **Default mode** is local, with "server" one click away. The AI summary checkbox stays and needs `ANTHROPIC_API_KEY` in the environment.
- **It doesn't update itself:** the update check is skipped, because an update would replace it with a published build.
- **The real run** (`16-lab-local-real`, 2026-11-05, 192 s clip at 720p 30 fps, full transcript, sharing the GPU and CPU with other jobs): hud 0 s (the pipeline's parquet is reused), detect 10 s, moves 144 s, transcript 202 s, facts 0 s, about 6 minutes. VRAM couldn't be split from the other jobs: the card peaked at 15.6 GB with one using about 13 GB and at 15.4 GB in the run below. It needs about 10 GB of free commit on top of the 6 GB to keep free, mostly the 8 DataLoader workers of the moves stage. `VM_WORKERS=2` takes that down but moves took 751 s, so 8 stays the default; with too little commit the run fails with an OpenCV "Insufficient memory" in the UI.
- **What it found:** the transcript's HUD inputs assumed a 60 fps video. On this 30 fps clip moves stopped at 83 s and states flattened to "attacking"; fixed in `lab_analyze.hud_rows` (72 moves became 207, reaching 183 s, and 22 openers linked to a move instead of 3). The hosted transcript gets the same fix with the service's next deploy, since desktop uploads are 30 fps.
- **Proof for the published builds:** `scripts/check_published_binary.py <exe>` fails when the binary has a string of `lab_local.rs`. It runs in `.gitea/workflows/ci.yml` on a default `cargo build`, in `build-windows.yml`, `.github/workflows/release.yml`, `release.sh` and `release.ps1` after the build. CI also runs `cargo check --features local-models` so our build keeps compiling, and `cargo test --lib published_build`.

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
answers: `analysis_starting`, `quota_exhausted`, `one_at_a_time`, `queue_full`, `too_long`, `plain_429`, `gateway`, `gateway_after_start` = charged, but nginx's 502 answers), `upload` (ok, `too_big`,
`html_413`, `storage_full`, `banned`), `uploadDelayMs`, `holdAt` (a job stops at a stage), `failAt`, `refundLost`, `job404`,
`result` (ok, bad, no_transcript, ai_withheld, gateway), `jobDelete` (ok, gateway), `fileDelete`, `tokenOutage`.

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

`14-lab` restarts on the `local-models` build (`connection.restart({ E2E_LOCAL_BUILD: "1" })`) to reach the local form, and back afterwards.
`15-lab-hosted` checks that the default build has neither the form nor the commands, with `SPAGHETTI_LAB_MODELS` set.
`16-lab-local-real` (`E2E_REAL_MODELS=1`) is the local build's real run, see "Local build".

### Live tests on dev

`app/src-tauri/src/lab_live_dev.rs`: ignored tests, not in CI, that need `LAB_LIVE_DEV=1` and refuse any other server:

```bash
cd app/src-tauri
LAB_LIVE_DEV=1 cargo test --lib lab_dev_0 -- --ignored --nocapture --test-threads=1   # one step at a time, 0 to 8
```

`lab_dev_real_clip` (with `LAB_DEV_CLIP=<video>`) is one whole video through the app's path, signing in and out by itself. Each run pays for an AI summary, so a marker in the work folder refuses a second one.

They run the shipped code (`run_remote` is `run_lab_analysis_remote` without its window), with the app's own environment
variables pointed at dev, a data folder in the temp dir and the credential-store service `SpaghettiLabLiveDevTest`. Only the
no-browser sign-in is test code. They print status codes, field names and counts. Steps 2, 3 and 4 are real analyses: each
costs server CPU and real AI-summary money, so run them one at a time, on purpose. Step 7 needs the account at 0 free and 0
credits, set by the server's admin. Step 8 ends the session and removes the credentials and the temp folder.

What real answers showed that the mock and the docs had wrong (all fixed):
- `ai_summary.langs.*.errors` and `warnings` are `{kind, text}` objects from the fact check, strings only when the call itself
  failed. `String(x)` showed "[object Object]" in the withheld details.
- A job's `created`, `started` and `finished` are numbers (epoch seconds), and `meta` has `filename`, later `size`, `fps`,
  `duration_s`, `openers`, `characters`, `capabilities`, `events`.

To get a job that fails, upload something the service's `probe` refuses (under 360p; the app's own check is skipped by going in at
its upload). A test pattern isn't reliable: 10 s of `testsrc2` at 720p ended `done`, see below.

### Open on the server side (found on dev, 2026-11-04)

- **A non-game video can end `done`:** 0 openers, invented characters, charged and not given back, and the AI summary was paid for.
- **The AI summary is often withheld after being paid for:** 1 of 8 language answers passed the fact check. Percentages the
  model works out itself (8 of 14 openers = 57%) count as `unsupported_percentage`, and multi-line notes as `note_without_timestamp`.
