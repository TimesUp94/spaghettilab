// ── The app's real client against the DEV server ─────────────────────────────
//
// Ignored tests, off by default and not in CI: they spend real AI-summary money and CPU on a shared server.
// Run them one at a time, in order, each only after the last one looked right (one analysis at a time):
//
//   cd app/src-tauri
//   LAB_LIVE_DEV=1 cargo test --lib lab_dev_0 -- --ignored --nocapture --test-threads=1
//
// 0 clips, 1 auth, 2 run, 3 reuse, 4 credits, 5 cancel, 6 failed, 7 exhausted (the account set to nothing
// left first), 8 stale session and clean-up. They run the shipped code: `run_remote` is the command without
// its window, `lab_auth::auth()` the app's session, with the app's own environment variables pointed at dev,
// a data folder in the temp dir and its own credential-store service. Production is refused.
//
// Only the no-browser sign-in is test code: the account file (outside the repo) gives a web JWT, which gets a
// login code, and from there on it's the app's exchange. The website's Discord click needs a person. Status
// codes, field names and counts are printed; never a token, the JWT, the account or an id.

use super::*;
use crate::lab_auth::TokenStore as _;
use rand::RngCore as _;

const API: &str = "https://dev.backend.spaghettishowdown.it";
const WEB: &str = "https://dev.spaghettishowdown.it";
const SERVICE: &str = "SpaghettiLabLiveDevTest";
/// A stuck run shouldn't sit on a shared server for ever.
const RUN_LIMIT: Duration = Duration::from_secs(30 * 60);

const REPLAY: &str = "live_match";
const SOURCE_DIR: &str = r"G:\Projects\replanal-data\ggst_dataset\degraded";
/// Name, source file, start second, length: the middle of a match, with several combo starts.
const CLIPS: [(&str, &str, u32, u32); 2] = [("match1.mp4", "inputs_20261022_181805.mp4", 30, 90), ("match2.mp4", "inputs_20261022_184800.mp4", 90, 90)];

fn work_dir() -> PathBuf {
    std::env::temp_dir().join("spaghetti-lab-live-dev")
}

fn clip(name: &str) -> String {
    work_dir().join("clips").join(name).to_string_lossy().to_string()
}

fn setup() {
    assert_eq!(std::env::var("LAB_LIVE_DEV").as_deref(), Ok("1"), "these tests spend real money: set LAB_LIVE_DEV=1");
    std::env::remove_var("SPAGHETTI_LAB_NO_PROD");
    std::env::set_var("SPAGHETTI_LAB_API_URL", API);
    std::env::set_var("SPAGHETTI_LAB_WEB_URL", WEB);
    std::env::set_var("SPAGHETTI_LAB_DATA_DIR", work_dir().join("data"));
    std::env::set_var("SPAGHETTI_LAB_KEYRING_SERVICE", SERVICE);
    assert_eq!(api::api_base().unwrap(), API);
    assert!(lab::lab_root().starts_with(work_dir()), "the cache must be the test's");
}

fn run<T>(f: impl std::future::Future<Output = T>) -> T {
    tauri::async_runtime::block_on(f)
}

/// Prints each check and fails the test at the end of a step, so that one answer doesn't hide the next.
#[derive(Default)]
struct Checks(Vec<String>);

impl Checks {
    fn check(&mut self, name: &str, ok: bool) {
        println!("[{}] {}", if ok { "pass" } else { "FAIL" }, name);
        if !ok {
            self.0.push(name.to_string());
        }
    }
    fn eq<T: PartialEq + std::fmt::Debug>(&mut self, name: &str, got: T, want: T) {
        let ok = got == want;
        self.check(&format!("{}: {:?}{}", name, got, if ok { String::new() } else { format!(" (expected {:?})", want) }), ok);
    }
    fn finish(self) {
        assert!(self.0.is_empty(), "failed: {:#?}", self.0);
    }
}

/// Field names and types, never a value.
fn shape(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(_) => "bool".into(),
        Value::Number(_) => "num".into(),
        Value::String(_) => "str".into(),
        Value::Array(a) => format!("[{}; {}]", a.first().map(shape).unwrap_or_default(), a.len()),
        Value::Object(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}: {}", k, shape(v))).collect::<Vec<_>>().join(", ")),
    }
}

fn show(name: &str, a: &Answer) {
    println!("  {}: {} retry_after={:?} {}", name, a.status, a.retry_after, if a.json.is_null() { format!("(not JSON, {} bytes)", a.text.len()) } else { shape(&a.json) });
}

fn account_fields() -> (String, String) {
    let path = dirs::home_dir().unwrap().join(".config").join("spaghetti-lab-dev").join("test-account.txt");
    let raw = std::fs::read_to_string(path).expect("the test account file");
    let text = serde_json::from_str::<Value>(&raw).ok().and_then(|v| v.get("body").and_then(|b| b.as_str()).map(str::to_string)).unwrap_or(raw);
    let field = |name: &str| {
        text.lines().find_map(|l| {
            let (k, v) = l.trim().split_once(':')?;
            k.trim().eq_ignore_ascii_case(name).then(|| v.split_whitespace().next().unwrap_or("").to_string())
        })
    };
    (field("email").expect("email field"), field("password").expect("password field"))
}

/// The website's part of the login, without a browser. The web JWT dies with this function.
async fn login_code(challenge: &str) -> String {
    let (email, password) = account_fields();
    let c = api::client_for(API);
    let login = c.post(format!("{}/playeruser_login", API)).json(&json!({"playerUserEmail": email, "password": password})).send().await.expect("login");
    let login = Answer::read(login).await.unwrap();
    println!("  /playeruser_login: {}", login.status);
    let jwt = login.json["user"]["token"].as_str().expect("user.token").to_string();
    let body = json!({"code_challenge": challenge, "code_challenge_method": "S256", "redirect_port": 54321});
    let a = Answer::read(c.post(format!("{}/api/desktop-auth/authorize", API)).bearer_auth(&jwt).json(&body).send().await.unwrap()).await.unwrap();
    println!("  /api/desktop-auth/authorize: {} fields={}", a.status, a.json.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>().join(",")).unwrap_or_default());
    a.json["code"].as_str().expect("code").to_string()
}

fn new_verifier() -> String {
    let mut b = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut b);
    lab_auth::b64url(&b)
}

/// The app's exchange and store, with a code from `login_code`.
async fn sign_in(auth: &Arc<Auth>) -> lab_auth::Account {
    let verifier = new_verifier();
    let code = login_code(&lab_auth::challenge_for(&verifier)).await;
    auth.exchange(&code, &verifier).await.expect("exchange")
}

fn store() -> lab_auth::KeyringStore {
    lab_auth::KeyringStore::new(SERVICE.to_string())
}

fn stored_refresh() -> Option<String> {
    store().get("refresh_token").unwrap()
}

/// A session left by an earlier step (the app's start: nothing in memory, the refresh token in the store).
fn session() -> Arc<Auth> {
    setup();
    lab_auth::auth().unwrap()
}

async fn quota_now(auth: &Arc<Auth>) -> Quota {
    let q = lab_auth::quota(auth).await.expect("quota").expect("quota answer");
    println!("  quota: free={}/{} credits={} resets_at={}", q.free_remaining, q.free_per_week, q.credits, q.week_resets_at);
    q
}

// ── a watched run ───────────────────────────────────────────────────────────

/// What the UI would see, with times since the start, and the ids the saved record held.
#[derive(Clone, Default)]
struct Watch {
    stages: Arc<Mutex<Vec<(f64, String)>>>,
    sources: Arc<Mutex<Vec<String>>>,
    positions: Arc<Mutex<Vec<i64>>>,
    upload_events: Arc<AtomicU64>,
    file_id: Arc<Mutex<Option<String>>>,
    job_id: Arc<Mutex<Option<String>>>,
}

impl Watch {
    fn emit(&self) -> Emit {
        let w = self.clone();
        let start = Instant::now();
        Arc::new(move |event, payload| {
            let t = start.elapsed().as_secs_f64();
            if event == "lab-quota" {
                if let Ok(q) = serde_json::from_value::<Quota>(payload) {
                    println!("  {:7.1}s quota event: free={} credits={}", t, q.free_remaining, q.credits);
                }
                return;
            }
            let stage = payload["stage"].as_str().unwrap_or("").to_string();
            if let Some(s) = payload["source"].as_str() {
                w.sources.lock().unwrap().push(s.to_string());
            }
            if let Some(p) = payload["position"].as_i64() {
                w.positions.lock().unwrap().push(p);
            }
            if stage == "upload" {
                w.upload_events.fetch_add(1, Ordering::SeqCst);
            }
            let mut stages = w.stages.lock().unwrap();
            if stages.last().map(|s| s.1.as_str()) != Some(stage.as_str()) {
                println!("  {:7.1}s stage {} ({})", t, stage, payload["line"].as_str().unwrap_or(""));
                stages.push((t, stage));
            }
        })
    }
    fn saw(&self, stage: &str) -> bool {
        self.stages.lock().unwrap().iter().any(|s| s.1 == stage)
    }
    fn current(&self) -> String {
        self.stages.lock().unwrap().last().map(|s| s.1.clone()).unwrap_or_default()
    }
    /// The file and job ids from the app's saved record, so the clean-up can be checked afterwards.
    fn follow_record(&self, id: &str) {
        let (w, id) = (self.clone(), id.to_string());
        std::thread::spawn(move || loop {
            if let Some(j) = read_job(&lab::lab_root(), &id) {
                if j.file_id.is_some() {
                    *w.file_id.lock().unwrap() = j.file_id;
                }
                if j.job_id.is_some() {
                    *w.job_id.lock().unwrap() = j.job_id;
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        });
    }
}

fn start_run(w: &Watch, id: &str, video: String, again: bool) -> tauri::async_runtime::JoinHandle<Result<Value, RemoteError>> {
    w.follow_record(id);
    let (emit, id) = (w.emit(), id.to_string());
    tauri::async_runtime::spawn(async move {
        let options = RemoteOptions { again: Some(again), video_path: Some(video) };
        run_remote(emit, String::new(), id, Some(options)).await
    })
}

/// What's left after a run ended: the uploaded file on the server, the record and the temp copy here.
async fn check_cleanup(c: &mut Checks, auth: &Arc<Auth>, w: &Watch, id: &str) {
    let root = lab::lab_root();
    c.check("the saved job record is gone", read_job(&root, id).is_none());
    let temp_left = std::fs::read_dir(root.join(".tmp")).map(|d| d.count()).unwrap_or(0);
    c.eq("temp copies left", temp_left, 0);
    c.eq("deletes waiting", read_pending(&root).len(), 0);
    let file = w.file_id.lock().unwrap().clone();
    match file {
        Some(f) => {
            // Already deleted by the run: a second DELETE finds nothing
            let a = auth.send_simple(Method::DELETE, &api::path_file(&f), CALL_TIMEOUT).await.expect("delete answer");
            c.eq("DELETE of the uploaded file after the run", a.status, 404);
        }
        None => println!("  (no file id reached the record)"),
    }
}

fn result_report(c: &mut Checks, v: &Value) {
    c.eq("schema", v["schema"].as_str(), Some("spaghetti-lab-result/1"));
    let openers = v["openers"].as_array().map_or(0, |o| o.len());
    c.check(&format!("openers present: {}", openers), openers > 0);
    println!("  top level: {}", v.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>().join(",")).unwrap_or_default());
    println!("  video={} duration_s={} fps={} characters={} capabilities={}", v["video"], v["duration_s"], v["fps"], v["characters"], v["capabilities"]);
    println!("  timings_s={}", v["timings_s"]);
    if let Some(o) = v["openers"].get(0) {
        println!("  openers[0]: {}", shape(o));
    }
    c.check("transcript matches capabilities", v["capabilities"]["transcript"].as_bool() == Some(v["transcript"].is_object()));
    println!("  facts: {}", shape(&v["facts"]).chars().take(600).collect::<String>());
    match v.get("ai_summary").filter(|a| !a.is_null()) {
        None => println!("  ai_summary: absent"),
        Some(a) => {
            println!("  ai_summary: provider={} model={} shape={}", a["provider"], a["model"], shape(a).chars().take(400).collect::<String>());
            for (lang, l) in a["langs"].as_object().into_iter().flatten() {
                println!("    {}: grounded={} text_chars={} errors={} warnings={}", lang, l["grounded"], l["text"].as_str().map_or(0, |t| t.chars().count()), l["errors"], l["warnings"].as_array().map_or(0, |w| w.len()));
                c.check(&format!("{}: grounded and text agree", lang), l["grounded"].as_bool() == Some(l["text"].is_string()));
            }
        }
    }
}

fn dump_result(name: &str, v: &Value) {
    let p = work_dir().join(name);
    let _ = std::fs::write(p, serde_json::to_string_pretty(v).unwrap_or_default());
}

async fn wait_run(c: &mut Checks, w: &Watch, h: tauri::async_runtime::JoinHandle<Result<Value, RemoteError>>, id: &str) -> Result<Value, RemoteError> {
    match tokio::time::timeout(RUN_LIMIT, h).await {
        Ok(r) => r.unwrap(),
        Err(_) => {
            c.check("the run ended in time", false);
            let _ = cancel_lab_analysis_remote(id.to_string());
            panic!("run over the limit at stage {}, cancelled", w.current());
        }
    }
}

fn timings(w: &Watch, total: f64) {
    let stages = w.stages.lock().unwrap();
    let parts: Vec<String> = stages.windows(2).map(|p| format!("{} {:.1}s", p[0].1, p[1].0 - p[0].0)).collect();
    println!("  timings: {} | total {:.1}s", parts.join(", "), total);
}

async fn post_start(auth: &Arc<Auth>, file_id: &str) -> Answer {
    let url = auth.url(api::PATH_OPENERS);
    let body = json!({"fileId": file_id});
    auth.send(move |cl| {
        let (u, b) = (url.clone(), body.clone());
        async move { Ok(cl.post(u).json(&b).timeout(Duration::from_secs(120))) }
    })
    .await
    .expect("start answer")
}

// ── steps ───────────────────────────────────────────────────────────────────

/// The clips, by stream copy from the read-only dataset, and two 10 s test patterns (step 6).
#[test]
#[ignore]
fn lab_dev_0_clips() {
    setup();
    let dir = work_dir().join("clips");
    std::fs::create_dir_all(&dir).unwrap();
    let ffmpeg = crate::ffmpeg_bin().expect("ffmpeg");
    for (name, source, start, len) in CLIPS {
        let out = dir.join(name);
        let src = Path::new(SOURCE_DIR).join(source);
        let st = std::process::Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-ss", &start.to_string(), "-t", &len.to_string(), "-i"])
            .arg(&src)
            .args(["-map", "0:v:0", "-an", "-c", "copy", "-movflags", "+faststart"])
            .arg(&out)
            .status()
            .unwrap();
        assert!(st.success());
        println!("{}: {} bytes", name, std::fs::metadata(&out).unwrap().len());
    }
    for (name, size) in [("pattern_1080p.mp4", "1920x1080"), ("pattern_240p.mp4", "320x240")] {
        let out = dir.join(name);
        let st = std::process::Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", &format!("testsrc2=size={}:rate=30", size), "-t", "10", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", "-pix_fmt", "yuv420p"])
            .arg(&out)
            .status()
            .unwrap();
        assert!(st.success());
        println!("{}: {} bytes", name, std::fs::metadata(&out).unwrap().len());
    }
}

#[test]
#[ignore]
fn lab_dev_1_auth() {
    setup();
    run(async {
        let mut c = Checks::default();
        let auth = lab_auth::auth().unwrap();
        c.check("signed out before", !lab_auth::lab_account().await.unwrap().signed_in);

        // A raw exchange and revoke beside the app's session: the real shapes
        let verifier = new_verifier();
        let code = login_code(&lab_auth::challenge_for(&verifier)).await;
        let raw_client = auth.client();
        let body = json!({"grant_type": "authorization_code", "code": code, "code_verifier": verifier});
        let raw = Answer::read(raw_client.post(auth.url(api::PATH_TOKEN)).json(&body).send().await.unwrap()).await.unwrap();
        show("token (raw, values hidden)", &raw);
        let t: api::TokenResponse = raw.parse().expect("TokenResponse parses");
        c.eq("token_type", t.token_type.as_deref(), Some("Bearer"));
        c.eq("scope", t.scope.as_deref(), Some("lab"));
        println!("  expires_in={}", t.expires_in);
        let reused = Answer::read(raw_client.post(auth.url(api::PATH_TOKEN)).json(&body).send().await.unwrap()).await.unwrap();
        c.eq("a code used twice", (reused.status, reused.error().map(str::to_string)), (400, Some("invalid_grant".to_string())));
        let r = json!({"refresh_token": t.refresh_token});
        let rv = Answer::read(raw_client.post(auth.url(api::PATH_REVOKE)).json(&r).send().await.unwrap()).await.unwrap();
        show("revoke", &rv);
        c.eq("revoke", rv.status, 200);
        let rv2 = Answer::read(raw_client.post(auth.url(api::PATH_REVOKE)).json(&r).send().await.unwrap()).await.unwrap();
        c.eq("revoke again", rv2.status, 200);
        let dead = json!({"grant_type": "refresh_token", "refresh_token": t.refresh_token});
        let d = Answer::read(raw_client.post(auth.url(api::PATH_TOKEN)).json(&dead).send().await.unwrap()).await.unwrap();
        c.eq("refresh after revoke", (d.status, d.error().map(str::to_string)), (400, Some("invalid_grant".to_string())));

        // The app's sign-in: exchange, credential store, what the webview gets
        let acc = sign_in(&auth).await;
        c.check("signed in", acc.signed_in && !acc.session_only && !acc.store_unavailable);
        c.check("has a nickname", acc.nickname.is_some());
        println!("  has_avatar={}", acc.avatar.is_some());
        let first = stored_refresh().expect("refresh token in the credential store");
        c.check("refresh token prefix", first.starts_with("sslab_rt_"));

        // Status and quota, raw and through the commands
        let s = Answer::read(auth.client().get(auth.url(api::PATH_STATUS)).send().await.unwrap()).await.unwrap();
        show("status", &s);
        println!("  capabilities={}", s.json["capabilities"]);
        println!("  running={} queued={} max_concurrent={} max_queue={}", s.json["running"], s.json["queued"], s.json["max_concurrent"], s.json["max_queue"]);
        let st = lab_auth::lab_service_status().await.unwrap();
        c.check("status available", st.available);
        c.check("status has capabilities", st.capabilities.is_some());
        c.eq("free_per_week in status", st.free_per_week, Some(2));
        let q = auth.send_simple(Method::GET, api::PATH_QUOTA, CALL_TIMEOUT).await.unwrap();
        show("quota", &q);
        c.check("the command reads the quota", lab_auth::lab_quota().await.unwrap().is_some());
        let before = quota_now(&auth).await;

        // Rotation, one refresh for eight callers: a second one would present a rotated token and end the session
        let access = auth.access_token(None).await.unwrap();
        c.check("access token prefix", access.starts_with("sslab_at_"));
        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let (a, u) = (auth.clone(), access.clone());
                tauri::async_runtime::spawn(async move { a.access_token(Some(u)).await })
            })
            .collect();
        let mut got = Vec::new();
        for t in tasks {
            got.push(t.await.unwrap().expect("refresh"));
        }
        got.dedup();
        c.eq("distinct tokens among eight callers", got.len(), 1);
        c.check("access token rotated", got[0] != access);
        let second = stored_refresh().unwrap();
        c.check("refresh token rotated and stored", second != first);
        let old = Answer::read(auth.client().get(auth.url(api::PATH_QUOTA)).bearer_auth(&access).send().await.unwrap()).await.unwrap();
        println!("  the access token before the rotation: {} error={:?}", old.status, old.error());
        let q2 = quota_now(&auth).await;
        c.eq("session alive after the concurrent refresh", q2.credits, before.credits);

        // 401 invalid_token: refresh once, retry once
        let bad = Answer::read(auth.client().get(auth.url(api::PATH_QUOTA)).bearer_auth("sslab_at_notarealtoken").send().await.unwrap()).await.unwrap();
        c.eq("a made-up access token", (bad.status, bad.error().map(str::to_string)), (401, Some("invalid_token".to_string())));
        auth.set_access_for_test("sslab_at_notarealtoken").await;
        c.check("401 is refreshed and retried", lab_auth::lab_quota().await.unwrap().is_some());
        c.check("that refresh rotated", stored_refresh().unwrap() != second);

        // 403 scope: a bug to report, not a sign-out
        let token = auth.access_token(None).await.unwrap();
        let raw403 = Answer::read(auth.client().get(auth.url("/api/support/me")).bearer_auth(&token).send().await.unwrap()).await.unwrap();
        c.eq("raw /api/support/me", (raw403.status, raw403.error().map(str::to_string)), (403, Some("scope".to_string())));
        let mapped = auth.send_simple(Method::GET, "/api/support/me", CALL_TIMEOUT).await;
        c.eq("mapped", mapped.err().map(|e| e.code), Some("scope".to_string()));
        c.check("still signed in after a 403", lab_auth::lab_account().await.unwrap().signed_in);
        c.finish();
    });
}

/// A full run through the app's pipeline: the first free analysis.
#[test]
#[ignore]
fn lab_dev_2_run() {
    let auth = session();
    run(async {
        let mut c = Checks::default();
        c.check("a restarted app finds the session", lab_auth::lab_account().await.unwrap().signed_in);
        let before = quota_now(&auth).await;
        c.check("a free analysis is left", before.free_remaining > 0);
        let w = Watch::default();
        let t0 = Instant::now();
        let h = start_run(&w, REPLAY, clip("match1.mp4"), false);
        // The quota while it runs
        let mut seen_running = false;
        while !h.inner().is_finished() {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if !seen_running && lab_remote_running().map_or(false, |r| r.phase == "running" || r.phase == "queued") {
                seen_running = true;
                let during = quota_now(&auth).await;
                c.eq("one free analysis taken while it runs", during.free_remaining, before.free_remaining - 1);
                c.eq("credits untouched", during.credits, before.credits);
            }
        }
        let out = wait_run(&mut c, &w, h, REPLAY).await;
        timings(&w, t0.elapsed().as_secs_f64());
        c.eq("source", w.sources.lock().unwrap().first().cloned(), Some("free".to_string()));
        for stage in ["check", "transcode", "upload", "start"] {
            c.check(&format!("saw stage {}", stage), w.saw(stage));
        }
        println!("  upload events: {}", w.upload_events.load(Ordering::SeqCst));
        let v = match out {
            Ok(v) => v,
            Err(e) => {
                c.check(&format!("the run succeeded, got {}: {}", e.code, e.message), false);
                check_cleanup(&mut c, &auth, &w, REPLAY).await;
                c.finish();
                return;
            }
        };
        let cached = lab::read_cached(&lab::lab_root(), REPLAY).unwrap();
        c.check("the cache holds the validated result", cached.as_ref() == Some(&v));
        result_report(&mut c, &v);
        dump_result("result_run1.json", &v);
        let after = quota_now(&auth).await;
        c.eq("quota after", (after.free_remaining, after.credits), (before.free_remaining - 1, before.credits));
        check_cleanup(&mut c, &auth, &w, REPLAY).await;
        c.finish();
    });
}

/// The second analysis: the last free one. The same fileId again, while running and when done, is free.
/// Uses the app's upload and start directly, because a whole run deletes its file.
#[test]
#[ignore]
fn lab_dev_3_reuse() {
    let auth = session();
    run(async {
        let mut c = Checks::default();
        let id = "live_reuse";
        let before = quota_now(&auth).await;
        let w = Watch::default();
        let (tx, rx) = watch::channel(false);
        let ctx = Ctx { auth: auth.clone(), root: lab::lab_root(), id: id.into(), emit: w.emit(), cancel: rx, poll: POLL, poll_error: POLL_AFTER_ERROR };
        let video = PathBuf::from(clip("match1.mp4"));
        let total = std::fs::metadata(&video).unwrap().len();
        let t0 = Instant::now();
        let answer = upload_answer(&ctx, &video, total).await.expect("upload");
        println!("  upload: {:.1} s for {} bytes", t0.elapsed().as_secs_f64(), total);
        show("upload", &answer);
        c.eq("compression_status", answer.json["data"]["compression_status"].as_str(), Some("skipped"));
        let file_id = classify_upload(&answer).expect("file id");

        let first = post_start(&auth, &file_id).await;
        show("POST /openers", &first);
        c.eq("POST status", first.status, 202);
        c.eq("analysis_source", first.json["analysis_source"].as_str(), Some("free"));
        c.eq("free count in the answer", first.json["free_remaining"].as_i64(), Some(before.free_remaining - 1));
        let job = match classify_start(&first) {
            StartOutcome::Started(j, q) => {
                c.check("the answer's quota parses", q.is_some());
                j
            }
            _ => panic!("the start answer isn't a job"),
        };
        println!("  status={} stage={:?} position={:?}", job.status, job.stage, job.position);

        // Asked again while it runs
        let during = post_start(&auth, &file_id).await;
        show("POST again while it runs", &during);
        c.eq("again while running", (during.status, during.json["reused"].as_bool()), (200, Some(true)));
        c.check("the same job", during.json["id"].as_str() == Some(job.id.as_str()));
        c.eq("no second charge", during.json["free_remaining"].as_i64(), Some(before.free_remaining - 1));

        // To its end, by polling (the app's follow would also fetch and store)
        let mut last = String::new();
        let t1 = Instant::now();
        let view = loop {
            let a = auth.send_simple(Method::GET, &api::path_job(&job.id), CALL_TIMEOUT).await.expect("poll");
            let j: JobView = a.parse().expect("JobView parses");
            let now = format!("{} {:?}", j.status, j.stage);
            if now != last {
                println!("  {:7.1}s {} {}", t1.elapsed().as_secs_f64(), now, shape(&a.json).chars().take(300).collect::<String>());
                last = now;
            }
            if !matches!(j.status.as_str(), "queued" | "running") {
                break j;
            }
            assert!(t1.elapsed() < RUN_LIMIT, "over the limit");
            tokio::time::sleep(POLL).await;
        };
        c.eq("job status", view.status.as_str(), "done");

        let done = post_start(&auth, &file_id).await;
        show("POST again when done", &done);
        c.eq("again when done", (done.status, done.json["reused"].as_bool()), (200, Some(true)));
        let after_reuse = quota_now(&auth).await;
        c.eq("reuse charged nothing", (after_reuse.free_remaining, after_reuse.credits), (before.free_remaining - 1, before.credits));

        // A saved record at this point is what the app resumes from: its own path collects and cleans
        let saved = SavedJob { file_id: Some(file_id.clone()), job_id: Some(job.id.clone()), player_user_id: auth.player_id().await, ..Default::default() };
        save_job(&ctx.root, id, &saved);
        *w.file_id.lock().unwrap() = Some(file_id);
        let v = continue_saved(&ctx, saved).await.expect("the resumed job gives the result");
        result_report(&mut c, &v);
        dump_result("result_run2.json", &v);
        check_cleanup(&mut c, &auth, &w, id).await;
        drop(tx);
        c.finish();
    });
}

/// The third analysis: free ones gone, a credit. A cached replay refuses first, without spending.
#[test]
#[ignore]
fn lab_dev_4_credits() {
    let auth = session();
    run(async {
        let mut c = Checks::default();
        let before = quota_now(&auth).await;
        c.eq("no free analysis left", before.free_remaining, 0);
        let w0 = Watch::default();
        let refused = start_run(&w0, REPLAY, clip("match2.mp4"), false).await.unwrap();
        c.eq("a cached replay asks first", refused.err().map(|e| e.code), Some("already_cached".to_string()));
        c.eq("no upload for that", w0.upload_events.load(Ordering::SeqCst), 0);

        let w = Watch::default();
        let t0 = Instant::now();
        let h = start_run(&w, REPLAY, clip("match2.mp4"), true);
        let out = wait_run(&mut c, &w, h, REPLAY).await;
        timings(&w, t0.elapsed().as_secs_f64());
        c.eq("source", w.sources.lock().unwrap().first().cloned(), Some("credit".to_string()));
        let after = quota_now(&auth).await;
        match out {
            Ok(v) => {
                result_report(&mut c, &v);
                dump_result("result_run3.json", &v);
                c.eq("one credit spent", (after.free_remaining, after.credits), (0, before.credits - 1));
            }
            Err(e) => c.check(&format!("the run succeeded, got {}: {}", e.code, e.message), false),
        }
        check_cleanup(&mut c, &auth, &w, REPLAY).await;
        c.finish();
    });
}

/// Cancelling before the charge (nothing spent), then cancelling a running analysis (not given back).
#[test]
#[ignore]
fn lab_dev_5_cancel() {
    let auth = session();
    run(async {
        let mut c = Checks::default();
        let id = "live_cancel";
        let status = lab_auth::lab_service_status().await.unwrap();
        println!("  service: running={:?} queued={:?}", status.running, status.queued);

        // During the upload
        let before = quota_now(&auth).await;
        let w = Watch::default();
        let h = start_run(&w, id, clip("match2.mp4"), false);
        while !w.saw("upload") {
            assert!(!h.inner().is_finished(), "the run ended early");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let answer = cancel_lab_analysis_remote(id.to_string()).expect("cancel");
        c.eq("phase known during the upload", answer.phase.as_str(), "before_charge");
        let e = h.await.unwrap().err().expect("cancelled");
        c.eq("ended", (e.code.as_str(), e.phase.as_deref()), ("cancelled", Some("before_charge")));
        let q = quota_now(&auth).await;
        c.eq("nothing spent", (q.free_remaining, q.credits), (before.free_remaining, before.credits));
        check_cleanup(&mut c, &auth, &w, id).await;

        // While running, in a model stage
        let before = quota_now(&auth).await;
        let w = Watch::default();
        let t0 = Instant::now();
        let h = start_run(&w, id, clip("match2.mp4"), false);
        while !matches!(w.current().as_str(), "hud" | "detect" | "moves" | "facts") {
            assert!(!h.inner().is_finished(), "the run ended before it could be cancelled");
            assert!(t0.elapsed() < RUN_LIMIT, "over the limit");
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        println!("  stage when cancelling: {} at {:.1}s; positions seen: {:?}", w.current(), t0.elapsed().as_secs_f64(), w.positions.lock().unwrap());
        let during = quota_now(&auth).await;
        c.eq("charged while running", during.free_remaining + during.credits, before.free_remaining + before.credits - 1);
        // What the warning dialog asks first
        let phase = lab_remote_cancel_phase(id.to_string()).await.expect("cancel phase");
        c.eq("the warning's phase", phase.phase.as_str(), "running");
        let job_id = w.job_id.lock().unwrap().clone().expect("job id in the record");
        let t1 = Instant::now();
        cancel_lab_analysis_remote(id.to_string()).expect("cancel");
        let e = h.await.unwrap().err().expect("cancelled");
        println!("  cancel took {:.1}s", t1.elapsed().as_secs_f64());
        c.eq("ended", (e.code.as_str(), e.phase.as_deref()), ("cancelled", Some("running")));
        c.eq("not given back", e.refunded, Some(false));
        let q = quota_now(&auth).await;
        c.eq("still charged", q.free_remaining + q.credits, before.free_remaining + before.credits - 1);
        let j = auth.send_simple(Method::GET, &api::path_job(&job_id), CALL_TIMEOUT).await.expect("job");
        show("the job after the cancel", &j);
        c.eq("job status", j.json["status"].as_str(), Some("cancelled"));
        check_cleanup(&mut c, &auth, &w, id).await;
        tokio::time::sleep(Duration::from_secs(5)).await;
        let s = lab_auth::lab_service_status().await.unwrap();
        println!("  service afterwards: running={:?} queued={:?}", s.running, s.queued);
        c.finish();
    });
}

/// A job that fails: the analysis comes back. The service refuses a 240p video right after its download,
/// before the chain runs, so no AI money. The app itself refuses under 360p, so this goes in at its upload.
/// A test pattern at 720p isn't a safe way to get a failure: it can end `done` with invented characters,
/// charged and with an AI summary paid for. If the chain starts anyway, this cancels at once.
/// Also the real encode, on this PC only: the 1080p pattern to 720p.
#[test]
#[ignore]
fn lab_dev_6_failed() {
    let auth = session();
    run(async {
        let mut c = Checks::default();
        let id = "live_failed";
        let before = quota_now(&auth).await;
        let w = Watch::default();
        let (tx, rx) = watch::channel(false);
        let ctx = Ctx { auth: auth.clone(), root: lab::lab_root(), id: id.into(), emit: w.emit(), cancel: rx, poll: POLL, poll_error: POLL_AFTER_ERROR };

        let ffmpeg = crate::ffmpeg_bin().expect("ffmpeg");
        let big = PathBuf::from(clip("pattern_1080p.mp4"));
        let info = probe(&ffmpeg, &big).await.expect("probe");
        c.check("the pattern needs an encode", needs_encode(info, std::fs::metadata(&big).unwrap().len()));
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("replay.mp4");
        let t0 = Instant::now();
        let sent = prepare(&ctx, &ffmpeg, &big, info, &out).await.expect("encode");
        let copy = probe(&ffmpeg, &out).await.expect("probe the copy");
        println!("  encode: {:.1} s, {} bytes", t0.elapsed().as_secs_f64(), sent);
        c.eq("the copy's height", copy.height, 720);

        let video = PathBuf::from(clip("pattern_240p.mp4"));
        let total = std::fs::metadata(&video).unwrap().len();
        let file_id = upload(&ctx, &video, total).await.expect("upload");
        *w.file_id.lock().unwrap() = Some(file_id.clone());
        let saved = SavedJob { file_id: Some(file_id.clone()), player_user_id: auth.player_id().await, ..Default::default() };
        save_job(&ctx.root, id, &saved);
        let t1 = Instant::now();
        let (c2, f2) = (ctx.clone(), file_id.clone());
        let h = tauri::async_runtime::spawn(async move { start_and_follow(&c2, saved, &f2).await });
        let mut charged = false;
        while !h.inner().is_finished() {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if !charged && w.saw("start") {
                charged = true;
                let during = quota_now(&auth).await;
                c.eq("charged while it runs", during.free_remaining + during.credits, before.free_remaining + before.credits - 1);
            }
            if matches!(w.current().as_str(), "hud" | "detect" | "moves" | "facts" | "summary") {
                c.check("the chain never starts", false);
                let _ = tx.send(true);
                break;
            }
        }
        let e = h.await.unwrap().err().expect("a failed job");
        println!("  ended after {:.1} s: {} ({})", t1.elapsed().as_secs_f64(), e.code, e.message);
        c.eq("error code", e.code.as_str(), "job_failed");
        c.eq("analysis_refunded", e.refunded, Some(true));
        let q = quota_now(&auth).await;
        c.eq("count restored", (q.free_remaining, q.credits), (before.free_remaining, before.credits));
        check_cleanup(&mut c, &auth, &w, id).await;
        c.finish();
    });
}

/// The account set to nothing left (by the server's admin): the server's answer and the app's.
#[test]
#[ignore]
fn lab_dev_7_exhausted() {
    let auth = session();
    run(async {
        let mut c = Checks::default();
        let id = "live_exhausted";
        let q = quota_now(&auth).await;
        c.eq("the account has nothing left", (q.free_remaining, q.credits), (0, 0));

        // The app's own check: no upload
        let w0 = Watch::default();
        let e = start_run(&w0, id, clip("match1.mp4"), false).await.unwrap().err().expect("refused");
        c.eq("the app stops first", e.code.as_str(), "quota_exhausted");
        c.eq("no upload", w0.upload_events.load(Ordering::SeqCst), 0);
        c.check("it carries the quota", e.quota.is_some());
        c.check("it remembers the error for the screen", lab_remote_take_error(id.to_string()).is_some());

        // The server's answer, past that check
        let (tx, rx) = watch::channel(false);
        let w = Watch::default();
        let ctx = Ctx { auth: auth.clone(), root: lab::lab_root(), id: id.into(), emit: w.emit(), cancel: rx, poll: POLL, poll_error: POLL_AFTER_ERROR };
        let video = PathBuf::from(clip("pattern_240p.mp4"));
        let total = std::fs::metadata(&video).unwrap().len();
        let file_id = upload(&ctx, &video, total).await.expect("upload");
        *w.file_id.lock().unwrap() = Some(file_id.clone());
        let a = post_start(&auth, &file_id).await;
        show("POST /openers", &a);
        c.eq("status and error", (a.status, a.error().map(str::to_string)), (429, Some("quota_exhausted".to_string())));
        c.check("Retry-After", a.retry_after.is_some());
        let refused = match classify_start(&a) {
            StartOutcome::Fail(e) => e,
            _ => panic!("not refused"),
        };
        c.eq("the app's reading", refused.code.as_str(), "quota_exhausted");
        c.check("with the quota fields", refused.quota.is_some());
        delete_file(&auth, &ctx.root, &file_id, None).await;
        check_cleanup(&mut c, &auth, &w, id).await;
        drop(tx);

        // The page the offer opens
        let page = api::web_page("support").unwrap();
        let r = api::client_for(WEB).get(format!("{}{}", WEB, page)).send().await.expect("support page");
        c.eq("/support on the website", r.status().as_u16(), 200);
        c.finish();
    });
}

/// Ends the session, so last: sign-out with revoke; then a new session, a rotation, and a stale refresh token
/// (it ends the whole session on the server, and the app then shows signed out). Removes the credentials.
#[test]
#[ignore]
fn lab_dev_8_stale_and_cleanup() {
    let auth = session();
    run(async {
        let mut c = Checks::default();
        c.check("the session still works", matches!(lab_auth::lab_quota().await, Ok(Some(_))));
        let current = stored_refresh().expect("a session");
        let out = lab_auth::lab_sign_out().await;
        c.check("sign-out", out.is_ok());
        c.check("signed out", !lab_auth::lab_account().await.unwrap().signed_in);
        c.check("the credential store is empty", stored_refresh().is_none() && store().get("player").unwrap().is_none());
        let dead = json!({"grant_type": "refresh_token", "refresh_token": current});
        let d = Answer::read(auth.client().post(auth.url(api::PATH_TOKEN)).json(&dead).send().await.unwrap()).await.unwrap();
        c.eq("refresh after the sign-out", (d.status, d.error().map(str::to_string)), (400, Some("invalid_grant".to_string())));

        // A new session, rotated once, then the old token presented
        sign_in(&auth).await;
        let a = stored_refresh().unwrap();
        let access = auth.access_token(None).await.unwrap();
        auth.access_token(Some(access)).await.expect("rotation");
        c.check("rotated", stored_refresh().unwrap() != a);
        let stale = Arc::new(lab_auth::MemoryStore::default());
        stale.set("refresh_token", &a).unwrap();
        let other = Auth::new(API.to_string(), stale.clone());
        let e = other.access_token(None).await.err().expect("a stale token refused");
        c.eq("the stale token, through the app", e.code.as_str(), "signed_out");
        c.check("that instance forgot it", stale.get("refresh_token").unwrap().is_none());

        // The real session is over too, and the app says so
        auth.set_access_for_test("sslab_at_notarealtoken").await;
        let e = lab_auth::lab_quota().await.err().map(|e| e.code);
        c.eq("the current session after that", e.as_deref(), Some("signed_out"));
        c.check("signed out in the app", !lab_auth::lab_account().await.unwrap().signed_in);

        let s = store();
        s.del("refresh_token").unwrap();
        s.del("player").unwrap();
        c.check("nothing left in the test service", stored_refresh().is_none());
        c.finish();
    });
    let _ = std::fs::remove_dir_all(work_dir());
    println!("removed {}", work_dir().display());
}
