// ── spaghettishowdown.it sign-in for the Lab tab ─────────────────────────────
//
// Loopback login with PKCE, token exchange, refresh, revoke, the OS credential store. Contract and
// reasons: docs/desktop-login.md. Rules that shape this file:
// - Tokens never leave Rust: not to the webview, SQLite, files or logs. The webview gets `Account`.
// - The refresh token lives in the OS store; the access token only in memory. No file fallback: if
//   the store fails, the session lives in memory until the app closes (`session_only`).
// - Presenting an already-rotated refresh token ends the whole session on the server, so refresh is
//   serialized behind one mutex, runs in its own task (a cancelled caller can't lose the answer),
//   writes the new token to the store before anything uses it, and never blindly retries a request
//   that may have reached the server. A lost answer means signing in again. Only one instance of the
//   app uses a session: lib.rs keeps it single-instance, and an isolated one (SPAGHETTI_LAB_DATA_DIR)
//   gets its own credential-store service (lab_api::keyring_service).

use crate::lab_api::{self as api, Answer, Player, RemoteError, TokenResponse};
use base64::Engine as _;
use futures_util::stream::{FuturesUnordered, StreamExt as _};
use rand::RngCore as _;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::watch;

const ACCOUNT_REFRESH: &str = "refresh_token";
const ACCOUNT_PLAYER: &str = "player";
/// A first sign-in may include a Discord login.
const LOGIN_TIMEOUT: Duration = Duration::from_secs(600);
/// Refresh before an access token gets this close to its end.
const ACCESS_MARGIN: Duration = Duration::from_secs(60);
/// One connection to the login listener gets this long to send its request, in all.
const CALLBACK_READ: Duration = Duration::from_secs(5);

// ── credential store ────────────────────────────────────────────────────────

pub trait TokenStore: Send + Sync {
    fn get(&self, account: &str) -> Result<Option<String>, String>;
    fn set(&self, account: &str, value: &str) -> Result<(), String>;
    fn del(&self, account: &str) -> Result<(), String>;
}

/// The OS store: Credential Manager, Keychain or Secret Service.
pub struct KeyringStore {
    service: String,
}

impl KeyringStore {
    pub fn new(service: String) -> Self {
        KeyringStore { service }
    }
    fn entry(&self, account: &str) -> Result<keyring::Entry, String> {
        keyring::Entry::new(&self.service, account).map_err(|e| e.to_string())
    }
}

impl TokenStore for KeyringStore {
    fn get(&self, account: &str) -> Result<Option<String>, String> {
        match self.entry(account)?.get_password() {
            Ok(v) => Ok(Some(v)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }
    fn set(&self, account: &str, value: &str) -> Result<(), String> {
        self.entry(account)?.set_password(value).map_err(|e| e.to_string())
    }
    fn del(&self, account: &str) -> Result<(), String> {
        match self.entry(account)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// Tests only (and a way to simulate a failing store).
#[cfg(test)]
#[derive(Default)]
pub struct MemoryStore {
    pub values: Mutex<HashMap<String, String>>,
    pub fail_writes: std::sync::atomic::AtomicBool,
    pub fail_reads: std::sync::atomic::AtomicBool,
    pub fail_deletes: std::sync::atomic::AtomicBool,
}

#[cfg(test)]
impl TokenStore for MemoryStore {
    fn get(&self, account: &str) -> Result<Option<String>, String> {
        if self.fail_reads.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("store unavailable".into());
        }
        Ok(self.values.lock().unwrap().get(account).cloned())
    }
    fn set(&self, account: &str, value: &str) -> Result<(), String> {
        if self.fail_writes.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("store unavailable".into());
        }
        self.values.lock().unwrap().insert(account.into(), value.into());
        Ok(())
    }
    fn del(&self, account: &str) -> Result<(), String> {
        if self.fail_deletes.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("store unavailable".into());
        }
        self.values.lock().unwrap().remove(account);
        Ok(())
    }
}

// ── session ─────────────────────────────────────────────────────────────────

/// All the webview ever learns about the session.
#[derive(Debug, Clone, Serialize, PartialEq, Default)]
pub struct Account {
    pub signed_in: bool,
    pub nickname: Option<String>,
    pub avatar: Option<String>,
    /// The OS credential store failed: this session ends when the app closes.
    pub session_only: bool,
    /// The OS credential store can't be read right now, so whether a sign-in is kept there is unknown:
    /// not the same as signed out. Saved jobs wait, and nothing is forgotten.
    pub store_unavailable: bool,
}

#[derive(Default)]
struct Session {
    access: Option<String>,
    expires_at: Option<Instant>,
    /// Only when the OS store failed (`session_only`)
    memory_refresh: Option<String>,
    player: Option<Player>,
    session_only: bool,
    /// `/token` said Retry-After: no refresh before this
    not_before: Option<Instant>,
}

pub struct Auth {
    api: String,
    client: reqwest::Client,
    store: Arc<dyn TokenStore>,
    /// The single mutex: held across the whole refresh request.
    session: tokio::sync::Mutex<Session>,
}

impl Auth {
    pub fn new(api: String, store: Arc<dyn TokenStore>) -> Arc<Auth> {
        let client = api::client_for(&api);
        Arc::new(Auth { api, client, store, session: tokio::sync::Mutex::new(Session::default()) })
    }

    pub fn client(&self) -> reqwest::Client {
        self.client.clone()
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.api, path)
    }

    pub async fn account(&self) -> Account {
        let s = self.session.lock().await;
        self.account_of(&s)
    }

    fn account_of(&self, s: &Session) -> Account {
        let stored = self.store.get(ACCOUNT_REFRESH);
        let signed_in = s.access.is_some() || s.memory_refresh.is_some() || matches!(stored, Ok(Some(_)));
        let player = if signed_in { s.player.clone().or_else(|| self.stored_player()) } else { None };
        Account {
            signed_in,
            nickname: player.as_ref().and_then(|p| p.nickname.clone()),
            // only an https image: it goes straight into an <img>
            avatar: player.as_ref().and_then(|p| p.avatar.clone()).filter(|a| a.starts_with("https://")),
            session_only: s.session_only,
            store_unavailable: stored.is_err(),
        }
    }

    fn stored_player(&self) -> Option<Player> {
        self.store.get(ACCOUNT_PLAYER).ok().flatten().and_then(|t| serde_json::from_str(&t).ok())
    }

    /// The signed-in player's id (saved jobs and pending file deletes belong to one player).
    pub async fn player_id(&self) -> Option<String> {
        let s = self.session.lock().await;
        if !self.account_of(&s).signed_in {
            return None;
        }
        s.player.clone().or_else(|| self.stored_player()).map(|p| p.player_user_id)
    }

    /// Keep the tokens of a /token answer. The refresh token is written before anyone can use the
    /// new access token; a store that fails moves the session into memory.
    fn adopt(&self, s: &mut Session, t: TokenResponse) {
        match self.store.set(ACCOUNT_REFRESH, &t.refresh_token) {
            Ok(()) => {
                s.memory_refresh = None;
                s.session_only = false;
                if let Ok(j) = serde_json::to_string(&t.player) {
                    let _ = self.store.set(ACCOUNT_PLAYER, &j);
                }
            }
            Err(e) => {
                eprintln!("[LAB] the credential store failed, keeping this session in memory: {}", api::scrub(&e));
                // What's stored was just rotated: presenting it later would end this session
                let _ = self.store.del(ACCOUNT_REFRESH);
                s.memory_refresh = Some(t.refresh_token.clone());
                s.session_only = true;
            }
        }
        s.player = Some(t.player);
        s.expires_at = Some(Instant::now() + Duration::from_secs(t.expires_in));
        s.access = Some(t.access_token);
        s.not_before = None;
    }

    fn forget(&self, s: &mut Session) -> Result<(), String> {
        *s = Session::default();
        let refresh = self.store.del(ACCOUNT_REFRESH);
        let _ = self.store.del(ACCOUNT_PLAYER);
        refresh
    }

    /// The server said this session is over (or the user disconnected this PC on the website).
    pub async fn clear(&self) {
        let mut s = self.session.lock().await;
        if let Err(e) = self.forget(&mut s) {
            eprintln!("[LAB] the credential store kept an ended sign-in: {}", api::scrub(&e));
        }
    }

    /// One refresh, the session lock held. Connection failures that never reached the server are
    /// retried; anything that may have reached it isn't (the token may already be rotated).
    async fn refresh_locked(&self, s: &mut Session) -> Result<String, RemoteError> {
        if let Some(t) = s.not_before {
            let now = Instant::now();
            if t > now {
                let left = (t - now).as_secs().max(1);
                return Err(rate_limited(left));
            }
        }
        let token = if s.session_only {
            s.memory_refresh.clone()
        } else {
            // read the store each time: a second instance of the app may have rotated it
            self.store
                .get(ACCOUNT_REFRESH)
                .map_err(|e| RemoteError::new("credential_store", format!("Could not read the sign-in from the system's credential store: {}", e)))?
        };
        let Some(token) = token else { return Err(RemoteError::signed_out()) };
        s.access = None;
        s.expires_at = None;
        let body = serde_json::json!({"grant_type": "refresh_token", "refresh_token": token});
        let mut tries = 0;
        let response = loop {
            match self.client.post(self.url(api::PATH_TOKEN)).json(&body).timeout(Duration::from_secs(60)).send().await {
                Ok(r) => break r,
                Err(e) if e.is_connect() && tries < 2 => {
                    tries += 1;
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                Err(e) => return Err(api::net_error(e)),
            }
        };
        let a = Answer::read(response).await?;
        match a.status {
            200 => {
                let t: TokenResponse = a.parse().ok_or_else(|| RemoteError::new("network", "The server's sign-in answer could not be read."))?;
                let access = t.access_token.clone();
                self.adopt(s, t);
                Ok(access)
            }
            // Expired, reused, or disconnected from the website's list of PCs
            400 => {
                if a.error() == Some("invalid_grant") {
                    if let Err(e) = self.forget(s) {
                        eprintln!("[LAB] the credential store kept an ended sign-in: {}", api::scrub(&e));
                    }
                    return Err(RemoteError::signed_out());
                }
                Err(RemoteError::unavailable())
            }
            429 => {
                let secs = a.retry_after.unwrap_or(60);
                s.not_before = Some(Instant::now() + Duration::from_secs(secs));
                Err(rate_limited(secs))
            }
            // A missing route looks like an auth failure on this server: keep the session
            401 | 404 => Err(RemoteError::login_unavailable()),
            _ => Err(RemoteError::unavailable()),
        }
    }

    /// A usable access token. `used`: the one a request just got a 401 with, so it must not come
    /// back; if another request has refreshed in the meantime, its token is used instead of
    /// rotating again. The work runs in its own task: dropping this future (cancel) can't lose the
    /// answer of a refresh that reached the server.
    pub async fn access_token(self: &Arc<Self>, used: Option<String>) -> Result<String, RemoteError> {
        self.token_lasting(used, ACCESS_MARGIN).await
    }

    /// Refresh now if the access token wouldn't last this long. For a request that takes a long time
    /// and is checked only at its end: the proxy may hold a whole upload before the backend sees it.
    pub async fn ensure_lasting(self: &Arc<Self>, need: Duration) -> Result<(), RemoteError> {
        self.token_lasting(None, need).await.map(|_| ())
    }

    async fn token_lasting(self: &Arc<Self>, used: Option<String>, need: Duration) -> Result<String, RemoteError> {
        let me = self.clone();
        tauri::async_runtime::spawn(async move {
            let mut s = me.session.lock().await;
            if let (Some(a), Some(exp)) = (s.access.clone(), s.expires_at) {
                if exp > Instant::now() + need && used.as_deref() != Some(a.as_str()) {
                    return Ok(a);
                }
            }
            me.refresh_locked(&mut s).await
        })
        .await
        .map_err(|e| RemoteError::new("network", e.to_string()))?
    }

    /// Send with the access token. On 401: refresh once, retry once, then signed out. `build` makes
    /// the request each time, so a streamed body can start over. Answers that mean the same on every
    /// route are turned into errors here; anything else goes back to the caller.
    pub async fn send<F, Fut>(self: &Arc<Self>, build: F) -> Result<Answer, RemoteError>
    where
        F: Fn(reqwest::Client) -> Fut,
        Fut: Future<Output = Result<reqwest::RequestBuilder, RemoteError>>,
    {
        let token = self.access_token(None).await?;
        let r = build(self.client.clone()).await?.bearer_auth(&token).send().await.map_err(api::net_error)?;
        let a = Answer::read(r).await?;
        if a.status != 401 {
            return shared(a);
        }
        let token = self.access_token(Some(token)).await?;
        let r = build(self.client.clone()).await?.bearer_auth(&token).send().await.map_err(api::net_error)?;
        let a = Answer::read(r).await?;
        if a.status == 401 {
            self.clear().await;
            return Err(RemoteError::signed_out());
        }
        shared(a)
    }

    /// `send` for a request without a body.
    pub async fn send_simple(self: &Arc<Self>, method: reqwest::Method, path: &str, timeout: Duration) -> Result<Answer, RemoteError> {
        let url = self.url(path);
        self.send(|c| {
            let (u, m) = (url.clone(), method.clone());
            async move { Ok(c.request(m, u).timeout(timeout)) }
        })
        .await
    }

    /// Exchange the login's code. A rejected code is a failed sign-in, not the end of a session. A
    /// sign-in this one replaces is revoked, so it doesn't stay on the website's list of PCs.
    pub async fn exchange(&self, code: &str, verifier: &str) -> Result<Account, RemoteError> {
        let body = serde_json::json!({"grant_type": "authorization_code", "code": code, "code_verifier": verifier});
        let r = self.client.post(self.url(api::PATH_TOKEN)).json(&body).timeout(Duration::from_secs(60)).send().await.map_err(api::net_error)?;
        let a = Answer::read(r).await?;
        match a.status {
            200 => {
                let t: TokenResponse = a.parse().ok_or_else(|| RemoteError::new("login_failed", "The server's sign-in answer could not be read. Please try again."))?;
                let mut s = self.session.lock().await;
                let old = if s.session_only { s.memory_refresh.clone() } else { self.store.get(ACCOUNT_REFRESH).ok().flatten() };
                let new_token = t.refresh_token.clone();
                self.adopt(&mut s, t);
                let account = self.account_of(&s);
                drop(s);
                if let Some(old) = old.filter(|o| *o != new_token) {
                    self.revoke(&old).await;
                }
                Ok(account)
            }
            400 => Err(RemoteError::new("login_failed", "The sign-in didn't complete. Please try again.")),
            401 | 404 => Err(RemoteError::login_unavailable()),
            429 => Err(rate_limited(a.retry_after.unwrap_or(60))),
            _ => Err(RemoteError::unavailable()),
        }
    }

    async fn revoke(&self, token: &str) {
        let body = serde_json::json!({"refresh_token": token});
        match self.client.post(self.url(api::PATH_REVOKE)).json(&body).timeout(Duration::from_secs(20)).send().await {
            Ok(r) if r.status().is_success() => {}
            Ok(r) => eprintln!("[LAB] revoke answered {}; going on anyway", r.status().as_u16()),
            Err(_) => eprintln!("[LAB] revoke failed; going on anyway"),
        }
    }

    /// Sign out here: tell the server if it can be reached, and forget the session either way. If the
    /// OS store won't let go of the token, say so: the app would still look signed in.
    pub async fn sign_out(&self) -> Result<(), RemoteError> {
        let mut s = self.session.lock().await;
        let token = if s.session_only { s.memory_refresh.clone() } else { self.store.get(ACCOUNT_REFRESH).ok().flatten() };
        if let Some(token) = token {
            self.revoke(&token).await;
        }
        self.forget(&mut s).map_err(|e| {
            eprintln!("[LAB] the credential store kept the sign-in: {}", api::scrub(&e));
            RemoteError::new("credential_store", "The sign-in ended on the server, but the system's credential store didn't remove it from this PC. Please try signing out again.")
        })
    }
}

fn rate_limited(secs: u64) -> RemoteError {
    RemoteError::new("rate_limited", format!("Too many sign-in requests. Please try again in {} seconds.", secs)).retry(Some(secs))
}

/// Answers that mean the same on every route.
fn shared(a: Answer) -> Result<Answer, RemoteError> {
    match (a.status, a.error()) {
        (403, Some("scope")) => Err(RemoteError::scope()),
        // The token lookup failed on the server: not a sign-out
        (503, Some("unavailable")) => Err(RemoteError::unavailable().retry(a.retry_after)),
        _ => Ok(a),
    }
}

// ── login ───────────────────────────────────────────────────────────────────

pub fn b64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

pub fn challenge_for(verifier: &str) -> String {
    b64url(&Sha256::digest(verifier.as_bytes()))
}

fn random_b64url() -> String {
    let mut b = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut b);
    b64url(&b)
}

fn page(title: &str, text: &str) -> String {
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{t}</title></head><body style=\"font-family:system-ui,sans-serif;background:#14141c;color:#eee;display:flex;align-items:center;justify-content:center;height:100vh;margin:0\"><div><h2>{t}</h2><p>{x}</p></div></body></html>",
        t = title,
        x = text
    )
}

async fn reply(stream: &mut tokio::net::TcpStream, status: &str, body: &str) {
    let head = format!(
        "HTTP/1.1 {}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n\r\n",
        status,
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).await;
    let _ = stream.write_all(body.as_bytes()).await;
    let _ = stream.shutdown().await;
}

/// The target of a GET on this connection. The time is for the whole request, so a client sending one
/// byte at a time gets no longer than a silent one.
async fn read_target(stream: &mut tokio::net::TcpStream) -> Option<String> {
    let mut buf = vec![0u8; 8192];
    let mut n = 0;
    let read = async {
        while n < buf.len() {
            let got = stream.read(&mut buf[n..]).await.ok()?;
            if got == 0 {
                break;
            }
            n += got;
            if buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        Some(())
    };
    tokio::time::timeout(CALLBACK_READ, read).await.ok()??;
    let text = String::from_utf8_lossy(&buf[..n]);
    let mut first = text.lines().next()?.split_whitespace();
    if first.next()? != "GET" {
        return None;
    }
    first.next().map(str::to_string)
}

/// What one connection means for the login: None is anything but the real callback, so the wait goes on.
async fn callback_answer(mut stream: tokio::net::TcpStream, state: &str) -> Option<Result<String, RemoteError>> {
    let Some(target) = read_target(&mut stream).await else {
        reply(&mut stream, "400 Bad Request", "").await;
        return None;
    };
    let Ok(u) = url::Url::parse(&format!("http://127.0.0.1{}", target)) else {
        reply(&mut stream, "400 Bad Request", "").await;
        return None;
    };
    if u.path() != "/callback" {
        reply(&mut stream, "404 Not Found", "").await;
        return None;
    }
    let q: HashMap<String, String> = u.query_pairs().into_owned().collect();
    if q.get("state").map(String::as_str) != Some(state) {
        reply(&mut stream, "400 Bad Request", &page("Sign-in refused", "This sign-in link doesn't match the one the app opened. Go back to Spaghetti Lab and open the page again from there.")).await;
        return None;
    }
    if q.contains_key("error") {
        reply(&mut stream, "200 OK", &page("Sign-in cancelled", "You can close this tab and go back to Spaghetti Lab.")).await;
        return Some(Err(RemoteError::new("login_denied", "The sign-in was cancelled in the browser.")));
    }
    let Some(code) = q.get("code").filter(|c| !c.is_empty()) else {
        reply(&mut stream, "400 Bad Request", &page("Sign-in failed", "Please start again from Spaghetti Lab.")).await;
        return Some(Err(RemoteError::new("login_failed", "The sign-in answer had no code. Please try again.")));
    };
    reply(&mut stream, "200 OK", &page("Signed in", "You can close this tab and go back to Spaghetti Lab.")).await;
    Some(Ok(code.clone()))
}

/// Wait for `GET /callback` with this login's `state`. Query values are percent-decoded before comparing.
/// Other paths get a 404 and a wrong `state` a 400, and neither ends the wait: anything on this PC can
/// reach the port, and it must not be able to end someone's login. Connections are served side by side,
/// so a silent or slow one can't hold up the browser's.
pub async fn wait_for_callback(listener: tokio::net::TcpListener, state: &str, timeout: Duration, mut cancel: watch::Receiver<bool>) -> Result<String, RemoteError> {
    let wait = async {
        let mut conns = FuturesUnordered::new();
        loop {
            tokio::select! {
                accepted = listener.accept() => {
                    let (stream, _) = accepted.map_err(|e| RemoteError::new("login_failed", e.to_string()))?;
                    conns.push(callback_answer(stream, state));
                }
                Some(answer) = conns.next(), if !conns.is_empty() => {
                    if let Some(answer) = answer {
                        return answer;
                    }
                }
            }
        }
    };
    tokio::select! {
        r = wait => r,
        _ = tokio::time::sleep(timeout) => Err(RemoteError::new("login_timeout", "The sign-in timed out. Please try again.")),
        _ = cancel.wait_for(|c| *c) => Err(RemoteError::new("cancelled", "Sign-in cancelled.")),
    }
}

struct LoginFlow {
    url: String,
    cancel: watch::Sender<bool>,
}

static LOGIN: Mutex<Option<LoginFlow>> = Mutex::new(None);
static AUTH: Mutex<Option<Arc<Auth>>> = Mutex::new(None);

/// The app's session (base URL and credential-store service come from the environment).
pub fn auth() -> Result<Arc<Auth>, RemoteError> {
    let mut g = AUTH.lock().unwrap();
    if let Some(a) = g.as_ref() {
        return Ok(a.clone());
    }
    let a = Auth::new(api::api_base()?, Arc::new(KeyringStore::new(api::keyring_service())));
    *g = Some(a.clone());
    Ok(a)
}

/// Open a website page in the system browser. In test mode the page is only requested, so the mock
/// sees exactly the URL and no browser opens.
fn open_external(app: &tauri::AppHandle, url: &str) -> Result<(), RemoteError> {
    if api::test_mode() {
        let u = url.to_string();
        let client = api::client_for(&u);
        tauri::async_runtime::spawn(async move {
            let _ = client.get(&u).send().await;
        });
        return Ok(());
    }
    #[allow(deprecated)]
    tauri_plugin_shell::ShellExt::shell(app)
        .open(url, None)
        .map_err(|e| RemoteError::new("browser", format!("Could not open the browser: {}", e)))
}

// ── commands ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct LabConfig {
    pub local_models: bool,
    pub web_url: String,
}

#[tauri::command]
pub fn lab_config() -> Result<LabConfig, RemoteError> {
    Ok(LabConfig {
        local_models: api::local_models_enabled(std::env::var_os("SPAGHETTI_LAB_MODELS").as_deref()),
        web_url: api::web_base()?,
    })
}

#[tauri::command]
pub async fn lab_account() -> Result<Account, RemoteError> {
    Ok(auth()?.account().await)
}

#[tauri::command]
pub async fn lab_sign_in(app: tauri::AppHandle) -> Result<Account, RemoteError> {
    let auth = auth()?;
    let web = api::web_base()?;
    // The page first: while the login isn't deployed, a readable message instead of a browser on a 404
    let client = api::client_for(&web);
    let probe = client
        .get(format!("{}{}", web, api::PATH_DESKTOP_LOGIN))
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .map_err(api::net_error)?;
    match probe.status().as_u16() {
        404 => return Err(RemoteError::login_unavailable()),
        s if s >= 500 => return Err(RemoteError::unavailable()),
        _ => {}
    }

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.map_err(|e| RemoteError::new("login_failed", e.to_string()))?;
    let port = listener.local_addr().map_err(|e| RemoteError::new("login_failed", e.to_string()))?.port();
    let state = random_b64url();
    let verifier = random_b64url();
    let url = api::login_url(&web, port, &state, &challenge_for(&verifier));
    let (tx, rx) = watch::channel(false);
    if let Some(old) = LOGIN.lock().unwrap().replace(LoginFlow { url: url.clone(), cancel: tx }) {
        let _ = old.cancel.send(true);
    }
    if let Err(e) = open_external(&app, &url) {
        LOGIN.lock().unwrap().take();
        return Err(e);
    }
    let code = wait_for_callback(listener, &state, LOGIN_TIMEOUT, rx).await;
    {
        // only if it's still this login's slot (a newer sign-in may have taken it)
        let mut g = LOGIN.lock().unwrap();
        if g.as_ref().map_or(false, |l| l.url == url) {
            g.take();
        }
    }
    let account = auth.exchange(&code?, &verifier).await?;
    crate::lab_remote::after_sign_in(auth.clone());
    Ok(account)
}

/// Open the same login page again (a popup blocker, a closed tab, or a login that landed on the home page).
#[tauri::command]
pub fn lab_sign_in_reopen(app: tauri::AppHandle) -> Result<(), RemoteError> {
    let url = LOGIN
        .lock()
        .unwrap()
        .as_ref()
        .map(|l| l.url.clone())
        .ok_or_else(|| RemoteError::new("login_failed", "No sign-in is waiting."))?;
    open_external(&app, &url)
}

#[tauri::command]
pub fn lab_sign_in_cancel() {
    if let Some(l) = LOGIN.lock().unwrap().take() {
        let _ = l.cancel.send(true);
    }
}

#[tauri::command]
pub async fn lab_sign_out() -> Result<Account, RemoteError> {
    if crate::lab_remote::running_replay().is_some() {
        return Err(RemoteError::new("busy", "An analysis is running. Cancel it or wait for it to end before signing out."));
    }
    let auth = auth()?;
    crate::lab_remote::retry_pending_deletes(&auth, &crate::lab_remote::lab_root_dir()).await;
    auth.sign_out().await?;
    Ok(auth.account().await)
}

/// Always without a token: a stale one would turn "not available" into a 401.
#[tauri::command]
pub async fn lab_service_status() -> Result<api::ServiceStatus, RemoteError> {
    service_status(&auth()?).await
}

pub async fn service_status(auth: &Arc<Auth>) -> Result<api::ServiceStatus, RemoteError> {
    let r = auth.client().get(auth.url(api::PATH_STATUS)).timeout(Duration::from_secs(30)).send().await.map_err(api::net_error)?;
    let a = Answer::read(r).await?;
    if a.status == 200 {
        Ok(a.parse().unwrap_or_default())
    } else {
        // 503 (either body), a proxy's 502, anything else: not available, nothing about the sign-in
        Ok(api::ServiceStatus::default())
    }
}

/// null when the analysis isn't available (production answers 503 here while it's off).
#[tauri::command]
pub async fn lab_quota() -> Result<Option<api::Quota>, RemoteError> {
    quota(&auth()?).await
}

pub async fn quota(auth: &Arc<Auth>) -> Result<Option<api::Quota>, RemoteError> {
    let a = match auth.send_simple(reqwest::Method::GET, api::PATH_QUOTA, Duration::from_secs(30)).await {
        Err(e) if e.code == "unavailable" => return Ok(None),
        other => other?,
    };
    if a.status == 200 {
        Ok(a.quota())
    } else {
        Ok(None)
    }
}

/// Open /support, /terms or /privacy in the system browser.
#[tauri::command]
pub fn lab_open_web(app: tauri::AppHandle, page: String) -> Result<(), RemoteError> {
    let path = api::web_page(&page).ok_or_else(|| RemoteError::new("browser", "Unknown page."))?;
    open_external(&app, &format!("{}{}", api::web_base()?, path))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Call = (String, String, String, String);

    /// A scripted server: each request is answered by `f(method, path, body, authorization)`.
    pub struct Mock {
        pub url: String,
        pub calls: Arc<Mutex<Vec<Call>>>,
    }

    pub fn mock<F>(f: F) -> Mock
    where
        F: Fn(&str, &str, &str, &str) -> (u16, String) + Send + Sync + 'static,
    {
        let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").unwrap());
        let url = format!("http://127.0.0.1:{}", server.server_addr().to_ip().unwrap().port());
        let calls: Arc<Mutex<Vec<Call>>> = Default::default();
        let f = Arc::new(f);
        // a few workers, so concurrent requests really overlap
        for _ in 0..4 {
            let (server, calls, f) = (server.clone(), calls.clone(), f.clone());
            std::thread::spawn(move || {
                for mut rq in server.incoming_requests() {
                    let mut body = Vec::new();
                    let _ = std::io::Read::read_to_end(rq.as_reader(), &mut body);
                    let body = String::from_utf8_lossy(&body).to_string();
                    let method = rq.method().to_string();
                    let path = rq.url().to_string();
                    let authz = rq.headers().iter().find(|h| h.field.equiv("Authorization")).map(|h| h.value.to_string()).unwrap_or_default();
                    calls.lock().unwrap().push((method.clone(), path.clone(), body.clone(), authz.clone()));
                    let (status, text) = f(&method, &path, &body, &authz);
                    let _ = rq.respond(tiny_http::Response::from_string(text).with_status_code(status));
                }
            });
        }
        Mock { url, calls }
    }

    pub fn token_json(n: usize) -> String {
        format!(r#"{{"access_token":"sslab_at_{n}","token_type":"Bearer","expires_in":3600,"refresh_token":"sslab_rt_{n}","scope":"lab","player":{{"player_user_id":"u-1","nickname":"Nick","avatar":null}}}}"#)
    }

    pub fn signed_in_auth(url: &str) -> (Arc<Auth>, Arc<MemoryStore>) {
        let store = Arc::new(MemoryStore::default());
        store.set(ACCOUNT_REFRESH, "sslab_rt_0").unwrap();
        store.set(ACCOUNT_PLAYER, r#"{"player_user_id":"u-1","nickname":"Nick","avatar":null}"#).unwrap();
        (Auth::new(url.to_string(), store.clone()), store)
    }

    fn run<T>(f: impl Future<Output = T>) -> T {
        tauri::async_runtime::block_on(f)
    }

    fn no_cancel() -> (watch::Sender<bool>, watch::Receiver<bool>) {
        watch::channel(false)
    }

    #[test]
    fn lab_pkce_vector() {
        // RFC 7636 appendix B
        assert_eq!(challenge_for("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
        let (a, b) = (random_b64url(), random_b64url());
        assert_ne!(a, b);
        assert_eq!(a.len(), 43);
        assert_eq!(challenge_for(&a).len(), 43);
    }

    async fn http_get(port: u16, target: &str) -> String {
        let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        s.write_all(format!("GET {} HTTP/1.1\r\nHost: x\r\n\r\n", target).as_bytes()).await.unwrap();
        let mut out = String::new();
        let _ = s.read_to_string(&mut out).await;
        out
    }

    async fn listener() -> (tokio::net::TcpListener, u16) {
        let l = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let p = l.local_addr().unwrap().port();
        (l, p)
    }

    /// Real socket: an unrelated request first, then a percent-encoded `state` and `code`.
    #[test]
    fn lab_callback_decodes_state_and_survives_stray_requests() {
        run(async {
            let (l, port) = listener().await;
            let (_tx, rx) = no_cancel();
            let w = tauri::async_runtime::spawn(async move { wait_for_callback(l, "st/at+e", Duration::from_secs(5), rx).await });
            assert!(http_get(port, "/favicon.ico").await.starts_with("HTTP/1.1 404"));
            assert!(http_get(port, "/?code=x&state=st%2Fat%2Be").await.starts_with("HTTP/1.1 404"));
            let ok = http_get(port, "/callback?code=abc%2Fd&state=st%2Fat%2Be").await;
            assert!(ok.starts_with("HTTP/1.1 200"));
            assert!(ok.to_lowercase().contains("referrer-policy: no-referrer"));
            assert!(!ok.contains("<script"));
            assert_eq!(w.await.unwrap().unwrap(), "abc/d");
        });
    }

    /// Anything on this PC can reach the port: a wrong state is refused and can't end the login.
    #[test]
    fn lab_callback_refuses_a_wrong_state_and_keeps_waiting() {
        run(async {
            let (l, port) = listener().await;
            let (_tx, rx) = no_cancel();
            let w = tauri::async_runtime::spawn(async move { wait_for_callback(l, "S1", Duration::from_secs(5), rx).await });
            let bad = http_get(port, "/callback?code=abc&state=WRONG").await;
            assert!(bad.starts_with("HTTP/1.1 400"));
            // a denial with the wrong state isn't a cancel either
            assert!(http_get(port, "/callback?error=access_denied&state=other").await.starts_with("HTTP/1.1 400"));
            assert!(http_get(port, "/callback?code=good&state=S1").await.starts_with("HTTP/1.1 200"));
            assert_eq!(w.await.unwrap().unwrap(), "good");
            // the listener is closed: nothing else can reach it
            assert!(tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_err());
        });
    }

    /// A silent client and one sending a byte at a time don't hold up the browser's request.
    #[test]
    fn lab_callback_is_not_held_up_by_slow_connections() {
        run(async {
            let (l, port) = listener().await;
            let (_tx, rx) = no_cancel();
            let w = tauri::async_runtime::spawn(async move { wait_for_callback(l, "S1", Duration::from_secs(30), rx).await });
            let _silent = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            let mut slow = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
            slow.write_all(b"G").await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            let started = Instant::now();
            assert!(http_get(port, "/callback?code=c&state=S1").await.starts_with("HTTP/1.1 200"));
            assert_eq!(w.await.unwrap().unwrap(), "c");
            assert!(started.elapsed() < Duration::from_secs(2));
        });
    }

    #[test]
    fn lab_a_callback_request_has_one_deadline_however_slowly_it_trickles() {
        run(async {
            let (l, port) = listener().await;
            let client = tauri::async_runtime::spawn(async move {
                let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
                // never a gap near the deadline, and never the end of the request
                for _ in 0..80 {
                    if s.write_all(b"x").await.is_err() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            });
            let (mut stream, _) = l.accept().await.unwrap();
            let started = Instant::now();
            assert_eq!(read_target(&mut stream).await, None);
            let took = started.elapsed();
            assert!(took >= CALLBACK_READ - Duration::from_millis(200) && took < CALLBACK_READ + Duration::from_secs(1), "{:?}", took);
            drop(stream);
            client.abort();
        });
    }

    #[test]
    fn lab_callback_cancel_in_the_browser_timeout_and_cancel_in_the_app() {
        run(async {
            let (l, port) = listener().await;
            let (_tx, rx) = no_cancel();
            let w = tauri::async_runtime::spawn(async move { wait_for_callback(l, "a/b", Duration::from_secs(5), rx).await });
            assert!(http_get(port, "/callback?error=access_denied&state=a%2Fb").await.starts_with("HTTP/1.1 200"));
            assert_eq!(w.await.unwrap().unwrap_err().code, "login_denied");

            let (l, _) = listener().await;
            let (_tx, rx) = no_cancel();
            assert_eq!(wait_for_callback(l, "S1", Duration::from_millis(50), rx).await.unwrap_err().code, "login_timeout");

            let (l, _) = listener().await;
            let (tx, rx) = no_cancel();
            let w = tauri::async_runtime::spawn(async move { wait_for_callback(l, "S1", Duration::from_secs(5), rx).await });
            tx.send(true).unwrap();
            assert_eq!(w.await.unwrap().unwrap_err().code, "cancelled");
        });
    }

    #[test]
    fn lab_exchange_keeps_refresh_in_store_and_shows_no_tokens() {
        run(async {
            let m = mock(|_, p, b, _| {
                assert_eq!(p, "/api/desktop-auth/token");
                assert!(b.contains("\"authorization_code\"") && b.contains("\"code_verifier\":\"v\""));
                (200, token_json(1))
            });
            let store = Arc::new(MemoryStore::default());
            let a = Auth::new(m.url.clone(), store.clone());
            let acc = a.exchange("c", "v").await.unwrap();
            assert_eq!(acc, Account { signed_in: true, nickname: Some("Nick".into()), avatar: None, session_only: false, store_unavailable: false });
            assert_eq!(store.get(ACCOUNT_REFRESH).unwrap().unwrap(), "sslab_rt_1");
            assert!(!serde_json::to_string(&acc).unwrap().contains("sslab_"));
            assert_eq!(store.values.lock().unwrap().len(), 2, "only the refresh token and the player are stored");
        });
    }

    #[test]
    fn lab_a_401_refreshes_once_and_retries_once() {
        run(async {
            let m = mock(|_, p, _, authz| match p {
                "/api/desktop-auth/token" => (200, token_json(1)),
                _ if authz == "Bearer sslab_at_1" => (200, r#"{"ok":1}"#.into()),
                _ => (401, r#"{"error":"invalid_token"}"#.into()),
            });
            let (a, store) = signed_in_auth(&m.url);
            // a token that was valid until the server dropped it
            {
                let mut s = a.session.lock().await;
                s.access = Some("sslab_at_0".into());
                s.expires_at = Some(Instant::now() + Duration::from_secs(3000));
            }
            let r = a.send_simple(reqwest::Method::GET, "/x", Duration::from_secs(5)).await.unwrap();
            assert_eq!(r.status, 200);
            assert_eq!(store.get(ACCOUNT_REFRESH).unwrap().unwrap(), "sslab_rt_1");
            let calls = m.calls.lock().unwrap();
            assert_eq!(calls.iter().filter(|c| c.1 == "/api/desktop-auth/token").count(), 1);
            assert_eq!(calls.iter().filter(|c| c.1 == "/x").count(), 2);
        });
    }

    #[test]
    fn lab_a_second_401_signs_out() {
        run(async {
            let m = mock(|_, p, _, _| if p.contains("token") { (200, token_json(1)) } else { (401, r#"{"error":"invalid_token"}"#.into()) });
            let (a, store) = signed_in_auth(&m.url);
            let e = a.send_simple(reqwest::Method::GET, "/x", Duration::from_secs(5)).await.unwrap_err();
            assert_eq!(e.code, "signed_out");
            assert_eq!(store.get(ACCOUNT_REFRESH).unwrap(), None);
            assert!(!a.account().await.signed_in);
        });
    }

    #[test]
    fn lab_invalid_grant_is_signed_out_but_server_trouble_is_not() {
        run(async {
            let m = mock(|_, _, _, _| (400, r#"{"error":"invalid_grant"}"#.into()));
            let (a, store) = signed_in_auth(&m.url);
            assert_eq!(a.access_token(None).await.unwrap_err().code, "signed_out");
            assert_eq!(store.get(ACCOUNT_REFRESH).unwrap(), None);

            for (status, body, code) in [
                (500, r#"{"error":"unavailable"}"#, "unavailable"),
                (503, r#"{"error":"unavailable"}"#, "unavailable"),
                (404, "", "login_unavailable"),
                (401, r#"{"error":"Authentication required","code":"credentials_required"}"#, "login_unavailable"),
                (502, "<html>", "unavailable"),
            ] {
                let m = mock(move |_, _, _, _| (status, body.to_string()));
                let (a, store) = signed_in_auth(&m.url);
                assert_eq!(a.access_token(None).await.unwrap_err().code, code);
                assert_eq!(store.get(ACCOUNT_REFRESH).unwrap().unwrap(), "sslab_rt_0", "{status} must keep the session");
            }
        });
    }

    #[test]
    fn lab_scope_and_lookup_failure_are_not_a_sign_out() {
        run(async {
            for (status, body, code) in [(403, r#"{"error":"scope"}"#, "scope"), (503, r#"{"error":"unavailable"}"#, "unavailable")] {
                let m = mock(move |_, p, _, _| if p.contains("token") { (200, token_json(1)) } else { (status, body.to_string()) });
                let (a, store) = signed_in_auth(&m.url);
                let e = a.send_simple(reqwest::Method::GET, "/x", Duration::from_secs(5)).await.unwrap_err();
                assert_eq!(e.code, code);
                assert!(store.get(ACCOUNT_REFRESH).unwrap().is_some());
            }
        });
    }

    /// The theft protection: presenting a rotated token ends the session, so concurrent 401s must
    /// produce exactly one refresh, with the current token.
    #[test]
    fn lab_concurrent_requests_refresh_once() {
        run(async {
            let refreshes = Arc::new(AtomicUsize::new(0));
            let r2 = refreshes.clone();
            let m = mock(move |_, p, b, authz| {
                if p.contains("token") {
                    let n = r2.fetch_add(1, Ordering::SeqCst);
                    assert!(b.contains(&format!("sslab_rt_{}", n)), "a rotated refresh token was presented");
                    std::thread::sleep(Duration::from_millis(150));
                    return (200, token_json(n + 1));
                }
                if authz.contains("sslab_at_1") {
                    (200, "{}".into())
                } else {
                    (401, r#"{"error":"invalid_token"}"#.into())
                }
            });
            let (a, _store) = signed_in_auth(&m.url);
            let mut tasks = Vec::new();
            for _ in 0..4 {
                let a = a.clone();
                tasks.push(tauri::async_runtime::spawn(async move {
                    a.send_simple(reqwest::Method::GET, "/x", Duration::from_secs(5)).await.map(|r| r.status)
                }));
            }
            for t in tasks {
                assert_eq!(t.await.unwrap().unwrap(), 200);
            }
            assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn lab_the_new_token_is_stored_before_use() {
        run(async {
            let store = Arc::new(MemoryStore::default());
            let seen = Arc::new(Mutex::new(None));
            let (s2, seen2) = (store.clone(), seen.clone());
            let m = mock(move |_, p, _, _| {
                if p.contains("token") {
                    (200, token_json(7))
                } else {
                    *seen2.lock().unwrap() = s2.get(ACCOUNT_REFRESH).unwrap();
                    (200, "{}".into())
                }
            });
            store.set(ACCOUNT_REFRESH, "sslab_rt_0").unwrap();
            let a = Auth::new(m.url.clone(), store.clone());
            a.send_simple(reqwest::Method::GET, "/x", Duration::from_secs(5)).await.unwrap();
            assert_eq!(seen.lock().unwrap().as_deref(), Some("sslab_rt_7"));
        });
    }

    #[test]
    fn lab_a_failing_store_keeps_the_session_in_memory() {
        run(async {
            let m = mock(|_, _, b, _| (200, token_json(if b.contains("sslab_rt_1") { 2 } else { 1 })));
            let store = Arc::new(MemoryStore::default());
            // an older sign-in in the store: rotated from now on, so it must not stay there
            store.set(ACCOUNT_REFRESH, "sslab_rt_old").unwrap();
            store.fail_writes.store(true, Ordering::SeqCst);
            let a = Auth::new(m.url.clone(), store.clone());
            let acc = a.exchange("c", "v").await.unwrap();
            assert!(acc.signed_in && acc.session_only);
            assert!(store.values.lock().unwrap().is_empty());
            // it still refreshes, from memory
            assert_eq!(a.access_token(Some("sslab_at_1".into())).await.unwrap(), "sslab_at_2");
        });
    }

    #[test]
    fn lab_retry_after_blocks_refresh() {
        run(async {
            let m = mock(|_, _, _, _| (429, r#"{"error":"too_many_requests"}"#.into()));
            let (a, store) = signed_in_auth(&m.url);
            assert_eq!(a.access_token(None).await.unwrap_err().code, "rate_limited");
            assert_eq!(a.access_token(None).await.unwrap_err().code, "rate_limited");
            assert_eq!(m.calls.lock().unwrap().len(), 1, "no new request before Retry-After");
            assert!(store.get(ACCOUNT_REFRESH).unwrap().is_some());
        });
    }

    #[test]
    fn lab_sign_out_revokes_and_forgets_even_if_the_server_fails() {
        run(async {
            let m = mock(|_, p, b, _| {
                assert_eq!(p, "/api/desktop-auth/revoke");
                assert!(b.contains("sslab_rt_0"));
                (500, r#"{"error":"unavailable"}"#.into())
            });
            let (a, store) = signed_in_auth(&m.url);
            a.sign_out().await.unwrap();
            assert_eq!(store.get(ACCOUNT_REFRESH).unwrap(), None);
            assert!(!a.account().await.signed_in);
            assert_eq!(m.calls.lock().unwrap().len(), 1);
        });
    }

    /// The UI must not go on showing a sign-in the server just revoked as if nothing had happened.
    #[test]
    fn lab_sign_out_says_so_when_the_store_keeps_the_token() {
        run(async {
            let m = mock(|_, _, _, _| (200, r#"{"ok":true}"#.into()));
            let (a, store) = signed_in_auth(&m.url);
            store.fail_deletes.store(true, Ordering::SeqCst);
            assert_eq!(a.sign_out().await.unwrap_err().code, "credential_store");
            store.fail_deletes.store(false, Ordering::SeqCst);
            a.sign_out().await.unwrap();
            assert!(!a.account().await.signed_in);
        });
    }

    #[test]
    fn lab_an_unreadable_store_is_not_signed_out() {
        run(async {
            let m = mock(|_, _, _, _| (200, token_json(1)));
            let (a, store) = signed_in_auth(&m.url);
            store.fail_reads.store(true, Ordering::SeqCst);
            let acc = a.account().await;
            assert!(acc.store_unavailable && !acc.signed_in);
            assert_eq!(a.access_token(None).await.unwrap_err().code, "credential_store");
            store.fail_reads.store(false, Ordering::SeqCst);
            let acc = a.account().await;
            assert!(acc.signed_in && !acc.store_unavailable);
            assert_eq!(store.get(ACCOUNT_REFRESH).unwrap().as_deref(), Some("sslab_rt_0"), "nothing was forgotten");
        });
    }

    #[test]
    fn lab_a_new_sign_in_revokes_the_one_it_replaces() {
        run(async {
            let m = mock(|_, p, _, _| if p.ends_with("/token") { (200, token_json(1)) } else { (200, r#"{"ok":true}"#.into()) });
            let (a, store) = signed_in_auth(&m.url);
            a.exchange("c", "v").await.unwrap();
            assert_eq!(store.get(ACCOUNT_REFRESH).unwrap().as_deref(), Some("sslab_rt_1"));
            let calls = m.calls.lock().unwrap();
            let revokes: Vec<_> = calls.iter().filter(|c| c.1 == "/api/desktop-auth/revoke").collect();
            assert_eq!(revokes.len(), 1);
            assert!(revokes[0].2.contains("sslab_rt_0"));
        });
    }

    /// A proxy may hold an upload before the backend checks the token, so it must last the whole transfer.
    #[test]
    fn lab_a_token_can_be_required_to_last() {
        run(async {
            let m = mock(|_, _, _, _| (200, token_json(1)));
            let (a, _) = signed_in_auth(&m.url);
            {
                let mut s = a.session.lock().await;
                s.access = Some("sslab_at_0".into());
                s.expires_at = Some(Instant::now() + Duration::from_secs(600));
            }
            a.ensure_lasting(Duration::from_secs(300)).await.unwrap();
            assert_eq!(m.calls.lock().unwrap().len(), 0, "ten minutes are enough for five");
            a.ensure_lasting(Duration::from_secs(1800)).await.unwrap();
            assert_eq!(m.calls.lock().unwrap().len(), 1);
            assert_eq!(a.access_token(None).await.unwrap(), "sslab_at_1");
        });
    }

    #[test]
    fn lab_status_never_sends_a_token_and_503_is_just_not_available() {
        run(async {
            for (status, body) in [(503, r#"{"error":"Opener analysis is not available"}"#), (503, r#"{"available":false}"#), (502, "<html>")] {
                let m = mock(move |_, _, _, _| (status, body.to_string()));
                let (a, store) = signed_in_auth(&m.url);
                assert!(!service_status(&a).await.unwrap().available);
                assert!(store.get(ACCOUNT_REFRESH).unwrap().is_some());
                assert_eq!(m.calls.lock().unwrap()[0].3, "");
            }
            let m = mock(|_, _, _, _| (200, r#"{"available":true,"running":0,"queued":1,"max_concurrent":1,"max_queue":8,"capabilities":{"transcript":false},"free_per_week":2}"#.into()));
            let (a, _) = signed_in_auth(&m.url);
            let s = service_status(&a).await.unwrap();
            assert!(s.available);
            assert_eq!(s.queued, Some(1));
        });
    }

    #[test]
    fn lab_a_network_error_mentions_no_url() {
        run(async {
            let (a, _) = signed_in_auth("http://127.0.0.1:9");
            let e = a.access_token(None).await.unwrap_err();
            assert_eq!(e.code, "network");
            assert!(!e.message.contains("127.0.0.1"));
        });
    }

    #[cfg(windows)]
    #[test]
    fn lab_keyring_round_trip() {
        let s = KeyringStore::new("SpaghettiLabUnitTest".into());
        s.del("t").unwrap();
        assert_eq!(s.get("t").unwrap(), None);
        s.set("t", "sslab_rt_fake").unwrap();
        assert_eq!(s.get("t").unwrap().as_deref(), Some("sslab_rt_fake"));
        s.del("t").unwrap();
        assert_eq!(s.get("t").unwrap(), None);
    }

    /// Manual: the app's real token code against the dev server, started by
    /// app/e2e/live/dev-auth-check.mjs, which passes a fresh code. Prints status codes and field
    /// names, never a value.
    #[test]
    #[ignore]
    fn lab_live_dev_check() {
        run(async {
            let base = api::check_base(&std::env::var("LAB_LIVE_API").expect("LAB_LIVE_API"), false).unwrap();
            assert!(base.starts_with("https://dev."), "this check runs on dev only");
            let code = std::env::var("LAB_LIVE_CODE").expect("LAB_LIVE_CODE");
            let verifier = std::env::var("LAB_LIVE_VERIFIER").expect("LAB_LIVE_VERIFIER");
            let store = Arc::new(MemoryStore::default());
            let a = Auth::new(base, store.clone());
            let fields = |v: &serde_json::Value| v.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>().join(",")).unwrap_or_else(|| "(not an object)".into());
            let null_uuid = "00000000-0000-0000-0000-000000000000";

            let acc = a.exchange(&code, &verifier).await.expect("exchange");
            println!("exchange: ok, signed_in={} has_nickname={} has_avatar={}", acc.signed_in, acc.nickname.is_some(), acc.avatar.is_some());
            let first_refresh = store.get(ACCOUNT_REFRESH).unwrap().unwrap();
            let first_access = a.access_token(None).await.unwrap();
            println!("token prefixes: access={} refresh={}", first_access.starts_with("sslab_at_"), first_refresh.starts_with("sslab_rt_"));

            let c = a.client();
            let s = Answer::read(c.get(a.url(api::PATH_STATUS)).send().await.unwrap()).await.unwrap();
            println!("status: {} fields={} parses={}", s.status, fields(&s.json), s.parse::<api::ServiceStatus>().is_some());
            if let Some(caps) = s.json.get("capabilities") {
                println!("status.capabilities fields={}", fields(caps));
            }

            let q = a.send_simple(reqwest::Method::GET, api::PATH_QUOTA, Duration::from_secs(30)).await;
            match &q {
                Ok(q) => println!("quota: {} fields={} parses={}", q.status, fields(&q.json), q.quota().is_some()),
                Err(e) => println!("quota: mapped to code {}", e.code),
            }

            match a.send_simple(reqwest::Method::GET, &api::path_job(null_uuid), Duration::from_secs(30)).await {
                Ok(j) => println!("job(unknown id): {} fields={}", j.status, fields(&j.json)),
                Err(e) => println!("job(unknown id): mapped to code {}", e.code),
            }
            // Outside the allow-list: 403 scope, not a sign-out
            match a.send_simple(reqwest::Method::GET, &api::path_file(null_uuid), Duration::from_secs(30)).await {
                Ok(x) => println!("GET user-file (out of scope): {} error={:?}", x.status, x.error()),
                Err(e) => println!("GET user-file (out of scope): mapped to code {}", e.code),
            }
            match a.send_simple(reqwest::Method::DELETE, &api::path_file(null_uuid), Duration::from_secs(30)).await {
                Ok(x) => println!("DELETE unknown file: {} fields={}", x.status, fields(&x.json)),
                Err(e) => println!("DELETE unknown file: mapped to code {}", e.code),
            }

            // Rotation by the app's own code path
            let second_access = a.access_token(Some(first_access.clone())).await.expect("refresh");
            let second_refresh = store.get(ACCOUNT_REFRESH).unwrap().unwrap();
            println!("refresh: rotated refresh={} rotated access={}", second_refresh != first_refresh, second_access != first_access);
            let old = Answer::read(c.get(a.url(api::PATH_QUOTA)).bearer_auth(&first_access).send().await.unwrap()).await.unwrap();
            println!("old access token after refresh: {} error={:?}", old.status, old.error());
            let bad = Answer::read(c.get(a.url(api::PATH_QUOTA)).bearer_auth("sslab_at_notarealtoken").send().await.unwrap()).await.unwrap();
            println!("made-up access token: {} error={:?}", bad.status, bad.error());

            // Field names of a raw refresh answer, parsed by the app's own struct
            let body = serde_json::json!({"grant_type": "refresh_token", "refresh_token": second_refresh});
            let raw = Answer::read(c.post(a.url(api::PATH_TOKEN)).json(&body).send().await.unwrap()).await.unwrap();
            println!("raw refresh: {} fields={}", raw.status, fields(&raw.json));
            if let Some(p) = raw.json.get("player") {
                println!("player fields={} id_is_string={}", fields(p), p["player_user_id"].is_string());
            }
            let t: TokenResponse = raw.parse().expect("TokenResponse parses");
            println!("token_type_bearer={} scope_lab={} expires_in={}", t.token_type.as_deref() == Some("Bearer"), t.scope.as_deref() == Some("lab"), t.expires_in);
            store.set(ACCOUNT_REFRESH, &t.refresh_token).unwrap();

            let unknown = serde_json::json!({"grant_type": "refresh_token", "refresh_token": "sslab_rt_notarealtoken"});
            let b = Answer::read(c.post(a.url(api::PATH_TOKEN)).json(&unknown).send().await.unwrap()).await.unwrap();
            println!("made-up refresh token: {} error={:?}", b.status, b.error());

            // Revoke by the app's code path, then the refresh token must be dead
            a.sign_out().await.unwrap();
            println!("signed out: signed_in={}", a.account().await.signed_in);
            let dead = serde_json::json!({"grant_type": "refresh_token", "refresh_token": t.refresh_token});
            let after = Answer::read(c.post(a.url(api::PATH_TOKEN)).json(&dead).send().await.unwrap()).await.unwrap();
            println!("refresh after revoke: {} error={:?}", after.status, after.error());
            let again = serde_json::json!({"refresh_token": t.refresh_token});
            let r = Answer::read(c.post(a.url(api::PATH_REVOKE)).json(&again).send().await.unwrap()).await.unwrap();
            println!("revoke again: {} fields={}", r.status, fields(&r.json));
        });
    }
}
