// ── Spaghetti Showdown API: every URL and answer shape the desktop Lab uses ─────
//
// The login contract and the hosted analysis API are described in docs/desktop-login.md. Everything
// that could change on the server side lives here: base URLs, paths and response types. Bodies are
// parsed defensively because the same route can answer JSON, plain text (the global limiter) or HTML
// (nginx). Nothing in this module may log a token; `scrub` runs over every error text.

use serde::{Deserialize, Serialize};
use std::time::Duration;

pub const DEFAULT_API_URL: &str = "https://backend.spaghettishowdown.it";
pub const DEFAULT_WEB_URL: &str = "https://spaghettishowdown.it";
pub const DEFAULT_KEYRING_SERVICE: &str = "SpaghettiLab";

pub const PATH_TOKEN: &str = "/api/desktop-auth/token";
pub const PATH_REVOKE: &str = "/api/desktop-auth/revoke";
pub const PATH_STATUS: &str = "/api/spaghetti-lab/openers/status";
pub const PATH_QUOTA: &str = "/api/spaghetti-lab/openers/quota";
pub const PATH_OPENERS: &str = "/api/spaghetti-lab/openers";
pub const PATH_UPLOAD: &str = "/api/user-files/upload";
pub const PATH_DESKTOP_LOGIN: &str = "/desktop-login";

pub fn path_job(id: &str) -> String {
    format!("{}/{}", PATH_OPENERS, enc(id))
}
pub fn path_job_result(id: &str) -> String {
    format!("{}/{}/result", PATH_OPENERS, enc(id))
}
pub fn path_file(id: &str) -> String {
    format!("/api/user-files/{}", enc(id))
}

/// The website pages the app may open (nothing else, so the webview can't open arbitrary links here).
pub fn web_page(name: &str) -> Option<&'static str> {
    match name {
        "support" => Some("/support"),
        "terms" => Some("/terms"),
        "privacy" => Some("/privacy"),
        _ => None,
    }
}

fn enc(id: &str) -> String {
    url::form_urlencoded::byte_serialize(id.as_bytes()).collect()
}

pub fn login_url(web: &str, port: u16, state: &str, challenge: &str) -> String {
    let q: String = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("redirect_port", &port.to_string())
        .append_pair("state", state)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .finish();
    format!("{}{}?{}", web, PATH_DESKTOP_LOGIN, q)
}

// ── configuration ───────────────────────────────────────────────────────────

pub fn is_loopback(base: &str) -> bool {
    url::Url::parse(base)
        .ok()
        .and_then(|u| u.host_str().map(|h| matches!(h, "127.0.0.1" | "localhost" | "[::1]" | "::1")))
        .unwrap_or(false)
}

/// A base URL without the trailing slash. https only, except loopback; with `no_prod` only loopback
/// (every test run sets SPAGHETTI_LAB_NO_PROD, so no test can reach a real server).
pub fn check_base(raw: &str, no_prod: bool) -> Result<String, String> {
    let base = raw.trim().trim_end_matches('/').to_string();
    let u = url::Url::parse(&base).map_err(|_| format!("Not a URL: {}", base))?;
    let loopback = is_loopback(&base);
    if u.scheme() != "https" && !(u.scheme() == "http" && loopback) {
        return Err(format!("{} must be https", base));
    }
    if no_prod && !loopback {
        return Err(format!("{} refused: SPAGHETTI_LAB_NO_PROD allows only loopback addresses", base));
    }
    Ok(base)
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| default.to_string())
}

fn no_prod() -> bool {
    std::env::var_os("SPAGHETTI_LAB_NO_PROD").is_some()
}

pub fn api_base() -> Result<String, RemoteError> {
    check_base(&env_or("SPAGHETTI_LAB_API_URL", DEFAULT_API_URL), no_prod()).map_err(|m| RemoteError::new("config", m))
}

pub fn web_base() -> Result<String, RemoteError> {
    check_base(&env_or("SPAGHETTI_LAB_WEB_URL", DEFAULT_WEB_URL), no_prod()).map_err(|m| RemoteError::new("config", m))
}

pub fn keyring_service() -> String {
    env_or("SPAGHETTI_LAB_KEYRING_SERVICE", DEFAULT_KEYRING_SERVICE)
}

/// Test hooks work only against loopback servers, so a stray variable can't weaken a real session.
pub fn test_mode() -> bool {
    std::env::var_os("SPAGHETTI_LAB_TEST_BROWSER").is_some()
        && api_base().map_or(false, |b| is_loopback(&b))
        && web_base().map_or(false, |b| is_loopback(&b))
}

/// Whether the local research mode is offered at all.
pub fn local_models_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value.map_or(false, |v| !v.is_empty())
}

pub fn client_for(base: &str) -> reqwest::Client {
    // reqwest is built without a default crypto provider (as the updater's); ignore "already installed"
    let _ = rustls::crypto::ring::default_provider().install_default();
    let ua = format!("Spaghetti Lab {} ({})", env!("CARGO_PKG_VERSION"), std::env::consts::OS);
    let mut b = reqwest::Client::builder()
        .user_agent(ua)
        .connect_timeout(Duration::from_secs(30))
        // never forward a bearer token anywhere the API didn't name itself
        .redirect(reqwest::redirect::Policy::none());
    if is_loopback(base) {
        b = b.no_proxy();
    }
    b.build().expect("HTTP client")
}

// ── answers ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Player {
    pub player_user_id: String,
    #[serde(default)]
    pub nickname: Option<String>,
    #[serde(default)]
    pub avatar: Option<String>,
}

/// `POST /api/desktop-auth/token`, 200. Debug never shows the tokens.
#[derive(Clone, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    #[allow(dead_code)] // part of the contract; the live check reads it
    pub token_type: Option<String>,
    pub expires_in: u64,
    pub refresh_token: String,
    #[serde(default)]
    pub scope: Option<String>,
    pub player: Player,
}

impl std::fmt::Debug for TokenResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TokenResponse {{ expires_in: {}, scope: {:?} }}", self.expires_in, self.scope)
    }
}

/// `GET /quota`, and the same four fields on POST answers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Quota {
    pub free_remaining: i64,
    pub free_per_week: i64,
    pub credits: i64,
    pub week_resets_at: String,
}

/// `GET /status`, 200. The 503 bodies are `{"error":..}` or `{"available":false}`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServiceStatus {
    #[serde(default)]
    pub available: bool,
    #[serde(default)]
    pub running: Option<i64>,
    #[serde(default)]
    pub queued: Option<i64>,
    #[serde(default)]
    pub max_concurrent: Option<i64>,
    #[serde(default)]
    pub max_queue: Option<i64>,
    #[serde(default)]
    pub capabilities: Option<serde_json::Value>,
    #[serde(default)]
    pub free_per_week: Option<i64>,
}

/// `GET /:jobId` and the POST answers (`Job.public` + what the backend adds).
#[derive(Debug, Clone, Deserialize)]
pub struct JobView {
    pub id: String,
    pub status: String,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub position: Option<i64>,
    #[serde(default)]
    pub analysis_source: Option<String>,
    #[serde(default)]
    pub analysis_refunded: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)] // part of the contract: a 200 without analysis_source
    pub reused: Option<bool>,
}

/// `POST /api/user-files/upload`, 200: `{"success":true,"data":{..}}`.
#[derive(Debug, Clone, Deserialize)]
pub struct UploadAnswer {
    pub data: UploadData,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UploadData {
    pub file_id: String,
    #[serde(default)]
    #[allow(dead_code)] // part of the contract: "skipped" for desktop uploads
    pub compression_status: Option<String>,
}

/// Any HTTP answer: status, `Retry-After`, the text, and the JSON when it is JSON.
#[derive(Debug, Clone)]
pub struct Answer {
    pub status: u16,
    pub retry_after: Option<u64>,
    pub text: String,
    pub json: serde_json::Value,
}

impl Answer {
    pub fn new(status: u16, retry_after: Option<u64>, text: String) -> Self {
        let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
        Answer { status, retry_after, text, json }
    }
    pub async fn read(r: reqwest::Response) -> Result<Answer, RemoteError> {
        let status = r.status().as_u16();
        let retry_after = r.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse::<u64>().ok());
        let text = r.text().await.map_err(net_error)?;
        Ok(Answer::new(status, retry_after, text))
    }
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }
    pub fn error(&self) -> Option<&str> {
        self.json.get("error").and_then(|e| e.as_str())
    }
    pub fn detail(&self) -> Option<&str> {
        self.json.get("detail").and_then(|e| e.as_str())
    }
    /// The server's own sentence, `error` before `detail` (never an HTML page).
    pub fn sentence(&self) -> Option<String> {
        self.error().or_else(|| self.detail()).map(scrub)
    }
    pub fn parse<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        serde_json::from_value(self.json.clone()).ok()
    }
    pub fn quota(&self) -> Option<Quota> {
        self.parse()
    }
}

// ── errors ──────────────────────────────────────────────────────────────────

/// What the remote commands fail with: `code` for the UI, a sentence to show. See
/// docs/desktop-login.md for the list.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RemoteError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refunded: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quota: Option<Quota>,
    /// `cancelled`: what the run was doing when it was cancelled
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

impl RemoteError {
    pub fn new(code: &str, message: impl AsRef<str>) -> Self {
        RemoteError { code: code.into(), message: scrub(message.as_ref()), retry_after: None, refunded: None, quota: None, phase: None }
    }
    pub fn retry(mut self, secs: Option<u64>) -> Self {
        self.retry_after = secs;
        self
    }
    pub fn signed_out() -> Self {
        RemoteError::new("signed_out", "Please sign in again.")
    }
    pub fn unavailable() -> Self {
        RemoteError::new("unavailable", "Lab analysis is not available right now. Please try again later.")
    }
    pub fn scope() -> Self {
        RemoteError::new("scope", "The server refused this request for the Lab sign-in. This is a bug in the app: please report it.")
    }
    pub fn login_unavailable() -> Self {
        RemoteError::new("login_unavailable", "Sign-in with spaghettishowdown.it isn't available yet. The rest of the app works as usual.")
    }
}

impl std::fmt::Display for RemoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

/// A connection problem, as a sentence: the URL adds nothing for the user and stays out of errors.
pub fn net_error(e: reqwest::Error) -> RemoteError {
    if e.is_timeout() {
        RemoteError::new("network", "The server took too long to answer.")
    } else {
        RemoteError::new("network", "Could not reach the Spaghetti Showdown servers. Check the connection and try again.")
    }
}

/// Remove anything token-like from a text that may be shown or logged.
pub fn scrub(text: &str) -> String {
    fn is_tok(c: char) -> bool {
        c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '~' | '%')
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let a = rest.find("sslab_");
        let b = rest.find("code=");
        let (i, marker, keep): (usize, usize, &str) = match (a, b) {
            (Some(x), Some(y)) if y < x => (y, "code=".len(), "code=[hidden]"),
            (Some(x), _) => (x, "sslab_".len(), "[token]"),
            (None, Some(y)) => (y, "code=".len(), "code=[hidden]"),
            (None, None) => {
                out.push_str(rest);
                return out;
            }
        };
        out.push_str(&rest[..i]);
        out.push_str(keep);
        let after = &rest[i + marker..];
        let end = after.find(|c: char| !is_tok(c)).unwrap_or(after.len());
        rest = &after[end..];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_base_url_rule() {
        assert_eq!(check_base("https://x.it/", false).unwrap(), "https://x.it");
        assert_eq!(check_base("http://127.0.0.1:5000", false).unwrap(), "http://127.0.0.1:5000");
        assert!(check_base("http://x.it", false).is_err());
        assert!(check_base("https://x.it", true).is_err());
        assert!(check_base("https://backend.spaghettishowdown.it", true).is_err());
        assert!(check_base("http://127.0.0.1:5000", true).is_ok());
        assert!(check_base("nonsense", false).is_err());
        // a loopback look-alike is not loopback
        assert!(!is_loopback("https://127.0.0.1.evil.example"));
        assert!(!is_loopback("https://localhost.evil.example"));
    }

    #[test]
    fn lab_defaults_are_production() {
        assert_eq!(check_base(DEFAULT_API_URL, false).unwrap(), "https://backend.spaghettishowdown.it");
        assert_eq!(check_base(DEFAULT_WEB_URL, false).unwrap(), "https://spaghettishowdown.it");
    }

    #[test]
    fn lab_local_mode_needs_the_variable() {
        assert!(!local_models_enabled(None));
        assert!(!local_models_enabled(Some(std::ffi::OsStr::new(""))));
        assert!(local_models_enabled(Some(std::ffi::OsStr::new("C:/models"))));
    }

    #[test]
    fn lab_scrub_removes_tokens_and_codes() {
        assert_eq!(scrub("bad sslab_rt_abc-DEF_123 here"), "bad [token] here");
        assert_eq!(scrub("/callback?code=abc123&state=x"), "/callback?code=[hidden]&state=x");
        assert_eq!(scrub("sslab_at_a sslab_rt_b"), "[token] [token]");
        assert_eq!(scrub("plain text"), "plain text");
        assert!(!RemoteError::new("x", "t sslab_at_secret").message.contains("secret"));
    }

    #[test]
    fn lab_answer_tolerates_any_body() {
        let a = Answer::new(429, Some(5), "Too many requests from this IP".into());
        assert_eq!(a.error(), None);
        assert_eq!(a.detail(), None);
        assert_eq!(a.sentence(), None);
        assert_eq!(Answer::new(502, None, "<html>Bad gateway</html>".into()).sentence(), None);
        let q = Answer::new(200, None, r#"{"free_remaining":1,"free_per_week":2,"credits":3,"week_resets_at":"2026-11-09T00:00:00Z","extra":1}"#.into());
        assert_eq!(q.quota().unwrap().credits, 3);
        assert!(Answer::new(409, None, r#"{"error":"analysis_starting"}"#.into()).quota().is_none());
    }

    #[test]
    fn lab_token_debug_hides_tokens() {
        let t: TokenResponse = serde_json::from_str(
            r#"{"access_token":"sslab_at_S1","token_type":"Bearer","expires_in":3600,"refresh_token":"sslab_rt_S2","scope":"lab","player":{"player_user_id":"u","nickname":"n","avatar":null}}"#,
        )
        .unwrap();
        let d = format!("{:?}", t);
        assert!(!d.contains("S1") && !d.contains("S2"));
    }

    #[test]
    fn lab_login_url_shape() {
        let u = login_url("https://w.it", 4567, "st/ate", "ch");
        assert_eq!(u, "https://w.it/desktop-login?redirect_port=4567&state=st%2Fate&code_challenge=ch&code_challenge_method=S256");
    }

    #[test]
    fn lab_only_known_pages_open() {
        assert_eq!(web_page("support"), Some("/support"));
        assert_eq!(web_page("https://evil.example"), None);
    }
}
