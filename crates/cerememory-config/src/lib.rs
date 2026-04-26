//! Configuration management for Cerememory.
//!
//! Loads configuration from three sources in order of precedence:
//! 1. Hard-coded defaults
//! 2. TOML configuration file (optional)
//! 3. Environment variables with `CEREMEMORY_` prefix

use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use cerememory_engine::EngineConfig;

/// Log level for tracing output.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trace => write!(f, "trace"),
            Self::Debug => write!(f, "debug"),
            Self::Info => write!(f, "info"),
            Self::Warn => write!(f, "warn"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// Log output format.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

/// LLM provider selection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmProvider {
    #[default]
    None,
    OpenAI,
    #[serde(alias = "anthropic")]
    Claude,
    #[serde(alias = "google")]
    Gemini,
}

impl std::fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::OpenAI => write!(f, "openai"),
            Self::Claude => write!(f, "claude"),
            Self::Gemini => write!(f, "gemini"),
        }
    }
}

/// Top-level server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Data directory for persistent storage.
    pub data_dir: String,

    /// HTTP server configuration.
    pub http: HttpConfig,

    /// gRPC server configuration.
    pub grpc: GrpcConfig,

    /// Authentication configuration.
    #[serde(default)]
    pub auth: AuthConfig,

    /// Security configuration.
    #[serde(default)]
    pub security: SecurityConfig,

    /// LLM provider configuration.
    #[serde(default)]
    pub llm: LlmConfig,

    /// Decay engine configuration.
    pub decay: DecayConfig,

    /// Dream processing configuration.
    pub dream: DreamConfig,

    /// Rate limiting configuration.
    pub rate_limit: RateLimitConfig,

    /// Logging configuration.
    pub log: LogConfig,
}

/// HTTP server settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HttpConfig {
    /// Port for the HTTP server.
    pub port: u16,

    /// Bind address (default: "127.0.0.1"). Use "0.0.0.0" for network-wide access.
    pub bind_address: String,

    /// Allowed CORS origins (empty = emit no CORS headers).
    pub cors_origins: Vec<String>,

    /// Trusted proxy CIDRs for forwarded client IP extraction.
    pub trusted_proxy_cidrs: Vec<String>,

    /// Enable Prometheus metrics endpoint.
    pub metrics_enabled: bool,
}

/// gRPC server settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GrpcConfig {
    /// Port for the gRPC server (None = disabled).
    pub port: Option<u16>,

    /// Path to TLS certificate file (PEM).
    pub tls_cert_path: Option<String>,

    /// Path to TLS private key file (PEM).
    pub tls_key_path: Option<String>,
}

/// Authentication settings.
///
/// API keys are stored as plain strings in the config file but wrapped
/// in `SecretString` at runtime to prevent accidental logging.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AuthConfig {
    /// Whether authentication is enabled.
    pub enabled: bool,

    /// API keys for Bearer token authentication (raw strings in config).
    #[serde(rename = "api_keys")]
    api_keys_raw: Vec<String>,
}

impl AuthConfig {
    /// Get API keys as SecretStrings.
    pub fn api_keys(&self) -> Vec<SecretString> {
        self.api_keys_raw
            .iter()
            .map(|k| SecretString::from(k.clone()))
            .collect()
    }

    /// Get exposed API key strings (for passing to auth layer).
    pub fn api_key_strings(&self) -> Vec<String> {
        self.api_keys_raw.clone()
    }

    /// Number of configured keys.
    pub fn key_count(&self) -> usize {
        self.api_keys_raw.len()
    }
}

impl std::fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthConfig")
            .field("enabled", &self.enabled)
            .field("api_keys", &format!("[{} keys]", self.api_keys_raw.len()))
            .finish()
    }
}

/// Security settings.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    /// Optional passphrase used to encrypt newly written redb store records.
    ///
    /// Existing plaintext records remain readable for migration compatibility.
    /// Prefer `CEREMEMORY_SECURITY__STORE_ENCRYPTION_PASSPHRASE` over TOML.
    #[serde(rename = "store_encryption_passphrase")]
    store_encryption_passphrase_raw: Option<String>,
    /// Whether full-text search indexes are persisted to disk.
    ///
    /// When omitted, encrypted stores use in-memory search indexes by default.
    pub persist_search_indexes: Option<bool>,
    /// Whether the tamper-evident audit log is enabled.
    ///
    /// When omitted, server-backed engines write `audit.jsonl` under data_dir.
    pub audit_log_enabled: Option<bool>,
    /// Optional path for the tamper-evident audit log.
    ///
    /// Prefer the default under data_dir unless operationally necessary.
    pub audit_log_path: Option<String>,
}

impl SecurityConfig {
    /// Return the configured store encryption passphrase, if present.
    pub fn store_encryption_passphrase(&self) -> Option<SecretString> {
        self.store_encryption_passphrase_raw
            .as_ref()
            .map(|value| SecretString::from(value.clone()))
    }

    /// Return the exposed store encryption passphrase for engine wiring.
    pub fn store_encryption_passphrase_exposed(&self) -> Option<&str> {
        self.store_encryption_passphrase_raw.as_deref()
    }

    pub fn audit_log_enabled(&self) -> bool {
        self.audit_log_enabled.unwrap_or(true)
    }
}

impl std::fmt::Debug for SecurityConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityConfig")
            .field(
                "store_encryption_passphrase",
                &self
                    .store_encryption_passphrase_raw
                    .as_ref()
                    .map(|_| "[REDACTED]"),
            )
            .field("persist_search_indexes", &self.persist_search_indexes)
            .field("audit_log_enabled", &self.audit_log_enabled)
            .field("audit_log_path", &self.audit_log_path)
            .finish()
    }
}

/// LLM provider settings.
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// LLM provider selection.
    pub provider: LlmProvider,

    /// API key for the LLM provider (raw string in config).
    #[serde(rename = "api_key")]
    api_key_raw: Option<String>,

    /// Model name override.
    pub model: Option<String>,

    /// Base URL override for the provider API.
    pub base_url: Option<String>,
}

impl LlmConfig {
    /// Get API key as SecretString.
    pub fn api_key(&self) -> Option<SecretString> {
        self.api_key_raw
            .as_ref()
            .map(|k| SecretString::from(k.clone()))
    }

    /// Get exposed API key string.
    pub fn api_key_exposed(&self) -> Option<&str> {
        self.api_key_raw.as_deref()
    }
}

impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConfig")
            .field("provider", &self.provider)
            .field("api_key", &self.api_key_raw.as_ref().map(|_| "[REDACTED]"))
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

/// Background decay settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecayConfig {
    /// Interval in seconds for background decay ticks.
    pub background_interval_secs: u64,
}

/// Background dream processing settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DreamConfig {
    /// Interval in seconds for background dream ticks.
    pub background_interval_secs: u64,
}

/// Rate limiting settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RateLimitConfig {
    /// Maximum requests per second.
    pub requests_per_second: u64,

    /// Burst size.
    pub burst: u32,
}

/// Logging settings.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LogConfig {
    /// Log level.
    pub level: LogLevel,

    /// Log output format.
    pub format: LogFormat,
}

// ─── Defaults ────────────────────────────────────────────────────────

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            data_dir: "./data".to_string(),
            http: HttpConfig::default(),
            grpc: GrpcConfig::default(),
            auth: AuthConfig::default(),
            security: SecurityConfig::default(),
            llm: LlmConfig::default(),
            decay: DecayConfig::default(),
            dream: DreamConfig::default(),
            rate_limit: RateLimitConfig::default(),
            log: LogConfig::default(),
        }
    }
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            port: 8420,
            bind_address: "127.0.0.1".to_string(),
            cors_origins: Vec::new(),
            trusted_proxy_cidrs: Vec::new(),
            metrics_enabled: false,
        }
    }
}

// GrpcConfig: derive Default (all fields are Option/default)
// AuthConfig: derive Default (bool=false, Vec=empty)

// LlmConfig derives Default via LlmProvider::None and Option defaults.

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            background_interval_secs: 3600,
        }
    }
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            background_interval_secs: 86_400,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 100,
            burst: 50,
        }
    }
}

// LogConfig derives Default via LogLevel::Info and LogFormat::Pretty defaults.

// ─── Loading ─────────────────────────────────────────────────────────

impl ServerConfig {
    /// Load configuration from defaults, optional TOML file, and environment variables.
    ///
    /// Precedence (highest wins): env vars > TOML file > defaults.
    /// Environment variables use the `CEREMEMORY_` prefix with `__` as separator.
    /// Example: `CEREMEMORY_HTTP__PORT=9000`
    pub fn load(config_path: Option<&str>) -> Result<Self, Box<figment::Error>> {
        let mut figment = Figment::from(Serialized::defaults(ServerConfig::default()));

        if let Some(path) = config_path {
            figment = figment.merge(Toml::file(path));
        }

        figment = figment.merge(Env::prefixed("CEREMEMORY_").split("__"));

        let mut config: Self = figment.extract().map_err(Box::new)?;

        // Special handling: CEREMEMORY_AUTH_API_KEYS as comma-separated string
        // (figment's env provider can't deserialize Vec<String> from a single env var)
        if let Ok(keys_str) = std::env::var("CEREMEMORY_AUTH_API_KEYS") {
            config.auth.api_keys_raw = keys_str.split(',').map(|s| s.trim().to_string()).collect();
        }

        if let Ok(cidrs_str) = std::env::var("CEREMEMORY_HTTP__TRUSTED_PROXY_CIDRS")
            .or_else(|_| std::env::var("CEREMEMORY_HTTP_TRUSTED_PROXY_CIDRS"))
        {
            config.http.trusted_proxy_cidrs =
                cidrs_str.split(',').map(|s| s.trim().to_string()).collect();
        }

        // Note: CEREMEMORY_LLM__API_KEY is handled by figment's env provider.

        // Expand tilde in data_dir (shell does not expand ~ in env vars or TOML)
        if config.data_dir.starts_with("~/") {
            if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))
            {
                config.data_dir = std::path::Path::new(&home)
                    .join(&config.data_dir[2..])
                    .to_string_lossy()
                    .into_owned();
            }
        }

        Ok(config)
    }

    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), String> {
        if self.data_dir.trim().is_empty() {
            return Err(
                "data_dir must not be empty. Set data_dir in config or CEREMEMORY_DATA_DIR env var."
                    .to_string(),
            );
        }

        if self.http.port == 0 {
            return Err(
                "HTTP port must be non-zero. Set http.port in config or CEREMEMORY_HTTP__PORT env var."
                    .to_string(),
            );
        }

        if let Some(grpc_port) = self.grpc.port {
            if grpc_port == 0 {
                return Err(
                    "gRPC port must be non-zero. Set grpc.port in config or CEREMEMORY_GRPC__PORT env var."
                        .to_string(),
                );
            }
            if grpc_port == self.http.port {
                return Err(format!(
                    "gRPC port ({}) and HTTP port ({}) must be different.",
                    grpc_port, self.http.port
                ));
            }
        }

        // TLS: both cert and key must be provided together
        match (&self.grpc.tls_cert_path, &self.grpc.tls_key_path) {
            (Some(_), None) | (None, Some(_)) => {
                return Err("Both tls_cert_path and tls_key_path must be set together".to_string());
            }
            _ => {}
        }

        if self
            .auth
            .api_keys_raw
            .iter()
            .any(|key| key.trim().is_empty())
        {
            return Err(
                "auth.api_keys must not contain blank values. Remove empty entries or whitespace-only keys.".to_string(),
            );
        }

        if self.auth.enabled && self.auth.api_keys_raw.is_empty() {
            return Err(
                "Auth is enabled but no API keys are configured. Add keys to [auth].api_keys or set CEREMEMORY_AUTH_API_KEYS env var."
                    .to_string(),
            );
        }

        if self
            .security
            .store_encryption_passphrase_raw
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(
                "security.store_encryption_passphrase must not be blank. Remove it or provide a non-empty passphrase."
                    .to_string(),
            );
        }

        if self
            .security
            .audit_log_path
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(
                "security.audit_log_path must not be blank. Remove it or provide a non-empty path."
                    .to_string(),
            );
        }

        for (idx, cidr) in self.http.trusted_proxy_cidrs.iter().enumerate() {
            validate_cidr(cidr).map_err(|msg| {
                format!(
                    "Invalid http.trusted_proxy_cidrs[{}] '{}': {}",
                    idx, cidr, msg
                )
            })?;
        }

        if self.rate_limit.requests_per_second == 0 {
            return Err(format!(
                "rate_limit.requests_per_second must be > 0 (got {}). This controls the sustained request rate per IP.",
                self.rate_limit.requests_per_second
            ));
        }

        if self.rate_limit.burst == 0 {
            return Err(format!(
                "rate_limit.burst must be > 0 (got {}). This controls burst capacity above the sustained rate.",
                self.rate_limit.burst
            ));
        }

        if self.decay.background_interval_secs == 0 {
            return Err(
                "decay.background_interval_secs must be > 0 to enable periodic decay processing."
                    .to_string(),
            );
        }

        if self.dream.background_interval_secs == 0 {
            return Err(
                "dream.background_interval_secs must be > 0 to enable periodic dream processing."
                    .to_string(),
            );
        }

        if self.http.bind_address.is_empty() {
            return Err("http.bind_address must not be empty".to_string());
        }
        if self.http.bind_address != "localhost"
            && self.http.bind_address.parse::<std::net::IpAddr>().is_err()
        {
            return Err(format!(
                "Invalid http.bind_address '{}'. Must be a valid IP address (e.g., \"127.0.0.1\", \"0.0.0.0\") or \"localhost\".",
                self.http.bind_address
            ));
        }

        // Log level and format are validated at deserialization time by serde.

        Ok(())
    }

    /// Convert to EngineConfig for the engine crate.
    pub fn to_engine_config(&self) -> EngineConfig {
        let base = std::path::Path::new(&self.data_dir);
        EngineConfig {
            raw_journal_path: Some(base.join("raw_journal.redb").to_string_lossy().into_owned()),
            episodic_path: Some(base.join("episodic.redb").to_string_lossy().into_owned()),
            semantic_path: Some(base.join("semantic.redb").to_string_lossy().into_owned()),
            procedural_path: Some(base.join("procedural.redb").to_string_lossy().into_owned()),
            emotional_path: Some(base.join("emotional.redb").to_string_lossy().into_owned()),
            index_path: Some(base.join("text_index").to_string_lossy().into_owned()),
            vector_index_path: Some(base.join("vectors.redb").to_string_lossy().into_owned()),
            background_decay_interval_secs: Some(self.decay.background_interval_secs),
            background_dream_interval_secs: Some(self.dream.background_interval_secs),
            store_encryption_passphrase: self.security.store_encryption_passphrase_raw.clone(),
            persist_search_indexes: self
                .security
                .persist_search_indexes
                .unwrap_or(self.security.store_encryption_passphrase_raw.is_none()),
            audit_log_path: self.security.audit_log_enabled().then(|| {
                self.security
                    .audit_log_path
                    .clone()
                    .unwrap_or_else(|| base.join("audit.jsonl").to_string_lossy().into_owned())
            }),
            ..EngineConfig::default()
        }
    }
}

fn validate_cidr(value: &str) -> Result<(), String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("must not be blank".to_string());
    }

    let (addr, prefix) = value
        .split_once('/')
        .ok_or_else(|| "must be in CIDR form like 10.0.0.0/8".to_string())?;

    let prefix: u8 = prefix
        .parse()
        .map_err(|_| "prefix length must be a valid integer".to_string())?;

    match addr
        .parse::<std::net::IpAddr>()
        .map_err(|_| "network address must be a valid IP address".to_string())?
    {
        std::net::IpAddr::V4(_) if prefix <= 32 => Ok(()),
        std::net::IpAddr::V6(_) if prefix <= 128 => Ok(()),
        std::net::IpAddr::V4(_) => Err("IPv4 prefix length must be <= 32".to_string()),
        std::net::IpAddr::V6(_) => Err("IPv6 prefix length must be <= 128".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::ExposeSecret;
    use std::io::Write;

    #[test]
    fn default_config_is_valid() {
        let config = ServerConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.http.port, 8420);
        assert_eq!(config.data_dir, "./data");
        assert!(!config.auth.enabled);
        assert!(config.http.cors_origins.is_empty());
        assert!(config.http.trusted_proxy_cidrs.is_empty());
        assert!(!config.http.metrics_enabled);
    }

    #[test]
    fn load_from_defaults_only() {
        let config = ServerConfig::load(None).unwrap();
        assert_eq!(config.http.port, 8420);
        assert_eq!(config.llm.provider, LlmProvider::None);
    }

    #[test]
    fn load_from_toml_file() {
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("cerememory.toml");
        let mut f = std::fs::File::create(&toml_path).unwrap();
        writeln!(
            f,
            r#"
data_dir = "/tmp/cere"

[http]
port = 9999

[log]
level = "debug"
"#
        )
        .unwrap();

        let config = ServerConfig::load(Some(toml_path.to_str().unwrap())).unwrap();
        assert_eq!(config.http.port, 9999);
        assert_eq!(config.data_dir, "/tmp/cere");
        assert_eq!(config.log.level, LogLevel::Debug);
        // Non-overridden fields keep defaults
        assert_eq!(config.llm.provider, LlmProvider::None);
    }

    #[test]
    fn env_overrides_toml() {
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("cerememory.toml");
        let mut f = std::fs::File::create(&toml_path).unwrap();
        writeln!(
            f,
            r#"
[http]
port = 9999
"#
        )
        .unwrap();

        // Test override via figment directly (env vars would require process-level changes)
        let figment = Figment::from(Serialized::defaults(ServerConfig::default()))
            .merge(Toml::file(&toml_path))
            .merge(("http.port", 7777u16));

        let config: ServerConfig = figment.extract().unwrap();
        assert_eq!(config.http.port, 7777);
    }

    #[test]
    fn invalid_toml_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("bad.toml");
        std::fs::write(&toml_path, "this is not valid toml [[[[").unwrap();

        let result = ServerConfig::load(Some(toml_path.to_str().unwrap()));
        assert!(result.is_err());
    }

    #[test]
    fn secret_debug_is_redacted() {
        let config = AuthConfig {
            enabled: true,
            api_keys_raw: vec!["sk-secret-123".to_string()],
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("sk-secret-123"));
        assert!(debug.contains("[1 keys]"));
    }

    #[test]
    fn llm_debug_is_redacted() {
        let config = LlmConfig {
            provider: LlmProvider::OpenAI,
            api_key_raw: Some("sk-secret-key".to_string()),
            model: None,
            base_url: None,
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("sk-secret-key"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn security_debug_is_redacted() {
        let config = SecurityConfig {
            store_encryption_passphrase_raw: Some("very-secret-passphrase".to_string()),
            persist_search_indexes: Some(false),
            audit_log_enabled: Some(true),
            audit_log_path: Some("/tmp/audit.jsonl".to_string()),
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("very-secret-passphrase"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn validate_catches_zero_port() {
        let mut config = ServerConfig::default();
        config.http.port = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_catches_same_ports() {
        let mut config = ServerConfig::default();
        config.grpc.port = Some(8420); // same as HTTP
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_catches_auth_no_keys() {
        let mut config = ServerConfig::default();
        config.auth.enabled = true;
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_blank_api_keys() {
        let mut config = ServerConfig::default();
        config.auth.api_keys_raw = vec!["   ".to_string()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_blank_store_encryption_passphrase() {
        let mut config = ServerConfig::default();
        config.security.store_encryption_passphrase_raw = Some("   ".to_string());
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_blank_audit_log_path() {
        let mut config = ServerConfig::default();
        config.security.audit_log_path = Some("   ".to_string());
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_invalid_trusted_proxy_cidr() {
        let mut config = ServerConfig::default();
        config.http.trusted_proxy_cidrs = vec!["not-a-cidr".to_string()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_catches_partial_tls() {
        let mut config = ServerConfig::default();
        config.grpc.tls_cert_path = Some("/path/to/cert.pem".to_string());
        // key_path not set
        assert!(config.validate().is_err());
    }

    #[test]
    fn to_engine_config_maps_paths() {
        let config = ServerConfig {
            data_dir: "/my/data".to_string(),
            decay: DecayConfig {
                background_interval_secs: 1800,
            },
            dream: DreamConfig {
                background_interval_secs: 7200,
            },
            ..ServerConfig::default()
        };
        let ec = config.to_engine_config();
        assert_eq!(ec.raw_journal_path.unwrap(), "/my/data/raw_journal.redb");
        assert_eq!(ec.episodic_path.unwrap(), "/my/data/episodic.redb");
        assert_eq!(ec.semantic_path.unwrap(), "/my/data/semantic.redb");
        assert_eq!(ec.background_decay_interval_secs, Some(1800));
        assert_eq!(ec.background_dream_interval_secs, Some(7200));
        assert_eq!(ec.audit_log_path.unwrap(), "/my/data/audit.jsonl");
    }

    #[test]
    fn to_engine_config_maps_store_encryption_passphrase() {
        let mut config = ServerConfig::default();
        config.security.store_encryption_passphrase_raw = Some("passphrase".to_string());

        let ec = config.to_engine_config();
        assert_eq!(
            ec.store_encryption_passphrase.as_deref(),
            Some("passphrase")
        );
        assert!(!ec.persist_search_indexes);
    }

    #[test]
    fn to_engine_config_allows_persisted_search_indexes_override() {
        let mut config = ServerConfig::default();
        config.security.store_encryption_passphrase_raw = Some("passphrase".to_string());
        config.security.persist_search_indexes = Some(true);

        let ec = config.to_engine_config();
        assert!(ec.persist_search_indexes);
    }

    #[test]
    fn to_engine_config_allows_audit_log_disable_and_override() {
        let mut config = ServerConfig::default();
        config.security.audit_log_enabled = Some(false);
        assert!(config.to_engine_config().audit_log_path.is_none());

        config.security.audit_log_enabled = Some(true);
        config.security.audit_log_path = Some("/tmp/custom-audit.jsonl".to_string());
        assert_eq!(
            config.to_engine_config().audit_log_path.as_deref(),
            Some("/tmp/custom-audit.jsonl")
        );
    }

    #[test]
    fn validate_rejects_zero_dream_interval() {
        let mut config = ServerConfig::default();
        config.dream.background_interval_secs = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_decay_interval() {
        let mut config = ServerConfig::default();
        config.decay.background_interval_secs = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn auth_api_keys_as_secret() {
        let config = AuthConfig {
            enabled: true,
            api_keys_raw: vec!["key1".to_string(), "key2".to_string()],
        };
        let secrets = config.api_keys();
        assert_eq!(secrets.len(), 2);
        assert_eq!(secrets[0].expose_secret(), "key1");
    }

    #[test]
    fn invalid_log_level_rejected_at_load() {
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("cerememory.toml");
        std::fs::write(&toml_path, "[log]\nlevel = \"invalid\"\n").unwrap();
        assert!(ServerConfig::load(Some(toml_path.to_str().unwrap())).is_err());
    }

    #[test]
    fn invalid_log_format_rejected_at_load() {
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("cerememory.toml");
        std::fs::write(&toml_path, "[log]\nformat = \"yaml\"\n").unwrap();
        assert!(ServerConfig::load(Some(toml_path.to_str().unwrap())).is_err());
    }

    #[test]
    fn toml_api_keys_field_name() {
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("cerememory.toml");
        let mut f = std::fs::File::create(&toml_path).unwrap();
        writeln!(
            f,
            r#"
[auth]
enabled = true
api_keys = ["sk-test-key"]
"#
        )
        .unwrap();

        let config = ServerConfig::load(Some(toml_path.to_str().unwrap())).unwrap();
        assert_eq!(config.auth.key_count(), 1);
    }

    #[test]
    fn toml_llm_api_key_field_name() {
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("cerememory.toml");
        let mut f = std::fs::File::create(&toml_path).unwrap();
        writeln!(
            f,
            r#"
[llm]
provider = "openai"
api_key = "sk-test-llm-key"
"#
        )
        .unwrap();

        let config = ServerConfig::load(Some(toml_path.to_str().unwrap())).unwrap();
        assert_eq!(config.llm.api_key_exposed(), Some("sk-test-llm-key"));
    }

    #[test]
    fn toml_store_encryption_passphrase_field_name() {
        let dir = tempfile::tempdir().unwrap();
        let toml_path = dir.path().join("cerememory.toml");
        let mut f = std::fs::File::create(&toml_path).unwrap();
        writeln!(
            f,
            r#"
[security]
store_encryption_passphrase = "test-passphrase"
"#
        )
        .unwrap();

        let config = ServerConfig::load(Some(toml_path.to_str().unwrap())).unwrap();
        assert_eq!(
            config.security.store_encryption_passphrase_exposed(),
            Some("test-passphrase")
        );
    }
}
