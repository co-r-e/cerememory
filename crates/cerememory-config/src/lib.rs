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

    /// LLM provider configuration.
    #[serde(default)]
    pub llm: LlmConfig,

    /// Decay engine configuration.
    pub decay: DecayConfig,

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

    /// Allowed CORS origins (empty = allow all).
    pub cors_origins: Vec<String>,
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

/// LLM provider settings.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// Provider name: "openai", "claude", "gemini", or "none".
    pub provider: String,

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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LogConfig {
    /// Log level: "trace", "debug", "info", "warn", "error".
    pub level: String,

    /// Log format: "pretty" or "json".
    pub format: String,
}

// ─── Defaults ────────────────────────────────────────────────────────

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            data_dir: "./data".to_string(),
            http: HttpConfig::default(),
            grpc: GrpcConfig::default(),
            auth: AuthConfig::default(),
            llm: LlmConfig::default(),
            decay: DecayConfig::default(),
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
        }
    }
}

// GrpcConfig: derive Default (all fields are Option/default)
// AuthConfig: derive Default (bool=false, Vec=empty)

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "none".to_string(),
            api_key_raw: None,
            model: None,
            base_url: None,
        }
    }
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            background_interval_secs: 3600,
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

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
        }
    }
}

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
            config.auth.api_keys_raw = keys_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        // Note: CEREMEMORY_LLM__API_KEY is handled by figment's env provider.

        Ok(config)
    }

    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), String> {
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

        if self.auth.enabled && self.auth.api_keys_raw.is_empty() {
            return Err(
                "Auth is enabled but no API keys are configured. Add keys to [auth].api_keys or set CEREMEMORY_AUTH_API_KEYS env var."
                    .to_string(),
            );
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

        const VALID_LOG_LEVELS: &[&str] = &["trace", "debug", "info", "warn", "error"];
        if !VALID_LOG_LEVELS.contains(&self.log.level.to_lowercase().as_str()) {
            return Err(format!(
                "Invalid log.level '{}'. Valid options: {}",
                self.log.level,
                VALID_LOG_LEVELS.join(", ")
            ));
        }

        const VALID_LOG_FORMATS: &[&str] = &["pretty", "json"];
        if !VALID_LOG_FORMATS.contains(&self.log.format.to_lowercase().as_str()) {
            return Err(format!(
                "Invalid log.format '{}'. Valid options: {}",
                self.log.format,
                VALID_LOG_FORMATS.join(", ")
            ));
        }

        Ok(())
    }

    /// Convert to EngineConfig for the engine crate.
    pub fn to_engine_config(&self) -> EngineConfig {
        let data_dir = &self.data_dir;
        EngineConfig {
            episodic_path: Some(format!("{data_dir}/episodic.redb")),
            semantic_path: Some(format!("{data_dir}/semantic.redb")),
            procedural_path: Some(format!("{data_dir}/procedural.redb")),
            emotional_path: Some(format!("{data_dir}/emotional.redb")),
            index_path: Some(format!("{data_dir}/text_index")),
            vector_index_path: Some(format!("{data_dir}/vectors.redb")),
            background_decay_interval_secs: Some(self.decay.background_interval_secs),
            ..EngineConfig::default()
        }
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
    }

    #[test]
    fn load_from_defaults_only() {
        let config = ServerConfig::load(None).unwrap();
        assert_eq!(config.http.port, 8420);
        assert_eq!(config.llm.provider, "none");
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
        assert_eq!(config.log.level, "debug");
        // Non-overridden fields keep defaults
        assert_eq!(config.llm.provider, "none");
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
            provider: "openai".to_string(),
            api_key_raw: Some("sk-secret-key".to_string()),
            model: None,
            base_url: None,
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("sk-secret-key"));
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
            ..ServerConfig::default()
        };
        let ec = config.to_engine_config();
        assert_eq!(ec.episodic_path.unwrap(), "/my/data/episodic.redb");
        assert_eq!(ec.semantic_path.unwrap(), "/my/data/semantic.redb");
        assert_eq!(ec.background_decay_interval_secs, Some(1800));
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
    fn validate_catches_invalid_log_level() {
        let mut config = ServerConfig::default();
        config.log.level = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_catches_invalid_log_format() {
        let mut config = ServerConfig::default();
        config.log.format = "yaml".to_string();
        assert!(config.validate().is_err());
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
}
