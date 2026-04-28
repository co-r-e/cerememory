use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

static BEARER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}").unwrap());
static BASIC_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{8,}").unwrap());
static AUTH_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?imx)
        ^(\s*authorization\s*:\s*)
        (?:Bearer\s+[A-Za-z0-9._~+/=-]{8,}|Basic\s+[A-Za-z0-9+/=]{8,})
        "#,
    )
    .unwrap()
});
static HEADER_SECRET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?imx)
        ^(\s*)(x-api-key|api-key|cookie|set-cookie)
        \s*:\s*
        [^\r\n]*
        "#,
    )
    .unwrap()
});
static ASSIGNMENT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?ix)
        \b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|client[_-]?secret|private[_-]?key|password|passwd)
        (\s*[:=]\s*)
        (?:
            "[^"\s]{8,}"
          | '[^'\s]{8,}'
          | [A-Za-z0-9._~+/=:-]{8,}
        )
        "#,
    )
    .unwrap()
});
static JSON_SECRET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?ix)
        ("(?:x[_-]?api[_-]?key|api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|client[_-]?secret|private[_-]?key|authorization|cookie|set[_-]?cookie|password|passwd)"\s*:\s*")
        [^"]+
        (")
        "#,
    )
    .unwrap()
});
static PRIVATE_KEY_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----")
        .unwrap()
});
static KNOWN_TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?x)
        \b(
            sk-[A-Za-z0-9_-]{16,}
          | sk_[A-Za-z0-9_-]{16,}
          | gh[pousr]_[A-Za-z0-9_]{20,}
          | github_pat_[A-Za-z0-9_]{20,}
          | xox[baprs]-[A-Za-z0-9-]{10,}
          | AKIA[0-9A-Z]{16}
        )\b
        "#,
    )
    .unwrap()
});

pub fn redact_string(input: &str) -> String {
    let text = AUTH_HEADER_RE.replace_all(input, "$1[REDACTED]");
    let text = BEARER_RE.replace_all(&text, "Bearer [REDACTED]");
    let text = BASIC_RE.replace_all(&text, "Basic [REDACTED]");
    let text = HEADER_SECRET_RE.replace_all(&text, "$1$2: [REDACTED]");
    let text = PRIVATE_KEY_BLOCK_RE.replace_all(&text, "[REDACTED PRIVATE KEY]");
    let text = JSON_SECRET_RE.replace_all(&text, "${1}[REDACTED]${2}");
    let text = KNOWN_TOKEN_RE.replace_all(&text, "[REDACTED]");
    ASSIGNMENT_RE
        .replace_all(&text, "$1$2[REDACTED]")
        .into_owned()
}

pub fn redact_value(value: Value) -> Value {
    match value {
        Value::String(text) => Value::String(redact_string(&text)),
        Value::Array(items) => Value::Array(items.into_iter().map(redact_value).collect()),
        Value::Object(object) => {
            let mut redacted = Map::new();
            for (key, value) in object {
                if is_secret_key(&key) {
                    redacted.insert(key, Value::String("[REDACTED]".to_string()));
                } else {
                    redacted.insert(key, redact_value(value));
                }
            }
            Value::Object(redacted)
        }
        other => other,
    }
}

fn is_secret_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(
        normalized.as_str(),
        "apikey"
            | "accesstoken"
            | "refreshtoken"
            | "token"
            | "auth"
            | "authorization"
            | "cookie"
            | "setcookie"
            | "password"
            | "passwd"
            | "secret"
            | "clientsecret"
            | "privatekey"
    ) || normalized.ends_with("secret")
        || normalized.ends_with("apikey")
        || normalized.ends_with("token")
        || normalized.ends_with("password")
        || normalized.ends_with("authorization")
        || normalized.ends_with("cookie")
        || normalized.ends_with("privatekey")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redacts_content_secret_patterns() {
        let redacted = redact_string(
            r#"Authorization: Bearer sk-test-123456789 api_key="abcdef123456" password=supersecret123"#,
        );
        assert!(redacted.contains("Authorization: [REDACTED]"));
        assert!(redacted.contains("api_key=[REDACTED]"));
        assert!(redacted.contains("password=[REDACTED]"));
        assert!(!redacted.contains("[REDACTED]\""));
        assert!(!redacted.contains("abcdef123456"));
        assert!(!redacted.contains("supersecret123"));
    }

    #[test]
    fn redacts_additional_common_secret_shapes() {
        let redacted = redact_string(
            "Authorization: Basic dXNlcjpwYXNz\nx-api-key: sk-abcdefghijklmnopqrstuvwxyz123456\nclient_secret=secret-value-12345\n-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\nAKIA1234567890ABCDEF",
        );

        assert!(redacted.contains("Authorization: [REDACTED]"));
        assert!(redacted.contains("x-api-key: [REDACTED]"));
        assert!(redacted.contains("client_secret=[REDACTED]"));
        assert!(redacted.contains("[REDACTED PRIVATE KEY]"));
        assert!(!redacted.contains("dXNlcjpwYXNz"));
        assert!(!redacted.contains("AKIA1234567890ABCDEF"));
    }

    #[test]
    fn redacts_entire_cookie_header_lines() {
        let redacted = redact_string(
            "Cookie: session=abcdef123456; csrf=deadbeefcafebabe\nSet-Cookie: refresh=supersecret123456; HttpOnly; Secure",
        );

        assert!(redacted.contains("Cookie: [REDACTED]"));
        assert!(redacted.contains("Set-Cookie: [REDACTED]"));
        assert!(!redacted.contains("abcdef123456"));
        assert!(!redacted.contains("deadbeefcafebabe"));
        assert!(!redacted.contains("supersecret123456"));
    }

    #[test]
    fn redacts_metadata_secret_patterns() {
        let value = json!({
            "token": "abcdef123456",
            "xApiKey": "sk-abcdefghijklmnopqrstuvwxyz123456",
            "sessionCookie": "session=abcdef123456; csrf=deadbeefcafebabe",
            "clientSecret": "secret-value-12345",
            "nested": {
                "message": "Authorization: Bearer deadbeefcafebabe"
            },
            "safe": "hello"
        });
        let redacted = redact_value(value);
        assert_eq!(redacted["token"], "[REDACTED]");
        assert_eq!(redacted["xApiKey"], "[REDACTED]");
        assert_eq!(redacted["sessionCookie"], "[REDACTED]");
        assert_eq!(redacted["clientSecret"], "[REDACTED]");
        assert_eq!(redacted["nested"]["message"], "Authorization: [REDACTED]");
        assert_eq!(redacted["safe"], "hello");
    }
}
