use std::time::Duration;

use cerememory_core::protocol::{EncodeBatchStoreRawRequest, EncodeStoreRawRequest};
use cerememory_core::types::{RawVisibility, SecrecyLevel};
use cerememory_recorder::capture::{capture_event_to_raw_request, CaptureEvent, CaptureEventType};
use cerememory_recorder::config::DEFAULT_MAX_EVENT_BYTES;
use cerememory_recorder::{
    ingest::ingest_reader, run_doctor, run_doctor_with_options, DoctorOptions, RecorderClient,
    RecorderConfig, Spool,
};
use serde_json::json;
use std::path::{Path, PathBuf};
use tokio::io::BufReader;
use uuid::Uuid;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn record(session_id: &str, text: &str) -> EncodeStoreRawRequest {
    capture_event_to_raw_request(CaptureEvent {
        session_id: session_id.to_string(),
        event_type: CaptureEventType::UserMessage,
        content: json!(text),
        turn_id: None,
        topic_id: None,
        action_id: None,
        source: Some("test".to_string()),
        speaker: None,
        timestamp: None,
        visibility: None,
        secrecy_level: None,
        metadata: None,
        meta: None,
    })
    .unwrap()
}

fn batch_response(session_id: &str) -> serde_json::Value {
    json!({
        "results": [{
            "record_id": Uuid::now_v7(),
            "session_id": session_id,
            "visibility": "normal",
            "secrecy_level": "sensitive"
        }]
    })
}

fn test_config(server_url: String, spool_dir: std::path::PathBuf) -> RecorderConfig {
    RecorderConfig::new(server_url)
        .unwrap()
        .with_spool_dir(spool_dir)
        .with_max_event_bytes(DEFAULT_MAX_EVENT_BYTES)
        .with_batch_max_records(1)
        .with_flush_interval(Duration::from_millis(50))
}

fn spool_dir(temp: &tempfile::TempDir) -> PathBuf {
    temp.path().join("spool")
}

fn create_private_dir(path: &Path) {
    std::fs::create_dir_all(path).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
}

#[tokio::test]
async fn client_sends_authorized_raw_batch() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .and(header("authorization", "Bearer secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch_response("sess-1")))
        .expect(1)
        .mount(&server)
        .await;

    let client =
        RecorderClient::with_timeout(&server.uri(), Some("secret"), Duration::from_secs(5))
            .unwrap();
    let response = client
        .send_raw_batch(&EncodeBatchStoreRawRequest {
            header: None,
            records: vec![record("sess-1", "hello")],
        })
        .await
        .unwrap();

    assert_eq!(response.results.len(), 1);
    assert_eq!(response.results[0].session_id, "sess-1");
}

#[tokio::test]
async fn client_trims_authorization_secret() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .and(header("authorization", "Bearer secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch_response("sess-1")))
        .expect(1)
        .mount(&server)
        .await;

    let client =
        RecorderClient::with_timeout(&server.uri(), Some(" secret "), Duration::from_secs(5))
            .unwrap();
    let response = client
        .send_raw_batch(&EncodeBatchStoreRawRequest {
            header: None,
            records: vec![record("sess-1", "hello")],
        })
        .await
        .unwrap();

    assert_eq!(response.results.len(), 1);
}

#[tokio::test]
async fn client_marks_429_and_5xx_retryable() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
        .expect(1)
        .mount(&server)
        .await;

    let client = RecorderClient::with_timeout(&server.uri(), None, Duration::from_secs(5)).unwrap();
    let err = client
        .send_raw_batch(&EncodeBatchStoreRawRequest {
            header: None,
            records: vec![record("sess-429", "hello")],
        })
        .await
        .unwrap_err();

    assert_eq!(err.status, Some(429));
    assert!(err.retryable);
}

#[tokio::test]
async fn client_rejects_mismatched_batch_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"results": []})))
        .expect(1)
        .mount(&server)
        .await;

    let client = RecorderClient::with_timeout(&server.uri(), None, Duration::from_secs(5)).unwrap();
    let err = client
        .send_raw_batch(&EncodeBatchStoreRawRequest {
            header: None,
            records: vec![record("sess-mismatch", "hello")],
        })
        .await
        .unwrap_err();

    assert!(!err.retryable);
    assert!(err.to_string().contains("1 request record"));
}

#[tokio::test]
async fn ingest_spools_failed_batch_and_flush_pending_deletes_it() {
    let failing_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(500).set_body_string("down"))
        .expect(1)
        .mount(&failing_server)
        .await;
    let temp = tempfile::tempdir().unwrap();
    let spool_dir = spool_dir(&temp);
    let config = test_config(failing_server.uri(), spool_dir.clone());
    let line = r#"{"session_id":"sess-spool","event_type":"user_message","content":"hello"}"#;
    let reader = BufReader::new(line.as_bytes());

    let stats = ingest_reader(reader, config).await.unwrap();
    assert_eq!(stats.accepted, 1);
    assert_eq!(stats.spooled, 1);
    assert_eq!(spool_file_count(&spool_dir), 1);

    let success_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch_response("sess-spool")))
        .expect(1)
        .mount(&success_server)
        .await;
    let flush_config = test_config(success_server.uri(), spool_dir.clone());
    let client = RecorderClient::new(&flush_config).unwrap();
    let flushed = Spool::new(spool_dir.clone())
        .flush_pending(&client, 10)
        .await
        .unwrap();

    assert_eq!(flushed, 1);
    assert_eq!(spool_file_count(&spool_dir), 0);
}

#[tokio::test]
async fn flush_pending_ignores_unrelated_jsonl_files() {
    let server = MockServer::start().await;
    let temp = tempfile::tempdir().unwrap();
    let spool_dir = spool_dir(&temp);
    create_private_dir(&spool_dir);
    let unrelated = spool_dir.join("notes.jsonl");
    std::fs::write(&unrelated, "not a recorder spool file\n").unwrap();
    let config = test_config(server.uri(), spool_dir.clone());
    let client = RecorderClient::new(&config).unwrap();

    let flushed = Spool::new(spool_dir)
        .flush_pending(&client, 10)
        .await
        .unwrap();

    assert_eq!(flushed, 0);
    assert!(unrelated.exists());
}

#[tokio::test]
async fn flush_pending_quarantines_unreadable_batch_and_continues() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch_response("sess-good")))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let spool_dir = spool_dir(&temp);
    create_private_dir(&spool_dir);
    std::fs::write(spool_dir.join("batch-000.jsonl"), "not-json\n").unwrap();
    Spool::new(spool_dir.clone())
        .spool_batch(&[record("sess-good", "send me")])
        .unwrap();

    let config = test_config(server.uri(), spool_dir.clone());
    let client = RecorderClient::new(&config).unwrap();
    let flushed = Spool::new(spool_dir.clone())
        .flush_pending(&client, 10)
        .await
        .unwrap();

    assert_eq!(flushed, 1);
    assert_eq!(spool_file_count(&spool_dir), 0);
    assert!(std::fs::read_dir(&spool_dir)
        .unwrap()
        .filter_map(Result::ok)
        .any(|entry| entry
            .file_name()
            .to_string_lossy()
            .starts_with("batch-000.jsonl.bad-")));
}

#[tokio::test]
async fn ingest_spools_then_errors_on_non_retryable_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid api_key=abcdef123456"))
        .expect(1)
        .mount(&server)
        .await;
    let temp = tempfile::tempdir().unwrap();
    let spool_dir = spool_dir(&temp);
    let config = test_config(server.uri(), spool_dir.clone());
    let line = r#"{"session_id":"sess-auth","event_type":"user_message","content":"hello"}"#;

    let err = ingest_reader(BufReader::new(line.as_bytes()), config)
        .await
        .unwrap_err();

    assert!(err.to_string().contains("send failed"));
    assert!(!err.to_string().contains("abcdef123456"));
    assert_eq!(spool_file_count(&spool_dir), 1);
}

#[tokio::test]
async fn ingest_preserves_repeated_content_without_explicit_identity() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch_response("sess-repeat")))
        .expect(2)
        .mount(&server)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let config = test_config(server.uri(), spool_dir(&temp));
    let input = concat!(
        r#"{"session_id":"sess-repeat","event_type":"user_message","content":"same"}"#,
        "\n",
        r#"{"session_id":"sess-repeat","event_type":"user_message","content":"same"}"#,
        "\n"
    );
    let stats = ingest_reader(BufReader::new(input.as_bytes()), config)
        .await
        .unwrap();

    assert_eq!(stats.accepted, 2);
    assert_eq!(stats.duplicate, 0);
    assert_eq!(stats.sent, 2);
}

#[tokio::test]
async fn ingest_spools_pending_batch_before_invalid_event_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(500).set_body_string("should not be called"))
        .expect(0)
        .mount(&server)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let spool_dir = spool_dir(&temp);
    let config = test_config(server.uri(), spool_dir.clone())
        .with_batch_max_records(10)
        .with_flush_interval(Duration::from_secs(3600));
    let input = concat!(
        r#"{"session_id":"sess-invalid","event_type":"user_message","content":"preserve me"}"#,
        "\n",
        "not json\n"
    );

    let err = ingest_reader(BufReader::new(input.as_bytes()), config)
        .await
        .unwrap_err();

    assert!(err.to_string().contains("invalid capture event on line 2"));
    assert_eq!(spool_file_count(&spool_dir), 1);
    let spooled = std::fs::read_to_string(
        std::fs::read_dir(&spool_dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .unwrap(),
    )
    .unwrap();
    let record: EncodeStoreRawRequest = serde_json::from_str(spooled.lines().next().unwrap())
        .expect("spool line should contain a raw request");
    assert_eq!(record.session_id, "sess-invalid");
    assert_eq!(
        std::str::from_utf8(&record.content.blocks[0].data).unwrap(),
        "preserve me"
    );
}

#[tokio::test]
async fn ingest_keeps_distinct_content_with_same_turn_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch_response("sess-turn")))
        .expect(2)
        .mount(&server)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let config = test_config(server.uri(), spool_dir(&temp));
    let input = concat!(
        r#"{"session_id":"sess-turn","turn_id":"turn-1","event_type":"user_message","content":"first"}"#,
        "\n",
        r#"{"session_id":"sess-turn","turn_id":"turn-1","event_type":"user_message","content":"second"}"#,
        "\n"
    );
    let stats = ingest_reader(BufReader::new(input.as_bytes()), config)
        .await
        .unwrap();

    assert_eq!(stats.accepted, 2);
    assert_eq!(stats.duplicate, 0);
    assert_eq!(stats.sent, 2);
}

#[tokio::test]
async fn ingest_dedupes_events_with_same_explicit_identity() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(200).set_body_json(batch_response("sess-dedupe")))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let config = test_config(server.uri(), spool_dir(&temp));
    let input = concat!(
        r#"{"session_id":"sess-dedupe","turn_id":"turn-1","event_type":"user_message","content":"same"}"#,
        "\n",
        r#"{"session_id":"sess-dedupe","turn_id":"turn-1","event_type":"user_message","content":"same"}"#,
        "\n"
    );
    let stats = ingest_reader(BufReader::new(input.as_bytes()), config)
        .await
        .unwrap();

    assert_eq!(stats.accepted, 1);
    assert_eq!(stats.duplicate, 1);
    assert_eq!(stats.sent, 1);
}

#[tokio::test]
async fn doctor_distinguishes_health_readiness_spool_and_auth_failure() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status":"ok"})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/readiness"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status":"ready"})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "code": "unauthorized",
            "message": "Invalid API key"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let report = run_doctor(test_config(server.uri(), spool_dir(&temp)))
        .await
        .unwrap();

    assert!(!report.ok());
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "health" && check.ok));
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "readiness" && check.ok));
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "spool" && check.ok));
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "auth" && !check.ok));
}

#[tokio::test]
async fn doctor_can_skip_raw_ingest_probe() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status":"ok"})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/readiness"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status":"ready"})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/encode/raw/batch"))
        .respond_with(ResponseTemplate::new(500).set_body_string("should not be called"))
        .expect(0)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/recall/raw"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "records": [],
            "total_candidates": 0
        })))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let report = run_doctor_with_options(
        test_config(server.uri(), spool_dir(&temp)),
        DoctorOptions {
            raw_ingest_probe: false,
        },
    )
    .await
    .unwrap();

    assert!(report.ok());
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "auth" && check.ok));
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "raw_ingest" && check.detail.contains("skipped")));
}

#[tokio::test]
async fn doctor_skip_raw_ingest_still_reports_auth_failure() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status":"ok"})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/readiness"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status":"ready"})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/recall/raw"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "code": "unauthorized",
            "message": "Invalid API key"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempfile::tempdir().unwrap();
    let report = run_doctor_with_options(
        test_config(server.uri(), spool_dir(&temp)),
        DoctorOptions {
            raw_ingest_probe: false,
        },
    )
    .await
    .unwrap();

    assert!(!report.ok());
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "auth" && !check.ok));
    assert!(report
        .checks
        .iter()
        .any(|check| check.name == "raw_ingest" && check.detail.contains("skipped")));
}

#[tokio::test]
async fn capture_defaults_are_normal_and_sensitive() {
    let req = record("sess-defaults", "hello");
    assert_eq!(req.visibility, RawVisibility::Normal);
    assert_eq!(req.secrecy_level, SecrecyLevel::Sensitive);
}

fn spool_file_count(path: &std::path::Path) -> usize {
    std::fs::read_dir(path)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jsonl"))
        .count()
}
