use cerememory_core::protocol::EncodeStoreRawRequest;
use cerememory_core::types::{
    ContentBlock, MemoryContent, MetaMemory, Modality, RawSource, RawSpeaker, RawVisibility,
    SecrecyLevel,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::redact::{redact_string, redact_value};
use crate::RecorderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureEventType {
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    Command,
    FileChange,
    SessionSummary,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureEvent {
    pub session_id: String,
    pub event_type: CaptureEventType,
    pub content: Value,
    #[serde(default)]
    pub turn_id: Option<String>,
    #[serde(default)]
    pub topic_id: Option<String>,
    #[serde(default)]
    pub action_id: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub speaker: Option<RawSpeaker>,
    #[serde(default)]
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(default)]
    pub visibility: Option<RawVisibility>,
    #[serde(default)]
    pub secrecy_level: Option<SecrecyLevel>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub meta: Option<MetaMemory>,
}

pub fn parse_capture_event_line(
    line: &str,
    line_number: usize,
    max_event_bytes: usize,
) -> Result<Option<CaptureEvent>, RecorderError> {
    let size = line.len();
    if size > max_event_bytes {
        return Err(RecorderError::EventTooLarge {
            size,
            limit: max_event_bytes,
        });
    }

    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let event = serde_json::from_str::<CaptureEvent>(trimmed).map_err(|err| {
        RecorderError::InvalidEvent {
            line: line_number,
            message: err.to_string(),
        }
    })?;

    if event.session_id.trim().is_empty() {
        return Err(RecorderError::InvalidEvent {
            line: line_number,
            message: "session_id must not be empty".to_string(),
        });
    }
    if content_to_text(&event.content)
        .map(|text| text.trim().is_empty())
        .unwrap_or(true)
    {
        return Err(RecorderError::InvalidEvent {
            line: line_number,
            message: "content must not be empty".to_string(),
        });
    }

    Ok(Some(event))
}

pub fn capture_event_to_raw_request(
    event: CaptureEvent,
) -> Result<EncodeStoreRawRequest, RecorderError> {
    let redacted_content = redact_value(event.content.clone());
    let meta = event.meta.clone().map(redact_meta_memory).transpose()?;
    let text = redact_string(&content_to_text(&redacted_content).ok_or_else(|| {
        RecorderError::InvalidEvent {
            line: 0,
            message: "content must not be empty".to_string(),
        }
    })?);
    if text.trim().is_empty() {
        return Err(RecorderError::InvalidEvent {
            line: 0,
            message: "content must not be empty".to_string(),
        });
    }

    let metadata = build_metadata(&event);

    Ok(EncodeStoreRawRequest {
        header: None,
        session_id: event.session_id.trim().to_string(),
        turn_id: trimmed_option(&event.turn_id),
        topic_id: trimmed_option(&event.topic_id),
        source: raw_source_for(event.event_type),
        speaker: event
            .speaker
            .unwrap_or_else(|| raw_speaker_for(event.event_type)),
        visibility: event.visibility.unwrap_or(RawVisibility::Normal),
        secrecy_level: event.secrecy_level.unwrap_or(SecrecyLevel::Sensitive),
        content: MemoryContent {
            blocks: vec![ContentBlock {
                modality: Modality::Text,
                format: "text/plain".to_string(),
                data: text.into_bytes(),
                embedding: None,
            }],
            summary: None,
        },
        metadata: Some(metadata),
        meta,
    })
}

pub fn capture_event_dedupe_key(event: &CaptureEvent) -> Option<String> {
    let has_explicit_identity = trimmed_option(&event.turn_id).is_some()
        || trimmed_option(&event.action_id).is_some()
        || event.timestamp.is_some();
    if !has_explicit_identity {
        return None;
    }

    Some(format!(
        "{}|{:?}|{}|{}|{}|{}|{}",
        event.session_id.trim(),
        event.event_type,
        trimmed_option(&event.turn_id).unwrap_or_default(),
        trimmed_option(&event.action_id).unwrap_or_default(),
        event
            .timestamp
            .map(|timestamp| timestamp.to_rfc3339())
            .unwrap_or_default(),
        trimmed_option(&event.source).unwrap_or_default(),
        content_to_text(&event.content).unwrap_or_default()
    ))
}

fn content_to_text(content: &Value) -> Option<String> {
    match content {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        value => serde_json::to_string(value).ok(),
    }
}

fn raw_source_for(event_type: CaptureEventType) -> RawSource {
    match event_type {
        CaptureEventType::UserMessage | CaptureEventType::AssistantMessage => {
            RawSource::Conversation
        }
        CaptureEventType::SessionSummary => RawSource::Summary,
        CaptureEventType::ToolCall
        | CaptureEventType::ToolResult
        | CaptureEventType::Command
        | CaptureEventType::FileChange
        | CaptureEventType::Error => RawSource::ToolIo,
    }
}

fn raw_speaker_for(event_type: CaptureEventType) -> RawSpeaker {
    match event_type {
        CaptureEventType::UserMessage => RawSpeaker::User,
        CaptureEventType::AssistantMessage => RawSpeaker::Assistant,
        CaptureEventType::ToolCall | CaptureEventType::ToolResult | CaptureEventType::Command => {
            RawSpeaker::Tool
        }
        CaptureEventType::FileChange
        | CaptureEventType::SessionSummary
        | CaptureEventType::Error => RawSpeaker::System,
    }
}

fn build_metadata(event: &CaptureEvent) -> Value {
    let mut object = Map::new();
    object.insert("recorder".to_string(), json!("cerememory-recorder"));
    object.insert("event_type".to_string(), json!(event.event_type));

    if let Some(action_id) = trimmed_option(&event.action_id) {
        object.insert("action_id".to_string(), json!(redact_string(&action_id)));
    }
    if let Some(source) = trimmed_option(&event.source) {
        object.insert("capture_source".to_string(), json!(redact_string(&source)));
    }
    if let Some(timestamp) = event.timestamp {
        object.insert("observed_at".to_string(), json!(timestamp.to_rfc3339()));
    }
    if let Some(cwd) = std::env::current_dir()
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok())
    {
        object.insert("recorder_cwd".to_string(), json!(cwd));
    }

    if let Some(metadata) = event.metadata.clone() {
        object.insert("capture_metadata".to_string(), redact_value(metadata));
    }

    Value::Object(object)
}

fn trimmed_option(value: &Option<String>) -> Option<String> {
    value
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn redact_meta_memory(meta: MetaMemory) -> Result<MetaMemory, RecorderError> {
    let value = serde_json::to_value(meta)?;
    let redacted = redact_value(value);
    Ok(serde_json::from_value(redacted)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_content(req: &EncodeStoreRawRequest) -> &str {
        std::str::from_utf8(&req.content.blocks[0].data).unwrap()
    }

    #[test]
    fn maps_user_message_to_sensitive_raw_conversation() {
        let event = CaptureEvent {
            session_id: "sess-1".to_string(),
            event_type: CaptureEventType::UserMessage,
            content: json!("hello"),
            turn_id: Some(" turn-1 ".to_string()),
            topic_id: None,
            action_id: None,
            source: Some("codex".to_string()),
            speaker: None,
            timestamp: None,
            visibility: None,
            secrecy_level: None,
            metadata: None,
            meta: None,
        };

        let req = capture_event_to_raw_request(event).unwrap();
        assert_eq!(req.source, RawSource::Conversation);
        assert_eq!(req.speaker, RawSpeaker::User);
        assert_eq!(req.visibility, RawVisibility::Normal);
        assert_eq!(req.secrecy_level, SecrecyLevel::Sensitive);
        assert_eq!(req.turn_id.as_deref(), Some("turn-1"));
        assert_eq!(text_content(&req), "hello");
    }

    #[test]
    fn maps_tool_call_to_tool_io() {
        let event = CaptureEvent {
            session_id: "sess-1".to_string(),
            event_type: CaptureEventType::ToolCall,
            content: json!({"tool": "exec", "cmd": "ls", "api_key": "abcdef123456"}),
            turn_id: None,
            topic_id: None,
            action_id: Some("act-1".to_string()),
            source: None,
            speaker: None,
            timestamp: None,
            visibility: None,
            secrecy_level: None,
            metadata: Some(json!({"tool": "exec"})),
            meta: None,
        };

        let req = capture_event_to_raw_request(event).unwrap();
        assert_eq!(req.source, RawSource::ToolIo);
        assert_eq!(req.speaker, RawSpeaker::Tool);
        assert!(text_content(&req).contains("\"tool\":\"exec\""));
        assert!(!text_content(&req).contains("abcdef123456"));
    }

    #[test]
    fn redacts_caller_supplied_meta_memory() {
        let mut meta = MetaMemory::unavailable("test");
        meta.intent = Some("api_key=abcdef123456".to_string());
        meta.evidence.push(cerememory_core::types::MetaEvidenceRef {
            excerpt: Some("Authorization: Bearer deadbeefcafebabe".to_string()),
            ..Default::default()
        });
        let event = CaptureEvent {
            session_id: "sess-1".to_string(),
            event_type: CaptureEventType::SessionSummary,
            content: json!("summary"),
            turn_id: None,
            topic_id: None,
            action_id: None,
            source: None,
            speaker: None,
            timestamp: None,
            visibility: None,
            secrecy_level: None,
            metadata: None,
            meta: Some(meta),
        };

        let req = capture_event_to_raw_request(event).unwrap();
        let meta = req.meta.unwrap();
        assert_eq!(meta.intent.as_deref(), Some("api_key=[REDACTED]"));
        assert_eq!(
            meta.evidence[0].excerpt.as_deref(),
            Some("Authorization: [REDACTED]")
        );
    }

    #[test]
    fn rejects_invalid_jsonl_empty_content_and_large_events() {
        let err = parse_capture_event_line("not json", 7, 1024).unwrap_err();
        assert!(err.to_string().contains("line 7"));

        let err = parse_capture_event_line(
            r#"{"session_id":"s","event_type":"user_message","content":""}"#,
            1,
            1024,
        )
        .unwrap_err();
        assert!(err.to_string().contains("content must not be empty"));

        let err = parse_capture_event_line(
            r#"{"session_id":"s","event_type":"user_message","content":"xxxxxxxx"}"#,
            1,
            10,
        )
        .unwrap_err();
        assert!(matches!(err, RecorderError::EventTooLarge { .. }));

        let padded = format!(
            "   {}   ",
            r#"{"session_id":"s","event_type":"user_message","content":"x"}"#
        );
        let err = parse_capture_event_line(&padded, 1, padded.trim().len()).unwrap_err();
        assert!(matches!(err, RecorderError::EventTooLarge { .. }));
    }

    #[test]
    fn dedupe_key_requires_explicit_identity() {
        let mut event = CaptureEvent {
            session_id: "sess-1".to_string(),
            event_type: CaptureEventType::UserMessage,
            content: json!("repeatable"),
            turn_id: None,
            topic_id: None,
            action_id: None,
            source: None,
            speaker: None,
            timestamp: None,
            visibility: None,
            secrecy_level: None,
            metadata: None,
            meta: None,
        };
        assert!(capture_event_dedupe_key(&event).is_none());

        event.turn_id = Some("turn-1".to_string());
        assert!(capture_event_dedupe_key(&event).is_some());

        let original = capture_event_dedupe_key(&event).unwrap();
        event.content = json!("different content in the same turn");
        assert_ne!(capture_event_dedupe_key(&event).unwrap(), original);
    }
}
