use cerememory_core::protocol::EncodeBatchStoreRawRequest;
use cerememory_core::types::{RawVisibility, SecrecyLevel};
use serde_json::json;
use uuid::Uuid;

use crate::capture::{capture_event_to_raw_request, CaptureEvent, CaptureEventType};
use crate::{RecorderClient, RecorderConfig, RecorderError, Spool};

#[derive(Debug, Clone)]
pub struct DoctorCheck {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Default)]
pub struct DoctorReport {
    pub checks: Vec<DoctorCheck>,
}

#[derive(Debug, Clone, Copy)]
pub struct DoctorOptions {
    pub raw_ingest_probe: bool,
}

impl Default for DoctorOptions {
    fn default() -> Self {
        Self {
            raw_ingest_probe: true,
        }
    }
}

impl DoctorReport {
    pub fn ok(&self) -> bool {
        self.checks.iter().all(|check| check.ok)
    }

    pub fn push(&mut self, name: &'static str, ok: bool, detail: impl Into<String>) {
        self.checks.push(DoctorCheck {
            name,
            ok,
            detail: detail.into(),
        });
    }

    pub fn render_text(&self) -> String {
        self.checks
            .iter()
            .map(|check| {
                format!(
                    "{} {} - {}",
                    if check.ok { "ok" } else { "fail" },
                    check.name,
                    check.detail
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub async fn run_doctor(config: RecorderConfig) -> Result<DoctorReport, RecorderError> {
    run_doctor_with_options(config, DoctorOptions::default()).await
}

pub async fn run_doctor_with_options(
    config: RecorderConfig,
    options: DoctorOptions,
) -> Result<DoctorReport, RecorderError> {
    let client = RecorderClient::new(&config)?;
    let spool = Spool::new(config.spool_dir.clone());
    let mut report = DoctorReport::default();

    match client.health().await {
        Ok(value) => report.push("health", true, format!("server returned {value}")),
        Err(err) => report.push("health", false, err.to_string()),
    }

    match client.readiness().await {
        Ok(value) => report.push("readiness", true, format!("server returned {value}")),
        Err(err) => report.push("readiness", false, err.to_string()),
    }

    match spool.ensure_writable() {
        Ok(()) => report.push(
            "spool",
            true,
            format!("writable at {}", spool.dir().display()),
        ),
        Err(err) => report.push("spool", false, err.to_string()),
    }

    if !options.raw_ingest_probe {
        match client.raw_recall_probe().await {
            Ok(_) => report.push("auth", true, "auth-protected raw recall probe accepted"),
            Err(err) if err.status == Some(401) => {
                report.push("auth", false, "server rejected CEREMEMORY_SERVER_API_KEY")
            }
            Err(err) if err.status == Some(429) => report.push("auth", false, "rate limited"),
            Err(err) => report.push("auth", false, err.to_string()),
        }
        report.push("raw_ingest", true, "skipped by --skip-raw-ingest-probe");
        return Ok(report);
    }

    let probe = doctor_probe_record()?;
    match client
        .send_raw_batch(&EncodeBatchStoreRawRequest {
            header: None,
            records: vec![probe],
        })
        .await
    {
        Ok(response) => report.push(
            "raw_ingest",
            true,
            format!("accepted {} record(s)", response.results.len()),
        ),
        Err(err) if err.status == Some(401) => {
            report.push("auth", false, "server rejected CEREMEMORY_SERVER_API_KEY")
        }
        Err(err) if err.status == Some(429) => report.push("raw_ingest", false, "rate limited"),
        Err(err) => report.push("raw_ingest", false, err.to_string()),
    }

    Ok(report)
}

fn doctor_probe_record() -> Result<cerememory_core::protocol::EncodeStoreRawRequest, RecorderError>
{
    capture_event_to_raw_request(CaptureEvent {
        session_id: format!("recorder-doctor-{}", Uuid::now_v7()),
        event_type: CaptureEventType::SessionSummary,
        content: json!("cerememory-recorder doctor probe"),
        turn_id: None,
        topic_id: None,
        action_id: None,
        source: Some("doctor".to_string()),
        speaker: None,
        timestamp: Some(chrono::Utc::now()),
        visibility: Some(RawVisibility::PrivateScratch),
        secrecy_level: Some(SecrecyLevel::Sensitive),
        metadata: Some(json!({"doctor_probe": true})),
        meta: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_distinguishes_failures() {
        let mut report = DoctorReport::default();
        report.push("health", true, "server returned ok");
        report.push("auth", false, "server rejected CEREMEMORY_SERVER_API_KEY");
        assert!(!report.ok());
        assert!(report.render_text().contains("fail auth"));
    }
}
