//! Cerememory Recorder companion.
//!
//! The recorder is intentionally a thin companion: it accepts observed events,
//! normalizes them into raw journal records, and writes those records to the
//! Cerememory HTTP API. It does not plan, execute work, or curate memories.

pub mod capture;
pub mod client;
pub mod config;
pub mod doctor;
pub mod error;
pub mod hook;
pub mod ingest;
pub mod redact;
pub mod spool;

pub use capture::{capture_event_to_raw_request, parse_capture_event_line, CaptureEvent};
pub use client::{RecorderClient, SendError};
pub use config::RecorderConfig;
pub use doctor::{run_doctor, run_doctor_with_options, DoctorOptions, DoctorReport};
pub use error::RecorderError;
pub use ingest::run_ingest;
pub use spool::Spool;
