//! Native Node.js bindings for Cerememory via napi-rs.
//!
//! Wraps `CerememoryEngine` directly, exposing CMP operations
//! as synchronous calls backed by a dedicated Tokio runtime.
//! No HTTP server required.

#[macro_use]
extern crate napi_derive;

mod engine;
mod types;
