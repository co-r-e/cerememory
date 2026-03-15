//! Native Python bindings for Cerememory via PyO3.
//!
//! Provides direct, in-process access to the CerememoryEngine without HTTP overhead.
//! Uses a tokio Runtime owned by the Python wrapper to bridge async Rust to sync Python.

#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod engine;
mod types;

#[pymodule]
fn cerememory_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::PyCerememoryEngine>()?;
    m.add_class::<types::PyMemoryRecord>()?;
    m.add_class::<types::PyEncodeStoreResponse>()?;
    m.add_class::<types::PyRecallQueryResponse>()?;
    m.add_class::<types::PyStatsResponse>()?;
    Ok(())
}
