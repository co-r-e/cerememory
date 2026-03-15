//! Python wrapper types for Cerememory data structures.
//!
//! Each wrapper holds a `serde_json::Value` internally and exposes typed
//! getter properties to Python. This approach avoids duplicating every field
//! as a PyO3 struct member and keeps the bridge layer thin.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use cerememory_core::error::CerememoryError;

// ─── Error mapping ─────────────────────────────────────────────────────

/// Convert a `CerememoryError` into an appropriate Python exception.
pub fn to_py_err(err: CerememoryError) -> PyErr {
    match err {
        CerememoryError::RecordNotFound(msg) => PyKeyError::new_err(msg),
        CerememoryError::Validation(msg) => PyValueError::new_err(msg),
        CerememoryError::StoreInvalid(msg) => PyValueError::new_err(msg),
        CerememoryError::ContentTooLarge { size, limit } => PyValueError::new_err(format!(
            "Content too large: {size} bytes exceeds limit of {limit} bytes"
        )),
        CerememoryError::ModalityUnsupported(msg) => PyTypeError::new_err(msg),
        CerememoryError::ForgetUnconfirmed => {
            PyValueError::new_err("Forget operation requires confirmation")
        }
        CerememoryError::WorkingMemoryFull => PyRuntimeError::new_err("Working memory at capacity"),
        CerememoryError::DecayEngineBusy { retry_after_secs } => PyRuntimeError::new_err(format!(
            "Decay engine busy, retry after {retry_after_secs}s"
        )),
        CerememoryError::ConsolidationInProgress => {
            PyRuntimeError::new_err("Consolidation in progress")
        }
        CerememoryError::Unauthorized(msg) => {
            PyRuntimeError::new_err(format!("Unauthorized: {msg}"))
        }
        CerememoryError::RateLimited { retry_after_secs } => {
            PyRuntimeError::new_err(format!("Rate limited, retry after {retry_after_secs}s"))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

// ─── serde_json::Value → PyObject helpers ──────────────────────────────

/// Convert a `serde_json::Value` into a Python object.
///
/// - null   → None
/// - bool   → bool
/// - number → int | float
/// - string → str
/// - array  → list
/// - object → dict
pub fn json_value_to_py(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(u) = n.as_u64() {
                Ok(u.to_object(py))
            } else {
                Ok(n.as_f64().unwrap_or(0.0).to_object(py))
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let items = arr
                .iter()
                .map(|v| json_value_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(items.to_object(py))
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_value_to_py(py, v)?)?;
            }
            Ok(dict.unbind().into_any())
        }
    }
}

// ─── PyMemoryRecord ────────────────────────────────────────────────────

/// A memory record returned from the engine.
///
/// Wraps the serialized JSON representation and exposes typed properties.
#[pyclass(name = "MemoryRecord")]
#[derive(Clone)]
pub struct PyMemoryRecord {
    inner: serde_json::Value,
}

impl PyMemoryRecord {
    pub fn from_value(val: serde_json::Value) -> Self {
        Self { inner: val }
    }
}

#[pymethods]
impl PyMemoryRecord {
    /// The record UUID as a string.
    #[getter]
    fn id(&self) -> Option<String> {
        self.inner
            .get("id")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// The store type (e.g. "episodic", "semantic").
    #[getter]
    fn store(&self) -> Option<String> {
        self.inner
            .get("store")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// ISO 8601 creation timestamp.
    #[getter]
    fn created_at(&self) -> Option<String> {
        self.inner
            .get("created_at")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// ISO 8601 last-updated timestamp.
    #[getter]
    fn updated_at(&self) -> Option<String> {
        self.inner
            .get("updated_at")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// ISO 8601 last-accessed timestamp.
    #[getter]
    fn last_accessed_at(&self) -> Option<String> {
        self.inner
            .get("last_accessed_at")
            .and_then(|v| v.as_str())
            .map(String::from)
    }

    /// Number of times this record has been accessed.
    #[getter]
    fn access_count(&self) -> u32 {
        self.inner
            .get("access_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
    }

    /// Fidelity state as a Python dict.
    #[getter]
    fn fidelity(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .get("fidelity")
            .map(|v| json_value_to_py(py, v))
            .transpose()?
            .unwrap_or_else(|| py.None()))
    }

    /// Emotion vector as a Python dict.
    #[getter]
    fn emotion(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .get("emotion")
            .map(|v| json_value_to_py(py, v))
            .transpose()?
            .unwrap_or_else(|| py.None()))
    }

    /// Content object as a Python dict.
    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .get("content")
            .map(|v| json_value_to_py(py, v))
            .transpose()?
            .unwrap_or_else(|| py.None()))
    }

    /// Associations list as Python list of dicts.
    #[getter]
    fn associations(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .get("associations")
            .map(|v| json_value_to_py(py, v))
            .transpose()?
            .unwrap_or_else(|| py.None()))
    }

    /// Metadata dict.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .get("metadata")
            .map(|v| json_value_to_py(py, v))
            .transpose()?
            .unwrap_or_else(|| py.None()))
    }

    /// Record version number.
    #[getter]
    fn version(&self) -> u32 {
        self.inner
            .get("version")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32
    }

    /// Extract the text content from the first text block, if any.
    fn text_content(&self) -> Option<String> {
        let blocks = self.inner.get("content")?.get("blocks")?.as_array()?;
        for block in blocks {
            if block.get("modality").and_then(|m| m.as_str()) == Some("text") {
                // data is stored as a byte array in JSON; try base64 or UTF-8
                if let Some(data) = block.get("data").and_then(|d| d.as_array()) {
                    let bytes: Vec<u8> = data
                        .iter()
                        .filter_map(|b| b.as_u64().map(|v| v as u8))
                        .collect();
                    return String::from_utf8(bytes).ok();
                }
            }
        }
        None
    }

    /// Return the full record as a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        let id = self.id().unwrap_or_else(|| "?".to_string());
        let store = self.store().unwrap_or_else(|| "?".to_string());
        format!("MemoryRecord(id={id}, store={store})")
    }
}

// ─── PyEncodeStoreResponse ─────────────────────────────────────────────

/// Response from an encode/store operation.
#[pyclass(name = "EncodeStoreResponse")]
#[derive(Clone)]
pub struct PyEncodeStoreResponse {
    pub record_id: String,
    pub store: String,
    pub initial_fidelity: f64,
    pub associations_created: u32,
}

#[pymethods]
impl PyEncodeStoreResponse {
    /// The UUID of the newly stored record.
    #[getter]
    fn record_id(&self) -> &str {
        &self.record_id
    }

    /// The store the record was placed in.
    #[getter]
    fn store(&self) -> &str {
        &self.store
    }

    /// The initial fidelity score (0.0 to 1.0).
    #[getter]
    fn initial_fidelity(&self) -> f64 {
        self.initial_fidelity
    }

    /// Number of associations created during encoding.
    #[getter]
    fn associations_created(&self) -> u32 {
        self.associations_created
    }

    /// Return as a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("record_id", &self.record_id)?;
        dict.set_item("store", &self.store)?;
        dict.set_item("initial_fidelity", self.initial_fidelity)?;
        dict.set_item("associations_created", self.associations_created)?;
        Ok(dict.unbind().into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "EncodeStoreResponse(record_id={}, store={}, fidelity={})",
            self.record_id, self.store, self.initial_fidelity
        )
    }
}

// ─── PyRecallQueryResponse ─────────────────────────────────────────────

/// Response from a recall/query operation.
#[pyclass(name = "RecallQueryResponse")]
#[derive(Clone)]
pub struct PyRecallQueryResponse {
    inner: serde_json::Value,
}

impl PyRecallQueryResponse {
    pub fn from_value(val: serde_json::Value) -> Self {
        Self { inner: val }
    }
}

#[pymethods]
impl PyRecallQueryResponse {
    /// The recalled memories as a list of dicts.
    #[getter]
    fn memories(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .get("memories")
            .map(|v| json_value_to_py(py, v))
            .transpose()?
            .unwrap_or_else(|| py.None()))
    }

    /// Total number of candidate records evaluated.
    #[getter]
    fn total_candidates(&self) -> u32 {
        self.inner
            .get("total_candidates")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
    }

    /// Return the full response as a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        let count = self
            .inner
            .get("memories")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let total = self.total_candidates();
        format!("RecallQueryResponse(returned={count}, total_candidates={total})")
    }
}

// ─── PyStatsResponse ───────────────────────────────────────────────────

/// Response from an introspect/stats operation.
#[pyclass(name = "StatsResponse")]
#[derive(Clone)]
pub struct PyStatsResponse {
    inner: serde_json::Value,
}

impl PyStatsResponse {
    pub fn from_value(val: serde_json::Value) -> Self {
        Self { inner: val }
    }
}

#[pymethods]
impl PyStatsResponse {
    /// Total number of records across all stores.
    #[getter]
    fn total_records(&self) -> u32 {
        self.inner
            .get("total_records")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
    }

    /// Record counts per store as a Python dict.
    #[getter]
    fn records_by_store(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .get("records_by_store")
            .map(|v| json_value_to_py(py, v))
            .transpose()?
            .unwrap_or_else(|| py.None()))
    }

    /// Total number of associations.
    #[getter]
    fn total_associations(&self) -> u32 {
        self.inner
            .get("total_associations")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
    }

    /// Average fidelity score across all records.
    #[getter]
    fn avg_fidelity(&self) -> f64 {
        self.inner
            .get("avg_fidelity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    }

    /// Average fidelity per store as a Python dict.
    #[getter]
    fn avg_fidelity_by_store(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self
            .inner
            .get("avg_fidelity_by_store")
            .map(|v| json_value_to_py(py, v))
            .transpose()?
            .unwrap_or_else(|| py.None()))
    }

    /// Whether background decay is enabled.
    #[getter]
    fn background_decay_enabled(&self) -> bool {
        self.inner
            .get("background_decay_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Return the full response as a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        let total = self.total_records();
        let fidelity = self.avg_fidelity();
        format!("StatsResponse(total_records={total}, avg_fidelity={fidelity:.3})")
    }
}
