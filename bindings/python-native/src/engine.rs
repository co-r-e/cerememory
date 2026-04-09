//! Python wrapper around `CerememoryEngine`.
//!
//! Owns a `tokio::runtime::Runtime` to bridge async Rust engine methods
//! to synchronous Python calls via `runtime.block_on()`.

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PyTuple};
use pyo3::{exceptions::PyTypeError, exceptions::PyValueError, Bound};

use cerememory_core::protocol::{
    DreamTickRequest, EncodeStoreRawRequest, EncodeStoreRequest, EncodeStoreResponse,
    ForgetRequest, RecallCue, RecallQueryRequest, RecallRawQueryRequest, RecordIntrospectRequest,
};
use cerememory_core::types::{
    ContentBlock, MemoryContent, Modality, RawSource, RawSpeaker, RawVisibility, SecrecyLevel,
    StoreType,
};
use cerememory_engine::{CerememoryEngine, EngineConfig};
use uuid::Uuid;

use crate::types::{
    json_value_to_py, to_py_err, PyEncodeStoreResponse, PyMemoryRecord, PyRecallQueryResponse,
    PyStatsResponse,
};

/// The main Cerememory engine with native Python bindings.
///
/// Creates an in-memory engine by default. All operations are synchronous
/// from Python's perspective (the async Rust runtime is managed internally).
///
/// Example::
///
///     from cerememory_native import Engine
///
///     engine = Engine()
///     record_id = engine.store("The capital of France is Paris.")
///     results = engine.recall("What is the capital of France?")
///     print(results.memories)
///
#[pyclass(name = "Engine")]
pub struct PyCerememoryEngine {
    runtime: std::sync::Mutex<tokio::runtime::Runtime>,
    engine: CerememoryEngine,
}

impl PyCerememoryEngine {
    fn runtime(&self) -> PyResult<std::sync::MutexGuard<'_, tokio::runtime::Runtime>> {
        self.runtime.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Tokio runtime lock poisoned: {e}"))
        })
    }

    fn py_value_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
        if value.is_none() {
            return Ok(serde_json::Value::Null);
        }
        if let Ok(v) = value.extract::<bool>() {
            return Ok(serde_json::Value::Bool(v));
        }
        if let Ok(v) = value.extract::<i64>() {
            return Ok(serde_json::Value::Number(v.into()));
        }
        if let Ok(v) = value.extract::<u64>() {
            return Ok(serde_json::Value::Number(v.into()));
        }
        if let Ok(v) = value.extract::<f64>() {
            return serde_json::Number::from_f64(v)
                .map(serde_json::Value::Number)
                .ok_or_else(|| PyValueError::new_err("metadata contains a non-finite number"));
        }
        if let Ok(v) = value.extract::<String>() {
            return Ok(serde_json::Value::String(v));
        }
        if let Ok(dict) = value.cast::<PyDict>() {
            let mut map = serde_json::Map::with_capacity(dict.len());
            for (key, item) in dict.iter() {
                let key = key.extract::<String>()?;
                map.insert(key, Self::py_value_to_json(&item)?);
            }
            return Ok(serde_json::Value::Object(map));
        }
        if let Ok(list) = value.cast::<PyList>() {
            let mut items = Vec::with_capacity(list.len());
            for item in list.iter() {
                items.push(Self::py_value_to_json(&item)?);
            }
            return Ok(serde_json::Value::Array(items));
        }
        if let Ok(tuple) = value.cast::<PyTuple>() {
            let mut items = Vec::with_capacity(tuple.len());
            for item in tuple.iter() {
                items.push(Self::py_value_to_json(&item)?);
            }
            return Ok(serde_json::Value::Array(items));
        }

        Err(PyTypeError::new_err(
            "metadata must be JSON-serializable (dict, list, string, number, bool, or null)",
        ))
    }

    /// Shared helper for `store` and `store_full` — builds the request and executes it.
    fn execute_store(
        &self,
        py: Python<'_>,
        text: &str,
        store: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> PyResult<EncodeStoreResponse> {
        let store_type = match store {
            Some(s) => Some(parse_store_type(s)?),
            None => None,
        };

        let req = EncodeStoreRequest {
            header: None,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: text.as_bytes().to_vec(),
                    embedding: None,
                }],
                summary: None,
            },
            store: store_type,
            emotion: None,
            context: None,
            metadata,
            associations: None,
        };

        py.detach(|| {
            let runtime = self.runtime()?;
            runtime
                .block_on(self.engine.encode_store(req))
                .map_err(to_py_err)
        })
    }
}

#[pymethods]
impl PyCerememoryEngine {
    /// Create a new CerememoryEngine.
    ///
    /// Args:
    ///     config: Optional configuration dict with keys:
    ///         - working_capacity (int): Working memory slot count (default: 7)
    ///         - recall_mode (str): "human" or "perfect" (default: "human")
    ///
    /// When no config is provided, an in-memory engine with default settings
    /// is created (suitable for development, testing, and embedded use).
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create tokio runtime: {e}"
                ))
            })?;

        let mut engine_config = EngineConfig::default();

        if let Some(cfg) = config {
            if let Some(capacity) = cfg.get_item("working_capacity")? {
                engine_config.working_capacity = capacity.extract::<usize>()?;
            }
            if let Some(mode) = cfg.get_item("recall_mode")? {
                let mode_str: String = mode.extract()?;
                engine_config.recall_mode = match mode_str.as_str() {
                    "perfect" => cerememory_core::types::RecallMode::Perfect,
                    "human" => cerememory_core::types::RecallMode::Human,
                    other => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid recall_mode: '{other}'. Use 'human' or 'perfect'"
                        )));
                    }
                };
            }
        }

        let engine = CerememoryEngine::new(engine_config).map_err(to_py_err)?;

        Ok(Self {
            runtime: std::sync::Mutex::new(runtime),
            engine,
        })
    }

    /// Store a text memory and return its record UUID as a string.
    ///
    /// Args:
    ///     text: The text content to store.
    ///     store: Optional store name ("episodic", "semantic", "procedural",
    ///            "emotional", "working"). If omitted, the engine auto-routes.
    ///     metadata: Optional JSON-serializable metadata to attach to the record.
    ///
    /// Returns:
    ///     The UUID of the newly created record.
    ///
    /// Raises:
    ///     ValueError: If the store name is invalid or content is too large.
    #[pyo3(signature = (text, store=None, metadata=None))]
    fn store(
        &self,
        py: Python<'_>,
        text: &str,
        store: Option<&str>,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let metadata = metadata.as_ref().map(Self::py_value_to_json).transpose()?;
        let resp = self.execute_store(py, text, store, metadata)?;
        Ok(resp.record_id.to_string())
    }

    /// Store a text memory and return a full EncodeStoreResponse.
    ///
    /// Like `store()` but returns the complete response object with
    /// record_id, store, initial_fidelity, and associations_created.
    ///
    /// Args:
    ///     text: The text content to store.
    ///     store: Optional store name.
    ///
    /// Returns:
    ///     EncodeStoreResponse with full details.
    #[pyo3(signature = (text, store=None, metadata=None))]
    fn store_full(
        &self,
        py: Python<'_>,
        text: &str,
        store: Option<&str>,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyEncodeStoreResponse> {
        let metadata = metadata.as_ref().map(Self::py_value_to_json).transpose()?;
        let resp = self.execute_store(py, text, store, metadata)?;
        Ok(PyEncodeStoreResponse {
            record_id: resp.record_id.to_string(),
            store: resp.store.to_string(),
            initial_fidelity: resp.initial_fidelity,
            associations_created: resp.associations_created,
        })
    }

    /// Recall memories matching a text query.
    ///
    /// Args:
    ///     query: The text query to search for.
    ///     limit: Maximum number of results (default: 10).
    ///     stores: Optional list of store names to search.
    ///     min_fidelity: Optional minimum fidelity threshold.
    ///     recall_mode: "human" (default) or "perfect".
    ///     reconsolidate: Whether to reconsolidate recalled memories.
    ///     activation_depth: Activation depth for spreading activation.
    ///
    /// Returns:
    ///     RecallQueryResponse containing matched memories.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query, limit=10, stores=None, min_fidelity=None, recall_mode=None, reconsolidate=true, activation_depth=2))]
    fn recall(
        &self,
        py: Python<'_>,
        query: &str,
        limit: u32,
        stores: Option<Vec<String>>,
        min_fidelity: Option<f64>,
        recall_mode: Option<&str>,
        reconsolidate: bool,
        activation_depth: u32,
    ) -> PyResult<PyRecallQueryResponse> {
        let store_types = match stores {
            Some(names) => {
                let mut types = Vec::with_capacity(names.len());
                for name in &names {
                    types.push(parse_store_type(name)?);
                }
                Some(types)
            }
            None => None,
        };

        let mode = match recall_mode {
            Some("perfect") => cerememory_core::types::RecallMode::Perfect,
            Some("human") | None => cerememory_core::types::RecallMode::Human,
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid recall_mode: '{other}'. Use 'human' or 'perfect'"
                )));
            }
        };

        let req = RecallQueryRequest {
            header: None,
            cue: RecallCue {
                text: Some(query.to_string()),
                ..Default::default()
            },
            stores: store_types,
            limit,
            min_fidelity,
            include_decayed: false,
            reconsolidate,
            activation_depth,
            recall_mode: mode,
        };

        let resp = py.detach(|| {
            let runtime = self.runtime()?;
            runtime
                .block_on(self.engine.recall_query(req))
                .map_err(to_py_err)
        })?;

        let val = serde_json::to_value(&resp).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to serialize recall response: {e}"
            ))
        })?;

        Ok(PyRecallQueryResponse::from_value(val))
    }

    /// Get a single record by UUID.
    ///
    /// Args:
    ///     id: The record UUID string.
    ///
    /// Returns:
    ///     MemoryRecord for the given ID.
    ///
    /// Raises:
    ///     KeyError: If the record does not exist.
    fn get_record(&self, py: Python<'_>, id: &str) -> PyResult<PyMemoryRecord> {
        let uuid = parse_uuid(id)?;

        let req = RecordIntrospectRequest {
            header: None,
            record_id: uuid,
            include_history: false,
            include_associations: true,
            include_versions: false,
        };

        let record = py.detach(|| {
            let runtime = self.runtime()?;
            runtime
                .block_on(self.engine.introspect_record(req))
                .map_err(to_py_err)
        })?;

        let val = serde_json::to_value(&record).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to serialize record: {e}"))
        })?;

        Ok(PyMemoryRecord::from_value(val))
    }

    /// Permanently delete records by UUID.
    ///
    /// Args:
    ///     ids: List of record UUID strings to delete.
    ///     confirm: Must be True to proceed (safety guard).
    ///
    /// Returns:
    ///     Number of records deleted.
    ///
    /// Raises:
    ///     ValueError: If confirm is False.
    ///     KeyError: If a record ID is not found (partial deletes may occur).
    #[pyo3(signature = (ids, confirm=false))]
    fn forget(&self, py: Python<'_>, ids: Vec<String>, confirm: bool) -> PyResult<u32> {
        let mut uuids = Vec::with_capacity(ids.len());
        for id_str in &ids {
            uuids.push(parse_uuid(id_str)?);
        }

        let req = ForgetRequest {
            header: None,
            record_ids: Some(uuids),
            store: None,
            temporal_range: None,
            cascade: false,
            confirm,
        };

        let deleted = py.detach(|| {
            let runtime = self.runtime()?;
            runtime
                .block_on(self.engine.lifecycle_forget(req))
                .map_err(to_py_err)
        })?;

        Ok(deleted)
    }

    /// Get engine statistics.
    ///
    /// Returns:
    ///     StatsResponse with total records, per-store counts, fidelity, etc.
    fn stats(&self, py: Python<'_>) -> PyResult<PyStatsResponse> {
        let resp = py.detach(|| {
            let runtime = self.runtime()?;
            runtime
                .block_on(self.engine.introspect_stats())
                .map_err(to_py_err)
        })?;

        let val = serde_json::to_value(&resp).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to serialize stats: {e}"))
        })?;

        Ok(PyStatsResponse::from_value(val))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (text, session_id, topic_id=None, source="conversation", speaker="user", visibility="normal", secrecy_level="public"))]
    fn store_raw(
        &self,
        py: Python<'_>,
        text: &str,
        session_id: &str,
        topic_id: Option<String>,
        source: &str,
        speaker: &str,
        visibility: &str,
        secrecy_level: &str,
    ) -> PyResult<String> {
        let req = EncodeStoreRawRequest {
            header: None,
            session_id: session_id.to_string(),
            turn_id: None,
            topic_id,
            source: parse_raw_source(source)?,
            speaker: parse_raw_speaker(speaker)?,
            visibility: parse_raw_visibility(visibility)?,
            secrecy_level: parse_secrecy_level(secrecy_level)?,
            content: MemoryContent {
                blocks: vec![ContentBlock {
                    modality: Modality::Text,
                    format: "text/plain".to_string(),
                    data: text.as_bytes().to_vec(),
                    embedding: None,
                }],
                summary: None,
            },
            metadata: None,
        };

        py.detach(|| {
            let runtime = self.runtime()?;
            let resp = runtime
                .block_on(self.engine.encode_store_raw(req))
                .map_err(to_py_err)?;
            Ok(resp.record_id.to_string())
        })
    }

    #[pyo3(signature = (query=None, session_id=None, limit=10, include_private_scratch=false, include_sealed=false))]
    fn recall_raw(
        &self,
        py: Python<'_>,
        query: Option<String>,
        session_id: Option<String>,
        limit: u32,
        include_private_scratch: bool,
        include_sealed: bool,
    ) -> PyResult<Py<PyAny>> {
        let req = RecallRawQueryRequest {
            header: None,
            session_id,
            query,
            temporal: None,
            limit,
            include_private_scratch,
            include_sealed,
            secrecy_levels: None,
        };

        let resp = py.detach(|| {
            let runtime = self.runtime()?;
            runtime
                .block_on(self.engine.recall_raw_query(req))
                .map_err(to_py_err)
        })?;
        let val = serde_json::to_value(&resp).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to serialize raw recall response: {e}"
            ))
        })?;
        json_value_to_py(py, &val)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (session_id=None, dry_run=false, max_groups=10, include_private_scratch=false, include_sealed=false, promote_semantic=true))]
    fn dream_tick(
        &self,
        py: Python<'_>,
        session_id: Option<String>,
        dry_run: bool,
        max_groups: u32,
        include_private_scratch: bool,
        include_sealed: bool,
        promote_semantic: bool,
    ) -> PyResult<Py<PyAny>> {
        let req = DreamTickRequest {
            header: None,
            session_id,
            dry_run,
            max_groups,
            include_private_scratch,
            include_sealed,
            promote_semantic,
            secrecy_levels: None,
        };
        let resp = py.detach(|| {
            let runtime = self.runtime()?;
            runtime
                .block_on(self.engine.lifecycle_dream_tick(req))
                .map_err(to_py_err)
        })?;
        let val = serde_json::to_value(&resp).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to serialize dream tick response: {e}"
            ))
        })?;
        json_value_to_py(py, &val)
    }

    fn __repr__(&self) -> String {
        "Engine(in_memory=True)".to_string()
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────

/// Parse a store type string into a `StoreType` enum.
fn parse_store_type(s: &str) -> PyResult<StoreType> {
    s.to_lowercase()
        .parse::<StoreType>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
}

fn parse_raw_source(s: &str) -> PyResult<RawSource> {
    match s.trim().to_lowercase().as_str() {
        "conversation" => Ok(RawSource::Conversation),
        "tool_io" => Ok(RawSource::ToolIo),
        "scratchpad" => Ok(RawSource::Scratchpad),
        "summary" => Ok(RawSource::Summary),
        "imported" => Ok(RawSource::Imported),
        other => Err(PyValueError::new_err(format!(
            "Invalid raw source: {other}"
        ))),
    }
}

fn parse_raw_speaker(s: &str) -> PyResult<RawSpeaker> {
    match s.trim().to_lowercase().as_str() {
        "user" => Ok(RawSpeaker::User),
        "assistant" => Ok(RawSpeaker::Assistant),
        "system" => Ok(RawSpeaker::System),
        "tool" => Ok(RawSpeaker::Tool),
        other => Err(PyValueError::new_err(format!(
            "Invalid raw speaker: {other}"
        ))),
    }
}

fn parse_raw_visibility(s: &str) -> PyResult<RawVisibility> {
    match s.trim().to_lowercase().as_str() {
        "normal" => Ok(RawVisibility::Normal),
        "private_scratch" => Ok(RawVisibility::PrivateScratch),
        "sealed" => Ok(RawVisibility::Sealed),
        other => Err(PyValueError::new_err(format!(
            "Invalid raw visibility: {other}"
        ))),
    }
}

fn parse_secrecy_level(s: &str) -> PyResult<SecrecyLevel> {
    match s.trim().to_lowercase().as_str() {
        "public" => Ok(SecrecyLevel::Public),
        "sensitive" => Ok(SecrecyLevel::Sensitive),
        "secret" => Ok(SecrecyLevel::Secret),
        other => Err(PyValueError::new_err(format!(
            "Invalid secrecy level: {other}"
        ))),
    }
}

/// Parse a UUID string.
fn parse_uuid(s: &str) -> PyResult<Uuid> {
    s.parse::<Uuid>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid UUID '{s}': {e}")))
}
