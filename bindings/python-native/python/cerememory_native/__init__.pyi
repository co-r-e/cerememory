"""Type stubs for cerememory_native."""

from typing import Any, Optional

class MemoryRecord:
    """A memory record returned from the engine."""

    @property
    def id(self) -> Optional[str]: ...
    @property
    def store(self) -> Optional[str]: ...
    @property
    def created_at(self) -> Optional[str]: ...
    @property
    def updated_at(self) -> Optional[str]: ...
    @property
    def last_accessed_at(self) -> Optional[str]: ...
    @property
    def access_count(self) -> int: ...
    @property
    def fidelity(self) -> Optional[dict[str, Any]]: ...
    @property
    def emotion(self) -> Optional[dict[str, Any]]: ...
    @property
    def content(self) -> Optional[dict[str, Any]]: ...
    @property
    def associations(self) -> Optional[list[dict[str, Any]]]: ...
    @property
    def metadata(self) -> Optional[dict[str, Any]]: ...
    @property
    def version(self) -> int: ...
    def text_content(self) -> Optional[str]: ...
    def to_dict(self) -> dict[str, Any]: ...

class EncodeStoreResponse:
    """Response from an encode/store operation."""

    @property
    def record_id(self) -> str: ...
    @property
    def store(self) -> str: ...
    @property
    def initial_fidelity(self) -> float: ...
    @property
    def associations_created(self) -> int: ...
    def to_dict(self) -> dict[str, Any]: ...

class RecallQueryResponse:
    """Response from a recall/query operation."""

    @property
    def memories(self) -> Optional[list[dict[str, Any]]]: ...
    @property
    def query_metadata(self) -> dict[str, Any] | None: ...
    @property
    def total_candidates(self) -> int: ...
    def to_dict(self) -> dict[str, Any]: ...

class StatsResponse:
    """Response from an introspect/stats operation."""

    @property
    def total_records(self) -> int: ...
    @property
    def records_by_store(self) -> dict[str, int]: ...
    @property
    def total_associations(self) -> int: ...
    @property
    def avg_fidelity(self) -> float: ...
    @property
    def avg_fidelity_by_store(self) -> dict[str, float]: ...
    @property
    def background_decay_enabled(self) -> bool: ...
    @property
    def raw_journal_records(self) -> int: ...
    @property
    def raw_journal_pending_dream(self) -> int: ...
    @property
    def dream_episodic_summaries(self) -> int: ...
    @property
    def dream_semantic_nodes(self) -> int: ...
    @property
    def background_dream_enabled(self) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...

class Engine:
    """Native Cerememory engine.

    Creates an in-memory engine by default. All operations are synchronous
    from Python's perspective.

    Args:
        config: Optional configuration dict with keys:
            - working_capacity (int): Working memory slot count (default: 7)
            - recall_mode (str): "human" or "perfect" (default: "human")
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None: ...
    def store(
        self,
        text: str,
        store: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a text memory. Returns the record UUID string."""
        ...
    def store_full(
        self,
        text: str,
        store: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EncodeStoreResponse:
        """Store a text memory. Returns full EncodeStoreResponse."""
        ...
    def recall(
        self,
        query: str,
        limit: int = 10,
        stores: Optional[list[str]] = None,
        min_fidelity: Optional[float] = None,
        recall_mode: Optional[str] = None,
        reconsolidate: bool = True,
        activation_depth: int = 2,
    ) -> RecallQueryResponse:
        """Recall memories matching a text query."""
        ...
    def get_record(self, id: str) -> MemoryRecord:
        """Get a single record by UUID."""
        ...
    def forget(self, ids: list[str], confirm: bool = False) -> int:
        """Delete records by UUID. Returns count deleted."""
        ...
    def stats(self) -> StatsResponse:
        """Get engine statistics."""
        ...
    def store_raw(
        self,
        text: str,
        session_id: str,
        topic_id: Optional[str] = None,
        source: str = "conversation",
        speaker: str = "user",
        visibility: str = "normal",
        secrecy_level: str = "public",
    ) -> str:
        """Store a raw journal record. Returns the raw record UUID string."""
        ...
    def recall_raw(
        self,
        query: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        include_private_scratch: bool = False,
        include_sealed: bool = False,
    ) -> dict[str, Any]:
        """Recall raw journal records. Returns a JSON-like dict."""
        ...
    def dream_tick(
        self,
        session_id: Optional[str] = None,
        dry_run: bool = False,
        max_groups: int = 10,
        include_private_scratch: bool = False,
        include_sealed: bool = False,
        promote_semantic: bool = True,
    ) -> dict[str, Any]:
        """Run a dream tick. Returns a JSON-like dict."""
        ...
