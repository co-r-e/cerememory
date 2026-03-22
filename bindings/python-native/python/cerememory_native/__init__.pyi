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
    def fidelity(self) -> dict[str, Any]: ...
    @property
    def emotion(self) -> dict[str, Any]: ...
    @property
    def content(self) -> dict[str, Any]: ...
    @property
    def associations(self) -> list[dict[str, Any]]: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
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
    def memories(self) -> list[dict[str, Any]]: ...
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
    def forget(self, ids: list[str], confirm: bool = True) -> int:
        """Delete records by UUID. Returns count deleted."""
        ...
    def stats(self) -> StatsResponse:
        """Get engine statistics."""
        ...
