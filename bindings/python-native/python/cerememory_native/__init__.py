"""Native Python bindings for Cerememory.

Provides direct, in-process access to the CerememoryEngine
without any HTTP overhead. Built with PyO3 + maturin.

Usage::

    from cerememory_native import Engine

    engine = Engine()
    record_id = engine.store("The capital of France is Paris.")
    results = engine.recall("capital of France")
    print(results.memories)
"""

from .cerememory_native import (
    PyCerememoryEngine as Engine,
    PyMemoryRecord as MemoryRecord,
    PyEncodeStoreResponse as EncodeStoreResponse,
    PyRecallQueryResponse as RecallQueryResponse,
    PyStatsResponse as StatsResponse,
)

__all__ = [
    "Engine",
    "MemoryRecord",
    "EncodeStoreResponse",
    "RecallQueryResponse",
    "StatsResponse",
]

__version__ = "0.1.0"
