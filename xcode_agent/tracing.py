"""Thread-safe, in-memory execution events with deliberately small metadata."""
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    agent_id: str
    event_type: str
    timestamp: datetime
    duration_ms: float | None = Field(default=None, ge=0)
    success: bool | None = None
    error_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceRecorder:
    """Store events by run ID; return copies so callers cannot alter stored history."""

    _METADATA_KEYS = frozenset({"task_id", "parent_agent_id", "parent_task_id", "role",
                                "round", "tool_name", "tool_call_id", "worker_running",
                                "attempt", "max_attempts", "delay_ms", "error_code", "retryable"})

    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, list[TraceEvent]] = {}

    @staticmethod
    def new_run_id() -> str:
        return uuid4().hex

    def record(self, run_id: str, agent_id: str, event_type: str, *,
               duration_ms: float | None = None, success: bool | None = None,
               error_type: str | None = None, metadata: dict[str, Any] | None = None) -> TraceEvent:
        # Only known identifiers and scalar values cross this boundary. In particular,
        # prompts, raw tool arguments, outputs and exception messages are never stored.
        safe = {key: value for key, value in (metadata or {}).items()
                if key in self._METADATA_KEYS and isinstance(value, (str, int, float, bool, type(None)))}
        event = TraceEvent(run_id=run_id, agent_id=agent_id, event_type=event_type,
                           timestamp=datetime.now(timezone.utc), duration_ms=duration_ms,
                           success=success, error_type=error_type, metadata=safe)
        with self._lock:
            self._runs.setdefault(run_id, []).append(event)
        return event.model_copy(deep=True)

    def events(self, run_id: str) -> list[TraceEvent]:
        with self._lock:
            return [event.model_copy(deep=True) for event in self._runs.get(run_id, [])]
