"""Tool definitions and standardized execution boundary."""
import json
from threading import RLock
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .errors import ToolArgumentError, ToolExecutionError


class ToolResult(BaseModel):
    ok: bool
    content: Any = None
    error: str | None = None
    error_code: str | None = Field(default=None, exclude=True)

    def as_message(self) -> str:
        return self.model_dump_json()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[str, type[BaseModel], Callable[..., Any]]] = {}
        self._shared_lock = RLock()

    def register(self, name: str, description: str, arguments: type[BaseModel], handler: Callable[..., Any]) -> None:
        if not name or name in self._tools:
            raise ValueError(f"Invalid or duplicate tool name: {name}")
        self._tools[name] = (description, arguments, handler)

    def get(self, name: str) -> Callable[..., Any] | None:
        entry = self._tools.get(name)
        return entry[2] if entry else None

    def clone(self, synchronize_handlers: bool = True) -> "ToolRegistry":
        """Snapshot definitions; optionally serialize calls to shared handler objects."""
        clone = ToolRegistry()
        for name, (description, arguments, handler) in self._tools.items():
            if synchronize_handlers:
                def guarded(*args: Any, _handler: Callable[..., Any] = handler, **kwargs: Any) -> Any:
                    with self._shared_lock:
                        return _handler(*args, **kwargs)
                clone.register(name, description, arguments, guarded)
            else:
                clone.register(name, description, arguments, handler)
        return clone

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": name, "description": description,
             "parameters": arguments.model_json_schema()}}
            for name, (description, arguments, _) in self._tools.items()
        ]

    def execute(self, name: str, raw_arguments: str) -> ToolResult:
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(ok=False, error=f"Unknown tool: {name}", error_code="tool_not_found")
        _, arguments, handler = entry
        try:
            data = json.loads(raw_arguments)
            if not isinstance(data, dict):
                raise ValueError("Arguments must be a JSON object")
            parsed = arguments.model_validate(data)
        except (ValueError, ValidationError) as exc:
            return ToolResult(ok=False, error=f"Invalid arguments: {exc}",
                              error_code=ToolArgumentError.error_code)
        try:
            return ToolResult(ok=True, content=handler(**parsed.model_dump()))
        except Exception as exc:
            return ToolResult(ok=False, error=f"Tool execution failed ({type(exc).__name__}): {exc}",
                              error_code=ToolExecutionError.error_code)
