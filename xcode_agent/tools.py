"""Tool definitions and standardized execution boundary."""
import json
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from jsonschema import FormatChecker, ValidationError as JSONSchemaValidationError, validate as validate_json
from jsonschema.validators import validator_for

from .errors import ToolArgumentError, ToolExecutionError


class ToolResult(BaseModel):
    ok: bool
    content: Any = None
    error: str | None = None
    error_code: str | None = Field(default=None, exclude=True)

    def as_message(self) -> str:
        return self.model_dump_json()


@dataclass(frozen=True)
class ExternalTool:
    description: str
    schema: dict[str, Any]
    handler: Callable[..., ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[str, type[BaseModel], Callable[..., Any]] | ExternalTool] = {}
        self._shared_lock = RLock()

    def register(self, name: str, description: str, arguments: type[BaseModel], handler: Callable[..., Any]) -> None:
        if not name or name in self._tools:
            raise ValueError(f"Invalid or duplicate tool name: {name}")
        self._tools[name] = (description, arguments, handler)

    def register_external(self, name: str, description: str, schema: dict[str, Any],
                          handler: Callable[..., ToolResult]) -> None:
        """Register an external tool while preserving its JSON Schema verbatim."""
        if not name or name in self._tools:
            raise ValueError(f"Invalid or duplicate tool name: {name}")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("External tool schema must describe an object")
        validator_for(schema).check_schema(schema)
        self._tools[name] = ExternalTool(description, deepcopy(schema), handler)

    def get(self, name: str) -> Callable[..., Any] | None:
        entry = self._tools.get(name)
        if isinstance(entry, ExternalTool):
            return entry.handler
        return entry[2] if entry else None

    def is_external(self, name: str) -> bool:
        return isinstance(self._tools.get(name), ExternalTool)

    def clone(self, synchronize_handlers: bool = True) -> "ToolRegistry":
        """Snapshot definitions; optionally serialize calls to shared handler objects."""
        clone = ToolRegistry()
        for name, entry in self._tools.items():
            if isinstance(entry, ExternalTool):
                def external_guarded(*args: Any, _handler: Callable[..., ToolResult] = entry.handler,
                                     **kwargs: Any) -> ToolResult:
                    with self._shared_lock:
                        return _handler(*args, **kwargs)
                clone.register_external(name, entry.description, entry.schema,
                                        external_guarded if synchronize_handlers else entry.handler)
                continue
            description, arguments, handler = entry
            if synchronize_handlers:
                def guarded(*args: Any, _handler: Callable[..., Any] = handler, **kwargs: Any) -> Any:
                    with self._shared_lock:
                        return _handler(*args, **kwargs)
                clone.register(name, description, arguments, guarded)
            else:
                clone.register(name, description, arguments, handler)
        return clone

    def schemas(self) -> list[dict[str, Any]]:
        result = []
        for name, entry in self._tools.items():
            if isinstance(entry, ExternalTool):
                result.append({"type": "function", "function": {"name": name,
                               "description": entry.description, "parameters": deepcopy(entry.schema)}})
            else:
                description, arguments, _ = entry
                result.append({"type": "function", "function": {"name": name,
                               "description": description, "parameters": arguments.model_json_schema()}})
        return result

    def execute(self, name: str, raw_arguments: str) -> ToolResult:
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(ok=False, error=f"Unknown tool: {name}", error_code="tool_not_found")
        if isinstance(entry, ExternalTool):
            try:
                data = json.loads(raw_arguments)
                validate_json(data, entry.schema, format_checker=FormatChecker())
            except (ValueError, JSONSchemaValidationError):
                return ToolResult(ok=False, error="Invalid MCP tool arguments", error_code=ToolArgumentError.error_code)
            try:
                return entry.handler(**data)
            except Exception:
                return ToolResult(ok=False, error="MCP tool execution failed", error_code=ToolExecutionError.error_code)
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
