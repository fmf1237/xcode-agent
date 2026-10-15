"""Synchronous facade over one persistent stdio MCP ClientSession."""
import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
from copy import deepcopy
import os
import re
from threading import Lock, Thread
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field

from .tools import ToolRegistry, ToolResult


class MCPConnectionError(RuntimeError):
    error_code = "mcp_connection_error"
    retryable = False


class MCPServerConfig(BaseModel):
    """Trusted application configuration; never populated from model output."""
    model_config = ConfigDict(frozen=True)

    server_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    command: str = Field(min_length=1)
    args: tuple[str, ...] = ()
    allowed_tools: frozenset[str] = frozenset()
    max_tools: int = Field(default=16, ge=1, le=64)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    call_timeout_s: float = Field(default=10.0, gt=0)


class MCPToolInfo(BaseModel):
    server_name: str
    tool_name: str
    qualified_name: str
    description: str
    input_schema: dict[str, Any]


class MCPClient:
    """Own one event-loop thread, MCP session and stdio subprocess until close()."""
    _TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._main_task: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._ready: Future[list[MCPToolInfo]] = Future()
        self._tools: dict[str, MCPToolInfo] = {}
        self._closed = False
        self._lifecycle_lock = Lock()
        self._call_lock = Lock()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._main_task = asyncio.current_task()
        self._stop = asyncio.Event()
        params = StdioServerParameters(command=self.config.command, args=list(self.config.args))
        server_stderr = open(os.devnull, "w", encoding="utf-8")
        try:
            async with stdio_client(params, errlog=server_stderr) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), self.config.connect_timeout_s)
                    listing = await asyncio.wait_for(session.list_tools(), self.config.connect_timeout_s)
                    infos: list[MCPToolInfo] = []
                    for tool in listing.tools:
                        if tool.name not in self.config.allowed_tools:
                            continue
                        if not self._TOOL_NAME.fullmatch(tool.name):
                            raise MCPConnectionError("MCP tool name cannot be exposed safely")
                        qualified = f"mcp__{self.config.server_name}__{tool.name}"
                        if len(qualified) > 64:
                            raise MCPConnectionError("MCP qualified tool name is too long")
                        infos.append(MCPToolInfo(server_name=self.config.server_name,
                                                 tool_name=tool.name, qualified_name=qualified,
                                                 description=tool.description or "MCP tool",
                                                 input_schema=deepcopy(tool.inputSchema)))
                    if len(infos) > self.config.max_tools:
                        raise MCPConnectionError("MCP tool exposure limit exceeded")
                    missing = self.config.allowed_tools - {info.tool_name for info in infos}
                    if missing:
                        raise MCPConnectionError("Configured MCP tool is unavailable")
                    self._session = session
                    self._tools = {info.tool_name: info for info in infos}
                    self._ready.set_result(infos)
                    await self._stop.wait()
        except BaseException:
            if not self._ready.done():
                self._ready.set_exception(MCPConnectionError("Could not connect to trusted MCP server"))
        finally:
            server_stderr.close()
            self._session = None

    def connect(self) -> list[MCPToolInfo]:
        with self._lifecycle_lock:
            if self._closed:
                raise MCPConnectionError("MCP client is closed")
            if self._thread is None:
                self._thread = Thread(target=lambda: asyncio.run(self._serve()),
                                      name=f"mcp-{self.config.server_name}", daemon=True)
                self._thread.start()
        try:
            infos = self._ready.result(timeout=self.config.connect_timeout_s + 1)
            if self._session is None or self._thread is None or not self._thread.is_alive():
                raise MCPConnectionError("MCP connection is closed")
            return [info.model_copy(deep=True) for info in infos]
        except (FutureTimeout, MCPConnectionError) as exc:
            self.close()
            raise MCPConnectionError("Could not connect to trusted MCP server") from exc

    def list_tools(self) -> list[MCPToolInfo]:
        return self.connect()

    def register_tools(self, registry: ToolRegistry) -> list[str]:
        """Expose only explicitly allowed names with original JSON Schema."""
        names = []
        for info in self.connect():
            def make_handler(raw_name: str):
                def handler(**kwargs: Any) -> ToolResult:
                    return self.call_tool(raw_name, kwargs)
                return handler
            registry.register_external(info.qualified_name, info.description,
                                       info.input_schema, make_handler(info.tool_name))
            names.append(info.qualified_name)
        return names

    @staticmethod
    def _content(result: Any) -> Any:
        if result.structuredContent is not None:
            structured = result.structuredContent
            return structured["result"] if isinstance(structured, dict) and set(structured) == {"result"} else structured
        blocks = result.content
        if len(blocks) == 1 and getattr(blocks[0], "type", None) == "text":
            return blocks[0].text
        return [block.model_dump(mode="json", exclude_none=True) for block in blocks]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(ok=False, error="MCP tool is not exposed", error_code="mcp_tool_not_found")
        with self._call_lock:
            if self._closed or self._session is None or self._loop is None:
                return ToolResult(ok=False, error="MCP connection is closed", error_code="mcp_connection_error")

            async def invoke():
                return await asyncio.wait_for(self._session.call_tool(tool_name, arguments=arguments),
                                              timeout=self.config.call_timeout_s)

            future = asyncio.run_coroutine_threadsafe(invoke(), self._loop)
            try:
                result = future.result(timeout=self.config.call_timeout_s + 1)
            except (TimeoutError, FutureTimeout):
                future.cancel()
                return ToolResult(ok=False, error="MCP tool call timed out", error_code="mcp_timeout")
            except Exception:
                return ToolResult(ok=False, error="MCP tool call failed", error_code="mcp_call_error")
            if result.isError:
                return ToolResult(ok=False, error="MCP server reported tool error", error_code="mcp_tool_error")
            return ToolResult(ok=True, content=self._content(result))

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        with self._call_lock:
            pass  # Wait for an active call to return before stopping its session.
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive() and self._loop is not None and self._main_task is not None:
                self._loop.call_soon_threadsafe(self._main_task.cancel)
                self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise MCPConnectionError("MCP server cleanup did not finish")

    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
