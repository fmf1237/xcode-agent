"""Synchronous, reusable agent loop."""
import time
from typing import Callable
from uuid import uuid4

from .context import ContextManager
from .llm import ChatClient
from .tools import ToolRegistry
from .tracing import TraceRecorder


class AgentLLMError(Exception):
    """The model request failed; no tool result can repair it."""


class MaxRoundsExceeded(Exception):
    """The model did not produce a final answer within the round budget."""


class AgentCancelled(Exception):
    """Cooperative cancellation observed between blocking operations."""


class Agent:
    def __init__(self, llm: ChatClient, tools: ToolRegistry, context: ContextManager | None = None,
                 max_rounds: int = 8, logger: Callable[[str], None] | None = None,
                 trace: TraceRecorder | None = None, agent_id: str | None = None):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.llm, self.tools = llm, tools
        self.context = context if context is not None else ContextManager()
        self.max_rounds, self.logger = max_rounds, logger
        self.trace = trace if trace is not None else TraceRecorder()
        self.agent_id = agent_id or uuid4().hex
        self.last_run_id: str | None = None

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def run(self, user_input: str, cancelled: Callable[[], bool] | None = None,
            *, run_id: str | None = None) -> str:
        run_id = run_id or self.trace.new_run_id()
        self.last_run_id = run_id
        run_start = time.monotonic()
        self.trace.record(run_id, self.agent_id, "agent_start")

        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise AgentCancelled("Agent execution cancelled")
        try:
            check_cancelled()
            self.context.add({"role": "user", "content": user_input})
            self.context.trim()
            for round_number in range(1, self.max_rounds + 1):
                check_cancelled()
                self._log(f"Round {round_number}: requesting model")
                llm_start = time.monotonic()
                self.trace.record(run_id, self.agent_id, "llm_start", metadata={"round": round_number})
                try:
                    response = self.llm.chat(self.context.messages, self.tools.schemas())
                except Exception as exc:
                    self.trace.record(run_id, self.agent_id, "llm_end", duration_ms=(time.monotonic() - llm_start) * 1000,
                                      success=False, error_type=type(exc).__name__, metadata={"round": round_number})
                    raise AgentLLMError(str(exc)) from exc
                else:
                    self.trace.record(run_id, self.agent_id, "llm_end", duration_ms=(time.monotonic() - llm_start) * 1000,
                                      success=True, metadata={"round": round_number})
                check_cancelled()
                if response.tool_calls:
                    calls = [{"id": call.id, "type": "function", "function": {"name": call.name,
                              "arguments": call.arguments}} for call in response.tool_calls]
                    self.context.add({"role": "assistant", "content": response.content, "tool_calls": calls})
                    for call in response.tool_calls:
                        check_cancelled()
                        self._log(f"Tool {call.name} ({call.id})")
                        tool_start = time.monotonic()
                        meta = {"round": round_number, "tool_name": call.name, "tool_call_id": call.id}
                        self.trace.record(run_id, self.agent_id, "tool_start", metadata=meta)
                        try:
                            result = self.tools.execute(call.name, call.arguments)
                        except Exception as exc:
                            self.trace.record(run_id, self.agent_id, "tool_end", duration_ms=(time.monotonic() - tool_start) * 1000,
                                              success=False, error_type=type(exc).__name__, metadata=meta)
                            raise
                        else:
                            self.trace.record(run_id, self.agent_id, "tool_end", duration_ms=(time.monotonic() - tool_start) * 1000,
                                              success=result.ok, error_type=None if result.ok else "ToolResultError", metadata=meta)
                        self.context.add({"role": "tool", "tool_call_id": call.id, "content": result.as_message()})
                        self._log(f"Tool result ({call.id}): {result.as_message()}")
                    self.context.trim()
                    continue
                answer = response.content or ""
                check_cancelled()
                self.context.add({"role": "assistant", "content": answer})
                self.context.trim()
                self._log(f"Final answer: {answer}")
                self.trace.record(run_id, self.agent_id, "agent_end", duration_ms=(time.monotonic() - run_start) * 1000,
                                  success=True)
                return answer
            raise MaxRoundsExceeded(f"No final answer after {self.max_rounds} model rounds")
        except Exception as exc:
            self.trace.record(run_id, self.agent_id, "agent_error", duration_ms=(time.monotonic() - run_start) * 1000,
                              success=False, error_type=type(exc).__name__)
            raise
