"""Synchronous, reusable agent loop."""
import time
import random
from typing import Callable
from uuid import uuid4

from .context import ContextManager
from .llm import ChatClient
from .tools import ToolRegistry, ToolResult
from .tracing import TraceRecorder
from .errors import (AgentCancelled, AgentLLMError, AgentTimeoutError,
                     classify_llm_exception)


class MaxRoundsExceeded(Exception):
    """The model did not produce a final answer within the round budget."""


class Agent:
    def __init__(self, llm: ChatClient, tools: ToolRegistry, context: ContextManager | None = None,
                 max_rounds: int = 8, logger: Callable[[str], None] | None = None,
                 trace: TraceRecorder | None = None, agent_id: str | None = None,
                 max_llm_attempts: int | None = None, retry_base_delay_s: float = 0.5,
                 retry_max_delay_s: float = 8.0, wait_fn: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 jitter_fn: Callable[[], float] = random.random):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        attempts = max_llm_attempts if max_llm_attempts is not None else getattr(llm, "max_attempts", 3)
        if attempts < 1 or retry_base_delay_s < 0 or retry_max_delay_s < 0:
            raise ValueError("Invalid LLM retry settings")
        self.llm, self.tools = llm, tools
        self.context = context if context is not None else ContextManager()
        self.max_rounds, self.logger = max_rounds, logger
        self.trace = trace if trace is not None else TraceRecorder()
        self.agent_id = agent_id or uuid4().hex
        self.last_run_id: str | None = None
        self.max_llm_attempts = attempts
        self.retry_base_delay_s, self.retry_max_delay_s = retry_base_delay_s, retry_max_delay_s
        self.wait_fn, self.clock, self.jitter_fn = wait_fn, clock, jitter_fn

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def run(self, user_input: str, cancelled: Callable[[], bool] | None = None,
            *, run_id: str | None = None, total_timeout_s: float | None = None) -> str:
        if total_timeout_s is not None and total_timeout_s <= 0:
            raise ValueError("total_timeout_s must be positive")
        run_id = run_id or self.trace.new_run_id()
        self.last_run_id = run_id
        run_start = self.clock()
        deadline = None if total_timeout_s is None else run_start + total_timeout_s
        self.trace.record(run_id, self.agent_id, "agent_start")

        def check_limits() -> None:
            if deadline is not None and self.clock() >= deadline:
                raise AgentTimeoutError("Agent total deadline exceeded")
            if cancelled is not None and cancelled():
                raise AgentCancelled("Agent execution cancelled")

        def wait_for_retry(delay: float) -> None:
            remaining = delay
            while remaining > 0:
                check_limits()
                chunk = min(remaining, 0.05)
                if deadline is not None:
                    chunk = min(chunk, max(0, deadline - self.clock()))
                if chunk <= 0:
                    check_limits()
                self.wait_fn(chunk)
                remaining -= chunk
            check_limits()

        try:
            check_limits()
            self.context.add({"role": "user", "content": user_input})
            self.context.trim()
            for round_number in range(1, self.max_rounds + 1):
                check_limits()
                self._log(f"Round {round_number}: requesting model")
                for attempt in range(1, self.max_llm_attempts + 1):
                    check_limits()
                    llm_start = self.clock()
                    meta = {"round": round_number, "attempt": attempt, "max_attempts": self.max_llm_attempts}
                    self.trace.record(run_id, self.agent_id, "llm_start", metadata=meta)
                    try:
                        response = self.llm.chat(self.context.messages, self.tools.schemas())
                    except Exception as exc:
                        error = classify_llm_exception(exc)
                        error_meta = {**meta, "error_code": error.error_code, "retryable": error.retryable}
                        self.trace.record(run_id, self.agent_id, "llm_end",
                                          duration_ms=(self.clock() - llm_start) * 1000,
                                          success=False, error_type=type(exc).__name__, metadata=error_meta)
                        check_limits()
                        retry_safe = not getattr(self.llm, "streaming", False) and not getattr(exc, "partial_output", False)
                        if not error.retryable or not retry_safe or attempt >= self.max_llm_attempts:
                            if error is exc:
                                raise
                            raise error from exc
                        suggested = getattr(error, "retry_after_s", None)
                        if suggested is not None:
                            delay = min(suggested, self.retry_max_delay_s)
                        else:
                            base = min(self.retry_base_delay_s * 2 ** (attempt - 1), self.retry_max_delay_s)
                            jitter = min(1.0, max(0.0, self.jitter_fn()))
                            delay = min(base * (0.8 + 0.4 * jitter), self.retry_max_delay_s)
                        if deadline is not None:
                            delay = min(delay, max(0, deadline - self.clock()))
                        self.trace.record(run_id, self.agent_id, "llm_retry", success=False,
                                          error_type=type(error).__name__,
                                          metadata={**error_meta, "delay_ms": round(delay * 1000)})
                        wait_for_retry(delay)
                    else:
                        self.trace.record(run_id, self.agent_id, "llm_end",
                                          duration_ms=(self.clock() - llm_start) * 1000,
                                          success=True, metadata=meta)
                        break
                check_limits()
                if response.tool_calls:
                    calls = [{"id": call.id, "type": "function", "function": {"name": call.name,
                              "arguments": call.arguments}} for call in response.tool_calls]
                    self.context.add({"role": "assistant", "content": response.content, "tool_calls": calls})
                    completed = 0
                    try:
                        for call in response.tool_calls:
                            check_limits()
                            self._log(f"Tool {call.name} ({call.id})")
                            tool_start = self.clock()
                            meta = {"round": round_number, "tool_name": call.name, "tool_call_id": call.id}
                            self.trace.record(run_id, self.agent_id, "tool_start", metadata=meta)
                            try:
                                result = self.tools.execute(call.name, call.arguments)
                            except Exception as exc:
                                self.trace.record(run_id, self.agent_id, "tool_end", duration_ms=(self.clock() - tool_start) * 1000,
                                                  success=False, error_type=type(exc).__name__, metadata=meta)
                                raise
                            else:
                                self.trace.record(run_id, self.agent_id, "tool_end", duration_ms=(self.clock() - tool_start) * 1000,
                                                  success=result.ok, error_type=None if result.ok else
                                                  ("ToolArgumentError" if result.error_code == "tool_argument_error" else
                                                   "ToolExecutionError" if result.error_code == "tool_execution_error" else
                                                   "ToolResultError"),
                                                  metadata={**meta, "error_code": result.error_code})
                            message = result.as_message()
                            self.context.add({"role": "tool", "tool_call_id": call.id, "content": message})
                            completed += 1
                            self._log(f"Tool result ({call.id}): {message}")
                    except Exception:
                        # The provider requires one tool reply for every declared call.
                        # Keep history valid while leaving unstarted tools untouched.
                        for pending in response.tool_calls[completed:]:
                            self.context.add({"role": "tool", "tool_call_id": pending.id,
                                              "content": ToolResult(ok=False, error="Tool call not completed").as_message()})
                        self.context.trim()
                        raise
                    self.context.trim()
                    continue
                answer = response.content or ""
                check_limits()
                self.context.add({"role": "assistant", "content": answer})
                self.context.trim()
                self._log(f"Final answer: {answer}")
                self.trace.record(run_id, self.agent_id, "agent_end", duration_ms=(self.clock() - run_start) * 1000,
                                  success=True)
                return answer
            raise MaxRoundsExceeded(f"No final answer after {self.max_rounds} model rounds")
        except Exception as exc:
            self.trace.record(run_id, self.agent_id, "agent_error", duration_ms=(self.clock() - run_start) * 1000,
                              success=False, error_type=type(exc).__name__,
                              metadata={"error_code": getattr(exc, "error_code", None)})
            raise
