"""Cancellation must not leave an incomplete tool-call group in session history."""
from pydantic import BaseModel
import pytest

from xcode_agent import (Agent, AgentCancelled, AgentTimeoutError, LLMRateLimitError,
                         LLMResponse, ToolCall, ToolRegistry, TraceRecorder)


class Args(BaseModel):
    value: int


class TwoCallsLLM:
    def chat(self, messages, tools):
        return LLMResponse(tool_calls=[
            ToolCall(id="first", name="touch", arguments='{"value":1}'),
            ToolCall(id="second", name="touch", arguments='{"value":2}'),
        ])


def test_cancel_between_tool_calls_preserves_complete_group():
    executed = []
    tools = ToolRegistry()
    tools.register("touch", "record value", Args, lambda value: executed.append(value))
    agent = Agent(TwoCallsLLM(), tools)
    with pytest.raises(AgentCancelled):
        agent.run("go", cancelled=lambda: bool(executed))
    assistant = next(m for m in agent.context.messages if m["role"] == "assistant")
    replies = [m for m in agent.context.messages if m["role"] == "tool"]
    assert executed == [1]
    assert [m["tool_call_id"] for m in replies] == [c["id"] for c in assistant["tool_calls"]]


def test_total_timeout_is_observed_after_blocking_model_returns():
    class Clock:
        now = 0.0
        def __call__(self):
            return self.now
    clock = Clock()
    class SlowLLM:
        def chat(self, messages, tools):
            clock.now = 2.0
            return LLMResponse(content="late")
    trace = TraceRecorder()
    agent = Agent(SlowLLM(), ToolRegistry(), trace=trace, clock=clock)
    with pytest.raises(AgentTimeoutError):
        agent.run("go", total_timeout_s=1)
    events = trace.events(agent.last_run_id)
    assert [event.event_type for event in events] == ["agent_start", "llm_start", "llm_end", "agent_error"]
    assert events[2].success is True
    assert events[-1].metadata["error_code"] == "agent_timeout"


def test_llm_retry_after_tool_result_does_not_repeat_tool():
    class SequenceLLM:
        def __init__(self):
            self.calls = 0
        def chat(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(tool_calls=[ToolCall(id="one", name="touch", arguments='{"value":1}')])
            if self.calls == 2:
                raise LLMRateLimitError("mock", retry_after_s=0)
            return LLMResponse(content="done")
    executed = []
    tools = ToolRegistry()
    tools.register("touch", "record value", Args, lambda value: executed.append(value))
    llm = SequenceLLM()
    agent = Agent(llm, tools)
    assert agent.run("go") == "done"
    assert executed == [1]
    assert llm.calls == 3
    assert sum(event.event_type == "llm_retry" for event in agent.trace.events(agent.last_run_id)) == 1
