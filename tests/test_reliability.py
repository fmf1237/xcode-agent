"""Offline reliability tests: no API key, network or real backoff sleeps."""
import json
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest
from openai import APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ConfigDict

from xcode_agent import (Agent, AgentCancelled, AgentFactory, AgentLLMError,
                         AgentTimeoutError, DependencyFailedError, LLMRateLimitError,
                         LLMResponse, LLMTimeoutError, TaskNode, ToolCall, ToolRegistry,
                         TraceRecorder)
from xcode_agent.errors import classify_llm_exception, parse_retry_after
from xcode_agent.llm import LLMClient


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def rate_limit(retry_after="2"):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(429, headers={"retry-after": retry_after}, request=request)
    return RateLimitError("private API detail", response=response, body=None)


def timeout_error():
    return APITimeoutError(request=httpx.Request("POST", "https://example.invalid/v1/chat/completions"))


def build(llm, clock=None, **kwargs):
    clock = clock or FakeClock()
    trace = TraceRecorder()
    agent = Agent(llm, ToolRegistry(), trace=trace, clock=clock, wait_fn=clock.wait,
                  jitter_fn=lambda: 0.5, **kwargs)
    return agent, clock, trace


def test_first_attempt_success_and_existing_api():
    llm = FakeLLM(LLMResponse(content="ok"))
    agent, clock, trace = build(llm)
    assert agent.run("hello") == "ok"
    assert llm.calls == 1 and clock.waits == []
    ends = [e for e in trace.events(agent.last_run_id) if e.event_type == "llm_end"]
    assert len(ends) == 1 and ends[0].metadata["attempt"] == 1 and ends[0].success


def test_transient_failure_retries_then_succeeds():
    llm = FakeLLM(timeout_error(), LLMResponse(content="ok"))
    agent, clock, trace = build(llm, retry_base_delay_s=0.5)
    assert agent.run("hello") == "ok"
    assert llm.calls == 2 and sum(clock.waits) == pytest.approx(0.5)
    events = trace.events(agent.last_run_id)
    ends = [e for e in events if e.event_type == "llm_end"]
    assert [(e.metadata["attempt"], e.success) for e in ends] == [(1, False), (2, True)]
    assert ends[0].metadata["error_code"] == LLMTimeoutError.error_code
    assert len([e for e in events if e.event_type == "llm_retry"]) == 1
    assert events[-1].event_type == "agent_end"


def test_attempt_limit_and_exponential_backoff():
    llm = FakeLLM(timeout_error(), timeout_error(), timeout_error())
    agent, clock, trace = build(llm, max_llm_attempts=3, retry_base_delay_s=1,
                                retry_max_delay_s=4)
    with pytest.raises(LLMTimeoutError) as caught:
        agent.run("hello")
    assert caught.value.error_code == "llm_timeout" and caught.value.retryable
    assert llm.calls == 3 and sum(clock.waits) == pytest.approx(3)
    events = trace.events(agent.last_run_id)
    assert len([e for e in events if e.event_type == "llm_retry"]) == 2
    assert events[-1].event_type == "agent_error"


def test_unknown_failure_is_not_retried():
    llm = FakeLLM(RuntimeError("private failure"))
    agent, clock, trace = build(llm)
    with pytest.raises(AgentLLMError) as caught:
        agent.run("secret prompt")
    assert caught.value.error_code == "llm_error" and not caught.value.retryable
    assert llm.calls == 1 and clock.waits == []
    assert "private" not in json.dumps([e.model_dump(mode="json") for e in trace.events(agent.last_run_id)])


def test_429_uses_retry_after_before_backoff_and_caps_wait():
    llm = FakeLLM(rate_limit("2"), LLMResponse(content="ok"))
    agent, clock, trace = build(llm, retry_base_delay_s=0.1, retry_max_delay_s=3)
    assert agent.run("hello") == "ok"
    assert sum(clock.waits) == pytest.approx(2)
    retry = next(e for e in trace.events(agent.last_run_id) if e.event_type == "llm_retry")
    assert retry.metadata["error_code"] == LLMRateLimitError.error_code
    assert retry.metadata["delay_ms"] == 2000
    later = FakeLLM(rate_limit("100"), LLMResponse(content="ok"))
    capped, clock2, _ = build(later, retry_max_delay_s=1)
    assert capped.run("hello") == "ok"
    assert sum(clock2.waits) == pytest.approx(1)


def test_retry_after_http_date_and_invalid_value():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_retry_after(format_datetime(now + timedelta(seconds=5)), now=now) == pytest.approx(5)
    assert parse_retry_after("not a date", now=now) is None
    assert parse_retry_after("-4", now=now) == 0


def test_cancel_during_retry_wait_stops_next_attempt():
    llm = FakeLLM(timeout_error())
    clock = FakeClock()
    cancelled = False
    def wait(seconds):
        nonlocal cancelled
        clock.wait(seconds)
        cancelled = True
    trace = TraceRecorder()
    agent = Agent(llm, ToolRegistry(), trace=trace, wait_fn=wait, clock=clock,
                  jitter_fn=lambda: 0.5)
    with pytest.raises(AgentCancelled) as caught:
        agent.run("go", cancelled=lambda: cancelled)
    assert caught.value.error_code == "agent_cancelled"
    assert llm.calls == 1 and len(clock.waits) == 1
    assert trace.events(agent.last_run_id)[-1].metadata["error_code"] == "agent_cancelled"


def test_agent_total_timeout_stops_retry():
    llm = FakeLLM(timeout_error())
    agent, clock, trace = build(llm, retry_base_delay_s=1)
    with pytest.raises(AgentTimeoutError) as caught:
        agent.run("go", total_timeout_s=0.1)
    assert caught.value.error_code == "agent_timeout" and not caught.value.retryable
    assert llm.calls == 1 and sum(clock.waits) == pytest.approx(0.1)
    assert trace.events(agent.last_run_id)[-1].metadata["error_code"] == "agent_timeout"


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def test_tool_argument_error_is_feedback_without_retry():
    calls = []
    tools = ToolRegistry()
    tools.register("test", "test", Args, lambda value: calls.append(value))
    llm = FakeLLM(LLMResponse(tool_calls=[ToolCall(id="x", name="test", arguments='{"value":"bad"}')]),
                  LLMResponse(content="recovered"))
    clock = FakeClock()
    trace = TraceRecorder()
    agent = Agent(llm, tools, trace=trace, clock=clock, wait_fn=clock.wait)
    assert agent.run("go") == "recovered"
    assert llm.calls == 2 and calls == [] and clock.waits == []
    message = json.loads(agent.context.messages[2]["content"])
    assert set(message) == {"ok", "content", "error"} and not message["ok"]
    tool_end = next(e for e in trace.events(agent.last_run_id) if e.event_type == "tool_end")
    assert tool_end.metadata["error_code"] == "tool_argument_error"


def test_tool_execution_failure_never_retries_handler():
    calls = []
    def broken(value):
        calls.append(value)
        raise RuntimeError("private tool secret")
    tools = ToolRegistry()
    tools.register("broken", "broken", Args, broken)
    llm = FakeLLM(LLMResponse(tool_calls=[ToolCall(id="x", name="broken", arguments='{"value":1}')]),
                  LLMResponse(content="recovered"))
    trace = TraceRecorder()
    agent = Agent(llm, tools, trace=trace)
    assert agent.run("go") == "recovered"
    assert calls == [1]
    events = trace.events(agent.last_run_id)
    assert next(e for e in events if e.event_type == "tool_end").metadata["error_code"] == "tool_execution_error"
    assert "private tool secret" not in str(events)


def test_multi_agent_dependency_failure_has_structured_code():
    class FailLLM:
        def chat(self, messages, tools):
            raise RuntimeError("private detail")
    trace = TraceRecorder()
    root = AgentFactory(lambda: FailLLM(), ToolRegistry(), trace=trace).create()
    results = root.dispatch([TaskNode(task_id="bad", description="fail"),
                             TaskNode(task_id="after", description="skip", depends_on=["bad"])])
    assert [r.status for r in results] == ["failed", "skipped"]
    assert results[1].error_code == DependencyFailedError.error_code
    skipped = next(e for e in trace.events(root.last_run_id) if e.event_type == "task_skipped")
    assert skipped.metadata["error_code"] == DependencyFailedError.error_code
    assert "private detail" not in str(trace.events(root.last_run_id))


def test_sdk_retries_disabled_and_classification_uses_type():
    client = LLMClient("fake", "https://example.invalid/v1", "fake-model", retries=2)
    assert client.max_attempts == 3 and client.client.max_retries == 0
    assert isinstance(classify_llm_exception(rate_limit()), LLMRateLimitError)
    assert isinstance(classify_llm_exception(RuntimeError("429")), AgentLLMError)
    assert not classify_llm_exception(RuntimeError("429")).retryable
    def fail(**kwargs):
        raise rate_limit("1")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(LLMRateLimitError) as caught:
        client.chat([{"role": "user", "content": "test"}], [])
    assert caught.value.retry_after_s == 1


def test_http_status_and_streaming_guard():
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    temporary = APIStatusError("temporary", response=httpx.Response(503, request=request), body=None)
    permanent = APIStatusError("permanent", response=httpx.Response(400, request=request), body=None)
    assert classify_llm_exception(temporary).retryable
    assert not classify_llm_exception(permanent).retryable
    llm = FakeLLM(timeout_error())
    llm.streaming = True
    agent, clock, trace = build(llm)
    with pytest.raises(LLMTimeoutError):
        agent.run("go")
    assert llm.calls == 1 and clock.waits == []
    assert not any(e.event_type == "llm_retry" for e in trace.events(agent.last_run_id))
