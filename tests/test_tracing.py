import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest
from pydantic import BaseModel

from xcode_agent import (Agent, AgentCancelled, AgentFactory, AgentLLMError,
                         LLMResponse, TaskNode, ToolCall, ToolRegistry, TraceRecorder)


class Args(BaseModel):
    value: int


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)

    def chat(self, messages, tools):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def event_types(events):
    return [event.event_type for event in events]


def test_normal_agent_event_order_and_fields():
    trace = TraceRecorder()
    tools = ToolRegistry()
    tools.register("double", "double", Args, lambda value: value * 2)
    llm = FakeLLM(LLMResponse(tool_calls=[ToolCall(id="c1", name="double", arguments='{"value":2}')]),
                  LLMResponse(content="four"))
    agent = Agent(llm, tools, trace=trace, agent_id="agent-1")
    assert agent.run("calculate") == "four"
    events = trace.events(agent.last_run_id)
    assert event_types(events) == ["agent_start", "llm_start", "llm_end", "tool_start", "tool_end",
                                   "llm_start", "llm_end", "agent_end"]
    assert all(e.run_id == agent.last_run_id and e.agent_id == "agent-1" for e in events)
    assert events[2].success and events[4].success and events[-1].success
    assert events[4].duration_ms >= 0
    assert events[4].metadata["tool_call_id"] == "c1"


def test_llm_failure_has_matching_end_and_agent_error():
    trace = TraceRecorder()
    agent = Agent(FakeLLM(RuntimeError("private failure detail")), ToolRegistry(), trace=trace)
    with pytest.raises(AgentLLMError):
        agent.run("private prompt")
    events = trace.events(agent.last_run_id)
    assert event_types(events) == ["agent_start", "llm_start", "llm_end", "agent_error"]
    assert events[2].success is False and events[2].error_type == "RuntimeError"
    assert events[3].error_type == "AgentLLMError"
    assert "private" not in json.dumps([e.model_dump(mode="json") for e in events])


def test_tool_failure_is_traced_and_agent_can_recover():
    trace = TraceRecorder()
    tools = ToolRegistry()
    def fail(value):
        raise RuntimeError("secret-tool-message")
    tools.register("fail", "fail", Args, fail)
    agent = Agent(FakeLLM(LLMResponse(tool_calls=[ToolCall(id="c", name="fail", arguments='{"value":1}')]),
                          LLMResponse(content="recovered")), tools, trace=trace)
    assert agent.run("go") == "recovered"
    events = trace.events(agent.last_run_id)
    assert event_types(events).count("tool_end") == 1
    assert next(e for e in events if e.event_type == "tool_end").success is False
    assert events[-1].event_type == "agent_end"
    assert "secret-tool-message" not in str(events)


def test_agent_cancellation_has_end_event():
    trace = TraceRecorder()
    agent = Agent(FakeLLM(), ToolRegistry(), trace=trace)
    with pytest.raises(AgentCancelled):
        agent.run("go", cancelled=lambda: True)
    assert event_types(trace.events(agent.last_run_id)) == ["agent_start", "agent_error"]


def test_parallel_children_share_run_but_keep_agent_ids():
    barrier = Barrier(2)
    class ParallelLLM:
        def chat(self, messages, tools):
            barrier.wait(timeout=3)
            return LLMResponse(content="ok")
    trace = TraceRecorder()
    factory = AgentFactory(lambda: ParallelLLM(), ToolRegistry(), trace=trace)
    root = factory.create()
    results = root.dispatch([TaskNode(task_id="a", description="one"),
                             TaskNode(task_id="b", description="two")], max_workers=2)
    assert all(result.status == "success" for result in results)
    assert results[0].run_id == results[1].run_id == root.last_run_id
    events = trace.events(root.last_run_id)
    for result in results:
        own = [e for e in events if e.agent_id == result.agent_id]
        assert {"child_created", "child_start", "agent_start", "agent_end", "task_success"} <= set(event_types(own))
        assert all(e.metadata.get("task_id") in (None, result.task_id) for e in own)
    assert results[0].agent_id != results[1].agent_id


def test_concurrent_dispatch_runs_have_separate_event_lists():
    class InstantLLM:
        def chat(self, messages, tools):
            return LLMResponse(content="ok")
    trace = TraceRecorder()
    factory = AgentFactory(lambda: InstantLLM(), ToolRegistry(), trace=trace)
    roots = [factory.create(), factory.create()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(root.dispatch, [TaskNode(task_id="same", description="work")])
                   for root in roots]
        results = [future.result() for future in futures]
    assert results[0][0].run_id != results[1][0].run_id
    for group in results:
        own = trace.events(group[0].run_id)
        assert {e.agent_id for e in own if e.event_type == "agent_start"} == {group[0].agent_id}


def test_failed_dependency_is_skipped_and_linked():
    class SelectiveLLM:
        def chat(self, messages, tools):
            if messages[-1]["content"] == "fail":
                raise RuntimeError("private text")
            return LLMResponse(content="ok")
    trace = TraceRecorder()
    root = AgentFactory(lambda: SelectiveLLM(), ToolRegistry(), trace=trace).create()
    results = root.dispatch([TaskNode(task_id="bad", description="fail"),
                             TaskNode(task_id="after", description="later", depends_on=["bad"])])
    assert [r.status for r in results] == ["failed", "skipped"]
    events = trace.events(root.last_run_id)
    assert "task_failed" in event_types(events) and "task_skipped" in event_types(events)
    skipped = next(e for e in events if e.event_type == "task_skipped")
    assert skipped.metadata["task_id"] == "after"
    assert skipped.metadata["parent_agent_id"] == root.identity.agent_id
    assert "private text" not in str(events)


def test_timeout_and_cooperative_cancel_are_traced():
    entered, release, finished = Event(), Event(), Event()
    class SlowLLM:
        def chat(self, messages, tools):
            entered.set()
            release.wait(timeout=3)
            finished.set()
            return LLMResponse(content="late")
    trace = TraceRecorder()
    root = AgentFactory(lambda: SlowLLM(), ToolRegistry(), trace=trace).create()
    try:
        result = root.dispatch([TaskNode(task_id="slow", description="sensitive prompt", timeout_s=0.05)])[0]
        assert entered.is_set() and result.status == "timed_out"
        assert "task_timed_out" in event_types(trace.events(result.run_id))
    finally:
        release.set()
    assert finished.wait(timeout=3)
    for _ in range(1000):
        if "task_cancelled" in event_types(trace.events(result.run_id)):
            break
        Event().wait(0.001)
    events = trace.events(result.run_id)
    assert "task_cancelled" in event_types(events)
    assert "agent_error" in event_types(events)
    assert "sensitive prompt" not in str(events)


def test_trace_filters_unapproved_metadata_and_returns_copies():
    trace = TraceRecorder()
    trace.record("run", "agent", "custom", metadata={"prompt": "secret", "api_key": "secret",
                                                       "task_id": "safe-id", "raw_arguments": "secret"})
    event = trace.events("run")[0]
    assert event.metadata == {"task_id": "safe-id"}
    event.metadata["task_id"] = "changed"
    assert trace.events("run")[0].metadata["task_id"] == "safe-id"
