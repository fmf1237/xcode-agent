import json

import pytest
from pydantic import BaseModel, ConfigDict

from xcode_agent import Agent, AgentLLMError, ContextManager, LLMResponse, MaxRoundsExceeded, ToolCall, ToolRegistry


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def call(id, name="double", arguments='{"value": 2}'):
    return ToolCall(id=id, name=name, arguments=arguments)


def build(*responses, max_rounds=8, max_messages=40):
    registry = ToolRegistry()
    registry.register("double", "Double an integer", Args, lambda value: value * 2)
    fake = FakeLLM(*responses)
    context = ContextManager(max_messages=max_messages)
    return Agent(fake, registry, context, max_rounds), fake, registry


def test_direct_answer():
    agent, fake, _ = build(LLMResponse(content="Hello"))
    assert agent.run("Hi") == "Hello"
    assert [m["role"] for m in agent.context.messages] == ["user", "assistant"]
    assert fake.requests[0] == [{"role": "user", "content": "Hi"}]


def test_one_tool_and_message_group():
    agent, fake, _ = build(LLMResponse(tool_calls=[call("a")]), LLMResponse(content="Four"))
    assert agent.run("Double two") == "Four"
    messages = agent.context.messages
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["id"] == messages[2]["tool_call_id"] == "a"
    assert json.loads(messages[2]["content"]) == {"ok": True, "content": 4, "error": None}
    assert len(fake.requests) == 2


def test_multiple_calls_in_one_response():
    agent, _, _ = build(LLMResponse(tool_calls=[call("a"), call("b", arguments='{"value": 3}')]),
                        LLMResponse(content="Done"))
    assert agent.run("Both") == "Done"
    assert [m.get("tool_call_id") for m in agent.context.messages if m["role"] == "tool"] == ["a", "b"]


def test_multiple_tool_rounds():
    agent, fake, _ = build(LLMResponse(tool_calls=[call("a")]),
                           LLMResponse(tool_calls=[call("b")]), LLMResponse(content="Done"))
    assert agent.run("Again") == "Done"
    assert len(fake.requests) == 3
    assert [m["role"] for m in agent.context.messages] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


@pytest.mark.parametrize("tool_call,fragment", [
    (call("a", name="missing"), "Unknown tool"),
    (call("a", arguments='{"value": "bad"}'), "Invalid arguments"),
    (call("a", arguments="{"), "Invalid arguments"),
])
def test_tool_errors_are_returned_to_model(tool_call, fragment):
    agent, _, _ = build(LLMResponse(tool_calls=[tool_call]), LLMResponse(content="Recovered"))
    assert agent.run("Try") == "Recovered"
    result = json.loads(agent.context.messages[2]["content"])
    assert not result["ok"] and fragment in result["error"]


def test_tool_exception_is_returned_to_model():
    agent, _, registry = build(LLMResponse(tool_calls=[call("a", "explode")]), LLMResponse(content="Recovered"))
    def explode(value):
        raise RuntimeError("boom")
    registry.register("explode", "Fail", Args, explode)
    assert agent.run("Try") == "Recovered"
    assert "Tool execution failed" in json.loads(agent.context.messages[2]["content"])["error"]


def test_max_rounds_keeps_complete_call_group():
    agent, _, _ = build(LLMResponse(tool_calls=[call("a")]), max_rounds=1)
    with pytest.raises(MaxRoundsExceeded):
        agent.run("Go")
    assert [m["role"] for m in agent.context.messages] == ["user", "assistant", "tool"]


def test_history_trim_preserves_groups_and_followup():
    agent, fake, _ = build(LLMResponse(tool_calls=[call("a")]), LLMResponse(content="First"),
                           LLMResponse(content="Second"), max_messages=3)
    assert agent.run("One") == "First"
    assert agent.run("Two") == "Second"
    assert [m["role"] for m in agent.context.messages] == ["user", "assistant"]
    assert fake.requests[-1] == [{"role": "user", "content": "Two"}]


def test_model_error_is_distinct():
    agent, _, _ = build(RuntimeError("offline"))
    with pytest.raises(AgentLLMError, match="offline"):
        agent.run("Hi")
