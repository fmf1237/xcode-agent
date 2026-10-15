"""Offline tests for stateless memory filtering and temporary injection."""
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from xcode_agent import Agent, ContextManager, LLMResponse, MemoryItem, MemoryStore, ToolCall, ToolRegistry
from xcode_agent.memory import select_memories


def note(item_id, content, score):
    now = datetime.now(timezone.utc)
    return MemoryItem(id=item_id, namespace="n", content=content, created_at=now,
                      updated_at=now, score=score)


def test_selection_sorts_deduplicates_and_fits_complete_notes():
    items = [note("low", "python low priority", -1),
             note("best", "Python must use version 3.11.", -9),
             note("duplicate", "python must use version 3.11!", -8),
             note("large", "python " + "X" * 200, -7),
             note("next", "python tests run offline", -6)]
    selected = select_memories(items, top_k=2, char_budget=70)
    assert [item.id for item in selected] == ["best", "next"]
    assert items[1].content == "Python must use version 3.11."


def test_near_duplicates_keep_distinct_numbers_and_constraints():
    long = "Project constraint: all tests must run offline and use Python 3.11."
    items = [note("one", long, -2),
             note("near", long.replace("constraint", "constraints"), -1.9),
             note("version", long.replace("3.11", "3.12"), -1.8),
             note("negative", long.replace("must run", "must not run"), -1.7)]
    selected = select_memories(items, top_k=4, char_budget=1000)
    assert [item.id for item in selected] == ["one", "version", "negative"]


@pytest.mark.parametrize("top_k,budget", [(0, 100), (101, 100), (True, 100), (1, -1), (1, True)])
def test_selection_rejects_invalid_limits(top_k, budget):
    with pytest.raises(ValueError):
        select_memories([], top_k=top_k, char_budget=budget)


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append(messages)
        return self.responses.pop(0)


def test_agent_injection_is_ephemeral_across_repeated_queries_and_empty_results():
    with MemoryStore() as store:
        store.save("n", "python use pytest for tests", item_id="a")
        llm = ScriptedLLM(LLMResponse(content="first"), LLMResponse(content="second"),
                          LLMResponse(content="third"))
        agent = Agent(llm, ToolRegistry(), memory=store, memory_namespace="n")
        agent.run("python")
        agent.run("python")
        agent.run("unmatchedword")
        for request in llm.requests[:2]:
            assert sum("Retrieved saved notes" in m.get("content", "") for m in request) == 1
        assert all("Retrieved saved notes" not in m.get("content", "") for m in llm.requests[2])
        assert all("Retrieved saved notes" not in m.get("content", "") for m in agent.context.messages)


def test_agent_deduplicates_and_skips_large_note_with_namespace_and_budget():
    with MemoryStore() as store:
        store.save("n", "python " + "X" * 300)
        store.save("n", "python requires pytest")
        store.save("n", "Python requires pytest!")
        store.save("other", "python secret from other namespace")
        llm = ScriptedLLM(LLMResponse(content="done"))
        agent = Agent(llm, ToolRegistry(), memory=store, memory_namespace="n",
                      memory_top_k=2, memory_char_budget=100)
        agent.run("python")
        injected = llm.requests[0][0]["content"]
        assert injected.count("requires pytest") == 1
        assert "secret" not in injected and "XXX" not in injected
        assert len(injected) <= 100


class Args(BaseModel):
    value: int


def test_memory_does_not_break_tool_pairing_or_history_trim():
    with MemoryStore() as store:
        store.save("n", "calculate with double tool")
        tools = ToolRegistry()
        tools.register("double", "Double", Args, lambda value: value * 2)
        llm = ScriptedLLM(LLMResponse(tool_calls=[ToolCall(id="t1", name="double",
                                                            arguments='{"value": 2}')]),
                          LLMResponse(content="four"), LLMResponse(content="next"))
        agent = Agent(llm, tools, ContextManager(max_messages=3), memory=store, memory_namespace="n")
        assert agent.run("calculate") == "four"
        assert [m["role"] for m in llm.requests[1]] == ["user", "user", "assistant", "tool"]
        assert llm.requests[1][-2]["tool_calls"][0]["id"] == llm.requests[1][-1]["tool_call_id"]
        assert agent.run("other") == "next"
        assert [m["role"] for m in agent.context.messages] == ["user", "assistant"]
