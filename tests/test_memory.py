"""Offline SQLite FTS5 memory and optional Agent integration tests."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from xcode_agent import (Agent, AgentFactory, LLMResponse, MemoryStore,
                         MemoryStoreError, ToolRegistry, TraceRecorder)


class CaptureLLM:
    def __init__(self):
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append(messages)
        return LLMResponse(content="ok")


def test_save_update_delete_keep_fts_index_consistent(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        saved = store.save("project", "alpha launch", item_id="note-1")
        assert saved.id == "note-1" and saved.score is None
        assert [item.id for item in store.search("project", "alpha")] == ["note-1"]
        updated = store.save("project", "beta release", item_id="note-1")
        assert updated.created_at == saved.created_at
        assert updated.content == "beta release"
        assert store.search("project", "alpha") == []
        assert [item.id for item in store.search("project", "beta")] == ["note-1"]
        assert store.delete("project", "note-1") is True
        assert store.delete("project", "note-1") is False
        assert store.search("project", "beta") == []
    with MemoryStore(path) as reopened:
        assert reopened.search("project", "beta") == []
        reopened.save("project", "persistent sqlite note")
    with MemoryStore(path) as reopened:
        assert len(reopened.search("project", "persistent")) == 1


def test_namespace_isolation_with_same_item_id():
    with MemoryStore() as store:
        store.save("alice", "alpha private", item_id="same")
        store.save("bob", "beta private", item_id="same")
        assert [item.content for item in store.search("alice", "private")] == ["alpha private"]
        assert [item.content for item in store.search("bob", "private")] == ["beta private"]
        assert not store.delete("other", "same")
        assert store.delete("alice", "same")
        assert store.search("alice", "alpha") == []
        assert [item.id for item in store.search("bob", "beta")] == ["same"]


def test_bm25_order_top_k_and_literal_query():
    with MemoryStore() as store:
        store.save("n", "python python python", item_id="frequent")
        store.save("n", "python", item_id="single")
        store.save("n", "database", item_id="other")
        matches = store.search("n", "python", top_k=2)
        assert len(matches) == 2
        assert [item.score for item in matches] == sorted(item.score for item in matches)
        assert [item.id for item in store.search("n", "python", top_k=1)] == [matches[0].id]
        assert store.search("n", '" OR NOT (*') == []


def test_chinese_unicode61_limitation_is_explicit():
    with MemoryStore() as store:
        store.save("zh", "北京天气", item_id="joined")
        store.save("zh", "北京 天气", item_id="separated")
        assert [item.id for item in store.search("zh", "北京")] == ["separated"]
        assert [item.id for item in store.search("zh", "北京天气")] == ["joined"]


@pytest.mark.parametrize("operation", [
    lambda store: store.save("", "text"),
    lambda store: store.save("n", ""),
    lambda store: store.save("n", "text", item_id=""),
    lambda store: store.search("n", "word", top_k=0),
    lambda store: store.search("n", "word", top_k=101),
    lambda store: store.search("n", "x" * 1001),
    lambda store: store.delete("", "id"),
])
def test_invalid_inputs(operation):
    with MemoryStore() as store:
        with pytest.raises(ValueError):
            operation(store)


def test_closed_store_raises():
    store = MemoryStore()
    store.close()
    store.close()
    with pytest.raises(MemoryStoreError):
        store.save("n", "text")
    with pytest.raises(MemoryStoreError):
        store.search("n", "text")
    with pytest.raises(MemoryStoreError):
        store.search("n", "!!!")


def test_agent_injects_only_budgeted_ephemeral_memory():
    with MemoryStore() as store:
        store.save("notes", "alpha " + "x" * 200)
        llm = CaptureLLM()
        agent = Agent(llm, ToolRegistry(), memory=store, memory_namespace="notes",
                      memory_top_k=1, memory_char_budget=90)
        assert agent.run("alpha") == "ok"
        injected = llm.requests[0][0]
        assert injected["role"] == "user" and "Retrieved saved notes" in injected["content"]
        assert "alpha" in injected["content"]
        assert len(injected["content"]) <= 90
        assert llm.requests[0][-1] == {"role": "user", "content": "alpha"}
        assert [message["role"] for message in agent.context.messages] == ["user", "assistant"]
        assert all("saved notes" not in str(message) for message in agent.context.messages)


def test_memory_lookup_failure_is_traced_without_llm_call():
    store = MemoryStore()
    store.close()
    llm, trace = CaptureLLM(), TraceRecorder()
    agent = Agent(llm, ToolRegistry(), trace=trace, memory=store, memory_namespace="n")
    with pytest.raises(MemoryStoreError):
        agent.run("query")
    assert llm.requests == []
    assert [event.event_type for event in trace.events(agent.last_run_id)] == ["agent_start", "agent_error"]


def test_agent_without_memory_preserves_existing_messages():
    llm = CaptureLLM()
    agent = Agent(llm, ToolRegistry())
    assert agent.run("hello") == "ok"
    assert llm.requests[0] == [{"role": "user", "content": "hello"}]
    with pytest.raises(ValueError):
        Agent(CaptureLLM(), ToolRegistry(), memory_namespace="unused")


def test_long_user_input_does_not_break_optional_lookup():
    with MemoryStore() as store:
        store.save("n", "alpha saved fact")
        llm = CaptureLLM()
        agent = Agent(llm, ToolRegistry(), memory=store, memory_namespace="n")
        assert agent.run("alpha " + "x" * 2_000) == "ok"
        assert any("saved fact" in message["content"] for message in llm.requests[0])


def test_factory_isolates_by_default_and_shares_only_when_explicit():
    with MemoryStore() as store:
        store.save("shared", "alpha project fact")
        clients = []
        def make_client():
            client = CaptureLLM()
            clients.append(client)
            return client
        isolated = AgentFactory(make_client, ToolRegistry(), memory=store)
        parent = isolated.create()
        child = isolated.create("leaf", parent)
        assert parent.agent.memory_namespace != child.agent.memory_namespace
        child.agent.run("alpha")
        assert [m["role"] for m in clients[-1].requests[0]] == ["user"]
        shared = AgentFactory(make_client, ToolRegistry(), memory=store,
                              shared_memory_namespace="shared")
        shared_parent = shared.create()
        shared_child = shared.create("leaf", shared_parent)
        assert shared_child.agent.memory_namespace == "shared"
        shared_child.agent.run("alpha")
        assert [m["role"] for m in clients[-1].requests[0]] == ["user", "user"]
        with pytest.raises(ValueError):
            AgentFactory(make_client, ToolRegistry(), memory=store, shared_memory_namespace="")


def test_shared_store_is_safe_across_threads():
    with MemoryStore() as store:
        def save_one(index):
            return store.save("workers", f"parallel record {index}", item_id=str(index))
        with ThreadPoolExecutor(max_workers=4) as pool:
            items = list(pool.map(save_one, range(20)))
        assert len(items) == 20
        assert len(store.search("workers", "parallel", top_k=20)) == 20
