"""Offline stdio MCP integration tests against local FastMCP subprocesses."""
import json
from pathlib import Path
import sys

import pytest

from xcode_agent import (Agent, AgentFactory, LLMResponse, MCPClient, MCPConnectionError,
                         MCPServerConfig, MemoryStore, SkillRegistry, ToolCall,
                         TaskNode, ToolRegistry, ToolResult, TraceRecorder)


DEMO_SERVER = Path(__file__).resolve().parents[1] / "examples" / "mcp_demo_server.py"


def config(name="demo", server=DEMO_SERVER, allowed=("add", "echo"), call_timeout=3):
    return MCPServerConfig(server_name=name, command=sys.executable, args=(str(server),),
                           allowed_tools=frozenset(allowed), connect_timeout_s=8,
                           call_timeout_s=call_timeout)


@pytest.fixture(scope="module")
def demo_client():
    with MCPClient(config()) as client:
        yield client


def test_discovery_and_json_schema_mapping(demo_client):
    infos = demo_client.list_tools()
    assert {item.qualified_name for item in infos} == {"mcp__demo__add", "mcp__demo__echo"}
    registry = ToolRegistry()
    assert set(demo_client.register_tools(registry)) == {item.qualified_name for item in infos}
    for item in infos:
        schema = next(entry["function"]["parameters"] for entry in registry.schemas()
                      if entry["function"]["name"] == item.qualified_name)
        assert schema == item.input_schema
    add = next(item for item in infos if item.tool_name == "add")
    assert set(add.input_schema["required"]) == {"a", "b"}
    assert add.input_schema["properties"]["a"]["type"] == "integer"


def test_external_schema_constraints_survive_registry_and_clone():
    schema = {"type": "object", "properties": {"count": {"type": "integer", "minimum": 1,
                                                            "maximum": 3}},
              "required": ["count"], "additionalProperties": False}
    registry = ToolRegistry()
    registry.register_external("mcp__fake__bounded", "Bounded", schema,
                               lambda count: ToolResult(ok=True, content=count))
    cloned = registry.clone()
    assert cloned.schemas()[0]["function"]["parameters"] == schema
    assert cloned.execute("mcp__fake__bounded", '{"count":2}').content == 2
    assert cloned.execute("mcp__fake__bounded", '{"count":0}').error_code == "tool_argument_error"
    assert cloned.execute("mcp__fake__bounded", '{"count":2,"extra":true}').error_code == "tool_argument_error"


def test_success_invalid_arguments_and_explicit_allowlist(demo_client):
    registry = ToolRegistry()
    demo_client.register_tools(registry)
    assert registry.execute("mcp__demo__add", '{"a":3,"b":5}').content == 8
    invalid = registry.execute("mcp__demo__add", '{"a":"secret","b":5}')
    assert not invalid.ok and invalid.error_code == "tool_argument_error"
    assert "secret" not in invalid.error
    invalid_json = registry.execute("mcp__demo__add", "{")
    assert not invalid_json.ok and invalid_json.error_code == "tool_argument_error"
    assert registry.execute("mcp__demo__missing", "{}").error_code == "tool_not_found"


def test_connection_failure_is_structured():
    bad = MCPServerConfig(server_name="missing", command="nonexistent-mcp-command-xcode-agent",
                          allowed_tools=frozenset({"add"}), connect_timeout_s=1)
    client = MCPClient(bad)
    with pytest.raises(MCPConnectionError) as error:
        client.connect()
    assert error.value.error_code == "mcp_connection_error"
    assert client._thread is None or not client._thread.is_alive()


@pytest.fixture
def exceptional_server(tmp_path):
    script = tmp_path / "server.py"
    script.write_text("""
import asyncio
from mcp.server.fastmcp import FastMCP
mcp = FastMCP('test')
@mcp.tool()
def fail() -> str:
    raise ValueError('PRIVATE_SERVER_ERROR')
@mcp.tool()
async def slow() -> str:
    await asyncio.sleep(2)
    return 'late'
if __name__ == '__main__':
    mcp.run(transport='stdio')
""", encoding="utf-8")
    return script


def test_mcp_server_error_is_structured_without_raw_exception(exceptional_server):
    with MCPClient(config("errors", exceptional_server, allowed=("fail",))) as client:
        registry = ToolRegistry()
        client.register_tools(registry)
        result = registry.execute("mcp__errors__fail", "{}")
        assert not result.ok and result.error_code == "mcp_tool_error"
        assert "PRIVATE_SERVER_ERROR" not in result.error


def test_mcp_timeout_does_not_retry(exceptional_server):
    with MCPClient(config("slow", exceptional_server, allowed=("slow",), call_timeout=0.1)) as client:
        registry = ToolRegistry()
        client.register_tools(registry)
        result = registry.execute("mcp__slow__slow", "{}")
        assert not result.ok and result.error_code == "mcp_timeout"


def test_close_releases_session_and_process():
    client = MCPClient(config("closed", allowed=("add",)))
    registry = ToolRegistry()
    client.register_tools(registry)
    assert registry.execute("mcp__closed__add", '{"a":1,"b":2}').content == 3
    client.close()
    client.close()
    assert not client._thread.is_alive()
    assert registry.execute("mcp__closed__add", '{"a":1,"b":2}').error_code == "mcp_connection_error"


def test_different_servers_can_expose_same_raw_tool():
    with MCPClient(config("left", allowed=("add",))) as left:
        with MCPClient(config("right", allowed=("add",))) as right:
            registry = ToolRegistry()
            left.register_tools(registry)
            right.register_tools(registry)
            assert {entry["function"]["name"] for entry in registry.schemas()} == {
                "mcp__left__add", "mcp__right__add"}
            assert registry.execute("mcp__left__add", '{"a":1,"b":2}').content == 3
            assert registry.execute("mcp__right__add", '{"a":3,"b":5}').content == 8


class FakeLLM:
    def __init__(self):
        self.step = 0
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append((messages, tools))
        self.step += 1
        if self.step == 1:
            return LLMResponse(tool_calls=[ToolCall(id="load", name="load_skill",
                                                    arguments='{"name":"helper"}')])
        if self.step == 2:
            return LLMResponse(tool_calls=[ToolCall(id="mcp", name="mcp__demo__echo",
                                                    arguments='{"text":"PRIVATE_ARGUMENT"}')])
        return LLMResponse(content="done")


def test_agent_tools_skills_memory_trace_compatibility(demo_client, tmp_path):
    folder = tmp_path / "helper"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: helper\ndescription: Helps with echo\n---\n"
                                    "Use echo when needed.\n", encoding="utf-8")
    registry = ToolRegistry()
    demo_client.register_tools(registry)
    with MemoryStore() as memory:
        memory.save("shared", "echo project note")
        llm, trace, logs = FakeLLM(), TraceRecorder(), []
        agent = Agent(llm, registry, memory=memory, memory_namespace="shared",
                      skills=SkillRegistry(tmp_path), trace=trace, logger=logs.append)
        assert agent.run("echo") == "done"
        assert "Retrieved saved notes" in str(llm.requests[0][0])
        assert "Use echo when needed" in str(llm.requests[1][0])
        assert json.loads(agent.context.messages[-2]["content"])["content"] == "PRIVATE_ARGUMENT"
        assert [m.get("tool_call_id") for m in agent.context.messages if m["role"] == "tool"] == ["load", "mcp"]
        assert sum(event.event_type == "tool_start" for event in trace.events(agent.last_run_id)) == 2
        assert "PRIVATE_ARGUMENT" not in str(trace.events(agent.last_run_id))
        assert "PRIVATE_ARGUMENT" not in str(logs)


def test_multi_agent_can_share_registered_mcp_tools(demo_client):
    registry = ToolRegistry()
    demo_client.register_tools(registry)

    class ChildLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(tool_calls=[ToolCall(id="add", name="mcp__demo__add",
                                                        arguments='{"a":3,"b":5}')])
            assert json.loads(messages[-1]["content"])["content"] == 8
            return LLMResponse(content="8")

    factory = AgentFactory(ChildLLM, registry)
    parent = factory.create()
    results = parent.dispatch([TaskNode(task_id="one", description="sum"),
                               TaskNode(task_id="two", description="sum")], max_workers=2)
    assert [result.status for result in results] == ["success", "success"]
    assert [result.output for result in results] == ["8", "8"]


def test_tool_exposure_limit():
    limited = MCPServerConfig(server_name="limited", command=sys.executable,
                              args=(str(DEMO_SERVER),), allowed_tools=frozenset({"add", "echo"}),
                              max_tools=1)
    with pytest.raises(MCPConnectionError):
        MCPClient(limited).connect()
