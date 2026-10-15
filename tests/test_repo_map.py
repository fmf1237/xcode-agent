"""Offline tests for the deliberately small, read-only Python repository map."""
import json
from pathlib import Path

import pytest

from xcode_agent import (Agent, AgentFactory, LLMResponse, RepoMap, TaskNode,
                         ToolCall, ToolRegistry, TraceRecorder, register_repo_map_tool)


def write(root: Path, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_functions_classes_methods_nested_and_async(tmp_path):
    write(tmp_path, "pkg/code.py", "def outer():\n    def inner(): pass\nclass Box:\n    def get(self): pass\n    async def put(self): pass\n")
    result = RepoMap(tmp_path).scan()
    assert [item.path for item in result.files] == ["pkg/code.py"]
    symbols = result.files[0].symbols
    assert [(s.qualified_name, s.kind, s.line) for s in symbols] == [
        ("outer", "function", 1), ("outer.inner", "function", 2),
        ("Box", "class", 3), ("Box.get", "method", 4), ("Box.put", "method", 5)]
    assert symbols[-1].is_async
    assert "method Box.get @ 4" in result.text


def test_multiple_files_empty_and_parse_failure(tmp_path):
    assert RepoMap(tmp_path).scan().files == []
    write(tmp_path, "b.py", "def b(): pass\n")
    write(tmp_path, "a.py", "def a(): pass\n")
    write(tmp_path, "broken.py", "def bad(:\n")
    result = RepoMap(tmp_path).scan()
    assert [item.path for item in result.files] == ["a.py", "b.py"]
    assert [(i.path, i.reason) for i in result.issues] == [("broken.py", "parse_error")]


def test_ignored_sensitive_and_size_limits(tmp_path):
    for name in (".git/secret.py", ".venv/lib.py", "__pycache__/cache.py",
                 "build/output.py", "settings.py", "secrets.py"):
        write(tmp_path, name, "def hidden(): pass\n")
    write(tmp_path, "okay.py", "def visible(): pass\n")
    write(tmp_path, "huge.py", "#" * 200)
    result = RepoMap(tmp_path, max_file_bytes=50).scan()
    assert [f.path for f in result.files] == ["okay.py"]
    assert [(i.path, i.reason) for i in result.issues] == [("huge.py", "too_large")]
    assert "hidden" not in result.text
    with pytest.raises(ValueError, match="ignored"):
        RepoMap(tmp_path).scan(".git")


@pytest.mark.parametrize("path", ["../outside", "/tmp", "C:\\Windows", "..\\outside"])
def test_path_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        RepoMap(tmp_path).scan(path)


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("def private(): pass\n", encoding="utf-8")
    try:
        (tmp_path / "link.py").symlink_to(outside)
        (tmp_path / "linked_dir").symlink_to(tmp_path.parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this host")
    result = RepoMap(tmp_path).scan()
    assert all(s.name != "private" for f in result.files for s in f.symbols)
    assert [(i.path, i.reason) for i in result.issues] == [("link.py", "symlink_skipped")]
    with pytest.raises(ValueError, match="symlink"):
        RepoMap(tmp_path).scan("linked_dir")


def test_output_budget_and_file_count(tmp_path):
    for index in range(4):
        write(tmp_path, f"f{index}.py", f"def function_{index}(): pass\n")
    result = RepoMap(tmp_path, max_files=2, max_output_chars=75).scan()
    assert len(result.files) == 2
    assert result.truncated and len(result.text) <= 75
    assert result.text.endswith("\n")
    tiny = RepoMap(tmp_path, max_output_chars=1).scan()
    assert tiny.text == "" and tiny.truncated


def test_invalid_utf8_is_isolated(tmp_path):
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe")
    write(tmp_path, "good.py", "def okay(): pass\n")
    result = RepoMap(tmp_path).scan()
    assert [(i.path, i.reason) for i in result.issues] == [("bad.py", "invalid_utf8")]
    assert [f.path for f in result.files] == ["good.py"]


def test_no_source_literals_or_secret_error_text(tmp_path):
    write(tmp_path, "module.py", "API_KEY = 'super-secret-value'\ndef safe(): return API_KEY\n")
    result = RepoMap(tmp_path).scan()
    assert "super-secret-value" not in result.model_dump_json()
    assert "safe" in result.text


class SequenceLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append((messages, tools))
        return self.responses.pop(0)


def test_agent_tool_call_and_trace_compatibility(tmp_path):
    write(tmp_path, "sample.py", "def example(): pass\n")
    registry = ToolRegistry()
    assert registry.get("repo_map") is None
    register_repo_map_tool(registry, tmp_path)
    trace = TraceRecorder()
    llm = SequenceLLM(LLMResponse(tool_calls=[ToolCall(id="map-1", name="repo_map", arguments='{"path":"."}')]),
                      LLMResponse(content="Found example"))
    agent = Agent(llm, registry, trace=trace)
    assert agent.run("Show structure") == "Found example"
    assert any(t["function"]["name"] == "repo_map" for t in llm.requests[0][1])
    messages = agent.context.messages
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["id"] == messages[2]["tool_call_id"] == "map-1"
    assert "example" in json.loads(messages[2]["content"])["content"]
    assert [e.event_type for e in trace.events(agent.last_run_id) if e.event_type.startswith("tool_")] == ["tool_start", "tool_end"]


def test_multi_agent_shared_tool_and_trace(tmp_path):
    write(tmp_path, "sample.py", "def example(): pass\n")
    registry = ToolRegistry()
    register_repo_map_tool(registry, tmp_path)
    trace = TraceRecorder()

    def llm_factory():
        return SequenceLLM(LLMResponse(tool_calls=[ToolCall(id="map-1", name="repo_map", arguments="{}")]),
                           LLMResponse(content="mapped"))

    root = AgentFactory(llm_factory, registry, trace=trace).create()
    results = root.dispatch([TaskNode(task_id="a", description="A"),
                             TaskNode(task_id="b", description="B")], max_workers=2)
    assert [r.status for r in results] == ["success", "success"]
    assert results[0].agent_id != results[1].agent_id
    events = trace.events(root.last_run_id)
    assert len([e for e in events if e.event_type == "tool_end" and e.success]) == 2
    assert len({e.agent_id for e in events if e.event_type == "tool_end"}) == 2
