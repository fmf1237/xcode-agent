"""Offline discovery, safety and progressive Skill loading tests."""
import json
import os

import pytest
from pydantic import BaseModel

from xcode_agent import (Agent, AgentFactory, ContextManager, LLMResponse, MemoryStore,
                         SkillError, SkillRegistry, ToolCall, ToolRegistry)


def write_skill(root, folder, name, description="A helpful skill", body="Follow the checklist."):
    directory = root / folder
    directory.mkdir()
    path = directory / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n", encoding="utf-8")
    return path


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append((messages, tools))
        return self.responses.pop(0)


def skill_call(name, call_id="skill-1"):
    return ToolCall(id=call_id, name="load_skill", arguments=json.dumps({"name": name}))


def test_scan_metadata_and_lazy_full_read(tmp_path):
    path = write_skill(tmp_path, "review", "review", '"Review source code"', "FIRST BODY")
    registry = SkillRegistry(tmp_path)
    assert [(item.name, item.description) for item in registry.list()] == [("review", "Review source code")]
    path.write_text(path.read_text(encoding="utf-8").replace("FIRST BODY", "SECOND BODY"), encoding="utf-8")
    assert "SECOND BODY" in registry.load("review")
    assert "FIRST BODY" not in registry.load("review")


def test_scan_refreshes_new_skills_and_rejects_changed_name(tmp_path):
    first = write_skill(tmp_path, "first", "first")
    registry = SkillRegistry(tmp_path)
    write_skill(tmp_path, "second", "second")
    assert [info.name for info in registry.list()] == ["first"]
    assert [info.name for info in registry.scan()] == ["first", "second"]
    first.write_text(first.read_text(encoding="utf-8").replace("name: first", "name: other"),
                     encoding="utf-8")
    with pytest.raises(SkillError, match="rescan"):
        registry.load("first")


@pytest.mark.parametrize("document", [
    "name: bad\n", "---\nname: bad\n---\n", "---\nname: ../bad\ndescription: bad\n---\n",
    "---\nname: good\nname: again\ndescription: bad\n---\n",
    "---\nname: good\ndescription: [invalid YAML]\n---\n",
])
def test_invalid_frontmatter_is_rejected(tmp_path, document):
    directory = tmp_path / "bad"
    directory.mkdir()
    (directory / "SKILL.md").write_text(document, encoding="utf-8")
    with pytest.raises(SkillError):
        SkillRegistry(tmp_path)


def test_duplicate_names_and_missing_file(tmp_path):
    write_skill(tmp_path, "a", "same")
    write_skill(tmp_path, "b", "same")
    with pytest.raises(SkillError, match="Duplicate skill name"):
        SkillRegistry(tmp_path)
    (tmp_path / "b" / "SKILL.md").unlink()
    with pytest.raises(SkillError):
        SkillRegistry(tmp_path)


def test_size_limit_and_path_traversal(tmp_path):
    path = write_skill(tmp_path, "safe", "safe")
    registry = SkillRegistry(tmp_path)
    with pytest.raises(SkillError):
        registry.load("../safe")
    with pytest.raises(SkillError):
        registry.load("unknown")
    path.write_bytes(path.read_bytes() + b"x" * SkillRegistry.MAX_FILE_BYTES)
    with pytest.raises(SkillError, match="exceeds"):
        registry.load("safe")
    with pytest.raises(SkillError, match="exceeds"):
        registry.scan()


def test_symlink_escape_is_rejected(tmp_path, monkeypatch):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    write_skill(outside, "external", "external")
    try:
        os.symlink(outside / "external", tmp_path / "linked", target_is_directory=True)
    except (OSError, NotImplementedError):
        # Windows may deny symlink creation without Developer Mode; still test the guard.
        linked = tmp_path / "linked"
        linked.mkdir()
        original = type(linked).is_symlink
        monkeypatch.setattr(type(linked), "is_symlink",
                            lambda self: self == linked or original(self))
    with pytest.raises(SkillError, match="Symbolic links"):
        SkillRegistry(tmp_path)


def test_agent_loads_only_after_model_selection_and_keeps_body_ephemeral(tmp_path):
    write_skill(tmp_path, "review", "review", "Review code", "PRIVATE SKILL BODY")
    registry = SkillRegistry(tmp_path)
    llm = FakeLLM(LLMResponse(tool_calls=[skill_call("review")]), LLMResponse(content="done"))
    agent = Agent(llm, ToolRegistry(), ContextManager(system_prompt="System rules"), skills=registry)
    assert agent.run("Please review") == "done"
    first, first_tools = llm.requests[0]
    second, _ = llm.requests[1]
    assert [m["role"] for m in first] == ["system", "user", "user"]
    assert "review: Review code" in first[1]["content"]
    assert "PRIVATE SKILL BODY" not in str(first)
    assert first[-1] == {"role": "user", "content": "Please review"}
    assert any(t["function"]["name"] == "load_skill" for t in first_tools)
    assert "PRIVATE SKILL BODY" in str(second)
    assert [m["role"] for m in agent.context.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert all("PRIVATE SKILL BODY" not in str(m) for m in agent.context.messages)
    assert agent.context.messages[2]["tool_calls"][0]["id"] == agent.context.messages[3]["tool_call_id"]


def test_skill_budget_and_maximum_count(tmp_path):
    write_skill(tmp_path, "a", "a", body="X" * 200)
    write_skill(tmp_path, "b", "b")
    registry = SkillRegistry(tmp_path)
    llm = FakeLLM(LLMResponse(tool_calls=[skill_call("a")]), LLMResponse(content="done"))
    agent = Agent(llm, ToolRegistry(), skills=registry, max_skills=1, skill_char_budget=90)
    agent.run("question")
    assert [item["function"]["parameters"]["properties"]["name"]["enum"]
            for item in llm.requests[0][1] if item["function"]["name"] == "load_skill"] == [["a"]]
    assert "Skill content exceeds character budget" in agent.context.messages[2]["content"]
    assert "XXXX" not in str(llm.requests[1][0])


def test_invalid_skill_call_is_feedback_not_execution(tmp_path):
    write_skill(tmp_path, "safe", "safe")
    llm = FakeLLM(LLMResponse(tool_calls=[ToolCall(id="x", name="load_skill",
                                                     arguments='{"name":"../outside"}')]),
                  LLMResponse(content="recovered"))
    agent = Agent(llm, ToolRegistry(), skills=SkillRegistry(tmp_path))
    assert agent.run("load") == "recovered"
    assert "not available" in agent.context.messages[2]["content"]


def test_reserved_tool_name_and_invalid_skill_limits(tmp_path):
    write_skill(tmp_path, "safe", "safe")
    registry = SkillRegistry(tmp_path)
    tools = ToolRegistry()
    tools.register("load_skill", "collision", Args, lambda value: value)
    with pytest.raises(ValueError, match="reserved"):
        Agent(FakeLLM(), tools, skills=registry)
    with pytest.raises(ValueError, match="reserved"):
        AgentFactory(lambda: FakeLLM(), tools, skills=registry)
    with pytest.raises(ValueError, match="Skill limits"):
        Agent(FakeLLM(), ToolRegistry(), skills=registry, skill_char_budget=0)


class Args(BaseModel):
    value: int


def test_skills_work_with_tools_memory_and_multi_agent(tmp_path):
    write_skill(tmp_path, "math", "math", body="Use the double tool.")
    registry = SkillRegistry(tmp_path)
    tools = ToolRegistry()
    tools.register("double", "Double", Args, lambda value: value * 2)
    with MemoryStore() as memory:
        memory.save("shared", "calculate project value")
        llm = FakeLLM(LLMResponse(tool_calls=[skill_call("math")]),
                      LLMResponse(tool_calls=[ToolCall(id="tool-1", name="double",
                                                       arguments='{"value":2}')]),
                      LLMResponse(content="4"))
        agent = Agent(llm, tools, memory=memory, memory_namespace="shared", skills=registry)
        assert agent.run("calculate") == "4"
        assert "Retrieved saved notes" in str(llm.requests[0][0])
        assert "Use the double tool" in str(llm.requests[2][0])
        assert [m.get("tool_call_id") for m in agent.context.messages if m["role"] == "tool"] == ["skill-1", "tool-1"]
        factory = AgentFactory(lambda: FakeLLM(LLMResponse(content="ok")), tools,
                               skills=registry, memory=memory)
        parent = factory.create()
        child = factory.create("leaf", parent)
        assert child.agent.skills is registry
        assert parent.agent.memory_namespace != child.agent.memory_namespace


def test_disabled_skills_preserve_original_request(tmp_path):
    write_skill(tmp_path, "unused", "unused")
    llm = FakeLLM(LLMResponse(content="ok"))
    agent = Agent(llm, ToolRegistry())
    agent.run("hello")
    assert llm.requests == [([{"role": "user", "content": "hello"}], [])]
