"""Deterministic offline demonstration: python examples/fake_demo.py"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xcode_agent import Agent, LLMResponse, ToolCall, ToolRegistry
from xcode_agent.file_tools import register_file_tools


class FakeLLM:
    def __init__(self):
        self.step = 0

    def chat(self, messages, tools):
        self.step += 1
        if self.step == 1:
            return LLMResponse(tool_calls=[ToolCall(id="demo-1", name="read_file",
                                                    arguments='{"path":"examples/demo.txt"}')])
        result = json.loads(messages[-1]["content"])
        return LLMResponse(content="File content: " + result["content"].strip())


registry = ToolRegistry()
register_file_tools(registry, Path(__file__).resolve().parents[1])
agent = Agent(FakeLLM(), registry, logger=print)
print(agent.run("Read the demo file"))
