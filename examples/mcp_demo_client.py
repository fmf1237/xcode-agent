"""Offline Agent + MCP demo. Run: python examples/mcp_demo_client.py"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xcode_agent import (Agent, LLMResponse, MCPClient, MCPServerConfig,
                         ToolCall, ToolRegistry)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            print("Discovered:", [item["function"]["name"] for item in tools])
            return LLMResponse(tool_calls=[ToolCall(id="demo-add", name="mcp__demo__add",
                                                    arguments='{"a":3,"b":5}')])
        print("MCP result:", messages[-1]["content"])
        return LLMResponse(content="3 + 5 = 8")


def main() -> None:
    server = Path(__file__).with_name("mcp_demo_server.py")
    config = MCPServerConfig(server_name="demo", command=sys.executable,
                             args=(str(server),), allowed_tools=frozenset({"add", "echo"}))
    with MCPClient(config) as client:
        registry = ToolRegistry()
        client.register_tools(registry)
        agent = Agent(FakeLLM(), registry)
        print("Answer:", agent.run("Calculate 3 + 5"))
    print("MCP connection closed")


if __name__ == "__main__":
    main()
