"""Synchronous, reusable agent loop."""
from typing import Callable

from .context import ContextManager
from .llm import ChatClient, LLMError
from .tools import ToolRegistry


class AgentLLMError(Exception):
    """The model request failed; no tool result can repair it."""


class MaxRoundsExceeded(Exception):
    """The model did not produce a final answer within the round budget."""


class Agent:
    def __init__(self, llm: ChatClient, tools: ToolRegistry, context: ContextManager | None = None,
                 max_rounds: int = 8, logger: Callable[[str], None] | None = None):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.llm, self.tools = llm, tools
        self.context = context if context is not None else ContextManager()
        self.max_rounds, self.logger = max_rounds, logger

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

    def run(self, user_input: str) -> str:
        self.context.add({"role": "user", "content": user_input})
        self.context.trim()
        for round_number in range(1, self.max_rounds + 1):
            self._log(f"Round {round_number}: requesting model")
            try:
                response = self.llm.chat(self.context.messages, self.tools.schemas())
            except Exception as exc:
                raise AgentLLMError(str(exc)) from exc
            if response.tool_calls:
                calls = [{"id": call.id, "type": "function", "function": {"name": call.name,
                          "arguments": call.arguments}} for call in response.tool_calls]
                self.context.add({"role": "assistant", "content": response.content, "tool_calls": calls})
                for call in response.tool_calls:
                    self._log(f"Tool {call.name} ({call.id})")
                    result = self.tools.execute(call.name, call.arguments)
                    self.context.add({"role": "tool", "tool_call_id": call.id, "content": result.as_message()})
                    self._log(f"Tool result ({call.id}): {result.as_message()}")
                self.context.trim()
                continue
            answer = response.content or ""
            self.context.add({"role": "assistant", "content": answer})
            self.context.trim()
            self._log(f"Final answer: {answer}")
            return answer
        raise MaxRoundsExceeded(f"No final answer after {self.max_rounds} model rounds")
