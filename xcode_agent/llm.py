"""Provider boundary. Tests can supply any object with a chat method."""
from typing import Any, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class LLMError(Exception):
    """Model transport or API failure after configured retries."""


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30, retries: int = 2):
        if not api_key or not model or not base_url:
            raise ValueError("api_key, base_url and model are required")
        if timeout <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries nonnegative")
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=retries)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        try:
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            return LLMResponse(
                content=message.content,
                tool_calls=[ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
                            for call in (message.tool_calls or [])],
            )
        except Exception as exc:
            raise LLMError(f"Model API request failed: {exc}") from exc
