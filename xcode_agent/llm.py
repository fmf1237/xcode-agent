"""Provider boundary. Tests can supply any object with a chat method."""
from typing import Any, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field

from .errors import AgentLLMError, classify_llm_exception


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


LLMError = AgentLLMError  # Keep the old import name for existing callers.


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30, retries: int = 2):
        if not api_key or not model or not base_url:
            raise ValueError("api_key, base_url and model are required")
        if timeout <= 0 or retries < 0:
            raise ValueError("timeout must be positive and retries nonnegative")
        self.model = model
        self.streaming = False  # chat() returns only complete, non-stream responses.
        self.max_attempts = retries + 1  # retries retains its existing public meaning.
        # The Agent owns all retries so their count, cancellation and Trace stay observable.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

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
            classified = classify_llm_exception(exc)
            if classified is exc:
                raise
            raise classified from exc
