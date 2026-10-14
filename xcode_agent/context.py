"""Session history; trim only whole user turns, preserving tool call groups."""
from copy import deepcopy
from typing import Any


class ContextManager:
    def __init__(self, max_messages: int = 40, system_prompt: str | None = None):
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        self.max_messages = max_messages
        self.system_prompt = system_prompt
        self.reset()

    def reset(self) -> None:
        self._messages: list[dict[str, Any]] = []
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})

    @property
    def messages(self) -> list[dict[str, Any]]:
        return deepcopy(self._messages)

    def add(self, message: dict[str, Any]) -> None:
        self._messages.append(deepcopy(message))

    def trim(self) -> None:
        start = 1 if self.system_prompt else 0
        while len(self._messages) > self.max_messages:
            users = [i for i in range(start, len(self._messages)) if self._messages[i]["role"] == "user"]
            if len(users) < 2:
                break  # Keep the entire latest turn, even when it exceeds the soft limit.
            del self._messages[users[0]:users[1]]
