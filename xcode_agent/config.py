"""Environment configuration, loaded at application startup."""
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    api_key: str
    base_url: str
    model: str
    workspace: Path
    max_rounds: int = Field(default=8, ge=1)
    history_limit: int = Field(default=40, ge=1)
    timeout: float = Field(default=30, gt=0)
    retries: int = Field(default=2, ge=0)
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", ""),
            workspace=Path(os.getenv("AGENT_WORKSPACE", ".")),
            max_rounds=int(os.getenv("AGENT_MAX_ROUNDS", "8")),
            history_limit=int(os.getenv("AGENT_HISTORY_LIMIT", "40")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "30")),
            retries=int(os.getenv("OPENAI_RETRIES", "2")),
            verbose=os.getenv("AGENT_VERBOSE", "true").lower() in {"1", "true", "yes"},
        )
