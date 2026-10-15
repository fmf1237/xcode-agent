"""Stable error codes and conservative LLM failure classification."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math

from openai import APIStatusError, APITimeoutError, RateLimitError


class AgentError(Exception):
    error_code = "agent_error"
    retryable = False


class AgentLLMError(AgentError):
    error_code = "llm_error"


class LLMTimeoutError(AgentLLMError):
    error_code = "llm_timeout"
    retryable = True


class LLMRateLimitError(AgentLLMError):
    error_code = "llm_rate_limit"
    retryable = True

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMTransientError(AgentLLMError):
    error_code = "llm_transient"
    retryable = True


class ToolArgumentError(AgentError):
    error_code = "tool_argument_error"


class ToolExecutionError(AgentError):
    error_code = "tool_execution_error"


class AgentTimeoutError(AgentError):
    error_code = "agent_timeout"


class AgentCancelled(AgentError):
    error_code = "agent_cancelled"


class DependencyFailedError(AgentError):
    error_code = "dependency_failed"


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Parse Retry-After seconds or an HTTP date; malformed values are ignored."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                return None
            seconds = (target - (now or datetime.now(timezone.utc))).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def classify_llm_exception(exc: Exception) -> AgentLLMError:
    """Use SDK types/status codes only; unknown failures remain non-retryable."""
    if isinstance(exc, AgentLLMError):
        return exc
    if isinstance(exc, APITimeoutError):
        return LLMTimeoutError(str(exc))
    if isinstance(exc, RateLimitError):
        return LLMRateLimitError(str(exc), parse_retry_after(exc.response.headers.get("retry-after")))
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            return LLMRateLimitError(str(exc), parse_retry_after(exc.response.headers.get("retry-after")))
        if exc.status_code == 408:
            return LLMTimeoutError(str(exc))
        if exc.status_code in (500, 502, 503, 504):
            return LLMTransientError(str(exc))
    return AgentLLMError(str(exc))
