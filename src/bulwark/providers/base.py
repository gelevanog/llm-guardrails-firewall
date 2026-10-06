"""Upstream provider interface: OpenAI chat.completions request body in, OpenAI response (or chunks) out.

Every provider speaks the same wire shape as the gateway's public API, so the guards only know one
format; the Anthropic provider converts at its edge.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any, Protocol

JsonDict = dict[str, Any]


class ProviderError(RuntimeError):
    """The upstream request failed and should not be retried. `status_code` is passed to the client."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class RetryableError(ProviderError):
    """Rate limit, overload, timeout or an empty answer: worth retrying with backoff."""

    def __init__(self, message: str, retry_after: float | None = None, status_code: int = 503) -> None:
        super().__init__(message, status_code)
        self.retry_after = retry_after


class FreeModelGuardError(ProviderError):
    """A non-`:free` model id was about to be sent to OpenRouter while the free-only guard is on."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class BudgetExceededError(ProviderError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=429)


def ensure_free_models(model_ids: Iterable[str]) -> None:
    """Refuse any model id that is not an OpenRouter free variant (suffix `:free`)."""
    paid = [model for model in model_ids if not str(model).endswith(":free")]
    if paid:
        raise FreeModelGuardError(
            f"free-only guard: refusing non-free OpenRouter model id(s): {', '.join(paid)} "
            "(set BULWARK_REQUIRE_FREE_MODELS=false to allow paid models)"
        )


class ChatProvider(Protocol):
    @property
    def label(self) -> str:
        """Provider name and default model, e.g. "openrouter/nvidia/nemotron-3-super-120b-a12b:free"."""
        ...

    @property
    def is_remote(self) -> bool:
        """True for real APIs (throttled, cached and counted in the call ledger during evals)."""
        ...

    async def complete(self, body: JsonDict) -> JsonDict:
        """Non-streaming chat completion; returns an OpenAI `chat.completion` object."""
        ...

    def stream(self, body: JsonDict) -> AsyncIterator[JsonDict]:
        """Streaming chat completion; yields OpenAI `chat.completion.chunk` objects."""
        ...


def message_text(message: JsonDict) -> str:
    """Plain text of an OpenAI message whose content is a string or a list of text parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""
