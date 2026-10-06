"""OpenAI and OpenRouter upstreams over plain HTTP.

The gateway is a proxy: it forwards the client's request body (after the input guards) as-is, including fields
this code does not know about, so it talks HTTP with httpx instead of re-typing the body through an SDK.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from bulwark.providers.base import (
    FreeModelGuardError,
    JsonDict,
    ProviderError,
    RetryableError,
    ensure_free_models,
)

OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/gelevanog/llm-guardrails-firewall",
    "X-Title": "Bulwark",
}


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        kind: Literal["openai", "openrouter"],
        api_key: str | None,
        base_url: str,
        default_model: str,
        fallback_models: list[str] | None = None,
        require_free: bool = False,
        timeout_seconds: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(f"{'OPENROUTER' if kind == 'openrouter' else 'OPENAI'}_API_KEY is not set", 500)
        self.kind = kind
        self.default_model = default_model
        self.fallback_models = list(fallback_models or [])
        self.require_free = require_free and kind == "openrouter"
        if self.require_free:
            ensure_free_models([default_model, *self.fallback_models])
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", **(OPENROUTER_HEADERS if kind == "openrouter" else {})}
        self._timeout = timeout_seconds
        self._transport = transport

    @property
    def label(self) -> str:
        return f"{self.kind}/{self.default_model}"

    @property
    def is_remote(self) -> bool:
        return True

    def prepare(self, body: JsonDict) -> JsonDict:
        """Fill the default model and OpenRouter's `models` fallback list; enforce the free-only guard."""
        prepared = dict(body)
        model = prepared.get("model")
        if not model or model in {"auto", "default"}:
            prepared["model"] = self.default_model
        if self.kind == "openrouter":
            if self.fallback_models and "models" not in prepared and prepared["model"] == self.default_model:
                prepared["models"] = [self.default_model, *self.fallback_models]
            if self.require_free:
                ensure_free_models([prepared["model"], *prepared.get("models", [])])
        return prepared

    def _client(self) -> httpx.AsyncClient:
        # A client per call: providers are shared across event loops (gateway, CLI, worker threads).
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    def _check_served(self, served: object) -> None:
        if self.require_free and isinstance(served, str) and served and not served.endswith(":free"):
            # Defence in depth: never accept an answer that a paid model produced.
            raise FreeModelGuardError(f"OpenRouter served non-free model {served!r}; refusing the answer")

    async def complete(self, body: JsonDict) -> JsonDict:
        prepared = self.prepare({**body, "stream": False})
        async with self._client() as client:
            try:
                response = await client.post(self._url, json=prepared, headers=self._headers)
            except httpx.TimeoutException as exc:
                raise RetryableError(f"upstream timeout: {exc}", status_code=504) from exc
            except httpx.TransportError as exc:
                raise RetryableError(f"upstream connection error: {exc}") from exc
        data = _json_or_error(response)
        self._check_served(data.get("model"))
        choices = data.get("choices")
        if not choices:
            raise RetryableError("upstream returned no choices")
        message = choices[0].get("message") or {}
        if not message.get("content") and not message.get("tool_calls"):
            finish = choices[0].get("finish_reason")
            if finish == "content_filter":
                raise ProviderError("upstream content filter blocked the answer", 422)
            if finish == "length":
                # A reasoning model spent the whole budget thinking; the same request would fail the same way.
                raise ProviderError("empty answer: max_tokens reached before any output (raise max_tokens)", 502)
            raise RetryableError(f"empty answer (finish_reason={finish})")
        return data

    async def stream(self, body: JsonDict) -> AsyncIterator[JsonDict]:
        prepared = self.prepare({**body, "stream": True})
        async with (
            self._client() as client,
            client.stream("POST", self._url, json=prepared, headers=self._headers) as response,
        ):
            if response.status_code >= 400:
                await response.aread()
                _json_or_error(response)
            checked = False
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue  # blank keep-alives and ": OPENROUTER PROCESSING" comments
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk: JsonDict = json.loads(payload)
                if "error" in chunk:
                    raise ProviderError(f"upstream stream error: {_error_message(chunk)}")
                if not checked and chunk.get("model"):
                    self._check_served(chunk["model"])
                    checked = True
                yield chunk


def _error_message(data: JsonDict) -> str:
    error: Any = data.get("error", data)
    if isinstance(error, dict):
        metadata = error.get("metadata")
        raw = metadata.get("raw") if isinstance(metadata, dict) else None
        return str(raw or error.get("message") or error)[:300]
    return str(error)[:300]


def _json_or_error(response: httpx.Response) -> JsonDict:
    status = response.status_code
    try:
        data = response.json()
    except ValueError:
        data = {"error": {"message": response.text[:300]}}
    if not isinstance(data, dict):
        data = {"error": {"message": str(data)[:300]}}
    # OpenRouter may report an upstream failure inside a 200 body.
    error = data.get("error")
    if isinstance(error, dict) and status < 400:
        code = error.get("code")
        status = code if isinstance(code, int) and code >= 400 else 502
    if status < 400:
        return data
    message = _error_message(data)
    if status == 429:
        retry_after = response.headers.get("retry-after")
        try:
            delay = float(retry_after) if retry_after else None
        except ValueError:
            delay = None
        raise RetryableError(f"rate limited: {message}", retry_after=delay, status_code=429)
    if status >= 500 or status in {408, 409}:
        raise RetryableError(f"upstream {status}: {message}", status_code=502 if status >= 500 else status)
    raise ProviderError(f"upstream {status}: {message}", status)
