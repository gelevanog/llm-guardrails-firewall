"""Budget-safe wrapper for real APIs: disk cache, throttle, retries with backoff, call budget, call ledger.

Used by the evaluation, the LLM judge, the agent demo and the playground's real-model mode. Every real request,
retries included, is appended to a JSONL ledger, so "how many API calls did this cost" is read from a
file, not remembered. The ledger stores model ids, status, latency and token counts; never prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import threading
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from bulwark.logging_config import get_logger
from bulwark.providers.base import BudgetExceededError, ChatProvider, JsonDict, ProviderError, RetryableError

log = get_logger(__name__)


class DiskCache:
    """One JSON file per request hash: re-running an eval replays answers instead of paying again."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key(label: str, body: JsonDict) -> str:
        material = json.dumps({"provider": label, "body": body}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> JsonDict | None:
        path = self._path(key)
        if not path.exists():
            return None
        data: JsonDict = json.loads(path.read_text(encoding="utf-8"))
        return data

    def put(self, key: str, response: JsonDict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")


class Throttle:
    """Spaces request starts at least `min_interval` seconds apart, across event loops and threads."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._next_start = 0.0
        self._lock = threading.Lock()

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:  # reserve a start slot; sleep outside the lock
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self.min_interval
        if start > now:
            await asyncio.sleep(start - now)


class CallLedger:
    """Counts real requests against a hard budget; one JSON line per request."""

    def __init__(self, path: Path | None, max_calls: int) -> None:
        self.path = path
        self.max_calls = max_calls
        self.calls = 0
        self._lock = threading.Lock()
        if path is not None and path.exists():
            with path.open(encoding="utf-8") as handle:
                self.calls = sum(1 for line in handle if line.strip())

    def reserve(self) -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceededError(f"call budget of {self.max_calls} real requests reached ({self.path})")
            self.calls += 1

    def record(self, **entry: object) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **entry}
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class ResilientProvider:
    def __init__(
        self,
        inner: ChatProvider,
        *,
        ledger: CallLedger | None = None,
        cache: DiskCache | None = None,
        throttle: Throttle | None = None,
        max_retries: int = 4,
        retry_base_seconds: float = 4.0,
        tag: str = "",
    ) -> None:
        self.inner = inner
        self.ledger = ledger or CallLedger(None, max_calls=10**9)
        self.cache = cache
        self.throttle = throttle
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.tag = tag

    @property
    def label(self) -> str:
        return self.inner.label

    @property
    def is_remote(self) -> bool:
        return self.inner.is_remote

    def with_tag(self, tag: str) -> ResilientProvider:
        """Same cache/budget/throttle, different ledger tag (e.g. "judge" vs "detector")."""
        return ResilientProvider(
            self.inner,
            ledger=self.ledger,
            cache=self.cache,
            throttle=self.throttle,
            max_retries=self.max_retries,
            retry_base_seconds=self.retry_base_seconds,
            tag=tag,
        )

    async def complete(self, body: JsonDict) -> JsonDict:
        key = DiskCache.key(self.inner.label, body) if self.cache else ""
        if self.cache and (hit := self.cache.get(key)) is not None:
            return {**hit, "_cached": True}
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            await self._before_request()
            started = time.monotonic()
            try:
                response = await self.inner.complete(body)
            except RetryableError as exc:
                last_error = exc
                self._record(body, "retryable_error", started, error=str(exc))
            except ProviderError as exc:
                self._record(body, "error", started, error=str(exc))
                raise
            else:
                self._record(body, "ok", started, response=response)
                if self.cache:
                    self.cache.put(key, response)
                return response
            if attempt < self.max_retries:
                delay = self._backoff(attempt, last_error)
                log.warning(
                    "llm.retry", tag=self.tag, attempt=attempt + 1, delay=round(delay, 1), error=str(last_error)[:160]
                )
                await asyncio.sleep(delay)
        raise last_error or ProviderError("request failed")

    async def stream(self, body: JsonDict) -> AsyncIterator[JsonDict]:
        # Retries are only safe before the first chunk reached the client.
        for attempt in range(self.max_retries + 1):
            await self._before_request()
            started = time.monotonic()
            emitted = False
            try:
                async for chunk in self.inner.stream(body):
                    emitted = True
                    yield chunk
            except RetryableError as exc:
                self._record(body, "retryable_error", started, error=str(exc))
                if emitted or attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self._backoff(attempt, exc))
                continue
            except ProviderError as exc:
                self._record(body, "error", started, error=str(exc))
                raise
            self._record(body, "ok", started, stream=True)
            return

    async def _before_request(self) -> None:
        if self.inner.is_remote:
            self.ledger.reserve()
            if self.throttle:
                await self.throttle.wait()

    def _backoff(self, attempt: int, error: ProviderError | None) -> float:
        if not self.inner.is_remote:
            return 0.0
        retry_after = error.retry_after if isinstance(error, RetryableError) else None
        if retry_after:
            return min(retry_after, 60.0)
        return float(min(self.retry_base_seconds * 2.0**attempt, 60.0) * (0.75 + random.random() / 2))

    def _record(
        self,
        body: JsonDict,
        status: str,
        started: float,
        *,
        error: str | None = None,
        response: JsonDict | None = None,
        stream: bool = False,
    ) -> None:
        if not self.inner.is_remote:
            return
        usage = (response or {}).get("usage") or {}
        self.ledger.record(
            tag=self.tag,
            provider=self.inner.label,
            requested_model=body.get("model"),
            served_model=(response or {}).get("model"),
            status=status,
            stream=stream,
            latency_s=round(time.monotonic() - started, 2),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            error=error[:300] if error else None,
        )
