"""Audit log of guard decisions: scores, actions, rules and reasons, never full prompts.

Each record holds the decisions of one request (or agent step) per stage, keyed hashes of the request
and of tool arguments (to correlate an incident with a request without storing it), and at most a few
short snippets of the text that triggered a guard, with email addresses, long numbers and URL query
strings masked. That is enough to review why something was blocked and too little to reconstruct the
conversation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import uuid
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bulwark.core import Decision
from bulwark.logging_config import get_logger

log = get_logger(__name__)

_EMAIL = re.compile(r"([\w.+-])[\w.+-]*@([\w-]+(?:\.[\w-]+)+)")
_DIGITS = re.compile(r"\d{4,}")
_QUERY = re.compile(r"(https?://[^\s?#]+)[?#]\S*")
_SECRETISH = re.compile(r"\b(?:sk-|ghp_|AKIA|AIza|xox|eyJ)[\w-]{6,}")
SNIPPET_CHARS = 80


def redact_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    text = _SECRETISH.sub("[secret]", text)
    text = _QUERY.sub(r"\1?…", text)
    text = _EMAIL.sub(r"\1…@\2", text)
    text = _DIGITS.sub(lambda m: "#" * len(m.group(0)), text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class GuardEntry(BaseModel):
    guard: str
    score: float
    action: str
    rules: list[str] = Field(default_factory=list)
    layer_scores: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""
    error: str | None = None


class StageEntry(BaseModel):
    stage: str
    action: str
    enforced: bool
    latency_ms: float
    guards: list[GuardEntry] = Field(default_factory=list)
    """Only guards that fired (the rest are summarized by the stage's action)."""
    source: str | None = None
    tool: str | None = None
    arguments_hash: str | None = None
    approval_id: str | None = None


class AuditRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))
    route: str
    policy: str
    tenant: str | None = None
    request_hash: str = ""
    action: str = "allow"
    """The strictest enforced action across stages."""
    blocked: bool = False
    stages: list[StageEntry] = Field(default_factory=list)
    snippets: list[str] = Field(default_factory=list)
    taint: str = "clean"
    untrusted_sources: int = 0
    upstream: str | None = None
    upstream_model: str | None = None
    stream: bool = False
    status: int = 200
    guard_ms: float = 0.0
    upstream_ms: float | None = None
    total_ms: float = 0.0


_ORDER = ["allow", "log_only", "flag", "sanitize", "require_approval", "block"]


class AuditLog:
    def __init__(self, key: bytes, max_entries: int = 2000, path: Path | None = None) -> None:
        self._key = key
        self._records: deque[AuditRecord] = deque(maxlen=max_entries)
        self._path = path
        self._lock = threading.Lock()

    def digest(self, data: bytes | str) -> str:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        return hmac.new(self._key, raw, hashlib.sha256).hexdigest()[:24]

    def new_record(self, *, route: str, policy: str, request: bytes | str, tenant: str | None = None) -> AuditRecord:
        return AuditRecord(route=route, policy=policy, tenant=tenant, request_hash=self.digest(request))

    def add_decision(
        self,
        record: AuditRecord,
        decision: Decision,
        *,
        text: str | None = None,
        source: str | None = None,
        tool: str | None = None,
        arguments: Any = None,
    ) -> None:
        """Append one stage; keep a redacted snippet of what fired (at most three per record)."""
        entry = StageEntry(
            stage=decision.stage,
            action=decision.action.value,
            enforced=decision.enforced,
            latency_ms=decision.latency_ms,
            source=source,
            tool=tool,
            arguments_hash=self.digest(json.dumps(arguments, sort_keys=True, default=str)) if arguments else None,
            approval_id=decision.approval.id if decision.approval else None,
            guards=[
                GuardEntry(
                    guard=r.guard,
                    score=round(r.score, 4),
                    action=r.action.value,
                    rules=sorted({f.rule for f in r.findings}),
                    layer_scores=r.layer_scores,
                    explanation=redact_snippet(r.explanation, 240),
                    error=r.error[:200] if r.error else None,
                )
                for r in decision.results
                if r.triggered or r.error
            ],
        )
        record.stages.append(entry)
        record.guard_ms = round(record.guard_ms + decision.latency_ms, 2)
        if decision.enforced and _ORDER.index(decision.action.value) > _ORDER.index(record.action):
            record.action = decision.action.value
        record.blocked = record.blocked or decision.blocked
        if text and decision.triggered and len(record.snippets) < 3:
            for result in decision.triggered:
                located = [f for f in result.findings if f.start is not None and f.end is not None]
                if located and located[0].start is not None and located[0].end is not None:
                    snippet = text[max(0, located[0].start - 10) : located[0].end + 10]
                else:
                    snippet = text[:SNIPPET_CHARS]
                record.snippets.append(redact_snippet(snippet))
                break

    def add(self, record: AuditRecord) -> None:
        with self._lock:
            self._records.append(record)
            if self._path is None:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(record.model_dump_json() + "\n")
            except OSError as exc:  # a full or read-only disk must not fail the user's request
                log.error("audit.write_failed", path=str(self._path), error=type(exc).__name__)

    def preload(self, records: list[AuditRecord]) -> None:
        with self._lock:
            self._records.extend(records)

    def recent(self, limit: int = 100) -> list[AuditRecord]:
        with self._lock:
            return list(self._records)[-limit:][::-1]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        guards: Counter[str] = Counter()
        for record in records:
            for stage in record.stages:
                guards.update(g.guard for g in stage.guards if g.action not in {"allow", "log_only"})
        return {
            "requests": len(records),
            "blocked": sum(r.blocked for r in records),
            "actions": dict(Counter(r.action for r in records)),
            "guards": dict(guards.most_common()),
            "policies": dict(Counter(r.policy for r in records)),
        }

    @staticmethod
    def load_jsonl(path: Path) -> list[AuditRecord]:
        if not path.exists():
            return []
        return [AuditRecord.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
