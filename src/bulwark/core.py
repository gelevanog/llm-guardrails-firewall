"""Shared types: actions, findings, guard results and decisions.

Every guard returns a `GuardResult` with a score in [0, 1], the action the policy chose for that score,
and a human-readable explanation built from `Finding`s (which rule or model fired, where, and why).
A `Decision` combines the guard results of one stage (input, untrusted content, output, tool call).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Action(StrEnum):
    """What happens when a guard fires. Ordered from least to most restrictive."""

    ALLOW = "allow"
    LOG_ONLY = "log_only"
    """Evaluate and audit, never change anything (shadow mode for rollouts)."""
    FLAG = "flag"
    """Let it through, mark it in the response headers and the audit log."""
    SANITIZE = "sanitize"
    """Remove or neutralize the offending part (a link, a hidden instruction, a secret) and continue."""
    REQUIRE_APPROVAL = "require_approval"
    """Hold a tool call until a human approves it."""
    BLOCK = "block"
    """Refuse: the request, the tool call or the answer does not go through."""

    @property
    def severity(self) -> int:
        return _SEVERITY[self]


_SEVERITY = {
    Action.ALLOW: 0,
    Action.LOG_ONLY: 1,
    Action.FLAG: 2,
    Action.SANITIZE: 3,
    Action.REQUIRE_APPROVAL: 4,
    Action.BLOCK: 5,
}


def strictest(actions: list[Action]) -> Action:
    return max(actions, key=lambda action: action.severity, default=Action.ALLOW)


Stage = Literal["input", "untrusted", "output", "tool_call"]


class Finding(BaseModel):
    """One piece of evidence: a rule match, a model score, a suspicious URL, a canary."""

    rule: str
    """Machine-readable id, e.g. `override.ignore_instructions` or `classifier`."""
    layer: str
    """heuristics | classifier | judge | html | canary | similarity | url | secret | schema | taint | ..."""
    score: float = Field(ge=0.0, le=1.0)
    message: str
    """Plain-English reason, safe to show to an operator."""
    start: int | None = None
    end: int | None = None
    """Character offsets in the original text, when the evidence can be located."""
    snippet: str = ""
    """The matched text (shown in the playground; the audit log stores a redacted, shortened form)."""
    view: str = "raw"
    """Which view of the text matched: raw, normalized, base64, hex, rot13, tag-chars, html-hidden, ..."""


class GuardResult(BaseModel):
    guard: str
    """injection | topic | untrusted | leakage | exfiltration | secrets | tool_call"""
    score: float = Field(ge=0.0, le=1.0)
    triggered: bool
    action: Action = Action.ALLOW
    explanation: str = ""
    findings: list[Finding] = Field(default_factory=list)
    layer_scores: dict[str, float] = Field(default_factory=dict)
    """Score per detection layer (heuristics, classifier, judge) for layered guards."""
    latency_ms: float = 0.0
    error: str | None = None
    """A layer failed; under a fail-closed policy the result is treated as triggered."""


class ApprovalRequest(BaseModel):
    """A tool call held for a human decision."""

    id: str = Field(default_factory=lambda: "apr_" + uuid.uuid4().hex[:16])
    tool: str
    arguments: dict[str, Any]
    reasons: list[str]
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    status: Literal["pending", "approved", "denied"] = "pending"


class Decision(BaseModel):
    """The combined result of one stage: the strictest action wins."""

    stage: Stage
    policy: str
    action: Action
    results: list[GuardResult] = Field(default_factory=list)
    text: str | None = None
    """The text to use downstream: sanitized, quarantined or spotlighted (None when nothing changed)."""
    approval: ApprovalRequest | None = None
    latency_ms: float = 0.0
    enforced: bool = True
    """False when the policy runs in monitor mode: the action is recorded but not applied."""

    @property
    def blocked(self) -> bool:
        return self.enforced and self.action is Action.BLOCK

    @property
    def needs_approval(self) -> bool:
        return self.enforced and self.action is Action.REQUIRE_APPROVAL

    @property
    def triggered(self) -> list[GuardResult]:
        return [result for result in self.results if result.triggered]

    @property
    def score(self) -> float:
        return max((result.score for result in self.results), default=0.0)

    def explain(self) -> str:
        """One line per guard that fired, e.g. for an error message or a log."""
        lines = [f"{r.guard} ({r.score:.2f}, {r.action.value}): {r.explanation}" for r in self.triggered]
        return "; ".join(lines) or "no guard fired"

    def summary(self) -> dict[str, Any]:
        """Compact, text-free view for API responses and headers."""
        return {
            "stage": self.stage,
            "policy": self.policy,
            "action": self.action.value,
            "enforced": self.enforced,
            "score": round(self.score, 3),
            "guards": [
                {
                    "guard": r.guard,
                    "score": round(r.score, 3),
                    "action": r.action.value,
                    "rules": sorted({f.rule for f in r.findings}),
                    "explanation": r.explanation,
                }
                for r in self.triggered
            ],
            **({"approval": self.approval.model_dump()} if self.approval else {}),
        }


def digest(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
