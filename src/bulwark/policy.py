"""Policies: which guards run on a route, their thresholds and actions, and the agent's tool rules.

A policy is a YAML file; a deployment ships several (one per route or tenant) and picks one per request.
Unknown keys are rejected, so a typo in a policy fails at startup instead of silently weakening it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bulwark.core import Action


class PolicyError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Layers(_Strict):
    heuristics: bool = True
    classifier: bool = True
    judge: bool = False
    """LLM judge for borderline scores (a real API call; off by default)."""
    combine: Literal["max", "corroborate"] = "corroborate"
    """How heuristics and classifier scores combine. `max`: either layer alone can flag and block.
    `corroborate`: the classifier raises a score only when the rules already saw something (heuristics score
    >= `corroborate_min`); on its own it stops just under the flag threshold, where the judge (if enabled)
    decides. Chosen by measurement: the classifier's false positives on business email (see README)."""
    corroborate_min: float = Field(default=0.25, ge=0.0, le=1.0)
    classifier_alone_cap: float = Field(default=0.45, ge=0.0, le=1.0)


class InjectionConfig(_Strict):
    enabled: bool = True
    layers: Layers = Field(default_factory=lambda: Layers(combine="max"))
    """User input: the classifier may flag on its own (it adds recall on direct attacks at a low false-positive
    rate). Untrusted content defaults to `corroborate` (see `Layers.combine`)."""
    flag_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    block_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    judge_band: tuple[float, float] = (0.3, 0.85)
    """Scores in [low, high) are sent to the judge when it is enabled; it can confirm or dismiss them."""
    action: Action = Action.BLOCK
    """Action at or above `block_threshold`."""
    borderline_action: Action = Action.FLAG
    """Action between `flag_threshold` and `block_threshold`."""

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.flag_threshold > self.block_threshold:
            raise ValueError("flag_threshold must not exceed block_threshold")
        return self


class TopicConfig(_Strict):
    enabled: bool = False
    allowed: list[str] = Field(default_factory=list)
    """If set, a request must match at least one of these topics (short greetings always pass)."""
    denied: list[str] = Field(default_factory=list)
    definitions: dict[str, list[str]] = Field(default_factory=dict)
    """Topic name -> regular expressions (case-insensitive) that indicate it."""
    action: Action = Action.BLOCK
    off_topic_action: Action = Action.BLOCK
    min_words_for_off_topic: int = 4

    @model_validator(mode="after")
    def _known_topics(self) -> Self:
        for name in [*self.allowed, *self.denied]:
            if name not in self.definitions:
                raise ValueError(f"topic {name!r} has no definition")
        for name, patterns in self.definitions.items():
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"invalid pattern in topic {name!r}: {exc}") from exc
        return self


class InputConfig(_Strict):
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    topics: TopicConfig = Field(default_factory=TopicConfig)


class Spotlight(StrEnum):
    OFF = "off"
    DELIMIT = "delimit"
    """Wrap in random, unforgeable boundary markers and tell the model the content is data."""
    DATAMARK = "datamark"
    """Interleave a marker character between words (Hines et al., 2024), so instructions read as data."""
    ENCODE = "encode"
    """Base64-encode the content (strong isolation; only capable models can still use the content)."""


class UntrustedAction(StrEnum):
    QUARANTINE = "quarantine"
    """Remove the flagged spans and leave a visible note saying so."""
    STRIP = "strip"
    """Remove the flagged spans silently."""
    FLAG = "flag"
    BLOCK = "block"
    """Drop the whole document and tell the model it was withheld."""


class UntrustedConfig(_Strict):
    enabled: bool = True
    layers: Layers = Field(default_factory=Layers)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    on_detection: UntrustedAction = UntrustedAction.QUARANTINE
    strip_hidden: bool = True
    """Always drop hidden HTML text, invisible characters and their payloads before the model sees them."""
    spotlight: Spotlight = Spotlight.DELIMIT
    untrusted_roles: list[str] = Field(default_factory=lambda: ["tool"])
    """Chat roles whose content is treated as untrusted by the gateway."""
    taint: bool = True
    """Untrusted content taints the session (see tools.on_taint)."""


class LeakageConfig(_Strict):
    enabled: bool = True
    canary: bool = True
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    """Share of the system prompt's 6-word shingles found in the answer."""
    min_verbatim_words: int = 15
    """A run of this many consecutive system-prompt words in the answer also counts as a leak."""
    action: Action = Action.BLOCK


class ExfiltrationConfig(_Strict):
    enabled: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    """Domains (and their subdomains) that answers may link to and load images from."""
    action: Action = Action.SANITIZE
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    block_unknown_images: bool = True
    """Any image from a domain outside the allow-list is removed (images load without a click)."""


class SecretsConfig(_Strict):
    enabled: bool = True
    action: Action = Action.SANITIZE
    pii_shield: bool = False
    """Also send answers to a PII Shield gateway (BULWARK_PII_SHIELD_URL) and mask what it finds."""


class OutputConfig(_Strict):
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    exfiltration: ExfiltrationConfig = Field(default_factory=ExfiltrationConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)


class Risk(StrEnum):
    READ = "read"
    """Reads data; no side effects (search, lookup)."""
    WRITE = "write"
    """Changes internal state (create a ticket, update a record)."""
    EXTERNAL = "external"
    """Acts outside the system or moves data out (send email, HTTP request, payment)."""


class DomainRule(_Strict):
    fields: list[str]
    """Argument names holding email addresses or URLs (strings or lists of strings)."""
    domains: list[str]
    action: Action = Action.REQUIRE_APPROVAL
    """Action when a value is outside these domains."""


class ToolRule(_Strict):
    risk: Risk = Risk.READ
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    """JSON Schema the arguments must satisfy."""
    on_schema_violation: Action = Action.BLOCK
    allowed_domains: DomainRule | None = None
    deny_patterns: dict[str, list[str]] = Field(default_factory=dict)
    """Argument name -> regular expressions that must not match its value."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TaintActions(_Strict):
    read: Action = Action.ALLOW
    write: Action = Action.ALLOW
    external: Action = Action.REQUIRE_APPROVAL

    def for_risk(self, risk: Risk) -> Action:
        action: Action = getattr(self, risk.value)
        return action


class ToolsConfig(_Strict):
    enabled: bool = True
    default: Action = Action.BLOCK
    """Action for a tool that is not in `allowed`."""
    allowed: dict[str, ToolRule] = Field(default_factory=dict)
    on_taint: TaintActions = Field(default_factory=TaintActions)
    """After untrusted content entered the conversation."""
    on_suspicious: TaintActions = Field(
        default_factory=lambda: TaintActions(read=Action.ALLOW, write=Action.REQUIRE_APPROVAL, external=Action.BLOCK)
    )
    """After untrusted content that a detector flagged entered the conversation."""
    untrusted_arguments: TaintActions = Field(
        default_factory=lambda: TaintActions(read=Action.ALLOW, write=Action.REQUIRE_APPROVAL, external=Action.BLOCK)
    )
    """An argument value (address, URL, number) that only appears in untrusted content, not in the
    user's or system's messages: the data flow an attacker controls."""


class Policy(_Strict):
    name: str
    description: str = ""
    mode: Literal["enforce", "monitor"] = "enforce"
    """monitor = evaluate and audit everything, change nothing (shadow mode)."""
    fail_mode: Literal["closed", "open"] = "closed"
    """closed: a layer that errors counts as a detection; open: continue with the remaining layers."""
    block_response: Literal["error", "refusal"] = "error"
    """Blocked input: an OpenAI-style HTTP 400 error, or a normal answer with `refusal_message`."""
    refusal_message: str = "Sorry, I can't help with that request."
    input: InputConfig = Field(default_factory=InputConfig)
    untrusted: UntrustedConfig = Field(default_factory=UntrustedConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def enforced(self) -> bool:
        return self.mode == "enforce"

    def uses_classifier(self) -> bool:
        return (self.input.injection.enabled and self.input.injection.layers.classifier) or (
            self.untrusted.enabled and self.untrusted.layers.classifier
        )

    def uses_judge(self) -> bool:
        return (self.input.injection.enabled and self.input.injection.layers.judge) or (
            self.untrusted.enabled and self.untrusted.layers.judge
        )


def load_policy(path: Path) -> Policy:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return Policy.model_validate(raw)
    except (yaml.YAMLError, ValueError) as exc:
        raise PolicyError(f"invalid policy {path.name}: {exc}") from exc


class PolicySet:
    def __init__(self, policies: list[Policy], default: str) -> None:
        self._policies = {policy.name: policy for policy in policies}
        if default not in self._policies:
            raise PolicyError(f"default policy {default!r} not found (have: {', '.join(self._policies)})")
        self.default_name = default

    @classmethod
    def from_dir(cls, directory: Path, default: str) -> PolicySet:
        files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
        if not files:
            raise PolicyError(f"no policy files in {directory}")
        return cls([load_policy(path) for path in files], default)

    @property
    def names(self) -> list[str]:
        return list(self._policies)

    def get(self, name: str | None = None) -> Policy:
        key = name or self.default_name
        try:
            return self._policies[key]
        except KeyError:
            raise PolicyError(f"unknown policy {key!r} (have: {', '.join(self._policies)})") from None

    def add(self, policy: Policy) -> None:
        self._policies[policy.name] = policy

    def __iter__(self) -> Iterator[Policy]:
        return iter(self._policies.values())
