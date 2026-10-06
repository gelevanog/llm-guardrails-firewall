"""Taint tracking for agents: what entered the conversation, from whom, and where tool arguments came from.

Two levels of taint, like a browser's "this page is not secure":
  * tainted:    untrusted content (a tool result, an email, a retrieved chunk) is in the context. No detector
                is complete, so from now on the model may be following someone else's instructions; risky
                tools need approval.
  * suspicious: some of that content was flagged by a detector; risky tools are blocked.

Argument provenance is the sharper signal. When the model calls `send_email(to="x@evil.test")`, the
tracker checks where "x@evil.test" came from. If the user or the system prompt mentioned it, it is the
user's intent. If it only appears inside an email the agent read, an outsider chose the recipient.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from bulwark.core import digest

TaintLevel = Literal["clean", "tainted", "suspicious"]
Origin = Literal["trusted", "untrusted", "unknown"]

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
_ACCOUNT = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b|\b\d{9,18}\b")
_RECIPIENT_FIELDS = re.compile(
    r"(?:^|_)(?:to|cc|bcc|recipients?|email|address|url|uri|endpoint|webhook|domain|host|iban|account)(?:_|$)", re.I
)


@dataclass(frozen=True)
class TaintSource:
    source: str
    flagged: bool
    score: float
    digest: str


@dataclass
class TaintState:
    sources: list[TaintSource] = field(default_factory=list)
    _trusted: list[str] = field(default_factory=list, repr=False)
    _untrusted: list[str] = field(default_factory=list, repr=False)

    @property
    def level(self) -> TaintLevel:
        if any(source.flagged for source in self.sources):
            return "suspicious"
        return "tainted" if self.sources else "clean"

    def add_trusted(self, text: str) -> None:
        """System prompt, the user's messages, and metadata the application vouches for (e.g. mail headers)."""
        if text:
            self._trusted.append(text.lower())

    def add_untrusted(self, text: str, source: str, *, flagged: bool = False, score: float = 0.0) -> None:
        self.sources.append(TaintSource(source=source, flagged=flagged, score=round(score, 4), digest=digest(text)))
        if text:
            self._untrusted.append(text.lower())

    def origin(self, value: str) -> Origin:
        needle = value.strip().lower()
        if len(needle) < 4:
            return "unknown"
        if any(needle in text for text in self._trusted):
            return "trusted"
        if any(needle in text for text in self._untrusted):
            return "untrusted"
        return "unknown"

    def untrusted_arguments(self, arguments: dict[str, Any]) -> list[tuple[str, str]]:
        """(argument path, value) pairs whose value only appears in untrusted content."""
        return [(path, atom) for path, atom in argument_atoms(arguments) if self.origin(atom) == "untrusted"]


def argument_atoms(arguments: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Values in tool arguments that say *where* data goes: addresses, URLs, account numbers, recipient fields."""
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            yield from argument_atoms(value, f"{path}.{key}" if path else str(key))
    elif isinstance(arguments, list):
        for index, value in enumerate(arguments):
            yield from argument_atoms(value, f"{path}[{index}]")
    elif isinstance(arguments, str):
        found = {m.group(0).rstrip(".,;") for m in _EMAIL.finditer(arguments)}
        found |= {m.group(0).rstrip(".,;") for m in _URL.finditer(arguments)}
        found |= {m.group(0) for m in _ACCOUNT.finditer(arguments)}
        leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if _RECIPIENT_FIELDS.search(leaf) and len(arguments.strip()) <= 120:
            found.add(arguments.strip())
        for atom in sorted(found):
            if len(atom) >= 4:  # shorter values cannot identify a destination
                yield path, atom
