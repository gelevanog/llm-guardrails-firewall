"""Output guards on a token stream: sanitize or stop mid-stream without waiting for the whole answer.

Text is released in pieces that are safe to judge on their own. The guard holds back:
  * any markdown link or image, HTML tag or bare URL that has started but not finished, so a URL is checked
    whole before any of it reaches the user;
  * the last few characters and the current word, so a secret or the canary is never released half-way.
Each releasable piece is checked for exfiltration URLs and secrets (sanitized in place) and, together with
everything released before, for system-prompt leakage. A blocking verdict stops the stream with a notice;
what was already released stays released (the hold-back keeps that to text that was safe at the time).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bulwark.core import Action, Decision, GuardResult, strictest
from bulwark.firewall import Firewall
from bulwark.policy import Policy

_OPENERS = re.compile(r"!?\[|<(?:img|a)\b|https?://", re.IGNORECASE)


def _closed(rest: str) -> bool:
    """Has the construct starting at `rest` (markdown link/image, HTML tag, URL) ended within the buffer?"""
    lowered = rest.lower()
    if lowered.startswith("http"):
        return re.search(r"[\s<>\"')\]]", rest) is not None
    if rest.startswith("<"):
        return ">" in rest
    bracket = rest.find("]")
    if bracket < 0:
        return "\n" in rest  # "[" without "]" before a line break is plain text
    after = rest[bracket + 1 :]
    if not after:
        return False  # "[label]" may still be followed by "(url)"
    if not after.startswith("("):
        return True
    return ")" in after or "\n" in after


STOP_NOTICE = "\n\n[Bulwark stopped this response: {reason}]"


@dataclass
class StreamStep:
    text: str
    stopped: bool = False


@dataclass
class StreamGuard:
    firewall: Firewall
    policy: Policy
    system_prompt: str | None = None
    canary: str | None = None
    conversation: str = ""
    hold_chars: int = 24
    max_hold: int = 1500
    emitted: str = ""
    pending: str = ""
    stopped: bool = False
    results: list[GuardResult] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    """The upstream text as received (for the audit and the playground)."""

    async def push(self, delta: str) -> StreamStep:
        if self.stopped or not delta:
            return StreamStep("", self.stopped)
        self.raw.append(delta)
        self.pending += delta
        cut = self._safe_cut()
        if cut <= 0:
            return StreamStep("")
        return await self._release(cut)

    async def finish(self) -> StreamStep:
        if self.stopped or not self.pending:
            return StreamStep("", self.stopped)
        return await self._release(len(self.pending))

    def _safe_cut(self) -> int:
        text = self.pending
        if len(text) > self.max_hold:
            return len(text) - self.hold_chars  # an unclosed "[" this long is just text
        cut = max(0, len(text) - self.hold_chars)
        for match in _OPENERS.finditer(text):
            if match.start() >= cut:
                break
            if not _closed(text[match.start() :]):
                cut = match.start()
                break
        # Never split a word: secrets and canaries have no spaces.
        while cut > 0 and not text[cut - 1].isspace():
            cut -= 1
        return cut

    async def _release(self, cut: int) -> StreamStep:
        chunk, self.pending = self.pending[:cut], self.pending[cut:]
        decision = await self.firewall.check_output(
            self.emitted + chunk,
            policy=self.policy,
            system_prompt=self.system_prompt,
            canary=self.canary,
            conversation=self.conversation,
            pii_shield=False,  # an HTTP call per released piece is too slow; see gateway buffering
        )
        triggered = [r for r in decision.results if r.triggered]
        if triggered:
            self._record(triggered)
        if decision.enforced and decision.action is Action.BLOCK:
            self.stopped = True
            reason = next(r.explanation for r in triggered if r.action is Action.BLOCK)
            return StreamStep(STOP_NOTICE.format(reason=reason), stopped=True)
        if decision.enforced and decision.text is not None:
            # Sanitization rewrote something; only the new part is released (earlier text was already safe).
            released = decision.text[len(self.emitted) :] if decision.text.startswith(self.emitted) else chunk
        else:
            released = chunk
        self.emitted += released
        return StreamStep(released)

    def _record(self, results: list[GuardResult]) -> None:
        for result in results:
            same = [r for r in self.results if r.guard == result.guard]
            if not same or result.score > same[0].score or result.explanation != same[0].explanation:
                self.results = [r for r in self.results if r.guard != result.guard] + [result]

    def decision(self) -> Decision:
        action = strictest([r.action for r in self.results])
        return Decision(
            stage="output",
            policy=self.policy.name,
            action=action,
            results=self.results,
            text=self.emitted,
            enforced=self.policy.enforced,
        )
