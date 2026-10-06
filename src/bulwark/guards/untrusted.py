"""Indirect prompt injection: inspect untrusted content before it enters the model's context.

RAG chunks, tool results, emails and web pages are written by someone other than the user. The guard:
  1. separates what a human would see from what is hidden (HTML tricks, invisible Unicode payloads) and
     drops the hidden part, scanning it with extra suspicion;
  2. runs the layered detector (heuristics with exact spans, classifier, optional judge);
  3. localizes the injected instruction to its sentences (spans from the rules, or sentence-level
     re-scoring when only the classifier fired) and quarantines, strips or withholds it per policy;
  4. spotlights what remains, so the model reads it as data.
The session is tainted either way: even clean-looking content may carry an instruction no detector saw.
"""

from __future__ import annotations

import asyncio
import re
import time

from pydantic import BaseModel, Field

from bulwark.core import Action, Finding, GuardResult
from bulwark.guards import heuristics
from bulwark.guards.injection import Detection, InjectionDetector, explain
from bulwark.guards.spotlight import spotlight
from bulwark.normalize import extract_html, find_hidden_payloads, strip_invisible
from bulwark.policy import UntrustedAction, UntrustedConfig

_SEGMENT = re.compile(r"[^\n]*?(?:[.!?](?=\s)|\n|$)")
_MIN_SEGMENT_CHARS = 12
_INVISIBLE_VIEWS = frozenset({"tag-chars", "variation-selectors"})


class Removal(BaseModel):
    start: int
    end: int
    reason: str
    snippet: str


class HiddenContent(BaseModel):
    reason: str
    text: str
    suspicious: bool


class UntrustedReport(BaseModel):
    source: str
    flagged: bool
    score: float
    sanitized: str
    """Content after hidden text and flagged spans were removed (before spotlighting)."""
    model_text: str
    """What the model receives: the sanitized content, spotlighted."""
    removals: list[Removal] = Field(default_factory=list)
    hidden: list[HiddenContent] = Field(default_factory=list)
    withheld: bool = False


def segments(text: str) -> list[tuple[int, int]]:
    """Sentence and line spans, used to expand a match to the whole instruction around it."""
    spans = []
    for match in _SEGMENT.finditer(text):
        if match.end() > match.start() and text[match.start() : match.end()].strip():
            spans.append((match.start(), match.end()))
    return spans


def expand_to_segments(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pieces = segments(text)
    expanded: list[tuple[int, int]] = []
    for start, end in spans:
        covering = [(s, e) for s, e in pieces if s < end and start < e]
        if covering:
            expanded.append((covering[0][0], covering[-1][1]))
        else:
            expanded.append((start, end))
    return merge_spans(expanded)


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class UntrustedContentGuard:
    def __init__(self, detector: InjectionDetector) -> None:
        self.detector = detector

    async def inspect(
        self,
        content: str,
        source: str,
        config: UntrustedConfig,
        *,
        fail_closed: bool = True,
        block_id: str | None = None,
    ) -> tuple[GuardResult, UntrustedReport]:
        started = time.perf_counter()
        findings: list[Finding] = []
        hidden: list[HiddenContent] = []

        # 1. What would a human see? Scan the hidden remainder separately.
        extraction = extract_html(content)
        visible = extraction.visible
        hidden_score = 0.0
        for segment in extraction.hidden:
            scanned = heuristics.scan(segment.text, "untrusted")
            suspicious = scanned.score >= 0.3
            hidden.append(HiddenContent(reason=segment.reason, text=segment.text[:400], suspicious=suspicious))
            if suspicious:
                score = min(1.0, 0.5 + scanned.score / 2)
                hidden_score = max(hidden_score, score)
                findings.append(
                    Finding(
                        rule="hidden.instruction",
                        layer="html",
                        score=round(score, 3),
                        message=(
                            f"hidden HTML text ({segment.reason}) contains instructions: {explain(scanned.findings, 1)}"
                        ),
                        snippet=segment.text[:160],
                        view="html-hidden",
                    )
                )
        for payload in find_hidden_payloads(visible):
            hidden.append(HiddenContent(reason=payload.kind, text=payload.text[:400], suspicious=True))

        # 2. Layered detection on the visible text (invisible payloads are decoded by the heuristics).
        detection = await self.detector.detect(visible, config.layers, "untrusted", fail_closed=fail_closed)
        findings.extend(detection.findings)
        score = max(detection.score, hidden_score)
        flagged = score >= config.threshold

        # 3. Localize and remove.
        removals: list[Removal] = []
        sanitized = visible
        withheld = False
        if flagged and config.on_detection in {UntrustedAction.QUARANTINE, UntrustedAction.STRIP}:
            spans = await self._localize(visible, detection, config)
            if spans:
                sanitized, removals = _remove(visible, spans, config.on_detection, findings)
            elif hidden_score < config.threshold:
                withheld = True
        if flagged and config.on_detection is UntrustedAction.BLOCK:
            withheld = True
        if withheld:
            removals = [Removal(start=0, end=len(visible), reason="whole document withheld", snippet=visible[:120])]
            sanitized = (
                f"[Bulwark withheld this {source} content: it contained instructions aimed at the AI "
                "and they could not be isolated.]"
            )
        if config.strip_hidden:
            sanitized = strip_invisible(sanitized)

        report = UntrustedReport(
            source=source,
            flagged=flagged,
            score=round(score, 4),
            sanitized=sanitized,
            model_text=spotlight(sanitized, config.spotlight, source, block_id=block_id),
            removals=removals,
            hidden=hidden,
            withheld=withheld,
        )
        if not flagged:
            action = Action.ALLOW
        elif config.on_detection is UntrustedAction.FLAG:
            action = Action.FLAG
        elif config.on_detection is UntrustedAction.BLOCK:
            action = Action.BLOCK
        else:
            action = Action.SANITIZE
        result = GuardResult(
            guard="untrusted",
            score=round(score, 4),
            triggered=flagged,
            action=action,
            explanation=(explain(findings) if flagged else "")
            + (f" ({len(removals)} span(s) removed)" if removals and not withheld else ""),
            findings=findings,
            layer_scores={
                **detection.layer_scores,
                **({"hidden_html": round(hidden_score, 4)} if extraction.hidden else {}),
            },
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error="; ".join(f"{k}: {v}" for k, v in detection.errors.items()) or None,
        )
        return result, report

    async def _localize(self, text: str, detection: Detection, config: UntrustedConfig) -> list[tuple[int, int]]:
        """Spans to remove: rule matches expanded to their sentences, else sentence-level re-scoring."""
        located = [
            f
            for f in detection.findings
            if f.start is not None
            and f.end is not None
            and f.score >= 0.2
            and f.view not in {"rot13", "reversed"}
            and f.end - f.start < len(text)
        ]
        if located:
            # Invisible payloads are removed exactly; the visible sentence carrying them may be legitimate.
            exact: list[tuple[int, int]] = []
            sentences: list[tuple[int, int]] = []
            for finding in located:
                assert finding.start is not None and finding.end is not None
                target = exact if finding.view in _INVISIBLE_VIEWS else sentences
                target.append((finding.start, finding.end))
            return merge_spans(expand_to_segments(text, sentences) + exact)
        pieces = [(s, e) for s, e in segments(text) if e - s >= _MIN_SEGMENT_CHARS][:60]
        if not pieces:
            return []
        flagged: list[tuple[int, int]] = []
        texts = [text[s:e] for s, e in pieces]
        scores = [heuristics.scan(t, "untrusted").score for t in texts]
        classifier = self.detector.classifier
        if config.layers.classifier and classifier is not None:
            model_scores = await asyncio.to_thread(classifier.score_many, texts)
            scores = [max(a, b) for a, b in zip(scores, model_scores, strict=True)]
        for (start, end), score in zip(pieces, scores, strict=True):
            if score >= config.threshold:
                flagged.append((start, end))
        return merge_spans(flagged)


def _remove(
    text: str, spans: list[tuple[int, int]], mode: UntrustedAction, findings: list[Finding]
) -> tuple[str, list[Removal]]:
    removals: list[Removal] = []
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        reason = explain([f for f in findings if f.start is not None and f.start < end and start < (f.end or 0)], 1)
        removals.append(
            Removal(start=start, end=end, reason=reason or "suspected instruction", snippet=text[start:end][:200])
        )
        out.append(text[cursor:start])
        if mode is UntrustedAction.QUARANTINE:
            out.append(f"[Bulwark removed {end - start} characters of suspected instructions here.] ")
        cursor = end
    out.append(text[cursor:])
    return "".join(out).strip(), removals
