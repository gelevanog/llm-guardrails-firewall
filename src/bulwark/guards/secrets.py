"""Secrets and payment data in model answers (built-in patterns), optionally PII via PII Shield.

An agent that read a config file, a log or a customer record can repeat an API key or a card number in its
answer; an injected instruction can ask it to. The built-in check covers the formats where a regex is
precise (provider key prefixes, private keys, JWTs, `password=` assignments, Luhn-valid card numbers).
Personal data (names, addresses, phone numbers in many formats) needs NER: with `pii_shield: true` the
answer is also sent to a PII Shield gateway, which masks what it finds.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from bulwark.core import Action, Finding, GuardResult
from bulwark.policy import SecretsConfig


@dataclass(frozen=True)
class SecretPattern:
    id: str
    pattern: re.Pattern[str]
    label: str


SECRET_PATTERNS: tuple[SecretPattern, ...] = (
    SecretPattern("openrouter_key", re.compile(r"\bsk-or-v1-[0-9a-f]{32,}\b"), "OpenRouter API key"),
    SecretPattern("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), "Anthropic API key"),
    SecretPattern("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}"), "OpenAI-style API key"),
    SecretPattern(
        "github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"), "GitHub token"
    ),
    SecretPattern("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "AWS access key id"),
    SecretPattern("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "Google API key"),
    SecretPattern("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    SecretPattern("stripe_key", re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{20,}\b"), "Stripe live key"),
    SecretPattern(
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----[\s\S]*?(?:-----END[^-]*-----|$)"),
        "private key",
    ),
    SecretPattern("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "JWT"),
    SecretPattern(
        "password_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)"
            r"\s*[:=]\s*[\"']?(?P<value>[^\s\"']{8,})"
        ),
        "password or key assignment",
    ),
)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            digit -= 9 if digit > 9 else 0
        total += digit
    return total % 10 == 0


def find_secrets(text: str) -> list[Finding]:
    findings: list[Finding] = []
    taken: list[tuple[int, int]] = []
    for secret in SECRET_PATTERNS:
        for match in secret.pattern.finditer(text):
            start, end = (
                (match.start("value"), match.end("value")) if "value" in secret.pattern.groupindex else match.span()
            )
            if any(s < end and start < e for s, e in taken):
                continue
            taken.append((start, end))
            findings.append(
                Finding(
                    rule=f"secrets.{secret.id}",
                    layer="secret",
                    score=0.95,
                    message=f"{secret.label} in the answer",
                    start=start,
                    end=end,
                    snippet=text[start:end][:6] + "…",
                )
            )
    for match in _CARD.finditer(text):
        if luhn_valid(match.group(0)) and not any(s < match.end() and match.start() < e for s, e in taken):
            taken.append(match.span())
            findings.append(
                Finding(
                    rule="secrets.card_number",
                    layer="secret",
                    score=0.9,
                    message="payment card number (Luhn-valid) in the answer",
                    start=match.start(),
                    end=match.end(),
                    snippet="****" + re.sub(r"\D", "", match.group(0))[-4:],
                )
            )
    return findings


def mask(text: str, findings: list[Finding]) -> str:
    out = text
    for finding in sorted((f for f in findings if f.start is not None), key=lambda f: -(f.start or 0)):
        assert finding.start is not None and finding.end is not None
        label = finding.rule.split(".", 1)[1].upper()
        out = out[: finding.start] + f"[REDACTED {label}]" + out[finding.end :]
    return out


class PiiShieldError(RuntimeError):
    pass


class PiiShieldClient:
    """Minimal client for PII Shield's `POST /v1/redact` (github.com/gelevanog/pii-redaction-gateway)."""

    def __init__(
        self,
        base_url: str,
        *,
        policy: str = "support-chat",
        timeout_seconds: float = 10.0,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/v1/redact"
        self.policy = policy
        self.timeout = timeout_seconds
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.transport = transport

    async def redact(self, text: str) -> tuple[str, list[Finding]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                response = await client.post(self.url, json={"text": text, "policy": self.policy}, headers=self.headers)
        except httpx.HTTPError as exc:
            raise PiiShieldError(f"PII Shield unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise PiiShieldError(f"PII Shield returned {response.status_code}")
        data = response.json()
        findings = [
            Finding(
                rule=f"pii_shield.{entity.get('type', 'PII').lower()}",
                layer="pii_shield",
                score=float(entity.get("score", 0.9)),
                message=f"{entity.get('type', 'PII')} found by PII Shield ({entity.get('action', 'redacted')})",
                start=entity.get("start"),
                end=entity.get("end"),
            )
            for entity in data.get("entities", [])
        ]
        return str(data.get("text", text)), findings


async def check_secrets(
    text: str, config: SecretsConfig, pii_shield: PiiShieldClient | None = None, *, fail_closed: bool = True
) -> tuple[GuardResult, str]:
    started = time.perf_counter()
    findings = find_secrets(text)
    sanitized = mask(text, findings)
    error = None
    if config.pii_shield:
        if pii_shield is None:
            error = "the policy asks for PII Shield but BULWARK_PII_SHIELD_URL is not set"
        else:
            try:
                redacted, pii = await pii_shield.redact(sanitized)
            except PiiShieldError as exc:
                error = str(exc)
            else:
                findings += pii
                sanitized = redacted
    if error and fail_closed:
        findings.append(Finding(rule="fail_closed", layer="policy", score=1.0, message=error))
    score = max((f.score for f in findings), default=0.0)
    triggered = bool(findings)
    action = config.action if triggered else Action.ALLOW
    if error and fail_closed:
        action = Action.BLOCK
    return (
        GuardResult(
            guard="secrets",
            score=score,
            triggered=triggered,
            action=action,
            explanation="; ".join(dict.fromkeys(f.message for f in findings)),
            findings=findings,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error=error,
        ),
        sanitized,
    )
