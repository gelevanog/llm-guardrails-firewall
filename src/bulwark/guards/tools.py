"""Tool-call guard for agents: allow-lists, argument validation, taint, and human approval.

Checks, in order, each with its own policy action (the strictest wins):
  1. the tool is allowed on this route;
  2. the arguments parse and satisfy the tool's JSON Schema;
  3. addresses and URLs in the arguments stay inside allowed domains;
  4. argument deny-patterns;
  5. taint: untrusted (or flagged) content is in the conversation and the tool is risky;
  6. provenance: an address, URL or account number in the arguments came only from untrusted content.

A `require_approval` outcome returns an `ApprovalRequest` (id, tool, arguments, reasons) instead of an
error: the agent pauses and a person decides, with the reasons in front of them.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from bulwark.core import Action, ApprovalRequest, Finding, GuardResult, strictest
from bulwark.policy import Risk, ToolsConfig
from bulwark.taint import TaintState

_EMAIL = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
_URL_HOST = re.compile(r"https?://([^/\s:?#]+)", re.IGNORECASE)
_SCORE = {
    Action.ALLOW: 0.0,
    Action.LOG_ONLY: 0.2,
    Action.FLAG: 0.4,
    Action.SANITIZE: 0.5,
    Action.REQUIRE_APPROVAL: 0.7,
    Action.BLOCK: 1.0,
}


def parse_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _domain_of(value: str) -> str | None:
    if match := _URL_HOST.search(value):
        return match.group(1).lower()
    if match := _EMAIL.search(value):
        return match.group(1).lower()
    return None


def _values(arguments: dict[str, Any], field: str) -> list[str]:
    value = arguments.get(field)
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def check_tool_call(
    name: str, arguments: str | dict[str, Any] | None, taint: TaintState, config: ToolsConfig
) -> tuple[GuardResult, ApprovalRequest | None]:
    started = time.perf_counter()
    findings: list[Finding] = []
    actions: list[Action] = []

    def add(rule: str, action: Action, message: str, snippet: str = "") -> None:
        if action is Action.ALLOW:
            return
        actions.append(action)
        findings.append(
            Finding(rule=rule, layer="tool_policy", score=_SCORE[action], message=message, snippet=snippet[:160])
        )

    rule = config.allowed.get(name)
    parsed: dict[str, Any] = {}
    if rule is None:
        add("tools.not_allowed", config.default, f"tool {name!r} is not allowed on this route")
    try:
        parsed = parse_arguments(arguments)
    except (ValueError, json.JSONDecodeError) as exc:
        add("tools.bad_arguments", Action.BLOCK, f"arguments are not a valid JSON object: {exc}")

    risk = rule.risk if rule else Risk.EXTERNAL
    if rule is not None and not findings:
        if rule.schema_:
            try:
                errors = sorted(Draft202012Validator(rule.schema_).iter_errors(parsed), key=lambda e: list(e.path))
            except SchemaError as exc:
                errors = []
                add("tools.bad_schema", Action.BLOCK, f"the policy's schema for {name!r} is invalid: {exc.message}")
            for error in errors[:3]:
                where = ".".join(str(p) for p in error.path) or "arguments"
                add("tools.schema", rule.on_schema_violation, f"{where}: {error.message}")
        if rule.allowed_domains:
            for field in rule.allowed_domains.fields:
                for value in _values(parsed, field):
                    domain = _domain_of(value)
                    allowed = rule.allowed_domains.domains
                    if domain and not any(domain == d or domain.endswith("." + d) for d in allowed):
                        add(
                            "tools.external_recipient",
                            rule.allowed_domains.action,
                            f"{field} {value!r} is outside the allowed domains ({', '.join(allowed)})",
                            value,
                        )
        for field, patterns in rule.deny_patterns.items():
            for value in _values(parsed, field) or ([str(parsed[field])] if field in parsed else []):
                for pattern in patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        add(
                            "tools.deny_pattern",
                            Action.BLOCK,
                            f"{field} matches a forbidden pattern ({pattern})",
                            value,
                        )

    level = taint.level
    if level == "suspicious":
        flagged = ", ".join(sorted({s.source for s in taint.sources if s.flagged}))
        add(
            "taint.suspicious",
            config.on_suspicious.for_risk(risk),
            f"a {risk.value} tool after flagged untrusted content entered the conversation (from {flagged})",
        )
    elif level == "tainted":
        sources = ", ".join(sorted({s.source for s in taint.sources}))
        add(
            "taint.untrusted_context",
            config.on_taint.for_risk(risk),
            f"a {risk.value} tool after untrusted content entered the conversation (from {sources})",
        )
    for path, atom in taint.untrusted_arguments(parsed):
        add(
            "taint.untrusted_argument",
            config.untrusted_arguments.for_risk(risk),
            f"{path} = {atom!r} appears only in untrusted content, not in the user's request",
            atom,
        )

    action = strictest(actions)
    reasons = list(dict.fromkeys(f.message for f in findings if f.score >= _SCORE[Action.REQUIRE_APPROVAL]))
    approval = (
        ApprovalRequest(tool=name, arguments=parsed, reasons=reasons) if action is Action.REQUIRE_APPROVAL else None
    )
    result = GuardResult(
        guard="tool_call",
        score=_SCORE[action],
        triggered=action is not Action.ALLOW,
        action=action,
        explanation="; ".join(dict.fromkeys(f.message for f in findings)),
        findings=findings,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return result, approval
