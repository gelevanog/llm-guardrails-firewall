"""Apply the guards to OpenAI chat.completions requests and responses.

Request: the system prompt gets the spotlighting note and a canary; the newest user message goes through
the input guards (earlier turns were checked when they were sent); every message in an untrusted role
(by default `tool`) is inspected, quarantined and spotlighted, and taints the conversation. The OpenAI API
resends the whole history each turn, so the taint state is rebuilt from the request: the gateway stays
stateless.

Response: text goes through the output guards (whole, or piece by piece when streaming); each tool call
goes through the tool-call guard and is passed on, withheld for approval, or removed.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from bulwark.core import Action, ApprovalRequest, Decision, digest, strictest
from bulwark.firewall import Firewall, GuardSession
from bulwark.guards.untrusted import UntrustedReport
from bulwark.policy import Policy
from bulwark.providers.base import JsonDict, message_text


class UntrustedCache:
    """Tool results are resent every turn; scan each distinct one once per policy."""

    def __init__(self, max_entries: int = 512) -> None:
        self._items: OrderedDict[str, tuple[Decision, UntrustedReport]] = OrderedDict()
        self.max_entries = max_entries

    def get(self, key: str) -> tuple[Decision, UntrustedReport] | None:
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, key: str, value: tuple[Decision, UntrustedReport]) -> None:
        self._items[key] = value
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)


@dataclass
class PreparedRequest:
    body: JsonDict
    session: GuardSession
    input_decision: Decision | None = None
    untrusted: list[tuple[str, Decision, UntrustedReport]] = field(default_factory=list)

    @property
    def decisions(self) -> list[Decision]:
        decisions = [self.input_decision] if self.input_decision else []
        return decisions + [decision for _, decision, _ in self.untrusted]


def _set_text(message: JsonDict, text: str) -> None:
    """Replace a message's text, keeping non-text parts (images) of multi-part content."""
    content = message.get("content")
    if isinstance(content, list):
        others = [part for part in content if not (isinstance(part, dict) and part.get("type") == "text")]
        message["content"] = [{"type": "text", "text": text}, *others]
    else:
        message["content"] = text


async def prepare_request(
    firewall: Firewall, policy: Policy, body: JsonDict, cache: UntrustedCache | None = None
) -> PreparedRequest:
    session = firewall.session(policy)
    prepared_body = copy.deepcopy(body)
    messages: list[JsonDict] = prepared_body["messages"]
    prepared = PreparedRequest(body=prepared_body, session=session)
    untrusted_roles = set(policy.untrusted.untrusted_roles) if policy.untrusted.enabled else set()

    system_parts = [message_text(m) for m in messages if m.get("role") in {"system", "developer"}]
    protected = session.protect_system_prompt("\n\n".join(system_parts))
    first_system = next((m for m in messages if m.get("role") in {"system", "developer"}), None)
    if first_system is None:
        messages.insert(0, {"role": "system", "content": protected.strip()})
    else:
        _set_text(first_system, protected)
        for message in messages:
            if message is not first_system and message.get("role") in {"system", "developer"}:
                message["content"] = ""
        messages[:] = [m for m in messages if not (m.get("role") in {"system", "developer"} and m.get("content") == "")]

    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    for index, message in enumerate(messages):
        role = message.get("role")
        text = message_text(message)
        if role in untrusted_roles:
            source = str(message.get("name") or role)
            key = f"{policy.name}:{source}:{digest(text, 32)}"
            cached = cache.get(key) if cache else None
            if cached is None:
                cached = await firewall.check_untrusted(text, source=source, policy=policy)
                if cache:
                    cache.put(key, cached)
            decision, report = cached
            session.taint.add_untrusted(text, source, flagged=report.flagged, score=report.score)
            if decision.text is not None:
                _set_text(message, decision.text)
            prepared.untrusted.append((source, decision, report))
        elif role == "user":
            if index == last_user:
                prepared.input_decision = await session.check_input(text)
            else:
                session.add_trusted(text)
    return prepared


@dataclass
class ToolCallOutcome:
    passed: list[JsonDict] = field(default_factory=list)
    withheld: list[ApprovalRequest] = field(default_factory=list)
    blocked: list[tuple[str, str]] = field(default_factory=list)
    decisions: list[tuple[JsonDict, Decision]] = field(default_factory=list)


def guard_tool_calls(session: GuardSession, calls: list[JsonDict]) -> ToolCallOutcome:
    outcome = ToolCallOutcome()
    for call in calls:
        function = call.get("function") or {}
        decision = session.check_tool_call(str(function.get("name", "")), function.get("arguments"))
        outcome.decisions.append((call, decision))
        if decision.blocked:
            outcome.blocked.append((str(function.get("name", "")), decision.explain()))
        elif decision.needs_approval and decision.approval is not None:
            outcome.withheld.append(decision.approval)
        else:
            outcome.passed.append(call)
    return outcome


def tool_notice(outcome: ToolCallOutcome) -> str:
    lines = [f"[Bulwark blocked a call to {name}: {reason}]" for name, reason in outcome.blocked]
    lines += [
        f"[Bulwark is holding a call to {a.tool} for human approval (id {a.id}): {'; '.join(a.reasons)}]"
        for a in outcome.withheld
    ]
    return "\n".join(lines)


def bulwark_extension(
    policy: Policy, decisions: list[Decision], approvals: list[ApprovalRequest], request_id: str
) -> dict[str, Any]:
    enforced = [d.action for d in decisions if d.enforced]
    return {
        "request_id": request_id,
        "policy": policy.name,
        "mode": policy.mode,
        "action": strictest(enforced).value if enforced else Action.ALLOW.value,
        "decisions": [d.summary() for d in decisions if d.triggered],
        "approvals": [a.model_dump() for a in approvals],
    }


def tool_call_arguments(call: JsonDict) -> Any:
    raw = (call.get("function") or {}).get("arguments")
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return raw
