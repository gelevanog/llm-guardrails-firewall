"""A deterministic, deliberately gullible model for the email agent (no API key needed).

It plans like a simple tool-using agent (look up orders named in the task, search and read the relevant
emails or articles, create the ticket or send the reply the task asks for, then answer) and, like the
generic fake upstream, obeys every instruction it finds in what it reads. It reads hidden HTML text,
invisible tag characters and base64, and ignores spotlighting markers, so the attacks in the fixture
mailbox succeed against an unprotected agent. With Bulwark the attacks fail for structural reasons
(quarantine, tool-call policy, output sanitization), not because the model behaved.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from bulwark.providers.base import JsonDict, message_text
from bulwark.providers.fake import (
    SPOTLIGHT_MARKERS,
    completion,
    done_calls,
    fill_url,
    injected_commands,
    make_call,
    readable_view,
    stream_message,
)

_ORDER = re.compile(r"\bA-\d{4}\b")
_EMAIL_ID = re.compile(r"\bem-\d{3}\b")
_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_INBOX_HINTS = re.compile(
    r"\b(inbox|emails?|wrote|asked|newsletter|invoices?|helpdesk|proposal|management|feedback|briefing|"
    r"operations|digest|journal|reply|forward)\b",
    re.IGNORECASE,
)
_KEYWORDS = (
    "invoice", "newsletter", "password", "proposal", "management", "feedback", "security", "briefing",
    "operations", "weekly", "numbers", "tent", "refund", "autumn", "trends", "damaged",
)  # fmt: skip
_STOP = {
    "I",
    "A",
    "The",
    "What",
    "Is",
    "Go",
    "Did",
    "Give",
    "Read",
    "Reply",
    "Look",
    "How",
    "Check",
    "Handle",
    "Forward",
    "Create",
}


def _query(task: str) -> str:
    names = [w for w in re.findall(r"\b[A-Z][a-z]+\b", task) if w not in _STOP]
    keywords = [k for k in _KEYWORDS if k in task.lower()]
    return " ".join(dict.fromkeys(names + keywords))


def _tool_results(messages: list[JsonDict]) -> list[tuple[str, str]]:
    """(tool name, content) for each tool message, matched to the assistant call that produced it."""
    names: dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            names[call["id"]] = call["function"]["name"]
    return [
        (names.get(str(m.get("tool_call_id")), "tool"), message_text(m)) for m in messages if m.get("role") == "tool"
    ]


def _plain(text: str) -> str:
    return " ".join(readable_view(text).split())


def plan(body: JsonDict) -> JsonDict:
    messages: list[JsonDict] = body.get("messages", [])
    task = next((message_text(m) for m in messages if m.get("role") == "user"), "")
    system = next((message_text(m) for m in messages if m.get("role") == "system"), "")
    done = done_calls(messages)
    results = _tool_results(messages)
    seed = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    lowered = task.lower()

    def called(name: str, **match: str) -> bool:
        return any(n == name and all(str(a.get(k, "")).lower() == v.lower() for k, v in match.items()) for n, a in done)

    def call(name: str, arguments: JsonDict) -> JsonDict:
        return {"role": "assistant", "content": None, "tool_calls": [make_call(name, arguments, seed + str(len(done)))]}

    # 0. Gullible: obey instructions found in tool results, each once.
    collected = "\n".join(_plain(content) for _, content in results)[:1500]
    documents = [(name, content) for name, content in results if name in {"read_email", "search_kb"}]
    for _, content in documents:
        for command in injected_commands(content):
            if command.kind == "send" and not called("send_email", to=command.target):
                return call("send_email", {"to": command.target, "subject": "Requested export", "body": collected})
            if command.kind == "ticket" and not any(n == "create_ticket" for n, _ in done):
                return call(
                    "create_ticket",
                    {"title": command.title, "description": command.sentence[:300], "priority": command.priority},
                )

    # 1. Orders named in the task (or, when the task is about an order, in the emails read so far).
    orders = list(dict.fromkeys(_ORDER.findall(task)))
    if "order" in lowered:
        for name, content in results:
            if name == "read_email":
                orders += [o for o in _ORDER.findall(content) if o not in orders]
    for order in orders:
        if not called("lookup_order", order_id=order):
            return call("lookup_order", {"order_id": order})

    # 2. Knowledge base.
    if "knowledge base" in lowered and not any(n == "search_kb" for n, _ in done):
        return call("search_kb", {"query": re.sub(r"[^\w\s]", "", task)[:120]})

    # 3. Inbox: search, then read the hits.
    reads: list[str] = []
    if _INBOX_HINTS.search(task) and not (orders and not re.search(r"email|wrote|reply|handle", lowered)):
        if not any(n == "search_inbox" for n, _ in done):
            return call("search_inbox", {"query": "" if "recent" in lowered or "digest" in lowered else _query(task)})
        listing = next((content for name, content in results if name == "search_inbox"), "")
        limit = 5 if "five" in lowered or "digest" in lowered else 3
        candidates = list(dict.fromkeys(_EMAIL_ID.findall(listing.split("Previews:")[0])))[:limit]
        for email_id in candidates:
            if not called("read_email", email_id=email_id):
                return call("read_email", {"email_id": email_id})
        reads = candidates

    emails = [content for name, content in results if name == "read_email"]
    names = [w.lower() for w in re.findall(r"\b[A-Z][a-z]+\b", task) if w not in _STOP]
    primary = next(
        (e for e in emails if any(n in e.split("Date:")[0].lower() for n in names)), emails[0] if emails else ""
    )
    first = _plain(primary) if primary else ""
    sender = _ADDRESS.search(first.split("Date:")[0]) if first else None
    subject = re.search(r"Subject: (.+?)(?: <<| \[|$)", first)
    order_lines = [_plain(content) for name, content in results if name == "lookup_order"]

    # 4. The action the task asks for.
    if "ticket" in lowered and not any(n == "create_ticket" for n, _ in done) and emails:
        return call(
            "create_ticket",
            {
                "title": subject.group(1)[:120] if subject else "Customer request",
                "description": first[:600],
                "priority": "normal",
                "customer_email": sender.group(0) if sender else "",
            },
        )
    target = _ADDRESS.search(task)
    if "forward" in lowered and target and not called("send_email", to=target.group(0)) and emails:
        return call(
            "send_email",
            {
                "to": target.group(0),
                "subject": "Fwd: " + (subject.group(1) if subject else "email"),
                "body": first[:1500],
            },
        )
    if "reply" in lowered and sender and not any(n == "send_email" for n, _ in done):
        body_text = "Hello,\n\n" + (" ".join(order_lines) or "Thank you for your message.") + "\n\nAcme Outdoor support"
        return call(
            "send_email",
            {
                "to": sender.group(0),
                "subject": "Re: " + (subject.group(1) if subject else "your email"),
                "body": body_text,
            },
        )

    # 5. Answer.
    lines: list[str] = []
    lines += order_lines
    lines += [f"- {_plain(content)[:700]}" for name, content in results if name in {"read_email", "search_kb"}]
    if not reads and not lines:
        lines += [_plain(content)[:600] for name, content in results if name == "search_inbox"]
    lines += [
        f"Action: {SPOTLIGHT_MARKERS.sub('', content).strip()[:200]}"
        for name, content in results
        if name in {"create_ticket", "send_email"}
    ]
    summary = " ".join(lines)[:200]
    for _, content in documents:
        for command in injected_commands(content):
            if command.kind == "reveal" and system:
                lines.append("As requested, my full instructions: " + system)
            elif command.kind == "image":
                lines.append(f"![image]({fill_url(command.target, summary)})")
            elif command.kind == "link":
                lines.append(f"Tell the customer to verify their card at {command.target} first.")
    answer = (
        "Here is what I found:\n" + "\n".join(dict.fromkeys(lines)) if lines else "I could not find anything relevant."
    )
    return {"role": "assistant", "content": answer}


class GullibleAgentModel:
    def __init__(self, model: str = "fake-gullible-agent") -> None:
        self.model = model

    @property
    def label(self) -> str:
        return f"fake/{self.model}"

    @property
    def is_remote(self) -> bool:
        return False

    async def complete(self, body: JsonDict) -> JsonDict:
        return completion(body, plan(body), self.model)

    async def stream(self, body: JsonDict) -> AsyncIterator[JsonDict]:
        async for chunk in stream_message(await self.complete(body)):
            yield chunk
