"""The demo agent's tools over a fixture mailbox, knowledge base and order database.

Tools return their result as text, split into what the application vouches for (`trusted`: mail
headers from the mail server, rows from our own order database) and what someone else wrote
(`untrusted`: email bodies, knowledge-base articles). Side effects (sent emails, created tickets) are
recorded, never performed.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from bulwark.normalize import extract_html

DATA = Path(__file__).parent / "data"
UNTRUSTED_TOOLS = frozenset({"search_inbox", "read_email", "search_kb"})

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_inbox",
            "description": (
                "Search the support inbox. Returns matching emails (id, sender, subject, date, preview), "
                "most relevant first."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keywords; empty for the latest emails."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_email",
            "description": "Read one email by id (for example em-101).",
            "parameters": {
                "type": "object",
                "properties": {"email_id": {"type": "string"}},
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the help-center knowledge base. Returns the most relevant articles.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id (format A-1234): item, status, carrier, tracking number.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                    "customer_email": {"type": "string"},
                },
                "required": ["title", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email from the support address to one recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "One email address."},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]


class Email(BaseModel):
    id: str
    sender: str
    subject: str
    date: str
    body: str
    format: str = "text"
    attack: str | None = None

    @property
    def sender_address(self) -> str:
        match = re.search(r"<([^>]+)>", self.sender)
        return match.group(1) if match else self.sender

    def plain(self) -> str:
        return extract_html(self.body).visible if self.format == "html" else self.body


class Article(BaseModel):
    id: str
    title: str
    body: str
    attack: str | None = None


def _encode(text: str) -> str:
    text = re.sub(r"\{\{tag:(.*?)\}\}", lambda m: "".join(chr(0xE0000 + ord(c)) for c in m.group(1)), text, flags=re.S)
    return re.sub(r"\{\{b64:(.*?)\}\}", lambda m: base64.b64encode(m.group(1).encode()).decode(), text, flags=re.S)


@dataclass(frozen=True)
class Fixture:
    company: str
    domain: str
    user_name: str
    user_email: str
    emails: tuple[Email, ...]
    articles: tuple[Article, ...]
    orders: dict[str, dict[str, str]]

    def email(self, email_id: str) -> Email | None:
        return next((e for e in self.emails if e.id == email_id.strip()), None)


@cache
def load_fixture(path: Path = DATA / "mailbox.yaml") -> Fixture:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    emails = tuple(
        Email(
            id=e["id"],
            sender=e["from"],
            subject=e["subject"],
            date=e["date"],
            body=_encode(e["body"]).strip(),
            format=e.get("format", "text"),
            attack=e.get("attack"),
        )
        for e in raw["emails"]
    )
    articles = tuple(Article(**a) for a in raw["knowledge_base"])
    return Fixture(
        company=raw["company"],
        domain=raw["domain"],
        user_name=raw["user"]["name"],
        user_email=raw["user"]["email"],
        emails=emails,
        articles=articles,
        orders={k: {kk: str(vv) for kk, vv in v.items()} for k, v in raw["orders"].items()},
    )


@dataclass
class ToolOutput:
    trusted: str
    """Metadata the application vouches for (headers, our own database rows)."""
    untrusted: str = ""
    """Content written by outsiders (email bodies, articles)."""

    @property
    def text(self) -> str:
        return "\n".join(part for part in (self.trusted, self.untrusted) if part)


@dataclass
class Workspace:
    """One agent run's world: the fixture plus the side effects the agent caused."""

    fixture: Fixture = field(default_factory=load_fixture)
    sent: list[dict[str, Any]] = field(default_factory=list)
    tickets: list[dict[str, Any]] = field(default_factory=list)

    def call(self, name: str, arguments: dict[str, Any]) -> ToolOutput:
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return ToolOutput(trusted=f"Error: unknown tool {name!r}.")
        try:
            output: ToolOutput = handler(**arguments)
        except TypeError as exc:
            return ToolOutput(trusted=f"Error: bad arguments for {name}: {exc}")
        return output

    def _search_inbox(self, query: str = "", **_: Any) -> ToolOutput:
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        emails = sorted(self.fixture.emails, key=lambda e: e.date, reverse=True)
        if words:
            scored = []
            for email in emails:
                haystack = f"{email.sender} {email.subject} {email.plain()}".lower()
                score = sum(word in haystack for word in words)
                if score:
                    scored.append((score, email))
            best = [e for _, e in sorted(scored, key=lambda pair: (pair[0], pair[1].date), reverse=True)][:6]
            matches = best or emails[:6]
        else:
            matches = emails[:6]
        header = f"{len(matches)} email(s), most relevant first:" if words else f"{len(matches)} most recent email(s):"
        lines = [f"- {e.id} | {e.date} | from {e.sender} | {e.subject}" for e in matches]
        previews = [f"[{e.id}] {' '.join(e.plain().split())[:160]}" for e in matches]
        return ToolOutput(trusted="\n".join([header, *lines]), untrusted="Previews:\n" + "\n".join(previews))

    def _read_email(self, email_id: str = "", **_: Any) -> ToolOutput:
        email = self.fixture.email(email_id)
        if email is None:
            return ToolOutput(trusted=f"No email with id {email_id!r}.")
        headers = f"Email {email.id}\nFrom: {email.sender}\nDate: {email.date}\nSubject: {email.subject}"
        return ToolOutput(trusted=headers, untrusted=email.body)

    def _search_kb(self, query: str = "", **_: Any) -> ToolOutput:
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        scored = sorted(
            self.fixture.articles,
            key=lambda a: -sum(w in f"{a.title} {a.body}".lower() for w in words),
        )
        top = scored[:2]
        return ToolOutput(
            trusted="Knowledge base results: " + ", ".join(f"{a.id} ({a.title})" for a in top),
            untrusted="\n\n".join(f"[{a.id}] {a.title}\n{a.body.strip()}" for a in top),
        )

    def _lookup_order(self, order_id: str = "", **_: Any) -> ToolOutput:
        order = self.fixture.orders.get(order_id.strip().upper())
        if order is None:
            return ToolOutput(trusted=f"No order {order_id!r}.")
        return ToolOutput(trusted=f"Order {order_id.strip().upper()}: " + json.dumps(order, ensure_ascii=False))

    def _create_ticket(
        self, title: str = "", description: str = "", priority: str = "normal", customer_email: str = "", **_: Any
    ) -> ToolOutput:
        ticket = {
            "id": f"T-{5000 + len(self.tickets) + 1}",
            "title": title,
            "description": description,
            "priority": priority,
            "customer_email": customer_email,
        }
        self.tickets.append(ticket)
        return ToolOutput(trusted=f"Created ticket {ticket['id']} ({priority}): {title}")

    def _send_email(self, to: str = "", subject: str = "", body: str = "", **_: Any) -> ToolOutput:
        self.sent.append({"to": to, "subject": subject, "body": body})
        return ToolOutput(trusted=f"Email sent to {to} (subject: {subject}).")
