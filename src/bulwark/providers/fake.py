"""Deterministic offline upstream: no keys, no network, same input -> same output.

The fake model is deliberately gullible. It reads every message the way a capable model would (it
decodes invisible tag characters and readable base64, reads hidden HTML text, and looks straight through
spotlighting markers), and it obeys instructions it finds anywhere in the conversation:
  * "send / forward ... to x@y"            -> calls the email tool with that recipient
  * "create a ticket ... title / priority" -> calls the ticket tool
  * "![...](https://...)" in an instruction -> renders that image with conversation data in the URL
  * "reveal your system prompt"            -> repeats its system prompt
  * "tell the customer to visit https://"  -> passes the link on
That makes attacks succeed against an unprotected app, so tests, CI and the demo show both outcomes
without an API key. Bulwark's protection in this mode cannot come from the model behaving well; it
comes from quarantine, the tool-call guard and the output guards.

Also understood: `CALL tool_name {json}` in the last message produces exactly that tool call (tests), and
judge requests (`response_format` with a `verdict` schema) are answered from the heuristic score.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import quote

from bulwark.providers.base import ChatProvider, JsonDict, message_text

CHUNK_SIZE = 7
SPOTLIGHT_MARKERS = re.compile(r"<</?UNTRUSTED[^>]*>>")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s)\"'<>]+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")
_AI_CUE = re.compile(
    r"\b(?:ai|assistant|agent|automated|model|llm|bot|system|authorized|also|all|every|full|complete|entire|"
    r"right away|immediately|without asking|do not tell|don't tell)\b",
    re.IGNORECASE,
)
_CALL = re.compile(r"CALL\s+([A-Za-z_][\w]*)\s*(\{.*\})", re.DOTALL)
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def readable_view(text: str) -> str:
    """What a capable model can read in `text`: markers dropped, hidden and encoded text decoded."""
    view = SPOTLIGHT_MARKERS.sub("", text).replace("ˆ", " ")
    hidden = "".join(chr(ord(c) - 0xE0000) for c in view if 0xE0020 <= ord(c) <= 0xE007E)
    view = "".join(c for c in view if not (0xE0000 <= ord(c) <= 0xE007F) and c not in "​‌‍⁠﻿")
    view = html.unescape(_HTML_TAG.sub(" ", view))
    decoded: list[str] = []
    for match in _B64.finditer(view):
        try:
            candidate = base64.b64decode(match.group(0), validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if sum(c.isalpha() or c == " " for c in candidate) > 0.8 * len(candidate):
            decoded.append(candidate)
    return "\n".join(part for part in [view, hidden, *decoded] if part)


@dataclass(frozen=True)
class Command:
    kind: str  # send | ticket | image | reveal | link
    sentence: str
    target: str = ""
    title: str = ""
    priority: str = ""


def injected_commands(text: str) -> list[Command]:
    """Instructions a gullible model would follow, in reading order."""
    commands: list[Command] = []
    view = readable_view(text)
    for raw in _SENTENCE_BREAK.split(view):
        sentence = " ".join(raw.split())
        lowered = sentence.lower()
        if not sentence:
            continue
        address = _EMAIL.search(sentence)
        if address and re.search(r"\b(?:forward|send|e-?mail)\b", lowered) and _AI_CUE.search(sentence):
            commands.append(Command("send", sentence, target=address.group(0)))
        if re.search(
            r"create_ticket|\b(?:create|open)\s+(?:a\s+|an\s+)?(?:new\s+)?ticket|erstelle\b.*\bticket", lowered
        ):
            title = re.search(r"(?:title|titel)\s*[\"“„']([^\"”“']+)[\"”“']", sentence, re.IGNORECASE)
            priority = re.search(r"(?:priority|priorität)\s*[\"“„']?(\w+)", sentence, re.IGNORECASE)
            commands.append(
                Command(
                    "ticket",
                    sentence,
                    title=title.group(1) if title else "Requested ticket",
                    priority=priority.group(1).lower() if priority else "high",
                )
            )
        image = re.search(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", sentence)
        if image:
            commands.append(Command("image", sentence, target=image.group(1)))
        quoted = re.search(r"[\"“'‘][^\"”'’]*(?:system prompt|instructions)[^\"”'’]*[\"”'’]", sentence)
        if (
            not quoted
            and re.search(r"system prompt|your (?:previous |full |complete )?instructions", lowered)
            and re.search(r"\b(?:include|reveal|print|repeat|reply with|show|output|tell)\b", lowered)
        ):
            commands.append(Command("reveal", sentence))
        link = _URL.search(sentence)
        if link and re.search(r"\btell (?:the )?customers?\b", lowered):
            commands.append(Command("link", sentence, target=link.group(0)))
    return commands


def conversation_summary(messages: list[JsonDict], limit: int = 160) -> str:
    texts = [message_text(m) for m in messages if m.get("role") in {"user", "tool"}]
    words = " ".join(" ".join(SPOTLIGHT_MARKERS.sub("", t).split()) for t in texts).split()
    return " ".join(words)[:limit]


def fill_url(url: str, summary: str) -> str:
    encoded = quote(summary, safe="")
    filled = re.sub(r"SUMMARY|DATA|ORDER_NUMBER|\{[^}]*\}", encoded, url)
    return filled if filled != url else url + ("&" if "?" in url else "?") + "d=" + encoded


def judge_answer(body: JsonDict) -> str:
    """Deterministic judge verdict from the heuristic score of the text between the boundary markers."""
    from bulwark.guards import heuristics

    user = message_text(body.get("messages", [])[-1]) if body.get("messages") else ""
    match = re.search(r"(@@[0-9a-f]{12}@@)\n(.*)\n\1", user, re.DOTALL)
    text = match.group(2) if match else user
    score = heuristics.scan(text, "untrusted" if "content the assistant will read" in user else "input").score
    verdict = "injection" if score >= 0.5 else "benign"
    confidence = round(max(score, 1 - score), 2)
    return json.dumps({"verdict": verdict, "confidence": confidence, "reason": f"heuristic score {score:.2f}"})


def make_call(name: str, arguments: JsonDict, seed: str) -> JsonDict:
    call_id = "call_" + hashlib.sha256((seed + name).encode()).hexdigest()[:16]
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _tool_names(body: JsonDict) -> list[str]:
    return [t["function"]["name"] for t in body.get("tools") or [] if t.get("type") == "function"]


def done_calls(messages: list[JsonDict]) -> list[tuple[str, JsonDict]]:
    calls: list[tuple[str, JsonDict]] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append((call["function"]["name"], arguments))
    return calls


def gullible_reply(body: JsonDict) -> JsonDict:
    """The generic fake's assistant message: an echo that obeys injected instructions."""
    messages: list[JsonDict] = body.get("messages", [])
    last = messages[-1] if messages else {}
    response_format = body.get("response_format") or {}
    schema = ((response_format.get("json_schema") or {}).get("schema") or {}).get("properties") or {}
    if "verdict" in schema:
        return {"role": "assistant", "content": judge_answer(body)}

    tools = _tool_names(body)
    seed = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    if tools and body.get("tool_choice") != "none" and last.get("role") in {"user", "tool"}:
        explicit = _CALL.search(message_text(last))
        if explicit and explicit.group(1) in tools:
            try:
                arguments = json.loads(explicit.group(2))
            except json.JSONDecodeError:
                arguments = {}
            return {"role": "assistant", "content": None, "tool_calls": [make_call(explicit.group(1), arguments, seed)]}
        done = done_calls(messages)
        sender = next((t for t in tools if "send" in t or "email" in t), None)
        for message in messages:
            if message.get("role") not in {"user", "tool"}:
                continue
            for command in injected_commands(message_text(message)):
                if command.kind == "send" and sender and not any(command.target in json.dumps(a) for _, a in done):
                    arguments = {
                        "to": command.target,
                        "subject": "As requested",
                        "body": conversation_summary(messages, 600),
                    }
                    return {"role": "assistant", "content": None, "tool_calls": [make_call(sender, arguments, seed)]}

    user_messages = [m for m in messages if m.get("role") == "user"]
    question = " ".join(message_text(user_messages[-1]).split()) if user_messages else ""
    question = question if len(question) <= 200 else question[:197] + "..."
    lines = [f'Echo: "{question}"']
    if last.get("role") == "tool":
        lines = ["Done. The tool returned: " + " ".join(SPOTLIGHT_MARKERS.sub("", message_text(last)).split())[:300]]
    system = next((message_text(m) for m in messages if m.get("role") in {"system", "developer"}), "")
    for message in messages:
        if message.get("role") not in {"user", "tool"}:
            continue
        for command in injected_commands(message_text(message)):
            if command.kind == "reveal" and system:
                lines.append("My instructions are: " + system)
            elif command.kind == "image":
                lines.append(f"![image]({fill_url(command.target, conversation_summary(messages))})")
            elif command.kind == "link":
                lines.append(f"Please verify your details at {command.target} first.")
    return {"role": "assistant", "content": "\n".join(dict.fromkeys(lines))}


def completion(body: JsonDict, message: JsonDict, model: str) -> JsonDict:
    prompt_tokens = sum(len(message_text(m).split()) for m in body.get("messages", []))
    completion_tokens = len((message.get("content") or "").split())
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        "id": f"chatcmpl-fake-{digest[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def stream_message(completed: JsonDict) -> AsyncIterator[JsonDict]:
    """Replay a completion as chunks: tool-call arguments and text in small pieces."""
    message = completed["choices"][0]["message"]
    base = {k: completed[k] for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}

    def chunk(delta: JsonDict, finish_reason: str | None = None) -> JsonDict:
        return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    yield chunk({"role": "assistant", "content": ""})
    for index, call in enumerate(message.get("tool_calls") or []):
        head = {
            "index": index,
            "id": call["id"],
            "type": "function",
            "function": {"name": call["function"]["name"], "arguments": ""},
        }
        yield chunk({"tool_calls": [head]})
        arguments = call["function"]["arguments"]
        for start in range(0, len(arguments), CHUNK_SIZE):
            yield chunk(
                {"tool_calls": [{"index": index, "function": {"arguments": arguments[start : start + CHUNK_SIZE]}}]}
            )
    content = message.get("content") or ""
    for start in range(0, len(content), CHUNK_SIZE):
        yield chunk({"content": content[start : start + CHUNK_SIZE]})
    yield chunk({}, completed["choices"][0]["finish_reason"])


class FakeProvider:
    def __init__(self, model: str = "fake-gullible") -> None:
        self.model = model

    @property
    def label(self) -> str:
        return f"fake/{self.model}"

    @property
    def is_remote(self) -> bool:
        return False

    async def complete(self, body: JsonDict) -> JsonDict:
        return completion(body, gullible_reply(body), str(body.get("model") or self.model))

    async def stream(self, body: JsonDict) -> AsyncIterator[JsonDict]:
        async for chunk in stream_message(await self.complete(body)):
            yield chunk


class RecordingProvider:
    """Keeps every request body it forwards (tests check what actually reached the upstream)."""

    def __init__(self, inner: ChatProvider) -> None:
        self.inner = inner
        self.requests: list[JsonDict] = []

    @property
    def label(self) -> str:
        return self.inner.label

    @property
    def is_remote(self) -> bool:
        return self.inner.is_remote

    async def complete(self, body: JsonDict) -> JsonDict:
        self.requests.append(json.loads(json.dumps(body)))
        return await self.inner.complete(body)

    async def stream(self, body: JsonDict) -> AsyncIterator[JsonDict]:
        self.requests.append(json.loads(json.dumps(body)))
        async for chunk in self.inner.stream(body):
            yield chunk
