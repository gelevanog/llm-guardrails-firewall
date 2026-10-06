"""Anthropic Messages API behind the OpenAI-compatible gateway (official `anthropic` SDK).

Converts OpenAI chat.completions bodies (system/user/assistant/tool messages, tool calls, tools,
tool_choice) to a Messages request and the answer back, including streaming. Sampling parameters
(`temperature`, `top_p`) are dropped: current Claude models reject them.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from bulwark.providers.base import JsonDict, ProviderError, RetryableError, message_text

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
# These models reject forced tool use (`tool_choice` any/tool -> 400); "required" degrades to "auto" for them.
_NO_FORCED_TOOL_CHOICE = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1")
_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _content_blocks(message: JsonDict) -> list[JsonDict]:
    blocks: list[JsonDict] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and part.get("text"):
                blocks.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:") and ";base64," in url:
                    media_type, data = url[5:].split(";base64,", 1)
                    blocks.append(
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
                    )
                elif url:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def to_anthropic_request(body: JsonDict, default_model: str) -> JsonDict:
    """OpenAI chat.completions body -> kwargs for `client.messages.create`."""
    system_parts: list[str] = []
    messages: list[JsonDict] = []

    def append(role: str, blocks: list[JsonDict]) -> None:
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for message in body.get("messages", []):
        role = message.get("role")
        if role in {"system", "developer"}:
            system_parts.append(message_text(message))
        elif role == "user":
            append("user", _content_blocks(message))
        elif role == "assistant":
            blocks = _content_blocks(message)
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": function.get("arguments", "")}
                blocks.append(
                    {"type": "tool_use", "id": call["id"], "name": function.get("name", ""), "input": arguments}
                )
            append("assistant", blocks)
        elif role == "tool":
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": message_text(message),
                    }
                ],
            )

    model = str(body.get("model") or "")
    request: JsonDict = {
        "model": model if model.startswith("claude-") else default_model,
        "max_tokens": int(body.get("max_completion_tokens") or body.get("max_tokens") or 16000),
        "messages": messages,
    }
    if system_parts:
        request["system"] = "\n\n".join(part for part in system_parts if part)
    if body.get("stop"):
        stop = body["stop"]
        request["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    tools = [tool["function"] for tool in body.get("tools") or [] if tool.get("type") == "function"]
    if tools:
        request["tools"] = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
            }
            for tool in tools
        ]
        choice = body.get("tool_choice")
        forced_allowed = not request["model"].startswith(_NO_FORCED_TOOL_CHOICE)
        if choice == "none":
            request["tool_choice"] = {"type": "none"}
        elif choice == "required" and forced_allowed:
            request["tool_choice"] = {"type": "any"}
        elif isinstance(choice, dict) and choice.get("type") == "function" and forced_allowed:
            request["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
    response_format = body.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if schema:
            request["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    return request


def from_anthropic_message(message: Any, created: int) -> JsonDict:
    text = "".join(block.text for block in message.content if block.type == "text")
    tool_calls = [
        {"id": block.id, "type": "function", "function": {"name": block.name, "arguments": json.dumps(block.input)}}
        for block in message.content
        if block.type == "tool_use"
    ]
    chat_message: JsonDict = {"role": "assistant", "content": text or None}
    if tool_calls:
        chat_message["tool_calls"] = tool_calls
    usage = message.usage
    return {
        "id": message.id,
        "object": "chat.completion",
        "created": created,
        "model": message.model,
        "choices": [
            {"index": 0, "message": chat_message, "finish_reason": _FINISH.get(message.stop_reason or "", "stop")}
        ],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        },
    }


def _translate_error(exc: Exception) -> ProviderError:
    if isinstance(exc, anthropic.RateLimitError):
        return RetryableError(f"rate limited: {exc.message}", status_code=429)
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500 or exc.status_code in {408, 409, 529}:
            return RetryableError(f"upstream {exc.status_code}: {exc.message}")
        return ProviderError(f"upstream {exc.status_code}: {exc.message}", exc.status_code)
    if isinstance(exc, anthropic.APIConnectionError):
        return RetryableError(f"upstream connection error: {exc}")
    return ProviderError(str(exc))


class AnthropicProvider:
    def __init__(
        self, *, api_key: str | None, default_model: str = DEFAULT_ANTHROPIC_MODEL, timeout_seconds: float = 120.0
    ) -> None:
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set", 500)
        self.default_model = default_model
        # SDK retries off: the gateway's resilient wrapper (or the client) decides about retries.
        self._client = AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout_seconds)

    @property
    def label(self) -> str:
        return f"anthropic/{self.default_model}"

    @property
    def is_remote(self) -> bool:
        return True

    async def complete(self, body: JsonDict) -> JsonDict:
        request = to_anthropic_request(body, self.default_model)
        try:
            message = await self._client.messages.create(**request)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            raise _translate_error(exc) from exc
        return from_anthropic_message(message, int(time.time()))

    async def stream(self, body: JsonDict) -> AsyncIterator[JsonDict]:
        request = to_anthropic_request(body, self.default_model)
        created = int(time.time())
        message_id, model = "", request["model"]
        tool_index = -1
        block_tool: dict[int, int] = {}

        def chunk(delta: JsonDict, finish_reason: str | None = None) -> JsonDict:
            return {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        try:
            async with self._client.messages.stream(**request) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        message_id, model = event.message.id, event.message.model
                        yield chunk({"role": "assistant", "content": ""})
                    elif event.type == "content_block_start" and event.content_block.type == "tool_use":
                        tool_index += 1
                        block_tool[event.index] = tool_index
                        call = {
                            "index": tool_index,
                            "id": event.content_block.id,
                            "type": "function",
                            "function": {"name": event.content_block.name, "arguments": ""},
                        }
                        yield chunk({"tool_calls": [call]})
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield chunk({"content": event.delta.text})
                        elif event.delta.type == "input_json_delta" and event.index in block_tool:
                            call = {
                                "index": block_tool[event.index],
                                "function": {"arguments": event.delta.partial_json},
                            }
                            yield chunk({"tool_calls": [call]})
                    elif event.type == "message_delta":
                        yield chunk({}, _FINISH.get(event.delta.stop_reason or "", "stop"))
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            raise _translate_error(exc) from exc
