"""OpenAI-compatible LLM firewall (FastAPI).

Point any OpenAI SDK at it (`base_url="http://localhost:8000/v1"`): requests go through the input and
untrusted-content guards, are forwarded to the configured upstream, and answers come back through the
output and tool-call guards. Streaming (SSE) is supported. Pick a policy per route (`/r/email-agent/v1`),
per header (`X-Bulwark-Policy`) or per tenant API key.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from bulwark import __version__
from bulwark.audit import AuditRecord
from bulwark.config import Settings
from bulwark.core import Action, ApprovalRequest, Decision
from bulwark.gateway.messages import (
    PreparedRequest,
    UntrustedCache,
    bulwark_extension,
    guard_tool_calls,
    prepare_request,
    tool_call_arguments,
    tool_notice,
)
from bulwark.gateway.runtime import Runtime, Tenant, build_runtime
from bulwark.logging_config import configure_logging, get_logger
from bulwark.policy import Policy, PolicyError
from bulwark.providers.base import JsonDict, ProviderError
from bulwark.stream import StreamGuard
from bulwark.taint import TaintState

log = get_logger(__name__)

POLICY_HEADER = "X-Bulwark-Policy"
REQUEST_HEADER = "X-Bulwark-Request-Id"
ACTION_HEADER = "X-Bulwark-Action"


def openai_error(status: int, message: str, error_type: str, code: str | None = None, **extra: Any) -> JSONResponse:
    error: JsonDict = {"message": message, "type": error_type, "param": None, "code": code, **extra}
    return JSONResponse({"error": error}, status_code=status)


class ScanRequest(BaseModel):
    stage: Literal["input", "untrusted", "output", "tool_call"] = "input"
    text: str = ""
    policy: str | None = None
    source: str = "document"
    """untrusted: where the content came from (tool name, URL, document id)."""
    system_prompt: str | None = None
    """output: the system prompt to compare against (leakage)."""
    tool: str | None = None
    arguments: dict[str, Any] | str | None = None
    untrusted_context: list[str] = Field(default_factory=list)
    """tool_call: untrusted content the agent has read so far (taints the session)."""
    trusted_context: list[str] = Field(default_factory=list)
    """tool_call: the user's messages and other trusted text (argument provenance)."""


class ApprovalDecision(BaseModel):
    decision: Literal["approve", "deny"]


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def resolve_tenant(runtime: Runtime, authorization: str | None) -> Tenant | None:
    if not runtime.tenants.enabled:
        return None
    tenant = runtime.tenants.lookup(_bearer(authorization))
    if tenant is None:
        raise HTTPException(status_code=401, detail="invalid or missing gateway API key")
    return tenant


def resolve_policy(
    runtime: Runtime, tenant: Tenant | None, route_policy: str | None, header_policy: str | None
) -> Policy:
    requested = route_policy or (header_policy if runtime.settings.allow_policy_header else None)
    if tenant is not None:
        if requested and requested != tenant.policy and requested not in tenant.allowed_policies:
            raise HTTPException(status_code=403, detail=f"tenant {tenant.name!r} may not use policy {requested!r}")
        requested = requested or tenant.policy
    try:
        return runtime.firewall.policies.get(requested or runtime.settings.default_policy)
    except PolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _audit_request(runtime: Runtime, record: AuditRecord, prepared: PreparedRequest, body: JsonDict) -> None:
    if prepared.input_decision:
        last_user: JsonDict = next((m for m in reversed(body["messages"]) if m.get("role") == "user"), {})
        content = last_user.get("content")
        runtime.audit.add_decision(record, prepared.input_decision, text=content if isinstance(content, str) else None)
    for source, decision, report in prepared.untrusted:
        runtime.audit.add_decision(record, decision, text=report.sanitized if report.flagged else None, source=source)
    record.taint = prepared.session.taint.level
    record.untrusted_sources = len(prepared.untrusted)


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or (runtime.settings if runtime else Settings())
    configure_logging(settings.log_level, settings.log_format)
    runtime = runtime or build_runtime(settings)
    cache = UntrustedCache()
    app = FastAPI(
        title="Bulwark",
        version=__version__,
        description="An LLM firewall against prompt injection, jailbreaks and data exfiltration.",
    )
    app.state.runtime = runtime

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        kind = {401: "authentication_error", 403: "permission_error"}.get(exc.status_code, "invalid_request_error")
        body = {"error": {"message": str(exc.detail), "type": kind, "param": None, "code": None}, "detail": exc.detail}
        return JSONResponse(body, status_code=exc.status_code)

    async def chat_completions(
        request: Request,
        body: JsonDict,
        route_policy: str | None,
        authorization: str | None,
        policy_header: str | None,
    ) -> Any:
        started = time.perf_counter()
        tenant = resolve_tenant(runtime, authorization)
        policy = resolve_policy(runtime, tenant, route_policy, policy_header)
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            return openai_error(400, "`messages` must be a non-empty list", "invalid_request_error")
        record = runtime.audit.new_record(
            route="chat.completions",
            policy=policy.name,
            request=await request.body(),
            tenant=tenant.name if tenant else None,
        )
        record.upstream = runtime.upstream.label
        record.stream = bool(body.get("stream"))
        prepared = await prepare_request(runtime.firewall, policy, body, cache)
        _audit_request(runtime, record, prepared, body)
        headers = {POLICY_HEADER: policy.name, REQUEST_HEADER: record.id}

        if prepared.input_decision is not None and prepared.input_decision.blocked:
            decision = prepared.input_decision
            record.total_ms = _ms(started)
            headers[ACTION_HEADER] = "block"
            log.info("request.blocked", policy=policy.name, guards=[r.guard for r in decision.triggered])
            if policy.block_response == "error":
                record.status = 400
                runtime.audit.add(record)
                response = openai_error(
                    400,
                    f"Request refused by Bulwark policy {policy.name!r}: {decision.explain()}",
                    "bulwark_policy_violation",
                    "input_blocked",
                    bulwark=bulwark_extension(policy, prepared.decisions, [], record.id),
                )
                response.headers.update(headers)
                return response
            runtime.audit.add(record)
            refusal = _refusal(body, policy, prepared.decisions, record.id)
            if body.get("stream"):
                return StreamingResponse(_replay(refusal), media_type="text/event-stream", headers=headers)
            return JSONResponse(refusal, headers=headers)

        upstream_body = prepared.body
        if body.get("stream"):
            return StreamingResponse(
                _stream(runtime, policy, prepared, upstream_body, record, started),
                media_type="text/event-stream",
                headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        upstream_started = time.perf_counter()
        try:
            answer = await runtime.upstream.complete(upstream_body)
        except ProviderError as exc:
            record.status = exc.status_code
            record.total_ms = _ms(started)
            runtime.audit.add(record)
            log.warning("upstream.error", status=exc.status_code, error=str(exc)[:200])
            response = openai_error(exc.status_code, str(exc), "upstream_error")
            response.headers.update(headers)
            return response
        record.upstream_ms = _ms(upstream_started)
        record.upstream_model = answer.get("model")
        answer.pop("_cached", None)
        decisions = list(prepared.decisions)
        approvals: list[ApprovalRequest] = []
        for choice in answer.get("choices") or []:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content:
                decision = await prepared.session.check_output(content)
                runtime.audit.add_decision(record, decision, text=content)
                decisions.append(decision)
                if decision.blocked:
                    message["content"] = f"[Bulwark blocked this answer: {decision.explain()}]"
                    choice["finish_reason"] = "content_filter"
                elif decision.text is not None:
                    message["content"] = decision.text
            calls = message.get("tool_calls") or []
            if calls:
                outcome = guard_tool_calls(prepared.session, calls)
                for call, decision in outcome.decisions:
                    name = (call.get("function") or {}).get("name")
                    runtime.audit.add_decision(record, decision, tool=name, arguments=tool_call_arguments(call))
                    decisions.append(decision)
                for approval in outcome.withheld:
                    runtime.approvals.add(approval)
                approvals += outcome.withheld
                if outcome.withheld or outcome.blocked:
                    message["tool_calls"] = outcome.passed or None
                    if not outcome.passed:
                        message.pop("tool_calls", None)
                        choice["finish_reason"] = "stop"
                    notice = tool_notice(outcome)
                    message["content"] = f"{message.get('content') or ''}\n{notice}".strip()
        extension = bulwark_extension(policy, decisions, approvals, record.id)
        answer["bulwark"] = extension
        headers[ACTION_HEADER] = extension["action"]
        record.total_ms = _ms(started)
        runtime.audit.add(record)
        log.info("request.done", policy=policy.name, action=extension["action"], total_ms=record.total_ms)
        return JSONResponse(answer, headers=headers)

    @app.post("/v1/chat/completions", tags=["OpenAI-compatible"])
    async def chat(
        request: Request,
        body: Annotated[JsonDict, Body()],
        authorization: Annotated[str | None, Header()] = None,
        x_bulwark_policy: Annotated[str | None, Header()] = None,
    ) -> Any:
        return await chat_completions(request, body, None, authorization, x_bulwark_policy)

    @app.post("/r/{policy}/v1/chat/completions", tags=["OpenAI-compatible"])
    async def chat_with_policy(
        policy: str,
        request: Request,
        body: Annotated[JsonDict, Body()],
        authorization: Annotated[str | None, Header()] = None,
    ) -> Any:
        return await chat_completions(request, body, policy, authorization, None)

    @app.get("/v1/models", tags=["OpenAI-compatible"])
    @app.get("/r/{policy}/v1/models", tags=["OpenAI-compatible"], include_in_schema=False)
    async def models(policy: str | None = None) -> JsonDict:
        upstream = runtime.upstream
        model_ids = [getattr(upstream, "default_model", None) or getattr(upstream, "model", None) or upstream.label]
        model_ids += list(getattr(upstream, "fallback_models", []))
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "bulwark"} for m in model_ids]}

    @app.post("/v1/scan", tags=["Firewall"])
    async def scan(payload: ScanRequest, authorization: Annotated[str | None, Header()] = None) -> JsonDict:
        """Run one stage on one text (the library over HTTP): RAG chunks, tool results, answers, tool calls."""
        tenant = resolve_tenant(runtime, authorization)
        policy = resolve_policy(runtime, tenant, None, payload.policy)
        record = runtime.audit.new_record(route=f"scan.{payload.stage}", policy=policy.name, request=payload.text)
        result = await run_scan(runtime, policy, payload)
        decision = Decision.model_validate(result["decision"])
        runtime.audit.add_decision(
            record, decision, text=payload.text, source=payload.source if payload.stage == "untrusted" else None
        )
        record.total_ms = decision.latency_ms
        runtime.audit.add(record)
        return {"request_id": record.id, **result}

    @app.get("/v1/approvals", tags=["Firewall"])
    async def list_approvals(authorization: Annotated[str | None, Header()] = None) -> JsonDict:
        resolve_tenant(runtime, authorization)
        return {"approvals": [a.model_dump() for a in runtime.approvals.list()]}

    @app.post("/v1/approvals/{approval_id}", tags=["Firewall"])
    async def decide_approval(
        approval_id: str, payload: ApprovalDecision, authorization: Annotated[str | None, Header()] = None
    ) -> JsonDict:
        """Approve or deny a held tool call. An approved call is returned so the client can execute it."""
        resolve_tenant(runtime, authorization)
        request = runtime.approvals.decide(approval_id, payload.decision == "approve")
        if request is None:
            raise HTTPException(status_code=404, detail="unknown or expired approval id")
        record = runtime.audit.new_record(route="approval", policy="-", request=approval_id)
        record.action = "allow" if request.status == "approved" else "block"
        runtime.audit.add(record)
        result: JsonDict = {"approval": request.model_dump()}
        if request.status == "approved":
            result["tool_call"] = {
                "id": "call_" + request.id[4:],
                "type": "function",
                "function": {"name": request.tool, "arguments": json.dumps(request.arguments)},
            }
        return result

    @app.get("/audit", tags=["Firewall"])
    async def audit(limit: int = 100, authorization: Annotated[str | None, Header()] = None) -> JsonDict:
        """Recent decisions: guard scores, actions, rules, hashes and short redacted snippets."""
        resolve_tenant(runtime, authorization)
        records = runtime.audit.recent(min(max(limit, 1), 1000))
        return {"summary": runtime.audit.summary(), "records": [r.model_dump(mode="json") for r in records]}

    @app.get("/health", tags=["ops"])
    async def health() -> JsonDict:
        return {
            "status": "ok",
            "version": __version__,
            "upstream": runtime.upstream.label,
            "require_free_models": settings.require_free_models,
            "policies": runtime.firewall.policies.names,
            "default_policy": runtime.firewall.policies.default_name,
            "layers": runtime.firewall.detector.available_layers(),
            "classifier": runtime.classifier_status,
            "judge": runtime.judge_status,
            "pii_shield": runtime.pii_shield_status,
            "tenants": runtime.tenants.enabled,
        }

    from bulwark.dashboard.views import register_dashboard

    register_dashboard(app, runtime)
    return app


async def run_scan(runtime: Runtime, policy: Policy, payload: ScanRequest) -> JsonDict:
    firewall = runtime.firewall
    if payload.stage == "input":
        decision = await firewall.check_input(payload.text, policy=policy)
        return {"decision": decision.model_dump(mode="json"), "text": payload.text}
    if payload.stage == "untrusted":
        decision, report = await firewall.check_untrusted(payload.text, source=payload.source, policy=policy)
        return {"decision": decision.model_dump(mode="json"), "text": decision.text, "report": report.model_dump()}
    if payload.stage == "output":
        decision = await firewall.check_output(payload.text, policy=policy, system_prompt=payload.system_prompt)
        return {"decision": decision.model_dump(mode="json"), "text": decision.text or payload.text}
    taint = TaintState()
    for text in payload.trusted_context:
        taint.add_trusted(text)
    for text in payload.untrusted_context:
        report = (await firewall.check_untrusted(text, source="context", policy=policy))[1]
        taint.add_untrusted(text, "context", flagged=report.flagged, score=report.score)
    decision = firewall.check_tool_call(payload.tool or "", payload.arguments, taint=taint, policy=policy)
    if decision.approval:
        runtime.approvals.add(decision.approval)
    return {"decision": decision.model_dump(mode="json"), "taint": taint.level}


def _refusal(body: JsonDict, policy: Policy, decisions: list[Decision], request_id: str) -> JsonDict:
    return {
        "id": f"chatcmpl-bulwark-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(body.get("model") or "bulwark"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": policy.refusal_message},
                "finish_reason": "content_filter",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "bulwark": bulwark_extension(policy, decisions, [], request_id),
    }


async def _replay(completion: JsonDict) -> AsyncIterator[str]:
    base = {k: completion[k] for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    content = completion["choices"][0]["message"]["content"]
    for delta, finish in (({"role": "assistant", "content": content}, None), ({}, "content_filter")):
        chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if finish:
            chunk["bulwark"] = completion["bulwark"]
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _sse(chunk: JsonDict) -> str:
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def _stream(
    runtime: Runtime, policy: Policy, prepared: PreparedRequest, body: JsonDict, record: AuditRecord, started: float
) -> AsyncIterator[str]:
    session = prepared.session
    guard = StreamGuard(
        runtime.firewall,
        policy,
        system_prompt=session.system_prompt,
        canary=session.canary,
        conversation=session.conversation_text(),
    )
    # PII Shield is an HTTP call: with it on, the answer is checked whole and then released.
    buffered = policy.output.secrets.pii_shield and runtime.firewall.pii_shield is not None
    base: JsonDict = {"object": "chat.completion.chunk"}
    calls: dict[int, JsonDict] = {}
    finish_reason: str | None = None
    held: list[str] = []
    upstream_started = time.perf_counter()

    def content_chunk(text: str) -> JsonDict:
        return {**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}

    try:
        async for chunk in runtime.upstream.stream(body):
            if "id" not in base and chunk.get("id"):
                base.update({k: chunk[k] for k in ("id", "created", "model") if k in chunk})
                record.upstream_model = chunk.get("model")
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("role") and not delta.get("content") and not delta.get("tool_calls"):
                    yield _sse(
                        {
                            **base,
                            "choices": [
                                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                            ],
                        }
                    )
                text = delta.get("content")
                if isinstance(text, str) and text:
                    if buffered:
                        held.append(text)
                    else:
                        step = await guard.push(text)
                        if step.text:
                            yield _sse(content_chunk(step.text))
                        if step.stopped:
                            finish_reason = "content_filter"
                            break
                for part in delta.get("tool_calls") or []:
                    slot = calls.setdefault(
                        int(part.get("index", 0)),
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    slot["id"] = part.get("id") or slot["id"]
                    function = part.get("function") or {}
                    slot["function"]["name"] += function.get("name") or ""
                    slot["function"]["arguments"] += function.get("arguments") or ""
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if guard.stopped:
                break
    except ProviderError as exc:
        record.status = exc.status_code
        log.warning("upstream.stream_error", status=exc.status_code, error=str(exc)[:200])
        yield _sse({"error": {"message": str(exc), "type": "upstream_error", "code": exc.status_code}})
        yield "data: [DONE]\n\n"
        _close(runtime, record, upstream_started, started)
        return

    decisions = list(prepared.decisions)
    if buffered and held:
        text = "".join(held)
        decision = await session.check_output(text)
        decisions.append(decision)
        runtime.audit.add_decision(record, decision, text=text)
        if decision.blocked:
            yield _sse(content_chunk(f"[Bulwark blocked this answer: {decision.explain()}]"))
            finish_reason = "content_filter"
        else:
            yield _sse(content_chunk(decision.text or text))
    elif not guard.stopped:
        tail = await guard.finish()
        if tail.text:
            yield _sse(content_chunk(tail.text))
        if tail.stopped:
            finish_reason = "content_filter"
    if not buffered and (guard.results or guard.raw):
        decision = guard.decision()
        decisions.append(decision)
        runtime.audit.add_decision(record, decision, text="".join(guard.raw))

    approvals: list[ApprovalRequest] = []
    if calls and finish_reason != "content_filter":
        outcome = guard_tool_calls(session, [calls[i] for i in sorted(calls)])
        for call, decision in outcome.decisions:
            runtime.audit.add_decision(
                record, decision, tool=(call.get("function") or {}).get("name"), arguments=tool_call_arguments(call)
            )
            decisions.append(decision)
        for index, call in enumerate(outcome.passed):
            yield _sse(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": {"tool_calls": [{"index": index, **call}]}, "finish_reason": None}
                    ],
                }
            )
        for approval in outcome.withheld:
            runtime.approvals.add(approval)
        approvals = outcome.withheld
        if outcome.withheld or outcome.blocked:
            yield _sse(content_chunk(("\n" if guard.emitted else "") + tool_notice(outcome)))
        finish_reason = "tool_calls" if outcome.passed else "stop"
    extension = bulwark_extension(policy, decisions, approvals, record.id)
    final = {
        **base,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}],
        "bulwark": extension,
    }
    yield _sse(final)
    yield "data: [DONE]\n\n"
    _close(runtime, record, upstream_started, started)


def _close(runtime: Runtime, record: AuditRecord, upstream_started: float, started: float) -> None:
    record.upstream_ms = _ms(upstream_started)
    record.total_ms = _ms(started)
    if record.action == Action.ALLOW.value and record.status >= 400:
        record.action = "error"
    runtime.audit.add(record)


def create_default_app() -> FastAPI:
    """Factory for `uvicorn bulwark.gateway.app:create_default_app --factory`."""
    return create_app(Settings())
