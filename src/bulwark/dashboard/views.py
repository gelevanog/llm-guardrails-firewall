"""Dashboard: playground, agent demo, evaluation results and the audit log. Jinja2 + htmx, no build step."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from bulwark.agent_demo.agent import AgentRun, run_agent
from bulwark.agent_demo.fake_model import GullibleAgentModel
from bulwark.agent_demo.scenarios import Score, load_scenarios, scenario, score_run, simulated_reviewer
from bulwark.core import Decision, Finding
from bulwark.dashboard.examples import EXAMPLES, Example
from bulwark.gateway.messages import prepare_request
from bulwark.gateway.runtime import Runtime
from bulwark.guards.untrusted import UntrustedReport
from bulwark.normalize import decoded_views, extract_html
from bulwark.policy import Policy
from bulwark.providers.base import ChatProvider, ProviderError
from bulwark.providers.fake import FakeProvider
from bulwark.taint import TaintState

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
PLAYGROUND_SYSTEM = "You are a helpful support assistant for Acme Outdoor, an outdoor-gear shop."


@dataclass
class Segment:
    text: str
    kind: str | None = None
    title: str = ""


@dataclass
class PlaygroundView:
    stage: str
    policy: str
    decision: Decision
    text: str
    segments: list[Segment]
    model_text: str | None = None
    report: UntrustedReport | None = None
    decoded: list[dict[str, str]] = field(default_factory=list)
    taint: str | None = None
    provider: str = "none"
    model: str | None = None
    raw_answer: str | None = None
    answer: str | None = None
    output_decision: Decision | None = None
    llm_ms: float | None = None
    error: str | None = None


def highlight(text: str, findings: list[Finding], removed: list[tuple[int, int]] | None = None) -> list[Segment]:
    """Split `text` into plain and highlighted segments (rule matches, removed spans)."""
    marks: list[tuple[int, int, str, str]] = []
    for start, end in removed or []:
        marks.append((start, end, "removed", "removed by Bulwark"))
    for finding in findings:
        if finding.start is None or finding.end is None or finding.end - finding.start >= len(text):
            continue
        marks.append((finding.start, finding.end, "hit", f"{finding.rule} · {finding.score:.2f} · {finding.view}"))
    segments: list[Segment] = []
    cursor = 0
    for start, end, kind, title in sorted(marks, key=lambda m: (m[0], -m[1])):
        if start < cursor:
            continue
        segments.append(Segment(text[cursor:start]))
        segments.append(Segment(text[start:end], kind, title))
        cursor = end
    segments.append(Segment(text[cursor:]))
    return [s for s in segments if s.text]


def with_layers(policy: Policy, classifier: bool, judge: bool) -> Policy:
    """A copy of the policy with the classifier / judge layers switched as the playground asks."""
    copy = policy.model_copy(deep=True)
    for layers in (copy.input.injection.layers, copy.untrusted.layers):
        layers.classifier = layers.classifier and classifier
        layers.judge = judge
    return copy


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


async def run_playground(
    runtime: Runtime,
    *,
    stage: str,
    text: str,
    policy_name: str,
    classifier: bool,
    judge: bool,
    provider_key: str,
    source: str = "document",
    system_prompt: str = "",
    tool: str = "",
    arguments: str = "",
    context: str = "",
) -> PlaygroundView:
    firewall = runtime.firewall
    base = firewall.policy(policy_name if policy_name in firewall.policies.names else None)
    policy = with_layers(base, classifier, judge)
    record = runtime.audit.new_record(route=f"playground.{stage}", policy=policy.name, request=text + arguments)
    view: PlaygroundView
    if stage == "untrusted":
        decision, report = await firewall.check_untrusted(text, source=source or "document", policy=policy)
        visible = extract_html(text).visible
        view = PlaygroundView(
            stage=stage,
            policy=policy.name,
            decision=decision,
            text=visible,
            segments=highlight(
                visible,
                decision.results[0].findings if decision.results else [],
                [(r.start, r.end) for r in report.removals],
            ),
            model_text=decision.text,
            report=report,
        )
    elif stage == "output":
        decision = await firewall.check_output(text, policy=policy, system_prompt=system_prompt or None)
        findings = [f for r in decision.results for f in r.findings]
        view = PlaygroundView(
            stage, policy.name, decision, text, highlight(text, findings), model_text=decision.text or text
        )
    elif stage == "tool_call":
        taint = TaintState()
        taint.add_trusted(text)
        for chunk in [c.strip() for c in context.split("\n---\n") if c.strip()]:
            _, report = await firewall.check_untrusted(chunk, source="context", policy=policy)
            taint.add_untrusted(chunk, "context", flagged=report.flagged, score=report.score)
        decision = firewall.check_tool_call(tool, arguments or "{}", taint=taint, policy=policy)
        view = PlaygroundView(stage, policy.name, decision, arguments, [Segment(arguments)], taint=taint.level)
        if decision.approval:
            runtime.approvals.add(decision.approval)
    else:
        decision = await firewall.check_input(text, policy=policy)
        findings = [f for r in decision.results for f in r.findings]
        view = PlaygroundView(stage, policy.name, decision, text, highlight(text, findings))
    view.decoded = (
        [{"kind": d.kind, "text": d.text[:400]} for d in decoded_views(text, rot13=False, reverse=False)]
        if stage in {"input", "untrusted"}
        else []
    )
    runtime.audit.add_decision(record, view.decision, text=view.text)

    provider = _provider(runtime, provider_key)
    if stage == "input" and provider is not None and not view.decision.blocked:
        view.provider = provider_key
        body = {
            "model": "auto",
            "messages": [{"role": "system", "content": PLAYGROUND_SYSTEM}, {"role": "user", "content": text}],
            "max_tokens": 1500,
        }
        prepared = await prepare_request(firewall, policy, body)
        started = time.perf_counter()
        try:
            answer = await provider.complete(prepared.body)
        except ProviderError as exc:
            view.error = str(exc)[:300]
        else:
            view.llm_ms = round((time.perf_counter() - started) * 1000, 1)
            view.model = answer.get("model")
            record.upstream, record.upstream_model, record.upstream_ms = provider.label, view.model, view.llm_ms
            raw = ((answer.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            view.raw_answer = raw
            output = await prepared.session.check_output(raw)
            view.output_decision = output
            runtime.audit.add_decision(record, output, text=raw)
            view.answer = (
                f"[Bulwark blocked this answer: {output.explain()}]" if output.blocked else (output.text or raw)
            )
    record.total_ms = round(view.decision.latency_ms + (view.llm_ms or 0.0), 2)
    runtime.audit.add(record)
    return view


def _provider(runtime: Runtime, key: str) -> ChatProvider | None:
    if key == "fake":
        return FakeProvider()
    if key == "real":
        return runtime.real_provider
    return None


@dataclass
class AgentComparison:
    scenario_id: str
    task: str
    kind: str
    model: str
    runs: list[tuple[AgentRun, Score]]


async def run_comparison(runtime: Runtime, scenario_id: str, model_key: str, reviewer: str) -> AgentComparison:
    item = scenario(scenario_id)
    provider: ChatProvider = GullibleAgentModel()
    if model_key == "real" and runtime.real_provider is not None:
        provider = (
            runtime.real_provider.with_tag("agent_demo")
            if hasattr(runtime.real_provider, "with_tag")
            else runtime.real_provider
        )
    approver = simulated_reviewer(item) if reviewer == "simulated" else None
    runs: list[tuple[AgentRun, Score]] = []
    for firewall in (None, runtime.firewall):
        run = await run_agent(item.task, provider=provider, firewall=firewall, approver=approver, scenario=item.id)
        runs.append((run, score_run(item, run)))
        if firewall is not None:
            record = runtime.audit.new_record(route="agent_demo", policy="email-agent", request=item.task)
            record.upstream, record.upstream_model = provider.label, run.model
            for decision in run.decisions:
                runtime.audit.add_decision(record, decision)
            for approval in run.approvals:
                runtime.approvals.add(approval)
            record.total_ms, record.upstream_ms = run.total_ms, run.model_ms
            runtime.audit.add(record)
    return AgentComparison(item.id, item.task, item.kind, provider.label, runs)


def register_dashboard(app: FastAPI, runtime: Runtime) -> None:
    def context(request: Request, page: str, **extra: Any) -> dict[str, Any]:
        return {
            "request": request,
            "page": page,
            "upstream": runtime.upstream.label,
            "classifier_status": runtime.classifier_status,
            "classifier_ready": runtime.firewall.classifier is not None,
            "judge_ready": runtime.firewall.detector.judge is not None,
            "real_ready": runtime.real_provider is not None,
            "real_label": runtime.real_provider.label if runtime.real_provider else None,
            "policies": runtime.firewall.policies.names,
            **extra,
        }

    def examples_by_group() -> dict[str, list[Example]]:
        groups: dict[str, list[Example]] = {}
        for example in EXAMPLES:
            groups.setdefault(example.group, []).append(example)
        return groups

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def playground_page(request: Request, example: str | None = None) -> HTMLResponse:
        chosen = next((e for e in EXAMPLES if e.id == example), EXAMPLES[0])
        return TEMPLATES.TemplateResponse(
            request,
            "playground.html",
            context(
                request,
                "playground",
                groups=examples_by_group(),
                examples=EXAMPLES,
                chosen=chosen,
                policy="email-agent" if chosen.stage == "tool_call" else "default",
                result=None,
                provider="none",
                classifier=True,
                judge=False,
            ),
        )

    @app.post("/playground", response_class=HTMLResponse, include_in_schema=False)
    async def playground_run(
        request: Request,
        stage: Annotated[str, Form()],
        policy: Annotated[str, Form()],
        text: Annotated[str, Form()] = "",
        source: Annotated[str, Form()] = "document",
        system_prompt: Annotated[str, Form()] = "",
        tool: Annotated[str, Form()] = "",
        arguments: Annotated[str, Form()] = "",
        tool_context: Annotated[str, Form()] = "",
        provider: Annotated[str, Form()] = "none",
        classifier: Annotated[str | None, Form()] = None,
        judge: Annotated[str | None, Form()] = None,
    ) -> HTMLResponse:
        result = await run_playground(
            runtime,
            stage=stage,
            text=text,
            policy_name=policy,
            classifier=classifier is not None,
            judge=judge is not None,
            provider_key=provider,
            source=source,
            system_prompt=system_prompt,
            tool=tool,
            arguments=arguments,
            context=tool_context,
        )
        partial = request.headers.get("HX-Request") == "true"
        chosen = Example(
            id="custom",
            group="",
            label="",
            stage=stage,
            text=text,
            source=source,
            system_prompt=system_prompt,
            tool=tool,
            arguments=arguments,
            context=[tool_context] if tool_context else [],
        )
        return TEMPLATES.TemplateResponse(
            request,
            "partials/scan_result.html" if partial else "playground.html",
            context(
                request,
                "playground",
                groups=examples_by_group(),
                examples=EXAMPLES,
                chosen=chosen,
                policy=policy,
                result=result,
                provider=provider,
                classifier=classifier is not None,
                judge=judge is not None,
            ),
        )

    @app.get("/agent", response_class=HTMLResponse, include_in_schema=False)
    async def agent_page(request: Request, scenario_id: str = "forward-invoices") -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "agent.html",
            context(
                request,
                "agent",
                scenarios=load_scenarios(),
                selected=scenario_id,
                comparison=None,
                model="fake",
                reviewer="pause",
            ),
        )

    @app.post("/agent/run", response_class=HTMLResponse, include_in_schema=False)
    async def agent_run(
        request: Request,
        scenario_id: Annotated[str, Form()],
        model: Annotated[str, Form()] = "fake",
        reviewer: Annotated[str, Form()] = "pause",
    ) -> HTMLResponse:
        comparison = await run_comparison(runtime, scenario_id, model, reviewer)
        partial = request.headers.get("HX-Request") == "true"
        return TEMPLATES.TemplateResponse(
            request,
            "partials/agent_result.html" if partial else "agent.html",
            context(
                request,
                "agent",
                scenarios=load_scenarios(),
                selected=scenario_id,
                comparison=comparison,
                model=model,
                reviewer=reviewer,
            ),
        )

    @app.get("/results", response_class=HTMLResponse, include_in_schema=False)
    async def results_page(request: Request) -> HTMLResponse:
        directory = runtime.settings.results_dir
        return TEMPLATES.TemplateResponse(
            request,
            "results.html",
            context(
                request,
                "results",
                detection=_load_json(directory / "detection.json"),
                classifiers=_load_json(directory / "classifier_choice.json"),
                agent=_load_json(directory / "agent.json"),
                calls=_load_json(directory / "calls_summary.json"),
                results_dir=str(directory),
            ),
        )

    @app.get("/audit-log", response_class=HTMLResponse, include_in_schema=False)
    async def audit_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "audit.html",
            context(
                request,
                "audit",
                records=runtime.audit.recent(200),
                summary=runtime.audit.summary(),
                approvals=runtime.approvals.list()[:20],
                audit_file=str(runtime.settings.audit_file) if runtime.settings.audit_file else None,
            ),
        )
