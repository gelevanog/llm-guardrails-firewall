"""Command line: `bulwark scan | serve | agent-demo | eval | models | calls | policies | download-model`."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from bulwark.config import DEFAULT_FREE_FALLBACKS, DEFAULT_FREE_MODEL, Settings
from bulwark.logging_config import configure_logging

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Bulwark: an LLM firewall against prompt injection, jailbreaks and data exfiltration.",
)
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation: detection by layer and dataset, agent attack success.")
models_app = typer.Typer(no_args_is_help=True, help="Free OpenRouter models.")
app.add_typer(eval_app, name="eval")
app.add_typer(models_app, name="models")
console = Console()
err = Console(stderr=True)

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Evaluation config YAML.")]


def _settings(**overrides: Any) -> Settings:
    settings = Settings(**overrides)
    configure_logging(settings.log_level, settings.log_format)
    return settings


def _read_text(text: str | None, file: Path | None) -> str:
    if file is not None:
        return file.read_text(encoding="utf-8")
    if text is None or text == "-":
        return sys.stdin.read()
    return text


@app.command()
def scan(
    text: Annotated[str | None, typer.Argument(help="Text to scan; '-' or omitted reads stdin.")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Read the text from a file.")] = None,
    stage: Annotated[str, typer.Option("--stage", "-s", help="input | untrusted | output")] = "input",
    policy: Annotated[str | None, typer.Option("--policy", "-p")] = None,
    classifier: Annotated[
        bool, typer.Option("--classifier/--no-classifier", help="Use the classifier layer (needs the extra).")
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full decision as JSON.")] = False,
) -> None:
    """Scan one text and print the decision. Exit code 2 when the policy would block it."""
    from bulwark.firewall import Firewall
    from bulwark.gateway.runtime import build_classifier
    from bulwark.policy import PolicySet

    settings = _settings(classifier_enabled=classifier, classifier_preload=False)
    model, _ = build_classifier(settings)
    firewall = Firewall(PolicySet.from_dir(settings.policies_dir, settings.default_policy), classifier=model)
    content = _read_text(text, file)

    async def run() -> Any:
        if stage == "untrusted":
            return (await firewall.check_untrusted(content, source="cli", policy=policy))[0]
        if stage == "output":
            return await firewall.check_output(content, policy=policy)
        return await firewall.check_input(content, policy=policy)

    decision = asyncio.run(run())
    if as_json:
        console.print_json(decision.model_dump_json())
    else:
        color = {"block": "red", "require_approval": "yellow", "sanitize": "cyan", "flag": "yellow"}.get(
            decision.action.value, "green"
        )
        console.print(
            f"[bold {color}]{decision.action.value}[/] · score {decision.score:.2f} · policy {decision.policy}"
        )
        for result in decision.results:
            layers = " ".join(f"{k}={v:.2f}" for k, v in result.layer_scores.items())
            console.print(f"  {result.guard:12} {result.score:.2f} {result.action.value:16} {layers}")
            if result.explanation:
                console.print(f"    {result.explanation}", markup=False, highlight=False)
        if decision.text is not None and decision.text != content:
            console.print("\n[bold]Text after Bulwark:[/]")
            console.print(decision.text, markup=False, highlight=False)
    if decision.blocked:
        raise typer.Exit(2)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (development).")] = False,
) -> None:
    """Run the gateway, API and dashboard."""
    import uvicorn

    uvicorn.run("bulwark.gateway.app:create_default_app", factory=True, host=host, port=port, reload=reload)


@app.command("agent-demo")
def agent_demo(
    scenario: Annotated[str | None, typer.Argument(help="Scenario id; omitted runs all of them.")] = None,
    real: Annotated[bool, typer.Option("--real", help="Use the configured free OpenRouter model.")] = False,
    classifier: Annotated[bool, typer.Option("--classifier/--no-classifier")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print every step.")] = False,
) -> None:
    """Run the email agent on its scenarios without and with Bulwark (fake gullible model by default)."""
    from bulwark.agent_demo.agent import run_agent
    from bulwark.agent_demo.fake_model import GullibleAgentModel
    from bulwark.agent_demo.scenarios import load_scenarios, score_run, simulated_reviewer
    from bulwark.firewall import Firewall
    from bulwark.gateway.runtime import build_classifier
    from bulwark.policy import PolicySet
    from bulwark.providers.base import ChatProvider
    from bulwark.providers.factory import budgeted, build_provider

    settings = _settings(classifier_enabled=classifier)
    model, _ = build_classifier(settings)
    firewall = Firewall(PolicySet.from_dir(settings.policies_dir, settings.default_policy), classifier=model)
    provider: ChatProvider = (
        budgeted(build_provider(settings, kind="openrouter"), settings, tag="agent_demo")
        if real
        else GullibleAgentModel()
    )
    items = [s for s in load_scenarios() if scenario is None or s.id == scenario]
    if not items:
        err.print(f"[red]unknown scenario {scenario!r}[/]")
        raise typer.Exit(1)
    table = Table(title=f"Agent demo · {provider.label}")
    for column in ("scenario", "kind", "Bulwark", "attack succeeded", "task completed", "approvals", "blocked calls"):
        table.add_column(column)

    async def run_all() -> None:
        for item in items:
            for protected in (False, True):
                run = await run_agent(
                    item.task,
                    provider=provider,
                    firewall=firewall if protected else None,
                    approver=simulated_reviewer(item),
                    scenario=item.id,
                )
                score = score_run(item, run)
                attack = (
                    "–" if score.attack_success is None else ("[red]yes[/]" if score.attack_success else "[green]no[/]")
                )
                table.add_row(
                    item.id,
                    item.kind,
                    "on" if protected else "off",
                    attack,
                    "yes" if score.task_completed else "[yellow]no[/]",
                    str(score.approvals),
                    str(score.blocked_calls),
                )
                if verbose:
                    for step in run.steps:
                        console.print(f"  [{step.status}] {step.title}: {step.detail[:200]}", markup=False)

    asyncio.run(run_all())
    console.print(table)


@app.command()
def policies(name: Annotated[str | None, typer.Argument()] = None) -> None:
    """List policies, or print one as JSON."""
    from bulwark.policy import PolicySet

    settings = _settings()
    policy_set = PolicySet.from_dir(settings.policies_dir, settings.default_policy)
    if name:
        console.print_json(policy_set.get(name).model_dump_json(by_alias=True))
        return
    for policy in policy_set:
        marker = " (default)" if policy.name == policy_set.default_name else ""
        console.print(f"[bold]{policy.name}[/]{marker}: {policy.description}")


@app.command("download-model")
def download_model(model: Annotated[str | None, typer.Argument()] = None) -> None:
    """Download the classifier (safetensors only) into the Hugging Face cache."""
    from bulwark.classifier import CANDIDATES, fetch_model

    settings = _settings()
    name = model or settings.classifier_model
    path = fetch_model(name)
    candidate = CANDIDATES.get(name)
    if candidate and candidate.tokenizer:
        fetch_model(candidate.tokenizer, revision="main", tokenizer_only=True)
    console.print(f"classifier ready: {path}")


# ------------------------------------------------------------------------------------------- eval
@eval_app.command("build-set")
def eval_build_set(
    source: Annotated[Path, typer.Option()] = Path("data/handwritten/source"),
    output: Annotated[Path, typer.Option()] = Path("data/handwritten/handwritten.jsonl"),
) -> None:
    """Compile the hand-written YAML source (with deterministic obfuscations) into JSONL."""
    from bulwark.eval.handwritten import build_set, set_stats, write_set

    items = build_set(source)
    write_set(items, output)
    console.print_json(json.dumps(set_stats(items)))


@eval_app.command("detection")
def eval_detection(
    config: ConfigOption = Path("configs/eval.yaml"),
    judge: Annotated[
        bool, typer.Option("--judge/--no-judge", help="Also run the LLM-judge layer (real calls).")
    ] = False,
    datasets: Annotated[str | None, typer.Option(help="Comma-separated dataset names (default: all).")] = None,
) -> None:
    """Precision / recall / F1 / FPR / latency per layer combination and dataset."""
    from bulwark.eval.report import write_report
    from bulwark.eval.runner import EvalRunner

    runner = EvalRunner.from_config(config, _settings())
    report = runner.detection(judge=judge, only=datasets.split(",") if datasets else None)
    for row in report["summary"]:
        console.print(
            f"{row['dataset']:14} {row['config']:30} P={row['precision'] or 0:.3f} R={row['recall'] or 0:.3f} "
            f"F1={row['f1'] or 0:.3f} FPR={row['fpr'] if row['fpr'] is not None else float('nan'):.3f}"
        )
    runner.calls_summary()
    console.print(f"report: {write_report(runner.out)}")


@eval_app.command("classifiers")
def eval_classifiers(config: ConfigOption = Path("configs/eval.yaml")) -> None:
    """Compare the candidate classifier models (classifier layer alone): quality and CPU latency."""
    from bulwark.eval.report import write_report
    from bulwark.eval.runner import EvalRunner

    runner = EvalRunner.from_config(config, _settings())
    for row in runner.classifier_choice()["models"]:
        console.print(f"{row['model']:50} F1={row.get('macro_f1') or 0:.3f} {row.get('ms_per_text', 0):.0f} ms/text")
    console.print(f"report: {write_report(runner.out)}")


@eval_app.command("agent")
def eval_agent(
    config: ConfigOption = Path("configs/eval.yaml"),
    real: Annotated[
        bool, typer.Option("--real/--fake", help="Real free model (OpenRouter) or the fake gullible one.")
    ] = False,
    scenarios: Annotated[str | None, typer.Option(help="Comma-separated scenario ids (default: all).")] = None,
) -> None:
    """Attack success rate and task completion, without vs with Bulwark."""
    from bulwark.eval.report import write_report
    from bulwark.eval.runner import EvalRunner

    runner = EvalRunner.from_config(config, _settings())
    result = runner.agent(real=real, only=scenarios.split(",") if scenarios else None)
    for mode in result["summary"]:
        console.print(mode)
    if real:
        runner.calls_summary()
    console.print(f"report: {write_report(runner.out)}")


@eval_app.command("report")
def eval_report(directory: Annotated[Path, typer.Argument()] = Path("results")) -> None:
    """Re-render results/report.md from the JSON artifacts."""
    from bulwark.eval.report import write_report

    console.print(f"report: {write_report(directory)}")


# ----------------------------------------------------------------------------------------- models
@models_app.command("free")
def models_free(
    smoke: Annotated[int, typer.Option(help="Smoke-test this many models with one tiny real call each.")] = 0,
    candidates: Annotated[
        str | None, typer.Option(help="Comma-separated ids to smoke-test (default: the configured ones).")
    ] = None,
    tools: Annotated[bool, typer.Option("--tools", help="Smoke-test tool calling instead of plain text.")] = False,
    ledger: Annotated[Path, typer.Option(help="Call ledger for the smoke calls.")] = Path("results/calls.jsonl"),
) -> None:
    """List OpenRouter models whose id ends in ':free' (no key needed); optionally smoke-test a few."""
    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()
    free = [m for m in response.json()["data"] if str(m["id"]).endswith(":free")]
    table = Table(title=f"{len(free)} free models")
    for column in ("id", "context", "tools", "structured output"):
        table.add_column(column)
    for model in free:
        params = model.get("supported_parameters") or []
        table.add_row(
            model["id"],
            str(model.get("context_length")),
            "yes" if "tools" in params else "",
            "yes" if "response_format" in params or "structured_outputs" in params else "",
        )
    console.print(table)
    if smoke:
        ids = candidates.split(",") if candidates else [DEFAULT_FREE_MODEL, *DEFAULT_FREE_FALLBACKS]
        asyncio.run(_smoke(ids[:smoke], ledger, tools))


async def _smoke(ids: list[str], ledger_path: Path, tools: bool) -> None:
    from bulwark.agent_demo.tools import TOOL_SCHEMAS
    from bulwark.providers.base import ProviderError, ensure_free_models
    from bulwark.providers.factory import build_provider
    from bulwark.providers.resilient import CallLedger, ResilientProvider, Throttle

    ensure_free_models(ids)
    settings = _settings()
    ledger = CallLedger(ledger_path, settings.llm_max_calls)
    throttle = Throttle(settings.llm_min_seconds_between_requests)
    results = []
    for model in ids:
        provider = ResilientProvider(
            build_provider(settings, kind="openrouter", model=model, fallback_models=[]),
            ledger=ledger,
            throttle=throttle,
            max_retries=1,
            tag="smoke_tools" if tools else "smoke",
        )
        body: dict[str, Any] = {"model": model, "max_tokens": 800}
        if tools:
            body["messages"] = [{"role": "user", "content": "What is the status of order A-1004? Use the tools."}]
            body["tools"] = TOOL_SCHEMAS
        else:
            body["messages"] = [{"role": "user", "content": "Reply with the single word: ok"}]
            body["reasoning"] = {"effort": "low", "exclude": True}
        try:
            answer = await provider.complete(body)
            message = answer["choices"][0]["message"]
            calls = [c["function"]["name"] + c["function"]["arguments"] for c in message.get("tool_calls") or []]
            results.append(
                {
                    "model": model,
                    "ok": True,
                    "served": answer.get("model"),
                    "answer": (message.get("content") or "").strip()[:60],
                    "tool_calls": calls,
                }
            )
        except ProviderError as exc:
            results.append({"model": model, "ok": False, "error": str(exc)[:160]})
    for row in results:
        console.print(row)
    out = ledger_path.parent / ("smoke_tools.json" if tools else "smoke.json")
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


@app.command()
def calls(ledger: Annotated[Path, typer.Argument()] = Path("results/calls.jsonl")) -> None:
    """Summarize the real-API call ledger."""
    from bulwark.eval.runner import summarize_ledger

    console.print_json(json.dumps(summarize_ledger(ledger)))


if __name__ == "__main__":
    app()
