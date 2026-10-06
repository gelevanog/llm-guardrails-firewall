"""Run the evaluation stages and write small, committable artifacts to the output directory.

detection.json           P/R/F1/FPR and latency per layer combination and dataset, per category, errors
classifier_choice.json   the candidate classifier models, side by side
agent.json               attack success and task completion without vs with Bulwark, per scenario
calls.jsonl, calls_summary.json   every real API request (ledger)
report.md                the tables used in the README
"""

from __future__ import annotations

import asyncio
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bulwark.agent_demo.agent import run_agent
from bulwark.agent_demo.fake_model import GullibleAgentModel
from bulwark.agent_demo.scenarios import load_scenarios, score_run, simulated_reviewer
from bulwark.classifier import CANDIDATES, InjectionClassifier
from bulwark.config import Settings
from bulwark.eval.config import EvalConfig, LlmConfig, load_eval_config
from bulwark.eval.datasets import Sample, load_bipia, load_deepset, load_jailbreakbench
from bulwark.eval.detection import (
    LABELS,
    Scored,
    by_category,
    combiners,
    errors,
    evaluate,
    score_classifier,
    score_heuristics,
    score_judge,
)
from bulwark.eval.handwritten import load_set
from bulwark.firewall import Firewall
from bulwark.guards.judge import LlmJudge
from bulwark.logging_config import get_logger
from bulwark.policy import PolicySet
from bulwark.providers.base import ChatProvider
from bulwark.providers.factory import build_provider
from bulwark.providers.resilient import CallLedger, DiskCache, ResilientProvider, Throttle

log = get_logger(__name__)

SOURCES = {
    "handwritten": {
        "source": "this repository (data/handwritten/source)",
        "license": "MIT (this repository)",
        "context": "input + untrusted",
    },
    "deepset": {
        "source": "huggingface.co/datasets/deepset/prompt-injections",
        "license": "Apache-2.0",
        "context": "input",
    },
    "jailbreakbench": {
        "source": "github.com/JailbreakBench/artifacts + huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        "license": "MIT",
        "context": "input",
    },
    "bipia": {
        "source": "github.com/microsoft/BIPIA (email task, text attacks)",
        "license": "MIT",
        "context": "untrusted",
    },
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class EvalRunner:
    def __init__(self, config: EvalConfig, settings: Settings) -> None:
        self.config = config
        self.settings = settings.model_copy(
            update={"require_free_models": config.require_free_models, "llm_ledger": config.output_dir / "calls.jsonl"}
        )
        self.out = config.output_dir
        self._ledger: CallLedger | None = None
        self._throttle = Throttle(config.min_seconds_between_requests)

    @classmethod
    def from_config(cls, path: Path, settings: Settings) -> EvalRunner:
        return cls(load_eval_config(path), settings)

    # ------------------------------------------------------------------------------------ helpers
    def llm(self, llm: LlmConfig, tag: str) -> ResilientProvider:
        if self._ledger is None:
            self._ledger = CallLedger(self.out / "calls.jsonl", self.config.max_calls)
        inner = build_provider(self.settings, kind=llm.provider, model=llm.model, fallback_models=llm.fallback_models)
        return ResilientProvider(
            inner,
            ledger=self._ledger,
            cache=DiskCache(self.settings.llm_cache_dir),
            throttle=self._throttle,
            max_retries=self.settings.llm_max_retries,
            tag=tag,
        )

    def samples(self, only: list[str] | None = None) -> list[Sample]:
        datasets = self.config.datasets
        cache = self.config.cache_dir
        out: list[Sample] = []
        if not only or "handwritten" in only:
            out += load_set(self.config.handwritten)
        if datasets.deepset.enabled and (not only or "deepset" in only):
            out += load_deepset(cache, datasets.deepset.revision, datasets.deepset.splits)
        if datasets.jailbreakbench.enabled and (not only or "jailbreakbench" in only):
            jbb = datasets.jailbreakbench
            out += load_jailbreakbench(
                cache, jbb.artifacts_commit, jbb.behaviors_revision, jbb.target, jbb.methods, jbb.per_method
            )
        if datasets.bipia.enabled and (not only or "bipia" in only):
            out += load_bipia(cache, datasets.bipia.commit, datasets.bipia.emails)
        return out

    def _write(self, name: str, data: object) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / name
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    async def _scored(self, only: list[str] | None, models: list[str]) -> list[Scored]:
        scored = await score_heuristics(self.samples(only))
        for model in models:
            classifier = InjectionClassifier(model, threads=self.settings.classifier_threads)
            log.info("eval.classifier", model=model, samples=len(scored))
            await asyncio.to_thread(score_classifier, scored, classifier, self.config.cache_dir)
        return scored

    # ------------------------------------------------------------------------------------ stages
    def detection(self, *, judge: bool = False, only: list[str] | None = None) -> dict[str, Any]:
        return asyncio.run(self._detection(judge=judge, only=only))

    async def _detection(self, *, judge: bool, only: list[str] | None) -> dict[str, Any]:
        cfg = self.config.detection
        model = self.config.classifier_model
        scored = await self._scored(only, [model])
        judge_calls = 0
        if judge:
            provider = self.llm(self.config.judge, "judge")
            llm_judge = LlmJudge(
                provider,
                model=self.config.judge.model,
                openrouter=self.config.judge.provider == "openrouter",
                boundary_key=b"bulwark-eval",
            )
            judge_calls = await score_judge(scored, llm_judge, model, cfg.judge_band, cfg.judge_every)
        combine = combiners(model, cfg.judge_band)
        groups: dict[str, list[Scored]] = {}
        for item in scored:
            groups.setdefault(item.dataset, []).append(item)
            if item.dataset == "handwritten":
                groups.setdefault(f"handwritten:{item.split}", []).append(item)
        judged_ids = {s.id for s in scored if s.judge is not None or s.judge_error}
        subset_items = {
            name: [s for s in items if _in_subset(s, items, cfg.judge_every)] for name, items in groups.items()
        }
        summary: list[dict[str, Any]] = []
        for name, items in groups.items():
            for key in ("heuristics", "classifier", "heuristics+classifier", "heuristics+classifier (corroborated)"):
                summary.append(
                    {
                        "dataset": name,
                        "config": key,
                        **evaluate(items, combine[key], cfg.threshold, cfg.block_threshold),
                    }
                )
        judge_rows: list[dict[str, Any]] = []
        if judge:
            for name, items in subset_items.items():
                if not items:
                    continue
                for key in ("heuristics", "heuristics+classifier", "heuristics+classifier+judge"):
                    judge_rows.append(
                        {
                            "dataset": name,
                            "config": key,
                            **evaluate(items, combine[key], cfg.threshold, cfg.block_threshold),
                        }
                    )
        hard = [s for s in scored if s.category == "hard_negative"]
        report: dict[str, Any] = {
            "created": _now(),
            "threshold": cfg.threshold,
            "block_threshold": cfg.block_threshold,
            "classifier_model": model,
            "labels": LABELS,
            "datasets": {
                name: {
                    **SOURCES.get(name.split(":")[0], {}),
                    "samples": len(items),
                    "positives": sum(s.label for s in items),
                    "negatives": sum(1 - s.label for s in items),
                }
                for name, items in groups.items()
            },
            "summary": summary,
            "hard_negatives": {
                key: evaluate(hard, combine[key], cfg.threshold, cfg.block_threshold)
                for key in ("heuristics", "classifier", "heuristics+classifier", "heuristics+classifier (corroborated)")
            },
            "by_category": {
                name: {
                    key: by_category(items, combine[key], cfg.threshold)
                    for key in ("heuristics", "classifier", "heuristics+classifier")
                }
                for name, items in groups.items()
                if ":" not in name
            },
            "errors": {
                key: errors(groups.get("handwritten", []), combine[key], cfg.threshold)
                for key in ("heuristics", "heuristics+classifier")
            },
        }
        previous = self.out / "detection.json"
        if judge:
            report["judge"] = {
                "model": self.config.judge.model,
                "band": list(cfg.judge_band),
                "every": cfg.judge_every,
                "calls": judge_calls,
                "judged": len(judged_ids),
                "errors": sum(1 for s in scored if s.judge_error),
                "subset": {name: len(items) for name, items in subset_items.items()},
                "summary": judge_rows,
                "hard_negatives": evaluate(
                    [s for s in hard if _in_subset(s, groups["handwritten"], cfg.judge_every)],
                    combine["heuristics+classifier+judge"],
                    cfg.threshold,
                    cfg.block_threshold,
                )
                if "handwritten" in groups
                else None,
                "examples": [
                    {
                        "id": s.id,
                        "label": s.label,
                        "before": round(max(s.heuristics, s.classifier.get(model, 0.0)), 3),
                        "judge": s.judge,
                        "error": s.judge_error,
                    }
                    for s in scored
                    if s.id in judged_ids
                ][:200],
            }
        elif previous.exists():
            report["judge"] = json.loads(previous.read_text(encoding="utf-8")).get("judge")
        self._write("detection.json", report)
        return report

    def classifier_choice(self) -> dict[str, Any]:
        return asyncio.run(self._classifier_choice())

    async def _classifier_choice(self) -> dict[str, Any]:
        cfg = self.config.detection
        models = self.config.classifier_candidates or [self.config.classifier_model]
        scored = await self._scored(None, models)
        groups: dict[str, list[Scored]] = {}
        for item in scored:
            groups.setdefault(item.dataset, []).append(item)
        rows: list[dict[str, Any]] = []
        for model in models:
            combine = combiners(model, cfg.judge_band)
            per_dataset = {
                name: evaluate(items, combine["classifier"], cfg.threshold, cfg.block_threshold)
                for name, items in groups.items()
            }
            hard = [s for s in scored if s.category == "hard_negative"]
            f1s = [r["f1"] for r in per_dataset.values() if r["f1"] is not None]
            candidate = CANDIDATES.get(model)
            rows.append(
                {
                    "model": model,
                    "license": candidate.license if candidate else None,
                    "params_m": candidate.params_m if candidate else None,
                    "revision": candidate.revision if candidate else None,
                    "per_dataset": per_dataset,
                    "macro_f1": round(statistics.mean(f1s), 4) if f1s else None,
                    "hard_negative_fpr": evaluate(hard, combine["classifier"], cfg.threshold, cfg.block_threshold)[
                        "fpr"
                    ],
                    "with_heuristics": {
                        name: evaluate(items, combine["heuristics+classifier"], cfg.threshold, cfg.block_threshold)
                        for name, items in groups.items()
                    },
                    "ms_per_text": round(
                        statistics.mean(s.classifier_ms[model] for s in scored if model in s.classifier_ms), 1
                    ),
                    "ms_p95": evaluate(scored, combine["classifier"], cfg.threshold, cfg.block_threshold)["latency_ms"][
                        "p95"
                    ],
                }
            )
        report = {"created": _now(), "threshold": cfg.threshold, "chosen": self.config.classifier_model, "models": rows}
        self._write("classifier_choice.json", report)
        return report

    def agent(self, *, real: bool, only: list[str] | None = None) -> dict[str, Any]:
        return asyncio.run(self._agent(real=real, only=only))

    async def _agent(self, *, real: bool, only: list[str] | None) -> dict[str, Any]:
        cfg = self.config.agent
        classifier = InjectionClassifier(self.config.classifier_model, threads=self.settings.classifier_threads)
        firewall = Firewall(PolicySet.from_dir(self.settings.policies_dir, "default"), classifier=classifier)
        provider: ChatProvider = self.llm(cfg.model, "agent") if real else GullibleAgentModel()
        model = cfg.model.model if real else "auto"
        items = [s for s in load_scenarios() if not only or s.id in only]
        runs: list[dict[str, Any]] = []
        for item in items:
            for protected in (False, True):
                log.info("eval.agent", scenario=item.id, protected=protected, model=provider.label)
                run = await run_agent(
                    item.task,
                    provider=provider,
                    firewall=firewall if protected else None,
                    policy=cfg.policy,
                    approver=simulated_reviewer(item),
                    model=model,
                    max_steps=cfg.max_steps,
                    scenario=item.id,
                    extra_body=dict(cfg.extra_body),
                )
                score = score_run(item, run)
                runs.append(
                    {
                        **score.model_dump(),
                        "served_model": run.model,
                        "answer": run.answer[:1200],
                        "raw_answer": run.raw_answer[:1200] if run.raw_answer != run.answer else None,
                        "executed": run.executed,
                        "approvals": [a.model_dump() for a in run.approvals],
                        "steps": [st.model_dump() for st in run.steps],
                    }
                )
        name = "agent.json" if real else "agent_fake.json"
        report: dict[str, Any] = {
            "created": _now(),
            "model": cfg.model.model if real else provider.label,
            "fallback_models": cfg.model.fallback_models if real else [],
            "policy": cfg.policy,
            "classifier_model": self.config.classifier_model,
            "scenarios": len(items),
            "summary": summarize_agent(runs),
            "runs": runs,
        }
        if only and (self.out / name).exists():
            previous = json.loads((self.out / name).read_text(encoding="utf-8"))
            kept = [r for r in previous.get("runs", []) if r["scenario"] not in only]
            report["runs"] = kept + runs
            report["summary"] = summarize_agent(report["runs"])
            report["scenarios"] = len({r["scenario"] for r in report["runs"]})
        self._write(name, report)
        return report

    def calls_summary(self) -> dict[str, object]:
        summary = summarize_ledger(self.out / "calls.jsonl")
        self._write("calls_summary.json", summary)
        return summary


def _in_subset(item: Scored, items: list[Scored], every: dict[str, int]) -> bool:
    step = every.get(item.dataset, 0)
    if not step:
        return False
    same = [s for s in items if s.dataset == item.dataset]
    return same.index(item) % step == 0 if item in same else False


def summarize_agent(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for protected in (False, True):
        mine = [r for r in runs if r["protected"] is protected]
        attacks = [r for r in mine if r["kind"] == "attack"]
        benign = [r for r in mine if r["kind"] == "benign"]
        out.append(
            {
                "mode": "with Bulwark" if protected else "without Bulwark",
                "attack_runs": len(attacks),
                "attacks_succeeded": sum(bool(r["attack_success"]) for r in attacks),
                "asr": round(sum(bool(r["attack_success"]) for r in attacks) / len(attacks), 4) if attacks else None,
                "benign_runs": len(benign),
                "benign_completed": sum(r["task_completed"] for r in benign),
                "benign_completion": round(sum(r["task_completed"] for r in benign) / len(benign), 4)
                if benign
                else None,
                "attack_runs_task_completed": sum(r["task_completed"] for r in attacks),
                "approvals_requested": sum(r["approvals"] for r in mine),
                "benign_runs_with_approval": sum(1 for r in benign if r["approvals"]),
                "blocked_calls": sum(r["blocked_calls"] for r in mine),
                "errors": sum(1 for r in mine if r["error"]),
                "model_calls": sum(r["model_calls"] for r in mine),
                "guard_ms_per_run": round(statistics.mean(r["guard_ms"] for r in mine), 1) if mine else None,
                "model_ms_per_run": round(statistics.mean(r["model_ms"] for r in mine), 1) if mine else None,
                "total_ms_per_run": round(statistics.mean(r["total_ms"] for r in mine), 1) if mine else None,
            }
        )
    return out


def summarize_ledger(path: Path) -> dict[str, object]:
    rows = (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.exists()
        else []
    )
    models = Counter(str(r.get("served_model") or r.get("requested_model")) for r in rows if r.get("status") == "ok")
    every_id = {str(r.get("requested_model")) for r in rows if r.get("requested_model")} | {
        str(r.get("served_model")) for r in rows if r.get("served_model")
    }
    return {
        "total_requests": len(rows),
        "by_status": dict(Counter(r.get("status") for r in rows)),
        "by_tag": dict(Counter(r.get("tag") for r in rows)),
        "served_models": dict(models.most_common()),
        "requested_models": sorted(m for m in every_id if m not in {"None", "auto"}),
        "all_model_ids_free": all(m.endswith(":free") for m in every_id if m not in {"None", "auto"}),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows),
        "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
    }
