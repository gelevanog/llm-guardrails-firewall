"""Render results/report.md from the JSON artifacts (the tables used in the README)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _f(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{value * 100:.{digits}f}%"


def write_report(directory: Path) -> Path:
    lines: list[str] = ["# Bulwark evaluation report", ""]
    detection = _load(directory / "detection.json")
    if detection:
        labels = detection["labels"]
        lines += [
            f"Detection run: {detection['created']}. Classifier: `{detection['classifier_model']}`. "
            f"Detected = score >= {detection['threshold']}; blocked = score >= {detection['block_threshold']}.",
            "",
            "## Datasets",
            "",
            "| Dataset | Samples | Attacks | Benign | Context | Source | License |",
            "|---|---:|---:|---:|---|---|---|",
        ]
        for name, info in detection["datasets"].items():
            lines.append(
                f"| {name} | {info['samples']} | {info['positives']} | {info['negatives']} | "
                f"{info.get('context', '')} | "
                f"{info.get('source', '')} | {info.get('license', '')} |"
            )
        lines += ["", "## Detection by layer and dataset", ""]
        lines += [
            "| Dataset | Layers | Precision | Recall | F1 | FPR | Blocked recall / FPR | Latency mean / p95 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in detection["summary"]:
            lines.append(
                f"| {row['dataset']} | {labels[row['config']]} | {_f(row['precision'])} | {_f(row['recall'])} | "
                f"{_f(row['f1'])} | {_pct(row['fpr'])} | "
                f"{_f(row['blocked']['recall'])} / {_pct(row['blocked']['fpr'])} | "
                f"{row['latency_ms']['mean']:.1f} / {row['latency_ms']['p95']:.1f} ms |"
            )
        lines += ["", "## False-positive rate on the hand-written hard negatives", ""]
        lines += ["| Layers | Flagged | FPR |", "|---|---:|---:|"]
        for key, row in detection["hard_negatives"].items():
            lines.append(f"| {labels[key]} | {row['fp']} / {row['negatives']} | {_pct(row['fpr'])} |")
        judge = detection.get("judge")
        if judge:
            lines += [
                "",
                f"## With the LLM judge (`{judge['model']}`, band {judge['band']}, {judge['calls']} calls, "
                f"{judge['errors']} errors)",
                "",
                "| Dataset (subset) | Layers | Precision | Recall | F1 | FPR | Latency mean / p95 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
            for row in judge["summary"]:
                lines.append(
                    f"| {row['dataset']} ({row['n']}) | {labels[row['config']]} | {_f(row['precision'])} | "
                    f"{_f(row['recall'])} | {_f(row['f1'])} | {_pct(row['fpr'])} | "
                    f"{row['latency_ms']['mean']:.0f} / {row['latency_ms']['p95']:.0f} ms |"
                )
    choice = _load(directory / "classifier_choice.json")
    if choice:
        datasets = list(choice["models"][0]["per_dataset"])
        lines += ["", "## Classifier choice (classifier layer alone)", ""]
        lines += [
            "| Model | Params | " + " | ".join(f"{d} F1 / FPR" for d in datasets) + " | Hard-negative FPR | ms / text |"
        ]
        lines += ["|---|---:|" + "---:|" * len(datasets) + "---:|---:|"]
        for row in choice["models"]:
            cells = " | ".join(
                f"{_f(row['per_dataset'][d]['f1'])} / {_pct(row['per_dataset'][d]['fpr'])}" for d in datasets
            )
            lines.append(
                f"| `{row['model']}` | {row['params_m']}M | {cells} | {_pct(row['hard_negative_fpr'])} | "
                f"{row['ms_per_text']:.0f} |"
            )
    for name, title in (("agent.json", "real model"), ("agent_fake.json", "fake gullible model")):
        agent = _load(directory / name)
        if not agent:
            continue
        lines += ["", f"## Agent: {title} (`{agent['model']}`), {agent['scenarios']} scenarios", ""]
        lines += [
            "| | Attack success rate | Benign tasks completed | Approvals requested | Blocked calls | "
            "Guard ms / run | Total s / run |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in agent["summary"]:
            lines.append(
                f"| {row['mode']} | {row['attacks_succeeded']} / {row['attack_runs']} ({_pct(row['asr'], 0)}) | "
                f"{row['benign_completed']} / {row['benign_runs']} ({_pct(row['benign_completion'], 0)}) | "
                f"{row['approvals_requested']} | {row['blocked_calls']} | {row['guard_ms_per_run'] or 0:.0f} | "
                f"{(row['total_ms_per_run'] or 0) / 1000:.1f} |"
            )
        lines += [
            "",
            "| Scenario | Kind | Without: attack / task | With: attack / task | With: approvals, blocked |",
            "|---|---|---|---|---|",
        ]
        by_id: dict[str, dict[bool, Any]] = {}
        for run in agent["runs"]:
            by_id.setdefault(run["scenario"], {})[run["protected"]] = run
        for scenario, pair in by_id.items():
            off, on = pair.get(False), pair.get(True)

            def cell(run: Any) -> str:
                if run is None:
                    return "–"
                attack = (
                    "–" if run["attack_success"] is None else ("**succeeded**" if run["attack_success"] else "failed")
                )
                return f"{attack} / {'done' if run['task_completed'] else 'not done'}" + (
                    " (error)" if run["error"] else ""
                )

            extra = f"{on['approvals']}, {on['blocked_calls']}" if on else "–"
            kind = (off or on or {}).get("kind", "")
            lines.append(f"| {scenario} | {kind} | {cell(off)} | {cell(on)} | {extra} |")
    calls = _load(directory / "calls_summary.json")
    if calls:
        lines += [
            "",
            "## Real API calls",
            "",
            f"{calls['total_requests']} requests; status {calls['by_status']}; by tag {calls['by_tag']}; "
            f"served models {calls['served_models']}; all model ids free: {calls['all_model_ids_free']}.",
        ]
    path = directory / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
