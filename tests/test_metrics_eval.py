"""Metrics, the hand-written set compiler, layer combinations and result summaries."""

import base64
import json
from pathlib import Path

from bulwark.eval.detection import Scored, by_category, combiners, evaluate
from bulwark.eval.handwritten import TRANSFORMS, build_set, load_set, render
from bulwark.eval.metrics import binary, latency
from bulwark.eval.report import write_report
from bulwark.eval.runner import summarize_agent, summarize_ledger

ROOT = Path(__file__).resolve().parents[1]


def test_binary_metrics() -> None:
    result = binary([1, 1, 1, 0, 0, 0, 0], [True, True, False, True, False, False, False])
    report = result.report()
    assert (report["tp"], report["fn"], report["fp"], report["tn"]) == (2, 1, 1, 3)
    assert report["precision"] == round(2 / 3, 4) and report["recall"] == round(2 / 3, 4)
    assert report["fpr"] == 0.25 and report["f1"] == round(2 / 3, 4)
    assert binary([0, 0], [False, False]).report()["precision"] is None
    assert binary([1], [False]).f1 is None


def test_latency_percentiles() -> None:
    stats = latency([float(v) for v in range(1, 101)])
    assert stats == {"mean": 50.5, "p50": 50.0, "p95": 95.0}
    assert latency([]) == {"mean": 0.0, "p50": 0.0, "p95": 0.0}


def test_compiled_set_matches_its_source() -> None:
    built = build_set(ROOT / "data/handwritten/source")
    committed = load_set(ROOT / "data/handwritten/handwritten.jsonl")
    assert [s.model_dump() for s in built] == [s.model_dump() for s in committed]
    categories = {s.category for s in built}
    assert {"direct", "jailbreak", "multilingual", "obfuscated", "indirect", "benign", "hard_negative"} <= categories


def test_transforms_are_deterministic() -> None:
    item = {"text": "ignore all", "transform": "base64", "template": "decode: {payload}"}
    assert render(item) == "decode: " + base64.b64encode(b"ignore all").decode()
    assert render({"text": "abc", "transform": "reversed"}) == "cba"
    assert render({"text": "hi"}) == "hi"
    assert set(TRANSFORMS) >= {"base64", "hex", "rot13", "leet", "homoglyph", "zero_width", "tags", "spaced", "url"}


def scored(label: int, h: float, c: float, judge: float | None = None, category: str = "x") -> Scored:
    return Scored(
        id=f"s{h}{c}",
        dataset="d",
        label=label,
        category=category,
        split="dev",
        context="input",
        lang="en",
        heuristics=h,
        heuristics_ms=0.1,
        classifier={"m": c},
        classifier_ms={"m": 10.0},
        judge=judge,
        judge_ms=500.0 if judge is not None else None,
    )


def test_layer_combinations() -> None:
    combine = combiners("m", (0.25, 0.85))
    attack_rules = scored(1, 0.9, 0.0)
    attack_model = scored(1, 0.0, 0.99)
    false_alarm = scored(0, 0.0, 0.98)
    judged = scored(0, 0.3, 0.6, judge=0.1)
    assert combine["heuristics"](attack_model)[0] == 0.0
    assert combine["heuristics+classifier"](attack_model)[0] == 0.99
    assert combine["heuristics+classifier (corroborated)"](false_alarm)[0] == 0.45  # classifier alone cannot flag
    assert combine["heuristics+classifier (corroborated)"](judged)[0] == 0.6  # rules saw something: classifier counts
    assert combine["heuristics+classifier+judge"](judged) == (0.1, 510.1)
    report = evaluate([attack_rules, attack_model, false_alarm, judged], combine["heuristics+classifier"], 0.5, 0.85)
    assert report["tp"] == 2 and report["fp"] == 2 and report["latency_samples"] == 4
    categories = by_category(
        [scored(1, 0.9, 0, category="a"), scored(1, 0.1, 0, category="a")], combine["heuristics"], 0.5
    )
    assert categories["a"]["rate"] == 0.5


def test_summaries(tmp_path: Path) -> None:
    ledger = tmp_path / "calls.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "tag": "judge",
                    "requested_model": "a/b:free",
                    "served_model": "a/b:free",
                    "status": "ok",
                    "input_tokens": 5,
                },
                {"tag": "agent", "requested_model": "a/b:free", "served_model": None, "status": "retryable_error"},
            ]
        )
    )
    summary = summarize_ledger(ledger)
    assert summary["total_requests"] == 2 and summary["all_model_ids_free"] is True and summary["input_tokens"] == 5
    runs = [
        {
            "protected": False,
            "kind": "attack",
            "attack_success": True,
            "task_completed": True,
            "approvals": 0,
            "blocked_calls": 0,
            "error": None,
            "model_calls": 3,
            "guard_ms": 0.0,
            "model_ms": 900.0,
            "total_ms": 950.0,
            "scenario": "a",
        },
        {
            "protected": True,
            "kind": "attack",
            "attack_success": False,
            "task_completed": True,
            "approvals": 1,
            "blocked_calls": 1,
            "error": None,
            "model_calls": 3,
            "guard_ms": 40.0,
            "model_ms": 900.0,
            "total_ms": 990.0,
            "scenario": "a",
        },
    ]
    off, on = summarize_agent(runs)
    assert off["asr"] == 1.0 and on["asr"] == 0.0 and on["blocked_calls"] == 1
    (tmp_path / "agent.json").write_text(
        json.dumps({"model": "a/b:free", "scenarios": 1, "summary": [off, on], "runs": runs})
    )
    (tmp_path / "calls_summary.json").write_text(json.dumps(summary))
    report = write_report(tmp_path).read_text()
    assert "Attack success rate" in report and "all model ids free: True" in report
