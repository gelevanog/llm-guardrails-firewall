"""Detection evaluation: score every sample once per layer, then combine layers offline.

Heuristics and the classifier run on every sample of every dataset (CPU, no API calls); the LLM judge runs
only where it would run in production (combined score inside the judge band), on a deterministic subset
that fits the free-tier budget. Untrusted samples go through the untrusted-content guard, so hidden HTML
and invisible-character handling count. Latency is measured per sample with batch size 1, as in a gateway.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bulwark.classifier import InjectionClassifier
from bulwark.core import digest
from bulwark.eval.datasets import Sample
from bulwark.eval.metrics import Binary, latency
from bulwark.guards.injection import InjectionDetector
from bulwark.guards.judge import JudgeError, LlmJudge
from bulwark.guards.untrusted import UntrustedContentGuard
from bulwark.normalize import decoded_views, extract_html
from bulwark.policy import Layers, UntrustedConfig


class Scored(BaseModel):
    id: str
    dataset: str
    label: int
    category: str
    split: str
    context: str
    lang: str
    heuristics: float = 0.0
    heuristics_ms: float = 0.0
    classifier: dict[str, float] = Field(default_factory=dict)
    classifier_ms: dict[str, float] = Field(default_factory=dict)
    judge: float | None = None
    judge_ms: float | None = None
    judge_error: str | None = None
    text: str = Field(default="", exclude=True)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


async def score_heuristics(samples: list[Sample]) -> list[Scored]:
    detector = InjectionDetector()
    guard = UntrustedContentGuard(detector)
    config = UntrustedConfig(layers=Layers(classifier=False))
    scored: list[Scored] = []
    for sample in samples:
        started = time.perf_counter()
        if sample.context == "untrusted":
            result, _ = await guard.inspect(sample.text, "eval", config)
            score = result.score
        else:
            score = (await detector.detect(sample.text, Layers(classifier=False), "input")).score
        scored.append(
            Scored(
                **sample.model_dump(exclude={"text"}),
                heuristics=score,
                heuristics_ms=_ms(started),
                text=sample.text,
            )
        )
    return scored


def classifier_input(sample: Scored) -> list[str]:
    """What the classifier layer sees: the visible text plus up to three decoded payloads."""
    text = extract_html(sample.text).visible if sample.context == "untrusted" else sample.text
    return [text, *[v.text for v in decoded_views(text, rot13=False, reverse=False)][:3]]


def score_classifier(
    scored: list[Scored], classifier: InjectionClassifier, cache_dir: Path, *, latency_every: int = 10
) -> None:
    """Fill `classifier[model]` for every sample (batched) and time every n-th sample alone (batch size 1,
    as in the gateway). Scores and timings are cached on disk per model."""
    path = cache_dir / "scores" / (classifier.model_name.replace("/", "__") + ".json")
    cache: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    classifier.load()
    keys = [digest(item.context + "\x00" + item.text, 32) for item in scored]
    todo = [(key, item) for key, item in zip(keys, scored, strict=True) if key not in cache]
    flat: list[str] = []
    owners: list[str] = []
    for key, item in todo:
        for text in classifier_input(item):
            flat.append(text)
            owners.append(key)
    for start in range(0, len(flat), 64):
        batch = flat[start : start + 64]
        for key, score in zip(owners[start : start + 64], classifier.score_many(batch), strict=True):
            entry = cache.setdefault(key, {"score": 0.0})
            entry["score"] = max(entry["score"], score)
    for index, (key, item) in enumerate(zip(keys, scored, strict=True)):
        if index % latency_every == 0 and "ms" not in cache[key]:
            started = time.perf_counter()
            classifier.score_many(classifier_input(item))
            cache[key]["ms"] = _ms(started)
    for key, item in zip(keys, scored, strict=True):
        item.classifier[classifier.model_name] = cache[key]["score"]
        if "ms" in cache[key]:
            item.classifier_ms[classifier.model_name] = cache[key]["ms"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache))


async def score_judge(
    scored: list[Scored], judge: LlmJudge, model: str, band: tuple[float, float], every: dict[str, int]
) -> int:
    """Ask the judge about subset samples whose combined score is in the band. Returns the number of calls."""
    calls = 0
    for item in _subset(scored, every):
        combined = max(item.heuristics, item.classifier.get(model, 0.0))
        if not band[0] <= combined < band[1]:
            continue
        calls += 1
        started = time.perf_counter()
        try:
            verdict = await judge.judge(item.text, item.context)
        except JudgeError as exc:
            item.judge_error = str(exc)[:200]
        else:
            item.judge = round(verdict.probability, 4)
        item.judge_ms = _ms(started)
    return calls


def _subset(scored: list[Scored], every: dict[str, int]) -> list[Scored]:
    out: list[Scored] = []
    counters: dict[str, int] = {}
    for item in scored:
        step = every.get(item.dataset, 0)
        position = counters.get(item.dataset, 0)
        counters[item.dataset] = position + 1
        if step and position % step == 0:
            out.append(item)
    return out


# ------------------------------------------------------------------------------------- combinations
Combiner = Callable[[Scored], tuple[float, float | None]]
"""sample -> (score, latency in ms or None when this sample's classifier latency was not measured)."""


def combiners(model: str, band: tuple[float, float]) -> dict[str, Combiner]:
    def classifier_ms(s: Scored, extra: float = 0.0) -> float | None:
        measured = s.classifier_ms.get(model)
        return None if measured is None else round(measured + extra, 3)

    def heuristics(s: Scored) -> tuple[float, float | None]:
        return s.heuristics, s.heuristics_ms

    def classifier(s: Scored) -> tuple[float, float | None]:
        return s.classifier.get(model, 0.0), classifier_ms(s)

    def both(s: Scored) -> tuple[float, float | None]:
        return max(s.heuristics, s.classifier.get(model, 0.0)), classifier_ms(s, s.heuristics_ms)

    def corroborated(s: Scored) -> tuple[float, float | None]:
        # The classifier alone cannot flag: it raises scores the rules already found suspicious (>= 0.25),
        # and otherwise stops just under the flag threshold (where a judge, if enabled, takes over).
        c = s.classifier.get(model, 0.0)
        h = s.heuristics
        score = max(h, c) if h >= 0.25 else (min(c, 0.45) if c >= 0.5 else h)
        return score, classifier_ms(s, s.heuristics_ms)

    def with_judge(s: Scored) -> tuple[float, float | None]:
        score, ms = both(s)
        judged = band[0] <= score < band[1] and (s.judge is not None or s.judge_error is not None)
        if not judged:
            return score, ms
        total = None if ms is None else ms + (s.judge_ms or 0.0)
        # Fail-closed: a judge call that failed counts as a detection.
        return (s.judge if s.judge is not None else 1.0), total

    return {
        "heuristics": heuristics,
        "classifier": classifier,
        "heuristics+classifier": both,
        "heuristics+classifier (corroborated)": corroborated,
        "heuristics+classifier+judge": with_judge,
    }


LABELS = {
    "heuristics": "Heuristics + normalization",
    "classifier": "Classifier alone",
    "heuristics+classifier": "Heuristics + classifier (max)",
    "heuristics+classifier (corroborated)": "Heuristics + classifier (corroborated)",
    "heuristics+classifier+judge": "Heuristics + classifier + LLM judge",
}


def evaluate(items: list[Scored], combine: Combiner, threshold: float, block_threshold: float) -> dict[str, Any]:
    flagged = Binary()
    blocked = Binary()
    latencies: list[float] = []
    for item in items:
        score, ms = combine(item)
        flagged.add(item.label, score >= threshold)
        blocked.add(item.label, score >= block_threshold)
        if ms is not None:
            latencies.append(ms)
    return {
        "n": len(items),
        "positives": flagged.positives,
        "negatives": flagged.negatives,
        **flagged.report(),
        "blocked": {"recall": blocked.report()["recall"], "fpr": blocked.report()["fpr"]},
        "latency_ms": latency(latencies),
        "latency_samples": len(latencies),
    }


def by_category(items: list[Scored], combine: Combiner, threshold: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        row = out.setdefault(item.category, {"n": 0, "flagged": 0, "label": item.label})
        row["n"] += 1
        row["flagged"] += int(combine(item)[0] >= threshold)
    for row in out.values():
        row["rate"] = round(row["flagged"] / row["n"], 4)
    return out


def errors(
    items: list[Scored], combine: Combiner, threshold: float, limit: int = 40
) -> dict[str, list[dict[str, Any]]]:
    false_positives, false_negatives = [], []
    for item in items:
        score = combine(item)[0]
        row = {
            "id": item.id,
            "category": item.category,
            "score": round(score, 3),
            "heuristics": item.heuristics,
            "classifier": {k: round(v, 3) for k, v in item.classifier.items()},
            "text": " ".join(item.text.split())[:220],
        }
        if item.label and score < threshold:
            false_negatives.append(row)
        elif not item.label and score >= threshold:
            false_positives.append(row)
    return {"false_positives": false_positives[:limit], "false_negatives": false_negatives[:limit]}
