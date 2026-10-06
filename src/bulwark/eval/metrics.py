"""Binary detection metrics and latency statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel


class Binary(BaseModel):
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.tn + self.fp

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / self.positives if self.positives else None

    @property
    def fpr(self) -> float | None:
        return self.fp / self.negatives if self.negatives else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None if p is None or r is None else 0.0
        return 2 * p * r / (p + r)

    def add(self, label: int, predicted: bool) -> None:
        if label and predicted:
            self.tp += 1
        elif label:
            self.fn += 1
        elif predicted:
            self.fp += 1
        else:
            self.tn += 1

    def report(self) -> dict[str, float | int | None]:
        def r(value: float | None) -> float | None:
            return None if value is None else round(value, 4)

        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": r(self.precision),
            "recall": r(self.recall),
            "f1": r(self.f1),
            "fpr": r(self.fpr),
        }


def binary(labels: Sequence[int], predictions: Sequence[bool]) -> Binary:
    result = Binary()
    for label, predicted in zip(labels, predictions, strict=True):
        result.add(label, predicted)
    return result


def latency(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)

    def pct(q: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return ordered[index]

    return {"mean": round(sum(ordered) / len(ordered), 2), "p50": round(pct(0.5), 2), "p95": round(pct(0.95), 2)}
