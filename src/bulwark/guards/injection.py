"""Layered prompt-injection and jailbreak detection: heuristics -> classifier -> (optional) LLM judge.

Each layer is cheap enough for the traffic it sees. Heuristics (microseconds) see every text and catch
obfuscated and multilingual phrasing the classifier was never trained on. The classifier (tens of
milliseconds on CPU) generalizes to unseen wordings. Their scores are combined with `max`, and only texts
whose combined score lands in the policy's judge band go to the LLM judge, which can confirm or dismiss
them.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from bulwark.classifier import InjectionClassifier
from bulwark.core import Action, Finding, GuardResult
from bulwark.guards import heuristics
from bulwark.guards.heuristics import Context
from bulwark.guards.judge import JudgeError, LlmJudge
from bulwark.logging_config import get_logger
from bulwark.normalize import decoded_views
from bulwark.policy import InjectionConfig, Layers

log = get_logger(__name__)


@dataclass
class Detection:
    score: float
    layer_scores: dict[str, float] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    judged: bool = False


@dataclass
class InjectionDetector:
    classifier: InjectionClassifier | None = None
    judge: LlmJudge | None = None

    def available_layers(self) -> list[str]:
        return ["heuristics"] + (["classifier"] if self.classifier else []) + (["judge"] if self.judge else [])

    async def detect(
        self,
        text: str,
        layers: Layers,
        context: Context = "input",
        judge_band: tuple[float, float] = (0.3, 0.85),
        fail_closed: bool = True,
    ) -> Detection:
        detection = Detection(score=0.0)
        if not text.strip():
            return detection

        if layers.heuristics:
            started = time.perf_counter()
            result = heuristics.scan(text, context)
            detection.timings_ms["heuristics"] = _ms(started)
            detection.layer_scores["heuristics"] = result.score
            detection.findings.extend(result.findings)

        if layers.classifier and self.classifier is not None:
            started = time.perf_counter()
            try:
                score = await asyncio.to_thread(self._classify, text)
            except Exception as exc:  # a broken model must not take the request down silently
                detection.errors["classifier"] = f"{type(exc).__name__}: {exc}"[:300]
                log.error("classifier.failed", error=str(exc)[:200])
            else:
                detection.layer_scores["classifier"] = score
                if score >= 0.5:
                    detection.findings.append(
                        Finding(
                            rule="classifier",
                            layer="classifier",
                            score=score,
                            message=f"the injection classifier scored this {score:.2f}",
                        )
                    )
            detection.timings_ms["classifier"] = _ms(started)

        combined = max(detection.layer_scores.values(), default=0.0)
        if layers.judge and self.judge is not None and judge_band[0] <= combined < judge_band[1]:
            started = time.perf_counter()
            try:
                verdict = await self.judge.judge(text, context)
            except JudgeError as exc:
                detection.errors["judge"] = str(exc)[:300]
                log.warning("judge.failed", error=str(exc)[:200])
            else:
                detection.judged = True
                detection.layer_scores["judge"] = round(verdict.probability, 4)
                combined = verdict.probability
                detection.findings.append(
                    Finding(
                        rule="judge." + ("injection" if verdict.injection else "benign"),
                        layer="judge",
                        score=round(verdict.probability, 4),
                        message=f"LLM judge ({verdict.model}): {verdict.reason}",
                    )
                )
            detection.timings_ms["judge"] = _ms(started)

        if detection.errors and fail_closed:
            combined = 1.0
            detection.findings.append(
                Finding(
                    rule="fail_closed",
                    layer="policy",
                    score=1.0,
                    message=f"layer(s) {', '.join(detection.errors)} failed and the policy is fail-closed",
                )
            )
        detection.score = round(combined, 4)
        return detection

    def _classify(self, text: str) -> float:
        assert self.classifier is not None
        # Also score decoded payloads: a base64 instruction is opaque to the model's tokenizer otherwise.
        extra = [view.text for view in decoded_views(text, rot13=False, reverse=False)][:3]
        return max(self.classifier.score_many([text, *extra]))


def injection_result(detection: Detection, config: InjectionConfig, started: float) -> GuardResult:
    """Map a detection to the policy's action for direct input."""
    if detection.score >= config.block_threshold:
        action = config.action
    elif detection.score >= config.flag_threshold:
        action = config.borderline_action
    else:
        action = Action.ALLOW
    return GuardResult(
        guard="injection",
        score=detection.score,
        triggered=action is not Action.ALLOW,
        action=action,
        explanation=explain(detection.findings),
        findings=detection.findings,
        layer_scores=detection.layer_scores,
        latency_ms=_ms(started),
        error="; ".join(f"{k}: {v}" for k, v in detection.errors.items()) or None,
    )


def explain(findings: list[Finding], limit: int = 3) -> str:
    """The strongest distinct reasons, most significant first."""
    seen: list[str] = []
    for finding in sorted(findings, key=lambda f: -f.score):
        if finding.score < 0.2 or finding.message in seen:
            continue
        seen.append(finding.message)
        if len(seen) == limit:
            break
    return "; ".join(seen)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
