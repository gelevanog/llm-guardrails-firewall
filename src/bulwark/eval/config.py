"""Evaluation config (configs/eval.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bulwark.config import ProviderKind
from bulwark.providers.base import ensure_free_models


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LlmConfig(_Strict):
    provider: ProviderKind = "openrouter"
    model: str
    fallback_models: list[str] = Field(default_factory=list)


class DeepsetConfig(_Strict):
    enabled: bool = True
    revision: str
    splits: list[str] = Field(default_factory=lambda: ["train", "test"])


class JbbConfig(_Strict):
    enabled: bool = True
    artifacts_commit: str
    behaviors_revision: str
    target: str = "gpt-4-0125-preview"
    methods: list[str] = Field(default_factory=lambda: ["PAIR", "JBC", "GCG", "prompt_with_random_search"])
    per_method: int = 100


class BipiaConfig(_Strict):
    enabled: bool = True
    commit: str
    emails: int = 50


class DatasetsConfig(_Strict):
    deepset: DeepsetConfig
    jailbreakbench: JbbConfig
    bipia: BipiaConfig


class DetectionConfig(_Strict):
    threshold: float = 0.5
    """Detected = combined score at or above (the policies' flag / quarantine threshold)."""
    block_threshold: float = 0.85
    judge_band: tuple[float, float] = (0.25, 0.85)
    judge_every: dict[str, int] = Field(default_factory=dict)
    """Judge subset per dataset: every n-th sample (free-tier budget)."""


class AgentConfig(_Strict):
    model: LlmConfig
    policy: str = "email-agent"
    max_steps: int = 8
    extra_body: dict[str, object] = Field(default_factory=dict)


class EvalConfig(_Strict):
    output_dir: Path = Path("results")
    cache_dir: Path = Path(".cache/datasets")
    handwritten: Path = Path("data/handwritten/handwritten.jsonl")
    datasets: DatasetsConfig
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    classifier_model: str
    classifier_candidates: list[str] = Field(default_factory=list)
    judge: LlmConfig
    agent: AgentConfig
    require_free_models: bool = True
    max_calls: int = 340
    min_seconds_between_requests: float = 3.0

    @model_validator(mode="after")
    def _free_only(self) -> EvalConfig:
        if self.require_free_models:
            for llm in (self.judge, self.agent.model):
                if llm.provider == "openrouter":
                    ensure_free_models([llm.model, *llm.fallback_models])
        return self


def load_eval_config(path: Path) -> EvalConfig:
    return EvalConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
