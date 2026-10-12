"""Evaluation config (configs/eval.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bulwark.config import ProviderKind
from bulwark.providers.base import FreeModelGuardError, ensure_free_models


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LlmConfig(_Strict):
    provider: ProviderKind = "openrouter"
    model: str
    fallback_models: list[str] = Field(default_factory=list)
    base_url: str | None = None
    """OpenAI-compatible endpoint for `provider: openai` (default: OPENAI_BASE_URL). Paid models only."""
    portable: bool = False
    """Judge requests without `temperature` and numeric schema bounds (some OpenAI-compatible gateways reject them)."""


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
    judge_limit: int | None = None
    """Stop after this many judge calls (a smoke run on a tiny token budget). None = no limit."""


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
                if llm.provider in {"openai", "anthropic"} or llm.base_url:
                    raise FreeModelGuardError(
                        f"free-only guard: provider {llm.provider!r} (model {llm.model!r}) is paid; set "
                        "`require_free_models: false` in the config and BULWARK_ALLOW_PAID_MODELS=1 to opt in"
                    )
                if llm.provider == "openrouter":
                    ensure_free_models([llm.model, *llm.fallback_models])
        return self

    def check_paid_opt_in(self, allow_paid_models: bool) -> None:
        """A config file alone cannot turn spending on: `require_free_models: false` also needs the env flag."""
        if not self.require_free_models and not allow_paid_models:
            raise FreeModelGuardError(
                "this evaluation config allows paid models (require_free_models: false); "
                "set BULWARK_ALLOW_PAID_MODELS=1 to run it"
            )


def load_eval_config(path: Path) -> EvalConfig:
    return EvalConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
