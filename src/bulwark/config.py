"""Runtime settings from environment variables (and an optional .env file). See .env.example."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from bulwark.classifier import DEFAULT_CLASSIFIER_MODEL

ProviderKind = Literal["fake", "openrouter", "openai", "anthropic"]

# Free OpenRouter models that answered with tool calls in the smoke test on 2026-10-06 (results/smoke_tools.json).
DEFAULT_FREE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_FREE_FALLBACKS = ["dots-studio/dots-3-note-preview:free", "google/gemma-4-31b-it:free"]
DEFAULT_JUDGE_MODEL = "dots-studio/dots-3-note-preview:free"
DEFAULT_JUDGE_FALLBACKS = ["nvidia/nemotron-3-ultra-550b-a55b:free"]
DEFAULT_MODELS: dict[str, str] = {
    "fake": "fake-gullible",
    "openrouter": DEFAULT_FREE_MODEL,
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-5",
}
PACKAGED_POLICIES = Path(__file__).parent / "policies"


def _split(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BULWARK_", extra="ignore")

    # ---- upstream LLM
    upstream_provider: ProviderKind = "fake"
    upstream_model: str = ""
    """Default model when the client sends none (or "auto"); per-provider default if empty."""
    upstream_fallback_models: Annotated[list[str], NoDecode] = Field(default_factory=list)
    require_free_models: bool = True
    """Refuse any OpenRouter model id that does not end in ":free" (requests and served models)."""
    upstream_timeout_seconds: float = 120.0

    openrouter_api_key: str | None = Field(default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY"))
    openai_api_key: str | None = Field(default=None, validation_alias=AliasChoices("OPENAI_API_KEY"))
    anthropic_api_key: str | None = Field(default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY"))
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias=AliasChoices("OPENROUTER_BASE_URL")
    )
    openai_base_url: str = Field(default="https://api.openai.com/v1", validation_alias=AliasChoices("OPENAI_BASE_URL"))

    # ---- policies and tenants
    policies_dir: Path = PACKAGED_POLICIES
    default_policy: str = "default"
    allow_policy_header: bool = True
    """Let clients pick a policy with the X-Bulwark-Policy header (tenants can still be restricted)."""
    tenants_file: Path | None = None

    # ---- detection layers
    classifier_enabled: bool = True
    classifier_model: str = DEFAULT_CLASSIFIER_MODEL
    classifier_threads: int = 4
    classifier_preload: bool = True
    """Load the classifier at startup instead of on the first request."""
    judge_provider: ProviderKind = "openrouter"
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_fallback_models: Annotated[list[str], NoDecode] = Field(default_factory=lambda: list(DEFAULT_JUDGE_FALLBACKS))

    # ---- PII Shield integration (optional): secrets/PII check of model answers over HTTP
    pii_shield_url: str | None = None
    """Base URL of a PII Shield gateway, e.g. http://localhost:8001. Unset = built-in secret patterns only."""
    pii_shield_policy: str = "support-chat"
    pii_shield_timeout_seconds: float = 10.0

    # ---- approvals, audit log and dashboard
    approval_ttl_seconds: int = 3600
    audit_file: Path | None = None
    audit_max_entries: int = 2000
    audit_key: str | None = None
    """Secret for the keyed hashes in the audit log. Unset = random per process."""
    results_dir: Path = Path("results")

    # ---- real-API budget (eval, judge, agent demo, playground)
    llm_cache_dir: Path = Path(".cache/llm")
    llm_ledger: Path = Path("results/calls.jsonl")
    llm_max_calls: int = 340
    llm_min_seconds_between_requests: float = 3.0
    llm_max_retries: int = 4

    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL", "BULWARK_LOG_LEVEL"))
    log_format: Literal["console", "json"] = Field(
        default="console", validation_alias=AliasChoices("LOG_FORMAT", "BULWARK_LOG_FORMAT")
    )

    _split_lists = field_validator("upstream_fallback_models", "judge_fallback_models", mode="before")(_split)

    def upstream_default_model(self) -> str:
        return self.upstream_model or DEFAULT_MODELS[self.upstream_provider]

    def has_key(self, provider: ProviderKind) -> bool:
        return provider == "fake" or bool(
            {"openrouter": self.openrouter_api_key, "openai": self.openai_api_key, "anthropic": self.anthropic_api_key}[
                provider
            ]
        )
