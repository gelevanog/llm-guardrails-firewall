"""Build providers from settings."""

from __future__ import annotations

from bulwark.config import DEFAULT_MODELS, ProviderKind, Settings
from bulwark.providers.anthropic_provider import AnthropicProvider
from bulwark.providers.base import ChatProvider
from bulwark.providers.fake import FakeProvider
from bulwark.providers.openai_compat import OpenAICompatibleProvider
from bulwark.providers.resilient import CallLedger, DiskCache, ResilientProvider, Throttle


def build_provider(
    settings: Settings,
    kind: ProviderKind | None = None,
    model: str | None = None,
    fallback_models: list[str] | None = None,
) -> ChatProvider:
    kind = kind or settings.upstream_provider
    if kind == "fake":
        return FakeProvider(model or "fake-gullible")
    if kind == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            default_model=model
            or (settings.upstream_model if settings.upstream_provider == "anthropic" else "")
            or DEFAULT_MODELS["anthropic"],
            timeout_seconds=settings.upstream_timeout_seconds,
        )
    default_model = model or (settings.upstream_model if settings.upstream_provider == kind else "")
    default_model = default_model or DEFAULT_MODELS[kind]
    return OpenAICompatibleProvider(
        kind=kind,
        api_key=settings.openrouter_api_key if kind == "openrouter" else settings.openai_api_key,
        base_url=settings.openrouter_base_url if kind == "openrouter" else settings.openai_base_url,
        default_model=default_model,
        fallback_models=settings.upstream_fallback_models if fallback_models is None else fallback_models,
        require_free=settings.require_free_models,
        timeout_seconds=settings.upstream_timeout_seconds,
    )


def budgeted(provider: ChatProvider, settings: Settings, tag: str) -> ResilientProvider:
    """Wrap a provider with the settings' cache, throttle, retries and call ledger (for real runs)."""
    return ResilientProvider(
        provider,
        ledger=CallLedger(settings.llm_ledger, settings.llm_max_calls),
        cache=DiskCache(settings.llm_cache_dir),
        throttle=Throttle(settings.llm_min_seconds_between_requests),
        max_retries=settings.llm_max_retries,
        tag=tag,
    )
