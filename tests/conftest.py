"""Shared fixtures. Nothing here needs an API key or a model download."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from fastapi import FastAPI

from bulwark.config import PACKAGED_POLICIES, Settings
from bulwark.firewall import Firewall
from bulwark.gateway.app import create_app
from bulwark.gateway.runtime import Runtime, build_runtime
from bulwark.policy import PolicySet
from bulwark.providers.fake import FakeProvider, RecordingProvider


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BULWARK_PII_SHIELD_URL"):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def policies() -> PolicySet:
    return PolicySet.from_dir(PACKAGED_POLICIES, "default")


@pytest.fixture
def firewall(policies: PolicySet) -> Firewall:
    """Heuristics only: the classifier is an optional extra and is tested separately when cached."""
    return Firewall(policies)


@pytest.fixture
def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        classifier_enabled=False, openrouter_api_key=None, audit_file=None, tenants_file=None, _env_file=None
    )


@pytest.fixture
def recorder() -> RecordingProvider:
    return RecordingProvider(FakeProvider())


@pytest.fixture
def runtime(settings: Settings, recorder: RecordingProvider) -> Runtime:
    return build_runtime(settings, upstream=recorder, classifier=None, judge=None)


@pytest.fixture
def app(settings: Settings, runtime: Runtime) -> FastAPI:
    return create_app(settings, runtime)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as http:
        yield http
