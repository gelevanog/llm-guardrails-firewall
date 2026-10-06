"""Wire settings into a running firewall: policies, classifier, judge, PII Shield, upstream, audit, approvals."""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from bulwark.audit import AuditLog
from bulwark.classifier import ClassifierUnavailableError, InjectionClassifier, classifier_installed
from bulwark.config import Settings
from bulwark.core import ApprovalRequest
from bulwark.firewall import Firewall
from bulwark.guards.judge import LlmJudge
from bulwark.guards.secrets import PiiShieldClient
from bulwark.logging_config import get_logger
from bulwark.policy import PolicySet
from bulwark.providers.base import ChatProvider, ProviderError
from bulwark.providers.factory import budgeted, build_provider

log = get_logger(__name__)


class Tenant(BaseModel):
    name: str
    api_key_sha256: str
    policy: str
    allowed_policies: list[str] = Field(default_factory=list)
    """Policies this tenant may pick with the X-Bulwark-Policy header or a /r/<policy>/ route."""


class TenantRegistry:
    """API key (by SHA-256) -> tenant. Without a tenants file the gateway is open (local development)."""

    def __init__(self, tenants: list[Tenant]) -> None:
        self._by_hash = {tenant.api_key_sha256.lower(): tenant for tenant in tenants}

    @classmethod
    def from_file(cls, path: Path | None) -> TenantRegistry:
        if path is None:
            return cls([])
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls([Tenant.model_validate(item) for item in raw.get("tenants", [])])

    @property
    def enabled(self) -> bool:
        return bool(self._by_hash)

    def lookup(self, api_key: str | None) -> Tenant | None:
        if not api_key:
            return None
        return self._by_hash.get(hashlib.sha256(api_key.encode("utf-8")).hexdigest())


class ApprovalStore:
    """Pending tool-call approvals, in memory with a TTL (one gateway replica; use sticky sessions or a DB beyond)."""

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 1000) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._items: OrderedDict[str, tuple[float, ApprovalRequest]] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, request: ApprovalRequest) -> None:
        with self._lock:
            self._items[request.id] = (time.monotonic(), request)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def _live(self) -> list[ApprovalRequest]:
        now = time.monotonic()
        for key in [k for k, (ts, _) in self._items.items() if now - ts > self.ttl]:
            del self._items[key]
        return [request for _, request in self._items.values()]

    def list(self) -> list[ApprovalRequest]:
        with self._lock:
            return list(reversed(self._live()))

    def get(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            self._live()
            item = self._items.get(approval_id)
            return item[1] if item else None

    def decide(self, approval_id: str, approve: bool) -> ApprovalRequest | None:
        with self._lock:
            self._live()
            item = self._items.get(approval_id)
            if item is None:
                return None
            request = item[1]
            if request.status == "pending":
                request.status = "approved" if approve else "denied"
            return request


@dataclass
class Runtime:
    settings: Settings
    firewall: Firewall
    upstream: ChatProvider
    audit: AuditLog
    tenants: TenantRegistry
    approvals: ApprovalStore
    real_provider: ChatProvider | None = None
    """A budgeted real model for the playground and agent demo (None without a key)."""
    classifier_status: str = "disabled"
    judge_status: str = "not configured"
    pii_shield_status: str = "not configured"


def build_classifier(settings: Settings) -> tuple[InjectionClassifier | None, str]:
    if not settings.classifier_enabled:
        return None, "disabled (heuristics only)"
    if not classifier_installed():
        log.warning("classifier.not_installed")
        return None, 'not installed (pip install "bulwark[classifier]"); heuristics only'
    classifier = InjectionClassifier(settings.classifier_model, threads=settings.classifier_threads)
    status = f"{settings.classifier_model} (loads on first use)"
    if settings.classifier_preload:
        try:
            classifier.load()
            status = f"{settings.classifier_model} (loaded in {classifier.load_seconds}s)"
        except ClassifierUnavailableError as exc:
            # Kept in place: every request will report the failure, and fail-closed policies refuse to pass.
            status = f"unavailable: {exc}"
            log.error("classifier.unavailable", error=str(exc)[:200])
    return classifier, status


def build_judge(settings: Settings) -> tuple[LlmJudge | None, str]:
    if not settings.has_key(settings.judge_provider):
        return None, "not configured (no key for the judge provider)"
    try:
        provider = budgeted(
            build_provider(
                settings,
                kind=settings.judge_provider,
                model=settings.judge_model,
                fallback_models=settings.judge_fallback_models,
            ),
            settings,
            tag="judge",
        )
    except ProviderError as exc:
        return None, f"unavailable: {exc}"
    judge = LlmJudge(provider, model=settings.judge_model, openrouter=settings.judge_provider == "openrouter")
    return judge, f"{settings.judge_provider}/{settings.judge_model} (used by policies with judge: true)"


def build_runtime(
    settings: Settings,
    *,
    upstream: ChatProvider | None = None,
    classifier: InjectionClassifier | str | None = "auto",
    judge: LlmJudge | str | None = "auto",
) -> Runtime:
    policies = PolicySet.from_dir(settings.policies_dir, settings.default_policy)
    if classifier == "auto":
        model, classifier_status = build_classifier(settings)
    else:
        model = classifier if isinstance(classifier, InjectionClassifier) else None
        classifier_status = f"{model.model_name} (provided)" if model else "disabled (heuristics only)"
    if judge == "auto":
        llm_judge, judge_status = build_judge(settings)
    else:
        llm_judge = judge if isinstance(judge, LlmJudge) else None
        judge_status = "provided" if llm_judge else "not configured"
    pii_shield = (
        PiiShieldClient(
            settings.pii_shield_url,
            policy=settings.pii_shield_policy,
            timeout_seconds=settings.pii_shield_timeout_seconds,
        )
        if settings.pii_shield_url
        else None
    )
    firewall = Firewall(policies, classifier=model, judge=llm_judge, pii_shield=pii_shield)
    upstream = upstream or build_provider(settings)
    # The dashboard's "real model": the configured upstream (or OpenRouter when a key is set), always through the
    # budget, cache and call ledger, so demo runs are counted and evaluated conversations replay from the cache.
    real: ChatProvider | None = None
    kind = settings.upstream_provider if settings.upstream_provider != "fake" else "openrouter"
    if settings.has_key(kind):
        try:
            real = budgeted(build_provider(settings, kind=kind), settings, tag="dashboard")
        except ProviderError as exc:
            log.warning("dashboard.real_unavailable", error=str(exc)[:200])
    key = settings.audit_key.encode() if settings.audit_key else secrets.token_bytes(32)
    audit = AuditLog(key, settings.audit_max_entries, settings.audit_file)
    if settings.audit_file is not None:
        audit.preload(AuditLog.load_jsonl(settings.audit_file)[-settings.audit_max_entries :])
    return Runtime(
        settings=settings,
        firewall=firewall,
        upstream=upstream,
        audit=audit,
        tenants=TenantRegistry.from_file(settings.tenants_file),
        approvals=ApprovalStore(settings.approval_ttl_seconds),
        real_provider=real,
        classifier_status=classifier_status,
        judge_status=judge_status,
        pii_shield_status=f"{settings.pii_shield_url} (policy {settings.pii_shield_policy})"
        if settings.pii_shield_url
        else "not configured",
    )
