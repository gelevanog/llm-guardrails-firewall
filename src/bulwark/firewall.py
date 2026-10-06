"""Library facade: `Firewall` (shared detectors and policies) and `GuardSession` (one conversation).

    firewall = Firewall.create(classifier=True)
    session = firewall.session("email-agent")
    system = session.protect_system_prompt("You are the support inbox assistant...")
    await session.check_input(user_message)                     # direct injection, jailbreak, topic
    safe = await session.check_untrusted(email_body, source="read_email")   # quarantine + spotlight
    messages.append({"role": "tool", "content": safe.text})
    decision = session.check_tool_call("send_email", arguments)  # allow / approval / block
    answer = await session.check_output(model_answer)            # leakage, exfiltration, secrets

Every check returns a `Decision` with per-guard scores, actions and explanations.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from pathlib import Path
from typing import Any

from bulwark.classifier import InjectionClassifier
from bulwark.config import PACKAGED_POLICIES
from bulwark.core import Action, ApprovalRequest, Decision, GuardResult, Stage, strictest
from bulwark.guards.exfiltration import check_exfiltration
from bulwark.guards.injection import InjectionDetector, injection_result
from bulwark.guards.judge import LlmJudge
from bulwark.guards.leakage import CANARY_PREFIX, check_leakage, make_canary, with_canary
from bulwark.guards.secrets import PiiShieldClient, check_secrets
from bulwark.guards.spotlight import spotlight_instructions
from bulwark.guards.tools import check_tool_call
from bulwark.guards.topic import check_topics
from bulwark.guards.untrusted import UntrustedContentGuard, UntrustedReport
from bulwark.policy import Policy, PolicySet
from bulwark.taint import TaintState


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


class Firewall:
    def __init__(
        self,
        policies: PolicySet,
        *,
        classifier: InjectionClassifier | None = None,
        judge: LlmJudge | None = None,
        pii_shield: PiiShieldClient | None = None,
        seed: bytes | None = None,
    ) -> None:
        self.policies = policies
        self.seed = seed
        """Evaluation and demo only: a fixed secret, so identical conversations produce identical requests (and
        cached model answers replay). Without it the secret is random per process."""
        self._secret = seed or secrets.token_bytes(32)
        self.detector = InjectionDetector(classifier=classifier, judge=judge)
        self.untrusted_guard = UntrustedContentGuard(self.detector)
        self.pii_shield = pii_shield

    @classmethod
    def create(
        cls,
        default_policy: str = "default",
        *,
        classifier: bool | str = False,
        judge: LlmJudge | None = None,
        pii_shield_url: str | None = None,
        policies_dir: Path | None = None,
    ) -> Firewall:
        """Packaged policies; `classifier=True` (or a model id) loads the classifier (needs the extra)."""
        model = None
        if classifier:
            model = InjectionClassifier(classifier) if isinstance(classifier, str) else InjectionClassifier()
        return cls(
            PolicySet.from_dir(policies_dir or PACKAGED_POLICIES, default_policy),
            classifier=model,
            judge=judge,
            pii_shield=PiiShieldClient(pii_shield_url) if pii_shield_url else None,
        )

    def derived(self, material: str, length: int) -> str:
        """A keyed hash: unpredictable without the secret, stable for the same input. Spotlight ids and canaries
        use it so a conversation's earlier turns stay byte-identical (provider prompt caching keeps working)."""
        return hmac.new(self._secret, material.encode("utf-8"), hashlib.sha256).hexdigest()[:length]

    @property
    def classifier(self) -> InjectionClassifier | None:
        return self.detector.classifier

    def policy(self, name: str | Policy | None = None) -> Policy:
        return name if isinstance(name, Policy) else self.policies.get(name)

    def session(self, policy: str | Policy | None = None) -> GuardSession:
        return GuardSession(self, self.policy(policy))

    # ------------------------------------------------------------------------------------- stages
    async def check_input(self, text: str, *, policy: str | Policy | None = None) -> Decision:
        """Direct attacks in a user message: injection / jailbreak (layered) and topic restrictions."""
        rules = self.policy(policy)
        started = time.perf_counter()
        results: list[GuardResult] = []
        injection = rules.input.injection
        if injection.enabled:
            stage_started = time.perf_counter()
            detection = await self.detector.detect(
                text,
                injection.layers,
                "input",
                judge_band=injection.judge_band,
                fail_closed=rules.fail_mode == "closed",
            )
            results.append(injection_result(detection, injection, stage_started))
        if rules.input.topics.enabled:
            results.append(check_topics(text, rules.input.topics))
        return _decision("input", rules, results, latency_ms=_ms(started))

    async def check_untrusted(
        self, content: str, *, source: str, policy: str | Policy | None = None
    ) -> tuple[Decision, UntrustedReport]:
        """Indirect injection in content the model will read; `Decision.text` is what to give the model."""
        rules = self.policy(policy)
        started = time.perf_counter()
        config = rules.untrusted
        if not config.enabled:
            report = UntrustedReport(source=source, flagged=False, score=0.0, sanitized=content, model_text=content)
            return _decision("untrusted", rules, [], text=content), report
        result, report = await self.untrusted_guard.inspect(
            content, source, config, fail_closed=rules.fail_mode == "closed", block_id=self.derived(source + content, 8)
        )
        decision = _decision("untrusted", rules, [result], latency_ms=_ms(started))
        # Spotlighting and hidden-text stripping always apply; flagged spans only when the policy is enforced.
        decision.text = report.model_text if decision.enforced or not report.flagged else content
        return decision, report

    async def check_output(
        self,
        text: str,
        *,
        policy: str | Policy | None = None,
        system_prompt: str | None = None,
        canary: str | None = None,
        conversation: str = "",
        pii_shield: bool = True,
    ) -> Decision:
        """The model's answer: system-prompt leakage, exfiltration links and images, secrets."""
        rules = self.policy(policy)
        started = time.perf_counter()
        results: list[GuardResult] = []
        sanitized = text
        output = rules.output
        if output.leakage.enabled:
            results.append(check_leakage(text, system_prompt, canary, output.leakage))
        if output.exfiltration.enabled:
            result, cleaned = check_exfiltration(sanitized, output.exfiltration, conversation)
            results.append(result)
            if result.triggered and result.action is Action.SANITIZE:
                sanitized = cleaned
        if output.secrets.enabled:
            config = output.secrets if pii_shield else output.secrets.model_copy(update={"pii_shield": False})
            result, cleaned = await check_secrets(
                sanitized, config, self.pii_shield, fail_closed=rules.fail_mode == "closed"
            )
            results.append(result)
            if result.triggered and result.action is Action.SANITIZE:
                sanitized = cleaned
        decision = _decision("output", rules, results, latency_ms=_ms(started))
        if decision.enforced and sanitized != text:
            decision.text = sanitized
        return decision

    def check_tool_call(
        self,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        taint: TaintState | None = None,
        policy: str | Policy | None = None,
    ) -> Decision:
        rules = self.policy(policy)
        started = time.perf_counter()
        if not rules.tools.enabled:
            return _decision("tool_call", rules, [])
        result, approval = check_tool_call(name, arguments, taint or TaintState(), rules.tools)
        return _decision("tool_call", rules, [result], approval=approval, latency_ms=_ms(started))


def _decision(
    stage: Stage,
    policy: Policy,
    results: list[GuardResult],
    *,
    text: str | None = None,
    approval: ApprovalRequest | None = None,
    latency_ms: float = 0.0,
) -> Decision:
    action = strictest([r.action for r in results if r.triggered])
    return Decision(
        stage=stage,
        policy=policy.name,
        action=action,
        results=results,
        text=text,
        approval=approval if policy.enforced else None,
        latency_ms=latency_ms,
        enforced=policy.enforced and action is not Action.LOG_ONLY,
    )


class GuardSession:
    """One conversation: its policy, taint state, canary and system prompt, and every decision made."""

    def __init__(self, firewall: Firewall, policy: Policy) -> None:
        self.firewall = firewall
        self.policy = policy
        self.taint = TaintState()
        self.canary = f"{CANARY_PREFIX}-{firewall.derived('canary', 12)}" if firewall.seed else make_canary()
        self.system_prompt: str | None = None
        self.decisions: list[Decision] = []
        self._conversation: list[str] = []

    def protect_system_prompt(self, prompt: str, *, spotlight_note: bool = True) -> str:
        """The system prompt to send: yours, plus the spotlighting note and the canary."""
        self.system_prompt = prompt
        self.canary = f"{CANARY_PREFIX}-{self.firewall.derived('canary:' + prompt, 12)}"
        self.taint.add_trusted(prompt)
        note = spotlight_instructions(self.policy.untrusted.spotlight) if spotlight_note else ""
        full = f"{prompt.rstrip()}\n\n{note}" if note else prompt
        return with_canary(full, self.canary) if self.policy.output.leakage.canary else full

    def add_trusted(self, text: str) -> None:
        """Data the application vouches for, e.g. email headers from the mail server."""
        self.taint.add_trusted(text)
        self._conversation.append(text)

    async def check_input(self, text: str) -> Decision:
        self.taint.add_trusted(text)
        self._conversation.append(text)
        return self._keep(await self.firewall.check_input(text, policy=self.policy))

    async def check_untrusted(self, content: str, *, source: str) -> Decision:
        decision, report = await self.firewall.check_untrusted(content, source=source, policy=self.policy)
        if self.policy.untrusted.taint:
            self.taint.add_untrusted(content, source, flagged=report.flagged, score=report.score)
        self._conversation.append(report.sanitized)
        self.last_report = report
        return self._keep(decision)

    def check_tool_call(self, name: str, arguments: str | dict[str, Any] | None) -> Decision:
        return self._keep(self.firewall.check_tool_call(name, arguments, taint=self.taint, policy=self.policy))

    async def check_output(self, text: str) -> Decision:
        decision = await self.firewall.check_output(
            text,
            policy=self.policy,
            system_prompt=self.system_prompt,
            canary=self.canary,
            conversation="\n".join(self._conversation),
        )
        return self._keep(decision)

    def conversation_text(self) -> str:
        return "\n".join(self._conversation)

    def _keep(self, decision: Decision) -> Decision:
        self.decisions.append(decision)
        return decision

    last_report: UntrustedReport | None = None
