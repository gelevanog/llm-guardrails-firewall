"""Topic and use-policy restrictions per route (e.g. a support bot only talks about orders and products).

Topics are defined in the policy as lists of regular expressions. A request that matches a denied topic is
refused; with an allow-list, a request must match at least one allowed topic (short greetings and thanks
always pass). This is deliberately simple and predictable: it states the bot's scope in reviewable YAML.
It is lexical, so it can be phrased around; a deployment that needs semantic topic control can add the
LLM judge on the same route.
"""

from __future__ import annotations

import re
import time

from bulwark.core import Action, Finding, GuardResult
from bulwark.policy import TopicConfig

_SMALL_TALK = re.compile(
    r"^\s*(?:hi|hello|hey|thanks?|thank\s+you|ok(?:ay)?|good\s+(?:morning|afternoon|evening)|bye|cheers|hallo|hola|"
    r"bonjour|danke|gracias|merci|привет|спасибо)\b[\s!.,?]*",
    re.IGNORECASE,
)


def matched_topics(text: str, config: TopicConfig, names: list[str]) -> dict[str, str]:
    """Topic name -> the first matching text."""
    hits: dict[str, str] = {}
    for name in names:
        for pattern in config.definitions.get(name, []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                hits[name] = match.group(0)
                break
    return hits


def check_topics(text: str, config: TopicConfig) -> GuardResult:
    started = time.perf_counter()
    findings: list[Finding] = []
    action = Action.ALLOW
    denied = matched_topics(text, config, config.denied)
    for name, snippet in denied.items():
        findings.append(
            Finding(
                rule=f"topic.denied.{name}",
                layer="topic",
                score=1.0,
                message=f"the request touches a topic this route refuses: {name}",
                snippet=snippet,
            )
        )
        action = config.action
    if not denied and config.allowed:
        allowed = matched_topics(text, config, config.allowed)
        words = len(re.findall(r"\w+", text))
        if not allowed and words >= config.min_words_for_off_topic and not _SMALL_TALK.fullmatch(text):
            findings.append(
                Finding(
                    rule="topic.off_topic",
                    layer="topic",
                    score=0.7,
                    message=f"the request is outside this route's topics ({', '.join(config.allowed)})",
                )
            )
            action = config.off_topic_action
    score = max((f.score for f in findings), default=0.0)
    return GuardResult(
        guard="topic",
        score=score,
        triggered=action is not Action.ALLOW,
        action=action,
        explanation="; ".join(f.message for f in findings),
        findings=findings,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
