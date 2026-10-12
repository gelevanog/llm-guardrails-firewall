"""Bulwark: an LLM firewall against prompt injection, jailbreaks and data exfiltration.

The public names load on first use, so a short-lived process that needs one module (the Claude Code hooks run
once per tool call) does not pay for importing every guard.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bulwark.core import Action, ApprovalRequest, Decision, Finding, GuardResult
    from bulwark.firewall import Firewall, GuardSession
    from bulwark.policy import Policy, PolicySet
    from bulwark.stream import StreamGuard
    from bulwark.taint import TaintState

__version__ = "0.1.0"

_EXPORTS = {
    "Action": "bulwark.core",
    "ApprovalRequest": "bulwark.core",
    "Decision": "bulwark.core",
    "Finding": "bulwark.core",
    "GuardResult": "bulwark.core",
    "Firewall": "bulwark.firewall",
    "GuardSession": "bulwark.firewall",
    "Policy": "bulwark.policy",
    "PolicySet": "bulwark.policy",
    "StreamGuard": "bulwark.stream",
    "TaintState": "bulwark.taint",
}

__all__ = [
    "Action",
    "ApprovalRequest",
    "Decision",
    "Finding",
    "Firewall",
    "GuardResult",
    "GuardSession",
    "Policy",
    "PolicySet",
    "StreamGuard",
    "TaintState",
    "__version__",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'bulwark' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value
