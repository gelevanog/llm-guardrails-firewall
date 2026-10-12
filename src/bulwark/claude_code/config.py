"""Hook configuration: which tools bring in untrusted content, what counts as risky, and what to do about it.

Defaults are built in; a YAML or JSON file (`--config PATH` or `BULWARK_HOOK_CONFIG`) overrides any field, and a
few environment variables override the rest (they are what a `settings.json` hook entry can set most easily):

    BULWARK_HOOK_SCAN_URL      a running Bulwark gateway's /v1/scan (adds the classifier); unset = heuristics only
    BULWARK_HOOK_SCAN_TIMEOUT  seconds for that call (default 2.0); on timeout or error the hook falls back locally
    BULWARK_HOOK_SCAN_TOKEN    bearer token for the gateway (tenant key), if it requires one
    BULWARK_HOOK_SCAN_POLICY   gateway policy to scan with (default: the gateway's default policy)
    BULWARK_HOOK_FAIL_CLOSED   1 = an internal hook error denies the tool call instead of letting it through
    BULWARK_HOOK_STATE_DIR     where per-session state lives (default: the user cache dir, bulwark/claude-code)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HookDecision(StrEnum):
    """Ordered from least to most restrictive. `allow` means "no opinion": the hook never auto-approves."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

    @property
    def severity(self) -> int:
        return {"allow": 0, "ask": 1, "deny": 2}[self.value]


class ByRisk(BaseModel):
    """A decision per kind of risky action."""

    model_config = ConfigDict(extra="forbid")

    network: HookDecision = HookDecision.ALLOW
    """Bash commands that talk to the network (curl, wget, ssh, scp, git push, ...)."""
    fetch: HookDecision = HookDecision.ALLOW
    """WebFetch, WebSearch and MCP browsing tools: a URL can carry data in its query string."""
    send: HookDecision = HookDecision.ALLOW
    """MCP tools that send, post, write or share (email, chat, issues, files)."""

    def get(self, risk: str) -> HookDecision:
        value: HookDecision = getattr(self, risk)
        return value


DEFAULT_NETWORK_COMMANDS = [
    r"\b(?:curl|wget|aria2c|xh|ssh|scp|sftp|rsync|ftp|tftp|telnet|nc|ncat|netcat|socat|nmap)\b",
    r"/dev/(?:tcp|udp)/",
    r"\b(?:Invoke-WebRequest|Invoke-RestMethod|iwr|irm)\b",
    r"\bgit\s+(?:push|clone|fetch|pull|remote\s+(?:add|set-url))\b",
    r"\bgh\s+(?:api|gist|issue|pr|release|repo)\b",
    r"\b(?:npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b|\bdocker\s+(?:push|login)\b",
    r"(?:^|[;&|(]\s*)(?:mail|sendmail|mutt|msmtp)\b",
    r"\b(?:pip3?|uv\s+pip|uv\s+add|npm|pnpm|yarn|gem|cargo)\s+(?:install|add|i)\b[^\n]*(?:https?://|git\+|--index-url)",
    r"\b(?:python3?|node|deno|bun|ruby|perl|php)\b[^\n]*\b(?:requests|urllib|httpx|aiohttp|http\.client|socket|fetch\(|"
    r"axios|Net::HTTP|LWP|smtplib)",
]


class HookConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ------------------------------------------------------------------ PostToolUse: what to scan
    scan_tools: str = r"^(?:WebFetch|WebSearch|Read|Bash|Grep|mcp__.*)$"
    """Tools whose results are untrusted content (a regex over the tool name). The settings.json matcher decides
    which calls reach the hook at all; this is a second filter, so a broad matcher is safe."""
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Score at or above which a tool result counts as a prompt injection (the default policy's threshold)."""
    max_chars: int = Field(default=100_000, ge=1_000)
    """Longer tool results are scanned up to this many characters (the head and the tail)."""

    # ------------------------------------------------------------------ PreToolUse: what is risky
    network_commands: list[str] = Field(default_factory=lambda: list(DEFAULT_NETWORK_COMMANDS))
    """Regexes over a Bash command that mark it as a network action."""
    fetch_tools: str = (
        r"^(?:WebFetch|WebSearch)$"
        r"|^mcp__.+__(?:\w*?[_-])?(?:fetch|navigate|browse|open_url|http|request|download|scrape|crawl)(?:[_-]|s?$)"
    )
    send_tools: str = (
        r"^mcp__.+__(?:\w*?[_-])?(?:send|post|reply|forward|create|write|update|delete|remove|upload|publish|share"
        r"|comment|invite|transfer|pay|execute|push|merge)(?:[_-]|s?$)"
    )
    """MCP tool names (case-insensitive) that act on someone's behalf."""
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "::1", "0.0.0.0"])
    """Destinations that are never treated as untrusted (and their subdomains), e.g. your company's domains."""

    # ------------------------------------------------------------------ PreToolUse: what to do
    on_taint: ByRisk = Field(default_factory=lambda: ByRisk(send=HookDecision.ASK))
    """Untrusted content entered the session, no detector fired."""
    on_suspicious: ByRisk = Field(
        default_factory=lambda: ByRisk(network=HookDecision.ASK, fetch=HookDecision.ASK, send=HookDecision.DENY)
    )
    """A scanned tool result was flagged as a prompt injection earlier in this session."""
    untrusted_destination: ByRisk = Field(
        default_factory=lambda: ByRisk(network=HookDecision.ASK, send=HookDecision.DENY)
    )
    """A URL, host or address in the arguments appears only in untrusted content, not in the user's messages."""
    flagged_destination: ByRisk = Field(
        default_factory=lambda: ByRisk(network=HookDecision.DENY, fetch=HookDecision.DENY, send=HookDecision.DENY)
    )
    """... and that content was flagged: an outsider's instruction chose where the data goes."""

    # ------------------------------------------------------------------ runtime (usually from the environment)
    scan_url: str | None = None
    scan_timeout: float = Field(default=2.0, gt=0.0, le=30.0)
    scan_token: str | None = Field(default=None, repr=False)
    scan_policy: str | None = None
    fail_closed: bool = False
    state_dir: Path | None = None
    state_ttl_hours: float = Field(default=72.0, gt=0.0)
    """State files of sessions that ended without a SessionEnd hook are pruned after this long."""

    def resolved_state_dir(self) -> Path:
        return self.state_dir or default_state_dir()


def default_state_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        return Path(base) / "bulwark" / "claude-code"
    cache = os.environ.get("XDG_CACHE_HOME") or (str(Path.home() / ".cache") if Path.home().exists() else "")
    return Path(cache or tempfile.gettempdir()) / "bulwark" / "claude-code"


_TRUE = {"1", "true", "yes", "on"}


def load_config(path: Path | None = None, env: Mapping[str, str] | None = None) -> HookConfig:
    environ = os.environ if env is None else env
    data: dict[str, Any] = {}
    file = path or (Path(environ["BULWARK_HOOK_CONFIG"]) if environ.get("BULWARK_HOOK_CONFIG") else None)
    if file is not None:
        text = file.read_text(encoding="utf-8")
        if file.suffix == ".json":
            loaded = json.loads(text)
        else:
            import yaml

            loaded = yaml.safe_load(text)
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"{file}: the hook config must be a mapping")
        data.update(loaded or {})
    if environ.get("BULWARK_HOOK_SCAN_URL"):
        data["scan_url"] = environ["BULWARK_HOOK_SCAN_URL"]
    if environ.get("BULWARK_HOOK_SCAN_TIMEOUT"):
        data["scan_timeout"] = float(environ["BULWARK_HOOK_SCAN_TIMEOUT"])
    if environ.get("BULWARK_HOOK_SCAN_TOKEN"):
        data["scan_token"] = environ["BULWARK_HOOK_SCAN_TOKEN"]
    if environ.get("BULWARK_HOOK_SCAN_POLICY"):
        data["scan_policy"] = environ["BULWARK_HOOK_SCAN_POLICY"]
    if environ.get("BULWARK_HOOK_FAIL_CLOSED"):
        data["fail_closed"] = environ["BULWARK_HOOK_FAIL_CLOSED"].strip().lower() in _TRUE
    if environ.get("BULWARK_HOOK_STATE_DIR"):
        data["state_dir"] = environ["BULWARK_HOOK_STATE_DIR"]
    return HookConfig.model_validate(data)


def fail_closed_requested(env: Mapping[str, str] | None = None) -> bool:
    """Read before anything else can fail, so a broken config file still honors the operator's choice."""
    environ = os.environ if env is None else env
    return environ.get("BULWARK_HOOK_FAIL_CLOSED", "").strip().lower() in _TRUE
