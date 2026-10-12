"""Per-session taint state for the Claude Code hooks: a small JSON file per session, guarded by a file lock.

Each hook is a separate process, so what one hook learned (a fetched page was an injection, the user named
`example.com`) has to be on disk for the next one. The file holds no page text and no prompt text, only:
  * the sources of untrusted content (tool, URL or path, flagged or not, score, the detector's reasons);
  * destinations (hosts and email addresses) seen in untrusted content, and whether flagged content named them;
  * destinations the user wrote in their own prompts (trusted).
That is enough for argument provenance: "this curl goes to a host that only a flagged web page mentioned".
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

Origin = Literal["trusted", "flagged", "untrusted", "unknown"]
Level = Literal["clean", "tainted", "suspicious"]

_URL = re.compile(r"\b(?:https?|ftp|wss?)://[^\s\"'<>)\]`|]+", re.IGNORECASE)
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@((?:[\w-]+\.)+[a-z]{2,24})\b", re.IGNORECASE)
# Bare host names ("send it to collect.evil.test"): a dot-separated name ending in a plausible top-level domain.
_HOST = re.compile(r"(?<![\w@./:-])((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+([a-z]{2,24}))(?![\w-])", re.IGNORECASE)
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
# Top-level domains for bare names; two-letter names count unless they are common file extensions.
_GTLD_WORDS = (
    "com net org edu gov mil int info biz name pro xyz top site online app dev io ai co me tv cloud tech store shop"
    " blog page live link click email services support systems network digital tools space website fun club news"
    " test example invalid localhost onion"
)
_GTLDS = frozenset(_GTLD_WORDS.split())
_FILE_EXTENSION_WORDS = "py md rs sh pl rb cc cs go so db gz xz el ml mk js ts hh in"
_FILE_EXTENSIONS = frozenset(_FILE_EXTENSION_WORDS.split())

MAX_SOURCES = 200
MAX_ATOMS = 5_000
LOCK_TIMEOUT_S = 5.0


def _host_of_url(url: str) -> str | None:
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    return normalize_host(host) if host else None


def normalize_host(host: str) -> str:
    host = host.strip().strip("[]").lower().rstrip(".")
    return host.removeprefix("www.")


def destinations(text: str, *, bare_hosts: bool = True) -> set[str]:
    """Hosts and email addresses a text mentions (lower-case). Emails are kept whole: `x@evil.test`."""
    found: set[str] = set()
    for match in _URL.finditer(text):
        host = _host_of_url(match.group(0).rstrip(".,;:!?"))
        if host and ("." in host or host == "localhost" or ":" in host):
            found.add(host)
    for match in _EMAIL.finditer(text):
        found.add(match.group(0).lower())
        found.add(normalize_host(match.group(1)))
    if bare_hosts:
        for match in _HOST.finditer(text):
            tld = match.group(2).lower()
            if tld in _GTLDS or (len(tld) == 2 and tld not in _FILE_EXTENSIONS):
                found.add(normalize_host(match.group(1)))
        found.update(_IPV4.findall(text))
    return {atom for atom in found if len(atom) >= 4}


def _related(host: str, other: str) -> bool:
    return host == other or host.endswith("." + other) or other.endswith("." + host)


@dataclass
class SessionState:
    session_id: str
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    sources: list[dict[str, Any]] = field(default_factory=list)
    """{tool, source, flagged, score, reasons, at}"""
    trusted: list[str] = field(default_factory=list)
    untrusted: dict[str, dict[str, Any]] = field(default_factory=dict)
    """destination -> {"flagged": bool, "source": where it was seen (the flagged source when there is one)}"""
    scan_failures: int = 0
    """Fail-closed mode: tool results that could not be scanned count as flagged."""

    @property
    def level(self) -> Level:
        if self.scan_failures or any(source["flagged"] for source in self.sources):
            return "suspicious"
        return "tainted" if self.sources else "clean"

    def flagged_sources(self) -> list[dict[str, Any]]:
        return [source for source in self.sources if source["flagged"]]

    # ------------------------------------------------------------------ updates
    def add_source(
        self, tool: str, source: str, *, flagged: bool, score: float, reasons: list[str], atoms: Iterable[str]
    ) -> None:
        self.sources.append(
            {
                "tool": tool,
                "source": source[:300],
                "flagged": flagged,
                "score": round(score, 4),
                "reasons": reasons[:5],
                "at": round(time.time(), 3),
            }
        )
        if len(self.sources) > MAX_SOURCES:  # keep every flagged source, drop the oldest clean ones
            clean = [i for i, s in enumerate(self.sources) if not s["flagged"]]
            for index in reversed(clean[: len(self.sources) - MAX_SOURCES]):
                del self.sources[index]
        for atom in atoms:
            entry = self.untrusted.get(atom)
            if entry is None:
                if len(self.untrusted) >= MAX_ATOMS and not flagged:
                    continue
                self.untrusted[atom] = {"flagged": flagged, "source": source[:200]}
            elif flagged and not entry["flagged"]:
                entry.update(flagged=True, source=source[:200])

    def add_trusted(self, atoms: Iterable[str]) -> None:
        known = set(self.trusted)
        for atom in sorted(atoms):
            if atom not in known and len(self.trusted) < MAX_ATOMS:
                self.trusted.append(atom)
                known.add(atom)

    # ------------------------------------------------------------------ provenance
    def origin(self, atom: str, allowed_hosts: Iterable[str] = ()) -> tuple[Origin, str | None]:
        """Where a destination came from, and the source that mentioned it (for flagged and untrusted)."""
        atom = atom.lower()
        is_email = "@" in atom
        host = atom.rsplit("@", 1)[-1]
        if any(host == h or host.endswith("." + h) for h in (normalize_host(a) for a in allowed_hosts)):
            return "trusted", None
        trusted = set(self.trusted)
        if atom in trusted or (
            not is_email and any(host == t or host.endswith("." + t) for t in trusted if "@" not in t)
        ):
            return "trusted", None
        for candidate, entry in self.untrusted.items():
            if not entry["flagged"]:
                continue
            if candidate == atom or ("@" not in candidate and _related(host, candidate)):
                return "flagged", str(entry["source"])
        for candidate, entry in self.untrusted.items():
            if candidate == atom or (not is_email and "@" not in candidate and _related(host, candidate)):
                return "untrusted", str(entry["source"])
        return "unknown", None

    # ------------------------------------------------------------------ serialization
    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "session_id": self.session_id,
            "created": self.created,
            "updated": self.updated,
            "level": self.level,
            "sources": self.sources,
            "trusted": self.trusted,
            "untrusted": self.untrusted,
            "scan_failures": self.scan_failures,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SessionState:
        return cls(
            session_id=str(data["session_id"]),
            created=float(data.get("created", time.time())),
            updated=float(data.get("updated", time.time())),
            sources=list(data.get("sources", [])),
            trusted=list(data.get("trusted", [])),
            untrusted=dict(data.get("untrusted", {})),
            scan_failures=int(data.get("scan_failures", 0)),
        )


class StateStore:
    """One JSON file per session (named by a hash of the session id) in a private directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path(self, session_id: str) -> Path:
        name = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{name}.json"

    def _ensure_dir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def load(self, session_id: str) -> SessionState:
        """A consistent snapshot (read under the lock); a missing file is a clean session."""
        path = self.path(session_id)
        if not path.exists():
            return SessionState(session_id)
        with self._locked(path):
            return self._read(path, session_id)

    @contextlib.contextmanager
    def update(self, session_id: str) -> Iterator[SessionState]:
        """Read-modify-write under an exclusive lock; the file is replaced atomically."""
        self._ensure_dir()
        path = self.path(session_id)
        with self._locked(path):
            state = self._read(path, session_id)
            yield state
            state.updated = time.time()
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state.to_json(), handle, ensure_ascii=False, separators=(",", ":"))
            tmp.replace(path)

    def delete(self, session_id: str) -> bool:
        path = self.path(session_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        path.with_suffix(".lock").unlink(missing_ok=True)
        return existed

    def prune(self, max_age_s: float) -> int:
        """Remove state of sessions not updated for `max_age_s` (sessions that never ran SessionEnd)."""
        if not self.directory.is_dir():
            return 0
        cutoff = time.time() - max_age_s
        removed = 0
        for path in self.directory.glob("*.json"):
            with contextlib.suppress(OSError):
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    path.with_suffix(".lock").unlink(missing_ok=True)
                    removed += 1
        return removed

    def all_paths(self) -> list[Path]:
        return sorted(self.directory.glob("*.json")) if self.directory.is_dir() else []

    @staticmethod
    def _read(path: Path, session_id: str) -> SessionState:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return SessionState(session_id)
        return SessionState.from_json(data)

    @contextlib.contextmanager
    def _locked(self, path: Path) -> Iterator[None]:
        self._ensure_dir()
        lock_path = path.with_suffix(".lock")
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _acquire(fd)
            try:
                yield
            finally:
                _release(fd)
        finally:
            os.close(fd)


if sys.platform == "win32":
    import msvcrt

    def _acquire(fd: int) -> None:
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("could not lock the Bulwark hook state file") from None
                time.sleep(0.01)

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire(fd: int) -> None:
        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError("could not lock the Bulwark hook state file") from None
                time.sleep(0.005)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
