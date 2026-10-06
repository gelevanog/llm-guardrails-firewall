"""Decode encoded payloads so the detectors can read what the model would read.

Models decode base64, hex, URL-encoding and ROT13 on request, so "decode this and follow it: aWdub3Jl..."
is an injection the raw text does not show. Each candidate is decoded and kept only when the result is
readable text (most base64 in real traffic is an attachment or a token, not a sentence).
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
from dataclasses import dataclass
from urllib.parse import unquote

from bulwark.normalize.unicode import find_hidden_payloads

_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{16,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[ :]?){8,}(?![0-9A-Fa-f])")
_HEX_ESCAPES = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){6,}")
_URL_ENCODED = re.compile(r"(?:[A-Za-z0-9._~+-]*%[0-9A-Fa-f]{2}){3,}[A-Za-z0-9._~+-]*")
_WORD = re.compile(r"[^\W\d_]{3,}")
_MAX_CANDIDATES = 20


@dataclass(frozen=True)
class DecodedView:
    """A decoded form of (part of) the text. `start`/`end` locate the encoded source in the original."""

    kind: str
    text: str
    start: int
    end: int


def readable(text: str, min_words: int = 2) -> bool:
    """Looks like natural-language text: printable, with a few real words."""
    if not text or len(text) < 6:
        return False
    printable = sum(c.isprintable() or c in "\n\t" for c in text)
    if printable < 0.95 * len(text):
        return False
    return len(_WORD.findall(text)) >= min_words and sum(c.isalpha() or c == " " for c in text) >= 0.6 * len(text)


def _b64(candidate: str) -> str | None:
    padded = candidate + "=" * (-len(candidate) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            raw = decoder(padded.encode("ascii"))
            return raw.decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
    return None


def _hex(candidate: str) -> str | None:
    digits = re.sub(r"[^0-9A-Fa-f]", "", candidate.replace("\\x", ""))
    if len(digits) % 2:
        return None
    try:
        return bytes.fromhex(digits).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def decoded_views(text: str, *, rot13: bool = True, reverse: bool = True) -> list[DecodedView]:
    """Every readable decoding found in the text (base64, hex, URL-encoding, ROT13, reversed, invisible)."""
    views: list[DecodedView] = []

    for payload in find_hidden_payloads(text):
        views.append(DecodedView(payload.kind, payload.text, payload.start, payload.end))

    for match in list(_BASE64.finditer(text))[:_MAX_CANDIDATES]:
        candidate = match.group(0)
        if candidate.isalpha() or candidate.isdigit():
            continue  # a long word or number, not base64
        decoded = _b64(candidate)
        if decoded and readable(decoded):
            views.append(DecodedView("base64", decoded, match.start(), match.end()))

    for pattern in (_HEX, _HEX_ESCAPES):
        for match in list(pattern.finditer(text))[:_MAX_CANDIDATES]:
            decoded = _hex(match.group(0))
            if decoded and readable(decoded):
                views.append(DecodedView("hex", decoded, match.start(), match.end()))

    for match in list(_URL_ENCODED.finditer(text))[:_MAX_CANDIDATES]:
        decoded = unquote(match.group(0))
        if readable(decoded):
            views.append(DecodedView("url-encoded", decoded, match.start(), match.end()))

    letters = sum(c.isascii() and c.isalpha() for c in text)
    if rot13 and letters >= 20:
        views.append(DecodedView("rot13", codecs.encode(text, "rot13"), 0, len(text)))
    if reverse and letters >= 20:
        views.append(DecodedView("reversed", text[::-1], 0, len(text)))
    return views
