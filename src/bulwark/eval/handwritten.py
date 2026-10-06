"""The hand-written evaluation set: readable YAML source, compiled to JSONL.

Obfuscated items are stored as their plain text plus a `transform` (base64, hex, rot13, leetspeak, homoglyphs,
zero-width characters, tag characters, letter spacing, reversal, URL-encoding), applied deterministically at
build time, so the source stays reviewable and the compiled file is reproducible (CI rebuilds and compares).
"""

from __future__ import annotations

import base64
import codecs
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from bulwark.eval.datasets import Sample

_HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "i": "і", "x": "х", "y": "у"}
_LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


def _homoglyph(text: str) -> str:
    # Every other eligible letter, so words stay mixed-script (that is what a real attack does).
    out, flip = [], False
    for char in text:
        if char in _HOMOGLYPHS:
            flip = not flip
            out.append(_HOMOGLYPHS[char] if flip else char)
        else:
            out.append(char)
    return "".join(out)


def _zero_width(text: str) -> str:
    return "".join(c + ("​" if c.isalpha() and i % 2 else "") for i, c in enumerate(text))


TRANSFORMS: dict[str, Callable[[str], str]] = {
    "base64": lambda t: base64.b64encode(t.encode()).decode(),
    "hex": lambda t: t.encode().hex(),
    "rot13": lambda t: codecs.encode(t, "rot13"),
    "leet": lambda t: "".join(_LEET.get(c, c) for c in t),
    "homoglyph": _homoglyph,
    "zero_width": _zero_width,
    "tags": lambda t: "".join(chr(0xE0000 + ord(c)) for c in t),
    "spaced": lambda t: " ".join(t),
    "reversed": lambda t: t[::-1],
    "url": lambda t: quote(t),
}


def render(item: dict[str, Any]) -> str:
    """The item's final text: plain, or `text` encoded by `transform` and placed into `template` if given."""
    text = str(item["text"]).strip()
    transform = item.get("transform")
    if not transform:
        return text
    encoded = TRANSFORMS[transform](text)
    template = item.get("template")
    return str(template).strip().replace("{payload}", encoded) if template else encoded


def build_set(source: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[str] = set()
    for path in sorted(source.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        split = str(raw.get("split", "dev"))
        for item in raw.get("items", []):
            if item["id"] in seen:
                raise ValueError(f"duplicate id {item['id']} in {path.name}")
            seen.add(item["id"])
            label = item["label"]
            samples.append(
                Sample(
                    id=item["id"],
                    dataset="handwritten",
                    text=render(item),
                    label=1 if label == "attack" else 0,
                    context=item.get("context", "input"),
                    category=item["category"],
                    split=split,
                    lang=item.get("lang", "en"),
                )
            )
    return samples


def write_set(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")


def load_set(path: Path) -> list[Sample]:
    return [Sample.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_stats(samples: list[Sample]) -> dict[str, Any]:
    return {
        "items": len(samples),
        "attacks": sum(s.label for s in samples),
        "benign": sum(1 - s.label for s in samples),
        "by_category": dict(Counter(s.category for s in samples)),
        "by_split": dict(Counter(s.split for s in samples)),
        "by_context": dict(Counter(s.context for s in samples)),
        "languages": dict(Counter(s.lang for s in samples)),
    }
