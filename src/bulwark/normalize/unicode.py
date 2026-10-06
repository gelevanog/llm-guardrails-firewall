"""Unicode normalization that undoes common obfuscation while keeping a map back to the original text.

Attackers hide instructions from humans and from naive filters with:
  * zero-width and bidi control characters inside words ("ig​nore"),
  * invisible Unicode tag characters (U+E0020-E007E) that spell ASCII text a model can still read
    ("ASCII smuggling"), and variation selectors that encode bytes,
  * homoglyphs (Cyrillic "о" for Latin "o", Greek "Α" for "A"), fullwidth and mathematical letters,
  * leetspeak ("1gn0r3") and letter spacing ("i g n o r e").

`normalize()` produces one canonical view for the detectors. Every character of the view remembers the
index of the original character it came from, so a match can be reported (and quarantined) at its
real position.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Zero-width, joiners, word joiner, invisible math operators, BOM, Mongolian vowel separator, soft hyphen,
# bidi embedding/override/isolate controls.
_INVISIBLE = frozenset(
    [
        *(0x00AD, 0x180E, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF),
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    ]
)
TAG_RANGE = range(0xE0000, 0xE0080)
_VARIATION_SELECTORS = (range(0xFE00, 0xFE10), range(0xE0100, 0xE01F0))

# Confusable letters that NFKC does not fold (Cyrillic, Greek, a few Latin look-alikes) -> ASCII.
_HOMOGLYPHS = {
    # Cyrillic lowercase
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p", "с": "c",
    "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ї": "i", "ј": "j", "ԁ": "d", "ԛ": "q", "ԝ": "w",
    "һ": "h", "ӏ": "l", "ɡ": "g",
    # Cyrillic uppercase
    "А": "A", "В": "B", "Е": "E", "Ё": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ї": "I", "Ј": "J", "Ԁ": "D", "Ԛ": "Q", "Ԝ": "W",
    "Һ": "H", "Ӏ": "I",
    # Greek
    "α": "a", "β": "b", "γ": "y", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
    "υ": "u", "χ": "x", "ω": "w", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Latin look-alikes
    "ı": "i", "ȷ": "j", "ł": "l", "ø": "o", "ß": "ss", "ꞵ": "b", "ɑ": "a", "ʀ": "r", "ᴅ": "d", "ᴇ": "e",
}  # fmt: skip
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i", "|": "l"}
# A token that mixes letters with leet symbols: "1gn0r3", "pr3v10us", "@ll". Pure numbers are left alone.
_LEET_TOKEN = re.compile(r"(?=[\w@$!|]*[A-Za-z])(?=[\w@$!|]*[0134578@$!|])[\w@$!|]{3,}")
# "i g n o r e", "i.g.n.o.r.e", "i-g-n-o-r-e": four or more single letters with one separator between them.
_WORD_RUN = re.compile(r"[^\W\d_]+")
# Emails and URLs keep their symbols ("@" in an address is not leetspeak).
_PROTECTED = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|https?://\S+|www\.\S+")
_SPACED = re.compile(r"(?<![\w])(?:[A-Za-z][ .\-_*]){3,}[A-Za-z](?![\w])")


def is_invisible(char: str) -> bool:
    code = ord(char)
    return (
        code in _INVISIBLE
        or code in TAG_RANGE
        or any(code in block for block in _VARIATION_SELECTORS)
        or unicodedata.category(char) == "Cf"
    )


@dataclass(frozen=True)
class NormalizedText:
    """A view of the text plus, for every character, the index of the original character it came from."""

    text: str
    index: tuple[int, ...]
    original_length: int

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Map a [start, end) span of the view back to the original text."""
        if not self.index:
            return 0, 0
        start = min(max(start, 0), len(self.index) - 1)
        end = min(max(end, start + 1), len(self.index))
        return self.index[start], min(self.index[end - 1] + 1, self.original_length)


@dataclass(frozen=True)
class HiddenPayload:
    """Text encoded in invisible characters (tag characters or variation selectors)."""

    kind: str
    text: str
    start: int
    end: int


def _fold_char(char: str) -> str:
    if is_invisible(char):
        return ""
    folded = unicodedata.normalize("NFKC", char)
    # Stray combining marks ("Zalgo" text) after composition carry no meaning for the detectors.
    return "".join(c for c in folded if unicodedata.category(c) != "Mn")


def _is_latin(char: str) -> bool:
    return char.isascii() and char.isalpha()


def _fold_homoglyphs(view: str) -> str:
    """Fold look-alike letters only inside mixed-script words ("ignоre" with a Cyrillic о).

    Words written entirely in Cyrillic or Greek are real words in those languages and stay untouched.
    """
    pieces = list(view)
    for match in _WORD_RUN.finditer(view):
        word = match.group(0)
        if any(_is_latin(c) for c in word) and any(c in _HOMOGLYPHS for c in word):
            for offset, char in enumerate(word):
                folded = _HOMOGLYPHS.get(char)
                if folded is not None and len(folded) == 1:
                    pieces[match.start() + offset] = folded
    return "".join(pieces)


def normalize(text: str, *, leet: bool = True, collapse_spacing: bool = True) -> NormalizedText:
    """Strip invisible characters, apply NFKC and homoglyph folding, undo leetspeak and letter spacing."""
    chars: list[str] = []
    index: list[int] = []
    for position, char in enumerate(text):
        for piece in _fold_char(char):
            chars.append(piece)
            index.append(position)
    view = _fold_homoglyphs("".join(chars))

    if leet:
        pieces = list(view)
        protected = [(m.start(), m.end()) for m in _PROTECTED.finditer(view)]
        for match in _LEET_TOKEN.finditer(view):
            if any(start < match.end() and match.start() < end for start, end in protected):
                continue
            token = match.group(0)
            letters = sum(c.isalpha() for c in token)
            if letters < 2 or letters < len(token) / 3:
                continue  # "A1", "B2B", serials like "X5000" stay as they are
            for offset, char in enumerate(token):
                # Symbols only count inside a word ("@ll", "pa$$word"), so "Stop!" stays "Stop!".
                if char.isdigit() or 0 < offset < len(token) - 1:
                    pieces[match.start() + offset] = _LEET.get(char, char)
        view = "".join(pieces)

    if collapse_spacing:
        keep = [True] * len(view)
        for match in _SPACED.finditer(view):
            separators = [c for c in match.group(0) if not c.isalpha()]
            # "i.g.n.o.r.e a.l.l": the most frequent separator joins letters, any other one separates words.
            joiner = max(set(separators), key=separators.count)
            for offset in range(match.start(), match.end()):
                if view[offset] == joiner:
                    keep[offset] = False
        view = "".join(c for c, k in zip(view, keep, strict=True) if k)
        index = [i for i, k in zip(index, keep, strict=True) if k]

    return NormalizedText(view, tuple(index), len(text))


def invisible_counts(text: str) -> dict[str, int]:
    """How many invisible characters of each kind the text contains."""
    counts: dict[str, int] = {}
    for char in text:
        code = ord(char)
        if code in TAG_RANGE:
            kind = "tag"
        elif any(code in block for block in _VARIATION_SELECTORS):
            kind = "variation_selector"
        elif 0x202A <= code <= 0x202E or 0x2066 <= code <= 0x2069:
            kind = "bidi"
        elif is_invisible(char):
            kind = "zero_width"
        else:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def find_hidden_payloads(text: str, min_chars: int = 4) -> list[HiddenPayload]:
    """Decode runs of tag characters (ASCII smuggling) and variation-selector byte encodings."""
    payloads: list[HiddenPayload] = []
    position = 0
    while position < len(text):
        code = ord(text[position])
        if code in TAG_RANGE:
            start = position
            decoded = []
            while position < len(text) and ord(text[position]) in TAG_RANGE:
                inner = ord(text[position]) - 0xE0000
                if 0x20 <= inner <= 0x7E:
                    decoded.append(chr(inner))
                position += 1
            if len(decoded) >= min_chars:
                payloads.append(HiddenPayload("tag-chars", "".join(decoded), start, position))
            continue
        if any(code in block for block in _VARIATION_SELECTORS):
            start = position
            data = bytearray()
            while position < len(text):
                code = ord(text[position])
                if 0xFE00 <= code <= 0xFE0F:
                    data.append(code - 0xFE00)
                elif 0xE0100 <= code <= 0xE01EF:
                    data.append(code - 0xE0100 + 16)
                else:
                    break
                position += 1
            decoded_text = data.decode("utf-8", errors="ignore")
            if len(decoded_text) >= min_chars and sum(c.isprintable() for c in decoded_text) >= 0.9 * len(decoded_text):
                payloads.append(HiddenPayload("variation-selectors", decoded_text, start, position))
            continue
        position += 1
    return payloads


def strip_invisible(text: str) -> str:
    """Remove every invisible character (used when sanitizing untrusted content)."""
    return "".join(char for char in text if not is_invisible(char))
