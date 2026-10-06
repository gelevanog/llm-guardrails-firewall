"""HTML to text, keeping track of content a human reader would not see.

Emails and web pages hide instructions for the model in white-on-white text, `display:none` blocks,
zero-size fonts, off-screen positioning, HTML comments and image alt text. The model reads all of it;
the person who forwarded the email saw none of it. `extract_html` returns the visible text and the
hidden segments separately, so the untrusted-content guard can scan the hidden part with suspicion and
drop it from what the model receives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

_HTML_HINT = re.compile(r"<(?:html|body|div|span|p|table|td|tr|br|img|a|font|style|h[1-6]|ul|li)\b[^>]*>", re.I)
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"})
_BLOCK = frozenset({"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "hr"})
_SKIP = frozenset({"script", "style", "head", "title"})
_WHITE = r"(?:#fff(?:fff)?\b|white\b|rgba?\(\s*255\s*,\s*255\s*,\s*255)"
_HIDDEN_STYLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("display:none", re.compile(r"display\s*:\s*none", re.I)),
    ("visibility:hidden", re.compile(r"visibility\s*:\s*hidden", re.I)),
    ("zero font size", re.compile(r"font-size\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%)?\s*(?:;|$|!)", re.I)),
    ("opacity:0", re.compile(r"opacity\s*:\s*0(?:\.0+)?\s*(?:;|$|!)", re.I)),
    ("white text", re.compile(r"(?<![-\w])color\s*:\s*" + _WHITE, re.I)),
    ("off-screen", re.compile(r"(?:left|top|text-indent)\s*:\s*-\d{3,}", re.I)),
    ("zero height", re.compile(r"(?:max-)?height\s*:\s*0(?:px)?\s*;[^\"']*overflow\s*:\s*hidden", re.I)),
)


@dataclass(frozen=True)
class HiddenSegment:
    text: str
    reason: str


@dataclass
class HtmlExtraction:
    visible: str
    hidden: list[HiddenSegment] = field(default_factory=list)
    is_html: bool = True


def looks_like_html(text: str) -> bool:
    return len(_HTML_HINT.findall(text)) >= 1


def _hidden_reason(tag: str, attrs: dict[str, str]) -> str | None:
    if "hidden" in attrs:
        return "hidden attribute"
    style = attrs.get("style", "")
    for reason, pattern in _HIDDEN_STYLES:
        if pattern.search(style):
            return reason
    if tag == "font":
        if re.fullmatch(_WHITE, attrs.get("color", "").strip(), re.I):
            return "white text"
        if attrs.get("size", "").strip() in {"0", "-10"}:
            return "zero font size"
    return None


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible: list[str] = []
        self.hidden: list[HiddenSegment] = []
        self._stack: list[tuple[str, str | None]] = []  # (tag, hidden reason inherited or own)
        self._buffer: dict[int, list[str]] = {}  # stack depth of the hiding element -> collected text
        self._skip = 0

    def _current_reason(self) -> tuple[int, str] | None:
        for depth, (_, reason) in enumerate(self._stack):
            if reason:
                return depth, reason
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): (value or "") for name, value in attrs}
        if tag in _SKIP:
            self._skip += 1
            return
        if tag in _BLOCK:
            self._emit("\n")
        for attribute in ("alt", "title"):
            if values.get(attribute, "").strip():
                self.hidden.append(HiddenSegment(values[attribute].strip(), f"{attribute} attribute"))
        if tag in _VOID:
            return
        reason = _hidden_reason(tag, values)
        self._stack.append((tag, reason))
        if reason and self._current_reason() == (len(self._stack) - 1, reason):
            self._buffer[len(self._stack) - 1] = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK:
            self._emit("\n")
        values = {name.lower(): (value or "") for name, value in attrs}
        for attribute in ("alt", "title"):
            if values.get(attribute, "").strip():
                self.hidden.append(HiddenSegment(values[attribute].strip(), f"{attribute} attribute"))

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag in _BLOCK:
            self._emit("\n")
        for depth in range(len(self._stack) - 1, -1, -1):
            if self._stack[depth][0] == tag:
                for closed in range(len(self._stack) - 1, depth - 1, -1):
                    self._close(closed)
                del self._stack[depth:]
                return

    def _close(self, depth: int) -> None:
        if depth in self._buffer:
            text = " ".join("".join(self._buffer.pop(depth)).split())
            reason = self._stack[depth][1] or "hidden"
            if text:
                self.hidden.append(HiddenSegment(text, reason))

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._emit(data)

    def handle_comment(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.hidden.append(HiddenSegment(text, "HTML comment"))

    def _emit(self, text: str) -> None:
        hidden = self._current_reason()
        if hidden is None:
            self.visible.append(text)
        else:
            depth = hidden[0]
            self._buffer.setdefault(depth, []).append(text)

    def finish(self) -> None:
        for depth in range(len(self._stack) - 1, -1, -1):
            self._close(depth)
        self._stack.clear()


def extract_html(text: str) -> HtmlExtraction:
    """Visible text and hidden segments of an HTML document (plain text is returned unchanged)."""
    if not looks_like_html(text):
        return HtmlExtraction(visible=text, hidden=[], is_html=False)
    parser = _Extractor()
    parser.feed(text)
    parser.close()
    parser.finish()
    visible = "".join(parser.visible)
    visible = re.sub(r"[ \t\r\f\v]+", " ", visible)
    visible = re.sub(r" *\n[ \n]*", "\n", visible).strip()
    return HtmlExtraction(visible=visible, hidden=parser.hidden, is_html=True)
