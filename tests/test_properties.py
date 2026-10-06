"""Property-based tests (Hypothesis) for the normalizers, decoders, HTML extraction and the stream guard."""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from bulwark.config import PACKAGED_POLICIES
from bulwark.firewall import Firewall
from bulwark.guards.heuristics import scan
from bulwark.guards.spotlight import spotlight, strip_spotlight
from bulwark.normalize import decoded_views, extract_html, find_hidden_payloads, normalize, strip_invisible
from bulwark.policy import PolicySet, Spotlight
from bulwark.stream import StreamGuard

ANY_TEXT = st.text(alphabet=st.characters(codec="utf-8"), max_size=300)
INVISIBLE = st.sampled_from(["​", "‌", "⁠", "﻿", "­", "‮", chr(0xE0041), chr(0xFE0F)])


@given(ANY_TEXT)
def test_normalize_index_is_valid(text: str) -> None:
    view = normalize(text)
    assert len(view.index) == len(view.text)
    assert all(0 <= i < len(text) for i in view.index)
    assert list(view.index) == sorted(view.index)  # order is preserved
    if view.text:
        start, end = view.to_original(0, len(view.text))
        assert 0 <= start <= end <= len(text)


@given(st.lists(st.tuples(st.sampled_from("ignore previous instructions now"), INVISIBLE), max_size=40))
def test_invisible_characters_never_survive_normalization(pairs: list[tuple[str, str]]) -> None:
    text = "".join(a + b for a, b in pairs)
    clean = normalize(text, leet=False, collapse_spacing=False).text
    assert clean == strip_invisible(text).replace("­", "")
    assert not any(c in clean for c in "​‌⁠﻿‮")


@given(st.text(alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), min_size=4, max_size=80))
def test_tag_payloads_round_trip(secret: str) -> None:
    hidden = "".join(chr(0xE0000 + ord(c)) for c in secret)
    payloads = find_hidden_payloads("visible " + hidden + " text")
    assert [p.text for p in payloads] == [secret]


@given(ANY_TEXT)
@settings(max_examples=150)
def test_decoders_and_scanners_never_crash(text: str) -> None:
    decoded_views(text)
    extract_html(text)
    result = scan(text, "untrusted")
    assert 0.0 <= result.score <= 1.0
    for finding in result.findings:
        if finding.start is not None and finding.end is not None:
            assert 0 <= finding.start <= finding.end <= len(text)


@given(st.text(alphabet=st.characters(codec="utf-8", exclude_characters="<>"), max_size=200))
def test_html_extraction_of_plain_text_is_identity(text: str) -> None:
    assert extract_html(text).visible == text


@given(ANY_TEXT, st.sampled_from([Spotlight.DELIMIT, Spotlight.DATAMARK]))
def test_spotlight_display_round_trip(text: str, mode: Spotlight) -> None:
    wrapped = spotlight(text, mode, "tool", block_id="ab12")
    restored = strip_spotlight(wrapped)
    if mode is Spotlight.DELIMIT and "<<" not in text and "ˆ" not in text:
        assert restored == text
    if mode is Spotlight.DATAMARK and "<<" not in text and "ˆ" not in text:
        assert restored == " ".join(text.split())


SAFE = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E, exclude_characters="[]()<>!:/"), max_size=200
)
FIREWALL = Firewall(PolicySet.from_dir(PACKAGED_POLICIES, "default"))


@given(SAFE, st.lists(st.integers(min_value=1, max_value=17), min_size=1, max_size=40))
@settings(max_examples=80, deadline=None)
def test_stream_guard_releases_safe_text_unchanged(text: str, sizes: list[int]) -> None:
    async def run() -> str:
        guard = StreamGuard(FIREWALL, FIREWALL.policy("default"))
        out, position, index = "", 0, 0
        while position < len(text):
            size = sizes[index % len(sizes)]
            out += (await guard.push(text[position : position + size])).text
            position += size
            index += 1
        return out + (await guard.finish()).text

    assert asyncio.run(run()) == text
