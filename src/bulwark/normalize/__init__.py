"""Normalization and decoding: show the detectors what the model will actually read."""

from bulwark.normalize.decode import DecodedView, decoded_views, readable
from bulwark.normalize.html import HiddenSegment, HtmlExtraction, extract_html, looks_like_html
from bulwark.normalize.unicode import (
    HiddenPayload,
    NormalizedText,
    find_hidden_payloads,
    invisible_counts,
    normalize,
    strip_invisible,
)

__all__ = [
    "DecodedView",
    "HiddenPayload",
    "HiddenSegment",
    "HtmlExtraction",
    "NormalizedText",
    "decoded_views",
    "extract_html",
    "find_hidden_payloads",
    "invisible_counts",
    "looks_like_html",
    "normalize",
    "readable",
    "strip_invisible",
]
