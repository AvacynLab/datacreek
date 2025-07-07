from __future__ import annotations

"""Utilities for detecting speech or writing modality."""

from functools import lru_cache

_SPOKEN_MARKERS = {
    "um",
    "uh",
    "er",
    "erm",
    "you know",
    "like",
}


@lru_cache(maxsize=1024)
def detect_modality(text: str) -> str:
    """Return ``"spoken"`` or ``"written"`` based on simple heuristics."""
    txt = text.lower()
    for w in _SPOKEN_MARKERS:
        if w in txt:
            return "spoken"
    # very short sentences with little punctuation often come from speech
    punct = sum(txt.count(p) for p in ".!?")
    words = len(txt.split())
    if words and punct / words < 0.05:
        return "spoken"
    return "written"
