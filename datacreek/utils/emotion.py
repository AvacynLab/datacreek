from __future__ import annotations

"""Simple emotion detection utilities."""

from functools import lru_cache
from typing import Dict, Iterable


_EMOTION_LEXICON: Dict[str, Iterable[str]] = {
    "happy": {"happy", "joy", "joyful", "glad", "delighted", "cheerful", "good", "great"},
    "sad": {"sad", "unhappy", "sorrow", "depressed", "down", "gloomy"},
    "angry": {"angry", "mad", "furious", "irate"},
    "surprised": {"surprised", "astonished", "amazed"},
    "fear": {"afraid", "scared", "fear", "terrified"},
}


@lru_cache(maxsize=1024)
def detect_emotion(text: str) -> str:
    """Return a rough emotion label for ``text``.

    The function performs a simple keyword search over a small
    lexicon to assign one of the known emotions. If no keyword is
    found, ``"neutral"`` is returned.
    """
    txt = text.lower()
    for label, words in _EMOTION_LEXICON.items():
        for w in words:
            if w in txt:
                return label
    return "neutral"

