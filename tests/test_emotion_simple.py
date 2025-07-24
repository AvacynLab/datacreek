import importlib

from datacreek.utils.emotion import detect_emotion


def test_detect_emotion_happy():
    detect_emotion.cache_clear()
    assert detect_emotion("I am very happy today!") == "happy"


def test_detect_emotion_neutral():
    detect_emotion.cache_clear()
    assert detect_emotion("Just words without emotion") == "neutral"


def test_detect_emotion_custom_lexicon():
    detect_emotion.cache_clear()
    custom = {"excited": {"excited", "thrilled"}}
    # bypass lru_cache hash requirement by calling the wrapped function
    assert detect_emotion.__wrapped__("so thrilled", custom) == "excited"
