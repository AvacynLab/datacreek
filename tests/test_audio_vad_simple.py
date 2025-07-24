import types
import pytest

import datacreek.utils.audio_vad as audio_vad


def test_split_on_silence_basic(monkeypatch):
    pattern = [False, True, True, False, False, True, False]
    frame_ms = 10
    join_ms = 20
    class FakeVad:
        def __init__(self, mode=3):
            self.calls = -1
        def is_speech(self, frame, sample_rate):
            self.calls += 1
            return pattern[self.calls]
    monkeypatch.setattr(audio_vad, "webrtcvad", types.SimpleNamespace(Vad=FakeVad))
    sample_rate = 16000
    frame_len = sample_rate * frame_ms // 1000 * 2
    pcm = b"\x00" * frame_len * len(pattern)
    segments = audio_vad.split_on_silence(pcm, sample_rate, frame_ms=frame_ms, join_ms=join_ms)
    assert segments == [(0.01, 0.03), (0.05, 0.07)]


def test_split_on_silence_missing_dep(monkeypatch):
    monkeypatch.setattr(audio_vad, "webrtcvad", None)
    with pytest.raises(ImportError):
        audio_vad.split_on_silence(b"\x00" * 2, 8000)
