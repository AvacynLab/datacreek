import sys
import types
import builtins
import pytest

from datacreek.parsers.whisper_audio_parser import WhisperAudioParser, matmul_8bit
from datacreek.parsers.youtube_parser import YouTubeParser


class DummyModel:
    def __init__(self):
        self.transcribed = []

    def transcribe(self, fp):
        self.transcribed.append(fp)
        return {"text": "hi"}


def test_whisper_parser_batch(monkeypatch):
    monkeypatch.setitem(sys.modules, 'datacreek.utils.whisper_batch', types.SimpleNamespace(transcribe_audio_batch=lambda x: ['ok']))
    parser = WhisperAudioParser()
    assert parser.parse('a.wav') == 'ok'


def test_whisper_parser_whisper(monkeypatch):
    monkeypatch.setitem(sys.modules, 'datacreek.utils.whisper_batch', types.SimpleNamespace(transcribe_audio_batch=None))
    fake_model = DummyModel()
    monkeypatch.setitem(sys.modules, 'whisper', types.SimpleNamespace(load_model=lambda name: fake_model))
    class FakeTorch:
        def __init__(self):
            self.cuda = types.SimpleNamespace(is_available=lambda: False)
            self.matmul = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'torch', FakeTorch())
    monkeypatch.setattr('datacreek.parsers.whisper_audio_parser.matmul_8bit', None, raising=False)
    parser = WhisperAudioParser()
    assert parser.parse('a.wav') == 'hi'
    assert fake_model.transcribed == ['a.wav']


def test_youtube_parser(monkeypatch):
    class FakeYT:
        def __init__(self, url):
            self.video_id = 'abc'
            self.title = 'T'
            self.author = 'A'
            self.length = 1
    monkeypatch.setitem(sys.modules, 'pytubefix', types.SimpleNamespace(YouTube=FakeYT))
    monkeypatch.setitem(
        sys.modules,
        'youtube_transcript_api',
        types.SimpleNamespace(
            YouTubeTranscriptApi=types.SimpleNamespace(
                get_transcript=lambda vid: [{"text": "foo"}, {"text": "bar"}]
            )
        ),
    )
    parser = YouTubeParser()
    text = parser.parse('http://youtube.com/watch?v=abc')
    assert 'Title: T' in text
    assert 'foo' in text and 'bar' in text
