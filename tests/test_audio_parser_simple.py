import sys
import types
import pytest

from datacreek.parsers.audio_parser import AudioParser
from datacreek.parsers.base import BaseParser


def _fake_sr_module(return_value="text", exc=None):
    class Recognizer:
        def record(self, source):
            return b"audio"

        def recognize_sphinx(self, audio):
            if exc:
                raise exc
            return return_value

    class AudioFile:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    return types.SimpleNamespace(
        Recognizer=Recognizer,
        AudioFile=AudioFile,
        UnknownValueError=UnknownValueError,
        RequestError=RequestError,
    )


def test_base_parser_not_implemented():
    parser = BaseParser()
    with pytest.raises(NotImplementedError):
        parser.parse("file")
    with pytest.raises(RuntimeError):
        parser.save("text", "out")


def test_audio_parser_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "speech_recognition", None)
    parser = AudioParser()
    with pytest.raises(ImportError):
        parser.parse("sample.wav")


def test_audio_parser_success(monkeypatch, tmp_path):
    fake_sr = _fake_sr_module(return_value="ok")
    monkeypatch.setitem(sys.modules, "speech_recognition", fake_sr)
    parser = AudioParser()
    result = parser.parse(str(tmp_path / "a.wav"))
    assert result == "ok"


def test_audio_parser_unknown_value(monkeypatch, tmp_path):
    fake_sr = _fake_sr_module()
    def raise_unknown(self, audio):
        raise fake_sr.UnknownValueError()

    fake_sr.Recognizer.recognize_sphinx = raise_unknown
    monkeypatch.setitem(sys.modules, "speech_recognition", fake_sr)
    parser = AudioParser()
    assert parser.parse(str(tmp_path / "a.wav")) == ""


def test_audio_parser_request_error(monkeypatch, tmp_path):
    fake_sr = _fake_sr_module()

    def raise_request(self, audio):
        raise fake_sr.RequestError("fail")

    fake_sr.Recognizer.recognize_sphinx = raise_request
    monkeypatch.setitem(sys.modules, "speech_recognition", fake_sr)
    parser = AudioParser()
    with pytest.raises(RuntimeError):
        parser.parse(str(tmp_path / "a.wav"))
