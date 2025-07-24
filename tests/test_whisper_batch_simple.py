import types
import sys
import logging
import datacreek.utils.whisper_batch as wb


def test_transcribe_no_dep(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(wb, "Whisper", None)
    wb._get_model.cache_clear()
    assert wb.transcribe_audio_batch(["a.wav"]) == [""]
    assert "whispercpp not available" in caplog.text


def test_transcribe_cpu(monkeypatch):
    calls = []

    class DummyModel:
        def transcribe(self, path, max_length):
            calls.append((path, max_length))
            return f"ok:{path}"

    dummy = DummyModel()

    def fake_get(model, fp16=True, device=None, int8=False):
        assert device == "cpu"
        assert int8
        return dummy

    fake_get.cache_clear = lambda: None
    monkeypatch.setattr(wb, "_get_model", fake_get)
    metrics = {}

    def record(name, value, labels):
        metrics[name] = (value, labels)

    monkeypatch.setitem(sys.modules, "datacreek.analysis.monitoring", types.SimpleNamespace(update_metric=record, whisper_fallback_total=None))
    monkeypatch.setattr(wb, "torch", None)
    times = [0.0, 1.0]
    monkeypatch.setattr(wb.time, "perf_counter", lambda: times.pop(0))

    out = wb.transcribe_audio_batch(["x.wav", "y.wav"], device="cpu", batch_size=2)
    assert out == ["ok:x.wav", "ok:y.wav"]
    assert calls == [("x.wav", 30), ("y.wav", 30)]
    assert metrics["whisper_xrt"][1]["device"] == "cpu"


def test_transcribe_fallback(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.calls = 0
        def transcribe(self, path, max_length):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Out of memory")
            return "done"

    model = DummyModel()

    def fake_get(*a, **k):
        return model

    fake_get.cache_clear = lambda: None
    monkeypatch.setattr(wb, "_get_model", fake_get)
    bumps = []
    fallback = types.SimpleNamespace(inc=lambda: bumps.append(True))
    metrics = []

    def record(name, value, labels):
        metrics.append((name, labels["device"]))

    monkeypatch.setitem(sys.modules, "datacreek.analysis.monitoring", types.SimpleNamespace(update_metric=record, whisper_fallback_total=fallback))
    monkeypatch.setattr(wb, "torch", types.SimpleNamespace(matmul=None))
    times = [0.0, 1.0]
    monkeypatch.setattr(wb.time, "perf_counter", lambda: times.pop(0))

    out = wb.transcribe_audio_batch(["f.wav"], device="cuda", batch_size=1)
    assert out == ["done"]
    assert bumps
    assert metrics[-1] == ("whisper_xrt", "cpu")
