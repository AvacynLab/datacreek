import types

import datacreek.backend.array_api as aapi


def test_cupy_unavailable(monkeypatch):
    def fake_import(name):
        if name == "cupy":
            raise ImportError
        import importlib
        return importlib.import_module(name)
    monkeypatch.setattr(aapi, "import_module", fake_import)
    assert not aapi._cupy_available()
    assert aapi.get_xp().__name__ == "numpy"


def test_cupy_available_with_gpu(monkeypatch):
    fake_cupy = types.SimpleNamespace(
        __name__="cupy",
        cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)),
    )
    monkeypatch.setattr(aapi, "import_module", lambda name: fake_cupy if name == "cupy" else __import__(name))
    assert aapi._cupy_available()
    assert aapi.get_xp().__name__ == "cupy"


def test_get_xp_from_array(monkeypatch):
    class DummyArray:
        __module__ = "cupy.core.core"
    fake_cupy = types.SimpleNamespace(__name__="cupy")
    monkeypatch.setattr(aapi, "import_module", lambda name: fake_cupy if name == "cupy" else __import__(name))
    assert aapi.get_xp(DummyArray()).__name__ == "cupy"
