import importlib
import sys
import types

import pytest

import datacreek.models as models


def make_stub(name: str, cls_name: str):
    module = types.ModuleType(name)
    module_cls = type(cls_name, (), {})
    setattr(module, cls_name, module_cls)
    return module, module_cls


def setup_stubs(monkeypatch):
    stubs = {}
    modules = [
        ("datacreek.models.llm_client", "LLMClient"),
        ("datacreek.models.llm_service", "LLMService"),
        ("datacreek.models.qa", "QAPair"),
        ("datacreek.models.cot", "COTExample"),
    ]
    results_module = types.ModuleType("datacreek.models.results")
    for name in [
        "QAGenerationResult",
        "COTGenerationResult",
        "CurationMetrics",
        "CurationResult",
        "ConversationResult",
        "PrefPairResult",
        "PrefListResult",
    ]:
        cls = type(name, (), {})
        setattr(results_module, name, cls)
        stubs[name] = cls
    extra = [
        ("datacreek.models.export_format", "ExportFormat"),
        ("datacreek.models.stage", "DatasetStage"),
        ("datacreek.models.task_status", "TaskStatus"),
    ]
    for full, cls_name in modules + extra:
        mod, cls = make_stub(full, cls_name)
        stubs[cls_name] = cls
        monkeypatch.setitem(sys.modules, full, mod)
    monkeypatch.setitem(sys.modules, "datacreek.models.results", results_module)
    return stubs


@pytest.fixture(autouse=True)
def reload_models():
    importlib.reload(models)
    yield
    importlib.reload(models)


def test_lazy_imports(monkeypatch):
    stubs = setup_stubs(monkeypatch)
    for name, cls in stubs.items():
        assert getattr(models, name) is cls


def test_unknown_attribute():
    with pytest.raises(AttributeError):
        models.__getattr__("Unknown")
