import importlib
import sys
import types
import pytest

import datacreek.generators as generators

@pytest.mark.parametrize(
    "name,mod",
    [
        ("QAGenerator", "qa_generator"),
        ("COTGenerator", "cot_generator"),
        ("VQAGenerator", "vqa_generator"),
        ("KGGenerator", "kg_generator"),
        ("ToolCallGenerator", "tool_generator"),
        ("ConversationGenerator", "conversation_generator"),
        ("MultiToolGenerator", "multi_tool_generator"),
        ("PrefPairGenerator", "pref_generator"),
        ("PrefListGenerator", "pref_generator"),
    ],
)
def test_getattr_imports(monkeypatch, name, mod):
    """Verify __getattr__ lazily imports the target class."""
    module = types.ModuleType(f"datacreek.generators.{mod}")
    Dummy = type(name, (), {})
    setattr(module, name, Dummy)
    monkeypatch.setitem(sys.modules, f"datacreek.generators.{mod}", module)
    cls = getattr(generators, name)
    assert cls is Dummy


def test_getattr_invalid():
    """Requesting an unknown name raises AttributeError."""
    with pytest.raises(AttributeError):
        generators.__getattr__("UnknownGenerator")


def test_all_exports():
    """Ensure selected generators are exported in __all__."""
    expected_names = [
        "QAGenerator",
        "COTGenerator",
        "VQAGenerator",
        "KGGenerator",
        "ToolCallGenerator",
        "ConversationGenerator",
        "MultiToolGenerator",
        "PrefPairGenerator",
        "PrefListGenerator",
    ]
    for expected in expected_names:
        assert expected in generators.__all__
