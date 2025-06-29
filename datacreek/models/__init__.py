__all__ = [
    "LLMClient",
    "QAPair",
    "COTExample",
    "QAGenerationResult",
    "COTGenerationResult",
    "CurationMetrics",
    "CurationResult",
    "ConversationResult",
    "PrefPairResult",
    "PrefListResult",
]


def __getattr__(name: str):
    """Lazily import model classes to avoid heavy dependencies."""

    if name == "LLMClient":
        from .llm_client import LLMClient as cls

        return cls
    if name == "QAPair":
        from .qa import QAPair as cls

        return cls
    if name == "COTExample":
        from .cot import COTExample as cls

        return cls
    if name == "QAGenerationResult":
        from .results import QAGenerationResult as cls

        return cls
    if name == "COTGenerationResult":
        from .results import COTGenerationResult as cls

        return cls
    if name == "CurationMetrics":
        from .results import CurationMetrics as cls

        return cls
    if name == "CurationResult":
        from .results import CurationResult as cls

        return cls
    if name == "ConversationResult":
        from .results import ConversationResult as cls

        return cls
    if name == "PrefPairResult":
        from .results import PrefPairResult as cls

        return cls
    if name == "PrefListResult":
        from .results import PrefListResult as cls

        return cls
    raise AttributeError(name)
