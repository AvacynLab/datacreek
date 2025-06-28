from dataclasses import dataclass
from typing import Any, Dict, List

from .cot import COTExample
from .qa import QAPair


@dataclass
class QAGenerationResult:
    """Output of QA generation."""

    summary: str
    qa_pairs: List[QAPair]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "qa_pairs": [p.to_dict() for p in self.qa_pairs],
        }


@dataclass
class CurationMetrics:
    """Statistics about QA pair curation."""

    total: int
    filtered: int
    retention_rate: float
    avg_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "filtered": self.filtered,
            "retention_rate": self.retention_rate,
            "avg_score": self.avg_score,
        }


@dataclass
class CurationResult:
    """Result of QA curation."""

    summary: str
    qa_pairs: List[QAPair]
    conversations: List[List[Dict[str, str]]]
    metrics: CurationMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "qa_pairs": [p.to_dict() for p in self.qa_pairs],
            "conversations": self.conversations,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class COTGenerationResult:
    """Output of chain-of-thought generation."""

    summary: str
    cot_examples: List[COTExample]
    conversations: List[List[Dict[str, str]]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "cot_examples": [ex.to_dict() for ex in self.cot_examples],
            "conversations": self.conversations,
        }
