from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@dataclass
class GenerationSettings:
    """Configuration options controlling LLM generation."""

    temperature: float = 0.7
    top_p: float = 0.95
    chunk_size: int = 4000
    overlap: int = 200
    chunk_method: str = "sliding"
    similarity_drop: float = 0.3
    retrieval_top_k: int = 3
    max_tokens: int = 4096
    num_pairs: int = 25
    num_cot_examples: int = 5
    num_cot_enhance_examples: Optional[int] = None
    batch_size: int = 32
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = field(default=None)
    summary_temperature: float = 0.1
    summary_max_tokens: int = 1024

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationSettings":
        """Create ``GenerationSettings`` from a raw dictionary."""
        defaults = cls()
        return cls(**{k: data.get(k, getattr(defaults, k)) for k in cls.__dataclass_fields__})

    def update(self, overrides: Dict[str, Any]) -> None:
        """Update fields from a dictionary of overrides."""
        for key, value in overrides.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


class GenerationSettingsModel(BaseModel):
    """Pydantic model for validating generation settings."""

    temperature: float = 0.7
    top_p: float = 0.95
    chunk_size: int = 4000
    overlap: int = 200
    chunk_method: str = "sliding"
    similarity_drop: float = 0.3
    retrieval_top_k: int = 3
    max_tokens: int = 4096
    num_pairs: int = 25
    num_cot_examples: int = 5
    num_cot_enhance_examples: Optional[int] = None
    batch_size: int = 32
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    summary_temperature: float = 0.1
    summary_max_tokens: int = 1024

    def to_settings(self) -> GenerationSettings:
        """Convert to :class:`GenerationSettings`."""
        return GenerationSettings.from_dict(self.model_dump())
