from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class QAPair:
    """Representation of a question-answer pair."""

    question: str
    answer: str
    rating: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {"question": self.question, "answer": self.answer}
        if self.rating is not None:
            data["rating"] = self.rating
        return data
