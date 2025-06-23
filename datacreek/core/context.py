"""Application context utilities."""
from pathlib import Path
from typing import Optional, Dict, Any
import os

from datacreek.utils.config import DEFAULT_CONFIG_PATH

class AppContext:
    """Context manager for global application state."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = {}
        self._ensure_data_dirs()

    def _ensure_data_dirs(self) -> None:
        dirs = [
            "data/pdf",
            "data/html",
            "data/youtube",
            "data/docx",
            "data/ppt",
            "data/txt",
            "data/output",
            "data/generated",
            "data/cleaned",
            "data/final",
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

