"""Application context utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from datacreek.utils.config import DEFAULT_CONFIG_PATH


class AppContext:
    """Context manager for global application state."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = {}
