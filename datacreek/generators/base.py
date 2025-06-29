import logging
from pathlib import Path
from typing import Any, Dict, Optional

from datacreek.models.llm_client import LLMClient
from datacreek.utils.config import get_generation_config, load_config, merge_configs

logger = logging.getLogger(__name__)


class BaseGenerator:
    """Common initializer for dataset generators."""

    def __init__(
        self,
        client: LLMClient,
        config_path: Optional[Path] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client = client
        if config_path or config_overrides:
            from datacreek.utils.config import load_config_with_overrides

            self.config = load_config_with_overrides(
                str(config_path) if config_path else None, config_overrides
            )
        else:
            self.config = client.config
        self.generation_config = get_generation_config(self.config)

    def process_document(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - base stub
        raise NotImplementedError
