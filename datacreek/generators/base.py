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
        if config_path:
            base_cfg = load_config(config_path)
        else:
            base_cfg = client.config
        if config_overrides:
            base_cfg = merge_configs(base_cfg, config_overrides)
        self.config = base_cfg
        self.generation_config = get_generation_config(self.config)

    def process_document(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - base stub
        raise NotImplementedError
