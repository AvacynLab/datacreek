import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.llm_client import LLMClient
from datacreek.utils import convert_to_conversation_format

from .base import BaseGenerator

logger = logging.getLogger(__name__)


class MultiToolGenerator(BaseGenerator):
    """Generate simple multi-tool conversations."""

    def __init__(
        self,
        client: LLMClient,
        config_path: Optional[Path] = None,
        kg: Optional[KnowledgeGraph] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(client, config_path, config_overrides)
        self.kg = kg

    def process_document(
        self,
        document_text: str,
        *,
        num_pairs: int = 25,
        verbose: bool = False,
        async_mode: bool = False,
    ) -> Dict[str, Any]:
        """Return multi-tool conversations generated from ``document_text``."""

        from .qa_generator import QAGenerator

        qa_gen = QAGenerator(self.client, self.config_path, kg=self.kg, config_overrides=None)

        if async_mode:
            result = asyncio.run(
                qa_gen.process_document_async(document_text, num_pairs=num_pairs, verbose=verbose)
            )
        else:
            result = qa_gen.process_document(document_text, num_pairs=num_pairs, verbose=verbose)

        conversations: List[Dict[str, Any]] = []
        for pair in result.qa_pairs:
            conv = convert_to_conversation_format([pair])[0]
            conv.insert(
                2,
                {
                    "role": "assistant",
                    "tool_call": {"name": "search", "arguments": {"query": pair.question}},
                },
            )
            conv.insert(
                3,
                {"role": "tool", "name": "search", "content": pair.answer},
            )
            conv.insert(
                4,
                {
                    "role": "assistant",
                    "tool_call": {"name": "calculator", "arguments": {"input": pair.answer}},
                },
            )
            conv.insert(
                5,
                {"role": "tool", "name": "calculator", "content": pair.answer},
            )
            conversations.append(
                {
                    "conversations": conv,
                    "chunk": pair.chunk,
                    "source": pair.source,
                }
            )

        return {"summary": result.summary, "conversations": conversations}
