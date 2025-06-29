import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.llm_client import LLMClient
from datacreek.models.results import PrefListResult, PrefPairResult

from .base import BaseGenerator

logger = logging.getLogger(__name__)


class PrefPairGenerator(BaseGenerator):
    """Generate simple pairwise preference data."""

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
        num_pairs: int = 10,
        verbose: bool = False,
        async_mode: bool = False,
    ) -> PrefPairResult:
        """Return pairwise preference data from ``document_text``."""

        from .qa_generator import QAGenerator

        qa_gen = QAGenerator(self.client, self.config_path, kg=self.kg, config_overrides=None)

        if async_mode:
            result = asyncio.run(
                qa_gen.process_document_async(
                    document_text, num_pairs=num_pairs * 2, verbose=verbose
                )
            )
        else:
            result = qa_gen.process_document(
                document_text, num_pairs=num_pairs * 2, verbose=verbose
            )

        pairs: List[Dict[str, Any]] = []
        qa_iter = iter(result.qa_pairs)
        for _ in range(num_pairs):
            try:
                a = next(qa_iter)
                b = next(qa_iter)
            except StopIteration:
                break
            pairs.append(
                {
                    "question": a.question,
                    "chosen": a.answer,
                    "rejected": b.answer,
                    "chunk": a.chunk,
                    "source": a.source,
                }
            )

        return PrefPairResult(summary=result.summary, pairs=pairs)

    async def process_document_async(
        self,
        document_text: str,
        *,
        num_pairs: int = 10,
        verbose: bool = False,
    ) -> PrefPairResult:
        """Asynchronous version of :meth:`process_document`."""

        from .qa_generator import QAGenerator

        qa_gen = QAGenerator(self.client, self.config_path, kg=self.kg, config_overrides=None)

        result = await qa_gen.process_document_async(
            document_text, num_pairs=num_pairs * 2, verbose=verbose
        )

        pairs: List[Dict[str, Any]] = []
        qa_iter = iter(result.qa_pairs)
        for _ in range(num_pairs):
            try:
                a = next(qa_iter)
                b = next(qa_iter)
            except StopIteration:
                break
            pairs.append(
                {
                    "question": a.question,
                    "chosen": a.answer,
                    "rejected": b.answer,
                    "chunk": a.chunk,
                    "source": a.source,
                }
            )

        return PrefPairResult(summary=result.summary, pairs=pairs)


class PrefListGenerator(BaseGenerator):
    """Generate simple listwise ranking data."""

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
        num_lists: int = 10,
        list_size: int = 3,
        verbose: bool = False,
        async_mode: bool = False,
    ) -> PrefListResult:
        """Return listwise preference data from ``document_text``."""

        from .qa_generator import QAGenerator

        qa_gen = QAGenerator(self.client, self.config_path, kg=self.kg, config_overrides=None)

        total = num_lists * list_size
        if async_mode:
            result = asyncio.run(
                qa_gen.process_document_async(document_text, num_pairs=total, verbose=verbose)
            )
        else:
            result = qa_gen.process_document(document_text, num_pairs=total, verbose=verbose)

        responses: List[Dict[str, Any]] = []
        qa_iter = iter(result.qa_pairs)
        for _ in range(num_lists):
            items = []
            for _ in range(list_size):
                try:
                    p = next(qa_iter)
                except StopIteration:
                    break
                items.append({"text": p.answer, "chunk": p.chunk, "source": p.source})
            if items:
                responses.append({"question": items[0]["text"], "answers": items})

        return PrefListResult(summary=result.summary, responses=responses)

    async def process_document_async(
        self,
        document_text: str,
        *,
        num_lists: int = 10,
        list_size: int = 3,
        verbose: bool = False,
    ) -> PrefListResult:
        """Asynchronous version of :meth:`process_document`."""

        from .qa_generator import QAGenerator

        qa_gen = QAGenerator(self.client, self.config_path, kg=self.kg, config_overrides=None)

        total = num_lists * list_size
        result = await qa_gen.process_document_async(
            document_text, num_pairs=total, verbose=verbose
        )

        responses: List[Dict[str, Any]] = []
        qa_iter = iter(result.qa_pairs)
        for _ in range(num_lists):
            items = []
            for _ in range(list_size):
                try:
                    p = next(qa_iter)
                except StopIteration:
                    break
                items.append({"text": p.answer, "chunk": p.chunk, "source": p.source})
            if items:
                responses.append({"question": items[0]["text"], "answers": items})

        return PrefListResult(summary=result.summary, responses=responses)
