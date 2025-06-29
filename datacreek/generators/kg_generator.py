import logging
from pathlib import Path
from typing import Any, Dict, Optional

from datacreek.core.knowledge_graph import KnowledgeGraph
from datacreek.models.llm_client import LLMClient
from datacreek.utils.config import get_prompt

from .base import BaseGenerator

logger = logging.getLogger(__name__)


class KGGenerator(BaseGenerator):
    """Generate QA examples from a knowledge graph."""

    def process_graph(
        self,
        kg: KnowledgeGraph,
        num_pairs: int = 25,
        *,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Return QA pairs generated from ``kg``.

        This is a simplified placeholder implementation that serializes a small
        portion of the graph and uses the QA prompt.
        """
        nodes = list(kg.graph.nodes(data=True))[:5]
        text_parts = []
        for node, data in nodes:
            label = data.get("label") or node
            summary = data.get("summary") or data.get("text", "")
            text_parts.append(f"{label}: {summary}")
        prompt = get_prompt(self.config, "qa_generation")
        prompt_filled = prompt.format(num_pairs=num_pairs, summary="", text="\n".join(text_parts))
        messages = [{"role": "system", "content": prompt_filled}]
        temperature = self.generation_config.temperature
        max_tokens = self.generation_config.max_tokens
        if verbose:
            logger.info("Generating QA from knowledge graph with %d nodes", len(nodes))
        response = self.client.chat_completion(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        from datacreek.utils.llm_processing import parse_qa_pairs

        qa_pairs = parse_qa_pairs(response)
        return {"qa_pairs": [p.to_dict() for p in qa_pairs]}
