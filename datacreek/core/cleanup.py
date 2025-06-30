from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .knowledge_graph import KnowledgeGraph
from datacreek.models.results import KGCleanupStats

if TYPE_CHECKING:  # pragma: no cover - avoid circular import at runtime
    from .dataset import DatasetBuilder


def cleanup_knowledge_graph(
    kg: KnowledgeGraph,
    *,
    dataset_builder: Optional[DatasetBuilder] = None,
    resolve_threshold: float = 0.8,
    resolve_aliases: dict[str, list[str]] | None = None,
    dedup_similarity: float = 1.0,
) -> KGCleanupStats:
    """Run standard cleanup operations on ``kg``.

    Parameters
    ----------
    dataset_builder:
        Optional :class:`DatasetBuilder` to log events on.
    resolve_threshold:
        Similarity used when merging entities.
    resolve_aliases:
        Mapping of canonical labels to aliases used during entity resolution.
    dedup_similarity:
        Similarity threshold for deduplicating chunk text.
    """
    if dataset_builder is not None:
        removed, cleaned = dataset_builder.cleanup_graph(
            resolve_threshold=resolve_threshold,
            resolve_aliases=resolve_aliases,
            dedup_similarity=dedup_similarity,
        )
    else:
        removed = kg.deduplicate_chunks(dedup_similarity)
        cleaned = kg.clean_chunk_texts()
        kg.resolve_entities(threshold=resolve_threshold, aliases=resolve_aliases)
    return KGCleanupStats(removed=removed, cleaned=cleaned)
