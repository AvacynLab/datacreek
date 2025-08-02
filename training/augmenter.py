"""Active-learning augmentation for high-loss samples.

This module identifies samples with high training loss and generates
paraphrased variants using a hypergraph of synonyms. The augmented
samples are re-inserted into the training pool every ``interval`` epochs
so that the model focuses on difficult examples.
"""

from __future__ import annotations

import math
from typing import List, Mapping, Sequence

__all__ = ["ActiveLearningAugmenter"]


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Compute the percentile of a sequence of numbers.

    A simple linear interpolation implementation is used instead of
    relying on NumPy so the function has no external dependencies.
    """

    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * percentile / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _generate_variants(
    text: str, synonym_map: Mapping[str, Sequence[str]], k: int
) -> List[str]:
    """Create ``k`` paraphrased variants by substituting synonyms.

    Parameters
    ----------
    text:
        The original sample.
    synonym_map:
        Mapping from token to a list of synonyms. The first synonym is
        used for deterministic behaviour.
    k:
        Number of variants to generate.
    """

    tokens = text.split()
    variants: List[str] = []
    for i in range(k):
        new_tokens = []
        for token in tokens:
            synonyms = synonym_map.get(token)
            if synonyms:
                new_tokens.append(synonyms[i % len(synonyms)])
            else:
                new_tokens.append(token)
        variants.append(" ".join(new_tokens))
    return variants


class ActiveLearningAugmenter:
    """Augment difficult samples at a fixed epoch interval.

    Parameters
    ----------
    synonym_map:
        Mapping of tokens to their synonyms extracted from the
        hypergraph.
    k:
        Number of variants to create for each selected sample.
    percentile:
        Percentile threshold above which a sample is considered
        difficult. Defaults to 95 (p95).
    interval:
        Augmentation interval in epochs. Augmentation occurs only when
        ``epoch % interval == 0``.
    """

    def __init__(
        self,
        synonym_map: Mapping[str, Sequence[str]],
        *,
        k: int = 1,
        percentile: float = 95.0,
        interval: int = 2,
    ) -> None:
        self.synonym_map = synonym_map
        self.k = k
        self.percentile = percentile
        self.interval = interval

    def augment(
        self,
        samples: Sequence[str],
        losses: Sequence[float],
        epoch: int,
    ) -> List[str]:
        """Return the dataset with augmented variants if at the right epoch.

        Parameters
        ----------
        samples:
            Original training samples.
        losses:
            Loss value for each sample.
        epoch:
            Current epoch number.
        """

        if len(samples) != len(losses):
            raise ValueError("samples and losses must have the same length")
        if epoch % self.interval != 0:
            return list(samples)

        threshold = _percentile(losses, self.percentile)
        augmented = list(samples)
        for sample, loss in zip(samples, losses):
            if loss > threshold:
                augmented.extend(_generate_variants(sample, self.synonym_map, self.k))
        return augmented
