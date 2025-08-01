"""Quality metrics used for ingestion validation.

This module provides helper functions to compute blur, entropy and
signal-to-noise ratio metrics. They implement the formulas detailed in
``AGENTS.md`` and are light on dependencies so they can run in unit tests.
"""

from __future__ import annotations

import math
import wave
from collections import Counter
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "blur_score",
    "text_entropy",
    "audio_snr",
    "image_dimensions",
    "audio_metrics",
]


def blur_score(path: str) -> float:
    r"""Return a blur metric for the image at ``path``.

    The score is derived from the variance of the Laplacian
    :math:`\mathrm{Blur} = \tfrac{1}{|I|}\sum (\nabla^2 I)^2` and mapped
    to ``(0, 1]`` such that lower values indicate sharper images.
    """

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=float)
    img.close()

    lap = (
        -4 * arr
        + np.roll(arr, 1, 0)
        + np.roll(arr, -1, 0)
        + np.roll(arr, 1, 1)
        + np.roll(arr, -1, 1)
    )
    var = float((lap**2).mean())
    return 1.0 / (1.0 + var)


def text_entropy(text: str) -> float:
    """Return Shannon entropy of ``text`` in bits per character."""

    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))


def audio_snr(pcm: bytes) -> float:
    """Approximate SNR of 16-bit PCM audio in decibels.

    The noise power is estimated from adjacent-sample differences and
    the result expressed as ``20·log10(signal_rms / noise_rms)``.
    """

    arr = np.frombuffer(pcm, dtype=np.int16).astype(float)
    if arr.size < 2:
        return float("inf")
    signal_rms = float(np.sqrt(np.mean(arr**2)))
    noise_rms = float(np.sqrt(np.mean(np.diff(arr) ** 2)) / math.sqrt(2))
    if noise_rms == 0:
        return float("inf")
    return 20.0 * math.log10(signal_rms / noise_rms)


def image_dimensions(path: str) -> Tuple[int, int]:
    """Return ``(width, height)`` for ``path``."""

    img = Image.open(path)
    size = img.size
    img.close()
    return size


def audio_metrics(path: str) -> Tuple[int, float, float]:
    """Return ``(sample_rate, duration, snr)`` for ``path``."""

    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
    dur = len(frames) / 2 / float(sr)
    snr = audio_snr(frames)
    return sr, dur, snr
