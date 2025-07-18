"""Batch Whisper.cpp transcription utilities."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Iterable, List

try:  # optional heavy dependency
    from whispercpp import Whisper  # type: ignore
except Exception:  # pragma: no cover - dependency missing
    Whisper = None  # type: ignore

__all__ = ["transcribe_audio_batch"]


@lru_cache(maxsize=1)
def _get_model(model: str = "tiny.en", fp16: bool = True, device: str | None = None):
    """Return a cached whisper.cpp model instance."""

    if Whisper is None:  # pragma: no cover - dependency missing
        return None
    # The Python bindings accept ``model_path`` and configuration options.
    # Device ``None`` lets the library select CUDA if available.
    return Whisper(model, fp16=fp16, device=device)


def transcribe_audio_batch(
    paths: Iterable[str],
    *,
    model: str = "tiny.en",
    max_seconds: int = 30,
    batch_size: int = 8,
    device: str | None = None,
) -> List[str]:
    """Return transcripts for ``paths`` using whisper.cpp in batches.

    Parameters
    ----------
    paths:
        Iterable of audio file paths. Each should contain at most ``max_seconds``
        of audio.
    model:
        Name or path of the whisper.cpp model to load. Defaults to ``"tiny.en"``.
    max_seconds:
        Maximum audio length per item in seconds. Longer files are processed but
        may reduce throughput.
    batch_size:
        Number of audio items processed per batch.
    device:
        Optional compute device (e.g. ``"cuda"``). ``None`` selects automatically.

    Returns
    -------
    list[str]
        Transcribed text for each audio path. Returns empty strings if
        ``whispercpp`` is unavailable.
    """

    all_paths = list(paths)
    model_inst = _get_model(model, fp16=True, device=device)
    if model_inst is None:  # pragma: no cover - dependency missing
        logging.getLogger(__name__).warning("whispercpp not available")
        return ["" for _ in all_paths]

    transcripts: List[str] = []
    start = time.perf_counter()
    for i in range(0, len(all_paths), batch_size):
        chunk = all_paths[i : i + batch_size]
        for path in chunk:
            try:
                text = model_inst.transcribe(path, max_length=max_seconds)
            except Exception:  # pragma: no cover - runtime error
                logging.getLogger(__name__).exception("failed to transcribe %s", path)
                text = ""
            transcripts.append(text)
    duration = time.perf_counter() - start
    if duration > 0:
        rate = len(all_paths) / duration
    else:  # pragma: no cover - extremely fast
        rate = float("inf")
    logging.getLogger(__name__).debug(
        "Whisper.cpp transcribed %d files in %.2fs (%.2f audio/s)",
        len(all_paths),
        duration,
        rate,
    )
    return transcripts
