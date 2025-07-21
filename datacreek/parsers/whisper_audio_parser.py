"""Audio parser based on OpenAI Whisper."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .base import BaseParser

try:  # optional quantised matmul
    from bitsandbytes.functional import matmul_8bit as _matmul
except Exception:  # pragma: no cover - dependency missing
    _matmul = None

matmul_8bit: Optional[Callable[..., torch.Tensor]] = _matmul


class WhisperAudioParser(BaseParser):  # type: ignore[misc]
    """Parse audio files using the Whisper model."""

    def parse(self, file_path: str) -> str:
        """Return transcription using whisper.cpp when available."""

        try:
            from datacreek.utils.whisper_batch import transcribe_audio_batch
        except Exception:
            transcribe_audio_batch = None

        if transcribe_audio_batch is not None:
            text = transcribe_audio_batch([file_path])[0]
            if text:
                return str(text)

        try:
            import whisper
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The whisper library is required for audio parsing"
            ) from exc

        model = whisper.load_model("base")
        use_int8 = matmul_8bit is not None and not torch.cuda.is_available()
        if use_int8:
            orig_matmul = torch.matmul
            torch.matmul = matmul_8bit
        try:
            result = model.transcribe(file_path)
        finally:
            if use_int8:
                torch.matmul = orig_matmul
        return str(result.get("text", ""))

    def save(self, content: str, output_path: str) -> None:
        # pragma: no cover - legacy
        """Persist ``content`` to ``output_path`` using parent helper."""
        super().save(content, output_path)
