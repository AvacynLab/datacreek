from __future__ import annotations

"""Audio parser based on OpenAI Whisper."""

from .base import BaseParser


class WhisperAudioParser(BaseParser):
    """Parse audio files using the Whisper model."""

    def parse(self, file_path: str) -> str:
        """Return transcription of ``file_path`` using whisper.cpp when available."""

        try:
            from datacreek.utils.whisper_batch import transcribe_audio_batch

            text = transcribe_audio_batch([file_path])[0]
            if text:
                return text
        except Exception:
            # Fallback to the slower Python implementation
            pass

        try:
            import whisper
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The whisper library is required for audio parsing"
            ) from exc

        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result.get("text", "")

    def save(self, content: str, output_path: str) -> None:  # pragma: no cover - legacy
        super().save(content, output_path)
