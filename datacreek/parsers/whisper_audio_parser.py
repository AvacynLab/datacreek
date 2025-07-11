from __future__ import annotations

"""Audio parser based on OpenAI Whisper."""

from .base import BaseParser


class WhisperAudioParser(BaseParser):
    """Parse audio files using the Whisper model."""

    def parse(self, file_path: str) -> str:
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
