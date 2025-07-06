from __future__ import annotations

"""Image captioning utilities."""

from functools import lru_cache

from transformers import pipeline


@lru_cache(maxsize=1)
def _get_model():
    """Return a cached BLIP captioning pipeline."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


def caption_image(path: str) -> str:
    """Return a caption for the image at ``path``."""
    model = _get_model()
    try:
        result = model(path)
    except Exception as exc:  # pragma: no cover - runtime errors
        raise RuntimeError("Failed to caption image") from exc
    if not result:
        return ""
    data = result[0]
    return data.get("generated_text") or data.get("caption", "")
