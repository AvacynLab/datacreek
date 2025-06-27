import os

from .base import BaseParser


class ImageParser(BaseParser):
    """Parser for image files using unstructured."""

    def parse(self, file_path: str) -> str:
        try:
            from unstructured.partition.image import partition_image
        except Exception as exc:  # pragma: no cover - dependency missing
            raise ImportError(
                "unstructured with image support is required for image parsing."
            ) from exc
        elements = partition_image(filename=file_path)
        texts = []
        for el in elements:
            t = getattr(el, "text", str(el))
            if t:
                texts.append(t)
        return "\n".join(texts)

    def save(self, content: str, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
