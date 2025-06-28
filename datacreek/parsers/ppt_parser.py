# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# PPTX parser logic

import os
from typing import Any, Dict

from .base import BaseParser


class PPTParser(BaseParser):
    """Parser for PowerPoint presentations"""

    def parse(
        self, file_path: str, *, use_unstructured: bool = True, return_elements: bool = False
    ) -> str | list[Any]:
        """Parse a PPTX file into plain text

        Args:
            file_path: Path to the PPTX file

        Returns:
            Extracted text from the presentation
        """
        try:
            from unstructured.partition.pptx import partition_pptx

            elements = partition_pptx(filename=file_path)
            if return_elements:
                return elements
            texts = [getattr(el, "text", str(el)) for el in elements if getattr(el, "text", None)]
            return "\n".join(texts)
        except Exception as exc:  # pragma: no cover - unexpected failures
            raise RuntimeError("Failed to parse PPTX with unstructured") from exc

    def save(self, content: str, output_path: str) -> None:
        """Save the extracted text to a file

        Args:
            content: Extracted text content
            output_path: Path to save the text
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
