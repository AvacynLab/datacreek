# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# PDF parser logic
import os
from typing import Any, Dict

from .base import BaseParser


class PDFParser(BaseParser):
    """Parser for PDF documents."""

    def parse(self, file_path: str, *, high_res: bool = False, ocr: bool = False) -> str:
        """Parse ``file_path`` and return extracted text.

        Parameters
        ----------
        file_path:
            Path to the PDF file.
        high_res:
            If ``True`` and :mod:`llamaparse` is available, use it for
            high resolution parsing of complex layouts.
        ocr:
            When ``True`` additionally run OCR on each page using
            :mod:`pytesseract`.
        """
        text = ""
        if high_res:
            try:
                from llamaparse import LlamaParse

                parser = LlamaParse()
                text = parser.parse(file_path)
            except ImportError as exc:
                raise ImportError(
                    "llamaparse is required for high resolution parsing. Install it with: pip install llamaparse"
                ) from exc
        else:
            try:
                from pdfminer.high_level import extract_text

                text = extract_text(file_path)
            except ImportError as exc:
                raise ImportError(
                    "pdfminer.six is required for PDF parsing. Install it with: pip install pdfminer.six"
                ) from exc

        if ocr:
            try:
                import pytesseract
                from pdf2image import convert_from_path

                images = convert_from_path(file_path)
                ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
                text += "\n" + ocr_text
            except ImportError as exc:
                raise ImportError(
                    "pdf2image and pytesseract are required for OCR mode. Install them with: pip install pdf2image pytesseract"
                ) from exc

        return text

    def save(self, content: str, output_path: str) -> None:
        """Save the extracted text to a file

        Args:
            content: Extracted text content
            output_path: Path to save the text
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
