# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Document parsers for different file formats
from .base import BaseParser
from .docx_parser import DOCXParser
from .html_parser import HTMLParser
from .pdf_parser import PDFParser
from .ppt_parser import PPTParser
from .txt_parser import TXTParser
from .youtube_parser import YouTubeParser

# Mapping of file extensions to parser classes
_PARSER_REGISTRY = {
    ".pdf": PDFParser,
    ".html": HTMLParser,
    ".htm": HTMLParser,
    ".docx": DOCXParser,
    ".pptx": PPTParser,
    ".txt": TXTParser,
}


def register_parser(ext: str, parser_cls: type[BaseParser]) -> None:
    """Register a custom parser class for the given file extension."""
    _PARSER_REGISTRY[ext.lower()] = parser_cls


def get_parser_for_extension(ext: str) -> BaseParser | None:
    cls = _PARSER_REGISTRY.get(ext.lower())
    return cls() if cls else None


__all__ = [
    "BaseParser",
    "PDFParser",
    "HTMLParser",
    "YouTubeParser",
    "DOCXParser",
    "PPTParser",
    "TXTParser",
    "register_parser",
    "get_parser_for_extension",
]
