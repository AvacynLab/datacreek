# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest different file formats

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import importlib

from datacreek.utils.config import get_path_config

def determine_parser(file_path: str, config: Dict[str, Any]):
    """Determine the appropriate parser for a file or URL"""
    from datacreek.parsers.pdf_parser import PDFParser
    from datacreek.parsers.html_parser import HTMLParser
    from datacreek.parsers.youtube_parser import YouTubeParser
    from datacreek.parsers.docx_parser import DOCXParser
    from datacreek.parsers.ppt_parser import PPTParser
    from datacreek.parsers.txt_parser import TXTParser
    
    # Check if it's a URL
    if file_path.startswith(('http://', 'https://')):
        # YouTube URL
        if 'youtube.com' in file_path or 'youtu.be' in file_path:
            return YouTubeParser()
        # HTML URL
        else:
            return HTMLParser()
    
    # File path - determine by extension
    if os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        
        parsers = {
            '.pdf': PDFParser(),
            '.html': HTMLParser(),
            '.htm': HTMLParser(),
            '.docx': DOCXParser(),
            '.pptx': PPTParser(),
            '.txt': TXTParser(),
        }
        
        if ext in parsers:
            return parsers[ext]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    raise FileNotFoundError(f"File not found: {file_path}")

def process_file(
    file_path: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a file using the appropriate parser
    
    Args:
        file_path: Path or URL of the resource to parse
        output_dir: Ignored, kept for backward compatibility
        output_name: Ignored, kept for backward compatibility
        config: Configuration dictionary (if None, uses default)

    Returns:
        Raw text content extracted from the source
    """
    # Determine parser based on file type
    parser = determine_parser(file_path, config)

    # Parse the file
    content = parser.parse(file_path)
    return content
