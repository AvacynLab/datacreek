# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest different file formats

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from datacreek.utils.config import get_path_config, load_config, get_generation_config
from datacreek.utils.text import split_into_chunks
from datacreek.core.dataset import DatasetBuilder
from datacreek.parsers import (
    get_parser_for_extension,
    HTMLParser,
    YouTubeParser,
)

logger = logging.getLogger(__name__)


def determine_parser(file_path: str, config: Dict[str, Any]):
    """Return a parser instance for the given resource."""
    # Check if it's a URL
    if file_path.startswith(("http://", "https://")):
        # YouTube URL
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return YouTubeParser()
        # HTML URL
        else:
            return HTMLParser()

    if os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        parser = get_parser_for_extension(ext)
        if parser:
            return parser
        logger.error("Unsupported file extension: %s", ext)
        raise ValueError(f"Unsupported file extension: {ext}")

    logger.error("File not found: %s", file_path)
    raise FileNotFoundError(f"File not found: {file_path}")


def process_file(
    file_path: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Parse ``file_path`` and optionally save the result."""

    cfg = config or load_config()

    parser = determine_parser(file_path, cfg)
    content = parser.parse(file_path)

    out_dir = Path(output_dir or get_path_config(cfg, "output", "parsed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        stem = Path(file_path).stem
        output_name = f"{stem}.txt"
    out_path = out_dir / output_name
    try:
        parser.save(content, str(out_path))
    except Exception:
        # saving should not block ingestion but should be logged
        logger.exception("Failed to save parsed content to %s", out_path)

    return content


def to_kg(
    text: str,
    dataset: DatasetBuilder,
    doc_id: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    build_index: bool = True,
) -> None:
    """Split ``text`` and populate ``dataset`` with nodes.

    Parameters
    ----------
    build_index: bool, optional
        Whether to rebuild the embedding index after inserting chunks. Set to
        ``False`` when ingesting many files sequentially to build the index once
        at the end.
    """

    cfg = config or load_config()
    gen_cfg = get_generation_config(cfg)

    chunks = split_into_chunks(
        text,
        chunk_size=gen_cfg.chunk_size,
        overlap=gen_cfg.overlap,
        method=gen_cfg.chunk_method,
        similarity_drop=gen_cfg.similarity_drop,
    )

    dataset.add_document(doc_id, source=doc_id)
    for i, chunk in enumerate(chunks):
        cid = f"{doc_id}_chunk_{i}"
        dataset.add_chunk(doc_id, cid, chunk)

    if build_index:
        dataset.graph.index.build()


def ingest_into_dataset(
    file_path: str,
    dataset: DatasetBuilder,
    doc_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience helper to parse ``file_path`` and load it into ``dataset``."""

    text = process_file(file_path, config=config)
    doc_id = doc_id or Path(file_path).stem
    to_kg(text, dataset, doc_id, config, build_index=True)
    return doc_id
