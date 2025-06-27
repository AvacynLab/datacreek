# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest different file formats

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from datacreek.core.dataset import DatasetBuilder
from datacreek.models.llm_client import LLMClient
from datacreek.parsers import HTMLParser, PDFParser, YouTubeParser, get_parser_for_extension
from datacreek.utils.config import get_generation_config, get_path_config, load_config
from datacreek.utils.text import split_into_chunks

logger = logging.getLogger(__name__)


def _resolve_input_path(file_path: str, config: Dict[str, Any]) -> str:
    """Resolve ``file_path`` using configured input directories."""

    if file_path.startswith(("http://", "https://")):
        return file_path

    if os.path.exists(file_path):
        return file_path

    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    candidates = []
    if ext:
        candidates.append(os.path.join(get_path_config(config, "input", ext), file_path))
    candidates.append(os.path.join(get_path_config(config, "input", "default"), file_path))

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    return file_path


def determine_parser(file_path: str, config: Dict[str, Any]):
    """Return a parser instance for the given resource."""
    # Check if it's a URL
    if file_path.startswith(("http://", "https://")):
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return YouTubeParser()
        from urllib.parse import urlparse

        path = urlparse(file_path).path
        ext = os.path.splitext(path)[1].lower()
        parser = get_parser_for_extension(ext)
        if parser:
            return parser
        return HTMLParser()

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    parser = get_parser_for_extension(ext)
    if parser:
        return parser
    logger.error("Unsupported file extension: %s", ext)
    raise ValueError(f"Unsupported file extension: {ext}")


def process_file(
    file_path: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    high_res: bool = False,
    ocr: bool = False,
    return_pages: bool = False,
    use_unstructured: bool | None = None,
) -> str:
    """Parse ``file_path`` and optionally save the result."""

    cfg = config or load_config()
    if use_unstructured is None:
        use_unstructured = cfg.get("ingest", {}).get("use_unstructured", True)

    resolved = _resolve_input_path(file_path, cfg)
    parser = determine_parser(resolved, cfg)
    parse_kwargs = {}
    if isinstance(parser, PDFParser):
        parse_kwargs = {"high_res": high_res, "ocr": ocr, "return_pages": return_pages}
    if hasattr(parser.parse, "__call__"):
        if "use_unstructured" in parser.parse.__code__.co_varnames:
            parse_kwargs["use_unstructured"] = use_unstructured
    content = parser.parse(resolved, **parse_kwargs)

    if return_pages and isinstance(content, tuple):
        text, pages = content
    else:
        text = content
        pages = None

    out_dir = Path(output_dir or get_path_config(cfg, "output", "parsed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        stem = Path(resolved).stem
        output_name = f"{stem}.txt"
    out_path = out_dir / output_name
    try:
        parser.save(text, str(out_path))
    except Exception:
        # saving should not block ingestion but should be logged
        logger.exception("Failed to save parsed content to %s", out_path)

    if return_pages:
        return text, pages or []
    return text


def to_kg(
    text: str,
    dataset: DatasetBuilder,
    doc_id: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    build_index: bool = True,
    pages: list[str] | None = None,
    source: str | None = None,
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

    dataset.add_document(doc_id, source=source or doc_id, text=text)

    if pages:
        chunk_idx = 0
        for page_num, page_text in enumerate(pages, start=1):
            page_chunks = split_into_chunks(
                page_text,
                chunk_size=gen_cfg.chunk_size,
                overlap=gen_cfg.overlap,
                method=gen_cfg.chunk_method,
                similarity_drop=gen_cfg.similarity_drop,
            )
            for chunk in page_chunks:
                cid = f"{doc_id}_chunk_{chunk_idx}"
                dataset.add_chunk(doc_id, cid, chunk, page=page_num)
                chunk_idx += 1
    else:
        chunks = split_into_chunks(
            text,
            chunk_size=gen_cfg.chunk_size,
            overlap=gen_cfg.overlap,
            method=gen_cfg.chunk_method,
            similarity_drop=gen_cfg.similarity_drop,
        )
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
    *,
    high_res: bool = False,
    ocr: bool = False,
    use_unstructured: bool | None = None,
    extract_entities: bool = False,
    extract_facts: bool = False,
    client: "LLMClient" | None = None,
) -> str:
    """Parse ``file_path`` and populate ``dataset`` with its content.

    Parameters
    ----------
    extract_entities:
        Run NER over inserted chunks if ``True``.
    extract_facts:
        Extract simple subject/predicate/object facts from chunks if ``True``.
        When enabled, ``client`` may supply an :class:`~datacreek.models.llm_client.LLMClient`
        instance to use for extraction.
    """

    result = process_file(
        file_path,
        config=config,
        high_res=high_res,
        ocr=ocr,
        use_unstructured=use_unstructured,
        return_pages=True,
    )
    if isinstance(result, tuple):
        text, pages = result
    else:
        text = result
        pages = None
    doc_id = doc_id or Path(file_path).stem
    to_kg(text, dataset, doc_id, config, build_index=True, pages=pages, source=file_path)

    if extract_entities:
        dataset.extract_entities()
    if extract_facts:
        dataset.extract_facts(client)
        dataset.history.append("Facts extracted on ingest")
    return doc_id
