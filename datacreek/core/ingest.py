# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest different file formats

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel

from datacreek.core.dataset import DatasetBuilder
from datacreek.models.llm_client import LLMClient
from datacreek.parsers import HTMLParser, PDFParser, YouTubeParser, get_parser_for_extension
from datacreek.utils.config import get_generation_config, load_config
from datacreek.utils.text import clean_text, split_into_chunks

logger = logging.getLogger(__name__)

# Directory containing user uploads. When set, ingestion is restricted to this
# path to avoid reading arbitrary files from the host machine.
UPLOAD_ROOT = os.getenv("DATACREEK_UPLOAD_DIR")


def validate_file_path(path: str) -> None:
    """Ensure ``path`` resides inside :data:`UPLOAD_ROOT` when configured."""

    if path.startswith(("http://", "https://")):
        return
    if not UPLOAD_ROOT:
        return
    abs_path = os.path.abspath(path)
    root = os.path.abspath(UPLOAD_ROOT)
    if not abs_path.startswith(root):
        raise ValueError(f"Access to {path} is not allowed")


@dataclass
class IngestOptions:
    """Configuration for ingesting a document."""

    config: Optional[Dict[str, Any]] = None
    high_res: bool = False
    ocr: bool = False
    use_unstructured: bool | None = None
    extract_entities: bool = False
    extract_facts: bool = False


class IngestOptionsModel(BaseModel):
    """Pydantic validation model for :class:`IngestOptions`."""

    config: Optional[Dict[str, Any]] = None
    high_res: bool = False
    ocr: bool = False
    use_unstructured: bool | None = None
    extract_entities: bool = False
    extract_facts: bool = False

    def to_options(self) -> IngestOptions:
        """Convert to :class:`IngestOptions`."""
        return IngestOptions(**self.model_dump())


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
    config: Optional[Dict[str, Any]] = None,
    *,
    high_res: bool = False,
    ocr: bool = False,
    return_pages: bool = False,
    use_unstructured: bool | None = None,
    return_elements: bool = False,
) -> str | list[Any] | tuple[str, list[str]]:
    """Parse ``file_path`` and return the extracted text."""

    cfg = config or load_config()
    if use_unstructured is None:
        use_unstructured = cfg.get("ingest", {}).get("use_unstructured", True)

    parser = determine_parser(file_path, cfg)
    parse_kwargs = {}
    if isinstance(parser, PDFParser):
        parse_kwargs = {"high_res": high_res, "ocr": ocr, "return_pages": return_pages}
    if hasattr(parser.parse, "__call__"):
        if "use_unstructured" in parser.parse.__code__.co_varnames:
            parse_kwargs["use_unstructured"] = use_unstructured
        if return_elements and "return_elements" in parser.parse.__code__.co_varnames:
            parse_kwargs["return_elements"] = True
    content = parser.parse(file_path, **parse_kwargs)

    if return_pages and isinstance(content, tuple):
        text, pages = content
    else:
        text = content
        pages = None

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
    elements: list[Any] | None = None,
    source: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
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

    cleaned_text = clean_text(text)
    cleaned_pages = [clean_text(p) for p in pages] if pages else None

    dataset.add_document(doc_id, source=source or doc_id, text=cleaned_text)

    if elements:
        chunk_idx = 0
        img_idx = 0
        current_page = 1
        for el in elements:
            page = getattr(getattr(el, "metadata", None), "page_number", None) or current_page
            current_page = page
            path = getattr(el, "image_path", None) or getattr(
                getattr(el, "metadata", None), "image_path", None
            )
            if path:
                img_id = f"{doc_id}_image_{img_idx}"
                dataset.add_image(doc_id, img_id, path, page=page)
                img_idx += 1
                continue
            text_el = getattr(el, "text", None)
            if not text_el:
                continue
            cleaned_el = clean_text(text_el)
            chunks = split_into_chunks(
                cleaned_el,
                chunk_size=gen_cfg.chunk_size,
                overlap=gen_cfg.overlap,
                method=gen_cfg.chunk_method,
                similarity_drop=gen_cfg.similarity_drop,
            )
            for chunk in chunks:
                cid = f"{doc_id}_chunk_{chunk_idx}"
                dataset.add_chunk(doc_id, cid, chunk, page=page)
                chunk_idx += 1
                if progress_callback:
                    progress_callback(chunk_idx)
    elif cleaned_pages:
        chunk_idx = 0
        for page_num, page_text in enumerate(cleaned_pages, start=1):
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
                if progress_callback:
                    progress_callback(chunk_idx)
    else:
        chunks = split_into_chunks(
            cleaned_text,
            chunk_size=gen_cfg.chunk_size,
            overlap=gen_cfg.overlap,
            method=gen_cfg.chunk_method,
            similarity_drop=gen_cfg.similarity_drop,
        )
        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}_chunk_{i}"
            dataset.add_chunk(doc_id, cid, chunk)
            if progress_callback:
                progress_callback(i + 1)

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
    options: IngestOptions | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> str:
    """Parse ``file_path`` and populate ``dataset`` with its content.

    Parameters
    ----------
    options:
        Optional :class:`IngestOptions` overriding individual parameters.
    extract_entities:
        Run NER over inserted chunks if ``True``.
    extract_facts:
        Extract simple subject/predicate/object facts from chunks if ``True``.
        When enabled, ``client`` may supply an :class:`~datacreek.models.llm_client.LLMClient`
        instance to use for extraction.
    """

    if options is not None:
        config = options.config
        high_res = options.high_res
        ocr = options.ocr
        use_unstructured = options.use_unstructured
        extract_entities = options.extract_entities
        extract_facts = options.extract_facts

    validate_file_path(file_path)

    try:
        result = process_file(
            file_path,
            config=config,
            high_res=high_res,
            ocr=ocr,
            use_unstructured=use_unstructured,
            return_pages=True,
            return_elements=True,
        )
    except Exception:
        logger.exception("Failed to parse %s", file_path)
        raise
    elements = None
    if isinstance(result, list):
        elements = result
        text = "\n".join(
            getattr(el, "text", str(el)) for el in elements if getattr(el, "text", None)
        )
        pages = None
    elif isinstance(result, tuple):
        text, pages = result
    else:
        text = result
        pages = None
    doc_id = doc_id or Path(file_path).stem
    try:
        to_kg(
            text,
            dataset,
            doc_id,
            config,
            build_index=True,
            pages=pages,
            elements=elements,
            source=file_path,
            progress_callback=progress_callback,
        )
    except Exception:
        logger.exception("Failed to build knowledge graph for %s", file_path)
        raise

    if extract_entities:
        try:
            dataset.extract_entities()
        except Exception:
            logger.exception("Failed to extract entities from %s", file_path)
            raise
    if extract_facts:
        try:
            dataset.extract_facts(client)
        except Exception:
            logger.exception("Failed to extract facts from %s", file_path)
            raise
        dataset.history.append("Facts extracted on ingest")
    return doc_id


async def ingest_into_dataset_async(
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
    options: IngestOptions | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> str:
    """Asynchronous wrapper around :func:`ingest_into_dataset`."""

    return await asyncio.to_thread(
        ingest_into_dataset,
        file_path,
        dataset,
        doc_id,
        config,
        high_res=high_res,
        ocr=ocr,
        use_unstructured=use_unstructured,
        extract_entities=extract_entities,
        extract_facts=extract_facts,
        client=client,
        options=options,
        progress_callback=progress_callback,
    )
