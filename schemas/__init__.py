"""Pydantic schemas used for ingestion payload validation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, constr


class BaseIngest(BaseModel):
    """Base schema for ingest payloads."""

    path: constr(min_length=1)

    model_config = ConfigDict(extra="forbid")


class ImageIngest(BaseIngest):
    """Schema for image ingestion payloads."""

    high_res: bool = False


class AudioIngest(BaseIngest):
    """Schema for audio ingestion payloads."""

    sample_rate: int | None = None


class PdfIngest(BaseIngest):
    """Schema for PDF ingestion payloads."""

    ocr: bool = False
