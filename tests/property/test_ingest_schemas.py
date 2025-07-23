import pytest

pytest.importorskip("pydantic")
from pydantic import ValidationError

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from schemas import AudioIngest, ImageIngest, PdfIngest


@given(path=st.text(min_size=1), high_res=st.booleans())
@settings(max_examples=1000)
def test_image_ingest_valid(path: str, high_res: bool) -> None:
    ImageIngest(path=path, high_res=high_res)


@given(path=st.just(""))
@settings(max_examples=1000)
def test_image_ingest_invalid(path: str) -> None:
    with pytest.raises(ValidationError):
        ImageIngest(path=path)


@given(
    path=st.text(min_size=1),
    sr=st.one_of(st.none(), st.integers(min_value=8_000, max_value=48_000)),
)
@settings(max_examples=1000)
def test_audio_ingest_valid(path: str, sr: int | None) -> None:
    AudioIngest(path=path, sample_rate=sr)


@given(path=st.just(""))
@settings(max_examples=1000)
def test_audio_ingest_invalid(path: str) -> None:
    with pytest.raises(ValidationError):
        AudioIngest(path=path)


@given(path=st.text(min_size=1), ocr=st.booleans())
@settings(max_examples=1000)
def test_pdf_ingest_valid(path: str, ocr: bool) -> None:
    PdfIngest(path=path, ocr=ocr)


@given(path=st.just(""))
@settings(max_examples=1000)
def test_pdf_ingest_invalid(path: str) -> None:
    with pytest.raises(ValidationError):
        PdfIngest(path=path)
