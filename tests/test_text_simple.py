import json
import sys

import pytest

# Ensure datacreek package can be imported
sys.path.insert(0, ".")

from datacreek.utils import text


def test_extract_json_from_text_code_block():
    raw = 'prefix\n```json\n{"x": 1}\n```\npost'
    assert text.extract_json_from_text(raw) == {"x": 1}


def test_extract_json_from_text_failure():
    with pytest.raises(ValueError):
        text.extract_json_from_text("no json here")


def test_clean_text_basic():
    result = text.clean_text("A  B")
    assert result == "A B"


def test_split_into_chunks_simple():
    data = "a\n\n" * 10
    chunks = text.split_into_chunks(data, chunk_size=5)
    assert chunks and chunks[0].startswith("a")
