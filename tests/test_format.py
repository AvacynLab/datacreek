import pytest
from datacreek.core.save_as import convert_format


def test_convert_format_invalid():
    with pytest.raises(ValueError):
        convert_format({"qa_pairs": []}, None, "bad")
