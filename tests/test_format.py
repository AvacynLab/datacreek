import pytest

from datacreek.core.save_as import convert_format


def test_convert_format_invalid():
    with pytest.raises(ValueError):
        convert_format({"qa_pairs": []}, None, "bad")


def test_convert_format_jsonl_roundtrip(tmp_path):
    qa = {"qa_pairs": [{"question": "q", "answer": "a"}]}
    out = tmp_path / "out.jsonl"
    convert_format(qa, str(out), "jsonl")
    assert out.read_text().strip() == '{"question": "q", "answer": "a"}'


def test_convert_format_in_memory():
    qa = {"qa_pairs": [{"question": "q", "answer": "a"}]}
    res = convert_format(qa, None, "alpaca")
    assert "instruction" in res
