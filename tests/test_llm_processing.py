from synthetic_data_kit.utils.llm_processing import (
    convert_to_conversation_format,
    parse_qa_pairs,
    parse_ratings,
)


def test_parse_qa_pairs():
    text = '[{"question": "Q1?", "answer": "A1"}, {"question": "Q2?", "answer": "A2"}]'
    pairs = parse_qa_pairs(text)
    assert pairs and len(pairs) == 2
    assert pairs[0]["question"] == "Q1?"


def test_parse_ratings():
    text = '[{"question": "Q1?", "answer": "A1", "rating": 8}]'
    rated = parse_ratings(text)
    assert rated and rated[0]["rating"] == 8


def test_convert_to_conversation_format():
    qa_pairs = [{"question": "Q?", "answer": "A"}]
    conv = convert_to_conversation_format(qa_pairs)
    assert conv[0][1]["content"] == "Q?"
