import sys
sys.path.insert(0, ".")
import pytest

from datacreek.utils import chunking


def test_sliding_window_basic():
    text = "abcdef"
    result = chunking.sliding_window_chunks(text, window_size=3, overlap=1)
    assert result == ["abc", "cde", "ef"]


def test_sliding_window_errors():
    with pytest.raises(ValueError):
        chunking.sliding_window_chunks("text", 0, 1)
    with pytest.raises(ValueError):
        chunking.sliding_window_chunks("text", 2, 2)


def test_chunk_by_tokens_basic():
    out = chunking.chunk_by_tokens("a b c d e f g", 3, overlap=1)
    assert out == ["a b c", "c d e", "e f g"]


def test_chunk_by_tokens_errors():
    with pytest.raises(ValueError):
        chunking.chunk_by_tokens("a b", 0)
    with pytest.raises(ValueError):
        chunking.chunk_by_tokens("a b c", 2, overlap=2)


def test_chunk_by_sentences_basic():
    text = "One. Two. Three. Four."
    out = chunking.chunk_by_sentences(text, max_sentences=2)
    assert out == ["One. Two", "Three. Four"]


def test_contextual_chunk_split():
    text = "zero one two three four five six seven eight"
    out = chunking.contextual_chunk_split(text, max_tokens=3, context_size=2)
    assert out[0] == "zero one two"
    assert out[1] == "one two three four five"
    assert out[2] == "four five six seven eight"


def test_semantic_chunk_split_basic():
    text = "Alpha Beta. Beta Gamma. Delta Epsilon. Zeta Eta."
    out = chunking.semantic_chunk_split(text, max_tokens=40, similarity_drop=0.0)
    assert out == ["Alpha Beta. Beta Gamma. Delta Epsilon", "Zeta Eta"]


def test_semantic_chunk_split_no_deps(monkeypatch):
    monkeypatch.setattr(chunking, "TfidfVectorizer", None)
    with pytest.raises(ImportError):
        chunking.semantic_chunk_split("x", 10)
    monkeypatch.setattr(chunking, "np", None)
    with pytest.raises(ImportError):
        chunking.semantic_chunk_split("y", 10)


def test_summarized_chunk_split(monkeypatch):
    monkeypatch.setattr(chunking, "semantic_chunk_split", lambda text, max_tokens, similarity_drop=0.3: ["a b", "c d"])
    out = chunking.summarized_chunk_split("irrelevant", max_tokens=5, summary_len=1)
    assert out == ["a b", "b c d"]
