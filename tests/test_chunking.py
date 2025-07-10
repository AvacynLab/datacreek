import importlib.util
import sys
import types
from pathlib import Path

import pytest

datacreek_pkg = sys.modules.setdefault("datacreek", types.ModuleType("datacreek"))
api_mod = types.ModuleType("datacreek.api")
api_mod.get_neo4j_driver = lambda: None
sys.modules.setdefault("datacreek.api", api_mod)
setattr(datacreek_pkg, "api", api_mod)
sys.modules.setdefault(
    "datacreek.core.dataset",
    types.SimpleNamespace(InvariantPolicy=types.SimpleNamespace(loops=0)),
)
core_pkg = types.ModuleType("datacreek.core")
dataset_mod = types.ModuleType("datacreek.core.dataset")


class InvariantPolicy:
    loops = 0


dataset_mod.InvariantPolicy = InvariantPolicy
core_pkg.dataset = dataset_mod
setattr(datacreek_pkg, "core", core_pkg)
sys.modules.setdefault("datacreek.core", core_pkg)
sys.modules["datacreek.core.dataset"] = dataset_mod

spec = importlib.util.spec_from_file_location(
    "chunking", Path(__file__).resolve().parents[1] / "datacreek/utils/chunking.py"
)
chunking = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(chunking)

contextual_chunk_split = chunking.contextual_chunk_split
semantic_chunk_split = chunking.semantic_chunk_split
sliding_window_chunks = chunking.sliding_window_chunks
summarized_chunk_split = chunking.summarized_chunk_split
chunk_by_tokens = chunking.chunk_by_tokens
chunk_by_sentences = chunking.chunk_by_sentences

SKIP_TFIDF = chunking.TfidfVectorizer is None


def test_sliding_window_chunks():
    text = "abcdefghij" * 10
    chunks = sliding_window_chunks(text, window_size=10, overlap=2)
    assert chunks[0] == text[:10]
    assert chunks[1].startswith(text[8:18])


def test_semantic_chunk_split():
    if SKIP_TFIDF:
        pytest.skip("scikit-learn not installed")
    text = "Sentence one. Sentence two related. Different topic."
    chunks = semantic_chunk_split(text, max_tokens=50, similarity_drop=0.5)
    assert len(chunks) >= 2


def test_contextual_chunk_split():
    text = (
        "Berlin est la capitale de l'Allemagne. La ville est un centre culturel majeur."
        " Encore une phrase."
    )
    chunks = contextual_chunk_split(text, max_tokens=6, context_size=3)
    assert len(chunks) > 1
    # second chunk should reuse tokens from the first chunk
    assert chunks[1].startswith("capitale de l'Allemagne")


def test_summarized_chunk_split():
    if SKIP_TFIDF:
        pytest.skip("scikit-learn not installed")
    text = "Sentence one. Sentence two about Berlin. Another."
    chunks = summarized_chunk_split(text, max_tokens=20, summary_len=3)
    assert len(chunks) >= 2
    assert "Sentence one" in chunks[1]


def test_chunk_by_tokens():
    text = "one two three four five six seven"
    chunks = chunk_by_tokens(text, max_tokens=3, overlap=1)
    assert chunks == ["one two three", "three four five", "five six seven"]


def test_chunk_by_sentences():
    text = "A. B. C. D."
    chunks = chunk_by_sentences(text, max_sentences=2)
    assert chunks == ["A. B", "C. D"]
