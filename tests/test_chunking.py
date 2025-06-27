from datacreek.utils.chunking import (
    contextual_chunk_split,
    semantic_chunk_split,
    sliding_window_chunks,
    summarized_chunk_split,
)


def test_sliding_window_chunks():
    text = "abcdefghij" * 10
    chunks = sliding_window_chunks(text, window_size=10, overlap=2)
    assert chunks[0] == text[:10]
    assert chunks[1].startswith(text[8:18])


def test_semantic_chunk_split():
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
    text = "Sentence one. Sentence two about Berlin. Another."
    chunks = summarized_chunk_split(text, max_tokens=20, summary_len=3)
    assert len(chunks) >= 2
    assert "Sentence one" in chunks[1]
