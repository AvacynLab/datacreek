from datacreek.utils.chunking import sliding_window_chunks, semantic_chunk_split


def test_sliding_window_chunks():
    text = "abcdefghij" * 10
    chunks = sliding_window_chunks(text, window_size=10, overlap=2)
    assert chunks[0] == text[:10]
    assert chunks[1].startswith(text[8:18])


def test_semantic_chunk_split():
    text = "Sentence one. Sentence two related. Different topic."
    chunks = semantic_chunk_split(text, max_tokens=50, similarity_drop=0.5)
    assert len(chunks) >= 2
