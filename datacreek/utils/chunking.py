"""Advanced text chunking utilities."""

from typing import List

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None


def sliding_window_chunks(text: str, window_size: int, overlap: int) -> List[str]:
    """Split ``text`` using a fixed-size sliding window with overlap."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + window_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - overlap
    return chunks


def semantic_chunk_split(text: str, max_tokens: int, similarity_drop: float = 0.3) -> List[str]:
    """Split text into semantically coherent chunks.

    This uses a naive TF‑IDF embedding of sentences and creates a new chunk
    whenever the cosine similarity between adjacent sentences drops below
    ``similarity_drop`` or the chunk would exceed ``max_tokens`` characters.
    """
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return []

    if TfidfVectorizer is None:
        raise ImportError("scikit-learn is required for semantic_chunk_split")

    vectorizer = TfidfVectorizer().fit(sentences)
    embeddings = vectorizer.transform(sentences).toarray()

    chunks = []
    current = sentences[0]
    current_len = len(current)
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i - 1], embeddings[i])
        if current_len + len(sentences[i]) > max_tokens or sim < similarity_drop:
            chunks.append(current.strip())
            current = sentences[i]
            current_len = len(current)
        else:
            current += ". " + sentences[i]
            current_len += len(sentences[i]) + 2
    if current:
        chunks.append(current.strip())
    return chunks


def contextual_chunk_split(text: str, max_tokens: int, context_size: int = 20) -> List[str]:
    """Split ``text`` and prepend minimal context to each chunk."""

    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk_tokens = words[i : i + max_tokens]
        prefix_tokens = words[max(0, i - context_size) : i]
        prefix = " ".join(prefix_tokens)
        chunk = (prefix + " " if prefix else "") + " ".join(chunk_tokens)
        chunks.append(chunk.strip())
    return chunks


def summarized_chunk_split(text: str, max_tokens: int, summary_len: int = 20) -> List[str]:
    """Split text and prefix each chunk with a short summary of the previous one."""

    base_chunks = semantic_chunk_split(text, max_tokens=max_tokens)
    out: List[str] = []
    prev = ""
    for chunk in base_chunks:
        summary = " ".join(prev.split()[-summary_len:])
        out.append((summary + " " if summary else "") + chunk)
        prev = chunk
    return out
