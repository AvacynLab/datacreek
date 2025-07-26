import builtins
from datacreek.utils import retrieval


def test_embedding_index_basic():
    idx = retrieval.EmbeddingIndex()
    idx.add("c1", "hello world")
    idx.add("c2", "hello there")
    idx.add("c3", "goodbye world")
    idx.build()
    nns = idx.nearest_neighbors(k=1)
    assert nns["c1"][0] == "c2"
    assert idx.search("hello", k=2)[0] in {0, 1}
    assert idx.get_text(2) == "goodbye world"
    assert idx.get_id(0) == "c1"
    emb = idx.embed("testing")
    assert emb.shape[0] == idx.transform(["testing"]).shape[1]


def test_embedding_index_remove():
    idx = retrieval.EmbeddingIndex()
    idx.add("a", "foo bar")
    idx.add("b", "bar baz")
    idx.build()
    idx.remove("a")
    assert "a" not in idx.ids
    assert idx._vectorizer is None
    idx.build()
    assert idx._vectorizer is not None and len(idx.ids) == 1


def test_embedding_index_fallback(monkeypatch):
    idx = retrieval.EmbeddingIndex()
    idx.add("1", "foo bar")
    idx.add("2", "foo baz")
    monkeypatch.setattr(retrieval, "TfidfVectorizer", None)
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sklearn.feature_extraction"):
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        idx.build()
        assert idx._vectorizer is None
        assert idx._matrix.shape[0] == 2
        nns = idx.nearest_neighbors(k=1)
        assert nns["1"][0] == "2"
        # keep patched import active so transform does not load sklearn
        assert idx.transform(["missing"]).size == 0
    finally:
        monkeypatch.setattr(builtins, "__import__", orig_import)
