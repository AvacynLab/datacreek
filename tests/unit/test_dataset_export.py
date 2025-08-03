from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from datacreek.utils import snapshot_tokenizer


class DummyTokenizer:
    """Minimal tokenizer stub writing deterministic JSON."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def save_pretrained(self, path: str | Path) -> None:  # pragma: no cover - trivial
        out = Path(path) / "tokenizer.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.payload, fh, ensure_ascii=False, sort_keys=True)


def test_snapshot_tokenizer(tmp_path, monkeypatch):
    tok = DummyTokenizer({"hello": 1})

    called = {}

    def fake_commit(path, repo):  # record arguments
        called["path"] = Path(path)
        called["repo"] = repo
        return "id"

    monkeypatch.setattr(
        "datacreek.utils.dataset_export.lakefs_commit", fake_commit
    )

    path, digest1 = snapshot_tokenizer(tok, path=tmp_path, repo="foo")
    assert path == tmp_path / "tokenizer.json"
    data = path.read_bytes()
    assert digest1 == sha256(data).hexdigest()
    assert called["repo"] == "foo" and called["path"] == tmp_path

    # Second snapshot should yield the same hash (stable export)
    _, digest2 = snapshot_tokenizer(tok, path=tmp_path, repo="foo")
    assert digest2 == digest1
