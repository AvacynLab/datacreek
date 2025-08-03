from __future__ import annotations

"""Helpers to snapshot tokenizers alongside dataset exports.

The snapshot is saved as ``tokenizer.json`` within the given LakeFS branch
working copy so that the exact tokenizer used for the dataset can be
reproduced. The file is committed using ``lakefs commit`` to ensure it shares
the same commit as the exported dataset.
"""

from hashlib import sha256
from pathlib import Path
from typing import Tuple

from .delta_export import lakefs_commit

__all__ = ["snapshot_tokenizer"]


def snapshot_tokenizer(tokenizer, *, path: str | Path, repo: str) -> Tuple[Path, str]:
    """Persist ``tokenizer`` to ``path`` and return its SHA256 digest.

    Parameters
    ----------
    tokenizer:
        Any object providing either ``save_pretrained`` (Hugging Face style)
        or a ``backend_tokenizer.save`` method.
    path:
        Directory representing the LakeFS branch working copy where the
        tokenizer should be stored. The file will be named ``tokenizer.json``.
    repo:
        Name of the LakeFS repository to commit to. The commit is best-effort;
        failures are logged but do not raise.

    Returns
    -------
    Tuple[Path, str]
        The path to the written ``tokenizer.json`` and its SHA256 hex digest.
    """

    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    json_path = dir_path / "tokenizer.json"

    # Hugging Face tokenizers expose ``save_pretrained`` while ``tokenizers``
    # library exposes ``backend_tokenizer.save``. Support both to keep the
    # utility generic.
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(dir_path)
    elif hasattr(getattr(tokenizer, "backend_tokenizer", None), "save"):
        tokenizer.backend_tokenizer.save(str(json_path))  # type: ignore[union-attr]
    else:  # pragma: no cover - defensive branch
        raise TypeError("Tokenizer does not implement a known save method")

    data = json_path.read_bytes()
    digest = sha256(data).hexdigest()
    # Commit the snapshot so that it is versioned with the dataset. The digest
    # allows callers to assert that downstream tokenizers match this snapshot.
    lakefs_commit(dir_path, repo)
    return json_path, digest
