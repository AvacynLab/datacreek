from __future__ import annotations

"""Helpers to snapshot tokenizers alongside dataset exports.

The snapshot is saved as ``tokenizer.json`` within the given LakeFS branch
working copy so that the exact tokenizer used for the dataset can be
reproduced. The file is committed using ``lakefs commit`` to ensure it shares
the same commit as the exported dataset.
"""

import json
from hashlib import sha256
from pathlib import Path
from typing import Tuple

from .delta_export import lakefs_commit

__all__ = ["snapshot_tokenizer"]


def snapshot_tokenizer(
    tokenizer,
    *,
    path: str | Path,
    repo: str,
    lakefs_client=None,
    metadata: str | Path | None = None,
) -> Tuple[Path, str]:
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
    lakefs_client:
        Optional LakeFS client exposing ``upload_object(repo, path, file, branch)``.
        When provided, ``tokenizer.json`` is uploaded before committing so it
        appears in the same LakeFS revision as the dataset export.
    metadata:
        Optional path to a ``metadata.json`` file that will receive the
        ``tokenizer_sha`` entry documenting the snapshot's digest.

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

    # Optionally store the digest in a metadata file so downstream consumers
    # can check that the tokenizer used for inference matches this snapshot.
    if metadata is None:
        meta_path = dir_path / "metadata.json"
    else:
        meta_path = Path(metadata)
    meta = {}
    if meta_path.exists():  # pragma: no cover - trivial
        meta = json.loads(meta_path.read_text())
    meta["tokenizer_sha"] = digest
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True))

    # Upload the file to LakeFS before committing so that it lands in the same
    # revision as the dataset export.
    if lakefs_client is not None:
        try:
            lakefs_client.upload_object(repo, str(json_path), branch="main")
        except Exception:  # pragma: no cover - network failure not fatal
            pass

    # Commit the snapshot so that it is versioned with the dataset. The digest
    # allows callers to assert that downstream tokenizers match this snapshot.
    lakefs_commit(dir_path, repo)
    return json_path, digest
