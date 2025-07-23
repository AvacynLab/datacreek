from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def export_delta(
    result: list | dict | str, *, root: str, org_id: int | str, kind: str
) -> Path:
    """Save ``result`` under a Delta Lake style partition and return the file path."""
    date = datetime.now(timezone.utc).date().isoformat()
    path = Path(root) / f"org_id={org_id}" / f"kind={kind}" / f"date={date}"
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "data.jsonl"
    with open(file_path, "w", encoding="utf-8") as fh:
        if isinstance(result, str):
            fh.write(result)
        else:
            if isinstance(result, dict):
                lines = [json.dumps(result, ensure_ascii=False)]
            else:
                lines = [json.dumps(r, ensure_ascii=False) for r in result]
            fh.write("\n".join(lines))
    return file_path


def lakefs_commit(path: Path, repo: str) -> None:
    """Run ``lakefs commit`` for ``repo`` pointing at ``path``."""
    try:
        subprocess.run(["lakefs", "commit", repo, "-m", f"export {path}"], check=True)
    except Exception:  # pragma: no cover - treat commit errors as non fatal
        logger.exception("LakeFS commit failed")
