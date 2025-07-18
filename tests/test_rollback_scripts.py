import os
import subprocess
from pathlib import Path


def test_backup_datastores(tmp_path):
    lmdb_src = tmp_path / "lmdb_src"
    lmdb_src.mkdir()
    (lmdb_src / "data.mdb").write_text("x")
    out = tmp_path / "out"
    env = os.environ.copy()
    env["LMDB_PATH"] = str(lmdb_src)
    env["REDIS_BACKUP_CMD"] = "touch"
    subprocess.check_call(["scripts/backup_datastores.sh", str(out)], env=env)
    assert (out / "lmdb_backup").exists()
    assert (out / "redis_dump.rdb").exists()


def test_revert_cache(tmp_path):
    backup = tmp_path / "back"
    lmdb_dst = tmp_path / "lmdb_dst"
    (backup / "lmdb_backup").mkdir(parents=True)
    (backup / "redis_dump.rdb").touch()
    env = os.environ.copy()
    env["LMDB_PATH"] = str(lmdb_dst)
    env["REDIS_RESTORE_CMD"] = "cat"  # no-op
    subprocess.check_call(["scripts/revert_cache.sh", str(backup)], env=env)
    assert lmdb_dst.exists()


def test_revert_ingestion_contains_restart():
    data = Path("scripts/revert_ingestion.sh").read_text()
    assert "docker compose restart ingestion" in data
