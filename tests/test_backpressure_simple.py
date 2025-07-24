import importlib
import os

from datacreek.utils import backpressure as bp


def test_basic_queue_operations(tmp_path, monkeypatch):
    bp.set_limit(2)
    assert bp.has_capacity()
    assert bp.acquire_slot()
    assert bp.active_count() == 1
    assert bp.acquire_slot()
    assert not bp.has_capacity()
    assert not bp.acquire_slot()
    bp.release_slot()
    assert bp.active_count() == 1
    bp.release_slot()
    assert bp.active_count() == 0


def test_acquire_slot_with_backoff(tmp_path, monkeypatch):
    bp.set_limit(1)
    assert bp.acquire_slot()
    spool = tmp_path / "spool"
    monkeypatch.setattr(bp, "time", importlib.import_module("time"))
    assert not bp.acquire_slot_with_backoff(
        1, 0.01, spool_dir=str(spool), spool_data={"a": 1}
    )
    files = list(spool.iterdir())
    assert len(files) == 1
    bp.release_slot()
