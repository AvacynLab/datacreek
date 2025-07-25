import importlib
import os
import sys
import types

import pytest

os.environ["DATACREEK_LIGHT_DATASET"] = "1"
# skip if datacreek cannot be imported due to missing deps
try:
    import datacreek
except Exception:
    pytest.skip("datacreek library unavailable", allow_module_level=True)

from datacreek.core.dataset import DatasetBuilder, DatasetType


def test_export_prompts_tag():
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.policy.loops = 0  # disable heavy policy enforcement during tests
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")
    records = ds.export_prompts()
    assert records and records[0]["tag"] == "inferred"


def test_export_prompts_metrics(monkeypatch):
    ds = DatasetBuilder(DatasetType.TEXT)
    ds.policy.loops = 0  # disable heavy policy enforcement during tests
    ds.add_document("d", source="s")
    ds.add_chunk("d", "c1", "hello")

    calls = []

    def dummy(metrics, **_):
        calls.append(metrics)

    monkeypatch.setattr("datacreek.core.dataset_light.push_metrics", dummy)

    ds.export_prompts()
    assert calls and calls[0]["prompts_exported"] == 1.0
