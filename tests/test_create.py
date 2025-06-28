import os
from pathlib import Path

from datacreek.core.create import _base_name, load_document_text, resolve_output_dir
from datacreek.utils import load_config


def test_base_name_helper():
    assert _base_name("/tmp/foo.txt") == "foo"
    assert _base_name(None) == "input"


def test_load_document_text(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello")
    assert load_document_text(str(p)) == "hello"


def test_resolve_output_dir(tmp_path):
    cfg_path = Path("configs/config.yaml")
    cfg = load_config(str(cfg_path))
    out_dir = resolve_output_dir(cfg_path)
    assert out_dir == cfg["paths"]["output"]["generated"]
