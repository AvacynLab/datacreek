import json

from datacreek.utils.dataset_cleanup import deduplicate_pairs
from datacreek.utils.delta_export import export_delta
from datacreek.utils.gitinfo import get_commit_hash


def test_deduplicate_pairs():
    pairs = [
        {"question": "q1", "answer": "a"},
        {"question": "q1", "answer": "a"},
        {"question": "q2", "answer": "b"},
    ]
    assert deduplicate_pairs(pairs) == [pairs[0], pairs[2]]


def test_export_delta_list(tmp_path):
    data = [{"x": 1}, {"y": 2}]
    path = export_delta(data, root=str(tmp_path), org_id="o", kind="k")
    text = path.read_text()
    assert json.loads(text.splitlines()[0]) == data[0]
    assert json.loads(text.splitlines()[1]) == data[1]


def test_export_delta_string(tmp_path):
    path = export_delta("hello", root=str(tmp_path), org_id=1, kind="txt")
    assert path.read_text() == "hello"


def test_get_commit_hash():
    h = get_commit_hash()
    assert len(h) == 40
