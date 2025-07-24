import json
import pytest
import datacreek.utils.llm_processing as lp
from datacreek.models.qa import QAPair

lp.logger.setLevel("DEBUG")

def test_parse_qa_pairs_json_cleaning():
    text = '[{"question":"q","answer":"a",},]'
    pairs = lp.parse_qa_pairs(text)
    assert pairs == [QAPair(question="q", answer="a")]


def test_parse_qa_pairs_regex_fallback():
    text = '{"question": "x", "answer": "y"} extra {"question": "z", "answer": "w"}'
    pairs = lp.parse_qa_pairs(text)
    assert pairs == [QAPair(question="x", answer="y"), QAPair(question="z", answer="w")]


def test_parse_ratings_single_and_multi():
    obj = '{"question": "q1", "answer": "a1", "rating": 1}'
    arr = '[{"question":"q2","answer":"a2","rating":2}, {"question":"q3","answer":"a3","rating":3}]'
    single = lp.parse_ratings(obj)
    multi = lp.parse_ratings(arr)
    assert single[0].rating == 1
    assert [p.rating for p in multi] == [2, 3]


def test_parse_ratings_bad_raises():
    with pytest.raises(ValueError):
        lp.parse_ratings("not json")


def test_convert_and_records_modify():
    q = QAPair(question="q", answer="a", rating=2.5, chunk="c", source="s")
    convs = lp.convert_to_conversation_format([q])
    assert convs[0][1]["content"] == "q"
    called = {}
    def modify(conv, pair):
        called["ok"] = pair.rating
    records = lp.qa_pairs_to_records([q], modify=modify)
    assert records[0]["chunk"] == "c" and called["ok"] == 2.5


def test_parse_ratings_code_block(caplog):
    caplog.set_level("DEBUG")
    text = 'Here:\n```json\n{"question":"q","answer":"a","rating":5}\n```'
    pairs = lp.parse_ratings(text)
    assert pairs == [QAPair(question="q", answer="a", rating=5.0, chunk=None, source=None)]


def test_parse_ratings_regex():
    text = '{"question": "q", "answer": "a", "rating": 4}}'
    pairs = lp.parse_ratings(text)
    assert pairs[0].rating == 4


def test_parse_ratings_pattern():
    original = [{"question": "foo", "answer": "bar", "chunk": "c1", "source": "s"}]
    text = 'blah foo blah "rating": 2.5 end'
    pairs = lp.parse_ratings(text, original_items=original)
    assert pairs[0].rating == 2.5 and pairs[0].chunk == "c1" and pairs[0].source == "s"

def test_parse_ratings_code_block_array():
    text = '```json\n[{"question":"a","answer":"b","rating":1},{"question":"c","answer":"d","rating":2}]\n```'
    pairs = lp.parse_ratings(text)
    assert [p.rating for p in pairs] == [1, 2]

def test_parse_ratings_regex_array():
    text = '{"question":"x","answer":"y","rating":2},{"question":"z","answer":"w","rating":3}'
    pairs = lp.parse_ratings(text)
    assert pairs[0].rating == 2


def test_parse_qa_pairs_load_exception(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    def bad_load(_):
        raise ValueError("boom")
    monkeypatch.setattr(lp.json, "loads", bad_load)
    text = '[{"question":"q","answer":"a"}]'
    pairs = lp.parse_qa_pairs(text)
    assert pairs == [QAPair(question="q", answer="a")]
    assert "Error during JSON extraction" in caplog.text


def test_parse_qa_pairs_no_pairs_error(caplog):
    lp.logger.setLevel("INFO")
    caplog.set_level("ERROR")
    pairs = lp.parse_qa_pairs("junk")
    assert pairs == []
    assert "Failed to parse QA pairs" in caplog.text
    lp.logger.setLevel("DEBUG")


def test_forced_code_block_parsing(monkeypatch):
    calls = {"i": 0}
    orig = lp.json.loads

    def fl(s):
        if calls["i"] < 2:
            calls["i"] += 1
            raise json.JSONDecodeError("bad", s, 0)
        return orig(s)

    monkeypatch.setattr(lp.json, "loads", fl)
    text = 'noise {} ```json\n[{"question":"q","answer":"a","rating":5}]\n```'
    pairs = lp.parse_ratings(text)
    assert [p.rating for p in pairs] == [5.0]


def test_parse_ratings_json5(monkeypatch):
    import sys, types, json
    mod = types.ModuleType("json5")
    mod.loads = lambda s: json.loads(s.replace("'", '"'))
    monkeypatch.setitem(sys.modules, "json5", mod)
    pairs = lp.parse_ratings("{'question':'q','answer':'a','rating':3}")
    assert pairs[0].rating == 3


def test_parse_ratings_json5_array(monkeypatch):
    import sys, types, json
    mod = types.ModuleType("json5")
    mod.loads = lambda s: json.loads(s.replace("'", '"'))
    monkeypatch.setitem(sys.modules, "json5", mod)
    text = "[{'question':'a','answer':'b','rating':1}, {'question':'c','answer':'d','rating':2}]"
    pairs = lp.parse_ratings(text)
    assert [p.rating for p in pairs] == [1, 2]


def test_parse_ratings_invalid_array_item():
    text = '[{"question":"a","answer":"b","rating":1}, {"question":"c","answer":"d"}]'
    assert lp.parse_ratings(text) == []


def test_parse_ratings_meta_handling():
    original = [
        {"question": "a", "answer": "b", "chunk": "c1", "source": "s1"},
        QAPair(question="c", answer="d", chunk="c2", source="s2"),
        "oops",
    ]
    text = '[{"question":"a","answer":"b","rating":1},{"question":"c","answer":"d","rating":2},{"question":"e","answer":"f","rating":3}]'
    pairs = lp.parse_ratings(text, original_items=original)
    assert [p.chunk for p in pairs] == ["c1", "c2", None]

def test_parse_qa_pairs_json_error(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    def bad_load(_):
        # Trigger the JSONDecodeError branch
        raise json.JSONDecodeError("boom", "{", 1)
    monkeypatch.setattr(lp.json, "loads", bad_load)
    text = '[{"question":"q","answer":"a"}]'
    pairs = lp.parse_qa_pairs(text)
    assert pairs == [QAPair(question="q", answer="a")]
    assert "Direct JSON parsing failed" in caplog.text


def test_forced_code_block_parsing(caplog):
    caplog.set_level("DEBUG")
    text = 'junk { bad } ```json\n{"question":"q","answer":"a","rating":5}\n```'
    pairs = lp.parse_ratings(text)
    assert pairs == [QAPair(question="q", answer="a", rating=5.0, chunk=None, source=None)]
    assert "code block" in caplog.text


def test_parse_ratings_json5(monkeypatch):
    import sys, types, json
    mod = types.ModuleType("json5")
    mod.loads = lambda s: json.loads(s.replace("'", '"'))
    monkeypatch.setitem(sys.modules, "json5", mod)
    pairs = lp.parse_ratings("{'question':'q','answer':'a','rating':3}")
    assert pairs[0].rating == 3


def test_convert_with_dict():
    pair = {"question": "x", "answer": "y"}
    conv = lp.convert_to_conversation_format([pair])[0]
    assert conv[1]["content"] == "x" and conv[2]["content"] == "y"

import builtins


def test_parse_qa_pairs_match_error(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    class BadMatch:
        def group(self, _):
            raise ValueError("err")
    monkeypatch.setattr(lp.re, "finditer", lambda *a, **k: [BadMatch()])
    pairs = lp.parse_qa_pairs('{"question":"q","answer":"a"}')
    assert pairs == []
    assert "Error extracting pair" in caplog.text


def test_parse_ratings_import_error(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    orig_import = builtins.__import__
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "json5":
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ValueError):
        lp.parse_ratings("{'question':'q','answer':'a','rating':1}")
    assert "json5 not available" in caplog.text


def test_parse_ratings_primary_error(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    def bad_load(_):
        raise RuntimeError("boom")
    monkeypatch.setattr(lp.json, "loads", bad_load)
    orig_import = builtins.__import__
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "json5":
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ValueError):
        lp.parse_ratings('{"question":"q","answer":"a","rating":1}')
    assert "Error in primary parsing approach" in caplog.text
