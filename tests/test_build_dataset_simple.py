import runpy
import sys
import types


def test_build_dataset_runs_main(monkeypatch):
    called = []
    fake = types.SimpleNamespace(main=lambda: called.append(True))
    monkeypatch.setitem(sys.modules, 'datacreek.core.scripts.build_dataset', fake)
    runpy.run_module('datacreek.build_dataset', run_name='__main__')
    assert called == [True]


def test_build_dataset_import_no_run(monkeypatch):
    called = []
    fake = types.SimpleNamespace(main=lambda: called.append(True))
    monkeypatch.setitem(sys.modules, 'datacreek.core.scripts.build_dataset', fake)
    runpy.run_module('datacreek.build_dataset')
    assert not called
