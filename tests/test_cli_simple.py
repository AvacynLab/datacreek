import runpy
from typer.testing import CliRunner
import sys
import types
import pytest

from datacreek import cli

runner = CliRunner()


def test_init_db_command(monkeypatch):
    called = {}
    fake_db = types.SimpleNamespace(init_db=lambda: called.setdefault('done', True))
    monkeypatch.setitem(sys.modules, 'datacreek.db', fake_db)
    result = runner.invoke(cli.app_cli, ['init-db-cmd'])
    assert result.exit_code == 0
    assert called.get('done')
    assert 'Database initialized' in result.output


def test_test_command(monkeypatch):
    monkeypatch.setattr('pytest.main', lambda args: 0)
    result = runner.invoke(cli.app_cli, ['test'])
    assert result.exit_code == 0


def test_cli_module_exec(monkeypatch):
    monkeypatch.setattr('pytest.main', lambda args: 0)
    monkeypatch.setitem(sys.modules, 'datacreek.db', types.SimpleNamespace(init_db=lambda: None))
    monkeypatch.setattr(sys, 'argv', ['cli.py', 'test'])
    with pytest.raises(SystemExit):
        runpy.run_module('datacreek.cli', run_name='__main__')
