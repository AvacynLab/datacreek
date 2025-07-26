import json
import subprocess
from pathlib import Path

from datacreek.utils import delta_export


def test_export_delta_variants(tmp_path, monkeypatch):
    fixed_date = '2024-01-01'
    class FakeDT:
        @staticmethod
        def now(tz=None):
            class FakeNow:
                @staticmethod
                def date():
                    class FakeDate:
                        def isoformat(self):
                            return fixed_date

                    return FakeDate()

            return FakeNow()
    monkeypatch.setattr(delta_export, 'datetime', FakeDT)

    p1 = delta_export.export_delta('hello', root=str(tmp_path), org_id='1', kind='txt')
    assert p1.read_text() == 'hello'
    # dict variant
    p2 = delta_export.export_delta({'a':1}, root=str(tmp_path), org_id=2, kind='json')
    assert json.loads(p2.read_text()) == {'a':1}
    # list variant
    p3 = delta_export.export_delta([{'x':1},{'y':2}], root=str(tmp_path), org_id=3, kind='list')
    lines = p3.read_text().splitlines()
    assert json.loads(lines[1]) == {'y':2}
    # paths include partition information
    assert Path(p1).parts[-4:-1] == (f'org_id=1', 'kind=txt', f'date={fixed_date}')


def test_lakefs_commit(monkeypatch):
    calls = []
    def fake_run(cmd, check):
        calls.append(cmd)
    monkeypatch.setattr(subprocess, 'run', fake_run)
    delta_export.lakefs_commit(Path('/tmp/file'), repo='repo')
    assert calls[0][:3] == ['lakefs', 'commit', 'repo']
    # errors are logged but ignored
    def boom(cmd, check):
        raise RuntimeError('fail')
    monkeypatch.setattr(subprocess, 'run', boom)
    delta_export.lakefs_commit(Path('/tmp/file'), repo='repo')
