import sys


# Stub out rich.progress if rich isn't installed
class DummyProgress:
    """Minimal stand-in for ``rich.progress.Progress`` used in tests."""

    def __init__(self, *args, **kwargs):
        self.started = False
        self.tasks = []

    def add_task(self, description, total=0):
        self.tasks.append((description, total))
        return 1

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


sys.modules["rich.progress"] = sys.modules.get(
    "rich.progress"
) or sys.modules.setdefault("rich.progress", sys.modules[__name__])
setattr(sys.modules["rich.progress"], "Progress", DummyProgress)


class DummyColumn:
    def __init__(self, *args, **kwargs):
        pass


setattr(sys.modules["rich.progress"], "BarColumn", DummyColumn)
setattr(sys.modules["rich.progress"], "TextColumn", DummyColumn)
setattr(sys.modules["rich.progress"], "TimeElapsedColumn", DummyColumn)
setattr(sys.modules["rich.progress"], "TimeRemainingColumn", DummyColumn)

from datacreek.utils.progress import create_progress, progress_context


def test_create_progress():
    p, tid = create_progress("desc", total=10)
    assert isinstance(p, DummyProgress)
    assert tid == 1


def test_progress_context():
    with progress_context("desc", 5) as (p, tid):
        assert p.started
        assert tid == 1
    assert not p.started
