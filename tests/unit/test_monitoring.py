import sys
import types
from importlib import reload

import pytest


def test_init_wandb(monkeypatch):
    dummy = types.SimpleNamespace(
        init=lambda **kwargs: types.SimpleNamespace(kwargs=kwargs)
    )
    monkeypatch.setitem(sys.modules, "wandb", dummy)
    from training import monitoring

    reload(monitoring)
    run = monitoring.init_wandb(project="proj", entity="me")
    assert run.kwargs["project"] == "proj" and run.kwargs["entity"] == "me"


class DummyGauge:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        self.value = None

    def set(self, value):
        self.value = value


def test_prometheus_logger(monkeypatch):
    dummy = types.SimpleNamespace(Gauge=DummyGauge, start_http_server=lambda port: None)
    monkeypatch.setitem(sys.modules, "prometheus_client", dummy)
    from training import monitoring

    reload(monitoring)
    logger = monitoring.PrometheusLogger()
    logger.log(training_loss=0.5, val_metric=0.4, gpu_vram_bytes=1.0, reward_avg=0.7)
    assert logger.training_loss.value == 0.5
    assert logger.val_metric.value == 0.4
    assert logger.gpu_vram_bytes.value == 1.0
    assert logger.reward_avg.value == 0.7


def test_early_stopping():
    from training.monitoring import EarlyStopping

    stopper = EarlyStopping(patience=2)
    assert not stopper.step(1.0)
    assert not stopper.step(1.1)
    assert stopper.step(1.2)
