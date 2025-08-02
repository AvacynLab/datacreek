from __future__ import annotations

"""Monitoring utilities for training metrics and callbacks."""

from dataclasses import dataclass
from typing import Optional

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore

try:
    from prometheus_client import Gauge, start_http_server
except Exception:  # pragma: no cover - optional dependency
    Gauge = None  # type: ignore
    start_http_server = None  # type: ignore


def init_wandb(project: str, **kwargs):
    """Initialize Weights & Biases logging.

    Parameters
    ----------
    project:
        Name of the W&B project.
    **kwargs:
        Additional arguments forwarded to :func:`wandb.init`.
    """
    if wandb is None:  # pragma: no cover - handled in tests
        raise ImportError("wandb is not installed")
    return wandb.init(project=project, **kwargs)


@dataclass
class PrometheusLogger:
    """Expose core training metrics through Prometheus gauges."""

    port: Optional[int] = None

    def __post_init__(self) -> None:
        if Gauge is None:  # pragma: no cover - optional dependency
            raise ImportError("prometheus_client is not installed")
        self.training_loss = Gauge("training_loss", "Current training loss")
        self.val_metric = Gauge("val_metric", "Validation metric")
        self.gpu_vram_bytes = Gauge("gpu_vram_bytes", "GPU VRAM usage in bytes")
        self.reward_avg = Gauge("reward_avg", "Average reward value")
        if self.port is not None and start_http_server:
            start_http_server(self.port)

    def log(
        self,
        training_loss: Optional[float] = None,
        val_metric: Optional[float] = None,
        gpu_vram_bytes: Optional[float] = None,
        reward_avg: Optional[float] = None,
    ) -> None:
        """Update metric gauges with new values."""
        if training_loss is not None:
            self.training_loss.set(training_loss)
        if val_metric is not None:
            self.val_metric.set(val_metric)
        if gpu_vram_bytes is not None:
            self.gpu_vram_bytes.set(gpu_vram_bytes)
        if reward_avg is not None:
            self.reward_avg.set(reward_avg)


class EarlyStopping:
    """Stop training when a monitored metric fails to improve."""

    def __init__(self, patience: int = 3, mode: str = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Update internal state and indicate whether training should stop."""
        if self.best is None:
            self.best = value
            return False
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
