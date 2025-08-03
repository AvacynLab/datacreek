from __future__ import annotations

"""Prometheus callback to expose training ETA through the HF Trainer."""

import time
from typing import Optional

try:  # pragma: no cover - optional dependency
    from transformers import TrainerCallback  # type: ignore
except Exception:  # pragma: no cover - optional dependency

    class TrainerCallback:  # type: ignore
        """Fallback base class when `transformers` is unavailable."""

        pass


try:  # pragma: no cover - optional dependency
    from prometheus_client import Gauge
except Exception:  # pragma: no cover - optional dependency
    Gauge = None  # type: ignore


class TrainingEtaSecondsCallback(TrainerCallback):
    """Report remaining training time to Prometheus.

    Parameters
    ----------
    total_steps:
        Optional override for the total number of optimization steps. If not
        provided, the callback will look for ``max_steps`` on the Trainer
        ``args`` or ``state`` objects.

    Notes
    -----
    The gauge ``training_eta_seconds`` is updated on every ``on_log`` event
    using a smoothed step-rate estimate:

    .. math::

        ETA = \frac{steps_{tot} - steps_{done}}{steps/sec}

    where ``steps/sec`` is derived from consecutive calls and exponentially
    smoothed (factor 0.9).
    """

    def __init__(self, total_steps: Optional[int] = None) -> None:
        if Gauge is None:  # pragma: no cover - optional dependency
            raise ImportError("prometheus_client is not installed")
        self.gauge = Gauge("training_eta_seconds", "ETA to finish")
        self.total_steps = total_steps
        self._last_time: Optional[float] = None
        self._last_step: Optional[int] = None
        self._rate: Optional[float] = None

    def on_log(self, args, state, control, **kwargs):  # type: ignore[override]
        """Hook executed by the Trainer to report metrics.

        Parameters
        ----------
        args, state, control:
            Objects provided by :class:`transformers.Trainer` describing the
            current training state. Only ``max_steps`` and ``global_step`` are
            accessed.
        """

        now = time.perf_counter()
        step = getattr(state, "global_step", 0)
        if self._last_time is not None and self._last_step is not None:
            dt = now - self._last_time
            ds = step - self._last_step
            if dt > 0 and ds > 0:
                inst_rate = ds / dt
                self._rate = (
                    inst_rate
                    if self._rate is None
                    else 0.9 * self._rate + 0.1 * inst_rate
                )
                steps_total = (
                    self.total_steps
                    or getattr(state, "max_steps", None)
                    or getattr(args, "max_steps", None)
                )
                if self._rate and steps_total:
                    eta = (steps_total - step) / self._rate
                    self.gauge.set(eta)
        self._last_time = now
        self._last_step = step
        return control
