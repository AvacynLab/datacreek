"""Model compression utilities."""

from __future__ import annotations

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


def prune_fractalnet(weights: "np.ndarray | list[float]", ratio: float = 0.5):
    """Return weights pruned by magnitude preserving ``ratio`` of parameters.

    Parameters
    ----------
    weights:
        Weight matrix or vector to prune.
    ratio:
        Fraction of weights to keep based on absolute value.

    Returns
    -------
    numpy.ndarray
        Pruned weight array with the same shape.
    """

    if np is None:
        flat = [
            abs(x) for x in (weights if isinstance(weights, list) else list(weights))
        ]
        k = int(len(flat) * ratio)
        if k <= 0:
            return [0.0 for _ in flat]
        thresh = sorted(flat)[-k]
        result = [
            w if abs(w) >= thresh else 0.0
            for w in (weights if isinstance(weights, list) else list(weights))
        ]
        return result if isinstance(weights, list) else np.array(result)

    flat = weights.flatten()
    k = int(len(flat) * ratio)
    if k <= 0:
        return np.zeros_like(weights)
    thresh = np.partition(np.abs(flat), -k)[-k]
    mask = np.abs(weights) >= thresh
    return weights * mask


class FractalNetPruner:
    """Utility to prune a FractalNet model by magnitude.

    Parameters
    ----------
    lambda_ : float, optional
        Threshold below which weights are zeroed. Default ``0.03``.
    """

    def __init__(self, lambda_: float = 0.03) -> None:
        self.lambda_ = float(lambda_)
        self.model = None

    def load(
        self, repo: str = "facebookresearch/fractalnet", name: str = "fractalnet"
    ) -> None:
        """Load pretrained model from ``repo`` if possible."""
        try:  # pragma: no cover - heavy optional dependency
            import torch

            self.model = torch.hub.load(repo, name, source="github")
        except Exception:
            self.model = None

    def _perplexity(self, eval_fn) -> float:
        """Return perplexity computed by ``eval_fn``."""
        return float(eval_fn(self.model))

    def prune(
        self,
        eval_fn,
        train_fn=None,
        *,
        baseline: float | None = None,
    ) -> tuple[bool, float]:
        """Prune model weights, fine-tune briefly and check perplexity change.

        Parameters
        ----------
        eval_fn:
            Callable taking the model and returning a perplexity estimate.
        train_fn:
            Optional callable performing a single fine-tuning epoch on the model.
        baseline:
            Optional pre-computed baseline perplexity. If ``None`` the
            perplexity is obtained via ``eval_fn`` before pruning.

        Returns
        -------
        tuple
            ``(accepted, perplexity)`` with acceptance boolean and the
            perplexity after pruning.
        """
        if self.model is None:
            self.load()
            if self.model is None:
                raise RuntimeError("FractalNet model unavailable")

        if baseline is None:
            baseline = self._perplexity(eval_fn)

        try:
            import torch
            import torch.nn as nn
        except Exception:  # pragma: no cover - torch missing
            torch = None  # type: ignore
            nn = None  # type: ignore

        try:
            from ..utils.config import load_config

            cfg = load_config()
            lam = float(cfg.get("compression", {}).get("magnitude", self.lambda_))
        except Exception:  # pragma: no cover - config missing
            lam = self.lambda_

        layers = []
        if torch is not None and nn is not None and hasattr(self.model, "modules"):
            for mod in self.model.modules():
                if isinstance(mod, (nn.Linear, nn.Conv2d)):
                    layers.append(mod)
        if not layers and hasattr(self.model, "named_parameters"):
            for _, param in self.model.named_parameters():
                layers.append(param)
        if not layers:
            layers.append(self.model)

        for layer in layers:
            param = layer.weight if hasattr(layer, "weight") else layer
            try:
                arr = param.detach().cpu().numpy()
            except Exception:  # pragma: no cover - tensor not torch
                arr = np.asarray(param)
            mask = np.abs(arr) >= lam
            arr = arr * mask
            if torch is not None and isinstance(param, torch.Tensor):
                param.data = torch.tensor(arr, dtype=param.data.dtype)
            else:
                setattr(self.model, getattr(layer, "name", "weight"), arr)

        if train_fn is not None:
            try:
                train_fn(self.model)
            except Exception:  # pragma: no cover - training optional
                pass
        perplexity = self._perplexity(eval_fn)
        delta = 0.0 if baseline == 0 else abs(perplexity - baseline) / baseline
        return delta <= 0.01, perplexity
