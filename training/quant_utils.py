"""Utilities for merging LoRA adapters and quantizing models to NF4.

This module provides helper functions to merge a base model with LoRA
(adapters) and then quantize the resulting weights to the 4‑bit NF4 format
using bitsandbytes.  A per‑channel scaling scheme is applied before
quantization to guard against overflow: for each output channel we compute

    scale = max(|w|) / 127

Weights are divided by their scale so that their range lies in [-127, 127]
prior to quantization.  The scale factors are returned alongside the
quantized weights so the original values can be approximately reconstructed
without risking NaNs during training.

Example
-------
>>> base = {"linear.weight": torch.randn(64, 64)}
>>> lora = {"linear.weight": torch.randn(64, 64) * 0.01}
>>> merged = merge_lora_weights(base, lora)
>>> qstate = quantize_state_dict_nf4(merged)
>>> export_gguf(qstate, "model.gguf")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from bitsandbytes import functional as bnb


@dataclass
class QuantizedParam:
    """Container holding NF4 quantization results for a parameter.

    Attributes
    ----------
    qweight:
        Quantized weights in 4‑bit NF4 format (dtype=torch.uint8).
    absmax:
        Absolute maxima used by bitsandbytes for dequantization.
    scale:
        Per‑channel scaling factors applied prior to quantization.
    """

    qweight: torch.Tensor
    absmax: torch.Tensor
    scale: torch.Tensor


def merge_lora_weights(
    base_state: Dict[str, torch.Tensor],
    lora_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Merge LoRA deltas into a base state dict.

    Parameters
    ----------
    base_state:
        Mapping of parameter names to tensors for the base model.
    lora_state:
        Mapping containing LoRA updates to be added to the base weights.

    Returns
    -------
    Dict[str, torch.Tensor]
        New state dict containing ``base_state`` with ``lora_state`` deltas
        added where keys overlap.  The original dictionaries are not
        modified.
    """

    merged = {name: tensor.clone() for name, tensor in base_state.items()}
    for name, delta in lora_state.items():
        if name in merged:
            merged[name] = merged[name] + delta
        else:
            merged[name] = delta.clone()
    return merged


def _per_channel_scale(weight: torch.Tensor) -> torch.Tensor:
    """Compute per‑channel scaling factors for NF4 quantization.

    The scaling is computed across the last dimension so that each output
    channel (typically rows for linear layers) has its own scale.  A small
    epsilon ensures we never divide by zero.
    """

    max_per_channel = weight.abs().max(dim=-1, keepdim=True).values
    scale = torch.clamp(max_per_channel / 127.0, min=1e-8)
    return scale


def quantize_state_dict_nf4(
    state: Dict[str, torch.Tensor],
    *,
    blocksize: int = 64,
) -> Dict[str, QuantizedParam]:
    """Quantize a state dict to NF4 with an overflow guard.

    Parameters
    ----------
    state:
        Mapping of parameter names to tensors to be quantized.  Tensors are
        expected to be at least one‑dimensional.
    blocksize:
        Block size parameter forwarded to ``bitsandbytes.functional.quantize_nf4``.

    Returns
    -------
    Dict[str, QuantizedParam]
        Mapping from parameter names to :class:`QuantizedParam` objects
        containing the quantized weights, absmax statistics and per‑channel
        scales.
    """

    quantized: Dict[str, QuantizedParam] = {}
    for name, weight in state.items():
        if weight.ndim == 1:
            weight = weight.unsqueeze(0)
        scale = _per_channel_scale(weight)
        normalized = torch.clamp(weight / scale, -1.0, 1.0)
        # ``quantize_nf4`` requires the number of elements per row to be
        # divisible by ``blocksize``.  Small test tensors therefore use
        # ``blocksize=1``.
        qweight, absmax = bnb.quantize_nf4(
            normalized, blocksize=min(blocksize, normalized.shape[-1])
        )
        quantized[name] = QuantizedParam(
            qweight=qweight, absmax=absmax, scale=scale.squeeze(0)
        )
    return quantized


def dequantize_state_dict_nf4(
    qstate: Dict[str, QuantizedParam]
) -> Dict[str, torch.Tensor]:
    """Reconstruct approximate weights from NF4 quantized parameters."""

    dequantized: Dict[str, torch.Tensor] = {}
    for name, param in qstate.items():
        recovered = bnb.dequantize_nf4(param.qweight, param.absmax)
        dequantized[name] = recovered * param.scale
    return dequantized


def export_gguf(qstate: Dict[str, QuantizedParam], path: str) -> None:
    """Export quantized state to a simple GGUF container.

    The actual GGUF specification is complex; for testing purposes we
    serialize a minimal container starting with a ``b"GGUF"`` header followed
    by ``torch.save`` of the quantized parameters.
    """

    data_path = Path(path)
    with data_path.open("wb") as f:
        f.write(b"GGUF")
        torch.save(qstate, f)


def merge_and_quantize(base_path: str, lora_path: str, out_path: str) -> None:
    """Utility combining merge and quantize steps for CLI use."""

    base_state = torch.load(base_path, weights_only=True)
    lora_state = torch.load(lora_path, weights_only=True)
    merged = merge_lora_weights(base_state, lora_state)
    qstate = quantize_state_dict_nf4(merged)
    export_gguf(qstate, out_path)
