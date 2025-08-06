"""Tests for SmoothQuant scale computation and bfloat4 quantization."""

import io
from pathlib import Path

import pytest
import torch

from training.quant_utils import (
    export_bfloat4_gguf,
    quantize_bfloat4,
    smoothquant_group_scales,
)


def test_smoothquant_scales_clip_outliers() -> None:
    """Scales above the 99th percentile should be clipped."""
    group_size = 4
    # Create 100 groups: 99 normal, 1 outlier with 10x larger values
    weights = torch.ones(100 * group_size)
    weights = weights.reshape(100, group_size) * 127
    weights[-1] = 1270  # last group has scale 10
    raw_scales = weights.abs().max(dim=-1).values / 127.0
    p99 = torch.quantile(raw_scales, 0.99)
    clipped = smoothquant_group_scales(weights, group_size=group_size)
    assert torch.all(clipped <= p99)
    assert clipped[-1].item() == pytest.approx(p99.item())


def test_quantize_bfloat4_range() -> None:
    """Quantized values should lie within the signed 4-bit range."""
    torch.manual_seed(0)
    weights = torch.randn(2, 8) * 0.1
    q, scales = quantize_bfloat4(weights, group_size=4)
    assert q.dtype == torch.int8
    assert q.min() >= -8 and q.max() <= 7
    assert scales.shape[-1] == weights.shape[-1] // 4


def test_quantize_bfloat4_default_group_size() -> None:
    """Default group size of 128 should produce one scale for 128 columns."""
    weights = torch.randn(1, 128) * 0.1
    q, scales = quantize_bfloat4(weights)
    assert q.shape[-1] == 128
    assert scales.shape[-1] == 1


def test_export_bfloat4_gguf(tmp_path: Path) -> None:
    """Quantized GGUF export should start with header and be loadable."""
    state = {"linear.weight": torch.randn(1, 128)}
    out_file = tmp_path / "model_bf4.gguf"
    export_bfloat4_gguf(state, out_file)
    with out_file.open("rb") as f:
        assert f.read(4) == b"GGUF"
        qstate = torch.load(io.BytesIO(f.read()), weights_only=False)
    assert "linear.weight" in qstate
    param = qstate["linear.weight"]
    assert param.qweight.dtype == torch.int8
