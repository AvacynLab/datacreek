import importlib

import pytest

from datacreek.analysis import graphwave_cuda


def test_estimate_stream_memory():
    assert graphwave_cuda.estimate_stream_memory(10, 3, dtype_size=4) == 240


def test_choose_stream_block_without_order():
    block = graphwave_cuda.choose_stream_block(10, limit_gb=0.0, dtype_size=4)
    assert block == 1


def test_choose_stream_block_with_order():
    block = graphwave_cuda.choose_stream_block(100, limit_gb=4.0, order=5)
    assert block == 5
