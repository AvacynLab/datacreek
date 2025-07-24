import random
from datacreek.analysis.privacy import k_out_randomized_response


def test_randomized_response_deterministic():
    items = ["a", "b", "c", "d"]
    random.seed(0)
    assert k_out_randomized_response(items, k=2) == ["d", "b", "d", "d"]


def test_randomized_response_extremes():
    items = ["a", "b", "c", "d"]
    random.seed(0)
    replaced = k_out_randomized_response(items, k=0)
    assert replaced == ["d", "d", "c", "c"]
    assert all(x in items for x in replaced)
    assert replaced != items
    random.seed(0)
    assert k_out_randomized_response(items, k=100) == items


def test_randomized_response_empty():
    assert k_out_randomized_response([], k=3) == []
