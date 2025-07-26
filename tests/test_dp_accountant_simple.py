import math
import datacreek.dp.accountant as acc


def test_renyi_epsilon_matches_manual():
    eps = [0.1, 0.2, 0.05]
    manual = math.log(sum(math.exp((2 - 1) * e) for e in eps)) / (2 - 1)
    assert math.isclose(acc.renyi_epsilon(eps, alphas=[2]), manual)


def test_compute_epsilon_alias():
    eps = [0.1, 0.2]
    assert acc.compute_epsilon(eps, alphas=[3]) == acc.renyi_epsilon(eps, alphas=[3])


def test_allow_request_gate():
    eps = [0.3, 0.4]
    assert not acc.allow_request(eps, 0.5, alphas=[2])
    assert acc.allow_request(eps, 1.1, alphas=[2])


def test_default_alphas_used():
    eps = [0.1]
    # Should match explicit call with the internal default range
    expected = acc.renyi_epsilon(eps, alphas=range(2, 33))
    assert math.isclose(acc.renyi_epsilon(eps), expected)
