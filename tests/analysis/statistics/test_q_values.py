"""
estimate_q_values: the default version and the low_memory version both give the q-values of the definition
(Storey and Tibshirani, 2003): with the p-values sorted ascending p(1) ... p(n) and n tests,
q(i) = min(1, min over j >= i of pi0 * m * p(j) / j), where the rank j of tied p-values is the number of p-values <= p(j).
The definition is written here independently of both versions, as a quadratic loop.
"""
import numpy as np
import pytest

from ClearMap.Analysis.Statistics.MultipleComparisonCorrection import estimate_q_values


def q_values_from_the_definition(p_values, *, m, pi0):
    flat = np.asarray(p_values, dtype=float).ravel()
    ranks = np.array([(flat <= p).sum() for p in flat])
    scaled = pi0 * m * flat / ranks
    q_values = np.array([min(1.0, scaled[flat >= p].min()) for p in flat])
    return q_values.reshape(np.shape(p_values))


def uniform_p_values(n, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, n)


def p_values_with_signal(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(0, 1, n - n // 6), rng.uniform(0, 0.01, n // 6)])


def p_values_with_ties(n, seed=0):
    """Rounded to 2 digits: many equal values, and exact 0 and 1 are present"""
    return np.concatenate([np.round(uniform_p_values(n, seed), 2), [0.0, 1.0]])


@pytest.fixture(params=[False, True], ids=['default', 'low_memory'])
def low_memory(request):
    return request.param


def test_q_values_of_a_hand_computed_example(low_memory):
    # sorted p: .005 .01 .03 .04 ; pi0 * m * p / rank: .02 .02 .04 .04 ; minimum from the largest down: .02 .02 .04 .04
    p_values = np.array([0.01, 0.04, 0.03, 0.005])

    q_values = estimate_q_values(p_values.copy(), pi0=1.0, low_memory=low_memory)

    np.testing.assert_allclose(q_values, [0.02, 0.04, 0.04, 0.02])


@pytest.mark.parametrize('make_p_values', [uniform_p_values, p_values_with_signal, p_values_with_ties])
@pytest.mark.parametrize('n', [1, 2, 7, 99, 100, 360])
@pytest.mark.parametrize('m', [None, 50, 500])  # The number of p-values, fewer tests, more tests
@pytest.mark.parametrize('pi0', [1.0, 0.3])
def test_q_values_are_those_of_the_definition(make_p_values, n, m, pi0, low_memory):
    p_values = make_p_values(n)
    n_tests = len(p_values) if m is None else m

    q_values = estimate_q_values(p_values.copy(), m=m, pi0=pi0, low_memory=low_memory)

    np.testing.assert_allclose(q_values, q_values_from_the_definition(p_values, m=n_tests, pi0=pi0))


@pytest.mark.parametrize('make_p_values', [uniform_p_values, p_values_with_signal, p_values_with_ties])
@pytest.mark.parametrize('m', [None, 500])
def test_the_low_memory_q_values_are_the_default_ones_with_an_estimated_pi0(make_p_values, m):
    p_values = make_p_values(358).reshape(2, -1)  # 100 p-values or more: pi0 is estimated by a spline

    default = estimate_q_values(p_values.copy(), m=m)

    low_memory = estimate_q_values(p_values.copy(), m=m, low_memory=True)  # It overwrites the p-values it gets

    assert low_memory.shape == p_values.shape
    np.testing.assert_allclose(low_memory, default)


def test_the_q_values_keep_the_shape_of_the_p_values(low_memory):
    p_values = uniform_p_values(360).reshape(40, 9)

    q_values = estimate_q_values(p_values.copy(), low_memory=low_memory)

    assert q_values.shape == p_values.shape


def test_the_low_memory_version_overwrites_the_p_values_it_gets_and_the_default_does_not():
    p_values = uniform_p_values(20)
    original = p_values.copy()

    estimate_q_values(p_values, low_memory=False)
    np.testing.assert_array_equal(p_values, original)

    estimate_q_values(p_values, low_memory=True)
    assert not np.array_equal(p_values, original)
