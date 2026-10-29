"""
estimate_q_values with low_memory=True runs (it failed on scipy.zeros and scipy.inf, which scipy no longer has)
and returns an array of the shape of the p-values.
"""
import numpy as np
import pytest

from ClearMap.Analysis.Statistics.MultipleComparisonCorrection import estimate_q_values


@pytest.fixture()
def p_values():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, 360).reshape(40, 9)


def test_the_low_memory_version_runs_and_keeps_the_shape_of_the_p_values(p_values):
    q_values = estimate_q_values(p_values.copy(), low_memory=True)  # It overwrites the p-values it gets

    assert q_values.shape == p_values.shape
    assert np.isfinite(q_values).all()
