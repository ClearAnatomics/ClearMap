import pytest
import numpy as np
from ClearMap.Analysis.vasculature.geometry_utils import angle_between_vectors, f_min, cartesian_to_polar


@pytest.fixture
def vectors_1():
    return np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]])


@pytest.fixture
def vectors_2():
    # In degrees, compared to vectors_1, this would be [0, 90, 180, 270, nan]
    return np.array([[1, 0, 0], [1, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 0]])


@pytest.mark.parametrize("in_degrees, expected", [
    (True, [0, 45, 0, 90, np.nan]),
    (False, [0, np.pi/4, np.pi, 3*np.pi/2, np.nan])
])
def test_angle_between_vectors(vectors_1, vectors_2, in_degrees, expected):
    expected = np.array(expected)
    result = angle_between_vectors(vectors_1, vectors_2, in_degrees)
    assert np.all(np.where(np.isnan(result))[0] == np.where(np.isnan(expected))[0])
    result = result[~np.isnan(result)]
    expected = expected[~np.isnan(expected)]
    print(result, expected)
    assert np.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("X, p, expected", [
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([0, 0, 1, 0]), np.array([3, 6, 9])),
])
def test_f_min(X, p, expected):
    result = f_min(X, p)
    assert np.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("x, y, expected_phi, expected_r", [
    (np.array([1, 1, -1, -1]), np.array([1, -1, -1, 1]), np.array([45, -45, -135, 135]) / 180, np.array([np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]) / np.sqrt(2)),
])
def test_cartesian_to_polar(x, y, expected_phi, expected_r):
    result_phi, result_r = cartesian_to_polar(x, y)
    assert np.allclose(result_phi, expected_phi, atol=1e-6)
    assert np.allclose(result_r, expected_r, atol=1e-6)
