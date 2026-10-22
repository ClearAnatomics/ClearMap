"""
The maxima detection with an h-max transform (maxima_detection.h_max, #46): the grey reconstruction behind it is the
one of scikit-image, a bump lower than h_max on the side of a peak is not a maximum of its own, on the 16-bit images
of the microscope as on float ones, and every maximum found gets its center.
"""
import numpy as np
import pytest
from skimage.morphology import reconstruction

from ClearMap.Analysis.Measurements import maxima_detection
from ClearMap.ImageProcessing import grey_reconstruction

PEAK, BUMP = (10, 10, 10), (10, 14, 10)  # A bump on the side of the peak, its own maximum without h_max


def _image(dtype):
    grid = np.indices((21, 25, 21)).astype(float)
    def blob(center, amplitude):
        return amplitude * np.exp(-sum((g - c) ** 2 for g, c in zip(grid, center)) / 2)
    return (100 + blob(PEAK, 1000) + blob(BUMP, 200)).astype(dtype)


def _saddle_and_bump_height(image):
    """The lowest value between the peak and the bump, and how high the bump rises above it"""
    profile = image[10, PEAK[1]:BUMP[1] + 1, 10].astype(float)
    return profile.min(), profile[-1] - profile.min()


def _n_maxima(image, h_max):
    maxima = maxima_detection.find_maxima(image, h_max=h_max, shape=5, threshold=200)  # Not the flat background
    return len(maxima_detection.find_center_of_maxima(image, maxima=maxima))


@pytest.mark.parametrize('dtype', [np.uint16, np.float32])
def test_the_h_max_transform_lowers_the_peaks_by_h_max_and_flattens_a_lower_bump(dtype):
    image = _image(dtype)
    saddle, bump_height = _saddle_and_bump_height(image)

    lower = maxima_detection.h_max_transform(image, h_max=int(bump_height) + 2)
    assert lower[PEAK] == pytest.approx(float(image[PEAK]) - (int(bump_height) + 2))
    assert lower[BUMP] == pytest.approx(saddle)  # The bump is gone, down to the saddle
    assert maxima_detection.h_max_transform(image, h_max=10)[BUMP] == pytest.approx(float(image[BUMP]) - 10)


@pytest.mark.parametrize('dtype', [np.uint16, np.float32])
def test_a_bump_lower_than_h_max_is_not_a_maximum_of_its_own(dtype):
    image = _image(dtype)
    bump_height = _saddle_and_bump_height(image)[1]

    assert _n_maxima(image, None) == 2
    assert _n_maxima(image, h_max=int(bump_height) + 2) == 1
    assert _n_maxima(image, h_max=int(bump_height) // 2) == 2


def test_an_integer_h_max_on_a_16_bit_image_does_not_wrap_around():
    image = _image(np.uint16) - 100  # Background at 0, below h_max: source - h_max would wrap to ~65535

    assert maxima_detection.h_max_transform(image, 5).max() == pytest.approx(image.max() - 5)


@pytest.mark.parametrize('dtype', [np.uint16, np.float32])
def test_the_reconstruction_is_the_one_of_scikit_image(dtype):
    mask = _image(dtype)
    seed = (mask.astype(np.float32) - 5).clip(0).astype(dtype)

    reconstructed = grey_reconstruction.reconstruct(seed, mask)

    np.testing.assert_array_equal(reconstructed, reconstruction(seed, mask))
    assert reconstructed.dtype == seed.dtype


def test_every_maximum_gets_its_center():
    maxima = np.zeros((10, 10, 10), dtype=bool)
    maxima[2, 2, 2] = maxima[7, 7, 7] = True

    centers = maxima_detection.find_center_of_maxima(maxima.astype(float), maxima=maxima)

    np.testing.assert_array_equal(centers, [[2, 2, 2], [7, 7, 7]])
