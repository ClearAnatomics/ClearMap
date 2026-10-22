"""
The grey reconstruction behind the h-max transform of the maxima detection (#46) is the one of scikit-image, in the
type of the seed.
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


@pytest.mark.parametrize('dtype', [np.uint16, np.float32])
def test_the_reconstruction_is_the_one_of_scikit_image(dtype):
    mask = _image(dtype)
    seed = (mask.astype(np.float32) - 5).clip(0).astype(dtype)

    reconstructed = grey_reconstruction.reconstruct(seed, mask)

    np.testing.assert_array_equal(reconstructed, reconstruction(seed, mask))
    assert reconstructed.dtype == seed.dtype
