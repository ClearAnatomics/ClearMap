"""
The postprocessing of TubeMap smooths a binary mask before filling it (P0015). With more than one smoothing iteration
(6 by default) the smoothed mask goes to a temporary file, which was made with the dtype of the mask. The smoothing
writes a bool array and refuses a sink of another dtype, so a uint8 mask (a mask made by another program, or a
combined mask copied from one) raised IncompatibleSource, after making a temporary file which failed the next run too.
"""
import numpy as np
import pytest

from ClearMap.IO import io_ops
from ClearMap.ImageProcessing.Experts import Vasculature as vasculature

SMOOTHING = {'iterations': 2}
SMOOTH_TMP_SUFFIX = '_smooth_tmp.npy'  # The name apply_smoothing gives the temporary file, next to the mask


@pytest.fixture
def uint8_mask_path(tmp_path):
    mask = np.zeros((20, 20, 20), np.uint8)
    mask[5:15, 5:15, 5:15] = 1
    path = tmp_path / 'binary_combined.npy'
    np.save(path, mask)
    return path


def smooth(mask_path):
    smoothed, tmp_file_path = vasculature.apply_smoothing(io_ops.open_ro(str(mask_path)), None, SMOOTHING, None,
                                                          processes=1, verbose=False)
    return np.load(tmp_file_path)


def test_a_uint8_mask_is_smoothed_into_a_binary_temporary_file(uint8_mask_path):
    smoothed = smooth(uint8_mask_path)

    assert smoothed.dtype == bool
    np.save(uint8_mask_path.with_name('bool_mask.npy'), np.load(uint8_mask_path).astype(bool))
    assert np.array_equal(smoothed, smooth(uint8_mask_path.with_name('bool_mask.npy')))


def test_the_uint8_temporary_file_of_a_failed_run_does_not_fail_the_next_one(uint8_mask_path):
    np.save(uint8_mask_path.with_name(f'{uint8_mask_path.stem}{SMOOTH_TMP_SUFFIX}'), np.load(uint8_mask_path))

    assert smooth(uint8_mask_path).dtype == bool
