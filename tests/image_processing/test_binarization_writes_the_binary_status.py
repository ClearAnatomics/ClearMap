"""
The vessel binarization can record, in a ``binary_status`` file, which path of the multi-path binarization flagged each
voxel (one bit per path, ``BINARY_STATUS``). The block function opened that file read-only and then added the bits into
it, so a binarization asked for its status raised. It also opened only the valid part of the block, while the steps read
the status of the whole block (overlap included): with several blocks, the shapes no longer matched.
"""
import numpy as np
import pytest

from ClearMap.ImageProcessing.Experts import Vasculature as vasculature

CLIP_LOW, CLIP_HIGH = 300, 1000
SATURATED_INTENSITY = 5000  # Above CLIP_HIGH: flagged by the clipping as 'High'
BRIGHT_INTENSITY = 900  # Below CLIP_HIGH, above the deconvolution threshold once clipped and normalized
DECONVOLUTION_THRESHOLD = 1500
SHAPE = (24, 24, 12)
SEVERAL_BLOCKS_SIZE_MAX = 5
BLOCK_SIZES = pytest.mark.parametrize('size_max', [SHAPE[2], SEVERAL_BLOCKS_SIZE_MAX], ids=['one block', 'several blocks'])


@pytest.fixture
def source_path(tmp_path):
    source = np.full(SHAPE, 500, dtype=np.uint16)
    source[4:10, 4:10, 2:8] = SATURATED_INTENSITY
    source[14:20, 14:20, 4:10] = BRIGHT_INTENSITY
    source[16:18, 16:18, 6:8] = 500  # A hole in the bright vessel, for the fill
    path = tmp_path / 'stitched.npy'
    np.save(path, source)
    return path


def binarize(source_path, *, size_max, with_status):
    binary_status_path = source_path.with_name('binary_status.npy')
    binarization_parameter = dict(vasculature.default_binarization_parameter,
                                  clip=dict(clip_range=(CLIP_LOW, CLIP_HIGH), save=None),
                                  deconvolve=dict(sigma=2, threshold=DECONVOLUTION_THRESHOLD, save=None),
                                  lightsheet=None, median=None, adaptive=None, equalize=None, vesselize=None,
                                  fill=dict(save=None), smooth=None,
                                  binary_status=str(binary_status_path) if with_status else None)
    # processes=0: the blocks run one after the other, so a block reads the finished neighbours in its overlap
    # whether or not the status is asked for
    processing_parameter = dict(vasculature.default_binarization_processing_parameter,
                                size_max=size_max, size_min=4, overlap=2, processes=0, optimization=False,
                                verbose=False)
    sink_path = source_path.with_name(f'binary_{with_status}.npy')
    vasculature.binarize(str(source_path), str(sink_path), binarization_parameter=binarization_parameter,
                         processing_parameter=processing_parameter)
    binary_status = np.load(binary_status_path) if with_status else None
    return np.load(sink_path), binary_status


@BLOCK_SIZES
def test_the_binary_status_holds_the_high_bit_where_the_clipping_flagged_the_voxels(source_path, size_max):
    _, binary_status = binarize(source_path, size_max=size_max, with_status=True)

    high = np.load(source_path) >= CLIP_HIGH
    assert np.array_equal((binary_status & vasculature.BINARY_STATUS['High']) > 0, high)


@BLOCK_SIZES
def test_the_binary_status_holds_the_deconvolved_bit_on_the_bright_vessel(source_path, size_max):
    _, binary_status = binarize(source_path, size_max=size_max, with_status=True)

    deconvolved = (binary_status & vasculature.BINARY_STATUS['Deconvolved']) > 0
    assert deconvolved[14:20, 14:20, 4:10].sum() > deconvolved[14:20, 14:20, 4:10].size // 2


@BLOCK_SIZES
def test_the_binary_is_the_voxels_with_a_status(source_path, size_max):
    binary, binary_status = binarize(source_path, size_max=size_max, with_status=True)

    assert np.array_equal(binary, binary_status > 0)


@BLOCK_SIZES
def test_asking_for_the_status_does_not_change_the_binary(source_path, size_max):
    binary_with_status, _ = binarize(source_path, size_max=size_max, with_status=True)
    binary_without_status, _ = binarize(source_path, size_max=size_max, with_status=False)

    assert np.array_equal(binary_with_status, binary_without_status)
