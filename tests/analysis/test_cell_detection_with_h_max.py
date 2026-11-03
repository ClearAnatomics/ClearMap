"""
The cell detection of a block with an h-max transform (maxima_detection.h_max, #46) finds the centers of its maxima:
a block is not an array of intensities.
"""
import numpy as np

import ClearMap.IO.IO as clearmap_io
import ClearMap.ParallelProcessing.Block as block
from ClearMap.ImageProcessing.Experts import Cells

ALL = (slice(None),) * 3


def test_the_cells_of_a_block_are_found_with_an_h_max_transform():
    image = np.full((15, 15, 15), 100, dtype=np.uint16)
    image[3:6, 3:6, 3:6] = 1000
    image[10:13, 10:13, 10:13] = 800
    parameter = dict(illumination_correction=None, background_correction=None, equalization=None, dog_filter=None,
                     maxima_detection=dict(h_max=50, shape=5, threshold=200, valid=True, save=False),
                     shape_detection=None, intensity_detection=None)

    cells = Cells.detect_cells_block(block.Block(source=clearmap_io.as_source(image), slicing=ALL, valid_slicing=ALL),
                                     parameter=parameter, n_threads=1)

    np.testing.assert_array_equal(np.sort(cells[['x', 'y', 'z']].to_numpy(), axis=0), [[4, 4, 4], [11, 11, 11]])
