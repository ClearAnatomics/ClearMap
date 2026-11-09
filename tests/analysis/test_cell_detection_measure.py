"""
The intensity of the cells is measured after the steps of the cell detection listed in intensity_detection["measure"].
A step which is disabled returns its input, so measuring it would silently give the intensity of the source (#60).
"""
import numpy as np
import pytest

from ClearMap.IO import dispatch
import ClearMap.ParallelProcessing.Block as block
from ClearMap.ImageProcessing.Experts import Cells
from ClearMap.Utils.exceptions import ClearMapValueError

ALL = (slice(None),) * 3
BACKGROUND_LEVEL = 100
CELL_LEVEL = 1000
BACKGROUND_CORRECTION = dict(shape=(5, 5), form='Disk', save=False)


def _detect_one_cell(background_correction, measure):
    image = np.full((21, 21, 9), BACKGROUND_LEVEL, dtype=np.uint16)
    image[8:11, 8:11, 3:6] = CELL_LEVEL
    parameter = dict(illumination_correction=None, background_correction=background_correction, equalization=None,
                     dog_filter=None,
                     maxima_detection=dict(h_max=None, shape=5, threshold=0, valid=True, save=False),
                     shape_detection=dict(threshold=0, save=False),
                     intensity_detection=dict(method='max', shape=3, measure=measure))
    cells = Cells.detect_cells_block(block.Block(source=dispatch.as_source(image), slicing=ALL, valid_slicing=ALL),
                                     parameter=parameter, n_threads=1)
    return cells.loc[cells['source'].idxmax()]  # The brightest: the cell


def test_the_intensity_after_the_background_correction_is_lower_than_that_of_the_source():
    cell = _detect_one_cell(BACKGROUND_CORRECTION, measure=['source', 'background_correction'])

    assert cell['source'] == CELL_LEVEL
    assert cell['background_correction'] < cell['source']


def test_measuring_the_intensity_after_a_disabled_step_is_refused():
    with pytest.raises(ClearMapValueError, match='background_correction'):
        _detect_one_cell(None, measure=['source', 'background_correction'])


def test_the_source_can_be_measured_when_every_step_is_disabled():
    cell = _detect_one_cell(None, measure=['source'])

    assert cell['source'] == CELL_LEVEL
