"""The intensity/size thresholds of the cell filtration are inclusive on both ends"""
import numpy as np
import pandas as pd
import pytest

from ClearMap.ImageProcessing.Experts.Cells import filter_cells

@pytest.fixture
def cells(tmp_path):
    path = tmp_path / 'cells.csv'
    pd.DataFrame({'size': [10, 20, 30, 40], 'source': [0, 100, 65534, 65535]}).to_csv(path, index=False)
    return path


def _filter(cells, **thresholds):
    sink = cells.with_name('filtered.csv')
    filter_cells(source=str(cells), sink=str(sink), thresholds=thresholds)
    return pd.read_csv(sink)


def test_saturated_cell_passes_the_default_max(cells):
    out = _filter(cells, source=(0, 65535))
    assert 65535 in out['source'].tolist()
    assert len(out) == 4


@pytest.mark.parametrize('bounds, expected', [((100, 65534), [100, 65534]),
                                             ((None, 100), [0, 100]),
                                             ((100, None), [100, 65534, 65535])])
def test_both_bounds_are_inclusive(cells, bounds, expected):
    out = _filter(cells, source=bounds)
    assert list(out['source']) == expected
