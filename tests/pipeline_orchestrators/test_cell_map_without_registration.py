"""
CellMap on a sample that is not registered to an atlas (P0012): the atlas alignment of the cells skips the annotation
(the cells table has no atlas columns), so the statistics per brain structure, which group the cells by
hemisphere and structure, cannot be made and are skipped too. They raised a KeyError on 'hemisphere'.
"""
import numpy as np
import pandas as pd
import pytest
import tifffile

from .conftest import RAW_FOLDER_NAME

CHANNEL = 'cells'
OME = {'axes': 'ZYX', 'PhysicalSizeX': 1.625, 'PhysicalSizeY': 1.625, 'PhysicalSizeZ': 3.0}


@pytest.fixture()
def cell_detector(experiment, channel_editor, tmp_path):
    channel_editor.add(CHANNEL, 'nuclei')
    path = tmp_path / RAW_FOLDER_NAME / CHANNEL / f'{CHANNEL}.ome.tif'
    path.parent.mkdir(parents=True)
    tifffile.imwrite(path, np.ones((8, 16, 16), np.uint16), metadata=OME)
    experiment.apply_ui_patch({'sample': {'channels': {CHANNEL: {'path': str(path)}}}})
    detector = experiment.get_worker('cell_map', channel=CHANNEL)
    detector.setup_if_needed()
    return detector


def cells_aligned_without_atlas():
    """The columns atlas_align writes when the sample is not registered: coordinates, size and intensity only"""
    return pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0], 'z': [1.0, 2.0], 'size': [5, 6], 'source': [100, 200]})


def test_the_statistics_per_structure_are_skipped_for_a_sample_not_registered(cell_detector, monkeypatch):
    assert not cell_detector.registration_processor.was_registered
    # Instead of reading the feather table (pyarrow)
    monkeypatch.setattr(cell_detector, 'get_cells_df', cells_aligned_without_atlas)

    with pytest.warns(UserWarning, match='registered'):
        cell_detector.export_collapsed_stats()

    assert not cell_detector.get_path('cells', channel=CHANNEL, asset_sub_type='stats', extension='.csv').exists()
