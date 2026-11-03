"""
The cells farther than a distance (in atlas voxels) from the surface of the brain are kept by the cell filter, when
cell_filtration.distance_from_surface_px of the config is above 0 (the "Distance from surface" field of the GUI,
which was linked to nothing: whatever was typed there did nothing).
The distance map is in atlas space: the coordinates of the cells are transformed to it first, as for the atlas alignment.
"""
import numpy as np
import pandas as pd
import pytest

from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.pipeline_orchestrators import cell_map

from .test_cell_map_without_registration import CHANNEL, cell_detector  # noqa: F401  (fixture)

DISTANCE_TO_SURFACE = np.arange(10)[:, None, None] * np.ones((10, 4, 4))  # The distance is the x index
ATLAS_COMPONENT_OF_THE_DISTANCE = 'distance_to_surface'


class FakeAsset:
    def read(self):
        return DISTANCE_TO_SURFACE


@pytest.fixture()
def atlas_assets_asked(cell_detector, monkeypatch):  # noqa: F811
    asked = []

    def get_asset(asset_type, *, channel=None, asset_sub_type=None, **_):
        asked.append((asset_type, asset_sub_type))
        return FakeAsset()

    monkeypatch.setattr(cell_detector, 'get', get_asset)
    return asked


def set_distance(cell_detector, distance_px):  # noqa: F811
    cell_detector.patch_channel({'cell_filtration': {'distance_from_surface_px': distance_px}})


def test_the_distance_from_surface_is_off_by_default(cell_detector):  # noqa: F811
    assert cell_detector.config['cell_filtration']['distance_from_surface_px'] == 0


def test_the_cells_are_kept_by_the_distance_map_of_the_atlas_and_the_cells_outside_the_atlas_are_not(
        cell_detector, atlas_assets_asked):  # noqa: F811
    coordinates = np.array([[1.5, 0, 0], [4.2, 1, 1], [9.0, 3, 3], [12.0, 0, 0], [-1.0, 0, 0]])

    uncrusted, farther_than_threshold = cell_detector.remove_crust(coordinates=coordinates, threshold=3,
                                                                   return_mask=True)

    assert farther_than_threshold.tolist() == [False, True, True, False, False]  # x of 12 and -1 are outside
    assert uncrusted.tolist() == [[4, 1, 1], [9, 3, 3]]
    assert atlas_assets_asked == [('atlas', ATLAS_COMPONENT_OF_THE_DISTANCE)]


def test_the_distance_of_a_cell_is_read_where_the_cell_is_in_the_atlas_not_in_the_sample(
        cell_detector, atlas_assets_asked, monkeypatch):  # noqa: F811
    set_distance(cell_detector, 4)
    monkeypatch.setattr(type(cell_detector.registration_processor), 'was_registered', property(lambda _: True))
    cells = pd.DataFrame({'x': [1.0, 3.0], 'y': [0.0, 0.0], 'z': [0.0, 0.0], 'size': [10, 10], 'source': [1, 1]})
    monkeypatch.setattr(cell_detector, 'get_coords', lambda coord_type: (cells, cells[['x', 'y', 'z']].to_numpy()))
    # In the atlas the first cell is far from the surface (x of 8) and the second one is not (x of 2)
    monkeypatch.setattr(cell_detector, 'transform_coordinates',
                        lambda coordinates: np.array([[8.0, 0, 0], [2.0, 0, 0]]))
    written = {}
    monkeypatch.setattr(cell_map.io_ops, 'write', lambda path, table: written.update(path=path, table=table))

    cell_detector._remove_cells_close_to_the_surface('filtered_cells', 4)

    assert written['path'] == 'filtered_cells'
    assert written['table']['x'].tolist() == [1.0]  # Their x in the sample would have kept the other cell


def test_filtering_by_the_distance_of_a_sample_not_registered_says_how_to_fix_it(cell_detector, monkeypatch):  # noqa: F811
    set_distance(cell_detector, 4)
    raw_cells = cell_detector.get_path('cells', channel=CHANNEL, asset_sub_type='raw')
    raw_cells.parent.mkdir(parents=True, exist_ok=True)
    raw_cells.touch()
    monkeypatch.setattr(cell_map.cell_detection, 'filter_cells', lambda **_: None)
    assert not cell_detector.registration_processor.was_registered

    with pytest.raises(MissingRequirementException, match='Register the sample or set the distance to 0'):
        cell_detector.filter_cells()


def test_filtering_without_a_distance_does_not_need_the_registration(cell_detector, monkeypatch):  # noqa: F811
    raw_cells = cell_detector.get_path('cells', channel=CHANNEL, asset_sub_type='raw')
    raw_cells.parent.mkdir(parents=True, exist_ok=True)
    raw_cells.touch()
    monkeypatch.setattr(cell_map.cell_detection, 'filter_cells', lambda **_: None)

    cell_detector.filter_cells()  # Raises if it asks for the distance map
