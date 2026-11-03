"""
The overlay preview of the tiles (Preview overlap, with the guessed overlaps) reads the 'auto' overlaps from the OME
metadata of the tiles, as the rigid alignment does: it failed on 'auto', the default ("int - str").
"""
import numpy as np
import pytest
import tifffile

import ClearMap.pipeline_orchestrators.stitching_orchestrator as stitching_orchestrator

from .conftest import RAW_FOLDER_NAME

TILE_SHAPE = (3, 24, 32)  # z, y, x
TILES_EXPRESSION = 'test_UltraII[<Y,2> x <X,2>]_C00.ome.tif'
OME_OVERLAPS = {'X': 6.4, 'Y': 3.2}  # In pixels, as the metadata of the microscope gives them


@pytest.fixture()
def tiled_channel(experiment, channel_editor, tmp_path, monkeypatch) -> str:
    folder = tmp_path / RAW_FOLDER_NAME / 'tiles'
    folder.mkdir(parents=True)
    for y in range(2):
        for x in range(2):
            tifffile.imwrite(folder / f'test_UltraII[{y:02d} x {x:02d}]_C00.ome.tif',
                             np.full(TILE_SHAPE, 100 * (y * 2 + x + 1), dtype=np.uint16),
                             metadata={'axes': 'ZYX', 'PhysicalSizeX': 1.625, 'PhysicalSizeY': 1.625,
                                       'PhysicalSizeZ': 3.0})
    channel_editor.add('test', 'nuclei')
    experiment.apply_ui_patch({'sample': {'channels': {'test': {'path': str(folder / TILES_EXPRESSION)}}}})
    experiment.get_worker('stitching').setup_if_needed()
    monkeypatch.setattr(stitching_orchestrator, 'parse_ome_info', _ome_info_of_the_microscope)
    return 'test'


def _ome_info_of_the_microscope(path):
    return {'overlap': OME_OVERLAPS}


def test_the_overlay_preview_reads_the_auto_overlaps_from_the_metadata(experiment, tiled_channel):
    rigid = experiment.cfg_coordinator.get_config_view('stitching')['channels'][tiled_channel]['rigid']
    assert (rigid['overlap_x'], rigid['overlap_y']) == ('auto', 'auto')

    overlay = experiment.get_worker('stitching').stitch_overlay(tiled_channel, color=False)

    assert overlay.shape == (2 * TILE_SHAPE[2] - round(OME_OVERLAPS['X']),  # x, y
                             2 * TILE_SHAPE[1] - round(OME_OVERLAPS['Y']))


def test_a_single_row_needs_no_overlap_along_the_other_axis(monkeypatch, tmp_path):
    """A row of tiles (no Y tag) has no neighbour along Y (#155): its metadata need not give a Y overlap"""
    def ome_info_without_y_overlap(path):
        return {'overlap': {'X': 6.4, 'Y': None}}
    monkeypatch.setattr(stitching_orchestrator, 'parse_ome_info', ome_info_without_y_overlap)
    rigid_cfg = {'overlap_x': 'auto', 'overlap_y': 'auto', 'projection_thickness': ['auto', 'auto', None]}

    overlaps, projection_thickness = stitching_orchestrator.define_auto_stitching_params(
        tmp_path / 'row_X00_C00.ome.tif', rigid_cfg, tile_axes=['X'])

    assert overlaps == [6.4, 0]
    assert projection_thickness == [6.4, 0, None]


def test_the_overlay_preview_of_a_single_row_with_auto_overlaps(experiment, channel_editor, tmp_path, monkeypatch):
    def ome_info_without_y_overlap(path):
        return {'overlap': {'X': OME_OVERLAPS['X'], 'Y': None}}
    folder = tmp_path / RAW_FOLDER_NAME / 'row'
    folder.mkdir(parents=True)
    for x in range(3):
        tifffile.imwrite(folder / f'row_X{x:02d}_C00.ome.tif', np.full(TILE_SHAPE, 100 * (x + 1), dtype=np.uint16),
                         metadata={'axes': 'ZYX', 'PhysicalSizeX': 1.625, 'PhysicalSizeY': 1.625, 'PhysicalSizeZ': 3.0})
    channel_editor.add('test', 'nuclei')
    experiment.apply_ui_patch({'sample': {'channels': {'test': {'path': str(folder / 'row_X<X,2>_C00.ome.tif')}}}})
    experiment.get_worker('stitching').setup_if_needed()
    monkeypatch.setattr(stitching_orchestrator, 'parse_ome_info', ome_info_without_y_overlap)

    overlay = experiment.get_worker('stitching').stitch_overlay('test', color=False)

    assert overlay.shape == (3 * TILE_SHAPE[2] - 2 * round(OME_OVERLAPS['X']), TILE_SHAPE[1])  # x, y
