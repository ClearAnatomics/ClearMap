"""
Tiles acquired as one file per Z plane (the expression has a Z tag besides X and Y, as the UltraMicroscope and Blaze
microscopes write them: ``..._UltraII[<Y,2> x <X,2>]_C00_xyz-Table Z<Z,4>.ome.tif``) are stacked into one file per
tile (a column) before anything uses the tiles, on every path of the GUI (#147, #192): not only by stitch().
"""
from pathlib import Path

import numpy as np
import pytest
import tifffile

import ClearMap.pipeline_orchestrators.stitching_orchestrator as stitching_orchestrator

from .conftest import RAW_FOLDER_NAME

N_TILES_Y, N_TILES_X, N_PLANES = 2, 2, 4
PLANE_SHAPE = (24, 32)  # y, x
OME_METADATA = {'axes': 'YX', 'PhysicalSizeX': 1.625, 'PhysicalSizeY': 1.625, 'PhysicalSizeZ': 3.0}  # As written
PLANES_EXPRESSION = '10-44-28_test_UltraII[<Y,2> x <X,2>]_C00_xyz-Table Z<Z,4>.ome.tif'


def _plane(y, x, z):
    """The value of a plane tells which tile and plane it is"""
    return np.full(PLANE_SHAPE, 1000 * (y * N_TILES_X + x + 1) + z, dtype=np.uint16)


@pytest.fixture()
def per_plane_channel(experiment, channel_editor, tmp_path) -> str:
    folder = tmp_path / RAW_FOLDER_NAME / 'planes'
    folder.mkdir(parents=True)
    for y in range(N_TILES_Y):
        for x in range(N_TILES_X):
            for z in range(N_PLANES):
                name = PLANES_EXPRESSION.replace('<Y,2>', f'{y:02d}').replace('<X,2>', f'{x:02d}')
                tifffile.imwrite(folder / name.replace('<Z,4>', f'{z:04d}'), _plane(y, x, z), photometric='minisblack',
                                 metadata=OME_METADATA)
    channel_editor.add('test', 'nuclei')
    experiment.apply_ui_patch({'sample': {'channels': {'test': {'path': str(folder / PLANES_EXPRESSION)}}}})
    experiment.get_worker('stitching').setup_if_needed()
    return 'test'


def test_the_overlay_preview_of_tiles_of_one_file_per_plane(experiment, per_plane_channel):
    overlap = 4
    experiment.apply_ui_patch({'stitching': {'channels': {per_plane_channel: {'rigid': {'overlap_x': overlap,
                                                                                         'overlap_y': overlap}}}}})

    overlay = experiment.get_worker('stitching').stitch_overlay(per_plane_channel, color=False)  # KeyError: 'Z'

    assert overlay.shape == (N_TILES_X * PLANE_SHAPE[1] - overlap, N_TILES_Y * PLANE_SHAPE[0] - overlap)  # x, y
    assert overlay[0, 0] == _plane(0, 0, N_PLANES // 2)[0, 0]  # The middle plane of the first tile




def test_the_planes_of_each_tile_are_stacked_into_one_column(experiment, per_plane_channel):
    experiment.get_worker('stitching').stack_columns(per_plane_channel)

    raw = experiment.sample_manager.get('raw', channel=per_plane_channel)
    assert raw.expression.tag_names() == ['Y', 'X']
    assert Path(raw.file_list[0]).name == '10-44-28_test_UltraII[00 x 00]_C00_xyz-Table Z.ome.npy'
    assert len(raw.file_list) == N_TILES_Y * N_TILES_X
    for path in raw.file_list:
        values = raw.expression.values(path)
        column = np.load(path)
        assert column.shape[-1] == N_PLANES  # z last, in ClearMap order
        assert [column[0, 0, z] for z in range(N_PLANES)] == [_plane(values['Y'], values['X'], z)[0, 0]
                                                                for z in range(N_PLANES)]


def test_stacking_twice_changes_nothing(experiment, per_plane_channel):
    stitching_worker = experiment.get_worker('stitching')
    stitching_worker.stack_columns(per_plane_channel)
    expression = experiment.sample_manager.get('raw', channel=per_plane_channel).expression.string()

    stitching_worker.stack_columns(per_plane_channel)

    assert experiment.sample_manager.get('raw', channel=per_plane_channel).expression.string() == expression


@pytest.mark.parametrize('use_of_the_tiles', ['align_channel_rigid', 'stitch_overlay',
                                              '_stitch_layout_wobbly'])  # A channel stitched with another's layout
def test_every_use_of_the_tiles_stacks_them_first(experiment, per_plane_channel, use_of_the_tiles, monkeypatch):
    stitching_worker = experiment.get_worker('stitching')
    seen_expressions = []

    def record_the_tiles(*args, **kwargs):  # What the use finds once it gets to the tiles
        seen_expressions.append(experiment.sample_manager.get('raw', channel=per_plane_channel).expression)
        raise StopIteration
    monkeypatch.setattr(type(experiment.sample_manager), 'check_has_all_tiles', record_the_tiles)  # rigid
    monkeypatch.setattr(type(stitching_worker), '_get_overlaps', record_the_tiles)  # overlay
    monkeypatch.setattr(stitching_orchestrator.stitching_rigid, 'load_layout', record_the_tiles)  # wobbly

    with pytest.raises(StopIteration):
        getattr(stitching_worker, use_of_the_tiles)(per_plane_channel)

    assert [expression.tag_names() for expression in seen_expressions] == [['Y', 'X']]


def test_the_conversion_to_npy_converts_the_columns_not_the_planes(experiment, per_plane_channel, monkeypatch):
    stitching_worker = experiment.get_worker('stitching')
    experiment.apply_ui_patch({'stitching': {'channels': {per_plane_channel: {'use_npy': True}}}})

    stitching_worker.convert_tiles_channel(per_plane_channel)

    raw = experiment.sample_manager.get('raw', channel=per_plane_channel)
    assert raw.expression.tag_names() == ['Y', 'X']
    assert all(Path(path).suffix == '.npy' for path in raw.file_list)  # Stacked in npy: nothing more to convert


def test_the_auto_overlaps_are_read_from_the_planes_before_they_are_stacked(experiment, per_plane_channel,
                                                                              monkeypatch):
    read_from = []

    def ome_info_with_overlaps(path):
        read_from.append(Path(path).name)
        return {'overlap': {'X': 204.8, 'Y': 102.4}}
    monkeypatch.setattr(stitching_orchestrator, 'parse_ome_info', ome_info_with_overlaps)
    rigid = experiment.cfg_coordinator.get_config_view('stitching')['channels'][per_plane_channel]['rigid']
    assert (rigid['overlap_x'], rigid['overlap_y']) == ('auto', 'auto')

    experiment.get_worker('stitching').stack_columns(per_plane_channel)

    rigid = experiment.cfg_coordinator.get_config_view('stitching')['channels'][per_plane_channel]['rigid']
    assert (rigid['overlap_x'], rigid['overlap_y']) == (205, 102)  # In pixels
    assert read_from == ['10-44-28_test_UltraII[00 x 00]_C00_xyz-Table Z0000.ome.tif']


@pytest.mark.parametrize('one_file_per_plane', [False, True])
def test_the_overlay_preview_of_a_single_row_of_tiles(experiment, channel_editor, tmp_path, one_file_per_plane):
    """A single row of tiles has no Y tag (#155): KeyError: 'Y'"""
    folder = tmp_path / RAW_FOLDER_NAME / 'row'
    folder.mkdir(parents=True)
    for x in range(N_TILES_X):
        if one_file_per_plane:
            for z in range(N_PLANES):
                tifffile.imwrite(folder / f'row_X{x:02d}_C00_xyz-Table Z{z:04d}.ome.tif', _plane(0, x, z),
                                 photometric='minisblack', metadata=OME_METADATA)
        else:
            column = np.stack([_plane(0, x, z) for z in range(N_PLANES)])
            tifffile.imwrite(folder / f'row_X{x:02d}_C00.ome.tif', column, photometric='minisblack',
                             metadata={**OME_METADATA, 'axes': 'ZYX'})
    expression = 'row_X<X,2>_C00_xyz-Table Z<Z,4>.ome.tif' if one_file_per_plane else 'row_X<X,2>_C00.ome.tif'
    channel_editor.add('test', 'nuclei')
    experiment.apply_ui_patch({'sample': {'channels': {'test': {'path': str(folder / expression)}}}})
    experiment.get_worker('stitching').setup_if_needed()
    overlap = 4
    experiment.apply_ui_patch({'stitching': {'channels': {'test': {'rigid': {'overlap_x': overlap,
                                                                              'overlap_y': overlap}}}}})

    overlay = experiment.get_worker('stitching').stitch_overlay('test', color=False)

    assert overlay.shape == (N_TILES_X * PLANE_SHAPE[1] - overlap, PLANE_SHAPE[0])  # x, y
