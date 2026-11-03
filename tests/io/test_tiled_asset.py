"""The grid of tiles of an asset: both axes always, from 0, even for a single row or column of tiles (#155)"""
import pytest

from ClearMap.IO.workspace2 import Workspace2
from ClearMap.IO.workspace_asset import Asset, TiledAsset

SAMPLE_ID = 'S1'


def tiled_asset(directory, expression, *file_names) -> Asset:
    workspace = Workspace2(directory, sample_id=SAMPLE_ID)
    workspace.ensure_pipeline('stitching', 'fos', sample_id=SAMPLE_ID, channel_content_type='cells',
                              create_channel=True)
    asset = Asset(directory, workspace.asset_types['raw'], workspace['fos'].channel_spec,
                  expression=expression, sample_id=SAMPLE_ID)
    for name in file_names:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(name)
    assert isinstance(asset, TiledAsset)
    return asset


def grid_files(rows, columns):
    return [f'tiles/t_{y:02d}_{x:02d}.tif' for y in rows for x in columns]


def test_the_tag_values_of_the_files_are_the_ones_of_their_names(tmp_path):
    asset = tiled_asset(tmp_path, 'tiles/t_<Y,2>_<X,2>.tif', *grid_files(range(1, 3), range(4, 6)))

    assert asset.file_tag_values == [{'Y': 1, 'X': 4}, {'Y': 1, 'X': 5}, {'Y': 2, 'X': 4}, {'Y': 2, 'X': 5}]


def test_the_grid_indices_are_the_column_and_row_of_each_file(tmp_path):
    asset = tiled_asset(tmp_path, 'tiles/t_<Y,2>_<X,2>.tif', *grid_files(range(2), range(2)))

    assert asset.tile_grid_indices == [{'X': 0, 'Y': 0}, {'X': 1, 'Y': 0}, {'X': 0, 'Y': 1}, {'X': 1, 'Y': 1}]
    assert asset.tile_grid_shape == {'X': 2, 'Y': 2}


def test_the_grid_starts_at_0_whatever_the_first_tag_values(tmp_path):
    asset = tiled_asset(tmp_path, 'tiles/t_<Y,2>_<X,2>.tif', *grid_files(range(1, 3), range(4, 7)))

    assert asset.tile_grid_ranges == {'X': range(4, 7), 'Y': range(1, 3)}  # The tag values the indices start from
    assert asset.tile_grid_indices[0] == {'X': 0, 'Y': 0}
    assert asset.tile_grid_indices[-1] == {'X': 2, 'Y': 1}
    assert asset.tile_grid_shape == {'X': 3, 'Y': 2}
    assert asset.n_files_expected == 6


@pytest.mark.parametrize('axis, other', [('X', 'Y'), ('Y', 'X')])
def test_a_single_row_or_column_has_one_tile_along_the_axis_it_has_no_tag_for(tmp_path, axis, other):
    asset = tiled_asset(tmp_path, f'tiles/t_<{axis},2>.tif', *(f'tiles/t_{i:02d}.tif' for i in range(3)))

    assert asset.tile_grid_ranges == {axis: range(3), other: range(1)}
    assert asset.tile_grid_indices == [{axis: i, other: 0} for i in range(3)]
    assert asset.tile_grid_shape == {axis: 3, other: 1}
    assert asset.n_files_expected == 3


def test_a_tiled_asset_shows_the_first_and_last_values_of_each_tag_in_its_repr(tmp_path):
    """repr raised TypeError (the minimum of dicts) as soon as the asset had two files"""
    asset = tiled_asset(tmp_path, 'tiles/t_<Y,2>_<X,2>.tif', *grid_files(range(1, 3), range(4, 7)))

    assert '{6 files, Y, X: (1, 4) -> (2, 6)}' in repr(asset)


def test_the_tag_ranges_have_every_integer_tag_of_the_files(tmp_path):
    asset = tiled_asset(tmp_path, 'tiles/t_<Y,2>_<X,2>_Z<Z,4>.tif',
                        *(f'tiles/t_{y:02d}_{x:02d}_Z{z:04d}.tif' for y in (1, 2) for x in (4,) for z in range(3)))

    assert asset.tag_ranges == {'Y': range(1, 3), 'X': range(4, 5), 'Z': range(3)}
    assert asset.tile_grid_ranges == {'X': range(4, 5), 'Y': range(1, 3)}  # Not the planes


def test_the_number_of_files_expected_of_planes_counts_the_missing_ones(tmp_path):
    workspace = Workspace2(tmp_path, sample_id=SAMPLE_ID)
    workspace.ensure_pipeline('stitching', 'fos', sample_id=SAMPLE_ID, channel_content_type='cells',
                              create_channel=True)
    planes = Asset(tmp_path, workspace.asset_types['raw'], workspace['fos'].channel_spec,
                   expression='planes/Z<Z,4>.tif', sample_id=SAMPLE_ID)
    for z in (0, 1, 3):  # The plane 2 is missing
        (tmp_path / 'planes').mkdir(exist_ok=True)
        (tmp_path / 'planes' / f'Z{z:04d}.tif').write_text('plane')

    assert planes.n_files_present == 3
    assert planes.n_files_expected == 4
