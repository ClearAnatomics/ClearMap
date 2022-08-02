import os
import tempfile
from shutil import rmtree

import numpy as np
import pandas as pd
import pytest
import tifffile

from ClearMap.Scripts.average_pval_cm2 import compare_groups
from ClearMap.Settings import clearmap_path
from ClearMap.IO import IO as clearmap_io
from ClearMap.Alignment import Annotation as annotation
import ClearMap.Analysis.Measurements.Voxelization as voxelization


atlas = clearmap_io.read(annotation.default_annotation_file)
hemispheres_atlas = clearmap_io.read(annotation.default_hemispheres_file)
rng = np.random.default_rng()  # seed=42


@pytest.fixture
def structures():
    df_path = os.path.join(os.path.dirname(clearmap_path),
                           'tests', 'data', 'structures.csv')
    return pd.read_csv(df_path)


@pytest.fixture
def fake_stats_table(structures):
    ids = [440, 7, 201, 1047, 776, 830, 928]
    counts = [20, 12, 50, 70, 8, 50, 203]
    df = pd.DataFrame({
        'Structure ID': ids * 2,
        'Hemisphere': [0]*len(ids) + [255]*len(ids),
        'Cell counts': counts + [e + 5 for e in counts],
        # 'Average cell size': [62.7, 52.03, 80.74, 60.71, 65.28, 98.48, 53.26]
    })
    # df = patch_stats_df(structures, df, ids)
    return df


def cells_table_from_stats_table(stats_df):  # FIXME: slow
    df = fix_df_column_names(stats_df)
    cells_df = pd.DataFrame(columns=('id', 'hemisphere', 'xt', 'yt', 'zt'))
    xts, yts, zts = [], [], []
    ids, hemispheres = [], []
    for row in df.itertuples():
        coords = fake_coords_from_struct(row.s_id, row.hem_id, row.cell_counts)  # TODO: see if .T
        ids.extend([row.s_id] * row.cell_counts)
        hemispheres.extend([row.hem_id] * row.cell_counts)
        # sizes.append([row.average_cell_size] * row.cell_counts)
        row_xts, row_yts, row_zts = coords
        xts.extend(row_xts)
        yts.extend(row_yts)
        zts.extend(row_zts)
        # TODO: x, y, zs
    cells_df['id'] = ids
    cells_df['hemisphere'] = hemispheres
    cells_df['xt'] = xts
    cells_df['yt'] = yts
    cells_df['zt'] = zts
    return cells_df


def fix_df_column_names(stats_df):
    df = stats_df.rename(columns={'Structure ID': 's_id',
                                  'Hemisphere': 'hem_id',
                                  'Cell counts': 'cell_counts'},
                         # 'Average cell size': 'average_cell_size'},
                         errors='raise')
    return df


def fake_coords_from_struct(structure_id, hemisphere_id, n_cells):
    xs, ys, zs = np.where(np.logical_and(atlas == structure_id, hemispheres_atlas == hemisphere_id))
    coords = np.vstack((xs, ys, zs))
    return rng.choice(coords, n_cells, replace=False, axis=1)


def patch_stats_df(structures, df, ids):
    matching_structures = structures[structures['id'].isin(ids)]
    df['Structure order'] = matching_structures['graph_order']  # FIXME: Does not work (all NaNs)
    df['Structure name'] = matching_structures['name']
    # df['Structure volume'] = matching_structures['volume']
    return df


def patch_df(structures, df):
    matching_structures = structures[structures['id'].isin(df['id'])]
    df['order'] = matching_structures['order']
    df['name'] = matching_structures['name']
    # df['volume'] = matching_structures['volume']
    df['color'] = matching_structures['color']
    return df


def distributed_stats_table(stats_table, n_samples, sd, mu=0, abs_min=0):
    """
    Create a set of stats tables of length n_samples, with a standard deviation of sd (by uniformly sampling)

    Parameters
    ----------
    n_samples int:
        The number of tables to return
    sd float:
        The standard deviation to sample the new tables from
    mu np.array:
        The average of the distribution in addition to the values of the counts. In other words, the amount
        that the distribution will be shifted by.

    Returns
    -------

    """
    dfs = []
    bins = np.linspace(-3*sd, 3*sd, n_samples)
    # shift = bins.max()
    # gaussian_distro = 1 / (sd * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu)**2 / (2 * sd**2))
    # bins += mu
    for epsilon in bins:  # FIXME: gaussian distribution
        tbl = stats_table.copy()
        tbl['Cell counts'] = np.round(tbl['Cell counts'] + mu + epsilon + abs_min)
        tbl = tbl.astype({'Cell counts': 'int64'})
        dfs.append(tbl)
    return dfs


@pytest.fixture
def fake_group_cell_counts(n_groups=2):
    for gp in range(n_groups):
        pass


@pytest.fixture
def fake_group_density_counts(n_groups=2):
    for gp in range(n_groups):
        pass


def voxelize(folder):
    voxelization_parameter = {'radius': (15, 15, 15),
                              'shape': atlas.shape,
                              'verbose': True}
    cells = pd.read_feather(os.path.join(folder, 'cells.feather'))
    coordinates = np.array([cells[axis] for axis in ['xt', 'yt', 'zt']]).T
    counts_file_path = os.path.join(folder, 'density_counts.tif')
    clearmap_io.delete_file(counts_file_path)
    voxelization.voxelize(coordinates, sink=counts_file_path, **voxelization_parameter)
    return coordinates, counts_file_path


def voxelize_region(folder):
    stats_path = os.path.join(folder, 'cells_stats.csv')
    stats_df = fix_df_column_names(pd.read_csv(stats_path))

    density_array = np.zeros(atlas.shape)

    for row in stats_df.itertuples():
        region_mask = np.logical_and(atlas == row.s_id, hemispheres_atlas == row.hem_id)
        density_array[region_mask] = row.cell_counts

    counts_file_path = os.path.join(folder, 'density_counts.tif')
    clearmap_io.delete_file(counts_file_path)
    tifffile.imsave(counts_file_path, density_array)


def make_fake_sample_file(gp_dir, sample_name):
    with open(os.path.join(gp_dir, 'sample_params.cfg'), 'w') as sample_file:
        sample_file.writelines(["base_directory = ''\n",
                                f"sample_id = '{sample_name}'\n"])


@pytest.mark.slow
def test_p_values_whole_region(fake_stats_table):
    # tmp_dir = '/tmp/test_group_stats_whole_region'
    # if os.path.exists(tmp_dir):
    #     rmtree(tmp_dir)
    # os.mkdir(tmp_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        all_gp_tables = []
        all_gp_dirs = []
        base_shifts = np.tile(np.array([-30, -10, 0, 30, -20, -10, 25]), 2)  # negative t value = 2nd sample > first
        for i in range(2):
            shifts = base_shifts * i
            gp_tables = distributed_stats_table(fake_stats_table, 5, 5, shifts, abs_min=35)
            gp_dirs = []
            for j, stats_table in enumerate(gp_tables):
                gp_dir = os.path.join(tmp_dir, f'group_{i}_sample_{j}')
                os.mkdir(gp_dir)
                gp_dirs.append(gp_dir)

                stats_table.to_csv(os.path.join(gp_dir, 'cells_stats.csv'))

                voxelize_region(gp_dir)

                make_fake_sample_file(gp_dir, f'sample_{i}_{j}')
                cells_table = cells_table_from_stats_table(stats_table)
                cells_table.to_feather(os.path.join(gp_dir, 'cells.feather'))

            all_gp_tables.append(gp_tables)
            all_gp_dirs.append(gp_dirs)

        compare_groups(tmp_dir, 'test_group_1', 'test_group_2', all_gp_dirs[0], all_gp_dirs[1])
        color_p_vals = tifffile.imread(os.path.join(tmp_dir, 'p_val_colors_test_group_1_test_group_2.tif'))
        color_p_vals = np.transpose(color_p_vals, (2, 0, 1, 3))
        for shift, row in zip(base_shifts, fix_df_column_names(fake_stats_table).itertuples()):
            region_mask = np.logical_and(atlas == row.s_id, hemispheres_atlas == row.hem_id)
            p_vals = color_p_vals[region_mask]
            unique_vals = np.unique(p_vals, axis=0)
            unique_vals = unique_vals.astype(bool).astype(np.int_) * 255
            assert len(unique_vals) == 1
            region_p_val = unique_vals[0]
            if shift <= -25:  # yellow
                assert (region_p_val == [255, 255, 0]).all()
            elif -25 < shift < -15:  # red
                assert (region_p_val == [255, 0, 0]).all()
            elif 15 < shift <= 25:  # green
                assert (region_p_val == [0, 255, 0]).all()
            elif shift >= 25:  # blue
                assert (region_p_val == [0, 255, 255]).all()
            else:
                assert (region_p_val == [0, 0, 0]).all()


@pytest.mark.slow
def test_p_values(fake_stats_table):
    # tmp_dir = '/tmp/test_group_stats'
    # if os.path.exists(tmp_dir):
    #     rmtree(tmp_dir)
    # os.mkdir(tmp_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        all_gp_tables = []
        all_gp_dirs = []
        for i in range(2):
            shifts = np.tile(np.array([15, -10, 0, 30, -20, -10, 25]) * i, 2)
            gp_tables = distributed_stats_table(fake_stats_table, 5, 5, shifts, abs_min=35)
            gp_dirs = []
            for j, stats_table in enumerate(gp_tables):
                gp_dir = os.path.join(tmp_dir, f'group_{i}_sample_{j}')
                os.mkdir(gp_dir)
                gp_dirs.append(gp_dir)

                stats_table.to_csv(os.path.join(gp_dir, 'cells_stats.csv'))

                make_fake_sample_file(gp_dir, f'sample_{i}_{j}')
                cells_table = cells_table_from_stats_table(stats_table)
                cells_table.to_feather(os.path.join(gp_dir, 'cells.feather'))

                voxelize(gp_dir)

            all_gp_tables.append(gp_tables)
            all_gp_dirs.append(gp_dirs)

        compare_groups(tmp_dir, 'test_group_1', 'test_group_2', all_gp_dirs[0], all_gp_dirs[1])
