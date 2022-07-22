import os

import numpy as np
import pandas as pd
from ClearMap.config.config_loader import ConfigLoader
from scipy import stats
import tifffile

from ClearMap.Alignment import Annotation as annotation
from ClearMap.Analysis.Statistics.GroupStatistics import remove_p_val_nans
from ClearMap.IO import IO as clearmap_io
from ClearMap.Analysis.Statistics import GroupStatistics as group_statistics
import ClearMap.Analysis.Statistics.MultipleComparisonCorrection as clearmap_FDR
from ClearMap.Utils.utilities import make_abs

colors = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255]
}


def stack_voxelizations(directory, f_list, suffix):
    """
    Regroup voxelisations to simplify further processing

    Parameters
    ----------
    directory
    f_list
    suffix

    Returns
    -------

    """
    for i, file_name in enumerate(f_list):
        img = clearmap_io.read(make_abs(directory, file_name))
        if i == 0:  # init on first image
            stacked_voxelizations = img[:, :, :, np.newaxis]
        else:
            stacked_voxelizations = np.concatenate((stacked_voxelizations, img[:, :, :, np.newaxis]), axis=3)
    stacked_voxelizations = stacked_voxelizations.astype(np.float32)
    try:
        clearmap_io.write(os.path.join(directory, f'stacked_density_{suffix}.tif'), stacked_voxelizations, bigtiff=True)
    except ValueError:
        pass
    return stacked_voxelizations


def average_voxelization_groups(stacked_voxelizations, directory, suffix):
    avg_voxelization = np.mean(stacked_voxelizations, axis=3)
    clearmap_io.write(os.path.join(directory, f'avg_density_{suffix}.tif'), avg_voxelization)


def get_colored_p_vals(p_vals, t_vals, significance, color_names):
    p_vals2_f, p_sign_f = get_p_vals_f(p_vals, t_vals, significance)
    return group_statistics.color_p_values(p_vals2_f, p_sign_f,
                                           positive_color=colors[color_names[0]],
                                           negative_color=colors[color_names[1]])


def is_density_file(f_name):  # FIXME: exclude debug
    # return 'density' in f_name and f_name.endswith('.tif')
    return os.path.basename(f_name) == 'density_counts.tif'  # FIXME add menu for alternatives


def find_density_file(target_dir):
    return [os.path.join(target_dir, f) for f in os.listdir(target_dir) if is_density_file(f)][0]


def dirs_to_density_files(directory, f_list):
    out = []
    for i, f_name in enumerate(f_list):
        f_name = make_abs(directory, f_name)
        if not is_density_file(f_name):
            f_name = find_density_file(f_name)
        out.append(f_name)
    return out


def get_p_vals_f(p_vals, t_vals, p_cutoff, new_orientation=(2, 0, 1)):  # FIXME: from sagittal to coronal view  specific to original orientation
    p_vals2 = np.clip(p_vals, None, p_cutoff)
    p_sign = np.sign(t_vals)
    return transpose_p_vals(new_orientation, p_sign, p_vals2)


def transpose_p_vals(new_orientation, p_sign, p_vals2):  # FIXME: check cm_rsp.sagittalToCoronalData
    p_vals2_f = np.transpose(p_vals2, new_orientation)
    p_sign_f = np.transpose(p_sign, new_orientation)
    return p_vals2_f, p_sign_f


def group_cells_counts(struct_ids, group_cells_dfs, sample_ids):
    all_ints = False
    atlas = clearmap_io.read(annotation.default_annotation_file)
    if all_ints:
        output = pd.DataFrame(columns=['id', 'hemisphere'] + [f'counts_{str(sample_ids[i]).zfill(2)}' for i in range(len(group_cells_dfs))])
    else:
        output = pd.DataFrame(columns=['id', 'hemisphere'] + [f'counts_{sample_ids[i]}' for i in range(len(group_cells_dfs))])

    output['id'] = np.tile(struct_ids, 2)  # for each hemisphere
    output['name'] = np.tile([annotation.find(_id, key='id')['name'] for _id in struct_ids], 2)
    output['volume'] = np.tile([(atlas == _id).sum() for _id in struct_ids], 2)
    output['hemisphere'] = np.repeat((0, 255), len(struct_ids))

    for multiplier, hem_id in zip((1, 2), (0, 255)):
        for j, sample_df in enumerate(group_cells_dfs):
            if all_ints:
                col_name = f'counts_{str(sample_ids[j]).zfill(2)}'  # TODO: option with f'counts_{j}'
            else:
                col_name = f'counts_{sample_ids[j]}'

            hem_sample_df = sample_df[sample_df['hemisphere'] == hem_id]
            for i, struct_id in enumerate(struct_ids):
                output.at[i*multiplier, col_name] = len(hem_sample_df[hem_sample_df['id'] == struct_id])  # FIXME: slow
    return output


def get_all_structs(dfs):
    structs = pd.Series()
    for df in dfs:
        structs = pd.concat((structs, df['id']))
    return np.sort(structs.unique())


def generate_summary_table(cells_dfs, p_cutoff=None):
    gp_names = list(cells_dfs.keys())

    grouped_counts = []

    total_df = pd.DataFrame()
    total_df['id'] = cells_dfs[gp_names[0]]['id']
    total_df['name'] = cells_dfs[gp_names[0]]['name']
    total_df['volume'] = cells_dfs[gp_names[0]]['volume']
    total_df['hemisphere'] = cells_dfs[gp_names[0]]['hemisphere']
    for i, gp_name in enumerate(gp_names):
        grouped_counts.append(pd.DataFrame())
        for col_name in cells_dfs[gp_name].columns:
            if 'count' in col_name:
                col = cells_dfs[gp_name][col_name]
                new_col_name = f'{gp_names[i]}_{col_name}'
                total_df[new_col_name] = col
                grouped_counts[i][new_col_name] = col
        total_df[f'mean_{gp_name}'] = grouped_counts[i].mean(axis=1)
        total_df[f'sd_{gp_name}'] = grouped_counts[i].std(axis=1)

    total_df, grouped_counts = sanitize_df(gp_names, grouped_counts, total_df)

    gp1 = grouped_counts[0].values.astype(np.int)
    gp2 = grouped_counts[1].values.astype(np.int)
    p_vals, p_signs = group_statistics.t_test_region_counts(gp1, gp2, p_cutoff=p_cutoff, signed=True)
    total_df['p_value'] = p_vals
    total_df['q_value'] = clearmap_FDR.estimate_q_values(p_vals)
    total_df['p_sign'] = p_signs
    return total_df


def sanitize_df(gp_names, grouped_counts, total_df):
    """
    Remove rows with all 0 or NaN in at least 1 group
    Args:
        gp_names:
        grouped_counts:
        total_df:

    Returns:

    """
    bad_idx = total_df[f'mean_{gp_names[0]}'] == 0  # FIXME: check that either not and
    bad_idx = np.logical_or(bad_idx, total_df[f'mean_{gp_names[1]}'] == 0)
    bad_idx = np.logical_or(bad_idx, np.isnan(total_df[f'mean_{gp_names[0]}']))
    bad_idx = np.logical_or(bad_idx, np.isnan(total_df[f'mean_{gp_names[1]}']))

    return total_df[~bad_idx], [grouped_counts[0][~bad_idx], grouped_counts[1][~bad_idx]]


def find_cells_df(target_dir):
    return [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('cells.feather')][0]


def dirs_to_cells_dfs(directory, dirs):
    out = []
    for i, f_name in enumerate(dirs):
        f_name = make_abs(directory, f_name)
        if not f_name.endswith('cells.feather'):
            f_name = find_cells_df(f_name)
        out.append(pd.read_feather(f_name))
    return out


def dir_to_sample_id(folder):
    cfg_loader = ConfigLoader(folder)
    return cfg_loader.get_cfg('sample')['sample_id']


def make_summary(directory, gp1_name, gp2_name, gp1_dirs, gp2_dirs, output_path=None, save=True):
    gp1_dfs = dirs_to_cells_dfs(directory, gp1_dirs)
    gp2_dfs = dirs_to_cells_dfs(directory, gp2_dirs)
    gp_cells_dfs = [gp1_dfs, gp2_dfs]
    structs = get_all_structs(gp1_dfs + gp2_dfs)

    gp1_sample_ids = [dir_to_sample_id(folder) for folder in gp1_dirs]
    gp2_sample_ids = [dir_to_sample_id(folder) for folder in gp2_dirs]
    sample_ids = [gp1_sample_ids, gp2_sample_ids]

    aggregated_dfs = {gp_name: group_cells_counts(structs, gp_cells_dfs[i], sample_ids[i])
                      for i, gp_name in enumerate((gp1_name, gp2_name))}
    total_df = generate_summary_table(aggregated_dfs)

    if output_path is None and save:
        output_path = os.path.join(directory, f'statistics_{gp1_name}_{gp2_name}.csv')
    if save:
        total_df.to_csv(output_path)
    return total_df


def density_files_are_comparable(directory, gp1_dirs, gp2_dirs):
    gp1_f_list = dirs_to_density_files(directory, gp1_dirs)
    gp2_f_list = dirs_to_density_files(directory, gp2_dirs)
    all_files = gp1_f_list + gp2_f_list
    sizes = [os.path.getsize(f) for f in all_files]
    return all(s == sizes[0] for s in sizes)


def compare_groups(directory, gp1_name, gp2_name, gp1_dirs, gp2_dirs, prefix='p_val_colors'):
    make_summary(directory, gp1_name, gp2_name, gp1_dirs, gp2_dirs)

    gp1_f_list = dirs_to_density_files(directory, gp1_dirs)
    gp2_f_list = dirs_to_density_files(directory, gp2_dirs)

    gp1_stacked_voxelizations = stack_voxelizations(directory, gp1_f_list, suffix=gp1_name)
    average_voxelization_groups(gp1_stacked_voxelizations, directory, gp1_name)
    gp2_stacked_voxelizations = stack_voxelizations(directory, gp2_f_list, suffix=gp2_name)
    average_voxelization_groups(gp2_stacked_voxelizations, directory, gp2_name)

    t_vals, p_vals = stats.ttest_ind(gp1_stacked_voxelizations, gp2_stacked_voxelizations, axis=3, equal_var=False)
    p_vals, t_vals = remove_p_val_nans(p_vals, t_vals)

    colored_p_vals_05 = get_colored_p_vals(p_vals, t_vals, 0.05, ('red', 'green'))
    colored_p_vals_01 = get_colored_p_vals(p_vals, t_vals, 0.01, ('green', 'blue'))
    p_vals_col_f = np.swapaxes(np.maximum(colored_p_vals_05, colored_p_vals_01), 2, 0).astype(np.uint8)  # FIXME: reorientation specific of initial orientation

    output_f_name = f'{prefix}_{gp1_name}_{gp2_name}.tif'
    output_file_path = os.path.join(directory, output_f_name)
    tifffile.imsave(output_file_path, p_vals_col_f, photometric='rgb', imagej=True)
    return p_vals_col_f
