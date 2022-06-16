import os

import numpy as np
from scipy import stats
import tifffile

from ClearMap.Analysis.Statistics.GroupStatistics import remove_p_val_nans
from ClearMap.IO import IO as clearmap_io
from ClearMap.Analysis.Statistics import GroupStatistics as group_statistics
from ClearMap.Utils.utilities import make_abs

colors = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255]
}


def process_condensed(directory, f_list, suffix):  # FIXME: description and better name
    for i, file_name in enumerate(f_list):
        img = clearmap_io.read(make_abs(directory, file_name))
        if i == 0:  # init on first image
            condensed = img[:, :, :, np.newaxis]
        else:
            condensed = np.concatenate((condensed, img[:, :, :, np.newaxis]), axis=3)
    clearmap_io.write(os.path.join(directory, f'condensed_{suffix}.tif'), condensed)

    condensed_avg = np.mean(condensed, axis=3)
    clearmap_io.write(os.path.join(directory, f'avg_{suffix}.tif'), condensed_avg)
    return condensed


def get_colored_p_vals(p_vals, t_vals, significance, color_names):
    p_vals2_f, p_sign_f = get_p_vals_f(p_vals, t_vals, significance)
    return group_statistics.color_p_values(p_vals2_f, p_sign_f,
                                           positive_color=colors[color_names[0]],
                                           negative_color=colors[color_names[1]])


def is_density_file(f_name):
    return 'density' in f_name and f_name.endswith('.tif')


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
    p_vals2 = p_vals.copy()
    p_vals2[p_vals2 > p_cutoff] = p_cutoff  # FIXME: clip
    p_sign = np.sign(t_vals)
    return transpose_p_vals(new_orientation, p_sign, p_vals2)


def transpose_p_vals(new_orientation, p_sign, p_vals2):  # FIXME: check cm_rsp.sagittalToCoronalData
    p_vals2_f = np.transpose(p_vals2, new_orientation)
    p_sign_f = np.transpose(p_sign, new_orientation)
    return p_vals2_f, p_sign_f

def compare_groups(directory, gp1_name, gp2_name, gp1_f_list, gp2_f_list, prefix='p_val_colors'):
    gp1_f_list = dirs_to_density_files(directory, gp1_f_list)
    gp2_f_list = dirs_to_density_files(directory, gp2_f_list)

    gp1_condensed = process_condensed(directory, gp1_f_list, suffix=gp1_name)
    gp2_condensed = process_condensed(directory, gp2_f_list, suffix=gp2_name)

    t_vals, p_vals = stats.ttest_ind(gp1_condensed, gp2_condensed, axis=3, equal_var=False)
    p_vals, t_vals = remove_p_val_nans(p_vals, t_vals)

    colored_p_vals_05 = get_colored_p_vals(p_vals, t_vals, 0.05, ('red', 'green'))
    colored_p_vals_01 = get_colored_p_vals(p_vals, t_vals, 0.01, ('green', 'blue'))
    p_vals_col_f = np.swapaxes(np.maximum(colored_p_vals_05, colored_p_vals_01), 2, 0).astype(np.uint8)  # FIXME: reorientation specific of initial orientation

    output_f_name = f'{prefix}_{gp1_name}_{gp2_name}.tif'
    output_file_path = os.path.join(directory, output_f_name)
    tifffile.imsave(output_file_path, p_vals_col_f, photometric='rgb', imagej=True)
    return p_vals_col_f
