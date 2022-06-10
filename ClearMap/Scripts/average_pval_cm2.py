import os
import numpy as np
from scipy import stats
import tifffile

from ClearMap.IO import IO as clearmap_io

colors = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255]
}


def process_condensed(directory, f_list, suffix):
    for i, file_name in enumerate(f_list):
        if os.path.isabs(file_name):
            f_path = file_name
        else:
            f_path = os.path.join(directory, file_name)
        img = clearmap_io.read(f_path)
        if i == 0:  # init on first image
            condensed = img[:, :, :, np.newaxis]
        else:
            condensed = np.concatenate((condensed, img[:, :, :, np.newaxis]), axis=3)
    clearmap_io.write(os.path.join(directory, f'condensed_{suffix}.tif'), condensed)

    condensed_avg = np.mean(condensed, axis=3)
    clearmap_io.write(os.path.join(directory, f'avg_{suffix}.tif'), condensed_avg)
    return condensed


def color_p_values(p_vals, p_sign, positive_color=[1, 0], negative_color=[0, 1], p_cutoff=None,
                   positive_trend=[0, 0, 1, 0], negative_trend=[0, 0, 0, 1], p_max=None):
    """

    Parameters
    ----------
    p_vals np.array
    p_sign np.array
    positive list
    negative list
    pcutoff float
    positivetrend list
    negativetrend list
    pmax float

    Returns
    -------

    """
    if p_max is None:
        p_max = p_vals.max()
    p_vals_inv = p_max - p_vals
    
    if p_cutoff is None:  # color given p values
        if len(positive_color) != len(negative_color):
            raise ValueError(f'Length of positive and negative colors do not match, '
                             f'got {len(positive_color)} and {len(negative_color)}')
        d = len(positive_color)   # 3D
        output_shape = p_vals.shape + (d,)  # 3D + color
        colored_p_vals = np.zeros(output_shape)

        # color
        for sign, col in ((1, positive_color), (-1, negative_color)):
            ids = sign*p_sign > 0  # positive of inverse for negative
            p_vals_i = p_vals_inv[ids]
            for i in range(len(col)):
                colored_p_vals[ids, i] = p_vals_i * col[i]
    else:  # split p_values according to cutoff
        if any([len(positive_color) != len(v) for v in (negative_color, positive_trend, negative_trend)]):
            raise ValueError('color_p_values: positive, negative, positive_trend and '
                             'negative_trend option must be equal length !')
        output_shape = p_vals.shape + (len(positive_color),)
        colored_p_vals = np.zeros(output_shape)

        idc = p_vals < p_cutoff
        ids = p_sign > 0
        # significant positive, non sig positive, sig neg, non sig neg
        for id_sign, idc_sign, w in ((1, 1, positive_color), (1, -1, positive_trend),
                                     (-1, 1, negative_color), (-1, -1, negative_trend)):
            ii = np.logical_and(id_sign*ids, idc_sign*idc)
            p_vals_i = p_vals_inv[ii]
            for i in range(len(w)):
                colored_p_vals[ii, i] = p_vals_i * w[i]

    return colored_p_vals


def get_pvals_f(pvals, tvals, pcutoff):
    pvals2 = pvals.copy()
    pvals2[pvals2 > pcutoff] = pcutoff
    psign = np.sign(tvals)
    # from sagital to coronal view
    _pvals2_f = np.transpose(pvals2, (2, 0, 1))
    _psign_f = np.transpose(psign, (2, 0, 1))
    return _pvals2_f, _psign_f


def get_colored_p_vals(p_vals, t_vals, significance, positive_color, negative_color):
    pvals2_f, psign_f = get_pvals_f(p_vals, t_vals, significance)
    return color_p_values(pvals2_f, psign_f, positive_color=positive_color, negative_color=negative_color)


def is_density_file(f_name):
    return 'density' in f_name and f_name.endswith('.tif')


def find_density_file(target_dir):
    return [os.path.join(target_dir, f) for f in os.listdir(target_dir) if is_density_file(f)][0]


def dirs_to_density_files(directory, f_list):
    out = []
    for i, f_name in enumerate(f_list):
        if not os.path.isabs(f_name):
            f_name = os.path.join(directory, f_name)
        if not is_density_file(f_name):
            f_name = find_density_file(f_name)
        out.append(f_name)
    return out


def compare_groups(directory, gp1_name, gp2_name, gp1_f_list, gp2_f_list,
                   prefix='p_val_colors'):

    gp1_f_list = dirs_to_density_files(directory, gp1_f_list)
    gp2_f_list = dirs_to_density_files(directory, gp2_f_list)

    gp1_condensed = process_condensed(directory, gp1_f_list, suffix=gp1_name)
    gp2_condensed = process_condensed(directory, gp2_f_list, suffix=gp2_name)

    t_vals, p_vals = stats.ttest_ind(gp1_condensed, gp2_condensed,
                                     axis=3, equal_var=False)

    invalid_idx = np.isnan(p_vals)
    p_vals[invalid_idx] = 1.0
    t_vals[invalid_idx] = 0

    colored_p_vals_05 = get_colored_p_vals(p_vals, t_vals, 0.05,
                                           positive_color=colors['red'], negative_color=colors['green'])
    colored_p_vals_01 = get_colored_p_vals(p_vals, t_vals, 0.01,
                                           positive_color=colors['green'], negative_color=colors['blue'])
    pvalscol_f = np.swapaxes(np.maximum(colored_p_vals_05, colored_p_vals_01), 2, 0).astype(np.uint8)

    output_f_name = f'{prefix}_{gp1_name}_{gp2_name}.tif'
    output_file_path = os.path.join(directory, output_f_name)
    tifffile.imsave(output_file_path, pvalscol_f, photometric='rgb', imagej=True)
    return pvalscol_f


if __name__ == '__main__':
    _gp1_file_list = [
        'cont/10w_density_branches.tif',
        'cont/12w_density_branches.tif',
        'cont/13w_density_branches.tif'
    ]
    _gp2_file_list = [
        'dep/2k_density_branches.tif',
        'dep/3k_density_branches.tif',
        'dep/6k_density_branches.tif'
    ]
    compare_groups('/data/vox_cont_dep', 'control', 'deprived', _gp1_file_list, _gp2_file_list)
