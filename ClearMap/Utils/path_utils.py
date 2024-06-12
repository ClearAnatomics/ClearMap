"""
Utility functions for working with paths
"""
import os

from ClearMap.config.config_loader import ConfigLoader


def is_feather(f):
    return f.endswith('cells.feather')


def is_density_file(f_name):
    f_name = os.path.basename(f_name)
    if 'debug' in f_name:
        return False
    # return 'density' in f_name and f_name.endswith('.tif')
    return f_name.endswith('density_counts.tif')  # FIXME add menu for alternatives


def find_density_file(target_dir):
    return find_file(target_dir, is_density_file, 'density')


def find_cells_df(target_dir):
    return find_file(target_dir, is_feather, 'feather')


def find_file(target_dir, check_func, file_type_name):
    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if check_func(f)]
    try:
        return files[0]
    except IndexError:
        raise RuntimeError(f'No {file_type_name} file found in {target_dir}')


def dir_to_sample_id(folder):
    cfg_loader = ConfigLoader(folder)
    return cfg_loader.get_cfg('sample')['sample_id']
