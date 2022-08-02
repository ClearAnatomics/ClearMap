import os

from ClearMap.config.config_loader import ConfigLoader


def is_density_file(f_name):  # FIXME: exclude debug
    # return 'density' in f_name and f_name.endswith('.tif')
    return os.path.basename(f_name) == 'density_counts.tif'  # FIXME add menu for alternatives


def find_density_file(target_dir):
    return [os.path.join(target_dir, f) for f in os.listdir(target_dir) if is_density_file(f)][0]


def find_cells_df(target_dir):
    return [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('cells.feather')][0]


def dir_to_sample_id(folder):
    cfg_loader = ConfigLoader(folder)
    return cfg_loader.get_cfg('sample')['sample_id']
