"""
Utility functions for working with paths
"""
import os
from pathlib import Path

from ClearMap.IO.assets_constants import EXTENSIONS
from ClearMap.config.config_loader import ConfigLoader


def is_feather(f):
    """
    Check if a file is a feather file

    .. warning::
    Currently, this function only checks if the file ends with 'cells.feather'

    Parameters
    ----------
    f : str or Path
        The file to check

    Returns
    -------

    """
    if isinstance(f, Path):
        f = str(f)
    return f.endswith('cells.feather')  # FIXME: check does not match name


def is_density_file(f_name):
    f_name = os.path.basename(f_name)
    if 'debug' in f_name:
        return False
    # return 'density' in f_name and f_name.endswith('.tif')
    return f_name.endswith('density_counts.tif')  # FIXME add menu for alternatives


def find_density_file(target_dir, channel, suffix=''):
    target_dir = Path(target_dir)
    extensions = [ext[1:] for ext in EXTENSIONS['image']]
    pattern = f'{channel}_density*{suffix}.'
    files = []
    for ext in extensions:
        files.extend(target_dir.glob(pattern + ext))
    return files[0] if files else None
    # return find_file(target_dir, is_density_file, 'density')


def find_cells_df(target_dir):
    """
    Find the first feather file in ``target_dir``

    Parameters
    ----------
    target_dir

    Returns
    -------

    """
    return find_file(target_dir, is_feather, 'feather')


def find_file(target_dir, check_func, file_type_name):
    """
    Find the first file corresponding to ``check_func`` in ``target_dir``

    Parameters
    ----------
    target_dir : str
        The directory to search
    check_func : callable
        A function that takes a file name and returns a boolean
    file_type_name : str
        The name of the file type to search for. This is used for error messages

    Returns
    -------

    """
    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if check_func(f)]
    try:
        return files[0]
    except IndexError:
        raise RuntimeError(f'No {file_type_name} file found in {target_dir}')


def dir_to_sample_id(folder):
    """
    Get the sample ID from a directory

    Parameters
    ----------
    folder : str
        The directory to check

    Returns
    -------

    """
    cfg_loader = ConfigLoader(folder)
    return cfg_loader.get_cfg('sample')['sample_id']
