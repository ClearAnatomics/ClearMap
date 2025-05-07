# -*- coding: utf-8 -*-
"""
utilities
=========

Various utilities that do not have a specific category
"""
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from operator import getitem

import numpy as np
import psutil

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import functools

from configobj import ConfigObj

from ClearMap.Utils.tag_expression import Expression
from ClearMap.Utils.exceptions import MissingRequirementException, SmiError, ParamsOrientationError


DEFAULT_ORIENTATION = (0, 0, 0)


colors = {
    "WHITE": '\033[1;37m',
    "GREEN": '\033[0;32m',
    "YELLOW": '\033[1;33;48m',
    "RED": '\033[1;31;48m',
    "BLINK": '\33[5m',
    "BLINK2": '\33[6m',
    "RESET": '\033[1;37;0m'
}


def colorize(msg, color):
    color = color.upper()
    color = colors[color]
    return "{color}{msg}{reset_color}".format(color=color, msg=msg, reset_color=colors["RESET"])


def runs_on_spyder():
    return any('SPYDER' in name for name in os.environ)


def runs_on_pycharm():
    return "PYCHARM_HOSTED" in os.environ


def runs_on_ui():
    return 'CLEARMAP_GUI_HOSTED' in os.environ


def smi_query(var_name, units=False):
    cmd = f'nvidia-smi --query-gpu={var_name} --format=noheader,csv'
    if not units:
        cmd += ',nounits'
    output = subprocess.check_output(cmd, shell=True)
    if "Failed to initialize NVML: Driver/library version mismatch" in output.decode('ascii'):
        raise SmiError(output)
    return output


def get_free_v_ram():
    return int(smi_query('memory.free'))


def get_percent_v_ram_use():
    percent = float(smi_query('memory.used')) / float(smi_query('memory.total'))
    return percent * 100


def gpu_util():
    gpu_percent = smi_query('utilization.gpu', units=True).decode('ascii').replace('%', '').strip()
    return int(gpu_percent)


def gpu_params(dest_file_path):  # Uses 1 query instead of 3 because too time-consuming
    cmd = 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=noheader,csv,nounits'
    cmd += f' > {dest_file_path}'
    subprocess.Popen(cmd, shell=True)


class CancelableProcessPoolExecutor(ProcessPoolExecutor):
    def immediate_shutdown(self):
        with self._shutdown_lock:
            self._shutdown_thread = True
            # statuses = [psutil.Process(_proc.pid).status() for _proc in self._processes.values()]
            terminated_procs = 0
            for proc in self._processes.values():
                status = psutil.Process(proc.pid).status()
                if status == 'sleeping':
                    proc.terminate()
                    terminated_procs += 1
            if not terminated_procs:
                for proc in self._processes.values():
                    proc.terminate()


def is_in_range(src_array, value_range):
    return np.logical_and(src_array >= value_range[0], src_array <= value_range[1])


def is_iterable(obj):
    try:
        _ = iter(obj)
        return True
    except TypeError:
        return False


def title_to_snake(string):
    out = re.sub('(?!^)([A-Z]+)', r'_\1', string).lower()
    out = out.replace(' ', '_')
    return out

def snake_to_title(string):
    return string.replace('_', ' ').title()


def backup_file(file_path):  # REFACTOR: put in workspace or IO
    base_path, ext = os.path.splitext(file_path)
    new_path = base_path + '.bcp' + ext
    shutil.copy(file_path, new_path)


def make_abs(directory, file_name):
    """Make file_name absolute if it is not"""
    if os.path.isabs(file_name):
        f_path = file_name
    else:
        f_path = os.path.join(directory, file_name)
    return f_path


def get_item_recursive(container, keys):
    try:
        return reduce(getitem, keys, container)
    except (TypeError, KeyError) as err:
        err_type = type(err).__name__
        err_msg = f'{err_type} attempting to read "{keys}"'
        if isinstance(container, ConfigObj):
            err_msg += f' path: {container.filename}'
        print(f'{err_msg} from "{container}"')
        raise err


def set_item_recursive(dictionary, keys_list, val, fix_missing_keys=True):
    def add_keys(d, keys):
        if not keys:
            return
        if keys[0] not in d.keys():
            d[keys[0]] = {}
        if keys[1:]:  # if keys left
            add_keys(d[keys[0]], keys[1:])

    if fix_missing_keys:
        add_keys(dictionary, keys_list[:-1])  # Fix missing keys recursively
    get_item_recursive(dictionary, keys_list[:-1])[keys_list[-1]] = val


def requires_assets(asset_specs):
    """
    Decorator to check if the required files exist before running the function.
    For the case of channel, it can be extracted (in this order) from
    - the kwargs
    - the FilePath object
    - the instance to which the wrapped function belongs

    Parameters
    ----------
    asset_specs: List[FilePath]
        List of FilePath objects
    """
    def decorator(func):
        def wraps(*args, **kwargs):
            instance = args[0]
            workspace = instance.workspace
            for f_p in asset_specs:
                channel = kwargs.get('channel')  # Get directly from the kwargs
                if channel is None:
                    if hasattr(f_p, 'channel'):  # Get from FilePath object
                        channel = f_p.channel
                    else:  # Get the global value from the instance
                        channel = getattr(instance, 'channel', None)
                f_path = workspace.filename(f_p.base, channel=channel, sample_id=f_p.prefix, asset_sub_type=f_p.postfix,
                                            extension=f_p.extension)
                msg = f'{type(instance).__name__}.{func.__name__} missing path: "{f_path}"'
                if Expression(f_path).tags:
                    file_list = workspace.file_list(f_p.base, sample_id=f_p.prefix, asset_sub_type=f_p.postfix,
                                                    extension=f_p.extension)
                    if not file_list:
                        raise MissingRequirementException(msg + ' Pattern but no file')
                    for f in file_list:
                        if not os.path.exists(f):
                            raise MissingRequirementException(msg + f' Missing tile {f}')
                elif not os.path.exists(f_path):
                    raise MissingRequirementException(msg)
            return func(*args, **kwargs)
        return wraps
    return decorator


def check_enough_temp_space(min_temp_space=200):
    free = get_free_temp_space()
    return free // (2**30) > min_temp_space  # GB


def get_free_temp_space():
    _, _, free = shutil.disk_usage(tempfile.gettempdir())
    return free


# FIXME: move to io and rename to smth similar to AssetSpec (without conflicting with the existing AssetSpec)
class FilePath:
    def __init__(self, base, prefix=None, postfix=None, extension=None, asset_sub_type=None):
        self.base = base
        self.prefix = prefix
        self.postfix = asset_sub_type or postfix
        self.extension = extension


def handle_deprecated_args(deprecated_args_map):  # FIXME: add a version_changed argument
    """
    Decorator to handle deprecated arguments by renaming them.
    It takes a dictionary and renames old arguments to new ones.

    Parameters
    ----------
    deprecated_args_map : dict
        Dictionary mapping old argument names to new ones.

    Returns
    -------

    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old_arg, new_arg in deprecated_args_map.items():
                if old_arg in kwargs:
                    warnings.warn(f"The '{old_arg}' argument is deprecated, use '{new_arg}' instead.",
                                  DeprecationWarning)
                    kwargs[new_arg] = kwargs.pop(old_arg)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def substitute_deprecated_arg(old_arg, new_arg, old_arg_name, new_arg_name):
    if new_arg is not None:
        raise ValueError(f'Cannot use both {old_arg_name} and {new_arg_name} arguments.')
    else:
        warnings.warn(f'The {old_arg_name} argument is deprecated. Use {new_arg_name} instead.',
                      DeprecationWarning, stacklevel=2)
        return old_arg


def validate_arg(arg_name, value, valid_values):
    """
    Check if the value is in the list of valid values and raise a ValueError if not.

    Parameters
    ----------
    arg_name
    value
    valid_values

    Returns
    -------
    value if it is in the list of valid values,
    otherwise raises a ValueError
    """
    if value not in valid_values:
        raise ValueError(f'Unknown {arg_name} "{value}". '
                         f'Supported values are "{valid_values}".')
    return value


def clear_cuda_cache():
    import torch
    torch.cuda.empty_cache()


# def topological_sort(graph, in_degree):
#     queue = deque([node for node in in_degree if in_degree[node] == 0])
#     sorted_list = []
#
#     while queue:
#         node = queue.popleft()
#         sorted_list.append(node)
#
#         for neighbor in graph[node]:
#             in_degree[neighbor] -= 1
#             if in_degree[neighbor] == 0:
#                 queue.append(neighbor)
#
#     if len(sorted_list) != len(in_degree):
#         raise ValueError("Cycle detected in channel dependencies")
#
#     return sorted_list


def check_stopped(func):
    """
    Decorator to check if the object is stopped (self.stopped) before running the function.
    If the object is stopped, the function returns immediately.
    The force argument can be used to force the function to run even if the object is stopped
    and reset the stopped flag.
    """
    def wrapper(self, *args, _force=False, **kwargs):
        if _force:
            self.stopped = False
        if self.stopped:
            return
        return func(self, *args, **kwargs)
    return wrapper


def validate_orientation(orientation, channel, raise_error=True):
    """
    Check that the orientation does not have redundant axes

    Parameters
    ----------
    orientation: tuple(int)
        The orientation to check
    """
    defined_axes = [abs(e) for e in orientation if e != 0]
    if len(defined_axes) != len(set(defined_axes)):
        if raise_error:
            raise ParamsOrientationError(f'Number of different defined axes in {orientation} is'
                                         f'{len(set(defined_axes))}. Defined axes cannot be duplicated. '
                                         f'Please amend duplicate axes.', channel=channel)
        else:
            warnings.warn(f'Invalid orientation {orientation} for {channel},'
                          f' using default {DEFAULT_ORIENTATION}')
            return DEFAULT_ORIENTATION
    return orientation


def sanitize_n_processes(processes):
    if processes < 0:
        processes = multiprocessing.cpu_count() + processes
    processes = max(processes, 1)
    return processes


def get_ok_n_ok_symbols():
    """
    1) Detect whether we can print✓/✗ in this terminal
    2) otherwise, use [OK]/[FAIL] instead

    Returns
    -------
    tuple
        ok_symbol, fail_symbol
    """
    enc = sys.stdout.encoding or ''
    try:
        '✓'.encode(enc)
        return '✓', '✗'
    except (UnicodeEncodeError, TypeError):
        return '[OK]', '[FAIL]'
