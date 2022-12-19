# -*- coding: utf-8 -*-
"""
utilities
=========

Various utilities that do not have a specific category
"""

import os
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from operator import getitem

import numpy as np
import psutil

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

from ClearMap.Utils.TagExpression import Expression
from ClearMap.Utils.exceptions import MissingRequirementException

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
    return subprocess.check_output(cmd, shell=True)


def get_free_v_ram():
    return int(smi_query('memory.free'))


def get_percent_v_ram_use():
    percent = float(smi_query('memory.used')) / float(smi_query('memory.total'))
    return percent * 100


def gpu_util():
    gpu_percent = smi_query('utilization.gpu', units=True).decode('ascii').replace('%', '').strip()
    return int(gpu_percent)


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
        iterator = iter(obj)
        return True
    except TypeError:
        return False


def title_to_snake(string):
    return re.sub('(?!^)([A-Z]+)', r'_\1', string).lower()


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
    return reduce(getitem, keys, container)


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


def requires_files(file_paths):
    def decorator(func):
        def wraps(*args, **kwargs):
            instance = args[0]
            workspace = instance.workspace
            for f_p in file_paths:
                f_path = workspace.filename(f_p.base, prefix=f_p.prefix, postfix=f_p.postfix, extension=f_p.extension)
                msg = f'{type(instance).__name__}.{func.__name__} missing path: "{f_path}"'
                if Expression(f_path).tags:
                    file_list = workspace.file_list(f_p.base, prefix=f_p.prefix, postfix=f_p.postfix,
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


# FIXME: move to io
class FilePath:
    def __init__(self, base, prefix=None, postfix=None, extension=None):
        self.base = base
        self.prefix = prefix
        self.postfix = postfix
        self.extension = extension
