# -*- coding: utf-8 -*-
"""
utilities
=========

Various utilities that do not have a specific category
"""

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import inspect
import multiprocessing
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
import functools
from operator import getitem
from types import MappingProxyType
from typing import List

import numpy as np
import psutil

from configobj import ConfigObj


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


def title_to_snake(string):
    """
        Convert a human/TitleCase/mixed string into canonical snake_case.

        Rules:
        - Inserts underscores at CamelCase boundaries (e.g., "TubeMap" -> "tube_map").
        - Keeps acronyms together (e.g., "HTTPServer" -> "http_server", "ROI3D" -> "roi3d").
        - Normalizes spaces, hyphens, slashes, and other punctuation to single underscores.
        - Collapses multiple underscores and trims leading/trailing underscores.
        - Lowercases the final result.

        Parameters
        ----------
        s : str
            Input string (e.g., tab title, pipeline name, or file-ish label).

        Returns
        -------
        str
            Snake_case version of the input. Empty string if input is None/empty.

        Examples
        --------
        >>> title_to_snake("TubeMap")
        'tube_map'
        >>> title_to_snake("Sample Info")
        'sample_info'
        >>> title_to_snake("Tract-Map")
        'tract_map'
        >>> title_to_snake("ROI3D")
        'roi3_d'  # FIXME: ideally 'roi3d' but hard to do robustly
        >>> title_to_snake("HTTPServerError")
        'http_server_error'
        """
    if not string:
        return ''

    string = string.strip()

    # Insert underscore between:
    # 1) a lowercase/digit and an uppercase letter: "version2File" -> "version2_File"
    string = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', string)

    # 2) acronym and the next CamelCase word: "HTTPServer" -> "HTTP_Server"
    #    (one or more capitals) followed by (Capital + lowercase)
    string = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', string)

    # Normalize any non-alphanumeric runs to underscores (keeps underscores too)
    string = re.sub(r'[^0-9A-Za-z]+', '_', string)

    # Collapse repeated underscores and trim
    string = re.sub(r'_+', '_', string).strip('_')

    return string.lower()
    # out = re.sub('(?!^)([A-Z]+)', r'_\1', string).lower()
    # out = out.replace(' ', '_')
    # return out

def snake_to_title(string):
    return string.replace('_', ' ').title()


def del_item_recursive(d: dict, keys: list) -> None:
    """Delete a nested key from a dict. No-op if path does not exist."""
    for key in keys[:-1]:
        d = d.get(key)
        if not isinstance(d, dict):
            return
    d.pop(keys[-1], None)


def get_item_recursive(container, keys):
    try:
        return functools.reduce(getitem, keys, container)
    except (TypeError, KeyError) as err:
        err_type = type(err).__name__
        err_msg = f'{err_type} attempting to read "{keys}"'
        if isinstance(container, ConfigObj):
            err_msg += f' path: {container.filename}'
        print(f'{err_msg} from "{container}"')
        raise err


def try_get_item_recursive(container, keys, default=Ellipsis):
    try:
        return get_item_recursive(container, keys)
    except (TypeError, KeyError):
        if default is Ellipsis:
            raise
        return default


def has_item_recursive(container, keys) -> bool:
    try:
        get_item_recursive(container, keys)
        return True
    except (TypeError, KeyError):
        return False


def set_item_recursive(dictionary, keys_list, val, fix_missing_keys=True):
    def add_keys(d, keys):
        if not keys:
            return
        if keys[0] not in d.keys():
            d[keys[0]] = {}
        if keys[1:]:  # if keys left
            if not isinstance(d[keys[0]], dict):
                # Avoid creating a sub-dict if there is already a non-dict value there
                raise TypeError(f"Non-dict at {keys[0]} while descending {keys}")
            add_keys(d[keys[0]], keys[1:])

    if fix_missing_keys:
        add_keys(dictionary, keys_list[:-1])  # Fix missing keys recursively
    else: # ensure the path exists and is all dicts
        cur = dictionary
        for k in keys_list[:-1]:
            cur = cur[k]  # KeyError if missing
            if not isinstance(cur, dict):
                raise TypeError(f"Non-dict at {k} while descending {keys_list}")

    # Since we ensured it exists and is a dict, write to var will modify in place
    get_item_recursive(dictionary, keys_list[:-1])[keys_list[-1]] = val
    return dictionary  # Return the modified dictionary for chaining


def deep_freeze(obj) -> object | MappingProxyType:
    """
    Recursively make a data structure immutable (freeze it).
    Only works for dict, list, set, tuple and scalars (str/int/float/None/...).

    .. note::
        This is faster than deepcopy and more explicit because
        if we try to modify a frozen object, we get an error immediately.

    Parameters
    ----------
    obj: object
        Typically a nested dict like a config

    Returns
    -------
    object | MappingProxyType
        An immutable version of the input object
    """
    if isinstance(obj, dict):
        # freeze children first, then wrap the dict so callers cannot mutate it
        frozen = {k: deep_freeze(v) for k, v in obj.items()}
        return MappingProxyType(frozen)
    elif isinstance(obj, list):
        return tuple(deep_freeze(x) for x in obj)
    elif isinstance(obj, set):
        return frozenset(deep_freeze(x) for x in obj)
    elif isinstance(obj, tuple):
        return tuple(deep_freeze(x) for x in obj)
    # scalars (str/int/float/None/...) pass through because in python they are already immutable
    return obj


def _ensure_list(obj) -> list:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]

def _dedupe_preserve_order(seq, key=lambda x: x):
    seen = set()
    out = []
    for x in seq:
        k = key(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def bytes_to_human(num):
    """
    Convert bytes to human-readable format.

    Parameters
    ----------
    num : int
        Number of bytes

    Returns
    -------
    str
        Human-readable format of the number of bytes
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    else:
        warnings.warn(f'Unable to convert {num} bytes to human-readable format. ')
        return f"{num:.2f} bytes"


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


def sanitize_n_processes(processes):
    if processes is None:
        processes = multiprocessing.cpu_count()
    elif isinstance(processes, str):
        warnings.warn(f'Using a string to specify the number of processes is deprecated. '
                      f'Please use an integer (positive or negative) or None.', DeprecationWarning)
        if processes.lower() == 'serial':
            processes = 1
        elif processes.lower() == '!serial':
            processes = multiprocessing.cpu_count() - 1
        else:
            raise ValueError(f'Unknown string value for processes: {processes}. '
                             f'Use None, "serial" or an integer.')
    if isinstance(processes, int):
        if processes < 0:
            processes = multiprocessing.cpu_count() + processes
        processes = max(1, processes)
        return processes
    else:
        raise ValueError(f'Processes must be an integer or None, got {processes} of type {type(processes)}')


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


def infer_origin_from_caller() -> str:
    """
    Infer an **origin** string from the caller's module, class and function name.
    This is typically the method that is mutating the config.

    Returns
    -------
    str
        The inferred origin string in the format "module.class.function" or "module.function" if
    """
    frame = inspect.stack()[1]
    mod = frame.frame.f_globals.get('__name__', '')
    func = frame.function
    cls = type(frame.frame.f_locals['self']).__name__ if 'self' in frame.frame.f_locals else ''
    origin = f"{mod}.{cls + '.' if cls else ''}{func}"
    return origin


def trim_or_pad(lst: List, target_len: int, pad_value=0):
    """
    Strip or pad a list to a target length.

    Parameters
    ----------
    lst: list
        The list to strip or pad
    target_len: int
        The target length
    pad_value: any
        The value to use for padding

    Returns
    -------
    list
        The stripped or padded list
    """
    if len(lst) > target_len:
        return lst[:target_len]
    elif len(lst) < target_len:
        return lst + [pad_value] * (target_len - len(lst))
    else:
        return lst
