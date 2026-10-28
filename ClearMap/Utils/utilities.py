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
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
import functools
from typing import List

import numpy as np
import psutil

from ClearMap.Utils.exceptions import ClearMapTypeError, ClearMapValueError


# The environment variables with which Slurm tells a job how many CPUs it may use (the cores of the node are not its own)
CPU_LIMIT_ENV_VARS = ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE')
PROCESSES_EXPECTED = 'None (all the CPUs), n, -n (all but n) or 0 (in the calling process)'

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


def available_cpu_count():
    """
    The number of CPUs this process may use, which is not the number of CPUs of the machine on a cluster (or in
    a container): ``multiprocessing.cpu_count()`` gives the cores of the node, and a job that starts that many
    processes on the few cores Slurm gave it fights with itself (and with the jobs of the others).

    This is the smallest of what each source says:

    * the CPUs the process is allowed to run on (``taskset``, the cpuset of a Slurm job with the cgroup plugin),
      or the CPUs of the machine where the system does not say (macOS, Windows)
    * the ``SLURM_CPUS_PER_TASK`` and ``SLURM_CPUS_ON_NODE`` of the job, which are all there is where Slurm
      does not restrict the CPUs the process can run on. A value that is not a positive integer is ignored.

    Returns
    -------
    int
        At least 1: every limit is, by construction.
    """
    return min([cpus_the_process_may_run_on(), *slurm_cpu_limits()])


def cpus_the_process_may_run_on():
    """The CPUs taskset, or Slurm with its affinity or cgroup plugin, lets this process run on"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # No such function on macOS and Windows: the CPUs of the machine
        return os.cpu_count() or 1


def slurm_cpu_limits():
    """What SLURM_CPUS_PER_TASK and SLURM_CPUS_ON_NODE say, where they are positive integers"""
    values = (os.environ.get(name, '').strip() for name in CPU_LIMIT_ENV_VARS)
    return [int(value) for value in values if value.isdecimal() and int(value) > 0]


def sanitize_n_processes(processes, can_run_in_calling_process=False):
    """
    The number of processes to use: None for all the CPUs of the job (:func:`available_cpu_count`), -n for all
    but n (at least 1) and n for n, but never more than the CPUs of the job: the number of a configuration written
    for a larger machine does not make a small job fight with itself.

    0 is no subprocess: the calling process does the work, one item after the other (to debug). It stays 0 for
    the code that can run that way (``can_run_in_calling_process``) and is 1 for the code that only needs a count
    (the threads of compiled code...). The former 'serial' is 0 and '!serial' -1, but they are deprecated.

    Raises
    ------
    ClearMapValueError
        If processes is a string other than the former ones.
    ClearMapTypeError
        If processes is neither None, an integer nor a string.
    """
    if isinstance(processes, str):
        warnings.warn('Using a string to specify the number of processes is deprecated. Please use an integer '
                      '(positive, negative, or 0 for the calling process) or None.', DeprecationWarning, stacklevel=2)
        if processes.lower() == 'serial':
            processes = 0
        elif processes.lower() == '!serial':
            processes = -1
        else:
            raise ClearMapValueError(f'Unknown string value for processes: {processes!r}', value=processes,
                                     expected=PROCESSES_EXPECTED)
    if processes is None:
        return available_cpu_count()
    if not isinstance(processes, int):
        raise ClearMapTypeError(f'processes must be None or an integer ({PROCESSES_EXPECTED}), got {processes!r} '
                                f'of type {type(processes).__name__}')
    if processes == 0:
        return 0 if can_run_in_calling_process else 1
    available = available_cpu_count()
    requested = available + processes if processes < 0 else processes
    return min(max(1, requested), available)  # A number relative to the CPUs never means the calling process


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
