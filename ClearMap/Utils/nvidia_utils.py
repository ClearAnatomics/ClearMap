# -*- coding: utf-8 -*-
"""
nvidia_utils
============

Everything graphics card related: the queries to the driver through ``nvidia-smi`` (free and used video memory,
utilisation) and the calls to torch/CUDA (availability, clearing the cache)."""

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import subprocess

from ClearMap.Utils.exceptions import SmiError


def smi_query(var_name, units=False):
    """
    Query one property of the GPU with ``nvidia-smi``.

    Parameters
    ----------
    var_name : str
        The property, as ``nvidia-smi --query-gpu`` names it (e.g. ``'memory.free'``, ``'utilization.gpu'``).
    units : bool
        Whether to keep the units in the output (e.g. ``'MiB'``, ``'%'``). Without them, only the number is returned.

    Returns
    -------
    bytes
        The raw output of ``nvidia-smi``: one line per GPU, without header.

    Raises
    ------
    SmiError
        If the driver and the library versions do not match (usually after a driver update, until a reboot).
    subprocess.CalledProcessError
        If ``nvidia-smi`` fails (e.g. there is no NVIDIA GPU).
    """
    cmd = f'nvidia-smi --query-gpu={var_name} --format=noheader,csv'
    if not units:
        cmd += ',nounits'
    output = subprocess.check_output(cmd, shell=True)
    if "Failed to initialize NVML: Driver/library version mismatch" in output.decode('ascii'):
        raise SmiError(output)
    return output


def get_free_v_ram():
    """
    The free video memory of the GPU.

    Returns
    -------
    int
        The free memory in MiB. Assumes a single GPU (``int`` fails on the several lines of a multi-GPU machine).
    """
    return int(smi_query('memory.free'))


def get_percent_v_ram_use():
    """
    The share of the video memory of the GPU which is in use.

    Returns
    -------
    float
        The used memory as a percentage of the total memory. Assumes a single GPU.
    """
    percent = float(smi_query('memory.used')) / float(smi_query('memory.total'))
    return percent * 100


def gpu_util():
    """
    The utilisation of the GPU.

    Returns
    -------
    int
        The percentage of the time over the last sample period during which a kernel was running on the GPU.
        Assumes a single GPU.
    """
    gpu_percent = smi_query('utilization.gpu', units=True).decode('ascii').replace('%', '').strip()
    return int(gpu_percent)


def gpu_params(dest_file_path):  # Uses 1 query instead of 3 because too time-consuming
    """
    Write the used memory, total memory and utilisation of the GPU to a file, without waiting for ``nvidia-smi``.

    The three values come from a single query (instead of one per value, as the functions above do) because each
    call to ``nvidia-smi`` is slow. The command runs in the background: the file is written once it finishes, and
    a failure of ``nvidia-smi`` is not reported.

    Parameters
    ----------
    dest_file_path : str | Path
        The file to write to: one line ``used (MiB), total (MiB), utilisation (%)`` per GPU, without units.
    """
    cmd = 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=noheader,csv,nounits'
    cmd += f' > {dest_file_path}'
    subprocess.Popen(cmd, shell=True)


def clear_cuda_cache():
    """Release the video memory which torch keeps cached but no longer uses (e.g. after a large computation)"""
    import torch
    torch.cuda.empty_cache()


def cuda_available() -> bool:
    """Whether torch can use a CUDA device (torch is imported here, so that the callers do not use it directly)"""
    import torch
    return torch.cuda.is_available()
