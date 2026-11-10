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

import shutil
import subprocess

from ClearMap.Utils.exceptions import SmiError

NVIDIA_SMI_EXECUTABLE = 'nvidia-smi'
NVIDIA_SMI_TIMEOUT_S = 10
"""How long to wait for ``nvidia-smi``: when the driver is broken it can hang instead of failing."""


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
    output_format = 'noheader,csv'
    if not units:
        output_format += ',nounits'
    output = subprocess.check_output([NVIDIA_SMI_EXECUTABLE, f'--query-gpu={var_name}', f'--format={output_format}'])
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
    cmd = [NVIDIA_SMI_EXECUTABLE, '--query-gpu=memory.used,memory.total,utilization.gpu',
           '--format=noheader,csv,nounits']
    with open(dest_file_path, 'w') as dest_file:  # The child keeps its own copy of the handle
        subprocess.Popen(cmd, stdout=dest_file)


def clear_cuda_cache():
    """Release the video memory which torch keeps cached but no longer uses (e.g. after a large computation)"""
    import torch
    torch.cuda.empty_cache()


def cuda_available() -> bool:
    """Whether torch can use a CUDA device (torch is imported here, so that the callers do not use it directly)"""
    import torch
    return torch.cuda.is_available()


def diagnose_nvidia_driver():
    """
    Check that the NVIDIA driver answers, when the machine has one.

    A machine without ``nvidia-smi`` (no NVIDIA card, a Mac...) is not an error. When the program exists but fails,
    the driver and the kernel module are most likely out of sync, which happens after an update of either until the
    machine is rebooted. The graphics stack (viewers, CUDA) then fails with many unrelated messages.

    Returns
    -------
    str | None
        None if there is no NVIDIA driver to check or if it answers. Otherwise a message for the user which says what
        ``nvidia-smi`` reported and what to do.
    """
    if shutil.which(NVIDIA_SMI_EXECUTABLE) is None:
        return None
    try:
        result = subprocess.run([NVIDIA_SMI_EXECUTABLE, '-L'], capture_output=True, text=True, check=False,
                                timeout=NVIDIA_SMI_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        report = f'did not answer within {NVIDIA_SMI_TIMEOUT_S} s'
    except OSError as err:
        report = f'could not be run ({err})'
    else:
        if result.returncode == 0:
            return None
        output_lines = (result.stdout + result.stderr).strip().splitlines()
        report = f'failed: "{output_lines[0] if output_lines else f"exit status {result.returncode}"}"'
    return (f'{NVIDIA_SMI_EXECUTABLE} {report}. The NVIDIA driver and the kernel module are probably out of sync '
            f'(this happens after an update of the driver or of the kernel). Reboot the machine and start ClearMap '
            f'again. Until then the GPU analyses and the viewers may fail.')
