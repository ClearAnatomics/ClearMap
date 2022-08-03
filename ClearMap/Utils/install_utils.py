"""
install_utils
=========

Utilities module with minimal dependencies (standard library only) for installation
"""
import sys
import subprocess

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


def assert_cuda():
    cmd = 'nvidia-smi -L'
    try:
        subprocess.check_output(cmd, shell=True)
        sys.exit(0)
    except subprocess.CalledProcessError as err:
        print('Error, a working installation of NVidia CUDA could not be detected on your computer.'
              ' This software is required to run ClearMap.')
        sys.exit(err.returncode)


def get_cuda_version():
    cmd = 'nvidia-smi -q -u'
    lines = subprocess.check_output(cmd, shell=True).splitlines()
    lines = [ln.decode('ASCII') for ln in lines]
    cuda_line = [ln for ln in lines if ln.lower().startswith('cuda version')][0]
    return [int(e) for e in cuda_line.split(':')[-1].strip().split('.')]


def get_cuda_line():
    """
    Get the line to put in the environment.yml file for the cuda version
    """
    cuda_version = get_cuda_version()
    error_msg = f'CUDA version {cuda_version} is not supported yet'
    major, minor = cuda_version
    if major == 11:
        if 3 < minor <= 6:
            return '  - cudatoolkit=11.6\n'
        elif minor <= 3:
            return '  - cudatoolkit=11.3\n'
        else:
            raise NotImplementedError(error_msg)
    elif major == 10:
        return '  - cudatoolkit=10.2\n'
    else:
        raise NotImplementedError(error_msg)


def patch_cuda_toolkit_version(cfg_path):
    with open(cfg_path, 'r') as cfg_file:
        cfg = cfg_file.readlines()
    ln_nb, cuda_line = [(i, ln) for i, ln in enumerate(cfg) if 'cudatoolkit' in ln][-1]
    cfg[ln_nb] = get_cuda_line()
    with open(cfg_path, 'w') as cfg_file:
        cfg_file.writelines(cfg)

