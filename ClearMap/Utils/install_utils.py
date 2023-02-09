"""
install_utils
=============

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

# # cmd = "conda search -c pytorch -c nvidia -c conda-forge pytorch=1.11.0=py3.7_cuda11.5_cudnn8.3.2_0 --info | grep cudatoolkit"
# def cuda_to_cuda_toolkit_version(cfg_path):   # FIXME: use from pytorch
#     """
#     Get the line to put in the environment.yml file for the cuda version
#     """
#     cuda_version = get_cuda_version()
#     error_msg = f'CUDA version {cuda_version} is not supported yet'
#     major, minor = cuda_version
#     if major == 12:
#         return '11.6'
#     elif major == 11:
#         if minor <= 3:
#             return '11.3'
#         else:
#             return '11.6'
#     elif major == 10:
#         return '10.2'
#     else:
#         raise NotImplementedError(error_msg)


def pytorch_to_cuda_toolkit_version(cfg_path):
    pytorch_version_str = get_pytorch_version_string(cfg_path)
    cmd = f'conda search -c pytorch -c nvidia -c conda-forge pytorch={pytorch_version_str} --info | grep cudatoolkit'
    cuda_info_ln = subprocess.check_output(cmd, shell=True)
    toolkit_version = cuda_info_ln.decode('ascii').split(',')[0].split('=')[-1]
    return toolkit_version


def get_best_match_cuda_version(versions, cuda_version=None):
    if cuda_version is None:
        cuda_version = get_cuda_version()
    major, minor = cuda_version
    for v in versions[::-1]:
        print(f'CUDA version: {cuda_version}, candidate version: {v}')
        v_major, v_minor = v
        if v_major > major:
            raise ValueError(f'Version not supported: {v}')
        else:
            if v_major < major or v_minor <= minor:
                return v
    else:
        raise ValueError(f'No match found between installed CUDA {cuda_version} and candidates {versions}')


def get_pytorch_builds_cuda_versions(python_version='3.7', pytorch_version='1.11', cuda_major_version='11'):
    builds = get_pytorch_build_strings(python_version, pytorch_version, cuda_major_version)
    cuda_versions = [[e.replace('cuda', '') for e in ln.split('_') if 'cuda' in e][0] for ln in builds]
    return [[int(e) for e in v.split('.')] for v in sorted(cuda_versions)]  # TODO: see for natsorted


def get_pytorch_build_strings(python_version='3.7', pytorch_version='1.11', cuda_major_version='11'):
    lines = get_pytorch_version_strings(python_version, pytorch_version, cuda_major_version)
    builds = [ln.split()[2] for ln in lines]
    return sorted(builds)


def get_pytorch_version_strings(python_version='3.7', pytorch_version='1.11', cuda_major_version='11'):
    if cuda_major_version is not None:
        version_pattern = f'{pytorch_version}=py{python_version}_cuda{cuda_major_version}*'
    else:
        version_pattern = f'{pytorch_version}=py{python_version}_cuda*'
    cmd = f'conda search -c pytorch pytorch={version_pattern}'
    cmd_result = subprocess.check_output(cmd, shell=True)
    lines = cmd_result.decode('ascii').splitlines()[2:]
    return lines


def get_pytorch_version_string(cfg_path, pytorch_version='1.11'):
    cuda_major_version, _ = get_cuda_version()
    python_version = get_python_version(cfg_path)
    pytorch_strings = get_pytorch_version_strings(python_version, pytorch_version, cuda_major_version=None)
    pytorch_cuda_versions = get_pytorch_builds_cuda_versions(python_version, pytorch_version, cuda_major_version=None)
    best_match = get_best_match_cuda_version(pytorch_cuda_versions, cuda_version=get_cuda_version())
    matching_string = pytorch_strings[pytorch_cuda_versions.index(best_match)]
    return '='.join(matching_string.split()[1:3])


def check_pytorch():
    import torch
    exit_code = int(not torch.cuda.is_available())
    sys.exit(exit_code)


def patch_environment_package_line(cfg_path, dest_path, package_name, package_substitution_function, **kwargs):
    """
    Patch the yaml environment file

    Parameters
    ----------
    cfg_path
    dest_path
    package_name
    package_substitution_function

    Returns
    -------

    """
    import yaml  # WARNING: Only guaranteed to be here in the tmp env
    with open(cfg_path, 'r') as in_file:
        cfg = yaml.safe_load(in_file)

    patched_dependencies = []
    for dep in cfg['dependencies']:
        if isinstance(dep, str) and dep.startswith(package_name):
            patched_dependencies.append(f'{package_name}={package_substitution_function(cfg_path, **kwargs)}')
        else:
            patched_dependencies.append(dep)

    cfg['dependencies'] = patched_dependencies
    if not dest_path:
        dest_path = cfg_path
    with open(dest_path, 'w', encoding='utf8') as out_file:
        yaml.dump(cfg, out_file, default_flow_style=False, allow_unicode=True)


def get_python_version(cfg_path, default_version='3.7'):
    import yaml  # WARNING: Only guaranteed to be here in the tmp env
    with open(cfg_path, 'r') as in_file:
        cfg = yaml.safe_load(in_file)
    lines = [ln for ln in cfg if isinstance(ln, str) and ln.startswith('python')]
    if lines:
        return lines[0].split('=')[-1]
    else:
        return default_version


def get_env_name(cfg_path):
    import yaml  # WARNING: Only guaranteed to be here in the tmp env
    with open(cfg_path, 'r') as in_file:
        cfg = yaml.safe_load(in_file)
    return cfg['name']


def patch_cuda_toolkit_version(cfg_path, dest_path):
    patch_environment_package_line(cfg_path, dest_path, 'cudatoolkit', pytorch_to_cuda_toolkit_version)


def patch_pytorch_cuda_version(cfg_path, dest_path):
    patch_environment_package_line(cfg_path, dest_path, 'pytorch', get_pytorch_version_string)

