# -*- coding: utf-8 -*-
"""
install_utils
=============

Utilities module with minimal dependencies (standard library only) for installation
"""
import sys
import platform
import subprocess
import json
from packaging.version import Version
from pathlib import Path

from ClearMap import Settings as clearmap_settings

try:
    import yaml  # WARNING: Only guaranteed to be here in the tmp env
except ImportError:
    pass

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright (c) 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


class EnvFileManager:
    def __init__(self, cfg_path, dest_path):
        self.cfg_path = cfg_path
        self.dest_path = dest_path
        self.__python_version = None
        with open(self.cfg_path, 'r') as in_file:
            self.cfg = yaml.safe_load(in_file)

    def __str__(self):
        return f'{self.dest_path} {self.cfg}'

    @property
    def python_version(self):
        if self.__python_version is None:
            self.__python_version = self.get_package_version('python')
        return self.__python_version

    def get_env_name(self):
        return self.cfg['name']

    def get_package_version(self, package_name):
        lines = [ln for ln in self.cfg['dependencies'] if isinstance(ln, str) and ln.startswith(package_name)]
        return lines[0].split('=')[-1]

    def write(self):
        dest_path = self.dest_path if self.dest_path else self.cfg_path
        with open(dest_path, 'w', encoding='utf8') as out_file:
            yaml.dump(self.cfg, out_file, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def patch_environment_package_line(self, package_name, pkg_version='', comparison_operator='='):
        """
        Patch the yaml environment file

        Parameters
        ----------
        package_name
        pkg_version

        Returns
        -------

        """
        patched_dependencies = []
        for dep in self.cfg['dependencies']:
            if isinstance(dep, str) and dep.startswith(package_name):
                version_str = (f'{comparison_operator}{pkg_version}') if pkg_version else ''
                patched_dependencies.append(f'{package_name}{version_str}')
            else:
                patched_dependencies.append(dep)

        self.cfg['dependencies'] = patched_dependencies
        self.write()

    def remove_dependency(self, package_name):
        patched_dependencies = []
        for dep in self.cfg['dependencies']:
            if isinstance(dep, str):
                if not dep.startswith(package_name):
                    patched_dependencies.append(dep)
            else:
                patched_dependencies.append(dep)

        self.cfg['dependencies'] = patched_dependencies
        self.write()

    def add_pip_option(self, option):
        if 'pip' in self.cfg['dependencies']:
            try:
                pip_section = [d for d in self.cfg['dependencies'] if isinstance(d, dict) and 'pip' in d.keys()][0]
                pip_section['pip'].append(option)
            except IndexError:
                self.cfg['dependencies'].append({'pip': [option]})
        else:
            self.add_dependency('pip')
            self.cfg['dependencies'].append({'pip': [option]})
        self.write()

    def add_pip_dependency(self, package_name):
        if 'pip' in self.cfg['dependencies']:
            try:
                pip_section = [d for d in self.cfg['dependencies'] if isinstance(d, dict) and 'pip' in d.keys()][0]
                pip_section['pip'].append(package_name)
            except IndexError:
                self.cfg['dependencies'].append({'pip': [package_name]})
        else:
            self.add_dependency('pip')
            self.cfg['dependencies'].append({'pip': [package_name]})
        self.write()


    def patch_env_var(self, var_name, var_val):
        if 'variables' in self.cfg.keys():
            self.cfg['variables'][var_name] = var_val
        else:
            self.cfg['variables'] = {var_name: var_val}
        self.write()

    def add_dependency(self, package_name, pkg_version=''):
        version_str = f"={pkg_version}" if pkg_version else ""
        self.cfg['dependencies'].append(f'{package_name}{version_str}')
        self.write()

    def remove_channel(self, channel_name):
        self.cfg['channels'] = [c for c in self.cfg['channels'] if c != channel_name]

    def add_channel(self, channel_name):
        if channel_name not in self.cfg['channels']:
            self.cfg['channels'].append(channel_name)


class CondaParser:
    @staticmethod
    def pkg_to_v_string(pkg):
        return f'{pkg["version"]}={pkg["build"]}'

    @staticmethod
    def get_conda_pkg_info(pkg_name, channels, version_pattern):
        channels_str = ' '.join([f'-c {c}' for c in channels])
        cmd = f'conda search {channels_str} {pkg_name}={version_pattern} --info --json'
        cmd_output = subprocess.check_output(cmd, shell=True).decode('ascii')
        info_blocks = json.loads(cmd_output)[pkg_name]
        return info_blocks


class PytorchVersionManager:  # TODO: inherit from condaparser ??
    def __init__(self, cfg_path, python_version, pytorch_version='1.11'):
        self.cfg_path = cfg_path
        self.python_version = python_version
        self.pytorch_version = pytorch_version
        self.__cuda_version = None
        self.used_cuda_major = None
        self._pytorch_info = None

    @staticmethod
    def assert_cuda():
        cmd = 'nvidia-smi -L'
        try:
            subprocess.check_output(cmd, shell=True)
            sys.exit(0)
        except subprocess.CalledProcessError as err:
            print('Error, a working installation of NVidia CUDA could not be detected on your computer.'
                  ' This software is required to run ClearMap.')
            sys.exit(err.returncode)

    @property
    def cuda_version(self):
        if self.__cuda_version is None:
            cmd = 'nvidia-smi -q -u'
            lines = subprocess.check_output(cmd, shell=True).splitlines()
            lines = [ln.decode('ASCII') for ln in lines]
            cuda_line = [ln for ln in lines if ln.lower().startswith('cuda version')][0]
            self.__cuda_version = Version(cuda_line.split(':')[-1].strip())
        return self.__cuda_version

    @property
    def cuda_major(self):
        return self.used_cuda_major or self.cuda_version.major

    @staticmethod
    def check_pytorch():
        import torch
        exit_code = int(not torch.cuda.is_available())
        sys.exit(exit_code)

    @property
    def pytorch_info(self):
        if self._pytorch_info is None:
            self._pytorch_info = self.get_pytorch_gpu_info()
        return self._pytorch_info

    def get_best_match_cuda_version(self, versions):
        print(f'CUDA version: {self.cuda_version}')
        for v in versions[::-1]:
            print(f'\tcandidate version: {v}')
            if self.cuda_version >= Version(v):
                return v
        else:
            raise ValueError(f'No match found between installed CUDA {self.cuda_version} and candidates {versions}')

    def toolkit_dep_to_version(self, dep_str):
        version_str = dep_str.split(' ')[-1].split(',')[0].split('=')[-1]
        return [int(e) for e in version_str.split('.')]

    def toolkit_version_from_torch_pkg(self, pkg):
        for dep in pkg['depends']:
            if 'cudatoolkit' in dep:
                return self.toolkit_dep_to_version(dep)
        raise ValueError(f'Dependency "cudatoolkit" not found in {pkg=}')

    def match_pytorch_to_toolkit(self):
        available_cudatoolkit_versions = self.get_toolkit_versions()
        for i, pkg in enumerate(self.pytorch_info[::-1]):
            # print(i, pkg)
            pytorch_toolkit_version = self.toolkit_version_from_torch_pkg(pkg)
            if pytorch_toolkit_version in available_cudatoolkit_versions:
                return pkg
        else:
            raise ValueError('No matching versions found')

    def match_pytorch_to_cuda(self):
        available_pytorch_cuda_versions = [self.torch_build_to_cuda_version(pkg['build']) for pkg in self.pytorch_info]
        best_match_cuda_version = self.get_best_match_cuda_version(available_pytorch_cuda_versions)
        print(f'{best_match_cuda_version=}')
        for i, pkg in enumerate(self.pytorch_info[::-1]):
            if self.torch_build_to_cuda_version(pkg['build']) == best_match_cuda_version:
                return best_match_cuda_version
        else:
            raise ValueError('No matching versions found')

    def get_pytorch_gpu_info(self):
        cuda_major = self.cuda_major
        version_pattern = f'{self.pytorch_version}=py{self.python_version}_cuda{cuda_major}*'
        return CondaParser.get_conda_pkg_info('pytorch', ['pytorch'], version_pattern)

    def get_pytorch_cpu_info(self):
        os_name = platform.system().lower()
        if os_name.startswith('linux') or os_name.startswith('windows'):
            version_pattern = f'{self.pytorch_version}=py{self.python_version}_cpu*'
        elif os_name.startswith('darwin'):
            version_pattern = f'{"*"}=cpu_py{"".join(self.python_version.split("."))}*'
        else:
            raise ValueError(f'Unknown platform {os_name}')
        return CondaParser.get_conda_pkg_info('pytorch', ['pytorch'], version_pattern)

    def get_toolkit_info(self):
        cuda_major = self.cuda_major
        while cuda_major > 9:
            try:
                return CondaParser.get_conda_pkg_info('cudatoolkit', ['nvidia'], cuda_major)
            except subprocess.CalledProcessError:
                cuda_major -= 1
                self.used_cuda_major = cuda_major

    def get_toolkit_versions(self):
        return [[int(e) for e in pkg['version'].split('.')[:2]] for pkg in self.get_toolkit_info()]

    def torch_build_to_cuda_version(self, build):
        return [e.replace('cuda', '') for e in build.split('_') if 'cuda' in e][0]


def patch_env(cfg_path, dest_path, use_cuda_torch=True, pip_mode=False, use_spyder=False, tmp_dir=None):
    """
    Patch the environment file to match the desired configuration. This is mostly to
    get a working pytorch installation with the correct cuda version.

    Parameters
    ----------
    cfg_path
    dest_path : str
        If evaluates to False, the cfg_path is overwritten
    use_cuda_torch : bool
        If True, install pytorch with cuda support
    pip_mode : bool
        If True, install pytorch with pip (since the nvidia channel is now considered a paid channel)
    use_spyder
    tmp_dir

    Returns
    -------

    """
    env_mgr = EnvFileManager(cfg_path, dest_path)

    if platform.system().lower().startswith('darwin'):
        if platform.processor().lower().startswith('x86'):
            env_mgr.add_dependency('nomkl')  # MacOS includes "accelerate" and does not need Intel MKL on Intel CPU
            env_mgr.patch_env_var('KMP_DUPLICATE_LIB_OK', 'TRUE')  # FIXME: find cleaner fix
        env_mgr.patch_environment_package_line('pyqt', '5.13', comparison_operator='<=')  # REFACTOR: get from other env file instead

    pytorch_v_mgr = PytorchVersionManager(cfg_path, env_mgr.python_version, env_mgr.get_package_version('pytorch'))
    if use_cuda_torch:
        if pip_mode:
            # remove the nvidia channel if it is present
            env_mgr.remove_channel('nvidia')
            env_mgr.remove_dependency('pytorch')
            env_mgr.remove_dependency('pytorch-cuda')
            env_mgr.remove_dependency('mkl')
            env_mgr.remove_dependency('cudatoolkit')
            env_mgr.add_pip_option('--extra-index-url https://download.pytorch.org/whl/cu121')
            env_mgr.add_pip_dependency('torch')
        else:
            # ensure that the nvidia channel is available
            env_mgr.add_channel('nvidia')
            if Version(pytorch_v_mgr.pytorch_version) >= Version('2.0'):
                pytorch_cuda_version = pytorch_v_mgr.match_pytorch_to_cuda()
                env_mgr.add_dependency('pytorch-cuda', pkg_version=pytorch_cuda_version)
                env_mgr.remove_dependency('cudatoolkit')
            else:
                torch_pkg = pytorch_v_mgr.match_pytorch_to_toolkit()
                torch_v_string = f"{torch_pkg['version']}={torch_pkg['build']}"
                env_mgr.patch_environment_package_line('pytorch', torch_v_string)

                toolkit_v_tuple = pytorch_v_mgr.toolkit_version_from_torch_pkg(torch_pkg)
                toolkit_v_string = f"{toolkit_v_tuple[0]}.{toolkit_v_tuple[1]}"
                env_mgr.patch_environment_package_line('cudatoolkit', toolkit_v_string)
    else:
        torch_pkg = pytorch_v_mgr.get_pytorch_cpu_info()[-1]
        torch_v_string = f"{torch_pkg['version']}={torch_pkg['build']}"
        env_mgr.patch_environment_package_line('pytorch', torch_v_string)

        env_mgr.remove_dependency('cudatoolkit')

    if use_spyder:
        env_mgr.add_dependency('spyder-kernels', pkg_version='2.4')

    if tmp_dir not in ('/tmp', '/tmp/'):
        print(f'Patching tmp_dir to {tmp_dir}')
        env_mgr.patch_env_var('TMP', tmp_dir)


if __name__ == '__main__':
    if sys.argv[1] == 'test_patch':
        clearmap_folder = Path(clearmap_settings.clearmap_path)
        patch_env(clearmap_folder.parent / sys.argv[2], clearmap_folder.parent / sys.argv[3],
                  use_cuda_torch=True, use_spyder=False, tmp_dir='/tmp/tmp')
