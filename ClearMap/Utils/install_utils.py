"""
install_utils
=============

Utilities module with minimal dependencies (standard library only) for installation
"""
import sys
import subprocess
try:
    import yaml  # WARNING: Only guaranteed to be here in the tmp env
except ImportError:
    pass

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


class EnvFileManager:
    def __init__(self, cfg_path, dest_path):
        self.cfg_path = cfg_path
        self.dest_path = dest_path
        self.__python_version = None
        with open(self.cfg_path, 'r') as in_file:
            self.cfg = yaml.safe_load(in_file)

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

    def patch_environment_package_line(self, package_name, pkg_version):
        """
        Patch the yaml environment file

        Parameters
        ----------
        package_name
        pkg_version_fn

        Returns
        -------

        """
        patched_dependencies = []
        for dep in self.cfg['dependencies']:
            if isinstance(dep, str) and dep.startswith(package_name):
                patched_dependencies.append(f'{package_name}={pkg_version}')
            else:
                patched_dependencies.append(dep)

        self.cfg['dependencies'] = patched_dependencies
        dest_path = self.dest_path if self.dest_path else self.cfg_path
        with open(dest_path, 'w', encoding='utf8') as out_file:
            yaml.dump(self.cfg, out_file, default_flow_style=False, allow_unicode=True)


class CondaParser:
    @staticmethod
    def parse_conda_info(lines, package_name):
        info_blocks = []
        block = None
        current_key = None
        for ln in lines:
            if ln.startswith(package_name):
                if block:
                    info_blocks.append(block)
                block = {}
                continue
            elif ln.startswith('----'):
                continue
            if block is not None:
                if ln.startswith('  - '):
                    block[current_key].append(ln[4:].strip())
                else:
                    elements = ln.split(':')
                    current_key = elements[0].strip()
                    if len(elements) > 1 and elements[1].strip():
                        block[current_key] = elements[1].strip()  # WARNING: pb if ':' in elements[1]
                    else:
                        block[current_key] = []
        if block:
            info_blocks.append(block)
        return info_blocks

    @staticmethod
    def pkg_to_v_string(pkg):
        return f'{pkg["version"]}={pkg["build"]}'

    @staticmethod
    def get_conda_pkg_info(pkg_name, channels, version_pattern):
        channels_str = ' '.join([f'-c {c}' for c in channels])
        cmd = f'conda search {channels_str} {pkg_name}={version_pattern} --info'
        lines = subprocess.check_output(cmd, shell=True).decode('ascii').splitlines()
        info_blocks = CondaParser.parse_conda_info(lines, pkg_name)
        return info_blocks


class CudaVersionManager:
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
            self.__cuda_version = [int(e) for e in cuda_line.split(':')[-1].strip().split('.')]
        return self.__cuda_version

    @staticmethod
    def check_pytorch():
        import torch
        exit_code = int(not torch.cuda.is_available())
        sys.exit(exit_code)

    @property
    def pytorch_info(self):
        if self._pytorch_info is None:
            self._pytorch_info = self.get_pytorch_info()
        return self._pytorch_info

    def get_best_match_cuda_version(self, versions):
        major, minor = self.cuda_version
        for v in versions[::-1]:
            print(f'CUDA version: {self.cuda_version}, candidate version: {v}')
            v_major, v_minor = v
            if v_major > major:
                raise ValueError(f'Version not supported: {v}')
            else:
                if v_major < major or v_minor <= minor:
                    return v
        else:
            raise ValueError(f'No match found between installed CUDA {self.cuda_version} and candidates {versions}')

    def cuda_matches(self, version):
        major, minor = self.cuda_version
        v_major, v_minor = version
        return v_major < major or v_minor <= minor

    def toolkit_dep_to_version(self, dep_str):
        version_str = dep_str.split(' ')[-1].split(',')[0].split('=')[-1]
        return [int(e) for e in version_str.split('.')]

    def toolkit_version_from_torch_pkg(self, pkg):
        for dep in pkg['dependencies']:
            if 'cudatoolkit' in dep:
                return self.toolkit_dep_to_version(dep)
        raise ValueError('Dependency not found')

    def match_pytorch_to_toolkit(self):
        available_cudatoolkit_versions = self.get_toolkit_versions()
        for i, pkg in enumerate(self.pytorch_info[::-1]):
            pytorch_toolkit_version = self.toolkit_version_from_torch_pkg(pkg)
            if pytorch_toolkit_version in available_cudatoolkit_versions:
                return pkg
        else:
            raise ValueError('No matching versions found')

    def get_pytorch_info(self):
        cuda_major = self.cuda_version[0] if self.used_cuda_major is None else self.used_cuda_major
        version_pattern = f'{self.pytorch_version}=py{self.python_version}_cuda{cuda_major}*'
        return CondaParser.get_conda_pkg_info('pytorch', ['pytorch'], version_pattern)

    def get_toolkit_info(self):
        cuda_major = self.cuda_version[0] if self.used_cuda_major is None else self.used_cuda_major
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


def patch_env(cfg_path, dest_path):
    env_mgr = EnvFileManager(cfg_path, dest_path)

    cuda_mgr = CudaVersionManager(cfg_path, env_mgr.python_version, env_mgr.get_package_version('pytorch'))

    torch_pkg = cuda_mgr.match_pytorch_to_toolkit()
    toolkit_v_tuple = cuda_mgr.toolkit_version_from_torch_pkg(torch_pkg)
    toolkit_v_string = f"{toolkit_v_tuple[0]}.{toolkit_v_tuple[1]}"
    env_mgr.patch_environment_package_line('cudatoolkit', toolkit_v_string)
    torch_v_string = f"{torch_pkg['version']}={torch_pkg['build']}"
    env_mgr.patch_environment_package_line('pytorch', torch_v_string)
