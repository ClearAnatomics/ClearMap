"""
install_utils
=============

Utilities module with minimal dependencies (standard library only) for installation
"""
import json
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

    def write(self):
        dest_path = self.dest_path if self.dest_path else self.cfg_path
        with open(dest_path, 'w', encoding='utf8') as out_file:
            yaml.dump(self.cfg, out_file, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def patch_environment_package_line(self, package_name, pkg_version):
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
                version_str = f"={pkg_version}" if pkg_version else ""
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

    def patch_env_var(self, var_name, var_val):
        if 'variables' in self.cfg.keys():
            self.cfg['variables'][var_name] = var_val
        else:
            self.cfg['variables'] = {var_name: var_val}
        self.write()


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
            self._pytorch_info = self.get_pytorch_gpu_info()
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
        for dep in pkg['depends']:
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

    def get_pytorch_gpu_info(self):
        cuda_major = self.cuda_version[0] if self.used_cuda_major is None else self.used_cuda_major
        version_pattern = f'{self.pytorch_version}=py{self.python_version}_cuda{cuda_major}*'
        return CondaParser.get_conda_pkg_info('pytorch', ['pytorch'], version_pattern)

    def get_pytorch_cpu_info(self):
        version_pattern = f'{self.pytorch_version}=py{self.python_version}_cpu*'
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


def patch_env(cfg_path, dest_path, use_cuda_torch=True, tmp_dir=None):
    env_mgr = EnvFileManager(cfg_path, dest_path)

    pytorch_v_mgr = PytorchVersionManager(cfg_path, env_mgr.python_version, env_mgr.get_package_version('pytorch'))
    if use_cuda_torch:
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

    if tmp_dir not in ('/tmp', '/tmp/'):
        print(f'Patching tmp_dir to {tmp_dir}')
        env_mgr.patch_env_var('TMP', tmp_dir)
