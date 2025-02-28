"""
This module provides a class to load configuration files from the ClearMap configuration directory.

The configuration files are used to store the parameters for the ClearMap processing steps.
Currently, they are stored as configobj files, but other formats like json or yaml
could be supported in the future through a simple plugin function to this module
(see the not_implemented functions). All that is required is that the object returned by these
plugin functions implements read, write, and reload methods, a filename attribute and behaves as
a python dictionary.
"""
import inspect
import os
import re
from pathlib import Path

import configobj
from packaging.version import Version
from importlib_metadata import version

clearmap_version = version('ClearMap')
VERSION_SUFFIX = f'v{Version(clearmap_version).major}_{Version(clearmap_version).minor}'

# FIXME: implement validation


INSTALL_CFG_DIR = Path(inspect.getfile(inspect.currentframe())).parent.absolute()  # Where this file resides (w cfgs)
CLEARMAP_CFG_DIR = Path('~/.clearmap/').expanduser()


def get_configobj_cfg(cfg_path, must_exist=True):
    cfg_path = clean_path(str(cfg_path))
    try:
        return configobj.ConfigObj(cfg_path, encoding="UTF8", indent_type='    ', unrepr=True, file_error=must_exist)
    except configobj.ConfigObjError as err:
        print(f'Could not read config file "{cfg_path}", some errors were encountered: "{err}"')


def get_yml_cfg(cfg_path):
    """

    Parameters
    ----------
    cfg_path (str)

    Returns Should return a dict like object with a write method and filename that gives the path
    -------

    """
    raise NotImplementedError


def get_json_cfg(cfg_path):
    """

    Parameters
    ----------
    cfg_path (str)

    Returns Should return a dict like object with a write method and filename that gives the path
    -------

    """
    raise NotImplementedError


tabs_alternatives = [
    ['sample', 'sample_info', 'sample info'],
    ['stitching'],
    ['registration'],
    ['cell_map', 'cell_counter'],
    ['vasculature', 'tube_map'],
    ['batch', 'batch_processing', 'batch processing'],
    ['group_analysis', 'group analysis']
]

if clearmap_version < '3.0.0':
    tabs_alternatives.insert(1, ['alignment', 'processing'])


alternative_names = tabs_alternatives + [['machine'], ['display']]
CONFIG_NAMES = [names[0] for names in alternative_names]
if 'alignment' not in CONFIG_NAMES:
    CONFIG_NAMES.append('alignment')  # For compatibility with older versions. Only in names, not tabs_alternatives


def get_alternatives(cfg_name):
    alternatives = [names for names in alternative_names if cfg_name in names]
    if not alternatives:
        raise ValueError(f'Could not find any alternative for {cfg_name}')
    return alternatives[0]


def flatten_alternatives(alternatives):
    flat = []
    for names in alternatives:
        flat.extend(names)
    return flat


def is_tab_file(cfg_name):
    cfg_name = cfg_name.replace('_params', '')
    cfg_name = ConfigLoader.strip_version_suffix(cfg_name)
    return cfg_name in flatten_alternatives(tabs_alternatives)


def clean_path(path):
    return os.path.normpath(os.path.expanduser(path))


def is_machine_file(cfg_name):
    return any([base in cfg_name for base in ('machine', 'preferences')])


def patch_cfg(cfg, default_cfg):
    for k, v in default_cfg.items():
        if k not in cfg.keys():
            cfg[k] = v  # everything below will match by definition
        else:
            if isinstance(v, dict):
                patch_cfg(cfg[k], v)


class ConfigLoader(object):
    loader_functions = {
        '.cfg': get_configobj_cfg,
        '.ini': get_configobj_cfg,
        '.yml': get_yml_cfg,
        '.json': get_json_cfg
    }
    supported_exts = tuple(loader_functions.keys())
    default_dir = CLEARMAP_CFG_DIR

    def __init__(self, src_dir):
        self._src_dir = None
        self.src_dir = src_dir
        self.sample_cfg_path = ''  # OPTIMISE: could use cached property
        self.preferences_path = ''
        self.cell_map_cfg_path = ''

    @property
    def src_dir(self):
        return self._src_dir

    @src_dir.setter
    def src_dir(self, value):
        self._src_dir = Path(value).expanduser()  # TODO: normpath?

    def get_cfg_path(self, cfg_name, must_exist=True):
        """
        Get the path to the configuration file with the given name.
        Several extensions are tried in order of preference.
        If None is found, the first possible option is returned if must_exist is False or
        a FileNotFoundError is raised if must_exist is True.

        Parameters
        ----------
        cfg_name: str
            The name (without params and extension) of the configuration file
        must_exist: bool
            Whether the file must exist. If missing and True, a FileNotFoundError is raised.

        Returns
        -------
        Path
            The path to the configuration file
        """

        variants = get_alternatives(cfg_name)
        for alternative_name in variants:
            if not alternative_name.endswith('params'):
                alternative_name += '_params'
            for ext in self.supported_exts:
                cfg_path = self.src_dir / f'{alternative_name}{ext}'
                if cfg_path.exists():
                    return cfg_path
        if not must_exist:  # If none found but not necessary, return the first possible option
            return self.src_dir / f'{cfg_name}_params{self.supported_exts[0]}'
        raise FileNotFoundError(f'Could not find file {cfg_name} in {self.src_dir} with variants {variants}')

    def get_cfg(self, cfg_name, must_exist=True):
        if '/' in str(cfg_name):  # Already a path
            cfg_path = cfg_name
        else:
            if is_tab_file(cfg_name):
                cfg_path = self.get_cfg_path(cfg_name, must_exist=must_exist)
            else:
                cfg_path = self.get_default_path(cfg_name, must_exist=must_exist)
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            if must_exist:
                raise FileNotFoundError(f'Could not find file {cfg_name} in {self.src_dir}')
            else:
                return None
        return self.get_cfg_from_path(cfg_path)

    @staticmethod
    def get_cfg_from_path(cfg_path):
        cfg_path = Path(cfg_path)
        return ConfigLoader.loader_functions[cfg_path.suffix](cfg_path)

    @staticmethod
    def strip_version_suffix(cfg_name):
        pattern = r'_v\d+_\d+$'
        return re.sub(pattern, '', cfg_name)


    @staticmethod
    def get_patched_cfg_from_path(cfg_path):
        cfg_path = Path(cfg_path)
        cfg = ConfigLoader.get_cfg_from_path(cfg_path)
        config_name = ConfigLoader.strip_version_suffix(cfg_path.stem)
        default_cfg = ConfigLoader.get_cfg_from_path(ConfigLoader.get_default_path(config_name))
        patch_cfg(cfg, default_cfg)
        cfg.write()
        return cfg

    @staticmethod
    def get_default_path(cfg_name, must_exist=True, from_package=False):  # FIXME: recursive w/ alternatives
        if cfg_name.endswith('_params'):
            cfg_name = cfg_name.replace('_params', '')
        paths_checked = []
        for name_variant in get_alternatives(cfg_name):
            name_variant += f'_params'
            for ext in ConfigLoader.supported_exts:
                cfg_path = ConfigLoader._name_to_default_path(f'{name_variant}_{VERSION_SUFFIX}',
                                                              ext, from_package=from_package)
                paths_checked.append(cfg_path)
                if os.path.exists(cfg_path):
                    return cfg_path
        else:
            if must_exist:
                raise FileNotFoundError(f'Could not find file {cfg_name}, checked {paths_checked}')
            else:  # Return first (default) ext if none found
                ext = ConfigLoader.supported_exts[0]
                return ConfigLoader._name_to_default_path(f'{cfg_name}_params_{VERSION_SUFFIX}',  ext, from_package=from_package)
                #return ConfigLoader._name_to_default_path(f'{cfg_name}_params', ConfigLoader.supported_exts[0],
                #                                          from_package=from_package)

    @staticmethod
    def _name_to_default_path(cfg_name, ext, from_package=False):
        prefix = 'default_' if is_tab_file(cfg_name) else ''
        cfg_name = f'{prefix}{cfg_name}{ext}'

        cfg_dir = INSTALL_CFG_DIR if from_package else ConfigLoader.default_dir
        cfg_path = clean_path(cfg_dir / cfg_name)
        return cfg_path


def get_configs(cfg_path, processing_params_path, machine_cfg_path=None):  # FIXME: fix missing stuff here
    if machine_cfg_path is None:
        machine_cfg_path = ConfigLoader.get_default_path('machine')
    sample_config = ConfigLoader.get_patched_cfg_from_path(cfg_path)
    processing_config = ConfigLoader.get_patched_cfg_from_path(processing_params_path)
    machine_config = ConfigLoader.get_patched_cfg_from_path(machine_cfg_path)

    return machine_config, sample_config, processing_config


def get_cfg_reader_function(cfg_path):
    ext = os.path.splitext(cfg_path)[-1]
    read_cfg = ConfigLoader.loader_functions[ext]
    return read_cfg
