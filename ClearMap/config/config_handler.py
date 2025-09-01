"""
This module provides a class to load/write configuration files from the ClearMap configuration directory.

The configuration files are used to store the parameters for the ClearMap processing steps.
Supported formats are .cfg (ConfigObj), .yml/.yaml (YAML) and .json (JSON).
Other formats (TOML, INI, XML, etc.)  could be supported in the future through
 a simple plugin function to this module.
"""
import os
import inspect
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Any, Dict

import configobj
from packaging.version import Version
from importlib_metadata import version

from ClearMap.IO.FileUtils import atomic_replace
from .utils import _handle_configobj_failed_parse, configobj_to_dict

clearmap_version = version('ClearMap')
VERSION_SUFFIX = f'v{Version(clearmap_version).major}_{Version(clearmap_version).minor}'

# FIXME: implement validation


INSTALL_CFG_DIR = Path(inspect.getfile(inspect.currentframe())).parent.absolute()  # Where this file resides (w cfgs)
CLEARMAP_CFG_DIR = Path('~/.clearmap/').expanduser()


# Optional YAML dependency
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


@dataclass
class ConfigProxy(dict):
    """
    Dict-like wrapper that remembers filename and provides write()/reload().
    Used to return from get_*_cfg functions.
    This is basically just a dict with I/O methods attached.
    Compatible with ConfigObj interface for yml/json/cfg files.
    """
    filename: str
    _loader: Callable[[Path], "ConfigProxy"]
    _dumper: Callable[[Path, dict], None]

    def write(self, outfile: Optional[Any] = None) -> None:
        """
        Backwards-compatible signature. If outfile is given (a file-like),
        we still write to our filename atomically, ignoring outfile (kept for compatibility).
        """
        path = Path(self.filename)
        tmp = path.with_suffix(f'{path.suffix}.tmp')
        self._dumper(tmp, dict(self))
        atomic_replace(tmp, path)

    def reload(self) -> "ConfigProxy":
        return self._loader(Path(self.filename))

# ---- Writers (path, dict) ----------------------------------------------------

def to_configobj(path: Path, data: dict) -> None:
    """
    Write a dict to a .cfg/.ini file using ConfigObj.

    Parameters
    ----------
    path: Path
        Path to the output .cfg/.ini file
    data: dict
        Data to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = configobj.ConfigObj(encoding="UTF8", indent_type="    ", unrepr=True)
    cfg.filename = str(path)
    cfg.clear()
    for k, v in data.items():
        cfg[k] = v
    cfg.write()

def to_yml(path: Path, data: dict) -> None:
    """
    Write a dict to a .yml/.yaml file using PyYAML.


    Parameters
    ----------
    path: Path
        Path to the output .yml/.yaml file
    data: dict
        Data to write

    Raises
    ------
    RuntimeError
        If PyYAML is not installed.
    """
    if not _HAS_YAML:
        raise RuntimeError('PyYAML is not installed; cannot write YAML configs.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def to_json(path: Path, data: dict) -> None:
    """
    Write a dict to a .json file using the json module.

    Parameters
    ----------
    path: Path
        Path to the output .json file
    data: dict
        Data to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# loader functions

def get_configobj_cfg(cfg_path: Path | str, must_exist: bool = True) -> ConfigProxy | None:
    """
    Load a .cfg/.ini file using ConfigObj and return as ConfigProxy.

    Parameters
    ----------
    cfg_path: Path | str
        Path to the .cfg/.ini file
    must_exist: bool
        If True, raise an error if the file does not exist.

    Returns
    -------
    ConfigProxy | None
        The configuration as a ConfigProxy object, or None if the file does not exist and must_exist is False.
    Raises
    ------
    ConfigParsingError
        If the file cannot be parsed.
    """
    cfg_path = str(clean_path(cfg_path))  # str to be future-proof. ConfigObj wants str
    try:
        cobj = configobj.ConfigObj(cfg_path, encoding="UTF8", indent_type='    ', unrepr=True, file_error=must_exist)
    except configobj.ConfigObjError as err:
        msg = str(err)  # Parse error message for
                        # "Parsing failed with several errors. First error at line 19."
                        # to reraise with more context
        if 'parsing failed' in msg.lower():
            _handle_configobj_failed_parse(err, Path(cfg_path))
        else:
            warnings.warn(f'Could not read config file "{cfg_path}", some errors were encountered: "{err}"')
            return None

    return ConfigProxy(
        filename=str(cfg_path),
        _loader=lambda p: get_configobj_cfg(p, must_exist=True),  # type: ignore
        _dumper=to_configobj,
        **(configobj_to_dict(cobj)),
    )


def get_yml_cfg(cfg_path: Path, must_exist: bool = True) -> ConfigProxy | None:
    """
    Load a .yml/.yaml file using PyYAML and return as ConfigProxy.

    Parameters
    ----------
    cfg_path: Path
        Path to the .yml/.yaml file
    must_exist: bool
        If True, raise an error if the file does not exist.

    Returns
    -------
    ConfigProxy | None
        The configuration as a ConfigProxy object, or None if the file does not exist and must_exist is False.
    """
    if not _HAS_YAML:
        raise RuntimeError('PyYAML is not installed; cannot read YAML configs.')
    cfg_path = Path(clean_path(cfg_path))
    if not cfg_path.exists() and must_exist:
        return None
    with cfg_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'YAML root must be a mapping; got {type(data)} in {cfg_path}')
    return ConfigProxy(
        filename=str(cfg_path),
        _loader=lambda p: get_yml_cfg(p, must_exist=True),  # type: ignore
        _dumper=to_yml,
        **data,
    )

def get_json_cfg(cfg_path: Path, must_exist: bool = True) -> ConfigProxy | None:
    """
    Load a .json file using the json module and return as ConfigProxy.

    Parameters
    ----------
    cfg_path: Path
        Path to the .json file
    must_exist: bool
        If True, raise an error if the file does not exist.

    Returns
    -------
    ConfigProxy | None
        The configuration as a ConfigProxy object, or None if the file does not exist and must_exist is False.
    """
    cfg_path = Path(clean_path(cfg_path))
    if not cfg_path.exists() and must_exist:
        return None
    with cfg_path.open('r', encoding='utf-8') as f:
        data = json.load(f) if cfg_path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError(f'JSON root must be a mapping; got {type(data)} in {cfg_path}')
    return ConfigProxy(
        filename=str(cfg_path),
        _loader=lambda p: get_json_cfg(p, must_exist=True),  # type: ignore
        _dumper=to_json,
        **data,
    )


def clean_path(path: str | Path) -> str:
    """Expand user (~) and normalize path."""
    return os.path.normpath(os.path.expanduser(str(path)))


def patch_cfg(cfg, default_cfg):
    """
    Recursively patch cfg with missing keys from default_cfg.
    Parameters
    ----------
    cfg: dict-like
        The configuration to patch.
    default_cfg: dict-like
        The default configuration to use for patching.
    """
    for k, v in default_cfg.items():
        if k not in cfg.keys():
            cfg[k] = v  # everything below will match by definition
        else:
            if isinstance(v, dict):
                if not isinstance(cfg[k], dict):
                    raise ValueError(f'Cannot merge dict into non-dict for key {k}.'
                                     f'Please first convert your config to a dict with e.g. configobj_to_dict.')
                patch_cfg(cfg[k], v)


"""
List of alternative names for configuration files.
"""

tabs_alternatives = [
    ['sample', 'sample_info', 'sample info'],
    ['stitching'],
    ['registration'],
    ['cell_map', 'cell_counter'],
    ['tract_map'],
    ['vasculature', 'tube_map'],
    ['batch', 'batch_processing', 'batch processing'],
    ['group_analysis', 'group analysis'],
    ['colocalization']
]

alternative_names = tabs_alternatives + [['machine'], ['display'], ['alignment', 'processing']]  # alignment for v2 compatibility


CONFIG_NAMES = [names[0] for names in alternative_names]


def get_alternatives(cfg_name: str) -> list[str]:
    """
    Get the list of alternative names for a given configuration name.

    Parameters
    ----------
    cfg_name: str
        The base name of the configuration file (without params and extension).

    Returns
    -------
    list[str]
        A list of alternative names for the configuration file.
    Raises
    ------
    ValueError
        If no alternatives are found for the given configuration name.
    """
    alternatives = [names for names in alternative_names if cfg_name in names]
    if not alternatives:
        raise ValueError(f'Could not find any alternative for {cfg_name}')
    return alternatives[0]


def flatten_alternatives(alternatives: list[list[str]]) -> list[str]:
    """Flatten a list of lists of alternative names into a single list."""
    flat = []
    for names in alternatives:
        flat.extend(names)
    return flat


def is_tab_file(cfg_name: str) -> bool:
    """Check if the given config name is a tab file (i.e. has alternatives)."""
    cfg_name = cfg_name.replace('_params', '')
    cfg_name = ConfigHandler.strip_version_suffix(cfg_name)
    return cfg_name in flatten_alternatives(tabs_alternatives)


def is_machine_file(cfg_name: str) -> bool:
    """Check if the given config name is a machine file (i.e. machine or preferences)."""
    return any([base in cfg_name for base in ('machine', 'preferences')])


class ConfigHandler:
    """
    Resolves logical config names to files and loads them via readers;
    writes using writer_functions (atomic).



    Class Attributes
    ----------------
    loader_functions: Dict[str, Callable[[Path, bool], Optional[ConfigProxy]]]
        A dictionary mapping file extensions to their corresponding loader functions.
    writer_functions: Dict[str, Callable[[Path, dict], None]]
        A dictionary mapping file extensions to their corresponding writer functions.
    default_dir: Path
        The default directory where configuration files are located.
    supported_exts: tuple
        A tuple of supported file extensions for configuration files.

    Attributes
    ----------
    src_dir: Path
        The source directory where configuration files are located.
    """

    # Readers by extension
    loader_functions: Dict[str, Callable[[Path, bool], Optional[ConfigProxy]]] = {
        ".cfg": get_configobj_cfg,
        ".ini": get_configobj_cfg,
        ".yml": get_yml_cfg,
        ".yaml": get_yml_cfg,
        ".json": get_json_cfg,
    }
    # Writers by extension
    writer_functions: Dict[str, Callable[[Path, dict], None]] = {
        ".cfg": to_configobj,
        ".ini": to_configobj,
        ".yml": to_yml,
        ".yaml": to_yml,
        ".json": to_json,
    }

    supported_exts = tuple(loader_functions.keys())
    default_dir = CLEARMAP_CFG_DIR

    def __init__(self, src_dir: Path | str):
        self._src_dir = None
        self.src_dir = src_dir

    @property
    def src_dir(self):
        return self._src_dir

    @src_dir.setter
    def src_dir(self, value: str | Path):
        self._src_dir = Path(value).expanduser()  # TODO: normpath?

    def get_cfg_path(self, cfg_name: str, must_exist: bool = True) -> Path:
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

    def get_cfg(self, cfg_name: str, must_exist: bool = True) -> ConfigProxy | None:
        """
        Get the configuration file with the given name.
        Several extensions are tried in order of preference.
        If None is found, None is returned if must_exist is False or
        a FileNotFoundError is raised if must_exist is True.

        Parameters
        ----------
        cfg_name: str
            The name (without params and extension) of the configuration file
        must_exist: bool
            Whether the file must exist. If missing and True, a FileNotFoundError is raised

        Returns
        -------
        ConfigProxy | None
            The configuration file as a ConfigProxy object or None if not found and must_exist is False
        """
        if '/' in str(cfg_name):  # Already a path
            cfg_path = Path(cfg_name)
        else:
            if is_tab_file(cfg_name):
                cfg_path = self.get_cfg_path(cfg_name, must_exist=must_exist)
            else:
                cfg_path = self.get_default_path(cfg_name, must_exist=must_exist)
        if not cfg_path.exists():
            if must_exist:
                raise FileNotFoundError(f'Could not find file {cfg_name} in {self.src_dir}')
            else:
                return None
        return self.get_cfg_from_path(cfg_path)

    @classmethod
    def get_cfg_from_path(cls, cfg_path: str | Path) -> ConfigProxy:
        """
        Load a configuration file from the given path using the appropriate loader function.
        The function is determined by the file extension.

        .. warning::

            The file must exist.

        Parameters
        ----------
        cfg_path: str | Path
            The path to the configuration file. Must exist.

        Returns
        -------
        ConfigProxy
            The configuration file as a ConfigProxy object.
        """
        cfg_path = Path(cfg_path)
        ext = cfg_path.suffix.lower()
        loader = cls.loader_functions[ext]
        cfg = loader(cfg_path, must_exist=True)
        if cfg is None:
            raise RuntimeError(f'Could not load config from {cfg_path} using loader for {ext}')
        return cfg

    @staticmethod
    def strip_version_suffix(cfg_name: str) -> str:
        """Strip version suffix from config name, e.g. _v2_3 from sample_v2_3_params"""
        pattern = r'_v\d+_\d+$'
        return re.sub(pattern, '', cfg_name)


    @classmethod
    def get_patched_cfg_from_path(cls, cfg_path: str | Path) -> ConfigProxy:
        """
        Load a configuration file from the given path and patch it with the default configuration.
        The default configuration is loaded from the ClearMap package directory.
        The function is determined by the file extension.
        The file must exist.

        Parameters
        ----------
        cfg_path: str | Path
            The path to the configuration file. Must exist.

        Returns
        -------
        ConfigProxy
            The patched configuration file as a ConfigProxy object.
        """
        cfg_path = Path(cfg_path)
        cfg = cls.get_cfg_from_path(cfg_path)
        config_name = cls.strip_version_suffix(cfg_path.stem)
        default_cfg = cls.get_cfg_from_path(cls.get_default_path(config_name))
        patch_cfg(cfg, default_cfg)
        cfg.write()
        return cfg

    @classmethod  # FIXME: check return type expected by client code
    def get_default_path(cls, cfg_name: str, must_exist: bool = True, from_package: bool = False) -> Path:
        """
        Get the path to the default configuration file with the given name.
        Several extensions are tried in order of preference.
        If None is found, the first possible option is returned if must_exist is False or
        a FileNotFoundError is raised if must_exist is True.

        Parameters
        ----------
        cfg_name: str
            The name (without params and extension) of the configuration file
        must_exist: bool
            Whether the file must exist. If missing and True, a FileNotFoundError is raised
        from_package: bool
            If True, look for the default config in the ClearMap package directory.
            Otherwise, look in the user's ~/.clearmap directory.

        Returns
        -------
        Path
            The path to the default configuration file

        Raises
        ------
        FileNotFoundError
            If the file does not exist and must_exist is True.
        """
        if cfg_name.endswith('_params'):
            cfg_name = cfg_name.replace('_params', '')
        paths_checked = []
        for name_variant in get_alternatives(cfg_name):
            name_variant += '_params'
            for ext in cls.supported_exts:
                cfg_path = cls._name_to_default_path(f'{name_variant}_{VERSION_SUFFIX}',
                                                              ext, from_package=from_package)
                paths_checked.append(cfg_path)
                if cfg_path.exists():
                    return cfg_path
        else:
            if must_exist:
                raise FileNotFoundError(f'Could not find file {cfg_name}, checked {paths_checked}')
            else:  # Return first (default) ext if none found
                ext0 = cls.supported_exts[0]
                return cls._name_to_default_path(f'{cfg_name}_params_{VERSION_SUFFIX}', ext0, from_package=from_package)
                # return cls._name_to_default_path(f'{cfg_name}_params', cls.supported_exts[0], from_package=from_package)

    @classmethod
    def _name_to_default_path(cls, cfg_name: str, ext: str, from_package: bool = False) -> Path:
        """Helper to construct a default config path from name + ext."""
        prefix = 'default_' if is_tab_file(cfg_name) else ''
        cfg_name = f'{prefix}{cfg_name}{ext}'

        cfg_dir = INSTALL_CFG_DIR if from_package else cls.default_dir
        cfg_path = clean_path(cfg_dir / cfg_name)
        return Path(cfg_path)

    @classmethod
    def dump(cls, path: Path, data: dict) -> None:
        """
        Write the given data to the specified path using the appropriate writer function.
        The function is determined by the file extension.
        The write is atomic (writes to a temp file then renames).

        Parameters
        ----------
        path: Path
            The path to the output configuration file
        data: dict
            The data to write

        Raises
        ------
        ValueError
            If no writer function is found for the given file extension.
        """
        ext = path.suffix.lower()
        writer_fn = cls.writer_functions.get(ext)
        if not writer_fn:
            raise ValueError(f'No writer for extension {ext}')
        tmp = path.with_suffix(f'{path.suffix}.tmp')
        writer_fn(tmp, data)
        atomic_replace(tmp, path)


# FIXME: fix missing stuff here
def get_configs(cfg_path: str | Path, processing_params_path: str | Path, machine_cfg_path: str | Path | None = None):
    """
    Get the machine, sample and processing configurations from the given paths.
    If machine_cfg_path is None, the default machine config path is used.

    Parameters
    ----------
    cfg_path: str | Path
        Path to the sample configuration file
    processing_params_path: str | Path
        Path to the processing parameters configuration file
    machine_cfg_path: str | Path | None
        Path to the machine configuration file. If None, the default machine config path is used.

    Returns
    -------
    Tuple[ConfigProxy, ConfigProxy, ConfigProxy]
        A tuple containing the machine, sample and processing configurations as ConfigProxy objects.
    """
    if machine_cfg_path is None:
        machine_cfg_path = ConfigHandler.get_default_path('machine')
    sample_config = ConfigHandler.get_patched_cfg_from_path(cfg_path)
    processing_config = ConfigHandler.get_patched_cfg_from_path(processing_params_path)
    machine_config = ConfigHandler.get_patched_cfg_from_path(machine_cfg_path)

    return machine_config, sample_config, processing_config

# REFACTOR: why is this not a class method of ConfigHandler?
def get_cfg_reader_function(cfg_path: Path | str) -> Callable[[Path, bool], Optional[ConfigProxy]]:
    """
    Get the appropriate configuration reader function for the given file path.
    The function is determined by the file extension.

    Parameters
    ----------
    cfg_path: Path | str
        The path to the configuration file.

    Returns
    -------
    Callable[[Path, bool], Optional[ConfigProxy]]
        The configuration reader function.
    """
    cfg_path = Path(cfg_path)
    ext = cfg_path.suffix.lower()
    read_cfg = ConfigHandler.loader_functions[ext]
    return read_cfg
