"""
This module provides a class to load/write configuration files from the ClearMap configuration directory.

The configuration files are used to store the parameters for the ClearMap processing steps.
Supported formats are .cfg (ConfigObj), .yml/.yaml (YAML) and .json (JSON).
Other formats (TOML, INI, XML, etc.)  could be supported in the future through
 a simple plugin function to this module.
"""
import inspect
import json
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Any, Dict, List, Mapping
from functools import cached_property

import configobj
from packaging.version import Version
from importlib_metadata import version as importlib_version

from ClearMap.IO.FileUtils import atomic_replace
from .utils import _handle_configobj_failed_parse, configobj_to_dict
from ..Utils.path_utils import clean_path, de_duplicate_path
from ..Utils.utilities import title_to_snake

clearmap_version = importlib_version('ClearMap')
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


def flatten_list(groups: list[list[str]]) -> list[str]:
    """Flatten a list of lists of item groups into a single list."""
    flat: list[str] = []
    for names in groups:
        flat.extend(names)
    return flat


class Scope(Enum):
    EXPERIMENT = 'experiment'
    GROUP = 'group'
    GLOBAL = 'global'


@dataclass(frozen=True)
class ConfigSpec:
    names: tuple[str, ...]       # ordered alternatives
    scope: Scope                 # EXPERIMENT or GLOBAL
    subdir: str = ''  # optional subfolder, e.g. "configs" or "settings"

"""
List of alternative names for configuration files.
"""

PIPELINE_SECTIONS = [
    ['sample', 'sample_info', 'sample info'],
    ['stitching'],
    ['registration'],
    ['cell_map', 'cell_counter'],
    ['tract_map'],
    ['vasculature', 'tube_map'],
    ['colocalization']
]

GROUP_SECTIONS = [
    ['batch_processing', 'batch', 'batch processing'],
    ['group_analysis', 'group analysis'],
]

GLOBAL_SECTIONS = [
    ['machine', 'preferences'],
    ['display'],
]

LEGACY_SECTIONS = [
    ['alignment', 'processing'],  # legacy names for stitching/registration tab
]


class ConfigAlternativesRegistry:
    """
    Registry of alternative names for configuration files.
    """
    def __init__(self):
        self._pipeline_groups: List[List[str]] = [list(g) for g in PIPELINE_SECTIONS]
        self._group_groups: List[List[str]] = [list(g) for g in GROUP_SECTIONS]
        self._global_groups: List[List[str]] = [list(g) for g in GLOBAL_SECTIONS]
        self._legacy_groups: List[List[str]] = [list(g) for g in LEGACY_SECTIONS]

        self._registry: Dict[str, ConfigSpec] = {}

        # experiment and group specs
        self._extend_registry(self._pipeline_groups, scope=Scope.EXPERIMENT)
        self._extend_registry(self._group_groups, scope=Scope.GROUP)
        self._extend_registry(self._global_groups, scope=Scope.GLOBAL)
        # compatibility specs
        for group in self._legacy_groups:  # FIXME: we need a flag to mark these as deprecated
            for name in group:
                self._registry.setdefault(name, ConfigSpec(names=tuple(group), scope=Scope.EXPERIMENT))

    def _extend_registry(self, groups, scope: Scope):
        for group in groups:
            spec = ConfigSpec(names=tuple(group), scope=scope)
            for name in group:
                self._registry[name] = spec

    @staticmethod
    def _normalise_name(cfg_name) -> str:
        cfg_name = ConfigHandler.strip_params(cfg_name)
        cfg_name = ConfigHandler.strip_version_suffix(cfg_name)
        return cfg_name

    def get_spec(self, cfg_name: str) -> ConfigSpec:  # REFACTOR: unused
        """
        Get the ConfigSpec for a given configuration name.

        Parameters
        ----------
        cfg_name: str
            The base name of the configuration file (without params and extension).

        Returns
        -------
        ConfigSpec
            The configuration specification.

        Raises
        ------
        KeyError
            If no specification is found for the given configuration name.
        """
        return self._registry[self._normalise_name(cfg_name)]

    @cached_property
    def alternative_names(self) -> list[list[str]]:
        """
        Get all alternative names registered.

        Returns
        -------
        list[str]
            A list of all alternative configuration names.
        """
        return list(self._pipeline_groups + self._group_groups + self._global_groups + self._legacy_groups)

    def get_alternatives(self, cfg_name: str) -> list[str]:
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
        cfg_name = self._normalise_name(cfg_name)
        alternatives = [names for names in self.alternative_names if cfg_name in names]
        if not alternatives:
            raise ValueError(f'Could not find any alternative for {cfg_name}')
        return alternatives[0]

    def is_local_file(self, cfg_name: str) -> bool:
        """Check if the given config name is a tab file (i.e. has alternatives)."""
        return self._normalise_name(cfg_name) in flatten_list(self._pipeline_groups + self._group_groups)

    def is_global_cfg(self, cfg_name: str) -> bool:
        """Check if the given config name is a global config (i.e. machine, display, preferences)."""
        return self._normalise_name(cfg_name) in flatten_list(self._global_groups)

    def is_legacy_cfg(self, cfg_name: str) -> bool:
        """Check if the given config name is a legacy config (i.e. alignment, processing)."""
        return self._normalise_name(cfg_name) in flatten_list(self._legacy_groups)

    def pipeline_to_section_name(self, pipeline_name: str) -> str:
        """
        Convert a pipeline name to a configuration section name.
        E.g. 'TubeMap' -> 'vasculature'

        Parameters
        ----------
        pipeline_name: str
            The name of the pipeline.

        Returns
        -------
        str
            The corresponding configuration section name.
        """
        tentative_section_name = title_to_snake(pipeline_name)
        for group in self.alternative_names:
            if tentative_section_name in group:
                return group[0]  # return canonical name
        return tentative_section_name  # assume name correct as is if all else fails

    @cached_property
    def canonical_config_names(self):
        """
        Get the canonical configuration names (first in each alternative group).

        Returns
        -------
        list[str]
            A list of canonical configuration names.
        """
        return [names[0] for names in (self._pipeline_groups + self._group_groups + self._global_groups)]

    @cached_property
    def canonical_pipeline_config_names(self):
        """
        Get the canonical configuration names for pipeline sections.

        Returns
        -------
        list[str]
            A list of canonical configuration names for pipeline sections.
        """
        return [names[0] for names in self._pipeline_groups]

    def to_canonical(self, cfg_name):
        return self.get_alternatives(cfg_name)[0]

    @staticmethod
    def get_channel_sections() -> tuple[str, ...]:  # REFACTOR: could use 'channels' in schema root for schema in schemas
        return ("sample", "stitching", "registration", "cell_map",
            "colocalization", "vasculature", "tract_map",)


ALTERNATIVES_REG = ConfigAlternativesRegistry()


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

# #################  WRITER FUNCTIONS  ####################

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
    cfg = configobj.ConfigObj(encoding='UTF8', indent_type='    ', unrepr=True)
    cfg.filename = str(path)
    cfg.clear()
    for k, v in data.items():
        cfg[k] = v
    cfg.write()

PRIORITY_KEYS = ('clearmap_version', 'clearmap_schema')

def prioritize_top_keys(obj, first=PRIORITY_KEYS):
    """Return a shallowly-reordered top-level mapping."""
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k in first:
        if k in obj:
            out[k] = obj[k]
    for k, v in obj.items():
        if k not in out:
            out[k] = v
    return out

class FlowList(list):
    """Marker type for inline (flow-style) sequences."""
    pass

def represent_flow_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowList, represent_flow_list)
yaml.add_representer(FlowList, represent_flow_list, Dumper=yaml.SafeDumper)

_SCALARS = (str, int, float, bool, type(None))

def is_all_scalars(seq):
    return all(isinstance(x, _SCALARS) for x in seq)

def mark_inline_sequences(obj, max_items=3):
    """Inline short sequences of scalars; convert tuples to lists (portable YAML)."""
    if isinstance(obj, dict):
        return {k: mark_inline_sequences(v, max_items) for k, v in obj.items()}

    # Treat both lists and tuples as sequences
    if isinstance(obj, (list, tuple)):
        # Recurse into children first
        items = [mark_inline_sequences(v, max_items) for v in obj]
        # Decide flow/block
        if len(items) <= max_items and is_all_scalars(items):
            return FlowList(items)        # -> [a, b, c]
        return items                      # -> block style with dashes

    return obj

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
    pretty = mark_inline_sequences(data, max_items=3)
    pretty = prioritize_top_keys(pretty)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(pretty, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

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

# #################  READER FUNCTIONS  ####################

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

    cfg = ConfigProxy(
        filename=str(cfg_path),
        _loader=lambda p: get_configobj_cfg(p, must_exist=True),  # type: ignore
        _dumper=to_configobj
    )
    cfg.update(configobj_to_dict(cobj))
    return cfg


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
    if not cfg_path.exists():
        if must_exist:
            raise FileNotFoundError(f'YAML config file {cfg_path} does not exist.')
        return None
    with cfg_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'YAML root must be a mapping; got {type(data)} in {cfg_path}')
    cfg = ConfigProxy(
        filename=str(cfg_path),
        _loader=lambda p: get_yml_cfg(p, must_exist=True),  # type: ignore
        _dumper=to_yml)
    cfg.update(data)
    return cfg


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
    if not cfg_path.exists():
        if must_exist:
            raise FileNotFoundError(f'JSON config file {cfg_path} does not exist.')
        return None
    with cfg_path.open('r', encoding='utf-8') as f:
        data = json.load(f) if cfg_path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError(f'JSON root must be a mapping; got {type(data)} in {cfg_path}')
    cfg = ConfigProxy(
        filename=str(cfg_path),
        _loader=lambda p: get_json_cfg(p, must_exist=True),  # type: ignore
        _dumper=to_json
    )
    cfg.update(data)
    return cfg
# #################################################################


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
    user_defaults_dir: Path
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
        ".yml": get_yml_cfg,
        ".yaml": get_yml_cfg,
        ".cfg": get_configobj_cfg,
        ".ini": get_configobj_cfg,
        ".json": get_json_cfg,
    }
    # Writers by extension
    writer_functions: Dict[str, Callable[[Path, dict], None]] = {
        ".yml": to_yml,
        ".yaml": to_yml,
        ".cfg": to_configobj,
        ".ini": to_configobj,
        ".json": to_json,
    }
    supported_exts = tuple(loader_functions.keys())
    user_defaults_dir = CLEARMAP_CFG_DIR  # Where to look for user defaults
    user_global_dir = CLEARMAP_CFG_DIR / 'config'  # Where to look for user global configs

    def __init__(self, src_dir: Path | str):
        self._src_dir = None
        self.roots = {
            "experiment": Path.cwd(),  # overridden by repository via src_dir setter
            "group": Path.cwd(),  # same as experiment for now
            "global": self.user_defaults_dir,  # ~/.clearmap
        }  # Before setting src_dir because setter uses it
        self.src_dir = src_dir

    @property
    def src_dir(self):
        return self._src_dir

    @src_dir.setter
    def src_dir(self, value: str | Path):
        self._src_dir = Path(value).expanduser().resolve()
        self.roots['experiment'] = self._src_dir
        self.roots['group'] = self._src_dir

    @staticmethod
    def is_global(cfg_name: str) -> bool:
        return ALTERNATIVES_REG.is_global_cfg(cfg_name)

    @staticmethod
    def is_local(cfg_name: str) -> bool:
        return ALTERNATIVES_REG.is_local_file(cfg_name)

    @staticmethod
    def strip_params(cfg_name: str, params_pattern: str = '_params') -> str:
        """Strip params from config name, e.g. _v2_params from sample_v2_params"""
        return re.sub(params_pattern, '', cfg_name)

    @staticmethod
    def strip_version_suffix(cfg_name: str) -> str:
        """Strip version suffix from config name, e.g. _v2_3 from sample_v2_3_params"""
        pattern = r'_v\d+_\d+$'
        return re.sub(pattern, '', cfg_name)

    @staticmethod
    def normalise_cfg_name(cfg_name: str) -> str:
        """Normalise config name by stripping params and version suffix."""
        name = str(cfg_name)
        name = ConfigHandler.strip_params(name)
        name = ConfigHandler.strip_version_suffix(name)
        return name

    @staticmethod
    def _layout_subdir_for_version(version: str, defaults: bool) -> Path:
        v = Version(version)
        if v >= Version('3.1'):  # >= 3.1 has “defaults/vX.Y/”
            prefix = Path('defaults') if defaults else Path()
            return prefix / f'v{v.major}.{v.minor}'
        else:
            return Path()

    @staticmethod
    def _filename_for(cfg_base: str, ext: str, *, version: str, is_exp_local: bool) -> str:
        v = Version(version)
        base = cfg_base  # already stripped of *_params and version bits before
        if v >= Version('3.1'):  # New layout: no prefix/suffix in filename
            return f'{base}{ext}'
        else:  # Legacy (<= 3.0)
            prefix = 'default_' if is_exp_local else ""
            suffix = '_params'
            if v == Version('3.0'):  # v3.0 specifically had _v3_0 suffix
                suffix += '_v3_0'
            return f'{prefix}{base}{suffix}{ext}'

    # ####### Canonical (where it should be in that version, don't scan) ##########
    def get_local_canonical_path(self, cfg_name: str) -> Path:
        canonical_name = ALTERNATIVES_REG.to_canonical(cfg_name)
        canonical_filename = self._filename_for(canonical_name, self.supported_exts[0],
                                                version=clearmap_version, is_exp_local=True)
        return self.src_dir / canonical_filename

    @classmethod
    def get_global_canonical_path(cls, cfg_name: str) -> Path:
        canonical_name = ALTERNATIVES_REG.to_canonical(cfg_name)
        return cls._name_to_shared_path(canonical_name, cls.supported_exts[0], root=cls.user_global_dir,
                                        version=clearmap_version, defaults=False)

    @classmethod
    def get_user_defaults_canonical_path(cls, cfg_name: str) -> Path:
        """
        Canonical path for a user default config for the *current* ClearMap version.

        - No scanning / fallback across versions
        - Uses current clearmap_version
        - Always uses the given ext (default: .yml for 3.1+)
        """
        canonical = ALTERNATIVES_REG.to_canonical(cls.normalise_cfg_name(cfg_name))
        return cls._name_to_shared_path(canonical, cls.supported_exts[0],
                                        root=cls.user_defaults_dir,
                                        version=str(Version(clearmap_version)),
                                        defaults=True)

    def get_canonical_path(self, cfg_name: str) -> Path:
        """
        Get the canonical path to the configuration file with the given name.
        The path is determined based on whether the config is local (tab/experiment)
        or global (machine/display/preferences).
        The rest is determined by the current ClearMap version and layout as well as preferred extension.

        Parameters
        ----------
        cfg_name: str
            The name (without params and extension) of the configuration file.

        Returns
        -------
        Path
            The canonical path to the configuration file
        """
        cfg_name = self.normalise_cfg_name(cfg_name)
        if ALTERNATIVES_REG.is_global_cfg(cfg_name):
            return self.get_global_canonical_path(cfg_name)
        elif ALTERNATIVES_REG.is_local_file(cfg_name):
            return self.get_local_canonical_path(cfg_name)
        else:
            raise ValueError(f'Config "{cfg_name}" is neither local nor global (maybe legacy); '
                             f'cannot get canonical path.')

    @classmethod
    def resolve_write_path(cls, name: str, *, base_dir: Path) -> Path:
        """
        Resolve the *target* path to write a config by name.
        - Tab/experiment sections -> experiment base_dir
        - Global sections (machine/display/preferences) -> user global directory
        """
        loader = cls(base_dir)
        return loader.get_canonical_path(name)

    @classmethod
    def _name_to_shared_path(cls, cfg_name: str, ext: str, *, root: Path,
                             version: Optional[str] = None, defaults: bool = True) -> Path:
        """
        Helper to construct a *shared* (i.e. non experiment-local) config path from name + ext.
        The path is either in the ClearMap package directory or in the user's ~/.clearmap directory.

        This is typically used for both default (package or user) and global (user) configs.

        The folder structure depends on the version and `defaults` flag:
          - defaults=True  → .../defaults/vX.Y/<name>.<ext>  (>= 3.1)
          - defaults=False → .../vX.Y/<name>.<ext>           (>= 3.1)
          - < 3.1          → root / legacy naming patterns

        Parameters
        ----------
        cfg_name: str
            The name (without params and extension) of the configuration file
        ext: str
            The file extension
        root: Path
            The root directory to look for the default config file.
        version: Optional[str]
            The version string in the major.minor format to use for the folder structure.
             If None, the current ClearMap version is used.

        """
        v = Version(version or clearmap_version)  # default to latest layout
        base = cls.strip_params(cfg_name)
        is_exp_local = ALTERNATIVES_REG.is_local_file(base)

        sub_dir = cls._layout_subdir_for_version(str(v), defaults=defaults)
        sub_dir = de_duplicate_path(root, sub_dir)  # avoid duplicate subdir suffix

        file_name = cls._filename_for(base, ext, version=str(v), is_exp_local=is_exp_local)
        return Path(root / sub_dir / file_name).expanduser().resolve()

    # _find paths: ####### Scan FS for existing files ##########
    def _find_local_path(self, cfg_name: str) -> Path | None:  # TODO: check
        """
        Scan the source directory for a *local* configuration file with the given name.
        The names are resolved via the alternatives registry.
        Several extensions are tried in order of preference.
        If None is found, None is returned.

        Parameters
        ----------
        cfg_name: str
            The name (with or without params and extension) of the configuration file

        Returns
        -------
        Path | None
            The path to the configuration file, or None if not found.
        """
        cfg_name = self.normalise_cfg_name(cfg_name)
        try:
            variants = ALTERNATIVES_REG.get_alternatives(cfg_name)
        except ValueError:
            variants = [cfg_name]

        src_dir = self.src_dir

        # Support for legacy file name patterns, newer to older
        search_patterns = [
            lambda base, ext: [src_dir / f"{base}{ext}"],  # >= 3.1 default_base.ext
            lambda base, ext: sorted(src_dir.glob(f"{base}_v*_*{ext}")),  # v3.0. WARNING: needs re if we want to limit to integers
            lambda base, ext: [src_dir / f"{base}_params{ext}"], # <= 3.0 default_base_params.ext
            lambda base, ext: sorted(src_dir.glob(f"{base}_params_v*_*{ext}"))  # Do we really have this ?
        ]

        for alternative_name in variants:
            for extension  in self.supported_exts:
                for ptrn_func in search_patterns:
                    for cfg_path in ptrn_func(alternative_name, extension):
                        if cfg_path.exists():
                            return cfg_path
        else:
            return None

    @classmethod
    def _find_shared_cfg_path(cls, cfg_name: str, *, root: Path,
                              version: Optional[str], defaults: bool) -> Path | None:
        """
        Get the path to a *shared* configuration file with the given name.
        Several extensions are tried in order of preference.
        If None is found, None is returned.

        This is used for both default (package or user) and global (user) configs.

        Parameters
        ----------
        cfg_name: str
            The name (without params and extension) of the configuration file
        must_exist: bool
            Whether the file must exist. If missing and True, a FileNotFoundError is raised
        root: Path
            The base directory to look for the default config file.
            If None, the directory is determined by from_package.
            Typically, this would be either the ClearMap package directory or the user's ~/.clearmap directory.
        version: Optional[str]
            The version string in the major.minor format to use for the folder structure.
             If None, the current ClearMap version is used.
        defaults: bool
            If True, look in the 'defaults' subdirectory (for >=v3.1).
            Otherwise, look at the root of base_dir. (or what it resolves to)

        Returns
        -------
        Path
            The path to the default configuration file

        Raises
        ------
        FileNotFoundError
            If the file does not exist and must_exist is True.
        """
        cfg_name = cls.normalise_cfg_name(cfg_name)

        versions_to_try = [version] if version else (clearmap_version, '3.0', '2.1')  # Currently 3 kinds of layout (read up to vX.Y)

        paths_checked = []  # FIXME: include this in caller errors
        for base in ALTERNATIVES_REG.get_alternatives(cfg_name):
            for ext in cls.supported_exts:
                for ver in versions_to_try:
                    candidate = cls._name_to_shared_path(base, ext, root=root, version=ver, defaults=defaults)
                    paths_checked.append(candidate)
                    if candidate.exists():
                        return candidate
        else:
            return None

    # ############## Get Paths: scan or canonical ###############
    def get_cfg_path(self, cfg_name: str, must_exist: bool = True) -> Path:
        """
        Get the path to the configuration file with the given name.
        Several extensions are tried in order of preference.
        If None is found, the canonical path is returned if must_exist is False or
        a FileNotFoundError is raised if must_exist is True.

        Parameters
        ----------
        cfg_name: str
            The name (without params and extension) of the configuration file.
            If present, they will be stripped.
        must_exist: bool
            Whether the file must exist. If missing and True, a FileNotFoundError is raised.

        Returns
        -------
        Path
            The path to the configuration file
        """
        cfg_name = self.normalise_cfg_name(cfg_name)
        if ALTERNATIVES_REG.is_global_cfg(cfg_name):
            raise ValueError(f'Config "{cfg_name}" is global; use get_global_path() instead of get_cfg_path().')
        cfg_path = self._find_local_path(cfg_name)
        if cfg_path is None:
            if must_exist:
                raise FileNotFoundError(f'Could not find file {cfg_name} in {self.src_dir} with variants:'
                                        f'{ALTERNATIVES_REG.get_alternatives(cfg_name)}')
            else:
                return self.get_local_canonical_path(cfg_name)
        else:
            return cfg_path

    @classmethod
    def get_global_path(cls, cfg_name: str, must_exist: bool = True, *, base_dir: Path | None = None,
                        version: Optional[str] = None) -> Path | None:
        """
        Get the path to the global configuration file with the given name.
        Several extensions are tried in order of preference.
        If None is found, the first possible option is returned if must_exist is False or
        a FileNotFoundError is raised if must_exist is True.

        Parameters
        ----------
        cfg_name: str
            The name (without params and extension) of the configuration file
        must_exist: bool
            Whether the file must exist. If missing and True, a FileNotFoundError is raised
        base_dir: Optional[Path]
            The base directory to look for the global config file.
            If None, the user's ~/.clearmap directory is used.
        version: Optional[str]
            The version string in the major.minor format to use for the folder structure.

        Returns
        -------
        Path
            The path to the global configuration file
        """
        root = Path(base_dir) if base_dir is not None else cls.user_global_dir
        version = version or clearmap_version

        if not must_exist:
            return cls._find_shared_cfg_path(cfg_name, root=root, version=version, defaults=False)
        else:
            global_path = cls._find_shared_cfg_path(cfg_name, root=root, version=version, defaults=False)
            if global_path is not None:
                return global_path
            else:
                if base_dir is not None:
                    raise FileNotFoundError(f'Could not find global config {cfg_name} in {base_dir}')
                if not must_exist:
                    loader = cls(root)
                    return loader.get_global_canonical_path(cfg_name)
                else:
                    legacy_root = cls.user_defaults_dir  # Old user_global_dir was same as default
                    global_path = cls._find_shared_cfg_path(cfg_name, root=legacy_root, version=version, defaults=False)
                if global_path is None and must_exist:
                    raise FileNotFoundError(f'Could not find global config "{cfg_name}" in "{root}" or "{legacy_root}"')
                else:
                    return global_path

    @classmethod
    def get_default_path(cls, cfg_name: str, must_exist: bool = True, *, base_dir: Path | None = None,
                         version: Optional[str] = None, from_package: bool = False) -> Path | None:
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
        base_dir: Optional[Path]
            The base directory to look for the default config file.
            If specified, this overrides the from_package parameter.
        version: Optional[str]
            The version string in the major.minor format to use for the folder structure.
            If None, the current ClearMap version is used.
        from_package: bool
            If True, look for the default config in the ClearMap package directory.
            Otherwise, look in the user's ~/.clearmap directory.
            .. warning::
                This only affects the default path if no base_dir is specified.

        Returns
        -------
        Path
            The path to the default configuration file
        """
        version = version or clearmap_version
        if (base_dir is None
            and not from_package
            and ALTERNATIVES_REG.is_global_cfg(cfg_name)
            and Version(version) >= Version('3.1')
        ):
            raise ValueError(f'Config "{cfg_name}" is global; use get_global_path() instead of get_default_path().')
        root = Path(base_dir) if base_dir is not None else (
            INSTALL_CFG_DIR if from_package else cls.user_defaults_dir
        )
        default_path = cls._find_shared_cfg_path(cfg_name, root=root, version=version, defaults=True)
        if default_path is not None:
            return default_path
        else:
            if must_exist:
                raise FileNotFoundError(f'Could not find default config "{cfg_name}" in "{root}"')
            else:
                canonical = ALTERNATIVES_REG.to_canonical(cfg_name)
                return cls._name_to_shared_path(canonical, cls.supported_exts[0], root=root,
                                                version=version, defaults=True)

    # ############# The configs themselves ###############
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
            cfg_path = Path(cfg_name).expanduser().resolve()
        else:
            if ALTERNATIVES_REG.is_local_file(cfg_name):
                cfg_path = self.get_cfg_path(cfg_name, must_exist=must_exist)
            elif ALTERNATIVES_REG.is_global_cfg(cfg_name):
                cfg_path = self.get_global_path(cfg_name, must_exist=must_exist)
            else:
                cfg_path = self.get_default_path(cfg_name, must_exist=must_exist)
        if not cfg_path.exists():
            if must_exist:
                raise FileNotFoundError(f'Could not find file {cfg_name} in {self.src_dir} @ {cfg_path}')
            else:
                return None
        return self.get_cfg_from_path(cfg_path)

    @classmethod
    def get_default_cfg(cls, cfg_name):
        try:
            return cls.get_cfg_from_path(cls.get_default_path(cfg_name, from_package=False))
        except FileNotFoundError:
            return cls.get_cfg_from_path(cls.get_default_path(cfg_name, from_package=True))

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
        if not cfg_path.exists():
            raise FileNotFoundError(f'Config file {cfg_path} does not exist.')
        ext = cfg_path.suffix.lower()
        loader = cls.loader_functions[ext]
        cfg = loader(cfg_path, must_exist=True)
        if cfg is None:
            raise RuntimeError(f'Could not load config from {cfg_path} using loader for {ext}')
        return cfg

    @classmethod
    def get_new_cfg_writer(cls, cfg_path: str | Path, *, must_not_exist: bool = True) -> ConfigProxy:
        """
        Create a new, empty ConfigProxy for cfg_path, without reading from disk.

        Parameters
        ----------
        cfg_path : str | Path
            Target config path (may or may not exist).
        must_not_exist : bool
            If True, raise if the file already exists.

        Returns
        -------
        ConfigProxy
            An empty ConfigProxy bound to cfg_path with the proper writer.
        """
        path = Path(cfg_path).expanduser().resolve()

        if must_not_exist and path.exists():
            raise FileExistsError(f'Config file {path} already exists.')

        ext = path.suffix.lower()
        dumper = cls.writer_functions.get(ext)
        if dumper is None:
            raise ValueError(f'No writer for extension {ext}')

        return ConfigProxy(
            filename=str(path),
            _loader=lambda p: cls.get_cfg_from_path(p),  # standard “read me back” loader
            _dumper=dumper,
        )

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

        if isinstance(data, Mapping):  # ConfigProxy
            payload = dict(data)
        else:
            payload = data
        writer_fn(tmp, payload)
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
        machine_cfg_path = ConfigHandler.get_global_path('machine')
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
