from __future__ import annotations

import shutil
import tempfile
import warnings
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Tuple

from packaging.version import Version
from importlib_metadata import version as importlib_version

import qdarkstyle
from PyQt5.QtWidgets import QApplication, QDialog

from ClearMap.IO.assets_constants import CHANNELS_ASSETS_TYPES_CONFIG
from ClearMap.Utils.tag_expression import Expression
from ClearMap.config.config_handler import ConfigHandler, ALTERNATIVES_REG

from ClearMap.gui.dialogs import RenameChannelsDialog, VerifyRenamingDialog
from ClearMap.gui.dialog_helpers import get_directory_dlg
from ClearMap.gui.gui_utils_base import ensure_qapp

clearmap_version = importlib_version('ClearMap')
VERSION_SUFFIX = f'v{Version(clearmap_version).major}_{Version(clearmap_version).minor}'


SUPPORTED_VERSIONS = [Version(v) for v in ('2.1', '3.0', '3.1')]

def _norm_ver(v) -> Version:
    return v if isinstance(v, Version) else Version(str(v))

def read_cfg(path: Path | str, must_exist: bool = True):
    path = Path(path).expanduser().resolve()
    if must_exist or path.exists():  # Read intent
        return ConfigHandler.get_cfg_from_path(path)
    else:  # write intent
        return ConfigHandler.get_new_cfg_writer(path)


# FILE_CONVERTERS[(from_v, to_v)][config_type] = converter_func
FILE_CONVERTERS: Dict[Tuple[Version, Version], Dict[str, Callable]] = {}
GLOBAL_FILE_CONVERTERS: Dict[tuple[Version, Version], Dict[str, Callable]] = {}

# PROJECT_CONVERTERS[(from_v, to_v)] = converter_func
PROJECT_CONVERTERS: Dict[Tuple[Version, Version], Callable] = {}


# The decorator to automatically add exp converters to the registry
def cfg_converter(from_v: str, to_v: str, config_type: str):
    """
    Register a file-level converter.

    Usage:
        @register_file_converter('3.0', '3.1', 'sample')
        def convert_sample_3_0_to_3_1(cfg_path: Path) -> Path:
            ...
    """
    from_v = _norm_ver(from_v)
    to_v = _norm_ver(to_v)
    def decorator(func: Callable) -> Callable:
        key = (from_v, to_v)
        if key not in FILE_CONVERTERS:
            FILE_CONVERTERS[key] = {}
        FILE_CONVERTERS[key][config_type] = func
        return func

    return decorator


def version_guard(from_v, to_v, key: str = 'clearmap_version'):
    """
    Wrap a converter that expects `config_v1, *args, **kwargs`.

    - Raises if version != from_v
    - Warns & returns filename if version == to_v
    """
    from_v = _norm_ver(from_v)
    to_v = _norm_ver(to_v)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(v1_path, *args, **kwargs):
            cfg = read_cfg(v1_path)
            current = Version(str(cfg.get(key, '0.0.0')))
            if current == to_v:
                warnings.warn(f'Config already in version {to_v}')
                return cfg.filename
            if current != from_v:
                raise ValueError(
                    f'Only version {from_v} is supported (got {current})'
                )
            return func(v1_path, *args, **kwargs)
        return wrapper
    return decorator


def global_cfg_converter(from_v, to_v, config_type: str):
    from_v = _norm_ver(from_v)
    to_v = _norm_ver(to_v)
    def decorator(func: Callable) -> Callable:
        GLOBAL_FILE_CONVERTERS.setdefault((from_v, to_v), {})[config_type] = func
        return func
    return decorator


# Project level converter decorator
def project_converter(from_v: str, to_v: str):
    """
    Register a project-level converter.

    Usage:
        @register_project_converter('3.0', '3.1')
        def convert_project_3_0_to_3_1(exp_dir: Path) -> None:
            ...
    """
    from_v = _norm_ver(from_v)
    to_v = _norm_ver(to_v)

    def decorator(func: Callable) -> Callable:
        PROJECT_CONVERTERS[(from_v, to_v)] = func
        return func

    return decorator


def detect_config_type(cfg_path: Path) -> tuple[str, str]:
    """
    Return (canonical_name, scope) where scope is 'experiment' or 'global'.
    """
    name = ALTERNATIVES_REG.to_canonical(cfg_path.stem)
    scope = 'global' if ALTERNATIVES_REG.is_global_cfg(name) else 'experiment'
    return name, scope


def get_configs(v1_path, v2_path=''):
    v1_path = Path(v1_path).expanduser()
    if not v2_path:
        v2_path = v1_path.with_name(f'{v1_path.stem}_{VERSION_SUFFIX}.cfg')  # can't use with_suffix without a dot
    v2_path = v2_path.expanduser()
    return read_cfg(v1_path), read_cfg(v2_path, must_exist=False)


@cfg_converter('2.1', '3.0', 'sample')
@version_guard('2.1', '3.0')
def convert_sample_2_1_to_3_0(v1_path, v2_path=''):
    v1_path = Path(v1_path).expanduser().resolve()
    config_v1, config_v2 = get_configs(v1_path, v2_path)
    def v1_or_default(key):
        return config_v1.get(key, default_cfg[key])

    default_cfg = ConfigHandler.get_default_cfg('sample')

    # Copy general parameters
    config_v2['clearmap_version'] = '3.0.0'
    config_v2['base_directory'] = config_v1.get('base_directory', '')
    config_v2['sample_id'] = config_v1.get('sample_id', '')
    config_v2['use_id_as_prefix'] = v1_or_default('use_id_as_prefix')
    config_v2['default_tile_extension'] = v1_or_default('default_tile_extension')
    config_v2['comments'] = config_v1.get('comments', '')
    config_v2['channels'] = {}

    # Copy sample parameters
    channel_names = [k for k, v in config_v1['resolutions'].items() if v]
    for i, channel_name in enumerate(channel_names):
        if config_v1['src_paths'][channel_name] is None:
            continue
        config_v2['channels'][channel_name] = {
            'data_type': None,
            'extension': config_v1['src_paths']['tile_extension'],  # TODO: check if .get with dot
            'path': config_v1['src_paths'][channel_name],
            'resolution': config_v1['resolutions'][channel_name],
            'orientation': config_v1['orientation'],
            'comments': '',
            'slicing': {k: config_v1[f'slice_{k}'] for k in 'xyz'}
        }

    config_v2.write()
    return config_v2.filename


@cfg_converter('2.1', '3.0', 'cell_map')
@version_guard('2.1', '3.0')
def convert_cell_map_2_1_to_3_0(v1_path, v2_path, channel_name='channel_0'):
    config_v1, config_v2 = get_configs(v1_path, v2_path)

    config_v2['clearmap_version'] = '3.0.0'
    config_v2[channel_name] = {
        'detection': config_v1['detection'],
        'cell_filtration': config_v1['cell_filtration'],
        'voxelization': config_v1['voxelization'],
        'run': config_v1['run'],
    }

    config_v2.write()
    return config_v2.filename


@cfg_converter('2.1', '3.0', 'alignment')
@version_guard('2.1', '3.0')
def convert_alignment_2_1_to_3_0(v1_path, v2_path='', sample_config=None):
    config_v1 = read_cfg(v1_path)
    v2_path = v2_path or v1_path

    out_stitching_cfg = _alignment_to_stitching_v3(v2_path, config_v1)

    out_registration_cfg = _alignment_to_registration_v3(v2_path, config_v1, sample_config)

    return out_stitching_cfg.filename, out_registration_cfg.filename


def _alignment_to_stitching_v3(output_path_base, config_v1):
    out_stitching_cfg = read_cfg(output_path_base.with_name(f'stitching_params.cfg'), must_exist=False)
    # Copy general parameters
    out_stitching_cfg['clearmap_version'] = '3.0.0'
    out_stitching_cfg['channels'] = {}
    # drop pipeline_name and conversion section
    # Copy stitching parameters
    channel_names = [k for k, use in config_v1['stitching']['run'].items() if use]
    for i, channel in enumerate(channel_names):
        out_stitching_cfg['channels'][channel] = {
            'use_npy': config_v1['conversion']['use_npy'],
            'run': config_v1['stitching']['run'][channel],
            'layout_channel': channel,
        }
        if i == 0:  # By default, stitch other channels to the first one
            # This (dict(...)) is weird but required to get the extra indentation level
            out_stitching_cfg['channels'][channel]['rigid'] = dict(config_v1['stitching']['rigid'])
            # Rename project_thickness to projection_thickness
            out_stitching_cfg['channels'][channel]['rigid']['projection_thickness'] = (
                out_stitching_cfg['channels'][channel]['rigid'].pop('project_thickness'))
            out_stitching_cfg['channels'][channel]['wobbly'] = dict(config_v1['stitching']['wobbly'])
    out_stitching_cfg.write()
    return out_stitching_cfg


def _alignment_to_registration_v3(output_path_base, config_v1, sample_config):
    out_registration_cfg = read_cfg(output_path_base.with_name(f'registration_params.cfg'), must_exist=False)
    out_registration_cfg['clearmap_version'] = '3.0.0'
    # Copy registration parameters
    out_registration_cfg['verbose'] = config_v1['registration']['resampling']['verbose']
    out_registration_cfg['atlas'] = {k: config_v1['registration']['atlas'][k]
                                     for k in ('id', 'structure_tree_id', 'align_files_folder')}
    autofluo_params_files = [v for k, v in config_v1['registration']['atlas'].items() if k.startswith('align_reference')]
    resample = not (config_v1['registration']['resampling']['skip'])

    channel_names = sample_config['channels'].keys()
    reference_channel = [c for c, v in sample_config['channels'].items() if v['data_type'] == 'autofluorescence']
    if not reference_channel:
        reference_channel = 'autofluorescence'
    else:
        reference_channel = reference_channel[0]

    out_registration_cfg['channels'] = {
        reference_channel: {
            'resample': resample,
            'resampled_resolution': config_v1['registration']['resampling']['autofluo_sink_resolution'],
            'align_with': 'atlas',
            'moving_channel': 'atlas',
            'params_files': autofluo_params_files,
            'landmarks_weights': [0] * len(autofluo_params_files),
        }
    }

    for channel in channel_names:
        if channel not in out_registration_cfg['channels']:
            out_registration_cfg['channels'][channel] = {
                'resample': resample,
                'resampled_resolution': config_v1['registration']['resampling']['raw_sink_resolution'],
                'align_with': reference_channel,
                'moving_channel': reference_channel,
                'params_files': [config_v1['registration']['atlas']['align_channels_affine_file']],
                'landmarks_weights': [0],
            }
    out_registration_cfg.write()
    return out_registration_cfg

@global_cfg_converter('2.1', '3.0', 'machine')
def convert_machine_2_1_to_3_0(v1_path, v2_path=''):
    return v1_path


def get_conversion_steps(previous_version: Version, new_version: Version) -> list[tuple[Version, Version]]:
    """
    Return a list of (from_v, to_v) steps to go from previous_version to new_version.

    Uses SUPPORTED_VERSIONS ordering; only upgrades are supported.
    """
    if previous_version == new_version:
        return []

    try:
        i = SUPPORTED_VERSIONS.index(previous_version)
        j = SUPPORTED_VERSIONS.index(new_version)
    except ValueError:
        raise NotImplementedError(
            f'Unsupported version(s): {previous_version} or {new_version}. '
            f'Supported: {SUPPORTED_VERSIONS}'
        )

    if i > j:
        raise NotImplementedError('Downgrades are not supported.')

    return [(SUPPORTED_VERSIONS[k], SUPPORTED_VERSIONS[k + 1]) for k in range(i, j)]


def convert(cfg_path, *, prev_v: str | Version = '2.1', new_v: str | Version = '3.0',
            backup: bool = False, overwrite: bool = False):
    """
    Convert a single config file between versions.

    - Detects type & scope via ALTERNATIVES_REG.
    - Chains multiple conversion steps if needed (2.1 → 3.0 → 3.1).
    """
    cfg_path = Path(cfg_path).expanduser().resolve()
    cfg_type, scope = detect_config_type(cfg_path)

    prev_v = _norm_ver(prev_v)
    new_v = _norm_ver(new_v)

    steps = get_conversion_steps(prev_v, new_v)
    if not steps:
        return cfg_path

    if backup:
        bak = cfg_path.with_suffix('.bak')
        if bak.exists():
            raise FileExistsError(f'Backup file already exists: {bak}')
        shutil.copyfile(cfg_path, bak)

    # Sample config is needed for alignment/processing conversions
    sample_cfg = None
    if scope == 'experiment' and cfg_type != 'sample':
        cfg_loader = ConfigHandler(cfg_path.parent)
        sample_cfg_path = cfg_loader.get_cfg_path('sample', must_exist=False)
        if Path(sample_cfg_path).exists():
            sample_cfg = read_cfg(sample_cfg_path)

    current_path = cfg_path

    for (from_v, to_v) in steps:
        if scope == 'global':
            registry = GLOBAL_FILE_CONVERTERS
        else:
            registry = FILE_CONVERTERS

        converters = registry.get((from_v, to_v), {})
        func = converters.get(cfg_type)
        if not func:
            raise NotImplementedError(
                f'No converter for {cfg_type} ({scope}) {from_v} → {to_v}'
            )

        # Decide output path for this step
        if overwrite and (from_v, to_v) == steps[-1]:
            # Last step: we will overwrite original file
            with tempfile.NamedTemporaryFile(suffix='.cfg', delete=False) as tmp:
                temp_out = Path(tmp.name)
            v2_path = temp_out
        else:
            v2_path = ''

        # Call converter – note: signatures vary slightly
        if cfg_type in ('alignment', 'processing'):
            res = func(current_path, v2_path, sample_config=sample_cfg)
        elif v2_path:
            res = func(current_path, v2_path)
        else:
            res = func(current_path)

        if isinstance(res, tuple):
            # Multi-output (e.g. alignment); we consider the last file the "main" one
            current_path = Path(res[-1])
        else:
            current_path = Path(res)

        # If overwrite on final step, move back onto original path
        if overwrite and (from_v, to_v) == steps[-1] and current_path != cfg_path:
            shutil.move(current_path, cfg_path)
            current_path = cfg_path

    return current_path

class ExperimentConfigConverter:
    def __init__(self, root: Path, app: QApplication):
        self.root = Path(root).expanduser().resolve()
        self.app = app
        self.cfg_loader = ConfigHandler(self.root)
        self._aborted = False

    def run(self):
        self._convert_sample()
        self._rename_channels()
        if self._aborted:
            return
        self._convert_alignment_and_cell_map()
        self._rename_assets()

    # each of these is 20–30 lines max, unit-testable
    def _convert_sample(self): raise NotImplementedError
    def _rename_channels(self): raise NotImplementedError
    def _convert_alignment_and_cell_map(self): raise NotImplementedError
    def _rename_assets(self): raise NotImplementedError



class ExperimentUpgrade_2_1_to_3_0(ExperimentConfigConverter):
    """
    Handles upgrade of a full experiment folder from 2.1 → 3.0.
    """

    def __init__(self, root: Path, app: QApplication):
        super().__init__(root, app)
        self.sample_cfg_path: Path | None = None
        self.alignment_cfg_path: Path | None = None
        self.cell_map_cfg_path: Path | None = None
        self.sample_cfg = None
        self.use_id_as_prefix: bool = False
        self.sample_id: str = ''
        self.new_channels: dict = {}

    def _convert_sample(self):
        # Get paths to the configuration files
        # FIXME: vasculature, batch, group_analysis if/when added
        self.sample_cfg_path, self.alignment_cfg_path, self.cell_map_cfg_path = [
            self.cfg_loader.get_cfg_path(name) for name in ('sample', 'alignment', 'cell_map')
        ]

        # Convert sample config
        v3_sample_cfg_path = convert(self.sample_cfg_path, prev_v='2.1', new_v='3.0',
                                     backup=True, overwrite=True)

        # Load converted sample
        self.sample_cfg = read_cfg(v3_sample_cfg_path)
        self.use_id_as_prefix = self.sample_cfg['use_id_as_prefix']
        self.sample_id = self.sample_cfg['sample_id']

    def _rename_channels(self):
        # Extract channels with paths
        channels = {name: data for name, data in self.sample_cfg['channels'].items() if data['path']}

        # Pop up dialog to rename channels and define data_type
        dialog = RenameChannelsDialog(channels)
        if dialog.exec() == QDialog.Accepted:
            self.new_channels = dialog.get_new_channels()
        else:
            self.new_channels = {}
            self._aborted = True
            return

        if not self.new_channels:
            self._aborted = True
            return

        # Rename channels in the v3 sample config file
        for old_name, (new_name, data_type) in self.new_channels.items():
            self.sample_cfg['channels'][new_name] = self.sample_cfg['channels'].pop(old_name)
            self.sample_cfg['channels'][new_name]['data_type'] = data_type
        self.sample_cfg.write()

    def _convert_alignment_and_cell_map(self):
        # alignment
        if self.alignment_cfg_path and Path(self.alignment_cfg_path).exists():
            convert(self.alignment_cfg_path, prev_v='2.1', new_v='3.0', backup=True, overwrite=False)

        # cell_map
        if self.cell_map_cfg_path and Path(self.cell_map_cfg_path).exists():
            convert(self.cell_map_cfg_path, prev_v='2.1', new_v='3.0', backup=True, overwrite=False)

    @staticmethod
    def fix_stray_underscore(name: str) -> str:
        """remove stray underscore before extension in name"""
        f_path = Path(name)
        if f_path.suffix and f_path.stem.endswith('_'):
            return f_path.stem[:-1] + f_path.suffix
        return name

    def _rename_assets(self):
        files_to_rename, folders_to_rename, main_folder = self.__get_files_and_folders_to_rename()
        files_to_rename = self._compute_new_file_names(files_to_rename)
        files_to_rename = self._compute_elastix_folders_new_names(files_to_rename, folders_to_rename)

        # Verify renaming
        files_to_rename = self.verify_renaming(files_to_rename)
        if self._aborted:
            return
        else:
            # Apply renaming
            for old_name, new_name in files_to_rename:
                (main_folder / old_name).rename(main_folder / new_name)

    def verify_renaming(self, files_to_rename: dict[str, str] = None):
        dialog = VerifyRenamingDialog(files_to_rename)
        if dialog.exec() == QDialog.Accepted:
            files_to_rename = dialog.get_selected_files()
        else:
            self._aborted = True
        return files_to_rename

    def _compute_elastix_folders_new_names(self, files_to_rename: dict[str, str], folders_to_rename: list[str]):
        """Handle the elastix folders renaming using channel info"""
        alignment_reference_channel = self.__get_autofluo_channel()
        regular_channel = self.__get_alignment_regular_channel()

        expr = Expression(str(Path(CHANNELS_ASSETS_TYPES_CONFIG['aligned']['basename']).parent))

        def channels_to_elx_folder(moving_channel: str, fixed_channel: str) -> str:
            """Use clearmap expression to build new folder name based on asset constants"""
            return expr.string(values={'moving_channel': moving_channel, 'fixed_channel': fixed_channel})

        elastix_folders = [d for d in folders_to_rename if d.startswith('elastix_')]
        out: dict[str, str] = deepcopy(files_to_rename)
        for folder in elastix_folders:
            if not alignment_reference_channel:
                warnings.warn(f'No reference channel found in the sample config file, Skipping {folder} renaming.')
                continue
            if folder == 'elastix_auto_to_reference':
                out[folder] = channels_to_elx_folder(moving_channel='atlas', fixed_channel=alignment_reference_channel)
            elif folder == 'elastix_resampled_to_auto':
                out[folder] = channels_to_elx_folder(moving_channel=alignment_reference_channel, fixed_channel=regular_channel)

        return out

    def __get_alignment_regular_channel(self) -> str | None:
        data_channel = [name for name, section in self.sample_cfg['channels'].items()
                        if section['data_type'] != 'autofluorescence']
        data_channel = data_channel[0] if data_channel else None
        return data_channel

    def __get_autofluo_channel(self) -> str | None:
        alignment_reference_channel = [name for name, section in self.sample_cfg['channels'].items()
                                       if section['data_type'] == 'autofluorescence']
        alignment_reference_channel = alignment_reference_channel[0] if alignment_reference_channel else None
        return alignment_reference_channel

    def _compute_new_file_names(self, files_to_rename: dict[str, str]) -> dict[str, str]:
        """Compute new file names based on channel renaming and sample id prefixing."""
        out: dict[str, str] = {}
        for old_name in files_to_rename:
            new_name = old_name  # By default, keep same name
            new_name = self._strip_sample_id(new_name)

            if 'arteries' in old_name:
                new_channel = self.new_channels['arteries'][0]
                new_name = old_name.replace("arteries", "")
            elif 'autofluorescence' in old_name:
                new_channel = next(
                    (name for name, data in self.new_channels.items() if data[1] == 'autofluorescence'),
                    None,
                )
                if new_channel:
                    # FIXME: may be surrounding underscores
                    new_name = old_name.replace("autofluorescence", "")
            else:  # The old (<v3) 'raw' channel (meaning first non autofluo channel)
                new_channel = self.new_channels['raw'][0]

            # Add channel prefix if found
            if new_channel:
                new_name = f'{new_channel}_{new_name}'

            # Add prefix back if needed
            if self.use_id_as_prefix:
                new_name = f'{self.sample_id}_{new_name}'

            out[old_name] = self.fix_stray_underscore(new_name)
        return out

    def _strip_sample_id(self, new_name: str) -> str:
        if self.use_id_as_prefix and new_name.startswith(f'{self.sample_id}_'):
            new_name = new_name[len(f'{self.sample_id}_'):]
        return new_name

    def __get_files_and_folders_to_rename(self) -> tuple[dict[str, str], list[str], Path]:
        main_folder = self.root
        excluded_exts = ('.log', '.html', '.cfg', '.bak')

        assets = [f for f in main_folder.iterdir() if f.suffix not in excluded_exts]

        folders_to_rename = ['elastix_auto_to_reference', 'elastix_resampled_to_auto']
        if self.use_id_as_prefix:
            folders_to_rename = [f'{self.sample_id}_{f}' for f in folders_to_rename]
        files_to_rename = {str(f.name): str(f.name) for f in assets if (main_folder / f).is_file()}
        return files_to_rename, folders_to_rename, main_folder


@project_converter('2.1', '3.0')
def convert_2_1_to_3_0(main_folder='', create_app=True):
    app = ensure_qapp()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    if not main_folder:
        main_folder = get_directory_dlg('~/')
    if not main_folder:
        return

    upgrader = ExperimentUpgrade_2_1_to_3_0(main_folder, app)
    upgrader.run()


def make_generic_3_0_to_3_1_converter(config_type: str):
    """
    Factory for trivial 3.0 → 3.1 converters, for both experiment and global configs.

    - Uses ALTERNATIVES_REG to decide whether to register as experiment or global.
    - Copies all fields as-is.
    - Writes to .yml (if no explicit v2_path is given).
    - Sets clearmap_schema = 3 and clearmap_version = '3.1.0'.
    """

    # Normalize the name and decide scope once at import time
    canonical = ALTERNATIVES_REG.to_canonical(config_type)
    is_global = ALTERNATIVES_REG.is_global_cfg(canonical)

    cfg_conv_decorator = global_cfg_converter if is_global else cfg_converter

    @cfg_conv_decorator('3.0', '3.1', canonical)
    @version_guard('3.0', '3.1')
    def _convert_3_0_to_3_1(v1_path, v2_path=''):
        v1_path = Path(v1_path).expanduser().resolve()

        if not v2_path:
            v2_path = ConfigHandler.resolve_write_path(canonical, base_dir=v1_path.parent)

        v2_path = Path(v2_path).expanduser().resolve()

        cfg_v1 = read_cfg(v1_path)
        cfg_v2 = read_cfg(v2_path, must_exist=False)
        # RESET: erase any existing content in cfg_v2
        for key in list(cfg_v2.keys()):
            del cfg_v2[key]

        # COPY (shallow copy of top-level keys is enough;
        #       nested sections stay as config-like objects.)
        for key in cfg_v1:
            cfg_v2[key] = cfg_v1[key]

        cfg_v2['clearmap_schema'] = 3
        cfg_v2['clearmap_version'] = '3.1.0'

        cfg_v2.write()
        return cfg_v2.filename

    return _convert_3_0_to_3_1


convert_sample_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('sample')
convert_stitching_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('stitching')
convert_registration_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('registration')
convert_cell_map_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('cell_map')
convert_tract_map_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('tract_map')
convert_colocalization_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('colocalization')
convert_vasculature_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('vasculature')
convert_batch_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('batch_processing')
convert_group_analysis_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('group_analysis')

convert_machine_3_0_to_3_1 = make_generic_3_0_to_3_1_converter('machine')


class ExperimentUpgrade_3_0_to_3_1(ExperimentConfigConverter):
    """
    Handles upgrade of a full experiment folder from 3.0 → 3.1.

    For this step, the only changes are:
    - switch configs from .cfg → .yml
    - add clearmap_schema: 3
    - bump clearmap_version to 3.1.0
    """

    def __init__(self, root: Path, app: QApplication):
        super().__init__(root, app)
        self.sample_cfg_path: Path | None = None

    def _convert_sample(self):
        # sample is optional; ConfigHandler will give us the “canonical” sample path
        self.sample_cfg_path = self.cfg_loader.get_cfg_path('sample', must_exist=False)
        if self.sample_cfg_path and Path(self.sample_cfg_path).exists():
            convert_sample_3_0_to_3_1(self.sample_cfg_path)

    def _rename_channels(self): pass

    def _rename_assets(self): pass

    def _convert_alignment_and_cell_map(self):
        """
        Convert all known experiment-level configs present in this folder.
        """
        converters = {
            'stitching': convert_stitching_3_0_to_3_1,
            'registration': convert_registration_3_0_to_3_1,
            'cell_map': convert_cell_map_3_0_to_3_1,
            'tract_map': convert_tract_map_3_0_to_3_1,
            'colocalization': convert_colocalization_3_0_to_3_1,
            'vasculature': convert_vasculature_3_0_to_3_1,
            'batch_processing': convert_batch_3_0_to_3_1,
            'group_analysis': convert_group_analysis_3_0_to_3_1,
        }

        for name, func in converters.items():
            cfg_path = self.cfg_loader.get_cfg_path(name, must_exist=False)
            if cfg_path and Path(cfg_path).exists():
                func(cfg_path)

@project_converter('3.0', '3.1')
def convert_3_0_to_3_1(main_folder: str | Path = '', create_app: bool = True):
    app = ensure_qapp()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    if not main_folder:
        main_folder = get_directory_dlg('~/')
    if not main_folder:
        return

    upgrader = ExperimentUpgrade_3_0_to_3_1(main_folder, app)
    upgrader.run()


def convert_versions(previous_version: str, new_version: str, *,
                     exp_dir: str | Path = '', create_app: bool = True):
    exp_dir = str(Path(exp_dir).expanduser().resolve())
    previous_version = _norm_ver(previous_version)
    new_version = _norm_ver(new_version)
    if previous_version == new_version:
        warnings.warn(f'No conversion needed: already at version {new_version}')
        return

    # Look up in registry
    converter = PROJECT_CONVERTERS.get((previous_version, new_version))
    if not converter:
        raise NotImplementedError(
            f'No converter registered for {previous_version} → {new_version}. '
            f'Supported: {SUPPORTED_VERSIONS}'
        )

    converter(exp_dir, create_app=create_app)
    print(f'\n✓ Upgrade complete: {previous_version} → {new_version}\n')


def test():
    # Example usage
    convert_sample_2_1_to_3_0('/ClearMap/config/default_sample_params.cfg',
                                         '/tmp/test_sample_v3.cfg')

    convert_cell_map_2_1_to_3_0('/ClearMap/config/default_cell_map_params.cfg',
                                           '/tmp/test_cell_map_v3.cfg')


if __name__ == '__main__':
    # test()
    convert_2_1_to_3_0()
