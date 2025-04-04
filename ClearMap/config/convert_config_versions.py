import os.path
import shutil
import tempfile
import warnings
from pathlib import Path

from packaging.version import Version
from importlib_metadata import version

import qdarkstyle
from PyQt5.QtWidgets import QApplication, QDialog

from ClearMap.IO.assets_constants import CHANNELS_ASSETS_TYPES_CONFIG
from ClearMap.Utils.tag_expression import Expression
from ClearMap.config.config_loader import ConfigLoader, get_configobj_cfg

from ClearMap.gui.dialogs import RenameChannelsDialog, VerifyRenamingDialog, get_directory_dlg

clearmap_version = version('ClearMap')
VERSION_SUFFIX = f'v{Version(clearmap_version).major}_{Version(clearmap_version).minor}'


def get_configs(v1_path, v2_path=''):
    v1_path = Path(v1_path).expanduser()
    if not v2_path:
        v2_path = v1_path.with_name(f'{v1_path.stem}_{VERSION_SUFFIX}.cfg')  # can't use with_suffix without a dot
    v2_path = v2_path.expanduser()
    return get_configobj_cfg(v1_path), get_configobj_cfg(v2_path, must_exist=False)


def convert_sample_config_2_1_0_to_3_0_0(v1_path, v2_path=''):
    config_v1, config_v2 = get_configs(v1_path, v2_path)
    if config_v1['clearmap_version'] == '3.0.0':
        warnings.warn('Sample config already in version 3.0.0')
        return config_v1.filename
    if config_v1['clearmap_version'] != '2.1.0':
        raise ValueError(f'Error converting {v1_path} '
                         f'with version {config_v1["clearmap_version"]}'
                         f'Only version 2.1.0 is supported')

    def v1_or_default(key):
        return config_v1.get(key, default_cfg[key])

    try:
        default_cfg = ConfigLoader.get_cfg_from_path(ConfigLoader.get_default_path('sample', from_package=False))
    except FileExistsError:
        default_cfg = ConfigLoader.get_cfg_from_path(ConfigLoader.get_default_path('sample', from_package=True))

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


def convert_cell_map_config_2_1_0_to_3_0_0(v1_path, v2_path, channel_name='channel_0'):
    config_v1, config_v2 = get_configs(v1_path, v2_path)
    if config_v1['clearmap_version'] == '3.0.0':
        warnings.warn('Sample config already in version 3.0.0')
        return config_v1.filename
    if config_v1['clearmap_version'] != '2.1.0':
        raise ValueError('Only version 2.1.0 is supported')

    config_v2['clearmap_version'] = '3.0.0'
    config_v2[channel_name] = {
        'detection': config_v1['detection'],
        'cell_filtration': config_v1['cell_filtration'],
        'voxelization': config_v1['voxelization'],
        'run': config_v1['run'],
    }

    config_v2.write()
    return config_v2.filename


def convert_alignment_config_2_1_0_to_3_0_0(v1_path, v2_path='', sample_config=None):
    config_v1 = get_configobj_cfg(v1_path)
    if config_v1['clearmap_version'] != '2.1.0':
        raise ValueError('Only version 2.1.0 is supported')

    v2_path = v2_path or v1_path

    out_stitching_cfg = alignment_to_stitching_v3(v2_path, config_v1)

    out_registration_cfg = alignment_to_registration_v3(v2_path, config_v1, sample_config)

    return out_stitching_cfg.filename, out_registration_cfg.filename


def alignment_to_stitching_v3(output_path_base, config_v1):
    out_stitching_cfg = get_configobj_cfg(output_path_base.with_name(f'stitching_params.cfg'),
                                          must_exist=False)
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


def alignment_to_registration_v3(output_path_base, config_v1, sample_config):
    out_registration_cfg = get_configobj_cfg(output_path_base.with_name(f'registration_params.cfg'),
                                             must_exist=False)
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


def convert_machine_config_2_1_0_to_3_0_0(v1_path, v2_path=''):
    return v1_path


def convert(cfg_path, backup=False, overwrite=False):
    conversion_funcs = {
        'sample': convert_sample_config_2_1_0_to_3_0_0,
        'alignment': convert_alignment_config_2_1_0_to_3_0_0,
        'processing': convert_alignment_config_2_1_0_to_3_0_0,
        'cell_map': convert_cell_map_config_2_1_0_to_3_0_0,
        'machine': convert_machine_config_2_1_0_to_3_0_0,
        # FIXME: implement vasculature, batch, group_analysis, display
    }
    config_type = next((k for k in conversion_funcs if k in str(cfg_path.stem)), None)
    if config_type is None:
        raise ValueError(f'Unknown config type for {cfg_path.stem} (whole path: {cfg_path})')

    if backup:
        if cfg_path.with_suffix('.bak').exists():
            raise FileExistsError(f'Backup file already exists: {cfg_path.with_suffix(".bak")}')
        shutil.copyfile(cfg_path, cfg_path.with_suffix('.bak'))

    if config_type != 'sample':  # We already have a sample config for v3
        config_loader = ConfigLoader(Path(cfg_path).parent)
        sample_cfg_path = config_loader.get_cfg_path('sample', must_exist=False)
        if os.path.exists(sample_cfg_path):  # FIXME: Make sure v3 sample config exists
            sample_cfg = config_loader.get_cfg_from_path(sample_cfg_path)
    if overwrite:
        with tempfile.NamedTemporaryFile(suffix='.cfg', delete=False) as tmp:
            dest = Path(tmp.name)
        args = [cfg_path, dest]
        if config_type in ('alignment', 'processing'):
            args.append(sample_cfg)
        dest_path = Path(conversion_funcs[config_type](*args))
        shutil.move(dest_path, cfg_path)
        dest_path = cfg_path
    else:
        if config_type in ('alignment', 'processing'):
            res = conversion_funcs[config_type](cfg_path, sample_config=sample_cfg)
        else:
            res = conversion_funcs[config_type](cfg_path)
        if isinstance(res, tuple):
            dest_path = Path(res[-1])
        else:
            dest_path = Path(res)
    return dest_path


def convert_v2_1_to_v3_0(main_folder='', create_app=True):
    if not create_app:
        app = QApplication.instance()
    else:
        app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    if not main_folder:
        main_folder = get_directory_dlg("~/")

    if not main_folder:
        return

    # Initialize ConfigLoader
    cfg_loader = ConfigLoader(main_folder)

    # Get paths to the configuration files
    # FIXME: vasculature, batch, group_analysis
    sample_cfg_path, alignment_cfg_path, cell_map_cfg_path = (
        [cfg_loader.get_cfg_path(name) for name in ('sample', 'alignment', 'cell_map')])

    # Convert sample config
    v3_sample_cfg_path = convert(sample_cfg_path, backup=True, overwrite=True)

    # Create SampleManager
    sample_cfg = ConfigLoader.get_cfg_from_path(v3_sample_cfg_path)
    use_id_as_prefix = sample_cfg['use_id_as_prefix']
    sample_id = sample_cfg['sample_id']

    # Extract channels with paths
    channels = {name: data for name, data in sample_cfg['channels'].items() if data['path']}

    # Pop up dialog to rename channels and define data_type
    dialog = RenameChannelsDialog(channels)
    if dialog.exec() == QDialog.Accepted:
        new_channels = dialog.get_new_channels()
    else:
        new_channels = {}

    if new_channels:
        print(f'{new_channels=}')
    else:
        return

    # Rename channels in the v3 sample config file
    for old_name, (new_name, data_type) in new_channels.items():
        sample_cfg['channels'][new_name] = sample_cfg['channels'].pop(old_name)
        sample_cfg['channels'][new_name]['data_type'] = data_type
    sample_cfg.write()

    # Convert alignment and cell_map config files
    convert(alignment_cfg_path, backup=True, overwrite=False)
    # convert(cell_map_cfg_path, backup=True, overwrite=False)  # FIXME: add if exists

    # Rename assets
    assets = [f for f in os.listdir(main_folder) if not f.endswith(('.log', '.html', '.cfg', '.bak'))]
    folders_to_rename = ['elastix_auto_to_reference', 'elastix_resampled_to_auto']
    if use_id_as_prefix:
        folders_to_rename = [f'{sample_id}_{f}' for f in folders_to_rename]
    files_to_rename = {f: f for f in assets if os.path.isfile(os.path.join(main_folder, f))}

    # Compute new names
    for file in files_to_rename:
        new_name = file
        if use_id_as_prefix and new_name.startswith(f'{sample_id}_'):  # Strip sample id if exists
            new_name = new_name[len(f'{sample_id}_'):]
        if 'arteries' in file:
            new_channel = new_channels["arteries"][0]
            new_name = f'{new_channel}_{file.replace("arteries", "")}'
        elif 'autofluorescence' in file:
            autofluo_channel = next((name for name, data in new_channels.items() if data[1] == 'autofluorescence'), None)
            if autofluo_channel:
                new_name = f'{autofluo_channel}_{file.replace("autofluorescence", "")}'  # FIXME: may be surrounding underscores
        else:
            new_channel = new_channels["raw"][0]
            new_name = f'{new_channel}_{new_name}'
        if use_id_as_prefix:
            new_name = f'{sample_id}_{new_name}'
        files_to_rename[file] = new_name

    alignment_reference_channel = [name for name, section in sample_cfg['channels'].items()
                                   if section['data_type'] == 'autofluorescence']
    if alignment_reference_channel:
        alignment_reference_channel = alignment_reference_channel[0]
    data_channel = [name for name, section in sample_cfg['channels'].items()
                                   if section['data_type'] != 'autofluorescence']
    if data_channel:
        data_channel = data_channel[0]
    for folder in folders_to_rename:
        if folder == 'elastix_auto_to_reference':
            if not alignment_reference_channel:
                warnings.warn('No reference channel found in the sample config file')
                continue
            new_name = Expression(str(Path(CHANNELS_ASSETS_TYPES_CONFIG['aligned']['basename']).parent))
            new_name = new_name.string(values={'moving_channel': 'atlas',
                                               'fixed_channel': alignment_reference_channel})
            files_to_rename[folder] = new_name
        elif folder == 'elastix_resampled_to_auto':
            if not alignment_reference_channel:
                warnings.warn('No reference channel found in the sample config file')
                continue
            new_name = Expression(str(Path(CHANNELS_ASSETS_TYPES_CONFIG['aligned']['basename']).parent))
            new_name = new_name.string(values={'moving_channel': alignment_reference_channel,
                                               'fixed_channel': data_channel})
            files_to_rename[folder] = new_name


    # Verify renaming
    dialog = VerifyRenamingDialog(files_to_rename)
    if dialog.exec() == QDialog.Accepted:
        files_to_rename = dialog.get_selected_files()
    else:
        return

    main_folder = Path(main_folder)
    for old_name, new_name in files_to_rename:
        os.rename(main_folder / old_name, main_folder / new_name)


def convert_versions(previous_version: str, new_verison: str, main_folder: str = '', create_app: bool = True):
    if previous_version == '2.1.0' and new_verison == '3.0.0':
        try:
            convert_v2_1_to_v3_0(main_folder, create_app=create_app)
        except Exception as err:
            warnings.warn(f'Error converting {main_folder}: {err}')
            return
    else:
        raise NotImplementedError(f'Conversion from {previous_version} to {new_verison} is not supported yet.')


def test():
    # Example usage
    convert_sample_config_2_1_0_to_3_0_0('/ClearMap/config/default_sample_params.cfg',
                                         '/tmp/test_sample_v3.cfg')

    convert_cell_map_config_2_1_0_to_3_0_0('/ClearMap/config/default_cell_map_params.cfg',
                                           '/tmp/test_cell_map_v3.cfg')


if __name__ == '__main__':
    # test()
    convert_v2_1_to_v3_0()
