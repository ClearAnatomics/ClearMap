import os.path
import shutil
import tempfile
from pathlib import Path

from packaging.version import Version
from importlib_metadata import version

from ClearMap.Settings import clearmap_path
from ClearMap.config.config_loader import ConfigLoader, get_configobj_cfg

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
    if config_v1['clearmap_version'] != '2.1.0':
        raise ValueError('Only version 2.1.0 is supported')

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


def convert_cell_map_config_2_1_0_to_3_0_0(v1_path, v2_path):
    config_v1, config_v2 = get_configs(v1_path, v2_path)
    if config_v1['clearmap_version'] != '2.1.0':
        raise ValueError('Only version 2.1.0 is supported')

    config_v2['clearmap_version'] = '3.0.0'
    config_v2['channel_0'] = {
        'detection': config_v1['detection'],
        'cell_filtration': config_v1['cell_filtration'],
        'voxelization': config_v1['voxelization'],
        'run': config_v1['run'],
    }

    config_v2.write()
    return config_v2.filename


def convert_alignment_config_2_1_0_to_3_0_0(v1_path, v2_path, channel_names=None):  # TODO: get actual channel name form sample config
    config_v1 = get_configobj_cfg(v1_path)
    if config_v1['clearmap_version'] != '2.1.0':
        raise ValueError('Only version 2.1.0 is supported')

    if not v2_path:
        v2_path = v1_path

    out_stitching_cfg = get_configobj_cfg(v2_path.with_name(f'stitching_config_{VERSION_SUFFIX}.cfg'), must_exist=False)
    out_registration_cfg = get_configobj_cfg(v2_path.with_name(f'registration_config_{VERSION_SUFFIX}.cfg'), must_exist=False)

    # Copy general parameters
    out_stitching_cfg['clearmap_version'] = '3.0.0'
    out_registration_cfg['clearmap_version'] = '3.0.0'
    # drop pipeline_name and conversion section

    # Copy stitching parameters
    channel_names = channel_names or [k for k, use in config_v1['stitching']['run'].items() if use]
    for i, channel in enumerate(channel_names):
        out_stitching_cfg[channel] = {}
        out_stitching_cfg[channel]['use_npy'] = config_v1['conversion']['use_npy']
        out_stitching_cfg[channel]['run'] = config_v1['stitching']['run'][channel]
        out_stitching_cfg[channel]['layout_channel'] = channel
        if i == 0:  # By default, stitch other channels to the first one
            # This (dict(...)) is weird but required to get the extra indentation level
            out_stitching_cfg[channel]['rigid'] = dict(config_v1['stitching']['rigid'])
            out_stitching_cfg[channel]['rigid']['projection_thickness'] = out_stitching_cfg[channel]['rigid'].pop('project_thickness')
            out_stitching_cfg[channel]['wobbly'] = dict(config_v1['stitching']['wobbly'])

    # Copy registration parameters
    out_registration_cfg['verbose'] = config_v1['registration']['resampling']['verbose']
    out_registration_cfg['atlas'] = {k: config_v1['registration']['atlas'][k]
                                     for k in ('id', 'structure_tree_id', 'align_files_folder')}
    params_files = [v for k, v in config_v1['registration']['atlas'].items() if k.startswith('align_reference')]
    out_registration_cfg['channels'] = {
        'autofluorescence': {
            'resample': not(config_v1['registration']['resampling']['skip']),
            'resampled_resolution': config_v1['registration']['resampling']['autofluo_sink_resolution'],
            'align_with': 'atlas',
            'moving_channel': 'atlas',
            'params_files': params_files,
            'landmarks_weights': [0] * len(params_files),
        }
    }
    for channel in channel_names:
        if channel not in out_registration_cfg['channels']:
            out_registration_cfg['channels'][channel] = {
                'resample': not(config_v1['registration']['resampling']['skip']),
                'resampled_resolution': config_v1['registration']['resampling']['raw_sink_resolution'],
                'align_with': 'autofluorescence',
                'moving_channel': channel,
                'params_files': [config_v1['registration']['atlas']['align_channels_affine_file']],
                'landmarks_weights': [0],
            }

    out_stitching_cfg.write()
    out_registration_cfg.write()
    return out_stitching_cfg.filename, out_registration_cfg.filename


def convert_machine_config_2_1_0_to_3_0_0(v1_path, v2_path=''):
    return v1_path


def convert(cfg_path, backup=False, overwrite=False):
    conversion_funcs = {
        'sample': convert_sample_config_2_1_0_to_3_0_0,
        'alignment': convert_alignment_config_2_1_0_to_3_0_0,
        'processing': convert_alignment_config_2_1_0_to_3_0_0,
        'cell_map': convert_cell_map_config_2_1_0_to_3_0_0,
        'machine': convert_machine_config_2_1_0_to_3_0_0,
    }
    config_type = next((k for k in conversion_funcs if k in str(cfg_path.stem)), None)
    if config_type is None:
        raise ValueError(f'Unknown config type for {cfg_path.stem} (whole path: {cfg_path})')

    if backup:
        if cfg_path.with_suffix('.bak').exists():
            raise FileExistsError(f'Backup file already exists: {cfg_path.with_suffix(".bak")}')
        shutil.copyfile(cfg_path, cfg_path.with_suffix('.bak'))
    if overwrite:
        with tempfile.NamedTemporaryFile(suffix='.cfg', delete=False) as tmp:
            dest = Path(tmp.name)
        dest_path = Path(conversion_funcs[config_type](cfg_path, dest))
        shutil.move(dest_path, cfg_path)
        dest_path = cfg_path
    else:
        dest_path = Path(conversion_funcs[config_type](cfg_path))
    return dest_path


def main():
    # Example usage
    convert_sample_config_2_1_0_to_3_0_0('/ClearMap/config/default_sample_params.cfg',
                                         '/tmp/test_sample_v3.cfg')

    convert_cell_map_config_2_1_0_to_3_0_0('/ClearMap/config/default_cell_map_params.cfg',
                                           '/tmp/test_cell_map_v3.cfg')


if __name__ == '__main__':
    main()
