"""
sample_preparation
==================

This is the part that is common to both pipelines to process the raw images.
It includes file conversion, stitching and registration
"""
import os
import platform
import re
import warnings
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

# noinspection PyPep8Naming
import matplotlib
from configobj import ConfigObj

from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE
from ClearMap.Utils.exceptions import MissingRequirementException, ClearMapAssetError
from ClearMap.gui.gui_utils import surface_project, setup_mini_brain

matplotlib.use('Qt5Agg')

import ClearMap.Settings as settings
from ClearMap.Utils.utilities import check_stopped, runs_on_ui
from ClearMap.config.atlas import ATLAS_NAMES_MAP
from ClearMap.processors.generic_tab_processor import TabProcessor, CanceledProcessing
# noinspection PyPep8Naming
import ClearMap.Alignment.Elastix as elastix
# noinspection PyPep8Naming
from ClearMap.Alignment.Annotation import Annotation
# noinspection PyPep8Naming
import ClearMap.IO.IO as clearmap_io
# noinspection PyPep8Naming
import ClearMap.Visualization.Qt.Plot3d as plot_3d
import ClearMap.Visualization.Plot3d as q_plot_3d
# noinspection PyPep8Naming
import ClearMap.Alignment.Resampling as resampling
# noinspection PyPep8Naming
import ClearMap.Alignment.Stitching.StitchingRigid as stitching_rigid
# noinspection PyPep8Naming
import ClearMap.Alignment.Stitching.StitchingWobbly as stitching_wobbly
from ClearMap.IO.metadata import define_auto_stitching_params, define_auto_resolution
from ClearMap.config.config_loader import get_configs, ConfigLoader, CLEARMAP_CFG_DIR
from ClearMap.config.update_config import update_default_config
from ClearMap.IO.assets_specs import TypeSpec, ChannelSpec
from ClearMap.IO.workspace2 import Workspace2
from ClearMap.Visualization.Color.Color import gray_image_to_rgb


__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


DEFAULT_ORIENTATION = (1, 2, 3)   # FIXME: defined in several places

class SampleManager(TabProcessor):
    """
    This class is used to manage the sample information
    Manage sample-level configurations and properties.
    Handle configurations related to the sample.
    Provide utility methods for checking sample properties.

    """
    def __init__(self):
        super().__init__()
        self.config_loader = None
        self.src_directory = None
        self.config = {}
        self.stitching_cfg = {}
        self.registration_cfg = {}

        self.incomplete_channels = []
        self.setup_complete = False

    def setup(self, cfgs=None, src_dir=None, watcher=None):
        """
        Setup the sample manager with the given configs.
        If watcher is provided, it will be used to display progress.

        Parameters
        ----------
        cfgs : tuple(str) | tuple(Path) | tuple(ConfigObj)
            (machine_cfg_path, sample_cfg_path, stitching_cfg_path, registration_cfg_path) or
            (machine_cfg, sample_cfg, stitching_cfg, registration_cfg)
        src_dir : str | Path | None
            The source directory of the sample
        watcher: ProgressWatcher | None
            The progress watcher to use
        """
        # Get source directory and setup the loader
        if src_dir and not cfgs:
            self.src_directory = Path(src_dir)
            cfgs = self.load_configs_from_dir()
        else:
            sample_cfg = cfgs[1]  # Path, name or config
            if isinstance(sample_cfg, (Path, str)) and '/' in str(sample_cfg):
                self.src_directory = Path(sample_cfg).parent
            elif isinstance(sample_cfg, ConfigObj):
                self.src_directory = Path(sample_cfg.filename).parent
            else:
                if self.src_directory is None:
                    raise ValueError('Source directory not set and required to load configs with names only')
        self.config_loader = ConfigLoader(self.src_directory)

        # Load the configs
        cfgs = list(cfgs)
        for i, cfg in enumerate(cfgs):
            if isinstance(cfg, (Path, str)):  # path not config itself
                cfgs[i] = self.config_loader.get_cfg(cfg)
        self.machine_config, self.config, self.stitching_cfg, self.registration_cfg = cfgs

        if watcher is not None:
            self.progress_watcher = watcher  # FIXME: in stitcher and registration too

        self.update_workspace()
        self.workspace.info()
        self.setup_complete = not self.incomplete_channels

    def load_processors_config(self, stitching_config=None, registration_config=None):
        if self.stitching_cfg is None:
            self.stitching_cfg = stitching_config

        if self.registration_cfg is None:
            self.registration_cfg = registration_config

        self.update_workspace()
        self.workspace.info()
        self.setup_complete = not self.incomplete_channels

    def rename_channels(self, names_map):
        for old_name, new_name in names_map:
            if new_name != old_name:
                if old_name not in self.config['channels'] and new_name in self.config['channels']:
                    pass
                else:
                    self.config['channels'][new_name] = self.config['channels'].pop(old_name)
                if old_name in self.workspace.asset_collections:
                    self.workspace.asset_collections[new_name] = self.workspace.asset_collections.pop(old_name)
                    self.workspace.asset_collections[new_name].channel_spec.name = new_name
                    # FIXME: update ChannelSpec.channel_names
                # The WS is updated through update_workspace

    def update_workspace(self):
        channel_names = self.config['channels'].keys()
        if not channel_names:
            first_channel = None
        else:
            first_channel = list(channel_names)[0]
        if self.workspace is None:
            self.workspace = Workspace2(self.src_directory, sample_id=self.prefix,
                                        default_channel=first_channel)
        self.incomplete_channels = []
        for channel, cfg in self.config['channels'].items():
            raw_path = cfg['path']
            if raw_path:
                if channel not in self.workspace.asset_collections:
                    content_type = cfg['data_type']
                    if not content_type:
                        raise ValueError(f'No data type specified for channel {channel}. '
                                         f'Cannot create asset without data type')
                        # FIXME: see if we could leave None and have minimum required assets
                    elif content_type == 'undefined':  # Difference with None is intention
                        continue
                    self.workspace.add_raw_data(file_path=raw_path,
                                                channel_id=channel,
                                                data_content_type=content_type,
                                                sample_id=self.prefix)
                else:
                    old_asset = self.workspace.asset_collections[channel]['raw']
                    if old_asset.expression != raw_path:
                        self.workspace.asset_collections[channel]['raw'] = old_asset.variant(expression=raw_path)
            else:
                self.incomplete_channels.append(channel)
        self.workspace.info()

    @property
    def prefix(self):
        """
        Get the prefix to use for the files

        Returns
        -------
        str
            The prefix to use, None to not use any
        """
        return self.config['sample_id'] if self.config['use_id_as_prefix'] else None

    @property
    def channels(self):
        return list(self.config['channels'].keys())

    @property
    def data_types(self):
        return [channel_cfg['data_type'] for channel_cfg in self.config['channels'].values()]

    @property
    def channels_to_detect(self):
        return [c for c, v in self.config['channels'].items() if CONTENT_TYPE_TO_PIPELINE[v['data_type']] == 'CellMap']

    @property
    def is_colocalization_compatible(self):
        return len(self.channels_to_detect) > 1

    @property
    def relevant_pipelines(self):
        """
        All the pipelines relevant to any of the sample channels

        Returns
        -------
        List[str]
            The relevant pipeline names
        """
        pipelines = [CONTENT_TYPE_TO_PIPELINE.get(d_type) for d_type in self.data_types]
        pipelines = list(set([p for p in pipelines if p is not None]))
        return pipelines

    def z_only(self, channel):
        """
        Check if the channel is z only (no x or y tiles)

        Parameters
        ----------
        channel : str
            The channel to check

        Returns
        -------
        bool
            True if the channel is z only
        """
        return self.get('raw', channel, prefix=self.prefix).tag_names == ['Z']

    def is_tiled(self, channel):
        asset = self.get('raw', channel, prefix=self.prefix)
        return asset.is_tiled and not self.z_only(channel)

    @property
    def autofluorescence_is_tiled(self):
        """
        Check if the autofluorescence channel is tiled (has x and y tiles)
        Returns
        -------
        bool
            True if the autofluorescence channel is tiled
        """
        return self.is_tiled(self.alignment_reference_channel)

    def has_tiles(self, channel=None):
        # extension = '.npy' if self.use_npy() else None
        # return len(clearmap_io.file_list(self.filename(channel, prefix=self.prefix, extension=extension)))
        # noinspection PyTypeChecker
        if channel is None:
            return bool(self.stitchable_channels)
        return self.get('raw', channel=channel, sample_id=self.prefix).n_tiles > 1

    def check_has_all_tiles(self, channel):
        """
        Check whether all the tiles of the channel exist on disk

        Parameters
        ----------
        channel : str
            The channel to check

        Returns
        -------
        bool
            True if all the tiles exist
        """
        extension = '.npy' if self.use_npy(channel) else None
        return self.get('raw', channel, extension=extension).exists

    @property
    def stitchable_channels(self):
        return self.get_stitchable_channels()

    def get_stitchable_channels(self):
        candidates = self.config['channels'].keys()
        try:
            assets = [self.get('raw', channel=c, sample_id=self.prefix) for c in candidates]
        except KeyError:  #  raw data not yet set up
            warnings.warn(f'Trying to get stitchable channels before raw data is set up')
            return []
        stitchable_channels = [c for c, asset in zip(candidates, assets) if asset.is_tiled]
        return stitchable_channels

    def can_convert(self, channel):
        asset = self.get('raw', channel=channel, sample_id=self.prefix)
        return asset.is_regular_file and not asset.variant(extension='.npy').exists

    @property
    def channels_to_convert(self):
        candidates = self.config['channels'].keys()
        return [c for c in candidates if self.can_convert(c)]

    def has_npy(self, channel=None):
        """
        Check if the channel is in npy format

        Parameters
        ----------
        channel : str
            The channel to check

        Returns
        -------
        bool
            True if the raw channel is in npy format
        """
        channels = [channel] if channel is not None else self.stitchable_channels
        return any([self.get('raw', channel=channel, sample_id=self.prefix).variant(extension='.npy').exists
                    for channel in channels])

    def use_npy(self, channel):
        asset = self.get('raw', channel=channel, prefix=self.prefix)
        cfg = self.stitching_cfg['channels'][channel]
        return cfg['use_npy'] and str(asset.expression).endswith('.npy') or asset.variant(extension='.npy').exists

    # FIXME: get_configs is currently incomplete
    def set_configs(self, cfg_paths):
        cfg_paths = [os.path.expanduser(p) for p in cfg_paths]
        self.machine_config, self.config, self.stitching_cfg, self.registration_cfg = get_configs(*cfg_paths)

    def get_configs(self):
        cfg = {
            'machine': self.machine_config,
            'sample': self.config,
            'stitching': self.stitching_cfg,
            'registration': self.registration_cfg
        }
        return cfg

    @property
    def alignment_reference_channel(self):
        for channel, cfg in self.config['channels'].items():
            if cfg['data_type'] == 'autofluorescence':
                return channel
        return

    def delete_resampled_files(self, channel):
        asset = self.get('resampled', channel=channel)
        if asset.exists:
            asset.delete()

    def stitched_shape(self, channel):
        asset = self.get('stitched', channel=channel, sample_id=self.prefix)
        if asset.exists:
            return asset.shape()
        elif self.resampled_shape(channel) is not None:
            raw_resampled_res_from_cfg = np.array(self.registration_cfg['channels'][channel]['resampled_resolution'])
            raw_res_from_cfg = np.array(self.config['channels'][channel]['resolution'])
            return self.resampled_shape(channel) * (raw_resampled_res_from_cfg / raw_res_from_cfg)
        else:
            raise FileNotFoundError(f'Could not get stitched shape without '
                                    f'stitched or resampled file for channel {channel}')

    def resampled_shape(self, channel):
        asset = self.workspace.get('resampled', channel=channel, sample_id=self.prefix)
        if asset.exists:
            return asset.shape()

    def needs_registering(self):  #  TODO: see if could be moved to RegistrationProcessor
        reg_cfg = self.registration_cfg
        alignment_partners = [cfg['align_with'] for cfg in reg_cfg['channels'].values()]

        def check_registered(channel):
            align_with = reg_cfg['channels'][channel]['align_with']
            moving_channel = reg_cfg['channels'][channel]['moving_channel']
            asset = self.get('aligned', channel=channel)
            fixed_channel = channel if align_with == moving_channel else align_with
            return asset.specify({'moving_channel': moving_channel, 'fixed_channel': fixed_channel}).exists

        if all([p is None for p in alignment_partners]):  # Alignment deselected for all channels
            return False
        else:
            ref_channel = self.alignment_reference_channel
            for channel in self.channels:
                if not ref_channel and reg_cfg['channels'][channel]['align_with'] == 'autofluorescence':
                    warnings.warn(f'This should not happen, {channel=} set for registration against '
                                  f'autofluorescence but no reference channel found')
                    return False
                else:
                    if not check_registered(channel):
                        return False

    def load_configs_from_dir(self):
        cfg_loader = ConfigLoader(self.src_directory)
        return [cfg_loader.get_cfg(name, must_exist) for name, must_exist in
                (('machine', True), ('sample', True), ('stitching', False), ('registration', False))]

    def asset_names_to_assets(self, asset_names, channel=None, sample_id=None):
        return [self.workspace.get(asset_name) for asset_name in asset_names]

    @staticmethod
    def compress(assets, format=None):
        for asset in assets:
            asset.compress(algorithm=format)

    @staticmethod
    def decompress(assets, check=True):
        for asset in assets:
            asset.decompress(check=check)

    @staticmethod
    def plot(assets, **kwargs):  # FIXME: what if len(assets) > 1 ? Should plot together
        for asset in assets:
            asset.plot(**kwargs)

    @staticmethod
    def convert(assets, new_extension, processes=None, verbose=False, **kwargs):
        for asset in assets:
            asset.convert(new_extension, processes=processes, verbose=verbose, **kwargs)

    @staticmethod
    def resample(assets, x_scale=1, y_scale=1, z_scale=1,
                 x_resolution=None, y_resolution=None, z_resolution=None,
                 x_shape=None, y_shape=None, z_shape=None,
                 orientation=None,  # TODO: add orientation
                 processes=None, verbose=False, **kwargs):
        resolution_params = {
            'x_scale': x_scale,
            'y_scale': y_scale,
            'z_scale': z_scale,
            'x_shape': x_shape,
            'y_shape': y_shape,
            'z_shape': z_shape,
            'x_resolution': x_resolution,
            'y_resolution': y_resolution,
            'z_resolution': z_resolution
        }
        resolution_params = {k: v for k, v in resolution_params.items() if v not in (1, None)}
        for asset in assets:
            if 'x_shape' in resolution_params.keys():
                resampling_params = {
                    'original_shape': asset.shape(),
                    'resampled_shape': tuple([resolution_params[f'{ax}_shape'] for ax in 'xyz'])}
            elif 'x_resolution' in resolution_params.keys():
                resampling_params = {
                    'resampled_resolution': tuple([resolution_params[f'{ax}_resolution'] for ax in 'xyz'])}
                # FIXME: needs original resolution
                #   resampling_params = {'original_resolution': ...}
            elif 'x_scale' in resolution_params.keys():
                original_shape = {ax: s for ax, s in zip('xyz', asset.shape())}
                resampling_params = {f'{ax}_shape': original_shape[ax] // resolution_params[f'{ax}_scale']
                                     for ax in 'xyz'}

            resampled_path = asset.path.with_suffix(f'.resampled.{asset.path.suffix}')
            resampling.resample(original=str(asset.path), resampled=resampled_path,
                                **resampling_params, orientation=orientation,
                                processes=processes, verbose=verbose, **kwargs)


class RegistrationProcessor(TabProcessor):
    """
    This class is used to manage the registration process
    Perform image registration operations.
    Manage atlas setup and transformations.
    Handle registration configurations.
    """
    def __init__(self, sample_manager):
        super().__init__()
        self.sample_manager = sample_manager
        self.config = {}
        self.annotators = {}
        self.mini_brains = {}  # 1 for each channel
        self.progress_watcher = None  # FIXME:
        self.__bspline_registration_re = re.compile(r"\d+\s-?\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+")
        self.__affine_registration_re = re.compile(r"\d+\s-\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+")
        self.__resample_re = ('Resampling: resampling',
                              re.compile(r".*?Resampling:\sresampling\saxes\s.+\s?,\sslice\s.+\s/\s\d+"))

        self.setup_complete = False

    def setup(self, sample_manager=None):
        self.sample_manager = sample_manager if sample_manager else self.sample_manager
        if not self.sample_manager.registration_cfg:
            raise ValueError('Registration config not set in sample manager')
        self.config = self.sample_manager.registration_cfg
        self.machine_config = self.sample_manager.machine_config
        if self.sample_manager.setup_complete:
            self.workspace = self.sample_manager.workspace
            self.setup_atlases()  # TODO: check if needed
            self.add_pipeline()
            self.setup_complete = True
        else:
            self.setup_complete = False  # FIXME: finish later

        # WARNING: must be called once registration pipeline has been added to the Workspace for that channel
        # self.parametrize_assets()

    def parametrize_assets(self):
        for channel in self.config['channels']:
            if self.config['channels'][channel]['align_with'] is None:
                continue
            if self.config['channels'][channel]['moving_channel'] in (None, 'intrinsically aligned'):
                continue
            for asset_type in ('fixed_landmarks', 'moving_landmarks', 'aligned'):
                try:
                    asset = self.get_elx_asset(asset_type, channel=channel)
                except KeyError:
                    continue  # FIXME: this should be handled more elegantly
                              #  the idea is to delay the parametrization
                              #  until the assets for all channels have been created
                except ClearMapAssetError:  # Check that align_with is None
                    warnings.warn(f'Could not parametrized {asset_type} for {channel=}')
                    continue
                if asset.is_expression:
                    fixed_channel, moving_channel = self.get_fixed_moving_channels(channel)
                    parametrized_asset = asset.specify({'moving_channel': moving_channel, 'fixed_channel': fixed_channel})
                    self.workspace.asset_collections[channel][asset_type] = parametrized_asset

    def add_pipeline(self):  # WARNING: hacky. Maybe add_pipeline_if_missing
        if self.workspace is None:
            return
        for channel in self.config['channels']:
            try:
                self.get('aligned', channel=channel)
            except KeyError:
                if self.sample_manager.setup_complete and channel in self.workspace.asset_collections:
                    self.workspace.add_pipeline('registration', channel_id=channel)
                    self.parametrize_assets()
                else:
                    warnings.warn('Workspace not setup, cannot add registration pipeline')

    def channels_to_resample(self):
        return [c for c, v in self.config['channels'].items() if v['resample']]

    def channels_to_register(self):
        return [c for c, v in self.config['channels'].items() if v['align_with'] is not None]

    @property
    def was_registered(self):  #  FIXME: much better if part of sample_manager
        return self.get_elx_asset('aligned',
                                  channel=self.sample_manager.alignment_reference_channel).exists

    @property
    def registration_params_files(self):
        align_dir = Path(settings.resources_path) / self.config['atlas']['align_files_folder']
        registration_params_files = {}
        for channel in self.config['channels']:
            params_file_names = self.config['channels'][channel]['params_files']
            registration_params_files[channel] = [align_dir / name for name in params_file_names]  # TODO: property
        return registration_params_files

    def plot_atlas(self, channel):  # FIXME: idealy part of sample_manager
        return q_plot_3d.plot(self.get('atlas', channel=channel, asset_sub_type='reference').path,
                              lut=self.machine_config['default_lut'])

    def clear_landmarks(self, channel=None):
        """
        Clear (remove) the landmarks files
        """
        channels = [channel] if channel else self.config['channels'].keys()
        for channel in channels:
            for landmark_type in ('fixed', 'moving'):
                asset = self.get_elx_asset(f'{landmark_type}_landmarks', channel=channel)
                if asset.exists:
                    asset.delete()

    def get_fixed_moving_channels(self, channel):
        moving_channel = self.config['channels'][channel]['moving_channel']
        align_with = self.config['channels'][channel]['align_with']
        if align_with is None:
            return None, moving_channel
        if not align_with:
            raise ValueError(f'Channel {channel} missing align_with in registration config')
        # fixed is whichever channel from ('channel', 'align_with') is not 'moving_channel'
        fixed_channel = channel if align_with == moving_channel else align_with
        return fixed_channel, moving_channel

    def get_elx_asset(self, asset_type, channel):
        fixed_channel, moving_channel = self.get_fixed_moving_channels(channel)
        if fixed_channel is None or moving_channel is None:
            return None

        asset = self.get(asset_type, channel=channel)
        if not asset.is_expression:
            return asset
        else:
            if asset.is_parametrized:
                return asset
            else:
                parametrized_asset = asset.specify({'moving_channel': moving_channel, 'fixed_channel': fixed_channel})
                self.workspace.asset_collections[channel][asset_type] = parametrized_asset
                return parametrized_asset

    def get_img_to_register(self, channel, other_channel):
        if other_channel == 'atlas':
            return self.get('atlas', channel=channel, asset_sub_type='reference')
        else:
            return self.get('resampled', channel=other_channel)

    def get_moving_image(self, channel):
        _, moving_channel = self.get_fixed_moving_channels(channel)
        return self.get_img_to_register(channel, moving_channel)

    def get_fixed_image(self, channel):
        fixed_channel, _ = self.get_fixed_moving_channels(channel)
        return self.get_img_to_register(channel, fixed_channel)

    def get_aligned_image(self, channel):
        aligned = self.get_elx_asset('aligned', channel=channel)
        return aligned.all_existing_paths(sort=True)[-1]  # The last step is the final result

    def resample_channel(self, channel, increment_main=False):  # set increment_main to True for channels > 0
        resampled_asset = self.get('resampled', channel=channel)
        if not runs_on_ui() and resampled_asset.exists:
            resampled_asset.delete()
        if resampled_asset.exists:
            raise FileExistsError(f'Resampled asset ({resampled_asset}) already exists')
        default_resample_parameter = {
            'processes': self.machine_config['n_processes_resampling'],
            'verbose': self.config['verbose']
        }  # WARNING: duplicate (use method ??)
        source_asset = self.get('stitched', channel=channel, default=None)
        source_asset = source_asset if source_asset.exists else self.get('raw', channel)
        if not source_asset.exists:
            raise FileNotFoundError(f'Cannot resample {channel}, source {source_asset} missing')

        if source_asset.is_tiled:
            src_res = define_auto_resolution(source_asset.file_list[0], self.sample_manager.config['channels'][channel]['resolution'])
        else:
            src_res = self.sample_manager.config['channels'][channel]['resolution']

        if source_asset.is_tiled:
            if 'Z' in source_asset.tag_names:
                n_planes = source_asset.expression.tag_range('Z')[1] + 1
            else:
                n_planes = clearmap_io.shape(source_asset.file_list[0])[0]
        else: # Stacked or single file, take the first dimension of the asset
            n_planes = source_asset.shape()[0]

        self.prepare_watcher_for_substep(n_planes, self.__resample_re, f'Resampling {channel}',
                                         increment_main=increment_main)

        result = resampling.resample(str(source_asset.path), resampled=str(resampled_asset.path),
                                     original_resolution=src_res,
                                     resampled_resolution=self.config['channels'][channel]['resampled_resolution'],
                                     workspace=self.workspace,
                                     **default_resample_parameter)
        try:
            pass
        except BrokenProcessPool:
            print('Resampling canceled')
            return
        assert result.array.max() != 0, f'Resampled {channel} has no data'
        assert resampled_asset.exists, f'Resampled {channel} not saved at {resampled_asset.path}'

    @property
    def n_registration_steps(self):
        n_steps_atlas_setup = 1
        n_steps_align = 2  # WARNING: probably 1 more when arteries included
        n_resampling_steps = len(self.sample_manager.channels_to_resample())
        return n_steps_atlas_setup + n_resampling_steps + n_steps_align

    @check_stopped
    def resample_for_registration(self, _force=False):
        for i, channel in enumerate(self.sample_manager.channels_to_resample()):
            self.resample_channel(channel, increment_main=i != 0)
            if self.stopped:
                return
        self.update_watcher_main_progress()

    @check_stopped
    def align(self, _force=False):
        try:
            for channel in self.channels_to_register():
                self.align_channel(channel)
                self.update_watcher_main_progress()
        except CanceledProcessing:
            print('Alignment canceled')
        self.stopped = False

    def align_channel(self, channel):
        fixed_channel, moving_channel = self.get_fixed_moving_channels(channel)
        if moving_channel is None or moving_channel == 'intrinsically aligned':
            return
        channel_cfg = self.config['channels'][channel]
        run_bspline = any(['bspline' in channel_cfg['params_files']])
        n_steps = 17000 if run_bspline else 2000
        regexp = self.__bspline_registration_re if run_bspline else self.__affine_registration_re
        self.prepare_watcher_for_substep(n_steps, regexp, f'Align {moving_channel} to {fixed_channel}')
        align_parameters = {
            "moving_image": self.get_moving_image(channel).existing_path,
            "fixed_image": self.get_fixed_image(channel).existing_path,

            'parameter_files': self.registration_params_files[channel],

            "result_directory": self.get_elx_asset('aligned', channel=channel).path.parent,
            'workspace': self.workspace,  # FIXME: use semaphore instead
            'check_alignment_success': True
        }

        landmarks_steps = [step for step, weight in zip(channel_cfg['params_files'], channel_cfg['landmarks_weights'])
                           if weight > 0]
        if landmarks_steps:
            if len(landmarks_steps) != len(self.registration_params_files[channel]):
                raise NotImplemented('Selecting landmarks for a subset of steps is currently not implemented')
            landmarks_files = {
                'moving_landmarks_path': self.get_elx_asset('moving_landmarks', channel=channel).path,
                'fixed_landmarks_path': self.get_elx_asset('fixed_landmarks', channel=channel).path,
            }
        else:
            landmarks_files = {'moving_landmarks_path': '', 'fixed_landmarks_path': ''}  # Disable landmarks w/ empty str
        elastix.align_from_dict(align_parameters, landmarks_files, landmarks_weights=channel_cfg['landmarks_weights'])

    def get_atlas_files(self):
        if not self.get('atlas', asset_sub_type='annotation',
                        channel=self.sample_manager.alignment_reference_channel).exists:
            self.setup_atlases()
        atlas_files = {}
        for channel in self.config['channels']:
            atlas_files[channel] = self.annotators[channel].get_atlas_paths()
        return atlas_files

    def __setup_source_atlas(self, atlas_base_name):
        default_annotator = Annotation(atlas_base_name, None, None, label_source='ABA json 2022')
        # TODO: use workspace instead
        channel_spec = ChannelSpec(channel_name='atlas', content_type='atlas')
        self.create_atlas_asset(default_annotator, channel_spec)

    def create_atlas_asset(self, annotator, channel_spec):  # FIXME: ensure that uses atlas subfolder from asset_constants
        try:
            atlas_asset = self.get('atlas', channel=channel_spec.name)
            return atlas_asset
        except KeyError:
            type_spec = TypeSpec(
                resource_type='atlas',
                type_name='atlas',
                file_format_category='image',
                relevant_pipelines=['registration']
            )
            atlas_asset = self.workspace.create_asset(type_spec, channel_spec=channel_spec, sample_id=self.sample_manager.prefix)
            for sub_type_name, file_path in annotator.get_atlas_paths().items():
                atlas_asset.type_spec.add_sub_type(sub_type_name, expression=os.path.abspath(file_path))
            atlas_asset.type_spec.add_sub_type('label', expression=annotator.label_file,
                                               extensions=['.json'])
            return atlas_asset

    def project_mini_brain(self, channel):  # FIXME: idealy part of sample_manager
        """
        Project the mini brain of the channel as a mask and a surface projection

        Parameters
        ----------
        channel: str
            The channel to project

        Returns
        -------
        np.ndarray, np.ndarray
            The mask and the projection
        """
        img = self.__transform_mini_brain(channel)
        mask, proj = surface_project(img)
        return mask, proj

    def __transform_mini_brain(self, channel):  # REFACTOR: move to preprocessor
        """
        Apply the set of transforms to the mini brain as defined by the crop and
        orientation parameters input by the user.

        Returns
        -------
        np.ndarray
            The transformed mini brain
        """
        def scale_range(rng, scale):
            for i in range(len(rng)):
                if rng[i] is not None:
                    rng[i] = round(rng[i] / scale)
            return rng

        def range_or_default(rng, scale):
            if rng is not None:
                return scale_range(rng, scale)
            else:
                return 0, None

        params = self.sample_manager.config['channels'][channel]
        orientation = params['orientation']
        img = self.mini_brains[channel]['array'].copy()
        x_scale, y_scale, z_scale = self.mini_brains[channel]['scaling']

        if axes_to_flip := [abs(axis) - 1 for axis in orientation if axis < 0]:
            img = np.flip(img, axes_to_flip)
        img = img.transpose([abs(axis) - 1 for axis in orientation])
        x_min, x_max = range_or_default(params['slicing']['x'], x_scale)
        y_min, y_max = range_or_default(params['slicing']['y'], y_scale)
        z_min, z_max = range_or_default(params['slicing']['z'], z_scale)
        img = img[x_min:x_max, y_min:y_max:, z_min:z_max]
        return img

    def setup_atlases(self):  # TODO: add possibility to load custom reference file (i.e. defaults to None in cfg)
        if not self.config:
            return  # Not setup yet. TODO: find better way around
        self.prepare_watcher_for_substep(0, None, 'Initialising atlases')

        sample_cfg = self.sample_manager.config['channels']
        atlas_cfg = self.config['atlas']

        atlas_base_name = ATLAS_NAMES_MAP[atlas_cfg['id']]['base_name']
        self.__setup_source_atlas(atlas_base_name)

        orientation = None
        # TODO: atlas variants as multichannel assets
        for channel in sample_cfg.keys():
            if sample_cfg[channel]['orientation'] != orientation:
                orientation = sample_cfg[channel]['orientation']
            slicing = sample_cfg[channel]['slicing']
            if slicing is not None and slicing.values() != (None, None, None):
                xyz_slicing = tuple(slice(None) if slc is None else slice(*slc) for slc in slicing.values())
            else:
                xyz_slicing = None

            if xyz_slicing is None and (orientation is None or orientation == DEFAULT_ORIENTATION):
                target_directory = settings.atlas_folder  # For the unchanged atlas
            else:
                target_directory = self.sample_manager.src_directory / 'atlas'  # FIXME: user asset_constants

            self.annotators[channel] = Annotation(atlas_base_name, xyz_slicing, orientation,
                                                  label_source=atlas_cfg['structure_tree_id'],
                                                  target_directory=target_directory)

            scaling, mini_brain = setup_mini_brain(atlas_base_name)
            self.mini_brains[channel] = {'scaling': scaling,
                                         'array': mini_brain}

            # Add to workspace
            channel_spec = self.get('raw', channel=channel).channel_spec
            atlas_asset = self.create_atlas_asset(self.annotators[channel], channel_spec)
            self.workspace.add_asset(atlas_asset)

        self.update_watcher_main_progress()


class StitchingProcessor(TabProcessor):
    """
    This class is used to manage the stitching process
    Handle image stitching operations.
    Manage stitching configurations and processes.
    """
    def __init__(self, sample_manager):
        super().__init__()
        self.sample_manager = sample_manager
        self.config = {}  # REFACTOR: only cfg
        self.progress_watcher = None  # FIXME:
        self.__wobbly_stitching_place_re = 'done constructing constraints for component'
        self.__wobbly_stitching_align_lyt_re = ('Alignment: Wobbly alignment',
                                                re.compile(r"Alignment:\sWobbly alignment \(\d+, \d+\)->\(\d+, \d+\) "
                                                           r"along axis [0-3] done: elapsed time: \d+:\d{2}:\d{2}.\d+"))
        self.__wobbly_stitching_stitch_re = ('Stitching: stitching',
                                             re.compile(r'Stitching: stitching wobbly slice \d+/\d+'))
        self.__rigid_stitching_align_re = ('done',
                                           re.compile(r"Alignment: aligning \(\d+, \d+\) with \(\d+, \d+\), alignment"
                                                      r" pair \d+/\d+ done, shift = \(-?\d+, -?\d+, -?\d+\),"
                                                      r" quality = -\d+\.\d+e\+\d+!"))

    def setup(self, sample_manager=None, convert_tiles=False):
        self.sample_manager = sample_manager if sample_manager else self.sample_manager
        if not self.sample_manager.stitching_cfg:
            raise ValueError('Stitching config not set in sample manager')
        self.config = self.sample_manager.stitching_cfg
        self.machine_config = self.sample_manager.machine_config
        self.workspace = self.sample_manager.workspace
        if convert_tiles:
            self.convert_tiles()  # TODO: check if needed

    @check_stopped
    def convert_tiles(self, _force=False):
        """
        Convert list of input files to numpy files for efficiency reasons

        Returns
        -------

        """
        use_npy_backup = {}
        for channel in self.sample_manager.stitchable_channels:
            cfg = self.config['channels'][channel]
            if _force:
                use_npy_backup[channel] = cfg['use_npy']
                cfg['use_npy'] = True
            self.convert_tiles_channel(channel)
        self.update_watcher_main_progress()
        if _force:
            for channel in self.sample_manager.stitchable_channels:
                self.config['channels'][channel]['use_npy'] = use_npy_backup[channel]

    def convert_tiles_channel(self, channel):
        if self.config['channels'][channel]['use_npy']:
            asset = self.get('raw', channel=channel, prefix=self.sample_manager.prefix)
            file_list = asset.file_list
            if not file_list or Path(file_list[0]).suffix == '.tif':
                try:
                    asset.convert('.npy', processes=self.machine_config['n_processes_file_conv'],
                                  workspace=self.workspace, verbose=self.verbose)
                except BrokenProcessPool:
                    print(f'File conversion of {channel} to numpy canceled')
                    return

    def get_stitching_order(self, strict=False):
        """
        Returns a list of trees (each tree is a list of channels in hierarchical order).
        Raises a ValueError if any channel is unreachable or if there is a cycle.
        """
        config = self.config
        channels = set(config['channels'].keys())  # All channels
        root_channels = [ch for ch, cfg in config['channels'].items()
                         if cfg.get('layout_channel', ch) == ch]
        visited = set()
        forest = {}

        def traverse_tree(root):
            stack = [root]
            tree = []
            while stack:
                node = stack.pop()
                tree.append(node)
                visited.add(node)
                # Find immediate children whose layout_channel == node
                for ch, c_cfg in config['channels'].items():
                    if ch not in visited and c_cfg.get('layout_channel', ch) == node:
                        stack.append(ch)
            return tree

        def tree_is_stitchable(root_channel, strict):
            cfg = self.config['channels'].get(root_channel)
            has_layout = self.get('layout', channel=root_channel, asset_sub_type='placed').exists
            if not (cfg['run'] or has_layout):
                msg = (f"Cannot stitch tree with root `{root_channel}`:"
                       f" {root_channel} does not have a layout and is not set for stitching.")
                if strict:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg, stacklevel=2)
                return False
            return True

        for root in root_channels:
            if not tree_is_stitchable(root, strict):
                continue
            if root not in visited:
                forest[root] = traverse_tree(root)

        unassigned = channels - visited
        if unassigned:
            raise ValueError(f"Unreachable or extra channels found: {unassigned}")
        return forest

    def stack_columns(self, channel):
        asset = self.get('raw', channel=channel, prefix=self.sample_manager.prefix)
        exp = asset.expression
        if exp.n_tags() >= 3:
            for y in range(exp.tag_max('Y')):
                for x in range(exp.tag_max('X')):
                    column_expression = exp.string(values={'X': x, 'Y': y})
                    dest = column_expression.replace('_xyz-Table Z<Z,4>', '')  # REFACTOR: find better way
                    clearmap_io.convert(column_expression, dest)
            # squash Z axis  # REFACTOR: find better way
            asset.expression = exp.string().replace('_xyz-Table Z<Z,4>', '')  # overwrite expression
            self.sample_manager.config['channels'][channel]['path'] = asset.expression
            self.sample_manager.config.write()

    def copy_or_stack(self, channel):
        """
        Copy or stack the channel data in case there is no X/Y tiling

        Parameters
        ----------
        channel : str
            The channel to copy or stack
        """
        clearmap_io.convert(self.get_path('raw', channel=channel),
                            self.get_path('stitched', channel=channel))


    def stitch(self):
        if self.stopped:
            return

        # Get the channels in dependency order (based on layout_channel)
        for stitching_tree in self.get_stitching_order().values():
            for channel in stitching_tree:
                self.stack_columns(channel)  # If x/y/z, stack first
                channel_cfg = self.config['channels'][channel]
                if channel == channel_cfg['layout_channel']:
                    if not channel_cfg['rigid']['skip']:
                        self.stitch_channel_rigid(channel)
                    if not channel_cfg['wobbly']['skip']:
                        self.stitch_channel_wobbly(channel)
                else:
                    self._stitch_layout_wobbly(channel)

                if self.stopped:
                    return

    def channel_was_stitched_rigid(self, channel):
        return self.get('layout', channel=channel, asset_sub_type='aligned_axis').exists

    @property
    def n_rigid_steps_to_run(self):
        cfg = self.config['channels']
        return [not cfg[channel]['rigid']['skip'] for channel in cfg.keys() if 'rigid' in cfg[channel]].count(True)

    # @check_stopped
    # @requires_assets([FilePath('raw')])
    def stitch_channel_rigid(self, channel, _force=False):
        if not self.sample_manager.check_has_all_tiles(channel):
            if self.sample_manager.use_npy(channel):
                self.convert_tiles_channel(channel)
                if not self.sample_manager.check_has_all_tiles(channel):
                    raise MissingRequirementException(f'Channel {channel} missing tiles')
            else:
                raise MissingRequirementException(f'Channel {channel} missing tiles')
        self.set_watcher_step(f'Stitching {channel} rigid')
        rigid_cfg = self.config['channels'][channel]['rigid']

        raw_asset = self.get('raw', channel=channel)
        if raw_asset.is_expression:
            params_file = raw_asset.file_list[0]
        else:
            params_file = raw_asset.existing_path
        overlaps, projection_thickness = define_auto_stitching_params(
            params_file,
            rigid_cfg)
        layout = self.get_wobbly_layout(channel, overlaps)
        if rigid_cfg['background_pixels'] is None:
            background_params = rigid_cfg['background_level']
        else:
            background_params = (rigid_cfg['background_level'],
                                 rigid_cfg['background_pixels'])
        max_shifts = [rigid_cfg[f'max_shifts_{ax}'] for ax in 'xyz']
        self.prepare_watcher_for_substep(len(layout.alignments), self.__rigid_stitching_align_re, 'Align layout rigid')
        try:
            stitching_rigid.align_layout_rigid_mip(layout, depth=projection_thickness, max_shifts=max_shifts,
                                                   ranges=[None, None, None], background=background_params,
                                                   clip=25000, processes=self.machine_config['n_processes_stitching'],
                                                   workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Stitching canceled')
            self.stopped = True  # FIXME: see if appropriate solution
            return  # WARNING: do not run stitching_wobbly if rigid failed
        layout.place(method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)
        self.update_watcher_main_progress()

        stitching_rigid.save_layout(self.filename('layout', channel=channel, asset_sub_type='aligned_axis'),
                                    layout)

    # @requires_assets([FilePath('raw')])  # TODO: optional requires npy + requires that channel is kwarg
    def get_wobbly_layout(self, channel, overlaps=None):
        if overlaps is None:
            rigid_cfg = self.config['channels'][channel]['rigid']
            overlaps, projection_thickness = define_auto_stitching_params(
                self.get('raw', channel=channel).file_list[0], rigid_cfg).as_source()
        extension = '.npy' if self.sample_manager.use_npy(channel) else None  # TODO: optional requires
        raw_expr = str(self.get_path('raw', channel=channel, extension=extension))
        layout = stitching_wobbly.WobblyLayout(expression=raw_expr, tile_axes=('X', 'Y'), overlaps=overlaps)
        return layout

    @property
    def n_wobbly_steps_to_run(self):
        out = len(self.sample_manager.stitchable_channels) - 1
        for channel in self.sample_manager.stitchable_channels:
            cfg = self.config['channels'][channel]
            if 'wobbly' in cfg and not cfg['wobbly']['skip']:
                out += 3
        return out

    def __align_layout_wobbly(self, channel, layout):
        wobbly_cfg = self.config['channels'][channel]['wobbly']
        max_shifts = [
            wobbly_cfg['max_shifts_x'],
            wobbly_cfg['max_shifts_y'],
            wobbly_cfg['max_shifts_z'],
        ]
        stack_validation_params = dict(
            method='foreground',
            valid_range=wobbly_cfg["stack_valid_range"],
            size=wobbly_cfg["stack_pixel_size"]
        )
        slice_validation_params = dict(
            method='foreground',
            valid_range=wobbly_cfg["slice_valid_range"],
            size=wobbly_cfg["slice_pixel_size"]
        )

        n_pairs = len(layout.alignments)
        self.prepare_watcher_for_substep(n_pairs, self.__wobbly_stitching_align_lyt_re, 'Align layout wobbly')
        try:
            stitching_wobbly.align_layout(layout, axis_range=(None, None, 3), max_shifts=max_shifts, axis_mip=None,
                                          stack_validation_params=stack_validation_params,
                                          prepare=dict(method='normalization', clip=None, normalize=True),
                                          slice_validation_params=slice_validation_params,
                                          prepare_slice=None,
                                          find_shifts=dict(method='tracing', cutoff=3 * np.sqrt(2)),
                                          processes=self.machine_config['n_processes_stitching'],
                                          workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return
        self.update_watcher_main_progress()

    def __place_layout_wobbly(self, layout):
        self.prepare_watcher_for_substep(len(layout.alignments) // 2,  # WARNING: bad estimation
                                         self.__wobbly_stitching_place_re, 'Place layout wobbly')
        try:
            n_processes = self.machine_config['n_processes_stitching']
            if platform.system().lower().startswith('darwin'):  # No parallel on MacOS
                n_processes = 1
            stitching_wobbly.place_layout(layout, min_quality=-np.inf,
                                          method='optimization',
                                          smooth=dict(method='window', window='bartlett', window_length=100,
                                                      binary=None),
                                          smooth_optimized=dict(method='window', window='bartlett',
                                                                window_length=20, binary=10),
                                          fix_isolated=False, lower_to_origin=True,
                                          processes=n_processes,
                                          workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return
        self.update_watcher_main_progress()

    def _stitch_layout_wobbly(self, channel):
        layout_channel = self.config['channels'][channel]['layout_channel']
        layout = stitching_rigid.load_layout(self.get_path('layout', channel=layout_channel, postfix='placed'))

        try:
            ref_asset = self.get('raw', channel=self.sample_manager.alignment_reference_channel)
            n_slices = len(ref_asset.file_list)  # TODO: find better proxy
        except KeyError:
            n_slices = clearmap_io.shape(self.get('raw', channel=channel).file_list[0])[0]
        self.prepare_watcher_for_substep(n_slices, self.__wobbly_stitching_stitch_re,
                                         'Stitch layout wobbly', True)
        try:
            layout_channel_asset = self.get('raw', channel=layout_channel)
            channel_asset = self.get('raw', channel=channel)
            if layout_channel != channel:
                layout_extension = Path(layout.sources[0].location).suffix  # Use the actual extension that was used
                if self.sample_manager.use_npy(channel):  # FIXME: check if we need to copy layout first
                    channel_pattern = channel_asset.with_extension(extension='.npy')
                else:
                    channel_pattern = channel_asset.path
                layout.replace_source_location(None, str(channel_pattern), method='infer')
            stitching_wobbly.stitch_layout(layout,
                                           sink=str(self.get_path('stitched', channel=channel)),
                                           method='interpolation',
                                           processes=self.machine_config['n_processes_stitching'],
                                           workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return

        self.update_watcher_main_progress()

    @check_stopped
    def stitch_channel_wobbly(self, channel, _force=False):  # Warning, will stitch channel on its own
        def get_layout_path(layout_sub_type):
            return self.get_path('layout', channel=channel, asset_sub_type=layout_sub_type)
# FIXME: check if ui_to_cfg is needed
        if not self.channel_was_stitched_rigid(channel):
            raise MissingRequirementException(f'Channel {channel} not stitched rigid yet')
        self.set_watcher_step('Stitching wobbly')
        layout = stitching_rigid.load_layout(get_layout_path('aligned_axis'))
        self.__align_layout_wobbly(channel, layout)
        if self.stopped:
            return
        stitching_rigid.save_layout(get_layout_path('aligned'), layout)

        self.__place_layout_wobbly(layout)
        if self.stopped:
            return
        stitching_rigid.save_layout(get_layout_path('placed'), layout)

        self._stitch_layout_wobbly(channel)
        if self.stopped:
            return

    def plot_stitching_results(self, channels=None, mode='side-by-side', parent=None):
        if channels is None:
            channels = self.sample_manager.stitchable_channels
        paths = []
        titles = []
        for c in channels:
            asset = self.get('stitched', channel=c)
            if asset.exists:
                paths.append(asset.path)
                titles.append(f'{c.title()} stitched')
            else:
                print(f'No stitched file for channel {c}')
        if not paths:
            raise MissingRequirementException('No stitched files found')
        if len(paths) == 1:
            paths = paths[0]
        if mode == 'overlay':
            titles = ' and '.join(titles)
            paths = [paths]
        elif mode == 'side-by-side':
            pass
        else:
            raise ValueError(f'Unknown mode {mode}')

        dvs = plot_3d.plot(paths, title=titles, arrange=False, lut='white', parent=parent)
        return dvs

    def stitch_overlay(self, channel, color=True):
        """
        This creates a *dumb* overlay of the tiles
        i.e. only using the fixed guess overlap

        Parameters
        ----------
        channel
        color

        Returns
        -------
        np.array(dtype=uint8)
            The overlay image
        """
        asset = self.get('raw', channel=channel, sample_id=self.sample_manager.prefix)
        positions = asset.positions
        tile_shape = {k: v for k, v in zip('XYZ', asset.tile_shape)}  # TODO: use asset.tile_grid_shape
        middle_z = int(tile_shape['Z'] / 2)
        overlaps = self._get_overlaps(channel)
        output_shape = self._compute_stitched_shape_from_overlaps(overlaps, positions, tile_shape)
        layers = [np.zeros(output_shape, dtype=int), np.zeros(output_shape, dtype=int)]
        if self.sample_manager.has_npy:
            files = asset.variant(extension='.npy').file_list
        else:
            files = asset.file_list
        for tile_path, pos in zip(files, positions):
            tile = self.__read_tile_middle_plane(tile_path, middle_z)

            starts = {ax: pos[ax] * tile_shape[ax] - pos[ax] * overlaps[ax] for ax in 'XY'}
            ends = {ax: starts[ax] + tile_shape[ax] for ax in starts.keys()}
            layer = layers[(pos['Y'] + pos['X']) % 2]  # Alternate colors
            layer[starts['X']: ends['X'], starts['Y']: ends['Y']] = tile
        if color:
            layers[0] = gray_image_to_rgb(layers[0], 'cyan', pseudo_z_score=True, range_max=255)
            layers[1] = gray_image_to_rgb(layers[1], 'magenta', pseudo_z_score=True, range_max=255)
        output_image = layers[0] + layers[1]
        if color:
            output_image = output_image.clip(0, 255).astype(np.uint8)
        return output_image

    def __read_tile_middle_plane(self, tile_path, middle_z):
        if self.sample_manager.has_npy:  # use memmap
            tile = clearmap_io.buffer(tile_path)[:, :, middle_z]
        else:
            tile = clearmap_io.read(tile_path)[:, :, middle_z]
        return tile

    def _compute_stitched_shape_from_overlaps(self, overlaps, positions, tile_shape):
        mosaic_shape = {ax: max([p[ax] for p in positions]) + 1 for ax in 'XY'}  # +1 because 0 indexing
        output_shape = [tile_shape[ax] * mosaic_shape[ax] - overlaps[ax] * (mosaic_shape[ax] - 1) for ax in 'XY']
        return output_shape

    def _get_overlaps(self, channel):
        layout_channel = self.config['channels'][channel]['layout_channel']
        overlaps = {k: self.config['channels'][layout_channel]['rigid'][f'overlap_{k.lower()}'] for k in 'XY'}
        return overlaps

    def overlay_layout_plane(self, layout):  # REFACTOR: move to e.g. layout class
        """Overlays the sources to check their placement.

        Arguments
        ---------
        layout : Layout class
          The layout with the sources to overlay.

        Returns
        -------
        image : array
          A color image.
        """
        dest_shape = tuple(layout.extent[:-1])
        full_lower = layout.lower
        middle_z = round(layout.sources[0].shape[-1] / 2)

        color_layers = [np.zeros(dest_shape, dtype=int), np.zeros(dest_shape, dtype=int)]
        # construct full image
        for src in layout.sources:
            tile = self.__read_tile_middle_plane(src.location,
                                                 middle_z)  # So as not to load the data into the list for memory efficiency
            is_odd = sum(src.tile_position) % 2
            layer = color_layers[is_odd]  # Alternate colors

            current_slicing = self.__compute_overlay_slicing(full_lower, src)
            layer[current_slicing] = tile

        cyan_image = gray_image_to_rgb(color_layers[0], 'cyan', pseudo_z_score=True, range_max=255)
        magenta_image = gray_image_to_rgb(color_layers[1], 'magenta', pseudo_z_score=True, range_max=255)

        # TODO: normalise
        output_image = np.clip(cyan_image + magenta_image, 0, 255).astype(np.uint8)

        return output_image

    def __compute_overlay_slicing(self, full_lower, src):
        l = src.lower
        u = src.upper
        current_slicing = tuple(slice(ll - fl, uu - fl) for ll, uu, fl in zip(l, u, full_lower))[:2]
        return current_slicing

    def plot_layout(self, channel, asset_sub_type='aligned_axis'):
        valid_sub_types = ("aligned_axis", "aligned", "placed")
        if asset_sub_type not in valid_sub_types:
            raise ValueError(f'Expected on of {valid_sub_types} for asset_sub_type, got "{asset_sub_type}"')
        layout = stitching_rigid.load_layout(self.get_path('layout', channel=channel, asset_sub_type=asset_sub_type))
        overlay = self.overlay_layout_plane(layout)
        return overlay


def init_sample_manager_and_processors(folder='', configs=None):
    sample_manager = SampleManager()
    if folder:
        sample_manager.setup(src_dir=folder)
    elif configs:
        sample_manager.setup(configs)

    stitcher = StitchingProcessor(sample_manager)
    stitcher.setup()

    registration_processor = RegistrationProcessor(sample_manager)
    registration_processor.setup()

    return {'sample_manager': sample_manager, 'stitcher': stitcher, 'registration_processor': registration_processor}
