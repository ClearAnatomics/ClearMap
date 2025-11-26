import os
import re
import warnings
from concurrent.futures.process import BrokenProcessPool
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, TypedDict

import numpy as np

from ClearMap import Settings as settings
from ClearMap.Alignment import Resampling as resampling, Elastix as elastix
from ClearMap.Alignment.Annotation import Annotation
from ClearMap.IO import IO as clearmap_io
from ClearMap.IO.assets_specs import ChannelSpec, TypeSpec
from ClearMap.IO.metadata import define_auto_resolution
from ClearMap.Utils.events import ChannelRenamed, UiAtlasIdChanged, UiAtlasStructureTreeIdChanged, \
    RegistrationStatusChanged
from ClearMap.Utils.exceptions import ClearMapAssetError, ParamsOrientationError
from ClearMap.Utils.utilities import runs_on_ui, check_stopped, DEFAULT_ORIENTATION, validate_orientation
from ClearMap.Visualization import Plot3d as q_plot_3d
from ClearMap.config.atlas import ATLAS_NAMES_MAP
from ClearMap.config.config_coordinator import ConfigCoordinator
from ClearMap.gui.gui_utils_images import surface_project, setup_mini_brain
from ClearMap.gui.widgets import ProgressWatcher
from ClearMap.pipeline_orchestrators.generic_orchestrators import PipelineOrchestrator, CanceledProcessing
from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager


class RegistrationStatus(Enum):
    NOT_SELECTED = 0
    MISSING_OUTPUTS = 1
    REGISTERED = 2


class RegistrationProcessor(PipelineOrchestrator):
    """
    This class is used to manage the registration process
    Perform image registration operations.
    Manage atlas setup and transformations.
    Handle registration configurations.
    """
    config_name = 'registration'
    def __init__(self, sample_manager: SampleManager, cfg_coordinator: ConfigCoordinator):
        super().__init__(cfg_coordinator)
        self.sample_manager: SampleManager = sample_manager
        self.annotators: Dict[str, Annotation] = {}  # 1 for each channel
        self.mini_brains: Dict[str, MiniBrain] = {}  # 1 for each channel
        self.progress_watcher: Optional[ProgressWatcher] = None  # FIXME:
        self.__bspline_registration_re = re.compile(r"\d+\s-?\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+")
        self.__affine_registration_re = re.compile(r"\d+\s-\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+")
        self.__resample_re = ('Resampling: resampling',
                              re.compile(r".*?Resampling:\sresampling\saxes\s.+\s?,\sslice\s.+\s/\s\d+"))

        self.setup_complete: bool = False

        self.subscribe(ChannelRenamed, self._on_channel_renamed)
        self.subscribe(UiAtlasIdChanged, self.setup_atlases)
        self.subscribe(UiAtlasStructureTreeIdChanged, self.setup_atlases)

    def setup(self, sample_manager: Optional[SampleManager] = None):
        self.sample_manager = sample_manager if sample_manager else self.sample_manager
        if not self.registration_config:
            raise ValueError('Registration config not set in config coordinator')

        if self.sample_manager is None:
            warnings.warn('SampleManager not provided, RegistrationProcessor setup incomplete')
            self.setup_complete = False
        elif self.sample_manager.setup_complete:
            self.workspace = self.sample_manager.workspace
            self.setup_atlases()  # TODO: check if needed
            self.add_pipeline()
            self.setup_complete = True
        else:
            self.setup_complete = False  # FIXME: finish later
            warnings.warn('SampleManager not setup, RegistrationProcessor setup incomplete')

        # WARNING: must be called once registration pipeline has been added to the Workspace for that channel
        # self.parametrize_assets()

    def _on_channel_renamed(self, event: ChannelRenamed):
        if event.old in self.annotators:
            self.annotators[event.new] = self.annotators.pop(event.old)
        if event.old in self.mini_brains:
            self.mini_brains[event.new] = self.mini_brains.pop(event.old)

    @property
    def ref_channel_cfg(self):
        if self.sample_manager is None:
            raise ValueError('CellDetector not properly initialized')
        ref_channel = self.sample_manager.alignment_reference_channel
        if ref_channel is None:
            raise ValueError('No alignment reference channel specified in sample manager')
        reg_cfg = self.registration_config
        if ref_channel not in reg_cfg['channels']:
            raise ValueError(f'Reference channel "{ref_channel}" not found in registration config')
        return reg_cfg['channels'][ref_channel]

    def get_registration_sequence_channels(self, first_channel, stop_channel='atlas'):
        out = [first_channel]
        registration_cfg = self.registration_config['channels']
        while True:
            next_channel = registration_cfg[out[-1]]['align_with']
            if next_channel in (None, stop_channel):
                break
            out.append(next_channel)
        return out

    def parametrize_assets(self):
        for channel in self.config['channels']:
            if self.config['channels'][channel]['align_with'] is None:
                continue
            if self.config['channels'][channel]['moving_channel'] in (None, 'intrinsically_aligned'):
                continue
            for asset_type in ('fixed_landmarks', 'moving_landmarks', 'aligned'):
                try:
                    asset = self.get_elx_asset(asset_type, channel=channel)
                except KeyError:
                    continue  # FIXME: this should be handled more elegantly
                              #  the idea is to delay the parametrization
                              #  until the assets for all channels have been created
                except ClearMapAssetError:  # Check that align_with is None
                    warnings.warn(f'Could not parametrize {asset_type} for {channel=}')
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

    @property
    def channels(self):
        return list(self.config['channels'].keys())

    def channels_to_resample(self):
        return [c for c, v in self.config['channels'].items() if v['resample']]

    def channels_to_register(self):
        return [c for c, v in self.config['channels'].items() if v['align_with'] is not None]

    def get_align_with(self, channel):
        return self.config['channels'][channel]['align_with']

    def get_moving_channel(self, channel: str) -> str:
        """
        Get the moving channel for a given channel

        .. warning::

            Contrary to get_fixed_moving_channels, this method does not
            check for the existence of the fixed channel. It simply returns
            the moving channel as specified in the config.

        Parameters
        ----------
        channel: str
            The channel to get the moving channel for

        Returns
        -------
        str
            The moving channel
        """
        return self.config['channels'][channel]['moving_channel']

    @property
    def was_registered(self):
        return self.registration_status() == RegistrationStatus.REGISTERED

    def channel_was_registered(self, channel):
        align_with = self.get_align_with(channel)
        moving_channel = self.get_moving_channel(channel)
        asset = self.get('aligned', channel=channel)
        fixed_channel = channel if align_with == moving_channel else align_with
        return asset.specify({'moving_channel': moving_channel, 'fixed_channel': fixed_channel}).exists

    def registration_status(self):
        reg_cfg = self.registration_config

        def is_selected(ch_cfg: dict) -> bool:
            # user opted-in this channel for registration?
            align_with = ch_cfg.get('align_with')
            return align_with not in (None, '', 'none')

        def is_intrinsically_aligned(ch_cfg: dict) -> bool:
            return ch_cfg.get('moving_channel') == 'intrinsically_aligned'

        any_selected = any(is_selected(reg_cfg['channels'][ch]) and not is_intrinsically_aligned(reg_cfg['channels'][ch])
                           for ch in self.channels)
        if not any_selected:
            return RegistrationStatus.NOT_SELECTED
        else:
            ref_channel = self.sample_manager.alignment_reference_channel
            for channel in self.channels:
                ch_cfg = reg_cfg['channels'][channel]
                if not is_selected(ch_cfg) or is_intrinsically_aligned(ch_cfg):
                    continue
                if not ref_channel and ch_cfg['align_with'] == 'autofluorescence':
                    raise ValueError(f'This should not happen, {channel=} set for registration against '
                                     f'autofluorescence but no reference channel found')
                elif not self.channel_was_registered(channel):
                        return RegistrationStatus.MISSING_OUTPUTS  # at least one not registered
            return RegistrationStatus.REGISTERED  # all selected channels are registered

    @property
    def registration_params_files(self):
        align_dir = Path(settings.resources_path) / self.config['atlas']['align_files_folder']
        registration_params_files = {}
        for channel in self.config['channels']:
            params_file_names = self.config['channels'][channel]['params_files']
            registration_params_files[channel] = [align_dir / name for name in params_file_names]  # TODO: define as property
        return registration_params_files

    def plot_atlas(self, channel):  # REFACTOR: idealy part of sample_manager
        atlas_path = self.get_path('atlas', channel=channel, asset_sub_type='reference')
        return q_plot_3d.plot(atlas_path, lut=self.machine_config['default_lut'])

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
        moving_channel = self.get_moving_channel(channel)
        align_with = self.config['channels'][channel]['align_with']
        if align_with is None:
            return None, moving_channel
        if not align_with:
            raise KeyError(f'Channel {channel} missing align_with in registration config')
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
            src_res = define_auto_resolution(source_asset.file_list[0],
                                             self.sample_manager.get_channel_resolution(channel))
        else:
            src_res = self.sample_manager.get_channel_resolution(channel)

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
        self.publish(RegistrationStatusChanged)

    def align_channel(self, channel):
        fixed_channel, moving_channel = self.get_fixed_moving_channels(channel)
        if moving_channel is None or moving_channel == 'intrinsically_aligned':
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
        channel_spec = ChannelSpec(channel='atlas', content_type='atlas')
        self.create_atlas_asset(default_annotator, channel_spec)

    def create_atlas_asset(self, annotator, channel_spec):  # FIXME: ensure that uses atlas subfolder from asset_constants
        try:
            atlas_asset = self.get('atlas', channel=channel_spec.name, default=None)
        except KeyError:
            atlas_asset = None
        if atlas_asset is not None:
            return atlas_asset
        else:
            type_spec = TypeSpec(resource_type='atlas', type_name='atlas',
                                 file_format_category='image', relevant_pipelines=['registration'])
            atlas_asset = self.workspace.create_asset(type_spec, channel_spec=channel_spec,
                                                      sample_id=self.sample_manager.prefix)
            return self.update_atlas_asset(channel_spec.name, annotator=annotator)

    def update_atlas_asset(self, channel, annotator=None):
        if annotator is None:
            annotator = self.annotators[channel]
            sample_cfg = self.cfg_coordinator.get_config_view('sample')['channels'][channel]
            if annotator.orientation != sample_cfg['orientation'] or \
                annotator.slicing != sample_cfg['slicing']:

                slicing = sample_cfg['slicing']
                if slicing is not None and slicing.values() != (None, None, None):
                    xyz_slicing = tuple(slice(None) if slc is None else slice(*slc) for slc in slicing.values())
                else:
                    xyz_slicing = None
                orientation = sample_cfg['orientation']
                if orientation == DEFAULT_ORIENTATION:
                    warnings.warn(f'Orientation not set for {channel}, skipping atlas setup')
                    return
                atlas_cfg = self.config['atlas']
                self.annotators[channel] = Annotation(atlas_base_name=ATLAS_NAMES_MAP[atlas_cfg['id']]['base_name'],
                                                      slicing=xyz_slicing, orientation=orientation,
                                                      label_source=atlas_cfg['structure_tree_id'],
                                                      target_directory=annotator.target_directory)
                annotator = self.annotators[channel]
        atlas_asset = self.get('atlas', channel=channel)
        for sub_type_name, file_path in annotator.get_atlas_paths().items():
            sub_type = atlas_asset.type_spec.add_sub_type(sub_type_name, expression=os.path.abspath(file_path))
            asset = self.workspace.asset_collections[channel].get(f'atlas_{sub_type_name}')
            if not asset:
                asset = self.workspace.create_asset(type_spec=sub_type, channel_spec=atlas_asset.channel_spec,
                                                    sample_id=self.sample_manager.prefix)
            else:
                asset.type_spec = sub_type
            self.workspace.asset_collections[channel][f'atlas_{sub_type_name}'] = asset  # FIXME: method in workspace2
        sub_type = atlas_asset.type_spec.add_sub_type('label', expression=annotator.label_file, extensions=['.json'])
        asset = self.workspace.asset_collections[channel].get('atlas_label')
        if not asset:
            asset = self.workspace.create_asset(type_spec=sub_type, channel_spec=atlas_asset.channel_spec,
                                                sample_id=self.sample_manager.prefix)
        else:
            asset.type_spec = sub_type
        self.workspace.asset_collections[channel]['atlas_label'] = asset
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

        params = self.cfg_coordinator.get_config_view('sample')['channels'][channel]
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

    def setup_atlases(self, event=None):  # TODO: add possibility to load custom reference file (i.e. defaults to None in cfg)
        if not self.config:
            return  # Not setup yet. TODO: find better way around
        self.prepare_watcher_for_substep(0, None, 'Initialising atlases')

        sample_cfg = self.cfg_coordinator.get_config_view('sample')['channels']
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
                target_directory = self.cfg_coordinator.base_dir / 'atlas'  # FIXME: use asset_constants

            try:
                orientation = validate_orientation(orientation, channel=channel, raise_error=True)
                if orientation == DEFAULT_ORIENTATION:
                    warnings.warn(f'Orientation not set for {channel}, skipping atlas setup')
                    continue
                self.annotators[channel] = Annotation(atlas_base_name, xyz_slicing, orientation,
                                                      label_source=atlas_cfg['structure_tree_id'],
                                                      target_directory=target_directory)

                scaling, mini_brain = setup_mini_brain(atlas_base_name)
                self.mini_brains[channel] = MiniBrain(scaling=scaling,
                                                      array=mini_brain)

                # Add to workspace
                asset = self.get('atlas', channel=channel, default=None)
                if asset is None or not asset.exists:
                    channel_spec = self.get('raw', channel=channel).channel_spec
                    atlas_asset = self.create_atlas_asset(self.annotators[channel], channel_spec)
                    self.workspace.add_asset(atlas_asset)
                else:
                    # FIXME: update_asset method in workspace2
                    self.workspace.asset_collections[channel]['atlas'] = self.update_atlas_asset(channel)
            except ParamsOrientationError:
                warnings.warn(f'Orientation not set for {channel}, skipping atlas setup and erasing annotators.')
                self.annotators[channel] = None
                self.mini_brains[channel] = None

        self.update_watcher_main_progress()


class MiniBrain(TypedDict):
    """
    A downscaled brain for quick visualization
    It includes the downscaled image and the scaling factors
    """
    scaling: tuple[float, float, float]
    array: np.ndarray
