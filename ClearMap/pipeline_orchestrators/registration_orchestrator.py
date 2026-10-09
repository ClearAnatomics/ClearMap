import functools
import os
import re
import warnings
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, TypedDict, TYPE_CHECKING

import numpy as np
from skimage import transform as sk_transform

from ClearMap import Settings as settings, Settings
from ClearMap.Alignment import Resampling as resampling, Elastix as elastix
from ClearMap.Alignment.Annotation import Annotation

from ClearMap.IO import source_geometry
from ClearMap.IO.source.backends.tif_backend import TifSource, parse_img_res
from ClearMap.IO.assets_specs import TypeSpec

from ClearMap.Utils.events import (ChannelRenamed, UiAtlasIdChanged,
                                   UiAtlasStructureTreeIdChanged,  RegistrationStatusChanged, WorkspaceChannelsUpdated)
from ClearMap.Utils.exceptions import (ClearMapAssetError, ParamsOrientationError, MissingRequirementException,
                                       NotAnOmeFile, MetadataError, ClearMapRuntimeError)
from ClearMap.Utils.utilities import (runs_on_ui, check_stopped, DEFAULT_ORIENTATION,
                                      validate_orientation,  sanitize_n_processes)

from ClearMap.config.atlas import ATLAS_NAMES_MAP
from ClearMap.config.config_coordinator import ConfigCoordinator

from ClearMap.pipeline_orchestrators.generic_orchestrators import IndependentChannelsPipelineOrchestrator, CanceledProcessing
from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager

if TYPE_CHECKING:
    from ClearMap.gui.widgets import ProgressWatcher
    from ClearMap.Visualization import Plot3d as q_plot_3d  # WARNING: Local imports, for reference only


def _atlas_orientation(orientation) -> Optional[tuple[int, ...]]:
    """
    The sample orientation as a tuple (the config stores a list), so that it compares
    with DEFAULT_ORIENTATION and with the orientation of an existing Annotation.
    """
    return None if orientation is None else tuple(orientation)


def _atlas_slicing(slicing) -> Optional[tuple[slice, slice, slice]]:
    """
    The sample slicing (config: ``{'x': [start, stop] | None, 'y': ..., 'z': ...}``) as the
    xyz slices applied to the atlas, or None when no axis is sliced.
    """
    if not slicing or all(slicing.get(ax) is None for ax in 'xyz'):
        return None
    return tuple(slice(None) if slicing.get(ax) is None else slice(*slicing[ax]) for ax in 'xyz')


def _atlas_target_directory(orientation: Optional[tuple[int, ...]],
                            xyz_slicing: Optional[tuple[slice, slice, slice]], cache_dir: Path) -> Path:
    """
    Where the atlas prepared for this orientation and slicing is stored.

    The unchanged atlas is the source atlas itself. Reoriented or cropped variants go to a
    cache shared by all experiments (`cache_dir`): their file names encode the atlas, the
    orientation and the slicing, so experiments with the same parameters reuse the same files
    (about 1.2 GB per variant for the 25 um ABA).
    """
    if xyz_slicing is None and (orientation is None or orientation == DEFAULT_ORIENTATION):
        return Path(settings.atlas_folder)
    return cache_dir


class RegistrationStatus(Enum):
    NOT_SELECTED = 0
    MISSING_OUTPUTS = 1
    REGISTERED = 2


class RegistrationProcessor(IndependentChannelsPipelineOrchestrator):
    """
    This class is used to manage the registration process
    Perform image registration operations.
    Manage atlas setup and transformations.
    Handle registration configurations.
    """
    _PARAMETRIZED_ASSET_TYPES = frozenset({'aligned', 'fixed_landmarks', 'moving_landmarks'})

    pipeline = 'registration'
    config_name = 'registration'

    def __init__(self, sample_manager: SampleManager, cfg_coordinator: ConfigCoordinator):
        super().__init__(cfg_coordinator)
        self.sample_manager: SampleManager = sample_manager
        self.annotators: Dict[str, Annotation] = {}  # 1 for each channel
        self._source_annotator: Optional[Annotation] = None  # see source_annotator
        self._source_annotator_key: Optional[tuple[str, str]] = None
        self.progress_watcher: Optional["ProgressWatcher"] = None  # FIXME:
        self.__bspline_registration_re = re.compile(r"\d+\s-?\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+")
        self.__affine_registration_re = re.compile(r"\d+\s-\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+")
        self.__resample_re = ('Resampling: resampling',
                              re.compile(r".*?Resampling:\sresampling\saxes\s.+\s?,\sslice\s.+\s/\s\d+"))

        self.subscribe(ChannelRenamed, self._on_channel_renamed)
        self.subscribe(WorkspaceChannelsUpdated, self._on_workspace_channels_updated)
        self.subscribe(UiAtlasIdChanged, self._on_atlas_config_changed)
        self.subscribe(UiAtlasStructureTreeIdChanged, self._on_atlas_config_changed)

    def setup(self, sample_manager: Optional[SampleManager] = None):
        self.sample_manager = sample_manager if sample_manager else self.sample_manager
        if not self.registration_config:
            raise ValueError('Registration config not set in config coordinator')

        if self.sample_manager is None:
            warnings.warn('SampleManager not provided, RegistrationProcessor setup incomplete')
            self._setup_done = False
        elif self.sample_ready:
            self.workspace = self.sample_manager.workspace
            self.setup_atlases()  # TODO: check if needed
            self.register_in_workspace()
            self.parametrize_assets()
            self._setup_done = True
        else:
            self._setup_done = False  # FIXME: finish later
            warnings.warn('SampleManager not setup, RegistrationProcessor setup incomplete')

        # WARNING: must be called once registration pipeline has been added to the Workspace for that channel
        # self.parametrize_assets()

    def get(self, asset_type, channel, asset_sub_type=None, **kwargs):
        """
        Get an asset, automatically resolving registration template
        variables for asset types that require parametrisation (e.g. registration
        (Elastix) assets, where
        moving/fixed channels are conditional).

        Parameters
        ----------
        asset_type : str
            The asset type name.
        channel : str
            The channel name.
        asset_sub_type : str or None
            Optional sub-type.
        **kwargs
            Forwarded to the parent ``get``.

        Returns
        -------
        Asset
            The resolved asset.
        """
        asset = super().get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs)

        if (asset_type not in self._PARAMETRIZED_ASSET_TYPES
                or not asset.is_expression
                or asset.is_parametrized):  # is_parametrized available only on ExpressionAsset
            return asset

        moving_channel = self.get_moving_channel(channel)
        if moving_channel in (None, 'intrinsically_aligned'):  # No alignment planned -> no parametrization needed
            return asset

        fixed_channel, moving_channel = self.get_fixed_moving_channels(channel)
        if fixed_channel is None or moving_channel is None:
            return asset

        parametrized = asset.specify({'moving_channel': moving_channel, 'fixed_channel': fixed_channel})
        self.workspace.asset_collections[channel][asset_type] = parametrized  # UPDATE WORKSPACE to cache
        return parametrized

    def _on_workspace_channels_updated(self, event: WorkspaceChannelsUpdated) -> None:
        """
        The sample channels in the workspace changed (e.g. one became complete): once set up,
        redo the setup, which is idempotent, so that it includes all the channels now ready,
        whatever the order in which they joined.
        """
        if self._setup_done and self.sample_ready:
            self.setup()

    def _on_atlas_config_changed(self, event) -> None:
        """The atlas or its structure tree changed: update the annotators (before the setup, it reads the config)"""
        if self.setup_complete:
            self.setup_atlases()

    def _on_channel_renamed(self, event: ChannelRenamed):
        if event.old in self.annotators:
            self.annotators[event.new] = self.annotators.pop(event.old)

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

    def get_transform_directories(self, first_channel, stop_channel='atlas'):
        reg_cfg = self.registration_config['channels']
        sequence = self.get_registration_sequence_channels(first_channel, stop_channel)

        # Check that the last channel in the sequence is aligned with the stop_channel
        last_align_with = reg_cfg[sequence[-1]]['align_with']
        if last_align_with != stop_channel:
            raise MissingRequirementException(f'Channel "{first_channel}" is not aligned to "{stop_channel}": '
                                              f'its registration sequence ({" -> ".join(sequence)}) '
                                              f'ends with align_with={last_align_with!r}.')

        # Compile the list of directories for the registration steps in the sequence while checking that they exist
        directories = []
        for channel in sequence:
            if reg_cfg[channel]['moving_channel'] in (None, 'intrinsically_aligned'):
                continue  # Identity: nothing to apply
            result_dir = self.get('aligned', channel=channel).path.parent
            if not result_dir.exists():
                raise MissingRequirementException(f'Elastix result directory "{result_dir}" for channel '
                                                  f'"{channel}" not found. Please run the registration first.')
            directories.append(result_dir)
        return directories

    def parametrize_assets(self):
        for channel in self.channels:
            channel_cfg = self.config['channels'][channel]
            if channel_cfg['align_with'] is None:
                continue
            if channel_cfg['moving_channel'] in (None, 'intrinsically_aligned'):  # No alignment planned -> no param
                continue
            for asset_type in self._PARAMETRIZED_ASSET_TYPES:
                try:
                    asset = self.get(asset_type, channel=channel)  # triggers parametrization and cache to WS
                except KeyError:
                    continue  #  the idea is to delay the parametrization
                              #  until the assets for all channels have been created
                except ClearMapAssetError:  # Check that align_with is None
                    warnings.warn(f'Could not parametrize {asset_type} for {channel=}')
                    continue

    @property
    def all_channels(self) -> list[str]:
        """
        ALL the channels of the registration config, complete or not (superset of :attr:`channels`).
        What the GUI shows and edits; nothing is processed from this list.
        """
        return list(self.config['channels'].keys())

    @property
    def channels(self) -> list[str]:
        """
        The channels of the registration config whose sample channel is complete (path and data type).
        The others are only hydrated in the GUI, they are processed once complete.
        """
        complete = self.sample_manager.complete_channels if self.sample_manager is not None else []
        return [channel for channel in self.all_channels if channel in complete]

    def all_channels_to_register(self) -> list[str]:
        """
        ALL the channels the config selects for registration, complete or not (superset of
        :meth:`channels_to_register`), e.g. to list the possible partners in the GUI.
        """
        return [c for c in self.all_channels if self.config['channels'][c]['align_with'] is not None]

    def channels_to_resample(self):
        return [c for c in self.channels if self.config['channels'][c]['resample']]

    def channels_to_register(self):
        """The channels selected for registration which can be processed (i.e. complete)"""
        return [c for c in self.all_channels_to_register() if c in self.channels]

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
        for channel in self.channels:
            params_file_names = self.config['channels'][channel]['params_files']
            registration_params_files[channel] = [align_dir / name for name in params_file_names]  # TODO: define as property
        return registration_params_files

    def plot_atlas(self, channel):  # REFACTOR: idealy part of sample_manager
        from ClearMap.Visualization import Plot3d as q_plot_3d
        atlas_path = self.get_path('atlas', channel=channel, asset_sub_type='reference')
        return q_plot_3d.plot(atlas_path, lut=self.machine_config['default_lut'])

    def clear_landmarks(self, channel=None):
        """
        Clear (remove) the landmarks files
        """
        channels = [channel] if channel else self.channels
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
            raise MissingRequirementException(f'Channel {channel} missing align_with in registration config')
        # fixed is whichever channel from ('channel', 'align_with') is not 'moving_channel'
        fixed_channel = channel if align_with == moving_channel else align_with
        return fixed_channel, moving_channel

    def get_elx_asset(self, asset_type, channel):
        fixed_channel, moving_channel = self.get_fixed_moving_channels(channel)
        if fixed_channel is None or moving_channel is None:
            return None
        else:
            return  self.get(asset_type, channel=channel)

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
            'processes': sanitize_n_processes(self.config['performance']['resampling']['n_processes']),
            'verbose': self.config['verbose']
        }  # WARNING: duplicate (use method ??)
        source_asset = self.get('stitched', channel=channel, default=None)
        if source_asset is None or not source_asset.exists:
            source_asset = self.get('raw', channel)
        if not source_asset.exists:
            raise FileNotFoundError(f'Cannot resample {channel}, source {source_asset} missing')

        if source_asset.is_tiled:
            src_res = define_auto_resolution(source_asset.file_list[0],
                                             self.sample_manager.get_channel_resolution(channel))
        else:
            src_res = self.sample_manager.get_channel_resolution(channel)

        if source_asset.is_tiled:
            if 'Z' in source_asset.tag_names:  # real tiles -> count planes
                n_planes = source_asset.expression.tag_range('Z')[1] + 1
            else:  # columns -> take z column shape
                n_planes = source_geometry.shape(source_asset.file_list[0])[0]
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

    def annotator_of(self, channel: str) -> Annotation:
        """
        The atlas annotator of a channel.

        Raises
        ------
        ClearMapRuntimeError
            If the atlas of this channel was not set up: the channel is not complete, the setup did not
            run yet, or it was skipped because the orientation of the channel is not set.
        """
        annotator = self.annotators.get(channel)
        if annotator is None:
            raise ClearMapRuntimeError(f'No atlas annotator for channel "{channel}": the atlas setup did not run '
                                       f'for it (channel incomplete, or orientation not set). '
                                       f'Channels with an annotator: {[c for c, a in self.annotators.items() if a]}.')
        return annotator

    def get_atlas_files(self):
        if not self.get('atlas', asset_sub_type='annotation',
                        channel=self.sample_manager.alignment_reference_channel).exists:
            self.setup_atlases()
        atlas_files = {}
        for channel in self.channels:
            atlas_files[channel] = self.annotator_of(channel).get_atlas_paths()
        return atlas_files

    @property
    def atlas_cache_dir(self) -> Path:
        """Where reoriented / cropped atlases are stored (machine config 'atlas_cache_folder')."""
        return Path(self.machine_config['atlas_cache_folder']).expanduser()

    @property
    def source_annotator(self) -> Annotation:
        """
        Annotator of the configured atlas, unoriented and uncropped, for orientation-independent
        lookups (structure ids, names, colours).
        Built on first use and cached until the atlas or the structure tree changes in the config.
        """
        atlas_cfg = self.config['atlas']
        atlas_base_name = ATLAS_NAMES_MAP[atlas_cfg['id']]['base_name']
        structure_tree_id = atlas_cfg['structure_tree_id']
        key = (atlas_base_name, structure_tree_id)
        if self._source_annotator_key != key:
            self._source_annotator = Annotation(atlas_base_name, None, None, label_source=structure_tree_id)
            self._source_annotator_key = key
        return self._source_annotator

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
            self.workspace.create_asset(type_spec, channel_spec=channel_spec, sample_id=self.sample_manager.prefix)
            return self.update_atlas_asset(channel_spec.name, annotator=annotator)

    def update_atlas_asset(self, channel, annotator=None):
        if annotator is None:
            annotator = self.annotator_of(channel)
            sample_cfg = self.cfg_coordinator.get_config_view('sample')['channels'][channel]
            orientation = _atlas_orientation(sample_cfg['orientation'])
            xyz_slicing = _atlas_slicing(sample_cfg['slicing'])
            if _atlas_orientation(annotator.orientation) != orientation or annotator.slicing != xyz_slicing:
                if orientation == DEFAULT_ORIENTATION:
                    warnings.warn(f'Orientation not set for {channel}, skipping atlas setup')
                    return
                atlas_cfg = self.config['atlas']
                self.annotators[channel] = Annotation(atlas_base_name=ATLAS_NAMES_MAP[atlas_cfg['id']]['base_name'],
                                                      slicing=xyz_slicing, orientation=orientation,
                                                      label_source=atlas_cfg['structure_tree_id'],
                                                      target_directory=_atlas_target_directory(orientation, xyz_slicing,
                                                                                               self.atlas_cache_dir))
                annotator = self.annotators[channel]
        else:
            if channel not in self.annotators:   # TODO: check if we only update
                self.annotators[channel] = annotator
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

    @property
    def mini_brain(self) -> "MiniBrain":
        """
        The downscaled annotation of the configured atlas, used for the orientation preview.
        Built on first use, once per atlas (see setup_mini_brain), not at setup.
        """
        atlas_base_name = ATLAS_NAMES_MAP[self.config['atlas']['id']]['base_name']
        scaling, array = setup_mini_brain(atlas_base_name)
        return MiniBrain(scaling=scaling, array=array)

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
        from ClearMap.gui.gui_utils_images import surface_project
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
        mini_brain = self.mini_brain
        img = mini_brain['array'].copy()
        x_scale, y_scale, z_scale = mini_brain['scaling']

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

        # TODO: atlas variants as multichannel assets
        for channel in self.channels:
            orientation = _atlas_orientation(sample_cfg[channel]['orientation'])
            xyz_slicing = _atlas_slicing(sample_cfg[channel]['slicing'])

            target_directory = _atlas_target_directory(orientation, xyz_slicing, self.atlas_cache_dir)

            try:
                orientation = validate_orientation(orientation, channel=channel, raise_error=True)
                if orientation == DEFAULT_ORIENTATION:
                    warnings.warn(f'Orientation not set for {channel}, skipping atlas setup')
                    continue
                self.annotators[channel] = Annotation(atlas_base_name, xyz_slicing, orientation,
                                                      label_source=atlas_cfg['structure_tree_id'],
                                                      target_directory=target_directory)

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

        self.update_watcher_main_progress()

    # Plot functions
    def __prepare_registration_results_graph(self, channel):
        img_paths = [self.get_fixed_image(channel).path, self.get_aligned_image(channel)]
        if not all([p.exists() for p in img_paths]):
            raise ValueError(f'Missing requirements {img_paths}')
        titles = [img.parent.stem if 'aligned_to' in str(img) else img.stem for img in img_paths]
        # TODO: replace result<N,1> by channel name
        return img_paths, titles

    def plot_registration_results(self, channel, composite=False, parent=None):
        from ClearMap.Visualization import Plot3d as q_plot_3d
        image_sources, titles = self.__prepare_registration_results_graph(channel)
        if composite:
            image_sources = [image_sources, ]
        dvs = q_plot_3d.plot(image_sources, title=titles, arrange=False, sync=True,
                             lut=self.machine_config['default_lut'], parent=parent)
        return dvs, titles


class MiniBrain(TypedDict):
    """
    A downscaled brain for quick visualization
    It includes the downscaled image and the scaling factors
    """
    scaling: tuple[float, float, float]
    array: np.ndarray


@functools.lru_cache(maxsize=4)
def setup_mini_brain(atlas_base_name, mini_brain_scaling=(5, 5, 5)):  # TODO: scaling in prefs
    """
    Create a downsampled version of the Allen Brain Atlas for the mini brain widget

    Cached: computed once per (atlas, scaling) for the process. The returned array is shared,
    hence read-only (copy it before modifying it).

    Parameters
    ----------
    mini_brain_scaling : tuple(int, int, int)
        The scaling factors for the mini brain. Default is (5, 5, 5)

    Returns
    -------
    tuple(scale, downsampled_array)
    """
    atlas_path = os.path.join(Settings.atlas_folder, f'{atlas_base_name}_annotation.tif')
    arr = TifSource(atlas_path).array
    mini_brain = sk_transform.downscale_local_mean(arr, mini_brain_scaling)
    mini_brain.flags.writeable = False  # Shared by the cache
    return mini_brain_scaling, mini_brain


def define_auto_resolution(img_path, cfg_res):
    if cfg_res == 'auto':
        cfg_res = ('auto', )*3
    out_res = deepcopy(cfg_res)
    if not cfg_res.count('auto'):
        return out_res

    parsed_res = None
    try:
        parsed_res = parse_img_res(img_path)
    except NotAnOmeFile as e:
        print(str(e))
        print('Defaulting to config values')
    except KeyError as e:
        print(f"Could not find resolution for image {img_path}, defaulting to config")

    if parsed_res is None and cfg_res.count('auto'):
        raise MetadataError(f"Could not determine auto config for file {img_path}")

    for i, ax_res in enumerate(cfg_res):
        if ax_res == 'auto':
            out_res[i] = parsed_res[i]

    return out_res
