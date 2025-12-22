"""
sample_info_management
======================

This is the part that pertains to the metadata and configuration of a sample.
It manages the sample-level configurations and properties, handles configurations related to the sample,
and provides utility methods for checking sample properties.
"""
import re
import warnings
from pathlib import Path
from typing import Optional, Callable, List, Dict, TYPE_CHECKING

import numpy as np

from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE
from ClearMap.Utils.events import ChannelRenamed, CfgChanged
if TYPE_CHECKING:
    from ClearMap.config.config_coordinator import ConfigCoordinator

# noinspection PyPep8Naming
import ClearMap.Alignment.Resampling as resampling
# noinspection PyPep8Naming
from ClearMap.IO.workspace2 import Workspace2

from .generic_orchestrators import OrchestratorBase

__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


class SampleManager(OrchestratorBase):
    """
    This class is used to manage the sample information
    Manage sample-level configurations and properties.
    Handle configurations related to the sample.
    Provide utility methods for checking sample properties.

    """
    config_name = 'sample'
    def __init__(self, config_coordinator: "ConfigCoordinator", src_dir: Optional[Path | str] = None):
        super().__init__(config_coordinator)

        self.incomplete_channels = []
        self.setup_complete = False
        self.workspace: Optional[Workspace2] = None  # Defined in update_workspace

        self._renamed_channels: dict[str, str] = {}

        self.subscribe(ChannelRenamed, self._on_channel_renamed)
        self.subscribe(CfgChanged, self._on_cfg_changed)

        self.resource_type_to_folder: Optional[dict] = None

        self.setup(src_dir=src_dir)

    def setup(self, src_dir: Optional[Path | str] = None):
        """
        Setup the sample manager with the given configs.

        Parameters
        ----------
        src_dir : str | Path | None
            The source directory of the sample
        """
        if src_dir is not None:
            self.cfg_coordinator.set_base_dir(src_dir)
            self.cfg_coordinator.load_all()

            workspace_path = self.cfg_coordinator.workspace_config_path
            if workspace_path.exists():
                workspace = Workspace2.from_yaml(workspace_path)
                self.workspace = workspace
                self.resource_type_to_folder = workspace.resource_type_to_folder

            self.update_workspace()
            self.setup_complete = (not self.incomplete_channels) and bool(self.config)

    def _on_cfg_changed(self, evt: CfgChanged) -> None:
        # Only reconcile when sample/channels subtree changed.
        if not evt.changed_keys:
            return
        channels_changed = any(k.startswith('sample.channels') or k == 'sample' for k in evt.changed_keys)
        if channels_changed:
            self.update_workspace()

    def patch_channel(self, channel, patch: dict):
        self.cfg_coordinator.submit_patch({self.config_name: {'channels': {channel: patch}}},
                                          sample_manager=self)

    def set_channel_expression(self, channel: str, expression: "str | tag_expression.Expression"):
        self.patch_channel(channel, patch={'path': expression})

    def set_channel_resolution(self, channel: str, resolution: tuple[float, float, float]):
        self.patch_channel(channel, patch={'resolution': list(resolution)})

    def _on_channel_renamed(self, event: ChannelRenamed):
        if self.workspace:
            try:
                self.workspace.rename_channel(event.old, event.new)
            except Exception:
                self.update_workspace()  # Full reconciliation if rename fails

    def rename_channels_in_workspace(self, names_map: Dict[str, str]):
        if not self.workspace:
            return
        for old_name, new_name in names_map.items():
            if old_name and old_name != new_name:
                self.workspace.rename_channel(old_name, new_name)

    def update_workspace(self):
        if not self.config or 'channels' not in self.config:
            # Nothing to do yet (cannot even create workspace) — config not loaded or new experiment
            self.incomplete_channels = []
            return

        self._ensure_workspace()

        self.incomplete_channels = []
        # Add or update channels in the workspace
        for channel, cfg in self.config['channels'].items():
            raw_path = cfg['path']
            if not raw_path:
                self.incomplete_channels.append(channel)
            else:
                data_content_type = cfg['data_type']
                if channel in self.workspace:  # exists -> update
                    self.workspace.update_raw_path(channel, expression=raw_path)
                    if data_content_type and data_content_type in CONTENT_TYPE_TO_PIPELINE:  # WARNING: no 'compound' here
                        channel_spec = self.workspace[channel].channel_spec
                        self.workspace.update_pipeline_assets(channel_spec, data_content_type, sample_id=self.prefix)
                else:  # new channel -> add
                    if data_content_type == 'undefined':  # Difference with None is intention
                        self.incomplete_channels.append(channel)
                        continue
                    self.workspace.add_raw_data(file_path=raw_path, channel_id=channel,
                                                data_content_type=data_content_type, sample_id=self.prefix)

        # Prune channels that are not in the config anymore
        self.workspace.prune_missing_channels(self.channels)

        self.workspace.ensure_default_channel(self.channels, self.channels[0] if self.channels else None)

        print(self.workspace.info())

        self.save_workspace()

    def save_workspace(self):
        workspace_cfg_path = self.cfg_coordinator.workspace_config_path
        if workspace_cfg_path.suffix in {'.yml', '.yaml'}:
            self.workspace.to_yaml(workspace_cfg_path)
        else:
            # legacy fallback
            self.workspace.save(workspace_cfg_path)

    def _ensure_workspace(self):
        if self.workspace is None:
            first_channel = self.channels[0] if self.channels else None
            workspace_cfg_path = self.cfg_coordinator.workspace_config_path
            if workspace_cfg_path.exists():
                if workspace_cfg_path.suffix in {'.yml', '.yaml'}:
                    self.workspace = Workspace2.from_yaml(workspace_cfg_path)
                else:  # legacy fallback
                    self.workspace = Workspace2.load(workspace_cfg_path)
            else:
                self.workspace = Workspace2(self.cfg_coordinator.base_dir,
                                            sample_id=self.prefix,
                                            default_channel=first_channel,
                                            resource_type_to_folder=self.resource_type_to_folder)
            self.resource_type_to_folder = self.workspace.resource_type_to_folder

    def set_resource_type_to_folder(self, new_mapping: dict, *,
                                    migrate: bool = False, dry_run: bool = False) -> dict[str, tuple[Path, Path]]:
        """
        Update the workspace's resource_type_to_folder layout.

        - If dry_run=True:
            * compute and return the migration plan,
            * DO NOT move files,
            * DO NOT change workspace or persist anything.

        - If dry_run=False:
            * apply layout to the workspace (optionally migrating),
            * update SampleManager.resource_type_to_folder,
            * persist the workspace.
        """
        self._ensure_workspace()

        plan = self.workspace.sync_resource_type_to_folder(new_mapping, migrate=migrate, dry_run=dry_run)

        if not dry_run:
            # Keep SM in sync with Workspace2
            self.resource_type_to_folder = self.workspace.resource_type_to_folder

            # Persist new layout + types/channels snapshot
            self.save_workspace()

        return plan

    @property
    def prefix(self) -> Optional[str]:
        """
        Get the prefix to use for the files

        Returns
        -------
        str
            The prefix to use, None to not use any
        """
        return self.config['sample_id'] if self.config['use_id_as_prefix'] else None

    @property
    def channels(self) -> list[str]:
        cfg = self.config
        if not cfg or 'channels' not in cfg:
            return []
        return list(cfg['channels'].keys())

    @property
    def renamed_channels(self) -> dict[str, str]:
        return dict(self._renamed_channels)  # copy to discourage mutation

    def set_renamed_channels(self, mapping: dict[str, str]) -> None:
        self._renamed_channels = dict(mapping or {})

    def clear_renamed_channels(self) -> None:
        self._renamed_channels = {}

    def infer_channel_index_from_name(self, path: Path | str) -> int | None:
        """
        Extracts Cxx from typical microscopy filenames, e.g. *_C00.ome.tif -> 0
        Returns None if no Cxx is found.
        """
        path = Path(path)
        match = re.search(r"[Cc](\d{2})", path.name)
        return int(match.group(1)) if match else None

    def data_type(self, channel: str) -> str:
        return self.config['channels'][channel]['data_type']

    @property
    def data_types(self) -> list[str]:
        return [channel_cfg['data_type'] for channel_cfg in self.config['channels'].values()]

    @property
    def channels_to_detect(self) -> list[str]:
        return self.get_channels_by_pipeline('CellMap', as_list=True)

    @property
    def is_colocalization_compatible(self) -> bool:
        return len(self.channels_to_detect) > 1

    @property
    def relevant_pipelines(self) -> list[str]:
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

    def z_only(self, channel) -> bool:
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
        return self.get('raw', channel, sample_id=self.prefix).tag_names == ['Z']

    def is_tiled(self, channel) -> bool:
        asset = self.get('raw', channel, sample_id=self.prefix)
        return asset.is_tiled and not self.z_only(channel)

    @property
    def autofluorescence_is_tiled(self) -> bool:
        """
        Check if the autofluorescence channel is tiled (has x and y tiles)
        Returns
        -------
        bool
            True if the autofluorescence channel is tiled
        """
        return self.is_tiled(self.alignment_reference_channel)

    def has_tiles(self, channel: Optional[str] = None) -> bool:
        # extension = '.npy' if self.use_npy() else None
        # return len(clearmap_io.file_list(self.filename(channel, sample_id=self.prefix, extension=extension)))
        # noinspection PyTypeChecker
        if channel is None:
            return bool(self.stitchable_channels)
        return self.get('raw', channel=channel, sample_id=self.prefix).n_tiles_present > 1

    def check_has_all_tiles(self, channel: str) -> bool:
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
    def stitchable_channels(self) -> list[str]:
        return self.get_stitchable_channels()

    def get_stitchable_channels(self) -> list[str]:
        candidates = self.config['channels'].keys()
        try:
            assets = [self.get('raw', channel=c, sample_id=self.prefix) for c in candidates]
        except KeyError:  #  raw data not yet set up
            warnings.warn(f'Trying to get stitchable channels before raw data is set up')
            return []
        stitchable_channels = [c for c, asset in zip(candidates, assets) if asset.is_tiled]
        return stitchable_channels

    def can_convert(self, channel: str) -> bool:
        asset = self.get('raw', channel=channel, sample_id=self.prefix)
        return asset.is_regular_file and not asset.variant(extension='.npy').exists

    @property
    def channels_to_convert(self) -> list[str]:
        candidates = self.config['channels'].keys()
        return [c for c in candidates if self.can_convert(c)]

    def has_npy(self, channel: Optional[str] = None) -> bool:
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

    def use_npy(self, channel: str) -> bool:
        asset = self.get('raw', channel=channel, sample_id=self.prefix)
        cfg = self.cfg_coordinator.get_config_view('stitching')['channels'][channel]
        return cfg['use_npy'] and str(asset.expression).endswith('.npy') or asset.variant(extension='.npy').exists

    @property
    def alignment_reference_channel(self) -> Optional[str]:
        return self.get_channels_by_type('autofluorescence') or None

    def delete_resampled_files(self, channel: str):
        asset = self.get('resampled', channel=channel)
        if asset.exists:
            asset.delete()

    def get_channel_resolution(self, channel: str) -> tuple[float, float, float]:
        """
        Get the resolution of the channel as defined in the sample config.

        Parameters
        ----------
        channel : str
            The channel to get the resolution for

        Returns
        -------
        tuple(float, float, float)
            The resolution of the channel in (x, y, z) format
        """
        return tuple(self.config['channels'][channel]['resolution'])

    def stitched_shape(self, channel: str) -> tuple[int, int, int]:
        asset = self.get('stitched', channel=channel, sample_id=self.prefix)
        if asset.exists:
            return asset.shape()
        elif self.resampled_shape(channel) is not None:
            reg_cfg = self.registration_config
            raw_resampled_res_from_cfg = np.array(reg_cfg['channels'][channel]['resampled_resolution'])
            raw_res_from_cfg = np.array(self.config['channels'][channel]['resolution'])
            return self.resampled_shape(channel) * (raw_resampled_res_from_cfg / raw_res_from_cfg)
        else:
            raise FileNotFoundError(f'Could not get stitched shape without '
                                    f'stitched or resampled file for channel {channel}')

    def resampled_shape(self, channel: str) -> Optional[tuple[int, int, int]]:
        asset = self.workspace.get('resampled', channel=channel, sample_id=self.prefix)
        if asset.exists:
            return asset.shape()

    def needs_registering(self, registration_processor: "RegistrationProcessor") -> bool:
        status = registration_processor.get_registration_status()
        from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationStatus
        return status == RegistrationStatus.MISSING_OUTPUTS

    def get_channels_by_condition(self, condition: Callable, missing_action: str = 'ignore',
                                  multiple_found_action: str ='ignore',
                                  as_list: bool = False, error_label: str = 'channel') -> str | list[str]:
        """
        Get the channel or list of channels that satisfy a given condition.
        The condition is specified as a function that takes a channel config and returns a boolean.
        e.g. to get the channel of type 'autofluorescence':
        get_channels_by_condition(lambda cfg: cfg['data_type'] == 'autofluorescence')

        Parameters
        ----------
        condition: function
            A function that takes a channel config and returns a boolean.
        missing_action: str
            What to do in case no matching channel is found. One of ['warn', 'raise', 'ignore']
        multiple_found_action: str
            What to do in case multiple matching channels are found. One of ['warn', 'raise', 'ignore']
        as_list: bool
            Whether to return the result as a list if a single channel is found.
        error_label: str
            The label to use in error messages.

        Returns
        -------
        str | List[str]
            The channel name or list of channels that match the condition.

        Raises
        ------
        KeyError
            If no channel is found and missing_action is 'raise'
            If multiple channels are found and multiple_found_action is 'raise'
        ValueError
            If an unknown action (missing_action or multiple_found_action) is specified
        """
        filtered = [chan for chan, cfg in self.config['channels'].items() if condition(cfg)]
        count = len(filtered)
        if count == 0:
            match missing_action.lower():
                case 'ignore':
                    return [] if as_list else ""
                case 'warn':
                    warnings.warn(f'No {error_label} found')
                    return [] if as_list else ""
                case 'raise':
                    raise KeyError(f'No {error_label} found')
                case _:
                    raise ValueError(f'Unknown missing action {missing_action}')
        elif count > 1:
            match multiple_found_action.lower():
                case 'ignore':
                    return filtered
                case 'warn':
                    warnings.warn(f'Multiple {error_label}s found')
                    return filtered
                case 'raise':
                    raise KeyError(f'Multiple {error_label}s found')
                case _:
                    raise ValueError(f'Unknown multiple found action {multiple_found_action}')
        else:  # count == 1
            result = filtered if as_list else filtered[0]
        return result

    def get_channels_by_type(self, channel_type: str, missing_action: str = 'warn',
                             multiple_found_action: str ='ignore', as_list: bool = False) -> str | list[str]:
        """
        Get the channel or list of channels that are of a given type.

        Parameters
        ----------
        channel_type: str
            Type of the channel as defined in asset_constants
        missing_action: str
            What to do in case the channel specified is not found. One of ['warn', 'raise', 'ignore']
        multiple_found_action: str
            What to do in case multiple matching channels are found. One of ['warn', 'raise', 'ignore']
        as_list: bool
            Whether to return the result as a list if a single channel is found.

        Returns
        -------
        str | List[str]
            The channel name or list of channels that match the type.

        Raises
        ------
        KeyError
            If no channel is found and missing_action is 'raise'
            If multiple channels are found and multiple_found_action is 'raise'
        ValueError
            If an unknown action (missing_action or multiple_found_action) is specified
        """
        return self.get_channels_by_condition(
            condition=lambda cfg: cfg['data_type'] == channel_type,
            missing_action=missing_action,
            multiple_found_action=multiple_found_action,
            as_list=as_list,
            error_label=channel_type
        )

    def get_channels_by_pipeline(self, pipeline_name: str, missing_action: str = 'ignore',
                                 multiple_found_action: str ='ignore', as_list: bool = False) -> str | list[str]:
        """
        Get the channels that are relevant for a given pipeline

        Parameters
        ----------
        pipeline_name : str
            The name of the pipeline
        missing_action : str
            What to do if no channel is found
            'ignore' : ignore and return empty list
            'warn' : warn and return empty list
            'raise' : raise an error
        multiple_found_action : str
            What to do if multiple channels are found
            'ignore' : ignore and return all channels
            'warn' : warn and return all channels
            'raise' : raise an error
        as_list: bool
            Whether to return the result as a list if a single channel is found.

        Returns
        -------
        List[str]
            The channels that are relevant for the pipeline

        Raises
        ------
        KeyError
            If no channel is found and missing_action is 'raise'
            If multiple channels are found and multiple_found_action is 'raise'
        ValueError
            If an unknown action (missing_action or multiple_found_action) is specified
        """
        if pipeline_name not in CONTENT_TYPE_TO_PIPELINE.values():
            raise ValueError(f'Unknown pipeline name {pipeline_name}. '
                             f'Options are: {list(CONTENT_TYPE_TO_PIPELINE.values())}')
        return self.get_channels_by_condition(
            condition=lambda cfg: CONTENT_TYPE_TO_PIPELINE[cfg['data_type']] == pipeline_name,
            missing_action=missing_action,
            multiple_found_action=multiple_found_action,
            as_list=as_list,
            error_label=pipeline_name
        )

    def infer_pipelines(self) -> set[str]:
        """
        Infer active *per-sample* pipelines based purely on this sample’s channels / types.
        Does not know/care about group/batch.
        """
        pipelines: set[str] = set()

        # 1) channel content types → pipelines
        for ch in self.channels:
            ct = self.data_type(ch)
            p = CONTENT_TYPE_TO_PIPELINE.get(ct)
            if p:
                pipelines.add(p)

        # 2) stitching: as soon as any tiled/pattern channel exists
        if self.stitchable_channels:
            pipelines.add('stitching')

        # 3) registration: if registration is actually meaningful
        # FIXME: check if we have an atlas or something
        pipelines.add('registration')

        # 4) compound/co-loc
        if self.is_colocalization_compatible:
            pipelines.add('Colocalization')

        return pipelines

    def asset_names_to_assets(self, asset_names: List[str], channel: Optional[str] = None,
                              sample_id: Optional[str] = None) -> List["WorkspaceAsset"]:
        return [self.workspace.get(asset_name) for asset_name in asset_names]

    @staticmethod
    def compress(assets: List["WorkspaceAsset"], format: Optional[str] = None):
        for asset in assets:
            asset.compress(algorithm=format)

    @staticmethod
    def decompress(assets: List["WorkspaceAsset"], check: bool = True):
        for asset in assets:
            asset.decompress(check=check)

    @staticmethod
    def plot(assets: List["WorkspaceAsset"], **kwargs):  # FIXME: what if len(assets) > 1 ? Should plot together
        for asset in assets:
            asset.plot(**kwargs)

    @staticmethod
    def convert(assets: List["WorkspaceAsset"], new_extension: str, processes: Optional[int] = None,
                verbose: bool = False, **kwargs):
        for asset in assets:
            asset.convert(new_extension, processes=processes, verbose=verbose, **kwargs)

    @staticmethod
    def resample(assets: List["WorkspaceAsset"], x_scale: float = 1, y_scale: float = 1, z_scale: float =1,
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
