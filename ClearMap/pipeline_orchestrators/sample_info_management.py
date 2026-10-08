"""
sample_info_management
======================

Sample-level metadata management, configuration synchronisation,
workspace reconciliation, and channel queries.

:class:`SampleManager` is the **root object** for a single ClearMap
experiment.  It owns the sample configuration (channel paths, resolutions,
orientations, data types) and keeps the
:class:`~ClearMap.IO.workspace2.Workspace2` in sync with it.
Every pipeline worker receives a ``SampleManager`` reference so it can
resolve asset paths without knowing the experiment layout.

Responsibilities
----------------

**Configuration access**
    :attr:`~SampleManager.channels`, :attr:`~SampleManager.data_types`,
    :meth:`~SampleManager.get_channel_resolution`, etc. expose the sample
    config as typed Python values, always reflecting the latest committed
    state from :class:`~ClearMap.config.config_coordinator.ConfigCoordinator`.

**Workspace reconciliation**
    :meth:`~SampleManager.update_workspace` ensures the
    :class:`~ClearMap.IO.workspace2.Workspace2` mirrors the current channel
    list — adding missing channels, updating raw-data paths, pruning deleted
    channels, and persisting ``workspace.yml``.

**Channel queries**
    :meth:`~SampleManager.get_channels_by_type`,
    :meth:`~SampleManager.get_channels_by_pipeline`, and
    :meth:`~SampleManager.get_channels_by_condition` provide filtered
    lookups with configurable ``missing_action`` / ``multiple_found_action``
    policies (``'ignore'``, ``'warn'``, ``'raise'``).

**Pipeline discovery**
    :meth:`~SampleManager.compute_relevant_pipelines` infers which
    pipelines are active so the
    :class:`~ClearMap.pipeline_orchestrators.experiment_controller.ExperimentController`
    knows which config sections to load.

**Asset access**
    Inherits :meth:`~.generic_orchestrators.OrchestratorBase.get` and
    :meth:`~.generic_orchestrators.OrchestratorBase.get_path`.

``@adjuster_safe`` marker
-------------------------
Methods decorated with :func:`adjuster_safe` are safe to call from config
adjusters before all channel configs are fully populated.
:func:`check_protocol_coverage` verifies at module load time that every
method in
:class:`~ClearMap.config.config_adjusters.type_hints.SampleManagerProtocol`
carries this marker.

Bootstrapping
-------------
Use :func:`build_sample_manager` rather than constructing
:class:`SampleManager` directly::

    from ClearMap.pipeline_orchestrators.sample_info_management import build_sample_manager

    sm = build_sample_manager('/path/to/experiment')

    print(sm.channels)                              # ['cfos', 'autofluorescence']
    print(sm.get_channels_by_pipeline('CellMap'))   # ['cfos']
    print(sm.get_channel_resolution('cfos'))        # (1.625, 1.625, 3.0)

    raw = sm.get('raw', channel='cfos')
    print(raw.is_tiled, raw.tile_grid_shape)        # True, array([3, 4])

See also
--------
:class:`~ClearMap.IO.workspace2.Workspace2` : Asset management layer.
:class:`~ClearMap.pipeline_orchestrators.experiment_controller.ExperimentController` :
    Owns SampleManager and wires it to pipeline workers.
:mod:`ClearMap.IO.assets_constants` : ``CONTENT_TYPE_TO_PIPELINE`` mapping.
"""
import atexit
import getpass
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List, Dict

import numpy as np

# noinspection PyPep8Naming
import ClearMap.Alignment.Resampling as resampling
# noinspection PyPep8Naming
from ClearMap.IO.workspace2 import Workspace2
from ClearMap.IO.workspace_asset import expression_is_tiled, Asset
from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE

from ..config.compound_keys import PairKey
from ..config.config_adjusters.type_hints import SampleManagerProtocol
from ..config.config_handler import ALTERNATIVES_REG
from ..config.config_coordinator import ConfigCoordinator, make_cfg_coordinator_factory

from ..Utils.events import ChannelRenamed, WorkspaceChannelsUpdated
from ..Utils.tag_expression import Expression
from ..Utils.event_bus import EventBus

from .generic_orchestrators import OrchestratorBase

__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


def adjuster_safe(fn):
    """
    No-op marker: 'this method tolerates incomplete channel configs'.
    This is meant to label SampleManager methods that are used in config adjusters (SampleManagerProtocol)
    and that can be safely called even if some channels have incomplete configs (e.g. missing path).
    """
    fn._adjuster_safe = True
    return fn


def _can_join_workspace(path: Optional[str], data_type: Optional[str]) -> bool:
    # 'undefined' differs from None semantically (intention) but neither can join yet
    return bool(path) and bool(data_type) and data_type != 'undefined'


def channel_can_join_workspace(channel_cfg) -> bool:
    """
    Whether a sample channel config is complete enough for the channel to be added to the workspace.

    .. note::
        Single source of truth for this rule: used by :meth:`SampleManager.update_workspace` and by
        the config-derived facts that must agree with the workspace (e.g. stitchable channels).
    """
    if not isinstance(channel_cfg, dict):
        return False
    return _can_join_workspace(channel_cfg.get('path'), channel_cfg.get('data_type'))


@dataclass(frozen=True, slots=True)
class ChannelWorkspaceInputs:
    """The part of a sample channel config the workspace is derived from."""
    name: str
    is_dict: bool  # malformed entries are reported as incomplete
    path: str
    data_type: Optional[str]

    @property
    def can_join(self) -> bool:
        return self.is_dict and _can_join_workspace(self.path, self.data_type)


@dataclass(frozen=True, slots=True)
class WorkspaceInputs:
    """
    Everything the workspace is derived from (see SampleManager.update_workspace).
    Comparable: equal inputs give the same workspace.
    """
    base_dir: str
    sample_id: Optional[str]  # file prefix, None when use_id_as_prefix is False
    channels: tuple[ChannelWorkspaceInputs, ...]

    @property
    def channel_names(self) -> list[str]:
        return [ch.name for ch in self.channels]


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
        self._published_channel_map: dict[str, str] = {}  # last channel->data_type published to the bus
        self._synced_workspace_inputs: Optional[WorkspaceInputs] = None  # inputs of the last workspace update

        self.subscribe(ChannelRenamed, self._on_channel_renamed)
        # The workspace is derived from the sample config: update it on every commit, before CfgChanged is published
        self.cfg_coordinator.add_post_commit_hook(self.sync_workspace)

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
            src_dir = Path(src_dir).expanduser().resolve()

            # Invalidate workspace if directory changed
            if self.workspace is not None:
                old_dir = Path(self.workspace.directory).resolve()
                if old_dir != src_dir:
                    self.workspace = None
                    self.resource_type_to_folder = None

            self.cfg_coordinator.set_base_dir(src_dir)
            # FIXME: add to set_base_dir ?
            self.cfg_coordinator.load('sample')  # need to load at least sample config to know pipelines
            if not self.config:
                if not src_dir.exists():
                    raise FileNotFoundError(f'Specified source directory {src_dir} does not exist')
                else:
                    return
                    # raise RuntimeError(f'Failed to load sample config from {src_dir}. Unknown error.')

            workspace_path = self.cfg_coordinator.workspace_config_path
            if workspace_path.exists():
                workspace = Workspace2.from_yaml(workspace_path)
                # Ensure loaded workspace points to the right directory
                if str(Path(workspace.directory).resolve()) != str(src_dir):
                    workspace.directory = str(src_dir)
                self.workspace = workspace
                self.resource_type_to_folder = workspace.resource_type_to_folder

            self.update_workspace()

            desired_sample_id = self.prefix  # None when use_id_as_prefix is False
            if desired_sample_id != self.workspace.sample_id:
                self.workspace.set_sample_id(desired_sample_id or '')

            sections = self.compute_required_sections()
            self.cfg_coordinator.load_all(sections)

            self.setup_complete = (not self.incomplete_channels) and bool(self.config)

    @adjuster_safe
    def compute_required_sections(self) -> set[str]:
        """
        Given this controller’s SampleManager (single-sample view),
        compute which config sections should exist for this experiment.
        """
        sections: set[str] = {'sample'}  # always  # TODO: check if ordered

        pipelines = self.compute_relevant_pipelines()

        for p in pipelines:
            sec = ALTERNATIVES_REG.pipeline_to_section_name(p)
            if sec:
                sections.add(sec)

        return sections

    def patch_channel(self, channel, patch: dict):
        self.cfg_coordinator.submit_patch({self.config_name: {'channels': {channel: patch}}},
                                          sample_manager=self)

    def set_channel_expression(self, channel: str, expression: str | Path | Expression) -> None:
        """
        Set the tile path expression of `channel`.

        The expression is validated (a malformed tag raises ValueError) and stored in its
        canonical string form (e.g. ``<X,2>`` becomes ``<X,I,2>``): the config only holds plain
        strings (an Expression has no ``__eq__``, so it would also defeat the change detection
        of the workspace inputs).
        """
        self.patch_channel(channel, patch={'path': str(Expression(expression))})

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

    def _workspace_inputs(self) -> Optional[WorkspaceInputs]:
        """The config values the workspace is derived from, or None if there are no channels yet."""
        cfg = self.config
        if not cfg or 'channels' not in cfg:
            return None
        channels = tuple(
            ChannelWorkspaceInputs(name, True, ch_cfg.get('path', ''), ch_cfg.get('data_type'))
            if isinstance(ch_cfg, dict) else ChannelWorkspaceInputs(name, False, '', None)
            for name, ch_cfg in cfg['channels'].items())
        return WorkspaceInputs(base_dir=str(Path(self.cfg_coordinator.base_dir).resolve()),
                               sample_id=self.prefix, channels=channels)

    def sync_workspace(self) -> None:
        """
        Bring the workspace in line with the sample config, if it changed since the last update.

        Runs after every config commit (ConfigCoordinator post-commit hook), before CfgChanged is
        published. No-op (no rebuild, no save) when the inputs of the workspace did not change.
        """
        inputs = self._workspace_inputs()
        if self.workspace is not None and inputs == self._synced_workspace_inputs:
            return
        self._update_workspace_from(inputs)

    def update_workspace(self):
        """Rebuild the config-derived part of the workspace unconditionally (see sync_workspace)."""
        self._update_workspace_from(self._workspace_inputs())

    def _update_workspace_from(self, inputs: Optional[WorkspaceInputs]) -> None:
        """
        .. warning::
            Must read the config only through `inputs`: sync_workspace skips the update when
            they are unchanged, so anything else read here would not trigger an update.
        """
        if inputs is None:
            # Nothing to do yet (cannot even create workspace) — config not loaded or new experiment
            self.incomplete_channels = []
            return

        self._ensure_workspace(inputs)

        desired_sample_id = inputs.sample_id  # should be self.prefix so None when use_id_as_prefix is False
        current_sample_id = self.workspace.sample_id
        if desired_sample_id != current_sample_id:
            self.workspace.set_sample_id(desired_sample_id or '')

        self.incomplete_channels = []
        # Add or update channels in the workspace
        for ch in inputs.channels:
            if not ch.is_dict or not ch.path or not ch.data_type:
                self.incomplete_channels.append(ch.name)
            elif ch.name in self.workspace:  # exists -> update
                self.workspace.update_raw_path(ch.name, expression=ch.path)
                if ch.data_type in CONTENT_TYPE_TO_PIPELINE:  # WARNING: no 'compound' here
                    channel_spec = self.workspace[ch.name].channel_spec
                    self.workspace.update_pipeline_assets(channel_spec, ch.data_type, sample_id=desired_sample_id)
            elif not ch.can_join:  # new channel not ready yet
                self.incomplete_channels.append(ch.name)
            else:  # new channel -> add
                self.workspace.add_raw_data(file_path=ch.path, channel_id=ch.name,
                                            data_content_type=ch.data_type, sample_id=desired_sample_id)

        # Prune channels that are not in the config anymore
        names = inputs.channel_names
        self.workspace.prune_missing_channels(names)

        print(self.workspace.info())

        self.save_workspace()
        self._synced_workspace_inputs = inputs  # before publishing: subscribers may read the workspace
        self._publish_channels_updated()

    def _workspace_channel_map(self) -> dict[str, str]:
        """channel -> data_type for the config channels that are registered in the workspace"""
        if self.workspace is None:
            return {}
        return {ch: self.data_type(ch) for ch in self.channels if ch in self.workspace}

    def _publish_channels_updated(self) -> None:
        """Publish WorkspaceChannelsUpdated if the workspace channel set/types changed since last publication."""
        after = self._workspace_channel_map()
        before = self._published_channel_map
        if after == before:
            return
        self._published_channel_map = after
        try:
            self.publish(WorkspaceChannelsUpdated(before=dict(before), after=dict(after)))
        except Exception as err:  # A failing subscriber must not break the config -> workspace reconciliation
            warnings.warn(f'A subscriber of WorkspaceChannelsUpdated failed: {err!r}')

    def save_workspace(self):
        workspace_cfg_path = self.cfg_coordinator.workspace_config_path
        if workspace_cfg_path.suffix in {'.yml', '.yaml'}:
            self.workspace.to_yaml(workspace_cfg_path)
        else:
            # legacy fallback
            self.workspace.save(workspace_cfg_path)

    def _ensure_workspace(self, inputs: Optional[WorkspaceInputs] = None):
        current_base = inputs.base_dir if inputs else str(Path(self.cfg_coordinator.base_dir).resolve())

        # Stale workspace pointing to a different directory
        if self.workspace is not None:
            ws_dir = str(Path(self.workspace.directory).resolve())
            if ws_dir != current_base:
                self.workspace = None

        if self.workspace is None:
            sample_id = inputs.sample_id if inputs else self.prefix
            workspace_cfg_path = self.cfg_coordinator.workspace_config_path
            if workspace_cfg_path.exists():
                if workspace_cfg_path.suffix in {'.yml', '.yaml'}:
                    self.workspace = Workspace2.from_yaml(workspace_cfg_path)
                else:  # legacy fallback
                    self.workspace = Workspace2.load(workspace_cfg_path)
                # Patch directory if YAML contains a stale path
                if str(Path(self.workspace.directory).resolve()) != current_base:
                    self.workspace.directory = current_base  # set and propagate to assets
                    self.save_workspace()  # persist so next load is correct
            else:
                self.workspace = Workspace2(current_base,
                                            sample_id=sample_id,
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
    @adjuster_safe
    def channels(self) -> list[str]:
        cfg = self.config
        if not cfg or 'channels' not in cfg:
            return []
        return list(cfg['channels'].keys())

    @property
    def pipeline_ready_channels(self) -> list[str]:
        """Channels with a meaningful data_type (excludes undefined/unconfigured)."""
        excluded = (None, 'undefined', 'no-pipeline')
        return [c for c in self.channels if self.data_type(c) not in excluded]

    @property
    @adjuster_safe
    def renamed_channels(self) -> dict[str, str]:
        return dict(self._renamed_channels)  # copy to discourage mutation

    def set_renamed_channels(self, mapping: dict[str, str]) -> None:
        self._renamed_channels = dict(mapping or {})

    def clear_renamed_channels(self) -> None:
        self._renamed_channels = {}

    def infer_channel_index_from_name(self, path: Path | str) -> int | None:
        """
        Extract channel index from typical microscopy filenames.

        Extracts Cxx from filenames like ``_C00.ome.tif`` -> 0.

        Parameters
        ----------
        path : Path or str
            Microscopy image filename.

        Returns
        -------
        int or None
            Channel index extracted from Cxx pattern, or None if not found.

        Examples
        --------
            >>> infer_channel_index_from_name("image_C03.ome.tif")
            3
        """
        path = Path(path)
        match = re.search(r"[Cc](\d{2})", path.name)
        return int(match.group(1)) if match else None

    @adjuster_safe
    def data_type(self, channel: str) -> str:            # WARNING: ConfigObj Section does not support get() method
        return self.config.get('channels', {}).get(channel, {}).get('data_type', 'undefined')

    @property
    def data_types(self) -> list[str]:
        return [self.data_type(ch) for ch in self.channels]

    @property
    @adjuster_safe
    def channels_to_detect(self) -> list[str]:
        return self.get_channels_by_pipeline('CellMap', as_list=True)

    @property
    @adjuster_safe
    def is_colocalization_compatible(self) -> bool:
        return len(self.channels_to_detect) > 1

    @adjuster_safe
    def colocalization_pairs(self) -> list[tuple[str, str]]:
        if not self.is_colocalization_compatible:
            return []
        src = self.channels_to_detect
        seen = set()
        deduped: list[str] = []
        for ch in src:
            if ch not in seen:
                deduped.append(ch)
                seen.add(ch)

        out: list[tuple[str, str]] = []
        for i in range(len(deduped)):
            for j in range(i + 1, len(deduped)):
                out.append((deduped[i], deduped[j]))  # preserve order from channels_to_detect
        return out

    @adjuster_safe
    def colocalization_pair_keys(self, *, oriented: bool) -> list[str]:
        return [str(PairKey(a, b, oriented=oriented)) for a, b in self.colocalization_pairs()]

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
    @adjuster_safe
    def stitchable_channels(self) -> list[str]:
        return self.get_stitchable_channels()

    @adjuster_safe
    def get_stitchable_channels(self) -> list[str]:
        """
        Channels that can join the workspace and whose raw path is a tile pattern (X and/or Y tag).

        .. warning::
            Derived from the sample config only, **not** from the workspace: the adjusters
            reconcile a config edit before the workspace is updated (after the commit), so a
            workspace-derived answer would lag one edit behind.

        Returns
        -------
        list[str]
            The stitchable channel names, in config order.
        """
        stitchable = []
        for channel, cfg in (self.config.get('channels') or {}).items():
            if not channel_can_join_workspace(cfg):
                continue
            try:
                if expression_is_tiled(cfg['path']):
                    stitchable.append(channel)
            except ValueError:  # Malformed tag (e.g. path being typed): not stitchable yet
                continue
        return stitchable

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
    @adjuster_safe
    def alignment_reference_channel(self) -> Optional[str]:
        try:
            return self.get_channels_by_type('autofluorescence') or None
        except KeyError:
            return None

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
        status = registration_processor.registration_status()
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
        channels_cfg = self.config.get('channels') or {}
        filtered = []
        for chan, cfg in channels_cfg.items():
            if not cfg:
                continue  # skip channels with no config (e.g. from incomplete channel list)
            try:
                if condition(cfg):
                    filtered.append(chan)
            except KeyError:
                continue  # incomplete channel config — skip silently
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

    @adjuster_safe
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
            condition=lambda cfg: cfg.get('data_type') == channel_type,
            missing_action=missing_action,
            multiple_found_action=multiple_found_action,
            as_list=as_list,
            error_label=channel_type
        )

    @adjuster_safe
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
        if pipeline_name == 'stitching':
            chs = self.stitchable_channels
            return chs if as_list or len(chs) != 1 else chs[0]
        if pipeline_name not in CONTENT_TYPE_TO_PIPELINE.values():
            raise ValueError(f'Unknown pipeline name {pipeline_name}. '
                             f'Options are: {list(CONTENT_TYPE_TO_PIPELINE.values())}')
        return self.get_channels_by_condition(
            condition=lambda cfg: CONTENT_TYPE_TO_PIPELINE.get(cfg.get('data_type')) == pipeline_name,
            missing_action=missing_action, multiple_found_action=multiple_found_action,
            as_list=as_list, error_label=pipeline_name
        )

    # TODO: check if we really need instance_kind here
    def get_instance_keys_by_pipeline(self, pipeline_name: str, *, instance_kind: str,
                                      oriented: bool = False) -> list[str]:
        if pipeline_name.lower() == 'colocalization' and instance_kind == 'pairs':
            return self.colocalization_pair_keys(oriented=oriented)
        raise ValueError(f'Unsupported pipeline/instance kind combination: {pipeline_name}/{instance_kind}')

    def compute_relevant_pipelines(self) -> set[str]:
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
                              sample_id: Optional[str] = None) -> List[Asset]:
        return [self.workspace.get(asset_name, channel=channel, sample_id=sample_id) for asset_name in asset_names]

    @staticmethod
    def compress(assets: List[Asset], format: Optional[str] = None):
        for asset in assets:
            asset.compress(algorithm=format)

    @staticmethod
    def decompress(assets: List[Asset], check: bool = True):
        for asset in assets:
            asset.decompress(check=check)

    @staticmethod
    def plot(assets: List[Asset], **kwargs):  # FIXME: what if len(assets) > 1 ? Should plot together
        for asset in assets:
            asset.plot(**kwargs)

    @staticmethod
    def convert(assets: List[Asset], new_extension: str, processes: Optional[int] = None,
                verbose: bool = False, **kwargs):
        for asset in assets:
            asset.convert(new_extension, processes=processes, verbose=verbose, **kwargs)

    @staticmethod
    def resample(assets: List[Asset], x_scale: float = 1, y_scale: float = 1, z_scale: float =1,
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



def _make_bootstrap_dir() -> Path:
    """
    Create a temporary directory for this session to hold config files
    until the experiment folder is set by the user.
    We prefer a user-configured temp if available (CLEARMAP_TMP env var)
    otherwise use the system temp folder.
    The folder is named "clearmap_bootstrap-<username>/session_<pid>" to avoid clashes
    if multiple instances are running.
    The folder is removed on exit if it stayed unused (i.e. still a bootstrap dir).
    Returns
    -------
    Path
        The path to the temporary bootstrap directory
    """
    root = Path(os.environ.get('CLEARMAP_TMP', tempfile.gettempdir()))
    session = root / f'clearmap_bootstrap-{getpass.getuser()}' / f'session_{os.getpid()}'
    session.mkdir(parents=True, exist_ok=True)
    return session


def build_sample_manager(src_dir='', bus: Optional[EventBus] = None):
    bootstrap_dir = _make_bootstrap_dir()

    @atexit.register
    def _cleanup_bootstrap():
        try:
            if bootstrap_dir.name.startswith('session_'):
                shutil.rmtree(bootstrap_dir, ignore_errors=True)
        except Exception:
            pass

    if bus is None:
        bus = EventBus()

    cfg_coordinator_factory = make_cfg_coordinator_factory(bus)
    cfg_coordinator = cfg_coordinator_factory(bootstrap_dir,
                                              config_groups=(ALTERNATIVES_REG._pipeline_groups,
                                                             ALTERNATIVES_REG._global_groups))

    cfg_coordinator.seed_missing_from_defaults(tabs_only=True)

    sample_manager = SampleManager(config_coordinator=cfg_coordinator, src_dir=None)

    if src_dir:
        sample_manager.setup(src_dir)
    return sample_manager


def check_protocol_coverage(cls, protocol_cls):
    """
    Check if all the decorated methods of SampleManager (meant to be used in SampleManagerProtocol)
    are present and decorated in the given class.

    Raises TypeError if any protocol member lacks the marker.
    """
    # Python 3.12+ has __protocol_attrs__; fall back to __annotations__
    names = getattr(protocol_cls, '__protocol_attrs__', None)
    if names is None:
        names = {n for n in dir(protocol_cls) if not n.startswith('_') and n not in dir(object)}

    missing = []
    for name in names:
        raw = getattr(cls, name, None)
        if raw is None:
            missing.append(f'{name} (not found)')
            continue
        fn = raw.fget if isinstance(raw, property) else raw
        if not getattr(fn, '_adjuster_safe', False):
            missing.append(name)

    if missing:
        raise TypeError(f'{cls.__name__} protocol members missing @adjuster_safe: {missing}')


check_protocol_coverage(SampleManager, SampleManagerProtocol)
