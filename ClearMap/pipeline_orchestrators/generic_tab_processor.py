"""
This module contains the generic processor classes that are used to define the processing steps and run the processing
This is inherited by all pipeline_orchestrators in ClearMap
"""
from __future__ import annotations

import inspect
import sys
import warnings
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from types import MappingProxyType
from typing import Mapping, Any, Iterator, Dict, Optional, List

from ClearMap.IO.workspace2 import Workspace2
from ClearMap.IO.workspace_asset import Asset
from ClearMap.Utils.utilities import handle_deprecated_args, deep_freeze
from ClearMap.config.config_coordinator import ConfigCoordinator
from ClearMap.gui.widgets import ProgressWatcher


def infer_origin_from_caller() -> str:
    """
    Infer an **origin** string from the caller's module, class and function name.
    This is typically the method that is mutating the config.

    Returns
    -------
    str
        The inferred origin string in the format "module.class.function" or "module.function" if
    """
    frame = inspect.stack()[1]
    mod = frame.frame.f_globals.get('__name__', '')
    func = frame.function
    cls = type(frame.frame.f_locals['self']).__name__ if 'self' in frame.frame.f_locals else ''
    origin = f"{mod}.{cls + '.' if cls else ''}{func}"
    return origin


class ProcessorSteps:
    def __init__(self, workspace, channel='', sub_step=''):
        self.channel: str = channel  # FIXME: see if works for CompoundChannelTabProcessor
        self.sub_step: str = sub_step
        self.workspace: Workspace2 = workspace

    @property
    def steps(self) -> List[str]:
        raise NotImplementedError

    def asset_from_step_name(self, step_name: str) -> Asset:
        raise NotImplementedError

    @property
    def existing_steps(self) -> List[str]:
        return [s for s in self.steps if self.step_exists(s)]

    @property
    def last_step(self) -> Optional[str]:
        return self.existing_steps[-1]

    def get_next_steps(self, step_name: str) -> List[str]:
        return self.steps[self.steps.index(step_name)+1:]

    def step_exists(self, step_name: str) -> bool:
        return self.asset_from_step_name(step_name).exists

    def remove_next_steps_files(self, target_step_name: str) -> None:
        for step_name in self.get_next_steps(target_step_name):
            asset = self.asset_from_step_name(step_name)
            if asset.exists:
                warnings.warn(f"WARNING: Remove previous step {step_name}, file {asset.path}")
                asset.path.unlink(missing_ok=True)

    def get_asset(self, step: str, step_back: bool = False, n_before: int = 0) -> Asset:
        """
        Get the asset corresponding to the step name, optionally picking the nth previous step if `n_before` is set.
        If the asset does not exist, it will try to get the previous step if `step_back` is True.

        Parameters
        ----------
        step: str
            Name of the step to get the asset for.
        step_back: bool
            If True, will try to get the previous step's asset if the current step's asset does not exist.
        n_before: int
            If set, will return the asset of the nth previous step instead of the current step.
            Useful when you want to get the asset source to the current step for example.

        Returns
        -------
        Asset
            The asset corresponding to the step name.
        """
        if n_before:
            step = self.steps[self.steps.index(step) - n_before]
        asset = self.asset_from_step_name(step)
        if not asset.exists:
            if step_back:  # FIXME: steps back only once ??
                asset = self.get_asset(self.steps[self.steps.index(step) - 1])
            else:
                raise IndexError(f'Could not find path "{asset}" and not allowed to step back')
        return asset


class TabProcessor:
    """
    Generic tab processor class.
    This class is inherited by all pipeline_orchestrators in ClearMap

    Base methods:
        - config access and mutation,
        - asset retrieval through workspace
        - progress watcher handling
        - process stopping
        - run (to be implemented in child classes)

    Attributes
    ----------
    coordinator: ConfigCoordinator
        The configuration coordinator to manage the configuration files.
    stopped: bool
        Flag to indicate if the process has been stopped.
    progress_watcher: ProgressWatcher or None
        The progress watcher to monitor the progress of the processing.
    workspace: Workspace or None
        The workspace to manage the assets.
    machine_config: dict
        The machine configuration dictionary.
    """
    processing_name = ''
    def __init__(self, coordinator: ConfigCoordinator):
        self.coordinator: ConfigCoordinator = coordinator
        self.stopped: bool = False
        self.progress_watcher: ProgressWatcher = None
        self.workspace: Workspace2 = None
        self.machine_config: Mapping[str, Any] = {}  # FIXME: remove and use coordinator.machine_config
        self.logger = None  # optional injected logger

    @property
    def config(self) -> Mapping[str, Mapping[str, Any]]:
        """
        Return an always fresh, read-only snapshot,
        to discourage mutation outside a session
        """
        return MappingProxyType(self.coordinator.get_config_view(self.processing_name))

    def apply_patch(self, patch: dict, *, origin: str) -> None:
        """
        Apply a multi-file patch through the coordinator (transactional).
        """
        patch = {"origin": origin, **patch}
        # Reuse your coordinator.apply (which already wraps edit_session)
        self.coordinator.apply(patch)   # FIXME: no validate arg ??

    def edit_session(self, *, origin: str, validate: bool = True):
        """
        Context manager: mutate a working copy; commit on exit.
        Usage:
            with self.edit_session(origin="stitching.something") as cfg:
                cfg['sample']['channels'][ch]['path'] = new_path
        """
        with self.coordinator.edit_session(origin=origin, validate=validate) as cfg:
            yield cfg  # FIXME: self.processing_name ??

    @handle_deprecated_args(
        {'postfix': 'asset_sub_type',
         'prefix': 'sample_id'}
    )
    def get(self, asset_type, channel='current', asset_sub_type=None, **kwargs):   # channel and asset_sub_type defined for completion
        asset = self.workspace.get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs)
        return asset

    def get_path(self, asset_type, channel='current', asset_sub_type=None, **kwargs):   # channel and asset_sub_type defined for completion
        return self.get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs).path

    def filename(self, *args, **kwargs):  # WARNING: deprecated
        """
        A shortcut to get the file path from the workspace

        Parameters
        ----------
        args: list
            Any positional argument accepted by workspace.filename
        kwargs: dict
            Any keyword argument accepted by workspace.filename

        Returns
        -------
        str
            The file path as a string
        """
        warnings.warn("TabProcessor.filename is deprecated, "
                      "use TabProcessor.workspace.get_path instead", DeprecationWarning)
        return self.workspace.filename(*args, **kwargs)

    def set_progress_watcher(self, watcher):
        self.progress_watcher = watcher

    def update_watcher_progress(self, val):
        if self.progress_watcher is not None:
            self.progress_watcher.increment(val)

    def update_watcher_main_progress(self, val=1):
        if self.progress_watcher is not None:
            self.progress_watcher.increment_main_progress(val)

    def set_watcher_step(self, step_name):
        if self.progress_watcher is not None:
            self.progress_watcher.main_step_name = step_name

    def prepare_watcher_for_substep(self, counter_size, pattern, title, increment_main=False):
        """
        Prepare the progress watcher for the coming processing step. The watcher will in turn signal changes to the
        progress bar

        Arguments
        ---------
        counter_size: int
            The progress bar maximum
        pattern: str or re.Pattern or (str, re.Pattern)
            The string to search for in the log to signal an increment of 1
        title: str
            The title of the step for the progress bar
        increment_main: bool
            Whether a new step should be added to the main progress bar
        """
        if self.progress_watcher is not None:
            self.progress_watcher.prepare_for_substep(counter_size, pattern, title)
            if increment_main:
                self.update_watcher_main_progress()

    def stop_process(self):  # REFACTOR: put in parent class ??
        self.stopped = True
        if executor := getattr(self.workspace, 'executor', None):
            if sys.version_info[:2] >= (3, 9):
                print('Canceling process')
                executor.shutdown(cancel_futures=True)  # The new clean version
            else:
                executor.immediate_shutdown()  # Dirty but we have no choice in python < 3.9
            self.workspace.executor = None
            # raise BrokenProcessPool
        elif process := getattr(self.workspace, 'process', None):
            process.terminate()
            # self.workspace.process.wait()
            self.workspace.process = None
            raise CanceledProcessing

    @property
    def verbose(self):
        return self.machine_config['verbosity'] == 'debug'

    def run(self):
        raise NotImplementedError

    # def setup(self):
    #     pass


class ChannelTabProcessor(TabProcessor):
    """
    Tab processor that is processing a single channel.

    The config is expected to have a 'channels' section with the channel name as key.
    When accessed, the config property returns the section for the current channel.
    """
    processing_name = ''
    def __init__(self, coordinator: ConfigCoordinator):
        super().__init__(coordinator)
        self.channel: str = ''

    @property
    def config(self) -> Mapping[str, Any]:
        """
        Read-only view of the config section for the current channel.
        Raises if no channel is set or section missing.
        """
        return self.__channel_cfg_view(self.processing_name)

    def __channel_cfg_view(self, cfg_name):
        """
        Read-only view of the config section for the current channel.
        Raises if no channel is set or section missing.
        """
        if not self.channel:
            raise ValueError(f'{self.__class__.__name__}.channel is not set')
        cfg = self.coordinator.get(cfg_name) or {}
        section = cfg.get('channels', {}).get(self.channel)
        if section is None:
            raise KeyError(f'Channel "{self.channel}" not found in config')
        return deep_freeze(section)

    @contextmanager
    def edit_session(self, *, origin: Optional[str] = None, validate: bool = True) \
            -> Iterator[Dict[str, Any]]:
        """
        Transactionally mutate the config section for the current channel.

        Usage:
            with self.mutate_channel_cfg(origin="cell.threshold") as ch:
                ch["threshold"] = 0.7

            or with self.mutate_channel_cfg() as ch:  # origin inferred from caller
                ch["threshold"] = 0.7

        Parameters
        ----------
        origin: str, optional
            Origin string for the change log; if None, inferred from caller.
        validate: bool
            If True (default), run all validators before committing.

        Raises
        ------
        ValueError
            If no channel is set.
        """
        if not self.channel:
            raise ValueError(f'{self.__class__.__name__}.channel is not set')

        if origin is None:
            origin = infer_origin_from_caller()

        with self.edit_session(origin=origin, validate=validate) as cfg:
            tab_cfg = cfg.setdefault(self.processing_name, {})
            chans = tab_cfg.setdefault('channels', {})
            ch_cfg = chans.setdefault(self.channel, {})
            yield ch_cfg  # allow in-place edits to the channel section

    def reload_config(self):
        """
        Deprecated no-op: configs are managed in-memory by the coordinator.
        Keep for compatibility with old call sites; consider removing once migrated.
        """
        warnings.warn("ChannelTabProcessor.reload_config() has been removed; "
                      "config is always in sync via ConfigCoordinator.",
                      category=DeprecationWarning)


class CompoundChannelTabProcessor(TabProcessor):
    """
    Tab processor that is processing a compound channels (made of several source channels)

    The config is expected to have a 'channels' section with the channel names as keys.
    When accessed, the config property returns the section for all channels.
    """
    processing_name = ''
    def __init__(self, coordinator: ConfigCoordinator):
        super().__init__(coordinator)
        self.channels: List[str] = []

    @property
    def config(self) -> Mapping[str, Any]:
        """
        Read-only view of the config section for the current channel.
        Raises if no channel is set or section missing.
        """
        return self.__channel_cfg_view(self.processing_name)

    def __channel_cfg_view(self, cfg_name):
        """
        Read-only view of the config section for the current channel.
        Raises if no channel is set or section missing.
        """
        if not self.channels:
            raise ValueError(f'{self.__class__.__name__}.channels is not set')
        cfg = self.coordinator.get(cfg_name) or {}
        channel_str = '-'.join(self.channels).lower()
        section = cfg.get('channels', {}).get(channel_str)
        if section is None:
            raise KeyError(f'Channel "{channel_str}" not found in config')
        return deep_freeze(section)

    @contextmanager
    def edit_session(self, *, origin: Optional[str] = None, validate: bool = True) \
            -> Iterator[Dict[str, Any]]:
        """
        Transactionally mutate the config section for all channels.

        Usage:
            with self.mutate_channels_cfg(origin="cell.threshold") as chans:
                chans["channel1"]["threshold"] = 0.7
                chans["channel2"]["threshold"] = 0.8

            or with self.mutate_channels_cfg() as chans:  # origin inferred from caller
                chans["channel1"]["threshold"] = 0.7
                chans["channel2"]["threshold"] = 0.8

        Parameters
        ----------
        origin: str, optional
            Origin string for the change log; if None, inferred from caller.
        validate: bool
            If True (default), run all validators before committing.

        Raises
        ------
        ValueError
            If no channel is set.
        """
        if not self.channels:
            raise ValueError(f'{self.__class__.__name__}.channels is not set')

        if origin is None:
            origin = infer_origin_from_caller()

        with self.edit_session(origin=origin, validate=validate) as cfg:
            tab_cfg = cfg.setdefault(self.processing_name, {})
            chans = tab_cfg.setdefault('channels', {})
            ch_cfg = chans.setdefault('-'.join(self.channels).lower(), {})
            yield ch_cfg  # allow in-place edits to the channel section

    def reload_config(self):
        """
        Deprecated no-op: configs are managed in-memory by the coordinator.
        Keep for compatibility with old call sites; consider removing once migrated.
        """
        warnings.warn("ChannelTabProcessor.reload_config() has been removed; "
                      "config is always in sync via ConfigCoordinator.",
                      category=DeprecationWarning)


class CanceledProcessing(BrokenProcessPool):  # TODO: better inheritance
    pass
