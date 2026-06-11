"""
This module contains the generic processor classes that are used to define the processing steps and run the processing
This is inherited by all pipeline_orchestrators in ClearMap
"""
import sys
import warnings
from abc import ABC, abstractmethod
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Any, Optional, List, Final, Callable, TYPE_CHECKING, Sequence

from ClearMap.IO.workspace2 import Workspace2
from ClearMap.IO.workspace_asset import Asset
from ClearMap.Utils.event_bus import BusSubscriberMixin
from ClearMap.Utils.exceptions import ClearMapRuntimeError
from ClearMap.Utils.utilities import handle_deprecated_args, deep_freeze, infer_origin_from_caller

if TYPE_CHECKING:    # WARNING: some circular imports below, use only for type checking with quotes
    from ClearMap.config.config_coordinator import ConfigCoordinator
    from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager
    from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor
    from ClearMap.pipeline_orchestrators.experiment_controller import AnalysisGroupController
    from ClearMap.gui.widgets import ProgressWatcher


class ProcessorSteps(ABC):
    """
    Ordered sequence of processing steps with disk-level asset tracking.

    Step order is resolved fresh on every access via an injected provider
    callable, so GUI-driven reordering takes effect immediately without
    processor reconstruction.
    """
    _default_steps: tuple[str, ...] = ()

    def __init__(self, workspace: Workspace2, channel: str | Sequence[str] = '',
                 step_order_provider: Callable[[], tuple[str, ...]] | None = None):
        """
        Parameters
        ----------
        workspace
        channel
        step_order_provider: Callable
        """
        self.channel: str | Sequence[str] = channel
        self.workspace: Workspace2 = workspace
        self._step_order_provider = step_order_provider

    @property
    def steps(self) -> tuple[str, ...]:
        if self._step_order_provider is not None:
            return self._step_order_provider()
        return self._default_steps

    @abstractmethod
    def asset_from_step_name(self, step_name: str) -> Asset:
        raise NotImplementedError

    @property
    def existing_steps(self) -> list[str]:
        return [s for s in self.steps if self.step_exists(s)]

    @property
    def last_step(self) -> Optional[str]:
        existing = self.existing_steps
        return existing[-1] if existing else None

    def step_exists(self, step_name: str) -> bool:
        return self.asset_from_step_name(step_name).exists

    def get_next_steps(self, step_name: str) -> list[str]:
        idx = self.steps.index(step_name)
        return list(self.steps[idx + 1:])

    def remove_next_steps_files(self, target_step_name: str) -> None:
        for step_name in self.get_next_steps(target_step_name):
            asset = self.asset_from_step_name(step_name)
            if asset.exists:
                warnings.warn(f'WARNING: Removing downstream step {step_name!r}: {asset.path}')
                asset.path.unlink(missing_ok=True)

    def get_asset(self, step: str, *, step_back: bool = False, n_before: int = 0) -> Asset:
        """
        Return the Asset for ``step``, optionally walking back if absent and ``step_back`` is True.

        Parameters
        ----------
        step : str
            Target step name.
        step_back : bool
            If True and the asset does not exist, walk backwards through
            all preceding steps until one is found.
        n_before : int
            If > 0, resolve to the step ``n_before`` positions earlier
            before checking existence.

        Returns
        -------
        Asset
            The asset corresponding to the step name.
        """
        idx = self.steps.index(step)
        if n_before:
            idx = max(0, idx - n_before)
            step = self.steps[idx]

        asset = self.asset_from_step_name(step)
        if asset.exists:
            return asset

        if not step_back:
            raise IndexError(f'Asset for step {step!r} not found: {asset.path}')

        # TODO: return how many before and step name
        for prev in reversed(self.steps[:idx]):
            candidate = self.asset_from_step_name(prev)
            if candidate.exists:
                return candidate

        raise IndexError(f'No existing asset found at or before step {step!r}')


class OrchestratorBase(BusSubscriberMixin):
    config_name = ''

    def __init__(self, coordinator: "ConfigCoordinator"):  # REFACTOR: pass event_bus explicitly (don't steal from coordinator)
        super().__init__(coordinator._bus)
        self.cfg_coordinator: "ConfigCoordinator" = coordinator
        self.workspace: Optional[Workspace2] = None
        self.logger = None  # optional injected logger
        self.registration_processor: Optional["RegistrationProcessor"] = None
        self.setup_complete: bool = False

    def get_alignment_ref_channel_reg_cfg(self) -> Mapping[str, Any]:
        if not getattr(self, 'registration_processor'):
            raise ValueError(f'{self.__class__.__name__} cannot call '
                             f'get_alignment_ref_channel_reg_cfg() without a registration_processor attribute')
        else:
            return self.registration_processor.ref_channel_cfg

    @handle_deprecated_args(
        {'postfix': 'asset_sub_type',
         'prefix': 'sample_id'}
    )
    def get(self, asset_type, channel='current', asset_sub_type=None, **kwargs):   # channel and asset_sub_type defined for completion
        if self.workspace is None:
            raise ClearMapRuntimeError(f'Cannot call {self.__class__.__name__}.get without a workspace. '
                                       f'Please ensure it is assigned by calling {self.__class__.__name__}.setup() '
                                       f'with a valid sample manager first.')
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

    def reload_config(self):
        """Deprecated no-op: configs are managed in-memory by the coordinator."""
        warnings.warn(f"{self.__class__.__name__}.reload_config() has been removed; "
                      "config now remains in sync via ConfigCoordinator.",
                      category=DeprecationWarning)

    @property
    def registration_config(self):
        return self.cfg_coordinator.get_config_view('registration')

    @property
    def config(self) -> Mapping[str, Mapping[str, Any]]:
        """
        Return an always fresh, read-only snapshot,
        to discourage mutation outside a session
        """
        return MappingProxyType(self.cfg_coordinator.get_config_view(self.config_name))

    @property
    def machine_config(self):
        return self.cfg_coordinator.get_config_view('machine')

    @property
    def verbose(self):
        return self.machine_config['verbosity'] == 'debug'

    def setup_if_needed(self):
        pass


class PipelineOrchestrator(OrchestratorBase):
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
    cfg_coordinator: ConfigCoordinator
        The configuration coordinator to manage the configuration files.
    stopped: bool
        Flag to indicate if the process has been stopped.
    progress_watcher: ProgressWatcher or None
        The progress watcher to monitor the progress of the processing.
    workspace: Workspace or None
        The workspace to manage the assets.
    """

    def __init__(self, coordinator: "ConfigCoordinator"):
        super().__init__(coordinator)
        self.stopped: bool = False
        self.progress_watcher: Optional["ProgressWatcher"] = None  # FIXME: ensure assigned
        self.sample_manager: Optional["SampleManager"] = None  # FIXME: ensure assigned

    def setup_if_needed(self):
        """

          .. warnings::

                This assumes self.channel is set if needed!
                and self.registration_processor is set if needed!
        Returns
        -------

        """
        if not self.setup_complete and self.sample_manager is not None:
            self.setup()

    def setup(self, sample_manager: Optional["SampleManager"] = None):
        self.sample_manager = sample_manager if sample_manager else self.sample_manager
        if not self.cfg_coordinator.get_config_view(self.config_name):
            raise ValueError(f'Config section "{self.config_name}" not found in config coordinator')
        if self.sample_manager is not None and sample_manager.setup_complete:
            self.workspace = self.sample_manager.workspace
            self.setup_complete = True
        else:
            self.setup_complete = False
            warnings.warn(f'Sample manager not setup yet. Cannot setup {self.__class__.__name__}.')

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
        pattern: str or re.Pattern or (str, re.Pattern) or None
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

    def run(self):
        raise NotImplementedError

    # def setup(self):
    #     pass

class _CurrentChannelSentinel:
    __slots__ = ()
CURRENT_CHANNEL: Final = _CurrentChannelSentinel()


class ChannelPipelineOrchestrator(PipelineOrchestrator):
    """
    Tab processor that is processing a single channel.

    The config is expected to have a 'channels' section with the channel name as key.
    When accessed, the config property returns the section for the current channel.
    """
    config_name = ''
    def __init__(self, coordinator: "ConfigCoordinator"):
        super().__init__(coordinator)
        self.channel: str = ''

    def get(self, asset_type, channel=CURRENT_CHANNEL, asset_sub_type=None, **kwargs):
        if channel is CURRENT_CHANNEL:
            channel = self.channel
        return super().get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs)

    @property
    def config(self) -> Mapping[str, Any]:
        """
        Read-only view of the config section for the current channel.
        Raises if no channel is set or section missing.
        """
        return self.channel_cfg_view(self.config_name)

    def channel_cfg_view(self, cfg_name):
        """
        Read-only view of the config section for the current channel.
        Raises if no channel is set or section missing.
        """
        if not self.channel:
            raise ValueError(f'{self.__class__.__name__}.channel is not set')
        cfg = self.cfg_coordinator.get_config_view(cfg_name) or {}
        section = cfg.get('channels', {}).get(self.channel)
        if section is None:
            raise KeyError(f'Channel "{self.channel}" not found in config')
        return deep_freeze(section)

    def patch_channel(self, patch: dict, *, origin: str = "") -> None:
        self.cfg_coordinator.submit_patch(
            {self.config_name: {"channels": {self.channel: patch}}},
            sample_manager=self.sample_manager,
            origin=origin or infer_origin_from_caller())


class CompoundChannelPipelineOrchestrator(PipelineOrchestrator):
    """
    Tab processor that is processing a compound channels (made of several source channels)

    The config is expected to have a 'channels' section with the channel names as keys.
    When accessed, the config property returns the section for all channels.
    """
    config_name = ''
    def __init__(self, coordinator: "ConfigCoordinator"):
        super().__init__(coordinator)
        self.channels: List[str] = []

    def get(self, asset_type, channel=CURRENT_CHANNEL, asset_sub_type=None, **kwargs):
        if channel is CURRENT_CHANNEL:
            channel = '-'.join(self.channels)
        return super().get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs)

    @property
    def config(self) -> Mapping[str, Any]:
        """
        Read-only view of the config section for the current channel.
        Raises if no channel is set or section missing.
        """
        return self.__channel_cfg_view(self.config_name)

    def __channel_cfg_view(self, cfg_name):
        """
        Read-only view of the config section for the current channel.
        Raises if no channel is set or section missing.
        """
        if not self.channels:
            raise ValueError(f'{self.__class__.__name__}.channels is not set')
        cfg = self.cfg_coordinator.get_config_view(cfg_name) or {}
        channel_str = '-'.join(self.channels).lower()
        section = cfg.get('channels', {}).get(channel_str)
        if section is None:
            raise KeyError(f'Channel "{channel_str}" not found in config')
        return deep_freeze(section)

    def patch_channel(self, patch: dict, *, origin: str = "") -> None:
        channel = '-'.join(self.channels).lower()
        self.cfg_coordinator.submit_patch(
            {self.config_name: {"channels": {channel: patch}}},
            sample_manager=self.sample_manager,
            origin=origin or infer_origin_from_caller())


class GroupOrchestratorBase:
    """
    Base for multi-experiment/group analyses.
    Depends on AnalysisGroupController (not a single ExperimentController).
    Provides progress + threading plumbing and per-sample worker access.
    """

    def __init__(self, *, group_controller: "AnalysisGroupController", groups: dict[str, list[str]] | None = None,
                channel: Optional[str] = None):
        self.group_controller = group_controller
        if groups is not None:
            self.group_controller.set_groups(groups)

        self.channel: Optional[str] = channel

        # injected by UI/tab
        self._progress_watcher = None             # ProgressWatcher-like
        self._wrap_in_thread: Optional[Callable] = None

    def set_progress_watcher(self, watcher) -> None:
        self._progress_watcher = watcher

    def set_thread_wrapper(self, wrapper_callable: Callable) -> None:
        """Typically main_window.wrap_in_thread; falls back to sync if not set."""
        self._wrap_in_thread = wrapper_callable

    @property
    def groups(self) -> dict[str, list[str]]:
        return self.group_controller.groups

    @property
    def results_folder(self) -> Path:
        return self.group_controller.group_base_dir

    def _any_sample_in(self, group_name: str) -> Path:
        g = self.groups.get(group_name, [])
        if not g:
            raise ValueError(f'Group "{group_name}" is empty or undefined')
        return Path(g[0])

    def _increment_progress_main(self, n: int = 1) -> None:
        if self._progress_watcher is not None:
            self._progress_watcher.increment_main_progress(n)

    def _threaded(self, func: Callable, *args, **kwargs):
        if self._wrap_in_thread is None:
            return func(*args, **kwargs)
        return self._wrap_in_thread(func, *args, **kwargs)

    def get_worker_for_sample(self, sample_src_dir: str | Path, pipeline: str, *, channel=None, substep=None):
        return self.group_controller.get_worker(sample_src_dir, pipeline, channel=channel, substep=substep)

    def get_sample_manager_for(self, sample_src_dir: str | Path):
        return self.group_controller.get_sample_manager(sample_src_dir)

    def run(self):
        """Implement in subclasses if you want a one-shot entrypoint."""
        raise NotImplementedError



class CanceledProcessing(BrokenProcessPool):  # TODO: better inheritance
    pass
