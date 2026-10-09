"""
generic_orchestrators
=====================

Base classes for all ClearMap processing workers.

Architecture
------------
The orchestrator hierarchy has three layers:

**Configuration + asset access**

:class:`OrchestratorBase` owns the
:class:`~ClearMap.config.config_coordinator.ConfigCoordinator` reference and
exposes read-only config views, workspace asset retrieval via
:meth:`~OrchestratorBase.get` / :meth:`~OrchestratorBase.get_path`, and
progress-watcher plumbing.  Everything else in ``pipeline_orchestrators/``
inherits from it.

**Single-experiment pipelines**

:class:`PipelineOrchestrator` adds stop/cancel logic and the
``wrap_in_thread`` interface used by the GUI.

:class:`ChannelPipelineOrchestrator` further narrows the scope to a single
channel: its :attr:`~ChannelPipelineOrchestrator.config` property
automatically returns the section for :attr:`~ChannelPipelineOrchestrator.channel`,
and :meth:`~ChannelPipelineOrchestrator.patch_channel` scopes config writes
to that channel without the caller having to know the full key path.

:class:`IndependentChannelsPipelineOrchestrator` processes several channels, each
on its own (stitching, registration): it is bound to none, so every asset access
names its channel.

:class:`CompoundChannelPipelineOrchestrator` is the equivalent for pipelines
that operate on one compound channel (a tuple of channels), such as TubeMap
(vessels and arteries) or colocalization (a pair).
:class:`PerCompoundConfigMixin` scopes its config to the entry of that compound
channel, for the pipelines with one config entry per compound (colocalization).

**Step tracking**

:class:`ProcessorSteps` is an independent ABC that tracks which pipeline
steps have already produced on-disk outputs.  It is composed into concrete
orchestrators that expose a binarization or graph-construction pipeline
so the GUI can resume from the last completed step without re-running
everything.

**Multi-experiment / group analyses**

:class:`GroupOrchestratorBase` sits outside the single-sample hierarchy.
It holds a reference to an
:class:`~ClearMap.pipeline_orchestrators.experiment_controller.AnalysisGroupController`
and provides per-sample worker and sample-manager access, plus progress and
threading plumbing.

Relationship to the GUI
-----------------------
The GUI never calls processing code directly.  Instead, each
``pipeline_orchestrators/`` tab creates and caches one or more workers
(subclasses of the classes here) via
:class:`~ClearMap.pipeline_orchestrators.experiment_controller.ExperimentController.get_worker`.
Config edits made in the GUI flow through
:class:`~ClearMap.config.config_coordinator.ConfigCoordinator` and are
visible to workers on the next ``self.config`` access without reconstruction.

Typical advanced-user pattern::

    from ClearMap.pipeline_orchestrators.sample_info_management import build_sample_manager
    from ClearMap.pipeline_orchestrators.cell_map import CellDetector

    sm = build_sample_manager('/path/to/experiment')
    detector = CellDetector(sm, cfg_coordinator, channel='cfos')
    detector.run_cell_detection()
    detector.filter_cells()
    detector.atlas_align()
"""
import sys
import warnings
from abc import ABC, abstractmethod
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Any, Optional, Final, Callable, TYPE_CHECKING, Sequence, ClassVar, Iterable

from ClearMap.IO.assets_constants import PIPELINE_NAMES
from ClearMap.IO.workspace2 import Workspace2
from ClearMap.IO.workspace_asset import Asset
from ClearMap.config.compound_keys import CompoundKey, compound_channel_key, compound_is_oriented
from ClearMap.Utils.event_bus import BusSubscriberMixin
from ClearMap.Utils.exceptions import ClearMapRuntimeError, ClearMapValueError, ClearMapConfigError, \
    ClearMapNotImplementedError, MissingChannelError
from ClearMap.Utils.utilities import handle_deprecated_args, deep_freeze, infer_origin_from_caller

if TYPE_CHECKING:    # WARNING: some circular imports below, use only for type checking with quotes
    from ClearMap.config.config_coordinator import ConfigCoordinator
    from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager
    from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor
    from ClearMap.pipeline_orchestrators.experiment_controller import AnalysisGroupController
    from ClearMap.gui.widgets import ProgressWatcher


def check_sample_channels_in_workspace(sample_manager: "SampleManager", channels: Iterable[str], *,
                                       required_by: str) -> None:
    """
    Raise if one of channels (sample channels) is not in the workspace of the sample.

    A sample channel is added to the workspace by the SampleManager, once its config is complete
    (path and data type), never by a processor: a processor only checks it is there.

    Raises
    ------
    MissingChannelError
        Naming the missing channels and the incomplete sample channels.
    """
    missing = [channel for channel in channels if channel not in sample_manager.workspace]
    if missing:
        raise MissingChannelError(f'{required_by} needs the sample channel(s) {missing}, which are not in the '
                                  f'workspace. A sample channel joins the workspace once its config is complete '
                                  f'(path and data type); incomplete sample channels: '
                                  f'{sample_manager.incomplete_channels}.')


def register_compound_channel(sample_manager: "SampleManager", pipeline: str,
                              parts: Sequence[str]) -> tuple[str, ...]:
    """
    Register the compound channel made of parts in the workspace of the sample, for pipeline:
    check its parts (sample channels, see check_sample_channels_in_workspace), and create the compound
    channel, and its pipeline assets, if missing. Idempotent: a later call for the same parts only adds
    the pipeline assets that are still missing.

    Its workspace id is compound_channel_key(parts, oriented=compound_is_oriented(pipeline)): the
    parts in their order if oriented, sorted otherwise. For an unoriented compound, the files named
    in another order of the parts are renamed (Workspace2.adopt_compound_order: before the parts were
    sorted, they were in the order of the sample config).

    The function behind CompoundChannelPipelineOrchestrator.register_in_workspace, for the code
    that reads the results of a pipeline without running its processors (e.g. group analyses).

    Returns
    -------
    tuple[str, ...]
        The workspace id of the compound channel.
    """
    oriented = compound_is_oriented(pipeline)
    compound_channel = compound_channel_key(parts, oriented=oriented)
    check_sample_channels_in_workspace(sample_manager, compound_channel,
                                       required_by=f'The {pipeline} compound channel {compound_channel}')
    workspace = sample_manager.workspace
    first_registration = compound_channel not in workspace
    workspace.ensure_pipeline(pipeline, compound_channel, sample_id=sample_manager.prefix,
                              channel_content_type='compound', create_channel=True)
    if first_registration and not oriented:  # Later calls find the order already adopted: no directory scan
        workspace.adopt_compound_order(compound_channel)
    return compound_channel


class ProcessorSteps(ABC):
    """
    Ordered sequence of processing steps with disk-level asset tracking.

    Step order is resolved fresh on every access via an injected provider
    callable, so GUI-driven reordering takes effect immediately without
    processor reconstruction.

    Parameters
    ----------
    workspace : Workspace2
        The workspace that owns the on-disk assets.
    channel : str or Sequence[str]
        Channel name(s) this step sequence operates on.
    step_order_provider : Callable[[], tuple[str, ...]] or None
        Optional callable that returns the current step order.  When provided
        it is called on every access to :attr:`steps`, allowing the GUI to
        reorder steps without reconstructing the processor.  Falls back to
        :attr:`_default_steps` when ``None``.
    """
    _default_steps: tuple[str, ...] = ()

    def __init__(self, workspace: Workspace2, channel: str | Sequence[str] = '',
                 step_order_provider: Callable[[], tuple[str, ...]] | None = None):
        self.channel: str | Sequence[str] = channel
        self.workspace: Workspace2 = workspace
        self._step_order_provider = step_order_provider

    @property
    def steps(self) -> tuple[str, ...]:
        """
        Ordered step names, resolved fresh on every access.

        Returns
        -------
        tuple of str
            Step names in execution order, either from the injected provider
            or from ``_default_steps``.
        """
        if self._step_order_provider is not None:
            return self._step_order_provider()
        return self._default_steps

    @abstractmethod
    def asset_from_step_name(self, step_name: str) -> Asset:
        """
        Return the on-disk asset that corresponds to *step_name*.

        Parameters
        ----------
        step_name : str
            A member of :attr:`steps`.

        Returns
        -------
        Asset
            The asset associated with that step.

        Raises
        ------
        NotImplementedError
            Must be implemented by every concrete subclass.
        """
        raise NotImplementedError

    @property
    def existing_steps(self) -> list[str]:
        """
        Steps whose output asset already exists on disk.

        Returns
        -------
        list of str
            Subset of :attr:`steps` for which :meth:`step_exists` is ``True``,
            in step order.
        """
        return [s for s in self.steps if self.step_exists(s)]

    @property
    def last_step(self) -> Optional[str]:
        """
        The last step whose output exists on disk, or ``None``.

        Returns
        -------
        str or None
            Last element of :attr:`existing_steps`, or ``None`` when no step
            has been run yet.
        """
        existing = self.existing_steps
        return existing[-1] if existing else None

    def step_exists(self, step_name: str) -> bool:
        """
        Check whether the output asset for *step_name* exists on disk.

        Parameters
        ----------
        step_name : str
            A member of :attr:`steps`.

        Returns
        -------
        bool
            ``True`` if the corresponding asset exists.
        """
        return self.asset_from_step_name(step_name).exists

    def get_next_steps(self, step_name: str) -> list[str]:
        """
        Return all steps that follow *step_name* in the pipeline.

        Parameters
        ----------
        step_name : str
            A member of :attr:`steps`.

        Returns
        -------
        list of str
            Steps after *step_name*, in execution order.  Empty list when
            *step_name* is the last step.
        """
        idx = self.steps.index(step_name)
        return list(self.steps[idx + 1:])

    def remove_next_steps_files(self, target_step_name: str) -> None:
        """
        Delete on-disk outputs for all steps that follow *target_step_name*.

        Useful when a step is re-run and downstream outputs are therefore stale.
        Warns before each deletion.

        Parameters
        ----------
        target_step_name : str
            The step *after* which all existing outputs will be deleted.
        """
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
    """
    Minimal base shared by all ClearMap processors and managers.

    Owns the :class:`~ClearMap.config.config_coordinator.ConfigCoordinator`
    reference and provides read-only config views, workspace access, and
    optional logging plumbing.

    Parameters
    ----------
    coordinator : ConfigCoordinator
        Central configuration manager.  All config reads and patch submissions
        go through this object.

    Attributes
    ----------
    cfg_coordinator : ConfigCoordinator
    workspace : Workspace2 or None
        Set by concrete subclasses after the sample is loaded.
    registration_processor : RegistrationProcessor or None
        Injected when atlas-space operations are needed.
    """
    config_name = ''

    def __init__(self, coordinator: "ConfigCoordinator"):  # REFACTOR: pass event_bus explicitly (don't steal from coordinator)
        super().__init__(coordinator._bus)
        self.cfg_coordinator: "ConfigCoordinator" = coordinator
        self.workspace: Optional[Workspace2] = None
        self.logger = None  # optional injected logger
        self.registration_processor: Optional["RegistrationProcessor"] = None

    def get_alignment_ref_channel_reg_cfg(self) -> Mapping[str, Any]:
        """
        Return the registration config for the alignment reference channel.

        Delegates to ``self.registration_processor.ref_channel_cfg``.

        Returns
        -------
        Mapping[str, Any]
            Read-only view of the reference channel's registration section.

        Raises
        ------
        ValueError
            If ``registration_processor`` has not been injected.
        """
        if not getattr(self, 'registration_processor'):
            raise ValueError(f'{self.__class__.__name__} cannot call '
                             f'get_alignment_ref_channel_reg_cfg() without a registration_processor attribute')
        else:
            return self.registration_processor.ref_channel_cfg

    @handle_deprecated_args(
        {'postfix': 'asset_sub_type',
         'prefix': 'sample_id'}
    )
    def get(self, asset_type, channel, asset_sub_type=None, **kwargs):   # channel and asset_sub_type defined for completion
        """
        Retrieve a workspace asset by type and channel.

        Parameters
        ----------
        asset_type : str
            Logical asset type, e.g. ``'stitched'``, ``'cells'``, ``'atlas'``.
        channel : str | tuple[str, ...] | None
            Channel name, compound channel tuple, or None for a global asset. Always explicit here;
            the single and compound channel orchestrators default it to their own channel(s).
        asset_sub_type : str or None, optional
            Optional sub-type qualifier, e.g. ``'raw'``, ``'filtered'``.
        **kwargs
            Forwarded to :meth:`~ClearMap.IO.workspace2.Workspace2.get`.

        Returns
        -------
        Asset
            The workspace asset object.

        Raises
        ------
        ClearMapRuntimeError
            If the workspace has not been initialised yet.
        """
        if self.workspace is None:
            raise ClearMapRuntimeError(f'Cannot call {self.__class__.__name__}.get without a workspace. '
                                       f'Please ensure it is assigned by calling {self.__class__.__name__}.setup() '
                                       f'with a valid sample manager first.')
        asset = self.workspace.get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs)
        return asset

    def get_path(self, asset_type, channel, asset_sub_type=None, **kwargs):   # channel and asset_sub_type defined for completion
        """
       Shortcut that returns the filesystem path of a workspace asset.

       Parameters
       ----------
       asset_type : str
           Logical asset type.
       channel : str | tuple[str, ...] | None
           Channel name, compound channel tuple, or None for a global asset. Always explicit.
       asset_sub_type : str or None, optional
           Optional sub-type qualifier.
       **kwargs
           Forwarded to :meth:`get`.

       Returns
       -------
       Path
           The path of the requested asset.
       """
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
        """
        Read-only view of the ``registration`` config section.

        Returns
        -------
        Mapping[str, Any]
        """
        return self.cfg_coordinator.get_config_view('registration')

    @property
    def config(self) -> Mapping[str, Mapping[str, Any]]:
        """
        Always-fresh, read-only snapshot of this processor's config section.

        The section is identified by :attr:`config_name`.

        Returns
        -------
        MappingProxyType
            Immutable view; request again to get the latest values after a patch.
        """
        return MappingProxyType(self.cfg_coordinator.get_config_view(self.config_name))

    @property
    def machine_config(self):
        """
        Read-only view of the ``machine`` (global params) config section (verbosity, start_folder, font_size…).

        Returns
        -------
        Mapping[str, Any]
        """
        return self.cfg_coordinator.get_config_view('machine')

    @property
    def verbose(self):
        """
        ``True`` when the machine verbosity is set to ``'debug'``.

        Returns
        -------
        bool
        """
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
    pipeline: str
        Class attribute: the pipeline this orchestrator is (a step of), one of PIPELINE_NAMES.
        A pipeline can have several orchestrators (e.g. TubeMap: BinaryVesselProcessor and
        VesselGraphProcessor), an orchestrator belongs to one pipeline. Compulsory for concrete
        orchestrators (checked when the class is defined); abstract bases declare abstract=True.
    """
    pipeline: ClassVar[str]

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        pipeline = getattr(cls, 'pipeline', None)
        if pipeline not in PIPELINE_NAMES:
            raise TypeError(f'{cls.__name__} must declare its pipeline as a class attribute, '
                            f'one of {sorted(PIPELINE_NAMES)}; got {pipeline!r}.')

    def __init__(self, coordinator: "ConfigCoordinator"):
        super().__init__(coordinator)
        self.stopped: bool = False
        self.progress_watcher: Optional["ProgressWatcher"] = None  # FIXME: ensure assigned
        self.sample_manager: Optional["SampleManager"] = None  # FIXME: ensure assigned
        self._setup_done: bool = False  # Whether setup() ran, see setup_complete

    @property
    def required_sample_channels(self) -> list[str]:
        """
        The sample channels this orchestrator works on.
        It can be set up once they are all in the workspace (see sample_ready).
        """
        raise ClearMapNotImplementedError(f'{self.__class__.__name__} must define required_sample_channels')

    @property
    def sample_ready(self) -> bool:
        """
        Whether the sample is loaded and the sample channels this orchestrator requires are in the workspace.
        A channel that is not complete yet (no path or data type) is not, but it only concerns the
        orchestrators that require it.
        """
        sample_manager = self.sample_manager
        return (sample_manager is not None and sample_manager.workspace_ready and
                sample_manager.has_channels_in_workspace(self.required_sample_channels))

    @property
    def setup_complete(self) -> bool:
        """
        Whether setup() ran and the sample channels this orchestrator requires are still in the workspace.
        Derived (not stored): the channels change after the setup, e.g. when the user adds one in the GUI.
        """
        return self._setup_done and self.sample_ready

    def setup_if_needed(self):
        """

          .. warning::

                This assumes self.channel is set if needed!
                and self.registration_processor is set if needed!
        """
        if not self.setup_complete and self.sample_manager is not None:
            self.setup()

    def setup(self, sample_manager: Optional["SampleManager"] = None):
        """
         Attach a sample manager and mark this processor as ready.

         Parameters
         ----------
         sample_manager : SampleManager, optional
             If provided, replaces the currently stored sample manager.
             When ``None``, the previously stored instance is reused.

         Raises
         ------
         ValueError
             If the config section identified by :attr:`config_name` is absent.
         """
        self.sample_manager = sample_manager if sample_manager else self.sample_manager
        if not self.cfg_coordinator.get_config_view(self.config_name):
            raise ValueError(f'Config section "{self.config_name}" not found in config coordinator')
        if self.sample_ready:
            self.workspace = self.sample_manager.workspace
            self._setup_done = True
        else:
            self._setup_done = False
            warnings.warn(f'Sample manager not setup yet. Cannot setup {self.__class__.__name__}.')

    def register_in_workspace(self) -> None:
        """
        Register in the workspace of the sample what this processor works on: check that its sample
        channels are there (the SampleManager adds them, a processor never does), create its compound
        channel if it has one, and add the assets of its pipeline that are missing. Idempotent,
        called by setup.

        Raises
        ------
        MissingChannelError
            If a sample channel of this processor is not in the workspace.
        """
        raise ClearMapNotImplementedError(f'{self.__class__.__name__} must define register_in_workspace')

    def set_progress_watcher(self, watcher):
        """
        Attach a progress watcher that drives progress-bar updates.

        Parameters
        ----------
        watcher : ProgressWatcher
            Watcher instance whose ``increment``, ``increment_main_progress``
            and ``main_step_name`` interface will be used.
        """

        self.progress_watcher = watcher

    def update_watcher_progress(self, val):
        """
        Increment the sub-step progress bar by *val*.

        Parameters
        ----------
        val : int
            Number of units to add.
        """
        if self.progress_watcher is not None:
            self.progress_watcher.increment(val)

    def update_watcher_main_progress(self, val=1):
        """
        Increment the main (top-level) progress bar by *val*.

        Parameters
        ----------
        val : int, optional
            Number of main steps completed.  Default is 1.
        """

        if self.progress_watcher is not None:
            self.progress_watcher.increment_main_progress(val)

    def set_watcher_step(self, step_name):
        """
        Update the displayed step name in the progress dialog.

        Parameters
        ----------
        step_name : str
            Human-readable name of the current processing step.
        """
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
        """
        Request cancellation of the current processing step.

        Sets :attr:`stopped` to ``True`` and attempts to shut down any running
        executor or subprocess attached to the workspace.  Raises
        :exc:`CanceledProcessing` when a subprocess is terminated.
        """

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
        """
        Execute the full processing pipeline for this orchestrator.

        Raises
        ------
        NotImplementedError
            Must be implemented by every concrete subclass.
        """
        raise NotImplementedError

    # def setup(self):
    #     pass

class _CurrentChannelSentinel:
    __slots__ = ()
CURRENT_CHANNEL: Final = _CurrentChannelSentinel()


class ChannelPipelineOrchestrator(PipelineOrchestrator, abstract=True):
    """
    Tab processor that is processing a single channel.

    The config is expected to have a 'channels' section with the channel name as key.
    When accessed, the config property returns the section for the current channel.
    """
    config_name = ''
    def __init__(self, coordinator: "ConfigCoordinator"):
        super().__init__(coordinator)
        self.channel: str = ''

    @property
    def required_sample_channels(self) -> list[str]:
        return [self.channel] if self.channel else []

    def get(self, asset_type, channel=CURRENT_CHANNEL, asset_sub_type=None, **kwargs):
        if channel is CURRENT_CHANNEL:
            channel = self.channel
        return super().get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs)

    def register_in_workspace(self) -> None:
        """Check that channel is in the workspace, and add the assets of the pipeline it misses"""
        check_sample_channels_in_workspace(self.sample_manager, [self.channel], required_by=self.__class__.__name__)
        self.workspace.ensure_pipeline(self.pipeline, self.channel, sample_id=self.sample_manager.prefix)

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
        """
        Submit a config patch scoped to the current channel.

        The patch is automatically nested under
        ``{config_name}.channels.{channel}`` before being forwarded to the
        coordinator.

        Parameters
        ----------
        patch : dict
            Flat-or-nested dict of values to update for this channel.
        origin : str, optional
            Human-readable description of the call site, used for audit
            logging.  Inferred from the call stack when empty.
        """
        self.cfg_coordinator.submit_patch(
            {self.config_name: {"channels": {self.channel: patch}}},
            sample_manager=self.sample_manager,
            origin=origin or infer_origin_from_caller())


class IndependentChannelsPipelineOrchestrator(PipelineOrchestrator, abstract=True):
    """
    Processor of several channels, each processed on its own (e.g. stitching, registration).

    Not bound to a channel: every asset access names its channel (get has no default channel),
    and channels lists the channels it processes.
    """
    def get(self, asset_type, channel, asset_sub_type=None, **kwargs):
        return super().get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs)

    @property
    def channels(self) -> list[str]:
        """The channels this processor processes, each on its own"""
        raise ClearMapNotImplementedError(f'{self.__class__.__name__} must define channels')

    @property
    def required_sample_channels(self) -> list[str]:
        return self.channels

    def register_in_workspace(self) -> None:
        """Register each of channels (see register_channel_in_workspace)"""
        for channel in self.channels:
            self.register_channel_in_workspace(channel)

    def register_channel_in_workspace(self, channel: str) -> None:
        """Check that channel is in the workspace, and add the assets of the pipeline it misses"""
        check_sample_channels_in_workspace(self.sample_manager, [channel], required_by=self.__class__.__name__)
        self.workspace.ensure_pipeline(self.pipeline, channel, sample_id=self.sample_manager.prefix)


class CompoundChannelPipelineOrchestrator(PipelineOrchestrator, abstract=True):
    """
    Processor of one compound channel: a channel made of several sample channels, its parts
    (e.g. the TubeMap vessels and arteries, or a colocalization pair).

    The config is the whole section (config_name). A pipeline with one config entry per
    compound channel adds PerCompoundConfigMixin.

    Attributes
    ----------
    compound_channel: tuple[str, ...]
        The parts, in the order of compound_channel_key (sorted unless oriented), whatever the order
        they are set in. This tuple is the workspace id of the compound channel, and the channel of
        get when none is given.
    oriented: bool
        Class attribute, derived from the pipeline (compound_keys.compound_is_oriented):
        whether the order of the parts carries meaning.
    """
    oriented: ClassVar[bool]

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs):
        super().__init_subclass__(abstract=abstract, **kwargs)
        if not abstract:
            cls.oriented = compound_is_oriented(cls.pipeline)

    def __init__(self, coordinator: "ConfigCoordinator"):
        super().__init__(coordinator)
        self._compound_channel: tuple[str, ...] = ()

    @property
    def compound_channel(self) -> tuple[str, ...]:
        return self._compound_channel

    @compound_channel.setter
    def compound_channel(self, parts: Sequence[str]) -> None:
        self._compound_channel = compound_channel_key(parts, oriented=self.oriented) if parts else ()

    @property
    def required_sample_channels(self) -> list[str]:
        return list(self.compound_channel)

    def get(self, asset_type, channel=CURRENT_CHANNEL, asset_sub_type=None, **kwargs):
        if channel is CURRENT_CHANNEL:
            if not self.compound_channel:
                raise ClearMapValueError(f'{self.__class__.__name__}.compound_channel is not set: '
                                         f'pass the channel explicitly or set up the processor first.')
            channel = self.compound_channel
        return super().get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs)

    def register_in_workspace(self) -> None:
        """Check the parts, create compound_channel and the assets of the pipeline if missing (see register_compound_channel)"""
        register_compound_channel(self.sample_manager, self.pipeline, self.compound_channel)


class PerCompoundConfigMixin:
    """
    For a CompoundChannelPipelineOrchestrator whose config section has one entry per compound
    channel, under ``channels.<compound key>`` (e.g. ``colocalization.channels['cfos-dapi']``):
    config is (and patch_channel writes) the entry of compound_channel.

    The key is str(CompoundKey(*compound_channel, oriented=oriented)), as written by the config
    adjusters (see SampleManager.colocalization_pair_keys), case included.
    """
    compound_channel: tuple[str, ...]
    oriented: ClassVar[bool]
    config_name: str
    cfg_coordinator: "ConfigCoordinator"
    sample_manager: Optional["SampleManager"]

    @property
    def compound_config_key(self) -> str:
        """The key of the config entry of compound_channel"""
        if not self.compound_channel:
            raise ClearMapValueError(f'{self.__class__.__name__}.compound_channel is not set')
        return str(CompoundKey(*self.compound_channel, oriented=self.oriented))

    @property
    def config(self) -> Mapping[str, Any]:
        """
        Read-only view of the config entry of compound_channel.

        Raises
        ------
        ClearMapConfigError
            If the section has no entry for compound_channel.
        """
        key = self.compound_config_key
        cfg = self.cfg_coordinator.get_config_view(self.config_name) or {}
        section = cfg.get('channels', {}).get(key)
        if section is None:
            raise ClearMapConfigError(f'No entry {key!r} in {self.config_name}.channels '
                                      f'(entries: {list(cfg.get("channels", {}))}).')
        return deep_freeze(section)

    def patch_channel(self, patch: dict, *, origin: str = "") -> None:
        """Submit a config patch scoped to the entry of compound_channel (see patch_channel of ChannelPipelineOrchestrator)"""
        self.cfg_coordinator.submit_patch(
            {self.config_name: {"channels": {self.compound_config_key: patch}}},
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
        """
        Attach a progress watcher for GUI progress-bar updates.

        Parameters
        ----------
        watcher : ProgressWatcher
        """
        self._progress_watcher = watcher

    def set_thread_wrapper(self, wrapper_callable: Callable) -> None:
        """
        Inject the GUI thread-wrapping helper (to keep the UI responsive).

        Parameters
        ----------
        wrapper_callable : Callable
            Typically ``main_window.wrap_in_thread``.  Falls back to
            synchronous execution when not set.
        """
        self._wrap_in_thread = wrapper_callable

    @property
    def groups(self) -> dict[str, list[str]]:
        """
        Mapping of group name → list of experiment source-directory paths.

        Returns
        -------
        dict[str, list[str]]
        """
        return self.group_controller.groups

    @property
    def results_folder(self) -> Path:
        """
        Root directory where group-level outputs are written.

        Returns
        -------
        Path

        Raises
        ------
        ValueError
            If ``results_folder`` has not been set via the group controller.
        """
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
        """
        Return the pipeline worker for a specific sample directory.

        Parameters
        ----------
        sample_src_dir : str or Path
            Root directory of the experiment.
        pipeline : str
            Pipeline name, e.g. ``'cell_map'``, ``'registration'``.
        channel : str or None, optional
            Channel key (required for per-channel pipelines).
        substep : str or None, optional
            Substep key (required for multi-substep pipelines such as vasculature).

        Returns
        -------
        PipelineOrchestrator
            The worker for that sample + pipeline combination.
        """
        return self.group_controller.get_worker(sample_src_dir, pipeline, channel=channel, substep=substep)

    def get_sample_manager_for(self, sample_src_dir: str | Path):
        """
        Return the SampleManager for a specific experiment directory.

        Parameters
        ----------
        sample_src_dir : str or Path
            Root directory of the experiment.

        Returns
        -------
        SampleManager
        """
        return self.group_controller.get_sample_manager(sample_src_dir)

    def run(self):
        """
         Execute the group-level analysis.

         Raises
         ------
         NotImplementedError
             Must be implemented by concrete subclasses.
         """
        raise NotImplementedError



class CanceledProcessing(BrokenProcessPool):  # TODO: better inheritance
    pass
