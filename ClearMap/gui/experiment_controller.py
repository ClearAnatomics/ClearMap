"""
experiment_controller
=====================

Coordinator for application flow between configuration, workspace/sample state,
and processing workers.

This module wires together three main subsystems:

- **ConfigCoordinator** — in-memory working model for all config files with
  atomic commit and validation semantics.
- **SampleManager** — runtime view of the sample + Workspace2 (assets on disk),
  reconciled from config and updated via events.
- **EventBus** — decoupled, typed pub/sub used by the UI and back-end.

Responsibilities
----------------
- Boot an experiment (open/clone/new), seed defaults, run adjusters, validate,
  and commit configuration.
- Construct and cache worker instances (per pipeline and channel), reconcile
  them when channels change, and expose a simple `get_worker(...)` API.
- Bridge UI patches into `ConfigCoordinator.submit_patch(...)` so all edits are
  validated and persisted atomically.
- Optionally seal and snapshot the working config for reproducible runs.
- Listen to UI/domain events (e.g., channel rename/changed) and keep worker
  topology consistent.

Thread-safety
-------------
- `ConfigCoordinator` provides its own locking; reads of `get_config_view()`
  are safe against mid-edit states.
- `EventBus` is thread-safe and resilient to dead weakrefs.

Design Notes
------------
- All persistence flows through **ConfigCoordinator**
- **SampleManager** keeps the Workspace2 in sync by subscribing to domain
  events (e.g., `ChannelRenamed`) and by reading the validated config view.
- Worker creation is **lazy**; reconciliation removes stale workers while new
  ones are built on first access.
"""
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, Any, Union

from ClearMap.Utils.event_bus import EventBus, BusSubscriberMixin
from ClearMap.Utils.events import WorkspaceChanged, UiChannelRenamed, UiChannelsChanged
from ClearMap.config.config_coordinator import ConfigCoordinator
from ClearMap.config.defaults_provider import DefaultsProvider
from ClearMap.pipeline_orchestrators.group_orchestrators import DensityGroupAnalysisOrchestrator
from ClearMap.pipeline_orchestrators.processor_launcher import ProcessorLauncher

from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager


# REFACTOR: move to a common place
ChannelKey = Optional[Union[str, Tuple[str, ...]]]  # str | (str,...) | None
RealChannelKey = Union[str, Tuple[str, ...]]  # str | (str,...) (skip None)
FactoryKey = Tuple[str, Optional[str]]           # (pipeline, substep)
WorkerKey  = Tuple[ChannelKey, Optional[str]]    # (channel_key, substep)


def swap_old_channel(ch_key: ChannelKey, old_chan: RealChannelKey, new_chan: RealChannelKey) -> ChannelKey:
    """
    Swap old_chan by new_chan in ch_key if present.
    If ch_key is a simple channel (str) or None -> str.replace() or None
    If ch_key is a tuple -> replace old_chan by new_chan in the tuple
    Parameters
    ----------
    ch_key: ChannelKey
        ChannelKey to update
    old_chan: RealChannelKey
        Channel name to replace
    new_chan: RealChannelKey
        Channel name to use as replacement

    Returns
    -------
    ChannelKey:
        new_ch_key with old_chan replaced by new_chan, or None if key was None
    """
    new_ch_key = None
    if ch_key == old_chan:
        new_ch_key = new_chan
    elif isinstance(ch_key, tuple) and old_chan in ch_key:
        # Compound channels, keep other channel and replace old by new
        new_ch_key = tuple(new_chan if x == old_chan else x for x in ch_key)
    return new_ch_key


def _channel_was_removed(ch_key: ChannelKey, removed: set[RealChannelKey]) -> bool:
    if ch_key is None:
        return False
    if isinstance(ch_key, tuple):
        return any(x in removed for x in ch_key)
    return ch_key in removed


class ExperimentController(BusSubscriberMixin):
    """
    Orchestrates experiment lifecycle, config edits, and worker topology.

    This controller is the single entrypoint for app flow. It owns the
    `ConfigCoordinator` and `SampleManager`, constructs per-pipeline workers,
    applies UI patches through the coordinator, and reacts to bus events that
    affect worker/channel state.

    Parameters
    ----------
    cfg_coordinator : ConfigCoordinator
        Central configuration manager (working view + validation + commit).
    sample_manager : SampleManager
        Provides sample semantics and manages the Workspace2.
    proc_launcher : ProcessorLauncher, optional
        Used for sealing/snapshotting and launching processors.
    evt_bus : EventBus
        Event bus for decoupled UI/backend communication.
    use_materializers : bool, default True
        Reserved for future materialization hooks (kept for compatibility).
    use_snapshots : bool, default False
        When True, runs seal+snapshot before launching processors.

    Attributes
    ----------
    _exp_dir : Optional[pathlib.Path]
        Current experiment directory (None until `set_experiment_dir()`).
    _workers : Dict[str, Dict[Tuple[ChannelKey, Optional[str]], object]]
        Cached workers keyed by pipeline and (channel_key, substep).
    _factories : Dict[Tuple[str, Optional[str]], Callable]
        Factory registry: (pipeline, substep) -> factory(sm, coord, channel_key).

    Signals
    -------
    - UiChannelRenamed  -> :meth:`on_channel_renamed`
    - UiChannelsChanged -> :meth:`on_channels_changed`

    Published Events
    ----------------
    - (optional) WorkspaceChanged(exp_dir=...) when the experiment directory is set.
      Ensure the event class exists before enabling.

    Key Operations
    --------------
    boot_open(exp_dir)
        Load configs, seed defaults, run adjusters, validate, and commit.
    boot_new(dest_dir, template_dir=None)
        Create from defaults or clone template, then open.
    apply_ui_patch(patch)
        Validate+commit a UI patch via `ConfigCoordinator.submit_patch(...)`.
    get_worker(pipeline, channel=None, substep=None)
        Lazy construction of workers. Cached until reconciliation.
    reconcile_workers_after_channel_change(before, after)
        Remove stale workers after channel removals; new ones remain lazy.
    seal_and_snapshot()
        Ensure committed config; optionally write an immutable snapshot.

    Invariants
    ----------
    - All config mutations go through `ConfigCoordinator.submit(...)` or
      `submit_patch(...)`; direct edits to files are disallowed.
    - `SampleManager` derives runtime state from committed config and bus events.
    - Workers never mutate global config directly; they request patches through
      controller-provided paths.

    """

    def __init__(self, *, cfg_coordinator: ConfigCoordinator, sample_manager: SampleManager,
                 proc_launcher: Optional[ProcessorLauncher] = None, evt_bus: EventBus,
                 use_materializers: bool = True, use_snapshots: bool = False) -> None:
        self.cfg_coordinator = cfg_coordinator
        self.sample_manager = sample_manager
        self.launcher = proc_launcher
        super().__init__(evt_bus)
        self.use_materializers = use_materializers
        self.use_snapshots = use_snapshots

        self._exp_dir: Optional[Path] = None
        self._workers: Dict[str, Dict[WorkerKey, object]] = defaultdict(dict)
        self._factories: Dict[tuple[str, Optional[str]], Callable] = {}  # pipeline(/sub) -> factory(sm, coord, key=None)
        self._factory_scope: Dict[tuple[str, Optional[str]], str] = {}  # "global" | "per_channel" | "per_pair"
        self._register_default_factories()

        self._hydrating: bool = False

        self.subscribe(UiChannelRenamed, self.on_channel_renamed)
        self.subscribe(UiChannelsChanged, self.on_channels_changed)
        # FIXME: also subscribe to dtype changed in config -> reconcile workers (pipelines)

    @property
    def hydrating(self) -> bool:
        return self._hydrating

    def _register_default_factories(self):
        # WARNING: Lazy imports for snappy boot and to avoid cycles

        # simple/global
        def stitching_factory(sm, coord, key=None):
            from ClearMap.pipeline_orchestrators.stitching_orchestrator import StitchingProcessor
            return StitchingProcessor(sm, coord)

        def registration_factory(sm, coord, key=None):
            from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor
            return RegistrationProcessor(sm, coord)
        # per-channel
        def cell_map_factory(sm, coord, key):
            from ClearMap.pipeline_orchestrators.cell_map import CellDetector
            return CellDetector(sm, coord, channel=key, registration_processor=reg_proc)
        def tract_map_factory(sm, coord, key):
            from ClearMap.pipeline_orchestrators.tract_map import TractMapProcessor
            return TractMapProcessor(sm, coord, channel=key, registration_processor=reg_proc)
        # per-pair (key is a tuple)
        def colocalization_factory(sm, coord, key):
            from ClearMap.pipeline_orchestrators.colocalization import ColocalizationProcessor
            return ColocalizationProcessor(sm, coord, channels=key, registration_processor=reg_proc)
        # vasculature as substeps (single pipeline, two steps)
        def binary_vessel_processor_factory(sm, coord, key=None):
            from ClearMap.pipeline_orchestrators.tube_map import BinaryVesselProcessor
            return BinaryVesselProcessor(sm, coord)

        def vessel_graph_processor_factory(sm, coord, key=None):
            from ClearMap.pipeline_orchestrators.tube_map import VesselGraphProcessor
            return VesselGraphProcessor(sm, coord, registration_processor=reg_proc)

        self.register_worker_factory('stitching',  stitching_factory, scope='global')
        self.register_worker_factory('registration', registration_factory, scope='global')

        # pull registration_processor now for factories that depend on it  # OPTIMISE: find a way to avoid instantiating too early
        reg_proc = self.get_worker('registration')

        self.register_worker_factory('cell_map', cell_map_factory, scope='per_channel')
        self.register_worker_factory('tract_map', tract_map_factory, scope='per_channel')
        self.register_worker_factory('colocalization', colocalization_factory, scope='per_pair')
        self.register_worker_factory('vasculature', binary_vessel_processor_factory,
                                     substep='binary', scope='global')
        self.register_worker_factory('vasculature', vessel_graph_processor_factory,
                                     substep='graph', scope='global')

    def register_worker_factory(self, pipeline: str, factory: Callable, substep: Optional[str] = None,
                                scope: str = 'global') -> None:
        """
        Register a factory function that creates workers for a given pipeline
        (and optionally substep). The factory will be called with (sample_manager, cfg_coordinator, channel_key).

        Parameters
        ----------
        pipeline : str
            Name of the processing pipeline (e.g., "registration", "cell_map").
        factory : Callable
            Factory function to create the worker. (built with the ctor of the worker)
        substep : Optional[str], default None
            Optional substep identifier within the pipeline.
        scope : str, default "global"
            Scope of the worker: "global", "per_channel", or "per_pair".
        """
        key = (pipeline, substep)
        self._factories[key] = factory
        self._factory_scope[key] = scope

    def set_workers_progress_watcher(self, watcher):  # REFACTOR: define watcher signature
        """Attach watcher to SampleManager and every existing worker."""
        # self.sample_manager.set_progress_watcher(watcher)
        for pipeline_workers in self._workers.values():
            for worker in pipeline_workers.values():  # each worker keyed by (channel, substep)
                if hasattr(worker, "set_progress_watcher"):
                    worker.set_progress_watcher(watcher)

    @property
    def exp_dir(self) -> Optional[Path]:
        return self._exp_dir

    def set_experiment_dir(self, exp_dir: str | Path) -> None:
        self._exp_dir = Path(exp_dir).resolve()
        self.cfg_coordinator.set_base_dir(self._exp_dir)
        self.sample_manager.setup(exp_dir)
        self.publish(WorkspaceChanged(exp_dir=str(self._exp_dir)))

    @staticmethod
    def _make_defaults_provider() -> DefaultsProvider:
        """
        Build a DefaultsProvider pointing to packaged defaults & schemas.
        Adjust paths to match your repo layout if needed.
        """
        base = Path(__file__).resolve().parent
        schemas_dir = base / 'schemas'  # put JSON Schemas here
        return DefaultsProvider(schemas_dir=schemas_dir, code_defaults={}, tabs_only=True)
        # TODO: check if tabs_only=True is OK

    def sample_path_exists(self) -> bool:
        return self.cfg_coordinator.config_exists_any('sample')

    def set_sample_id(self, sample_id: str) -> None:
        if self._exp_dir is not None:
            self.cfg_coordinator.set_base_dir(self._exp_dir)
        self.cfg_coordinator.submit_patch({'sample': {'sample_id': sample_id}},
                                          sample_manager=self.sample_manager,
                                          origin='ExperimentController.set_sample_id',)

    def _get_sample_cfg(self):
        sample_cfg = self.cfg_coordinator.get_config_view('sample')
        if not sample_cfg: # Load sample only because might be partial experiment
            sample_cfg = self.cfg_coordinator.load('sample')
        return sample_cfg

    def read_sample_version(self) -> str:
        return self._get_sample_cfg().get('clearmap_version', '')

    def get_or_init_sample_id(self):
        sample_id = self._get_sample_cfg().get('sample_id')
        return sample_id or None  # leave creation to UI prompt (app will call set_sample_id)

    def upgrade_configs(self, from_version: str, to_version: str) -> None:
        """
        Convert the config to the current ClearMap version in
        the current exp_dir.
        Reload all configs into the working model afterwards.

        Parameters
        ----------
        from_version: str
            The version the config is currently in.
        to_version: str
            The version to convert the config to.
        """
        from ClearMap.config.convert_config_versions import convert_versions
        convert_versions(from_version, to_version, exp_dir=self._exp_dir)
        self.cfg_coordinator.load_all()

    # ---- boot / workspace ----------------------------------------------------
    def boot_open(self, exp_dir: Path) -> None:
        """
        Open an existing experiment directory: load all configs, build working model,
        materialize/validate/commit (to normalize), and build tabs.
        """
        self._hydrating = True

        try:
            self.set_experiment_dir(exp_dir)
            print(f'Opening experiment at {self._exp_dir}')

            self.cfg_coordinator.load_all()

            defaults_provider = self._make_defaults_provider()
            self.cfg_coordinator.set_defaults_provider(defaults_provider)

            # Ensure configs will never have missing fields
            self.cfg_coordinator.seed_missing_from_defaults(tabs_only=True)

            # Normalize once (may add derived sections), then persist atomically
            self.cfg_coordinator.submit(sample_manager=self.sample_manager, do_run_adjusters=True,
                                        validate=True, commit=True)
        finally:
            self._hydrating = False

    def boot_new(self, dest_dir: Optional[Path] = None, template_dir: Optional[Path] = None) -> None:
        """
        Create a new experiment from defaults or clone a template, then open it.
        """
        if not dest_dir:
            dest_dir = self._exp_dir  # FIXME: check this fallback
        dest = Path(dest_dir).resolve()
        if template_dir:
            template_dir = Path(template_dir).resolve()
            self.cfg_coordinator.clone_from(template_dir, dest)
        else:
            self.cfg_coordinator.copy_from_defaults(dest)
        self.boot_open(dest)

    def clone_from(self, template_dir: Path, dest_dir: Path) -> None:
        """
        Explicit clone API (alias); then open the new experiment.
        """
        self.boot_new(dest_dir, template_dir)

    def ensure_config_present(self, name: str) -> Path | None:
        """
        Ensure the named config file exists in the current experiment directory,
        copying from defaults if needed. Returns the path or None if no
        default exists for that name.

        Parameters
        ----------
        name: str
            Logical config name, e.g. "sample", "registration", etc.

        Returns
        -------
        Path | None
            The path to the ensured config file, or None if no default exists.
        """
        return self.cfg_coordinator.ensure_present(name)

    def get_worker(self, pipeline: str, channel: ChannelKey = None, substep: Optional[str] = None):
        factory_key: FactoryKey = (pipeline, substep)
        try:
            factory = self._factories[factory_key]
        except KeyError:
            raise KeyError(f'No factory registered for pipeline "{pipeline}" ({substep=})')

        scope = self._factory_scope.get(factory_key, 'global')

        worker_key: WorkerKey = (channel, substep)
        pipeline_workers = self._workers[pipeline]  # what if pipeline not registered yet?
        worker = pipeline_workers.get(worker_key)
        if worker is not None:
            return worker
        else:  # FIXME: add pipeline to workspace or inside worker ctor?
            if scope == 'global':
                if channel is not None:
                    raise ValueError(
                        f'Pipeline "{pipeline}" is global (scope=global) but channel={channel!r} was requested.')
                worker = factory(self.sample_manager, self.cfg_coordinator)
            elif scope in ('per_channel', 'per_pair'):
                if channel is None:
                    raise ValueError(f'Pipeline "{pipeline}" (scope={scope}) requires a channel key, got channel=None.')
                worker = factory(self.sample_manager, self.cfg_coordinator, channel)
            else:
                raise ValueError(f'Unknown worker scope "{scope}" for pipeline "{pipeline}".')

            pipeline_workers[worker_key] = worker
            return worker

    def get_workers(self, pipeline: str, channels: Iterable[ChannelKey], substep: Optional[str] = None):
        return {ch: self.get_worker(pipeline, ch, substep=substep) for ch in channels}

    def worker_is_ready(self, worker_name):
        if worker_name != 'sample' and self.sample_manager.workspace is None:
            return False
        else:
            registration_worker = self.get_worker('registration')
            try:
                return not self.sample_manager.needs_registering(registration_worker)
            except Exception:
                return False

    def reconcile_workers(self, pipeline: str, desired_channels: Optional[Iterable[ChannelKey]],
                          substep: Optional[str] = None, keep_global: bool = True) -> Dict[ChannelKey, object]:
        """
        Ensure all desired workers exist and remove obsolete ones for this (pipeline, substep).
        desired_channels=None means 'global-only' (i.e., {None} if keep_global).
        """
        desired = set(desired_channels or [])

        factory_key = (pipeline, substep)
        scope = self._factory_scope.get(factory_key, 'global')
        if keep_global and scope == 'global':
            desired.add(None)

        # build missing (lazy via get_worker)
        for ch in desired:
            self.get_worker(pipeline, ch, substep=substep)

        pipeline_workers = self._workers[pipeline]

        # prune stale
        to_remove = []
        for (ch, current_substep), _worker in list(pipeline_workers.items()):
            if current_substep != substep:
                continue
            if ch not in desired:
                to_remove.append((ch, current_substep))
        for key in to_remove:
            pipeline_workers.pop(key, None)

        # return current (without substep key in API, map by channel)
        return {ch: pipeline_workers[(ch, substep)] for ch in desired if (ch, substep) in pipeline_workers}

    def reconcile_workers_after_channel_change(self, before: list[str], after: list[str]) -> None:
        """
        Prune workers that reference removed channels by delegating to reconcile_workers.
        Creation of workers for *new* channels is left lazy (on first use).
        """
        removed = set(c for c in (before or []) if c not in (after or []))

        if not removed:
            return

        for pipeline, workers in self._workers.items():
            # Keep all existing non-removed channel keys for this pipeline/substep
            by_sub: dict[Optional[str], set[ChannelKey]] = {}
            for (ch_key, substep) in list(workers.keys()):
                if not _channel_was_removed(ch_key, removed):
                    by_sub.setdefault(substep, set()).add(ch_key)

            # Reconcile per substep using the desired (filtered) set
            for substep, desired in by_sub.items():
                scope = self._factory_scope.get((pipeline, substep), 'global')
                keep_global = (scope == 'global')
                self.reconcile_workers(pipeline, desired_channels=desired,
                                       substep=substep, keep_global=keep_global)

    def rename_workers_channels(self, old_chan: RealChannelKey, new_chan: RealChannelKey):
        for pipeline, pipeline_workers in self._workers.items():
            # work on a copy of keys because we mutate the dict
            for (ch_key, substep) in list(pipeline_workers.keys()):
                new_ch_key = swap_old_channel(ch_key, old_chan, new_chan)
                if new_ch_key is not None:
                    worker = pipeline_workers.pop((ch_key, substep))
                    pipeline_workers[(new_ch_key, substep)] = worker

    def on_channel_renamed(self, evt: UiChannelRenamed) -> None:  # FIXME: shouldn't it be ChannelRenamed (sent by config coordinator AFTER)?
        """
        Apply a rename everywhere (workers, config), then publish one snapshot event.
        """
        try:
            self.rename_workers_channels(evt.old, evt.new)
        except Exception:
            pass

    def on_channels_changed(self, evt: UiChannelsChanged) -> None:
        """
        Handle add/remove; seed/drop per-channel sections, purge artifacts, then publish.
        """
        self.reconcile_workers_after_channel_change(before=evt.before, after=evt.after)

    def channel_snapshot(self):
        """
        Single “truth” event: current channel list + registration partner defaults,
        derived centrally via the registration worker.
        """
        channels = self.sample_manager.channels

        # derive partner defaults via registration rules (no UI logic here)
        partners = {}
        try:
            reg = self.get_worker('registration')
            for ch in channels:
                partners[ch] = {
                    'align_with': reg.get_align_with(ch),
                    'moving': reg.get_moving_channel(ch),
                }
        except Exception:
            # keep UI alive even if reg worker not ready
            partners = {ch: {'align_with': None, 'moving': None} for ch in channels}

        return channels, partners  # , 'renamed': rename_map}

    # ---- UI edits / persistence ---------------------------------------------
    def apply_ui_patch(self, patch: Dict[str, Any]) -> None:
        """
        Single write entrypoint: Params call this with a patch (persisted fields only).
        We merge it, derive dependent sections, validate, and atomically commit.
        """
        if not patch:
            return
        print(f'Applying UI patch keys: {list(patch.keys())}')
        self.cfg_coordinator.submit_patch(patch, sample_manager=self.sample_manager,
                                          do_run_adjusters=True, validate=True, commit=True)
        # self.evt_bus.publish(ConfigChanged({"rev": self.cfg_coordinator._rev}))

    # ---- run path (snapshot + launch) ---------------------------------------
    def seal_and_snapshot(self) -> Path:
        """
        Ensure the on-disk config is fully materialized & valid; take an immutable snapshot
        directory used by pipeline_orchestrators for reproducibility.
        """
        self.cfg_coordinator.submit(sample_manager=self.sample_manager, do_run_adjusters=True,
                                    validate=True, commit=True)
        if not self.use_snapshots:
            if self._exp_dir is None:
                raise ValueError('Workspace not set')
            return self._exp_dir  # legacy behavior: use working dir

        # FIXME: this is redundant with ConfigCoordinator.snapshot_to()
        snap_dir = self.launcher.seal_and_snapshot(self.cfg_coordinator)  # expects coordinator to expose working view
        print(f'Sealed snapshot at {snap_dir}')
        return snap_dir

    def launch_processor(self, processor_class, **opts):
        """
        Typical run: seal snapshot, then launch processor with config-dir=<snapshot>.
        """
        snap = self.seal_and_snapshot()
        return self.launcher.launch(processor_class, snap, **opts)

    # ---- read-only views for UI ---------------------------------------------
    def get_config_view(self) -> Dict[str, Any]:
        """
        Dict-like snapshot the UI can read to populate widgets.
        Do not expose repository paths here—UI shouldn’t hit disk.
        """
        return self.cfg_coordinator.get_config_view()


####################### Multi experiment controller ###########################
class AnalysisGroupController:
    """
    Manages multiple ExperimentControllers, one per experiment root (src_dir).
    Also acts as a factory/router to fetch workers tied to the correct sample.
    """
    def __init__(self, cfg_coordinator_factory, event_bus, exp_controller_factory):
        """
        cfg_coordinator_factory: callable(base_dir: Path) -> ConfigCoordinator
        exp_controller_factory:  callable(sample_manager, cfg_coordinator, event_bus) -> ExperimentController
        """
        self._cfg_coordinator_factory = cfg_coordinator_factory  # Built for each sample
        self._exp_controller_factory   = exp_controller_factory  # Built for each sample
        self._bus = event_bus

        self._controllers: dict[Path, ExperimentController] = {}
        self._groups: dict[str, list[str]] = {}
        self._results_folder: Path | None = None

        self._analysis_worker: Optional["DensityGroupAnalysisOrchestrator"] = None

        self._progress_watcher = None
        self._thread_wrapper   = None

    # ---------- external state ----------
    def set_groups(self, groups: dict[str, list[str]]):
        self._groups = {k: [str(Path(p)) for p in v] for k, v in groups.items()}

    def set_results_folder(self, results_folder: str | Path):
        self._results_folder = Path(results_folder)

    def set_progress_watcher(self, watcher):
        self._progress_watcher = watcher

    def set_thread_wrapper(self, wrapper):
        self._thread_wrapper = wrapper

    @property
    def groups(self) -> dict[str, list[str]]:
        return self._groups

    @property
    def results_folder(self) -> Path:
        if self._results_folder is None:
            raise ValueError("results_folder not set")
        return self._results_folder

    def _get_or_create_exp_controller(self, sample_src_dir: str | Path) -> "ExperimentController":
        root = Path(sample_src_dir).resolve()

        # Return if already cached
        if root in self._controllers:
            return self._controllers[root]

        # Otherwise, create new, cache, and return
        cfg_coordinator = self._cfg_coordinator_factory(base_dir=root)   # isolated config view for this sample
        sample_mgr = SampleManager(config_coordinator=cfg_coordinator, src_dir=root)
        exp_controller = self._exp_controller_factory(cfg_coordinator=cfg_coordinator, sample_manager=sample_mgr,
                                                      evt_bus=self._bus)
        self._controllers[root] = exp_controller
        return exp_controller

    def get_worker(self, sample_src_dir: str | Path, pipeline: str, *, channel=None, substep=None):
        exp_controller = self._get_or_create_exp_controller(sample_src_dir)
        return exp_controller.get_worker(pipeline, channel=channel, substep=substep)

    def get_sample_manager(self, sample_src_dir: str | Path) -> SampleManager:
        exp_controller = self._get_or_create_exp_controller(sample_src_dir)
        return exp_controller.sample_manager

    def get_density_orchestrator(self) -> "DensityGroupAnalysisOrchestrator":
        if self._analysis_worker is not None:
            return self._analysis_worker
        analysis_worker = DensityGroupAnalysisOrchestrator(group_controller=self)
        if self._progress_watcher:
            analysis_worker.set_progress_watcher(self._progress_watcher)
        if self._thread_wrapper:
            analysis_worker.set_thread_wrapper(self._thread_wrapper)
        self._analysis_worker = analysis_worker
        return analysis_worker
