from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

from ClearMap.Utils.event_bus import EventBus
from ClearMap.config.config_coordinator import ConfigCoordinator
from ClearMap.config.config_repository import ConfigRepository
from ClearMap.gui.tabs import TabRegistry
from ClearMap.pipeline_orchestrators.processor_launcher import ProcessorLauncher


class ExperimentController:
    """
    Single entrypoint for app flow:
      - Boot (open, new from defaults, clone)
      - Keep an in-memory working model in ConfigCoordinator
      - Apply UI patches (Params -> Patch), then materialize -> validate -> atomic commit
      - Decide which tabs exist; build/update them and inject callbacks
      - Seal + snapshot + launch pipeline_orchestrators
      - Publish high-level UI events on an EventBus
    """

    def __init__(self, *, cfg_repo: ConfigRepository, cfg_coordinator:  ConfigCoordinator,
                 tabs_registry: TabRegistry, proc_launcher: ProcessorLauncher,
                 evt_bus: EventBus, sample_manager_factory: Optional[Callable[[], Any]] = None,
                 use_materializers: bool = True, use_snapshots: bool = False) -> None:
        self.cfg_repo = cfg_repo
        self.cfg_coordinator = cfg_coordinator
        self.tabs_registry = tabs_registry
        self.launcher = proc_launcher
        self.evt_bus = evt_bus
        self.use_materializers = use_materializers
        self.use_snapshots = use_snapshots

        self._exp_dir: Optional[Path] = None
        self._tabs: List[Any] = []  # List[GenericTab]
        self.sample_manager = sample_manager_factory() if sample_manager_factory else None

    @property
    def exp_dir(self) -> Path:
        return self._exp_dir

    def set_experiment_dir(self, exp_dir: str | Path) -> None:
        self._exp_dir = Path(exp_dir).resolve()
        # FIXME: self.sample_manager.src_folder = src_folder or something
        # keep repository in sync with the workspace
        self.cfg_repo.set_base_dir(self._exp_dir)
        if self.evt_bus:
            self.evt_bus.publish('workspace/changed', {'exp_dir': str(self._exp_dir)})

    def sample_path_exists(self) -> bool:
        return self.cfg_repo.path_for('sample').exists()

    def set_sample_id(self, sample_id: str) -> None:
        cfg = self.cfg_repo.load('sample')
        cfg['sample_id'] = sample_id

        # self.tab_managers['sample_info'].params.shared_sample_params.sample_id = sample_id
        # self.tab_managers['sample_info'].params.write()

        cfg.write()  # FIXME: use atomic commit via apply_ui_patch?

    def read_sample_version(self) -> str:
        cfg = self.cfg_repo.load('sample')
        return cfg.get('clearmap_version', '')

    # REFACTOR: part of ConfigRepository?
    # FIXME: reload ConfigCoordinator/ConfigRepository working model too?
    def upgrade_configs(self, from_version: str, to_version: str) -> None:
        """Run your converter over the current exp_dir."""
        from ClearMap.config.convert_config_versions import convert_versions
        convert_versions(from_version, to_version, self._exp_dir)

    # ---- boot / workspace ----------------------------------------------------
    def boot_open(self, exp_dir: Path) -> None:
        """
        Open an existing experiment directory: load all configs, build working model,
        materialize/validate/commit (to normalize), and build tabs.
        """
        self.set_experiment_dir(exp_dir)
        self.cfg_repo.set_base_dir(self._exp_dir)  # repo tracks where files live
        print(f'Opening experiment at {self._exp_dir}')

        # Load disk -> working model (no commit yet)
        configs: Dict[str, Dict] = self.cfg_repo.load_all()
        for name, cfg in configs.items():
            self.cfg_coordinator.set_working(name, cfg)

        # Normalize once (may add derived sections), then persist atomically
        self._materialize_validate_commit()

        # Build tabs based on current working model
        self.install_or_update_tabs()

    def boot_new(self, dest_dir: Path, template_dir: Optional[Path] = None) -> None:
        """
        Create a new experiment from defaults or clone a template, then open it.
        """
        if not dest_dir:
            dest_dir = self._exp_dir  # FIXME: check this fallback
        dest = Path(dest_dir).resolve()
        if template_dir:
            template_dir = Path(template_dir).resolve()
            self.cfg_repo.clone_from(template_dir, dest)
        else:
            self.cfg_repo.copy_from_defaults(dest)
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
        return self.cfg_repo.ensure_present(name)

    # ---- tabs lifecycle ------------------------------------------------------
    # FIXME: check those
    def install_or_update_tabs(self) -> None:
        """
        Decide which tabs exist (via TabRegistry + validators/materializers),
        create or reuse instances, inject callbacks, and notify UI.
        """
        view = self.get_config_view()
        valid_tab_classes = self.tabs_registry.valid_tabs(view)  # -> Iterable[type[GenericTab]]
        print(f'Valid tabs: {[cls.__name__ for cls in valid_tab_classes]}')

        by_cls = {t.__class__: t for t in self._tabs}
        new_tabs: List[Any] = []

        for cls in valid_tab_classes:
            tab = by_cls.get(cls) or cls(sample_manager=self.sample_manager)
            # Params write path (narrow) and read-only view:
            # FIXME: missing methods in Tab interface?
            tab.params.bind_apply_patch(self.apply_ui_patch)     # Params → Patch → Controller
            tab.set_params_view(self.get_config_view)            # Params reads working model
            # Optionally give tabs a lightweight “service bag”
            if hasattr(tab, "set_services"):
                tab.set_services(workspace_dir=self._exp_dir, repo=self.cfg_repo, bus=self.evt_bus)
            new_tabs.append(tab)

        self._tabs = new_tabs
        self.evt_bus.publish("tabs_updated", [t.title for t in self._tabs])

    def tabs(self) -> Iterable[Any]:
        """Return the current tab instances for the UI to render."""
        return list(self._tabs)

    def on_tab_activated(self, tab_key: str) -> Tuple[bool, Optional[str]]:
        """
        Business rules when a tab is activated (called by UI). Returns (ok, msg).
        Example rules:
         - Post-processing tabs require registration/alignment first
         - Batch tabs get lazy initialization on first open
        """
        tab = self._find_tab_by_key(tab_key)
        if not tab:
            return False, None

        # Example gate for 'post' tabs
        if getattr(tab, "processing_type", None) == "post":
            try:
                needs_reg = self.sample_manager and self.sample_manager.needs_registering()
            except Exception:
                needs_reg = False
            if needs_reg:
                return False, "Registration not completed. Please run alignment first."

            # One-time heavy setup (if tab defines it)
            if (hasattr(tab, "finalise_workers_setup") and
                    not getattr(tab, "_finalised", False)):
                tab.finalise_workers_setup()
                tab._finalised = True

        # Example lazy init for 'batch' tabs
        if (getattr(tab, "processing_type", None) == "batch" and
                not getattr(tab, "_initialised", False)):
            if hasattr(tab, "initialise_batch"):
                tab.initialise_batch()
            tab._initialised = True

        return True, None

    # ---- UI edits / persistence ---------------------------------------------
    def apply_ui_patch(self, patch: Dict[str, Any]) -> None:
        """
        Single write entrypoint: Params call this with a patch (persisted fields only).
        We merge it, derive dependent sections, validate, and atomically commit.
        """
        if not patch:
            return
        print(f'Applying UI patch keys: {list(patch.keys())}')
        self.cfg_coordinator.apply(patch)
        self._materialize_validate_commit()
        self.evt_bus.publish("config_changed", {"rev": self.cfg_coordinator._rev})

        # If tab structure depends on edited fields (e.g., channels), refresh tabs
        if self._tabs_may_have_changed(patch):
            self.install_or_update_tabs()

    # ---- run path (snapshot + launch) ---------------------------------------
    def seal_and_snapshot(self) -> Path:
        """
        Ensure the on-disk config is fully materialized & valid; take an immutable snapshot
        directory used by pipeline_orchestrators for reproducibility.
        """
        self._materialize_validate_commit()
        if not self.use_snapshots:
            # Fallback: return the live config dir (not ideal but preserves old behavior)
            assert self._exp_dir is not None, "Workspace not set"
            return self._exp_dir

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

    # ---- helpers / internals -------------------------------------------------
    def _find_tab_by_key(self, key: str) -> Optional[Any]:
        for t in self._tabs:
            if getattr(t, "key", None) == key or getattr(t, "title", "") == key:
                return t
        return None

    def _tabs_may_have_changed(self, patch: Dict[str, Any]) -> bool:
        """
        Heuristic: if user edited sample structure (e.g., channels),
        we may need to rebuild tab set.
        """
        # Adjust to your structure; this keeps it coarse on purpose.
        return "sample" in patch or "channels" in patch.get("sample", {})

    def _materialize_validate_commit(self) -> None:
        """
        The core “pipeline”: derive dependent config, validate, and atomically persist.
        """
        if self.use_materializers:
            self.cfg_coordinator.materialize()
        self.cfg_coordinator.validate()
        self._atomic_commit_all()
        print(f'Committed rev={self.cfg_coordinator._rev}')

    def _atomic_commit_all(self) -> None:
        """
        Ask the repository to atomically write all working configs to disk.
        """
        if self._exp_dir is not None:
            self.cfg_repo.set_base_dir(self._exp_dir)
        self.cfg_coordinator.commit_all(self.cfg_repo)
