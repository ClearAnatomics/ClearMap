from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from ClearMap.config.config_loader import ConfigHandler


class SnapshotManager:
    """
    Writes an immutable copy of the current working configs to:
      <exp_dir>/config_snapshots/<YYYYMMDD_HHMMSS>/

    File names are derived from the ConfigLoader name→filename.
    """

    def __init__(self, *, experiment_dir: Path, known_names: Iterable[str]) -> None:
        self._exp_dir = Path(experiment_dir)
        self._known_names = list(known_names)

    def create_snapshot(self, working: Dict[str, Dict[str, Any]]) -> Path:
        """
        Persist all configs found in `working` into a new timestamped directory.
        Returns the snapshot directory path.
        """
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        snap_dir = self._exp_dir / 'config_snapshots' / now_str  # FIXME: from workspace2
        snap_dir.mkdir(parents=True, exist_ok=True)

        cfg_handler = ConfigHandler(self._exp_dir)

        manifest_lines = [f'Snapshot created: {now_str}', 'Files: ']
        for name in self._known_names:
            if name not in working:
                continue
            cfg_dict = working[name]
            # Resolve the current filename (or the first possible if missing)
            target_name = Path(cfg_handler.get_cfg_path(name, must_exist=False)).name
            target_path = snap_dir / target_name
            cfg_handler.dump(target_path, cfg_dict)

            manifest_lines.append(f' - {name} -> {target_name}')

        manifest = snap_dir / 'MANIFEST.txt'  # write a tiny manifest for traceability
        manifest.write_text('\n'.join(manifest_lines), encoding='utf-8')
        return snap_dir


class ProcessorLauncher:
    """
    Seals the working configs into an immutable snapshot dir and launches pipeline_orchestrators
    against that snapshot location (CLI/cluster-friendly).
    """
    def __init__(self, snapshot_manager: SnapshotManager | None = None) -> None:
        self._snap = snapshot_manager

    def seal_and_snapshot(self, coordinator) -> Path:
        """
        Create a sealed snapshot of the current working model.

        Parameters
        ----------
        coordinator: ConfigCoordinator
            The configuration coordinator holding the current working model.

        Returns
        -------
        Path to the snapshot directory.
        """
        if not self._snap:
            raise RuntimeError('SnapshotManager not configured')
        view = coordinator.get_config_view()
        return self._snap.create_snapshot(view)

    def launch(self, processor_class, snapshot_dir: Path, **opts) -> Any:
        """
        Processors can be launched here.

        Parameters
        ----------
        processor_class: Callable
            a callable taking (config_dir, **opts).
        snapshot_dir: Path
            the sealed config snapshot to use.
        opts: dict
            additional options to pass to the processor.

        Returns
        -------
        Whatever the processor returns.
        """
        proc = processor_class(config_dir=snapshot_dir, **opts)
        return proc.run()  # or whatever your pipeline_orchestrators expose   # FIXME:
