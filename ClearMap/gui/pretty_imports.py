import time
import importlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
import pathlib

import yaml

from ClearMap.config.early_boot import MachineConfig

EPSILON_DT = 0.0005  # minimum time to avoid zero-division in progress calculations


@dataclass(frozen=True)
class ImportTask:
    """
    A single staged-import task.

    - module-only: ImportTask("pyqtgraph", as_name="pg")
    - from-imports: ImportTask("pkg.mod", symbols=("Foo","Bar"))
    """
    module: str
    attrs: tuple[str, ...] = ()             # from x import (a, b, c)
    message: Optional[str] = None
    weight: Optional[float] = None            # optional manual weight override, else learned profile
    as_name: Optional[str] = None             # optional "import x as y"

    # Alternate constructors
    @classmethod
    def module_only(cls, module: str, *, as_name: Optional[str] = None, message: Optional[str] = None, weight: Optional[float] = None) -> "ImportTask":
        return cls(module=module, attrs=(), as_name=as_name, message=message, weight=weight)

    @classmethod
    def from_imports(cls, module: str, *attrs: str, message: Optional[str] = None, weight: Optional[float] = None) -> "ImportTask":
        return cls(module=module, attrs=tuple(attrs), as_name=None, message=message, weight=weight)

    # derived properties
    @property
    def has_attrs(self) -> bool:
        return len(self.attrs) > 0

    @property
    def export_name(self) -> str:
        """Name under which the *module* will be exported (ignored for from-imports)."""
        return self.as_name or self.module.rsplit(".", 1)[-1]

    @property
    def display_message(self) -> str:
        return self.message or f"Importing {self.module}…"

    @property
    def key(self) -> str:
        """Key for profiling/weighting (group by module)."""
        return self.module

    def __str__(self) -> str:
        return self.display_message



OnMsg = Optional[Callable[[str], None]]
OnProg = Optional[Callable[[int], None]]


def _profile_path() -> pathlib.Path:
    major, minor = MachineConfig._version.split('.')[:2]
    return pathlib.Path.home() / '.clearmap' / f'.import_profile_v{major}_{minor}.yml'


def _load_weights(tasks: List[ImportTask]) -> List[float]:
    """
    Read the import time weights from disk if available.
    If this is the first run and a profile is not available, assume even weights (1.0).

    Parameters
    ----------
    tasks : List[ImportTask]
        List of import tasks to be performed.

    Returns
    -------
    List[float]
        Weights to drive the progress bar.
    """
    path = _profile_path()
    persisted: Dict[str, float] = {}
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8')) or []
            persisted = {d.get('module'): float(d.get('weight', 1.0)) for d in data if isinstance(d, dict)}
        except Exception:
            persisted = {}

    raw = []
    for t in tasks:
        if t.weight is not None:
            raw.append(float(t.weight))
        else:
            raw.append(float(persisted.get(t.module, 1.0)))

    total = sum(raw) or 1.0
    return [x / total for x in raw]


def _save_weights(tasks: List[ImportTask], seconds: List[float]) -> None:
    """
    Save the import time weights to disk for next time.
    Best-effort only, ignore any errors.

    Parameters
    ----------
    tasks: List[ImportTask]
        List of import tasks that were performed.
    seconds: List[float]
        List of elapsed times for each task.
    """
    # Normalize to sum=1 and persist
    total = sum(seconds) or 1.0
    data = [{'module': t.module, 'weight': f'{s / total:.4f}'} for t, s in zip(tasks, seconds)]
    try:
        weights_f_path = _profile_path()
        weights_f_path.parent.mkdir(parents=True, exist_ok=True)
        weights_f_path.write_text(yaml.safe_dump(data, sort_keys=False, indent=2), encoding='utf-8')
    except Exception:
        pass  # not critical if we cannot write the profile


def run_staged_imports(tasks: List[ImportTask], *,
                       on_message: OnMsg = None, on_progress: OnProg = None,
                       target_namespace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Imports modules in stages, with progress callbacks and persistent weights.

    See `ImportTask` for details on how to specify each import.

    Returns a dict {export_name: object}, where export_name is either:
      - task.as_name (for whole-module alias), else last segment of module name
      - or the alias/name for each entry in task.names (from-imports)

    Parameters
    ----------
    tasks : List[ImportTask]
        List of import tasks to be performed.
    on_message : Optional[Callable[[str], None]], optional
        Callback for status messages, by default None.
    on_progress : Optional[Callable[[int], None]], optional
        Callback for progress updates (0-100), by default None.
    target_namespace : Optional[Dict[str, Any]], optional
        If provided, injects imported names into this dict (e.g. globals()), by default
    """
    expected_weights = _load_weights(tasks)

    exports: Dict[str, Any] = {}
    observed_seconds: List[float] = []
    total_time = 0.0

    def _emit_msg(msg: str):
        if on_message:
            on_message(msg)

    def _emit_progress(percent: float):
        if on_progress:
            on_progress(int(max(0, min(100, round(percent)))))  # clip percentage ([0..100])

    for i, task in enumerate(tasks):
        _emit_msg(task.display_message)

        start_t = time.perf_counter()

        # Import module itself.
        mod = importlib.import_module(task.module)

        # Import attributes or alias module as needed.
        if task.has_attrs:
            for name in task.attrs:
                attr = getattr(mod, name)
                exports[name] = attr
                if target_namespace is not None:
                    target_namespace[name] = attr
        else:
            name = task.export_name
            exports[name] = mod
            if target_namespace is not None:
                target_namespace[name] = mod

        # Measure actual import time for accurate display next time.
        dt = max(EPSILON_DT, time.perf_counter() - start_t)
        observed_seconds.append(dt)

        total_time += expected_weights[i]
        _emit_progress(total_time * 100.0)

    _save_weights(tasks, observed_seconds)
    _emit_progress(100.0)
    return exports
