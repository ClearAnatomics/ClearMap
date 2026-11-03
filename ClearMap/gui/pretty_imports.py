import ast
import sys
import time
import importlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path

import yaml

from ClearMap.config.early_boot import MachineConfig

CLEARMAP_PATH = Path(__file__).parent.parent


CLEARMAP_PREFIX = 'ClearMap.'
EPSILON_DT = 0.0005  # minimum time to avoid zero-division in progress calculations

OnMsg = Optional[Callable[[str], None]]
OnProg = Optional[Callable[[int], None]]

START_MARKER = '### SLOW IMPORTS ###'
END_MARKER   = '### END SLOW IMPORTS ###'


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


def _module_to_path(module: str) -> Path | None:
    """
    Convert 'ClearMap.foo.bar' → ClearMap/foo/bar.py
    """
    rel = module.split('.')[1:]  # drop 'ClearMap'
    path = CLEARMAP_PATH.joinpath(*rel).with_suffix('.py')
    if path.exists():
        return path
    init_py = CLEARMAP_PATH.joinpath(*rel, '__init__.py')
    if init_py.exists():
        return init_py
    return None


def _parse_regular_import(line: str) -> ImportTask | None:
    mod = line[len('import '):].strip()
    if mod.startswith(CLEARMAP_PREFIX):
        return ImportTask.module_only(mod)
    return None


def _parse_from_import(line: str) -> ImportTask | None:
    rest = line[len('from '):]
    try:
        module, modules_list = rest.split(' import ', 1)
    except ValueError:
        return None
    module = module.strip()
    if module.startswith(CLEARMAP_PREFIX):
        modules_list = modules_list.strip()

        # Remove optional parentheses around imported names: "from x import (a, b, c)"
        if modules_list.startswith('(') and modules_list.endswith(')'):
            modules_list = modules_list[1:-1].strip()

        names = tuple(n.strip() for n in modules_list.split(',') if n.strip())
        if names:
            return ImportTask.from_imports(module, *names)
    return None


def _read_clear_map_imports(py_file: Path) -> List[ImportTask]:
    """
    Read a single .py file and extract column-0 ClearMap imports in order.
    """
    tasks: List[ImportTask] = []
    try:
        acc_line = ''
        for line in py_file.read_text(encoding='utf-8').splitlines():
            line = line.rstrip()
            if line.startswith('import '):
                task = _parse_regular_import(line)
            elif line.startswith('from '):
                if '(' in line and ')' not in line:
                    acc_line += line.strip() + ' '  # accumulate multi-line from-import
                    continue  # skip multi-line from-imports for simplicity
                else:
                    task = _parse_from_import(line)
            elif acc_line:
                acc_line += line.strip() + ' '
                if '(' in acc_line and ')' in acc_line:
                    task = _parse_from_import(acc_line)
                    acc_line = ''  # reset accumulator after processing
                else:
                    continue
            else:
                continue
            if task is not None:
                tasks.append(task)
    except Exception:
        pass  # best-effort only
    return tasks


class _ImportGraph:
    def __init__(self) -> None:
        self._deps_cache: Dict[str, List[str]] = {}

    @staticmethod
    def _resolve_relative(from_module: str, level: int, mod: str | None) -> str | None:
        parts = from_module.split('.')
        if level <= 0:
            return mod
        if level >= len(parts):
            return None
        base = parts[:-level]
        if mod:
            base += mod.split('.')
        return '.'.join(base)

    @staticmethod
    def _candidate_submodule(parent: str, name: str) -> str | None:
        cand = f'{parent}.{name}'
        return cand if _module_to_path(cand) is not None else None

    def deps(self, module: str) -> List[str]:
        if module in self._deps_cache:
            return self._deps_cache[module]

        out: List[str] = []
        py_file = _module_to_path(module)
        if py_file is None:
            self._deps_cache[module] = out
            return out

        try:
            source = py_file.read_text(encoding='utf-8')
            tree = ast.parse(source)
        except Exception:
            self._deps_cache[module] = out
            return out

        for node in getattr(tree, 'body', ()):
            if getattr(node, 'col_offset', 0) != 0:
                continue

            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith(CLEARMAP_PREFIX):
                        out.append(name)

            elif isinstance(node, ast.ImportFrom):
                level = int(getattr(node, 'level', 0) or 0)
                parent = self._resolve_relative(module, level, node.module)

                if not parent or not parent.startswith(CLEARMAP_PREFIX):
                    continue

                out.append(parent)

                for a in node.names:
                    sym = a.name
                    if not sym or sym == '*':
                        continue
                    cand = self._candidate_submodule(parent, sym)
                    if cand is not None:
                        out.append(cand)

        seen: set[str] = set()
        dedup: List[str] = []
        for d in out:
            if d not in seen:
                seen.add(d)
                dedup.append(d)

        self._deps_cache[module] = dedup
        return dedup


def discover_import_tasks(entry_file: Path) -> List[ImportTask]:
    """
    Recursively discover *unprotected* ClearMap imports starting from entry_file.

    Rules:
      - Only column-0 imports
      - Only ClearMap.*
      - Order preserved by discovery
      - Recursive expansion via source files
      - sys.modules short-circuit
    """
    entry_file = Path(entry_file).resolve()

    in_block = False
    tasks: List[ImportTask] = []

    # Parse modules from main file
    acc_line = ''
    for line in entry_file.read_text(encoding='utf-8').splitlines():
        line = line.rstrip()
        task = None
        if line.strip() == START_MARKER:
            in_block = True
            continue
        elif not in_block:  # second to ensure we allow it to become True
            continue
        elif line.strip() == END_MARKER:
            break
        elif line.startswith('import '):
            task = _parse_regular_import(line)
        elif line.startswith('from '):
            if '(' in line and ')' not in line:
                acc_line += line.strip() + ' '  # accumulate multi-line from-import
                continue  # skip multi-line from-imports for simplicity
            else:
                task = _parse_from_import(line)
        elif acc_line:
            acc_line += line.strip() + ' '
            if '(' in acc_line and ')' in acc_line:
                task = _parse_from_import(acc_line)
                acc_line = ''  # reset accumulator after processing
            else:
                continue

        if task is not None and task.module.startswith(CLEARMAP_PREFIX):
            tasks.append(task)

    graph = _ImportGraph()

    seed_modules: List[str] = []
    seen_seed: set[str] = set()
    for t in tasks:
        if t.module not in seen_seed:
            seen_seed.add(t.module)
            seed_modules.append(t.module)

    visited: set[str] = set()
    visiting: set[str] = set()
    ordered: List[str] = []

    def dfs(m: str) -> None:
        if m in visited:
            return
        if m in set(sys.modules):
            visited.add(m)
            return
        if m in visiting:
            return
        visiting.add(m)
        for d in graph.deps(m):
            if d.startswith(CLEARMAP_PREFIX):
                dfs(d)
        visiting.remove(m)
        visited.add(m)
        ordered.append(m)

    for m in seed_modules:
        dfs(m)

    import_tasks: List[ImportTask] = []
    for m in ordered:
        if m not in set(sys.modules):
            import_tasks.append(ImportTask.module_only(m))

    return import_tasks


def _profile_path() -> Path:
    major, minor = MachineConfig._version.split('.')[:2]
    return Path.home() / '.clearmap' / f'.import_profile_v{major}_{minor}.yml'


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
