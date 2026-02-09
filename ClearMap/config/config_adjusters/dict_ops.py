from __future__ import annotations

from copy import deepcopy
from types import NoneType
from typing import Dict, Any, Set, Tuple, Optional

TuplePath = Tuple[str, ...]

def resolve_path(section: str, path: str) -> TuplePath:
    """
    Resolve a dotted path.
    - '.channels'        -> (section, 'channels')
    - '.performance.bin' -> (section, 'performance', 'bin')
    - 'vasculature.bin'  -> ('vasculature', 'bin')
    """
    if not path:
        raise ValueError('Empty path is not allowed.')
    if path.startswith('.'):
        tail = path[1:]
        if not tail:
            return (section,)
        parts = tuple(tail.split('.'))
        _validate_parts(parts, path)
        return (section,) + parts

    parts = tuple(path.split('.'))
    _validate_parts(parts, path)
    return parts


def _validate_parts(parts: tuple[str, ...], path: str):
    """Ensure a path's parts are valid (no empty parts). """
    if any(p == '' for p in parts):
        raise ValueError(f'Invalid path with consecutive dots: {path!r}')


def normalise_dict(current: Any) -> dict[str, Any]:
    return deepcopy(current) if isinstance(current, dict) else {}


def deep_merge_missing(dst: Dict[str, Any], src: Dict[str, Any], *,
                       skip_paths: Optional[Set[Tuple[str, ...]]] = None, _path: Tuple[str, ...] = tuple()) -> None:
    """
    Fill only missing keys in dst from src (recursive), skipping specific subtrees by dotted path.
    Paths are tuples like ('binarization',) or ('performance', 'binarization').
    """
    for k, v in src.items():
        if k == 'templates':
            continue

        path = _path + (k,)  # WARNING: small difference with above
        if skip_paths and path in skip_paths:
                continue

        if k not in dst:
            dst[k] = deepcopy(v)
        elif isinstance(dst[k], dict) and isinstance(v, dict):
            deep_merge_missing(dst[k], v, skip_paths=skip_paths, _path=path)


def merge_section_missing_only(section_cfg: Dict[str, Any], section_defaults: Dict[str, Any], *,
                               skip_paths: Optional[Set[Tuple[str, ...]]] = None) -> bool:
    before = deepcopy(section_cfg)
    if skip_paths:
        deep_merge_missing(section_cfg, section_defaults, skip_paths=skip_paths)
    else:
        deep_merge_missing(section_cfg, section_defaults)
    return section_cfg != before

#
# def merge_container_entries_missing_only(*, cur_map: dict[str, Any], template: dict[str, Any]) -> tuple[dict[str, Any], bool]:
#     changed = False
#     out: dict[str, Any] = {}
#
#     for k, v in (cur_map or {}).items():
#         entry = v if isinstance(v, dict) else {}
#         if entry is not v:
#             changed = True
#
#         before = deepcopy(entry)
#         deep_merge_missing(entry, template)
#         entry = normalize_json_compat(entry)
#         out[k] = entry
#
#         changed = changed or (entry != before)
#
#     return out, changed



def ensure_path_dict(root: dict, path: tuple[str, ...]) -> dict:
    node = root
    for p in path:
        val = node.get(p)
        if not isinstance(val, (dict, NoneType)):
            raise ValueError(f'Cannot ensure path {path}: intermediate {p} is not a dict or missing')
        if val is None:
            node[p] = {}
        node = node[p]
    return node


def get_nested(root: Any, path: tuple[str, ...], default: Any = None) -> Any:
    """
    Safe nested dict traversal. Returns `default` if:
      - any intermediate is not a dict
      - a key is missing
    """
    node = root
    for p in path:
        if not isinstance(node, dict):
            return default
        node = node.get(p, default)
    return node


def normalize_json_compat(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [normalize_json_compat(x) for x in obj]
    if isinstance(obj, list):
        return [normalize_json_compat(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize_json_compat(v) for k, v in obj.items()}
    return obj


def is_under(prefix: tuple[str, ...], path: tuple[str, ...]) -> bool:
    """
    Check if `path` is under `prefix`.

    Parameters
    ----------
    prefix
    path

    Returns
    -------

    """
    return len(path) >= len(prefix) and path[:len(prefix)] == prefix
