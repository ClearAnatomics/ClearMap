from __future__ import annotations

import functools
from copy import deepcopy
from operator import getitem
from types import MappingProxyType, NoneType
from typing import Dict, Any, Set, Tuple, Optional

from configobj import ConfigObj

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


def del_item_recursive(d: dict, keys: list) -> None:
    """Delete a nested key from a dict. No-op if path does not exist."""
    for key in keys[:-1]:
        d = d.get(key)
        if not isinstance(d, dict):
            return
    d.pop(keys[-1], None)


def get_item_recursive(container, keys):
    try:
        return functools.reduce(getitem, keys, container)
    except (TypeError, KeyError) as err:
        err_type = type(err).__name__
        err_msg = f'{err_type} attempting to read "{keys}"'
        if isinstance(container, ConfigObj):
            err_msg += f' path: {container.filename}'
        print(f'{err_msg} from "{container}"')
        raise err


def try_get_item_recursive(container, keys, default=Ellipsis):
    try:
        return get_item_recursive(container, keys)
    except (TypeError, KeyError):
        if default is Ellipsis:
            raise
        return default


def has_item_recursive(container, keys) -> bool:
    try:
        get_item_recursive(container, keys)
        return True
    except (TypeError, KeyError):
        return False


def set_item_recursive(dictionary, keys_list, val, fix_missing_keys=True):
    def add_keys(d, keys):
        if not keys:
            return
        if keys[0] not in d.keys():
            d[keys[0]] = {}
        if keys[1:]:  # if keys left
            if not isinstance(d[keys[0]], dict):
                # Avoid creating a sub-dict if there is already a non-dict value there
                raise TypeError(f"Non-dict at {keys[0]} while descending {keys}")
            add_keys(d[keys[0]], keys[1:])

    if fix_missing_keys:
        add_keys(dictionary, keys_list[:-1])  # Fix missing keys recursively
    else: # ensure the path exists and is all dicts
        cur = dictionary
        for k in keys_list[:-1]:
            cur = cur[k]  # KeyError if missing
            if not isinstance(cur, dict):
                raise TypeError(f"Non-dict at {k} while descending {keys_list}")

    # Since we ensured it exists and is a dict, write to var will modify in place
    get_item_recursive(dictionary, keys_list[:-1])[keys_list[-1]] = val
    return dictionary  # Return the modified dictionary for chaining


def deep_freeze(obj) -> object | MappingProxyType:
    """
    Recursively make a data structure immutable (freeze it).
    Only works for dict, list, set, tuple and scalars (str/int/float/None/...).

    .. note::
        This is faster than deepcopy and more explicit because
        if we try to modify a frozen object, we get an error immediately.

    Parameters
    ----------
    obj: object
        Typically a nested dict like a config

    Returns
    -------
    object | MappingProxyType
        An immutable version of the input object
    """
    if isinstance(obj, dict):
        # freeze children first, then wrap the dict so callers cannot mutate it
        frozen = {k: deep_freeze(v) for k, v in obj.items()}
        return MappingProxyType(frozen)
    elif isinstance(obj, list):
        return tuple(deep_freeze(x) for x in obj)
    elif isinstance(obj, set):
        return frozenset(deep_freeze(x) for x in obj)
    elif isinstance(obj, tuple):
        return tuple(deep_freeze(x) for x in obj)
    # scalars (str/int/float/None/...) pass through because in python they are already immutable
    return obj


def ensure_list(obj) -> list:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def dedupe_preserve_order(seq, key=lambda x: x):
    seen = set()
    out = []
    for x in seq:
        k = key(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out
