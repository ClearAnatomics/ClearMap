from __future__ import annotations

from copy import deepcopy
from typing import Tuple, Any, Iterable, Mapping, Dict

from ClearMap.Utils.utilities import DELETE, _REPLACE
from ClearMap.config.config_adjusters.dict_ops import ensure_path_dict
from ClearMap.config.config_adjusters.type_hints import PatchConflictError, KeysPath


def merge_patches(dst: dict, src: dict, *, _path: Tuple[str, ...] = ()) -> dict:
    """
    Combine two patch dicts without applying them.

    Composition rules:
      - DELETE and _REPLACE are strong operations.
      - dict values recurse (merge).
      - scalars overwrite.

    Conflict rules:
      - dst already has _REPLACE and src has *different* _REPLACE => raise.
      - dst already has _REPLACE and src provides a non-sentinel update => raise
        (prevents silent downgrade of authoritative replacement).
    """
    for k, v in src.items():
        path = _path + (k,)
        dv = dst.get(k)

        # DELETE always wins
        if v is DELETE:
            dst[k] = DELETE
            continue
        elif dv is DELETE: # If destination already deleted this key, do not allow resurrection
            raise PatchConflictError(path, dv, v, 'non-DELETE update would resurrect a DELETE')


        # REPLACE: detect conflicting REPLACE-vs-REPLACE
        if isinstance(v, _REPLACE):
            if isinstance(dv, _REPLACE) and dv.payload != v.payload:
                raise PatchConflictError(path, dv, v, 'two different REPLACE operations for the same path')
            dst[k] = v
            continue

        # If destination already asserts REPLACE, absorb dict updates into payload
        if isinstance(dv, _REPLACE):
            if isinstance(v, dict) and isinstance(dv.payload, dict):
                merge_patches(dv.payload, v, _path=path)  # note: path stays same for good errors
                continue
            raise PatchConflictError(path, dv, v, 'non-dict update would downgrade an existing REPLACE')

        # Recurse only into plain dicts
        if isinstance(v, dict) and isinstance(dv, dict):
            merge_patches(dv, v, _path=path)
        else:
            dst[k] = deepcopy(v)

    return dst


def _strip_template_like_keys_from_entry(entry: dict[str, Any]) -> None:
    entry.pop('templates', None)
    for k in list(entry.keys()):
        if isinstance(k, str) and k.endswith('_template'):
            entry.pop(k, None)


def _strip_template_like_keys_from_container(container: dict[str, Any]) -> None:
    container.pop('templates', None)
    for _, v in container.items():
        if isinstance(v, dict):
            _strip_template_like_keys_from_entry(v)

def _stable_order_keys(*, current_keys: Iterable[str], target_keys: Iterable[str]) -> list[str]:
    target_set = set(target_keys)
    ordered = [k for k in current_keys if k in target_set]
    for k in target_keys:
        if k not in ordered:
            ordered.append(k)
    return ordered


def _restrict_entry_to_template_keys(entry: dict[str, Any], template_base: dict[str, Any]) -> dict[str, Any]:
    keep = set(template_base.keys())
    return {k: v for k, v in entry.items() if k in keep}


def _raise_if_rename_map_collides(rename_map: Mapping[str, str]) -> None:
    inv: dict[str, list[str]] = {}
    for old, new in rename_map.items():
        inv.setdefault(new, []).append(old)
    collisions = {new: olds for new, olds in inv.items() if len(olds) > 1}
    if collisions:
        msg = ', '.join(f'{new}<-{olds}' for new, olds in collisions.items())
        raise ValueError(f'renamed_channels has collisions: {msg}')


# FIXME: merge with below using return_items arg
def _iter_patch_paths(p: Any, *, _path: tuple[str, ...] = ()) -> Iterable[tuple[str, ...]]:
    """
    Yield *assignment* paths in a patch dict, including sentinel leaves.
    We treat any non-dict value (including DELETE/_REPLACE/scalar/list) as a leaf assignment.
    """
    if not isinstance(p, dict):
        yield _path
        return
    for k, v in p.items():
        path = _path + (k,)
        if isinstance(v, dict):
            # recurse into dict patches
            yield from _iter_patch_paths(v, _path=path)
        else:
            # leaf assignment (DELETE, _REPLACE, scalar, list, etc.)
            yield path

def iter_patch_items(p: Any, *, _path: tuple[str, ...] = ()):
    """
    Yield (path, value) for leaf assignments in a patch dict.
    A leaf is any non-dict value (including DELETE/_REPLACE/scalar/list).
    """
    if not isinstance(p, dict):
        yield _path, p
        return
    for k, v in p.items():
        path = _path + (k,)
        if isinstance(v, dict):
            yield from iter_patch_items(v, _path=path)
        else:
            yield path, v


def set_patch_at_path(patch: Dict[str, Any], path: KeysPath, value: Any) -> None:
    parent = ensure_path_dict(patch, path[:-1]) if len(path) > 1 else patch
    parent[path[-1]] = value
