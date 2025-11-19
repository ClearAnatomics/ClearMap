from typing import Any, Dict, Iterable, Mapping

from ._schema import AggregatedValidationError


__all__ = ['run_semantic_checks']

SECTIONS_WITH_CHANNELS = (
    'stitching', 'registration', 'cell_map', 'colocalization', 'tract_map'
)


def _sample_channels(config: Mapping[str, Any]) -> set[str]:
    sample = config.get("sample") or {}
    chs = (sample.get("channels") or {})
    return set(chs.keys())


def _channels_in_section(config: Mapping[str, Any], section: str) -> Iterable[str]:
    sec = config.get(section) or {}
    chs = (sec.get("channels") or {})
    return chs.keys()


def _validate_channels_subset(config: Mapping[str, Any], section: str) -> list[str]:
    known = _sample_channels(config)
    names = set(_channels_in_section(config, section))
    unknown = sorted(names - known)
    if unknown:
        return [f"[{section}] channels: unknown {unknown}; not present in sample.channels"]
    return []


def _validate_registration_refs(config: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    known = _sample_channels(config)
    reg = (config.get("registration") or {})
    chs: Dict[str, Any] = (reg.get("channels") or {})
    tokens_align = {"atlas", "self", "other"}
    tokens_moving = {"atlas", "intrinsically_aligned", "self", "other"}
    for ch_name, spec in chs.items():
        spec = spec or {}
        aw = spec.get("align_with")
        mv = spec.get("moving_channel")
        if isinstance(aw, str) and aw not in tokens_align and aw not in known:
            errs.append(f"[registration] channels.{ch_name}.align_with → unknown '{aw}'")
        if isinstance(mv, str) and mv not in tokens_moving and mv not in known:
            errs.append(f"[registration] channels.{ch_name}.moving_channel → unknown '{mv}'")
    return errs


def _validate_stitching_semantics(config: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    known = _sample_channels(config)
    st = (config.get("stitching") or {})
    chs: Dict[str, Any] = (st.get("channels") or {})
    for name, spec in chs.items():
        spec = spec or {}
        target = spec.get("layout_channel")

        if target == 'undefined' or target is None:
            continue  # skip validation for undefined layout_channel

        if not isinstance(target, str) or target not in known:
            errs.append(f"[stitching] channels.{name}.layout_channel must be an existing channel (got {target!r})")
            continue
        is_provider = (target == name)  # self-referential -> this channel stitches itself
        if is_provider:
            for req in ("rigid", "wobbly"):
                if req not in spec:
                    errs.append(f"[stitching] channels.{name} is a provider but missing '{req}'")
        else:
            for forb in ("rigid", "wobbly"):
                if forb in spec:
                    errs.append(f"[stitching] channels.{name} is derived; remove '{forb}' (provider={target})")
        # numeric relations
        rigid = spec.get("rigid") or {}
        pt = rigid.get("projection_thickness")
        ox = rigid.get("overlap_x")
        oy = rigid.get("overlap_y")
        bl = rigid.get("background_level")
        bp = rigid.get("background_pixels")
        if isinstance(bl, (int, float)) and bl < 0:
            errs.append(f"[stitching] channels.{name}.rigid.background_level must be ≥ 0")
        if isinstance(bp, int) and bp < 0:
            errs.append(f"[stitching] channels.{name}.rigid.background_pixels must be ≥ 0")
        def _is_num(x): return isinstance(x, (int, float)) and not isinstance(x, bool)
        def _is_int(x): return isinstance(x, int) and not isinstance(x, bool)
        if isinstance(pt, (list, tuple)) and len(pt) >= 2 and _is_num(pt[0]) and _is_num(pt[1]) and _is_int(ox) and _is_int(oy):
            if pt[0] < ox:
                errs.append(f"[stitching] channels.{name}.rigid.projection_thickness[0] ({pt[0]}) < overlap_x ({ox})")
            if pt[1] < oy:
                errs.append(f"[stitching] channels.{name}.rigid.projection_thickness[1] ({pt[1]}) < overlap_y ({oy})")
    return errs


def _validate_batch_processing(config: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    bp = (config.get("batch_processing") or {})
    groups = (bp.get("groups") or {})
    known = set(groups.keys())
    comps = bp.get("comparisons") or []
    seen: set[tuple[str, str]] = set()
    for idx, pair in enumerate(comps):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            errs.append(f"[batch_processing] comparisons[{idx}] must be a pair [group_1, group_2]")
            continue
        g1, g2 = pair
        if g1 not in known:
            errs.append(f"[batch_processing] comparisons[{idx}][0] '{g1}' is not a defined group")
        if g2 not in known:
            errs.append(f"[batch_processing] comparisons[{idx}][1] '{g2}' is not a defined group")
        key = tuple(sorted((g1, g2)))
        if key in seen:
            errs.append(f"[batch_processing] comparisons[{idx}] duplicates {key[0]} vs {key[1]}")
        seen.add(key)
    return errs


def _validate_machine_cross(config: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    m = (config.get("machine") or {})
    mn = m.get("detection_chunk_size_min")
    mx = m.get("detection_chunk_size_max")
    if isinstance(mn, int) and isinstance(mx, int) and mn > mx:
        errs.append("[machine] detection_chunk_size_max should be ≥ detection_chunk_size_min")
    return errs


def run_semantic_checks(config: Mapping[str, Any], _sv=None) -> None:
    messages: list[str] = []
    if config.get("sample"):
        for sec in SECTIONS_WITH_CHANNELS:
            if config.get(sec):
                messages.extend(_validate_channels_subset(config, sec))
        if config.get("registration"):
            messages.extend(_validate_registration_refs(config))
        if config.get("stitching"):
            messages.extend(_validate_stitching_semantics(config))
    if config.get("batch_processing"):
        messages.extend(_validate_batch_processing(config))
    if config.get("machine"):
        messages.extend(_validate_machine_cross(config))
    if messages:
        raise AggregatedValidationError(messages)
