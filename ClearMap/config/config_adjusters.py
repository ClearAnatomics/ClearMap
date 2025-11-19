"""
Adjusters are pure functions (view, sm) -> patch
that can inspect the current config view and sample manager
and return a patch to be merged into the working config.
They are defined in the ADJUSTERS DEFINITION section and registered
with the @adjuster decorator
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Protocol, Mapping, Any, Dict, List, Optional, Set, Tuple, Iterable, Callable

from ClearMap.Utils.utilities import deep_merge, _dedupe_preserve_order, _ensure_list, DELETE, REPLACE, _REPLACE
from ClearMap.config.defaults_provider import DefaultsProvider, get_defaults_provider

# ################################# HELPERS #################################

ConfigView = Mapping[str, Any]
ConfigPatch = Dict[str, Any]
ConfigKeys = tuple[str, ...]
ConfigKeysLike = str | ConfigKeys


DEFAULTS_PROVIDER: Optional[DefaultsProvider] = get_defaults_provider()


def set_defaults_provider(provider) -> None:
    global DEFAULTS_PROVIDER
    DEFAULTS_PROVIDER = provider


def merge_patches(dst: dict, src: dict) -> dict:
    """
    Combine two patch dicts without applying them.
    - Preserve sentinels: DELETE and _REPLACE must flow through untouched.
    - Recurse only into plain dicts (not into _REPLACE payloads).
    """
    for k, v in src.items():
        # Sentinels: copy as-is, do not recurse, do not unwrap
        if v is DELETE or isinstance(v, _REPLACE):
            dst[k] = v
        else: # Both sides dicts and current dst not a sentinel: recurse
            dv = dst.get(k)
            if isinstance(v, dict) and isinstance(dv, dict) and not isinstance(dv, _REPLACE):
                merge_patches(dv, v)
            else:
                dst[k] = deepcopy(v)
    return dst


def to_config_keys(x: ConfigKeysLike) -> ConfigKeys:
    return x if isinstance(x, tuple) else tuple(x.split("."))


def to_config_keys_list(xs: Optional[Iterable[ConfigKeysLike]]) -> Optional[List[ConfigKeys]]:
    return None if xs is None else [to_config_keys(x) for x in xs]


# FIXME: no usage. and does not seem to use $rename
def _rename_channel_key(sec_dict: Dict[str, Any], old: str, new: str) -> None:
    """In-place: move channels[old] -> channels[new] if present."""
    if not isinstance(sec_dict, dict):
        return
    chs = sec_dict.get('channels') or {}
    if old in chs:
        ent = chs.pop(old)
        chs[new] = ent


def _deep_merge_missing(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Fill only missing keys in dst from src (recursive)."""
    for k, v in src.items():
        if k not in dst:
            dst[k] = deepcopy(v)
        elif isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_merge_missing(dst[k], v)


def _swap_placeholders(obj, *, channel: str, reference: Optional[str] = None) -> Any:
    """Recursively substitute ${channel} and ${reference} in strings within obj."""
    def sub_one(s: str) -> str:
        out = s.replace('${channel}', channel)
        if reference is not None:
            out = out.replace('${reference}', reference)
        return out
    if isinstance(obj, str):
        return sub_one(obj)
    if isinstance(obj, list):
        return [_swap_placeholders(x, channel=channel, reference=reference) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_swap_placeholders(x, channel=channel, reference=reference) for x in obj)
    if isinstance(obj, dict):
        return {k: _swap_placeholders(v, channel=channel, reference=reference) for k, v in obj.items()}
    return obj

################################# ENGINE ##################################

class SampleManagerProtocol(Protocol):
    @property
    def channels(self) -> List[str]: ...
    @property
    def stitchable_channels(self) -> List[str]: ...
    @property
    def renamed_channels(self) -> Dict[str, str]: ...
    def get_channels_by_pipeline(self, pipeline: str, as_list: bool = False) -> List[str]: ...
    def data_type(self, channel: str) -> Optional[str]: ...
    @property
    def alignment_reference_channel(self) -> Optional[str]: ...
    @property
    def is_colocalization_compatible(self) -> bool: ...
    @property
    def channels_to_detect(self) -> List[str]: ...


class Phase(str, Enum):
    PRE_VALIDATE  = 'pre-validate'
    POST_VALIDATE = 'post-validate'
    PRE_COMMIT    = 'pre-commit'
    POST_COMMIT   = 'post-commit'

class Step(str, Enum):
    # Creation/materialization steps
    APPLY_RENAMES             = 'apply-renames'         # sample/channel_x → renamed channels
    CREATE_PIPELINE_SECTIONS  = 'create-pipeline-sections'  # pipeline-level blocks
    CREATE_CHANNEL_SECTIONS   = 'create-channel-sections'  # example/channel_x → real channels
    CREATE_CHANNELS_PRUNE     = 'create-channels-prune'  # drop removed channels
    CREATE_CHANNELS_RECONCILE = 'create-channels-reconcile'  # add missing channels
    # Population steps
    POPULATE_DEFAULTS         = 'populate-defaults'  # fill missing keys with defaults
    ADJUST                    = 'adjust'            # custom adjustments
    # Normalization steps
    NORMALIZE_SHAPES          = 'normalize-shapes'  # scalars↔vectors, sentinels, etc.


PHASES_ORDER: Dict[Phase, Tuple[Step, ...]] = {
    Phase.PRE_VALIDATE: (
        Step.APPLY_RENAMES,
        Step.CREATE_PIPELINE_SECTIONS,
        Step.CREATE_CHANNEL_SECTIONS,
        Step.CREATE_CHANNELS_PRUNE,
        Step.CREATE_CHANNELS_RECONCILE,
        Step.POPULATE_DEFAULTS,
        Step.ADJUST,
        Step.NORMALIZE_SHAPES,
    ),
    Phase.POST_VALIDATE: (),
    Phase.PRE_COMMIT:    (),
    Phase.POST_COMMIT:   (),
}


AdjusterFn = Callable[[ConfigView, SampleManagerProtocol], ConfigPatch]

@dataclass(frozen=True)
class AdjusterSpec:
    name: str
    fn: AdjusterFn
    step: Step
    phase: Phase
    pipelines: Optional[Set[str]]             # None => all
    keys: Optional[Tuple[ConfigKeys, ...]]    # run only if these prefixes match changed keys
    order: int = 100  # lower runs first. Alphabetically if identical.

_REGISTRY: List[AdjusterSpec] = []


def adjuster(*, step: Step, phase: Phase = Phase.PRE_VALIDATE,
             pipelines: Optional[Iterable[str]] = None,
             keys: Optional[Iterable[ConfigKeysLike]] = None,
             order: int = 100) -> Callable[[AdjusterFn], AdjusterFn]:
    """
    Decorator for adjusters.
    Decorate a pure, idempotent (view, sm) -> patch function.
    """
    def _wrap(fn: AdjusterFn) -> AdjusterFn:
        _REGISTRY.append(
            AdjusterSpec(name=fn.__name__, fn=fn, step=step, phase=phase,
                pipelines=set(pipelines) if pipelines else None,
                keys=tuple(to_config_keys(k) for k in keys) if keys else None,
                order=order))
        return fn
    return _wrap


def _config_keys_overlap(changed_keys: Optional[Iterable[ConfigKeysLike]],
                         watched_prefixes: Optional[Tuple[ConfigKeysLike, ...]]) -> bool:
    """
    True if `changed` touches any `keys` prefix.
    Example:
      changed=["registration.channels.Ch488.align_with"]
      keys=("registration.channels",)  -> True
      keys=("stitching.channels",)     -> False
    If `keys` is None or `changed` is None/empty, return True (do not filter).

    .. note::

        This is made to know which adjusters to run based on config
        parts that have changed in the current config patch.
        This way if one adjuster alters part of the config that
        other adjusters depend on, those will be run too.
    """
    changed_t = to_config_keys_list(changed_keys)
    watched_prefixes_t = to_config_keys_list(watched_prefixes)
    if not watched_prefixes_t or not changed_t:
        return True
    for changed_k in changed_t:
        for watched_k in watched_prefixes_t:
            if len(changed_k) >= len(watched_k) and changed_k[:len(watched_k)] == watched_k:
                return True
    return False


def run_adjusters(*,
                  view: ConfigView,
                  sample_manager: SampleManagerProtocol,  # REQUIRED
                  phase: Phase = Phase.PRE_VALIDATE,
                  pipelines: Optional[Iterable[str]] = None,
                  changed_keys: Optional[Iterable[ConfigKeys]] = None) -> ConfigPatch:
    """
    Select adjusters matching the phase/pipelines/keys and run them
    in the order defined by PHASES_ORDER[phase]. Return a merged patch.
    """
    steps_order = PHASES_ORDER.get(phase, ())
    if not steps_order:
        return {}

    pipes_set = set(pipelines) if pipelines else None
    patch: ConfigPatch = {}
    working_view = deepcopy(dict(view))  # mutable to update as we go

    # Group specs by step (within this phase), then follow the plan
    by_step: Dict[Step, List[AdjusterSpec]] = {s: [] for s in steps_order}
    for spec in _REGISTRY:
        if spec.phase != phase:
            continue
        if spec.step not in steps_order:
            continue
        if pipes_set and spec.pipelines and spec.pipelines.isdisjoint(pipes_set):
            continue
        if not _config_keys_overlap(changed_keys, spec.keys):
            continue
        by_step[spec.step].append(spec)

    for step in steps_order:
        step_patch: ConfigPatch = {}
        # Sort by order, then name
        for spec in sorted(by_step.get(step, []), key=lambda s: (s.order, s.name)):
            step_res = spec.fn(working_view, sample_manager) or {}
            if step_res:
                step_patch = merge_patches(step_patch, step_res)  # accumulate the step’s patch
                working_view = deep_merge(working_view, step_res)  # ← crucial: update the view for later adjusters

        if step_patch:
            merge_patches(patch, step_patch)

    return patch


# ################################# ADJUSTERS DEFINITION #################################


@adjuster(step=Step.POPULATE_DEFAULTS,
          phase=Phase.PRE_VALIDATE,
          pipelines=None,
          keys=('sample.channels',))
def populate_sample_channel_defaults(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    For any sample.channels.<name>, fill missing keys from defaults/sample.yml:channel_template.
    Does not overwrite existing user values. Also normalizes tuple→list for orientation/resolution.
    """
    sample = deepcopy(view.get('sample') or {})
    channels = sample.get('channels') or {}
    if not channels:
        return {}

    channel_template = DEFAULTS_PROVIDER.get('sample')['channel_template']

    changed = False
    for ch_name, cfg in channels.items():
        if not isinstance(cfg, dict):
            cfg = {}
            channels[ch_name] = cfg
            changed = True
        before = deepcopy(cfg)
        _deep_merge_missing(cfg, channel_template)
        # normalize tuple->list
        if isinstance(cfg.get('orientation'), tuple):
            cfg['orientation'] = list(cfg['orientation'])
        if isinstance(cfg.get('resolution'), tuple):
            cfg['resolution'] = list(cfg['resolution'])
        if cfg != before:
            changed = True

    if changed:
        sample['channels'] = channels
        return {'sample': sample}
    return {}


# ############ Global rename triggered by SampleManager ############
@adjuster(step=Step.APPLY_RENAMES, phase=Phase.PRE_VALIDATE,
          pipelines=None, keys=None)
def apply_channel_renames(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Rename section channels (all but vasculature because structure is different)

    Parameters
    ----------
    view: ConfigView
        Current config view
    sm: SampleManagerProtocol
        Sample manager with renamed_channels mapping

    Returns
    -------
    ConfigPatch
        Patch with renamed channels applied
    """
    if not sm.renamed_channels:
        return {}

    patch: ConfigPatch = {}

    def move_channels(sec_name: str):
        sec = deepcopy(view.get(sec_name) or {})
        chs = deepcopy(sec.get('channels') or {})
        changed = False
        for old, new in sm.renamed_channels.items():
            if old in chs and new not in chs:
                chs[new] = chs.pop(old)
                changed = True
        if changed:
            sec['channels'] = chs
            patch[sec_name] = sec

    # FIXME: extract list of sections to rename elsewhere
    for section in ('sample', 'stitching', 'registration', 'cell_map', 'tract_map'):  # Vasculature has its own function
        move_channels(section)
    return patch


@adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
          phase=Phase.PRE_VALIDATE,
          pipelines=('stitching',),
          keys=('sample.channels', 'stitching.channels'))
def materialize_stitching_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Materialize (build) missing channels in stitching.channels.
    Build stitching.channels for all new sample channels using defaults/.../stitching.yml:templates.
    - Channels that serve as own layout use 'layout' template (has rigid/wobbly).
    - Others use 'derived' template (external layout).
    - ${channel} and ${reference} placeholders are expanded.
    """
    if not sm.stitchable_channels:
        return {}

    stitching_config = deepcopy(view.get('stitching') or {})
    existing_chan_cfg = deepcopy((stitching_config.get('channels') or {}))

    # Only act on channels that are absent or empty dicts
    channels_to_create = [ch for ch in sm.stitchable_channels
                          if not existing_chan_cfg.get(ch)]
    if not channels_to_create:
        return {}

    templates = DEFAULTS_PROVIDER.get('stitching')['templates']

    # If an existing layout channel is already present, prefer it as default root
    existing_layouts = [ch for ch, cfg in existing_chan_cfg.items() if (cfg or {}).get('layout_channel') == ch]
    default_root = existing_layouts[0] if existing_layouts else None

    for ch in channels_to_create:
        if default_root is None: # No root -> we create and becomes the layout/root
            existing_chan_cfg[ch] = _swap_placeholders(templates['layout'], channel=ch,
                                                       reference=ch)
            default_root = ch  # Update default root for subsequent channels
        else:  # Any other derive from root
            existing_chan_cfg[ch] = _swap_placeholders(templates['derived'], channel=ch,
                                                       reference=default_root)

    stitching_config['channels'] = REPLACE(existing_chan_cfg)
    return {'stitching': stitching_config}


@adjuster(step=Step.CREATE_CHANNELS_PRUNE,
          phase=Phase.PRE_VALIDATE,
          pipelines=('stitching',),
          keys=('sample.channels', 'stitching.channels'))
def prune_stitching_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Ensure stitching.channels keys are a subset of sample channels.
    Drops any stitching.channels.<ch> where ch not in sm.channels.
    (Adds nothing; creation is handled by your materializer.)
    """
    st = deepcopy(view.get('stitching') or {})
    chs = deepcopy(st.get('channels') or {})
    if not chs:
        return {}

    pruned = {k: v for k, v in chs.items() if k in sm.stitchable_channels}
    if pruned == chs:  # No change -> no patch
        return {}
    return {'stitching': {'channels': REPLACE(pruned)}}



@adjuster(step=Step.ADJUST,
          phase=Phase.PRE_VALIDATE,
          pipelines=('stitching',),
          keys=('stitching.channels',))
def adjust_stitching_channel_roles(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    ADJUSTER:
    - Ensure at least one layout channel exists.
    - Ensure every derived channel references an existing *layout* channel (single-hop check).
    - Apply role templates (layout/derived), expand placeholders, preserve overrides, strip opposite-role keys.
    - Leave deep chain resolution & cycle detection to the stitching_processor.
    """
    st = deepcopy(view.get('stitching') or {})
    channels = deepcopy(st.get('channels') or {})
    if not channels:
        return {}

    templates = DEFAULTS_PROVIDER.get('stitching')['templates']

    sorted_channels = list(channels)
    index_map = {k: i for i, k in enumerate(sorted_channels)}

    layout_channels_set = {ch for ch, cfg in channels.items() if (cfg or {}).get('layout_channel', ch) == ch}
    if not layout_channels_set:  # If None, set the first channel as layout
        root = sorted_channels[0]
        channels[root] = {**(channels.get(root) or {}), 'layout_channel': root}
        layout_channels_set.add(root)

    # Pick a stable root for fixes
    root = min(layout_channels_set, key=index_map.get)

    layout_keys = set(templates['layout'].keys())
    derived_keys = set(templates['derived'].keys())
    layout_only = layout_keys - derived_keys
    derived_only = derived_keys - layout_keys

    # Rebuild with role templates
    updated = {}
    for ch in sorted_channels:
        cfg = channels.get(ch) or {}
        target = cfg.get('layout_channel', ch)

        if target == ch:  # Layout channel
            base = _swap_placeholders(templates['layout'], channel=ch, reference=ch)
            to_strip = derived_only
        else:  # Derived → point to an existing layout or default to root
            if target not in channels or target not in layout_channels_set:
                target = root
            base = _swap_placeholders(templates['derived'], channel=ch, reference=target)
            to_strip = layout_only
        merged = deep_merge(base, cfg)

        for k in to_strip:
            merged.pop(k, None)
        updated[ch] = merged

    if updated == channels:
        return {}

    st['channels'] = REPLACE(updated)
    return {'stitching': st}


def _split_compound(compound_channel):
    if isinstance(compound_channel, str):
        if '-' not in compound_channel:
            raise ValueError(f'Channel "{compound_channel}" is not a compound A-B name.')
        elif compound_channel.count('-') > 1:
            raise NotImplementedError(f'Channel "{compound_channel}" has multiple "-" characters; cannot split.')
    return str(compound_channel).split('-', 1)


def _canonical_pair_key(pair: str | tuple[str, str]) -> str:
    """
    Return canonical 'A-B' (alphabetically sorted) from 'A-B' or (A,B).
    Assumes '-' is forbidden in atomic names.
    """
    if isinstance(pair, tuple):
        a, b = pair
    else:
        a, b = _split_compound(pair)
    a_, b_ = sorted((a, b))
    return f'{a_}-{b_}'


def _get_compound_key_candidate_ancestors(new_key: str, rename_map: dict) -> list[str]:
    """
    For a desired canonical 'A2-B', generate possible prior keys like 'A-B', 'A2-C', 'B-C' ...
    excluding self-pairs (e.g. 'A-A') and non canonical forms (e.g. 'B-A').

    Usage
    -----
    >>> _get_compound_key_candidate_ancestors('Ch561-Ch640', {'Ch488': 'Ch561', 'Ch561': 'Ch640'})
    ['Ch488-Ch561', 'Ch488-Ch640']
    >>> _get_compound_key_candidate_ancestors('Ch561-Ch640', {})
    []
    >>> _get_compound_key_candidate_ancestors('Ch561-Ch640', {'Ch488': 'Ch561'})
    ['Ch488-Ch561', 'Ch488-Ch640']
    """
    if not rename_map:
        return []

    # Ensure we treat the input as canonical
    a2, b2 = _split_compound(_canonical_pair_key(new_key))
    olds_a = [old for (old, new) in rename_map.items() if new == a2]
    olds_b = [old for (old, new) in rename_map.items() if new == b2]

    candidates: set[str] = set()
    new_key_canonical = _canonical_pair_key((a2, b2))

    # Pair old A with current B and with current/old A/B side counterparts
    for oa in olds_a:
        if oa != b2:
            candidates.add(_canonical_pair_key((oa, b2)))
        if oa != a2:
            candidates.add(_canonical_pair_key((oa, a2)))
        for ob in olds_b:
            if oa != ob:
                candidates.add(_canonical_pair_key((oa, ob)))

    # Pair old B with current A and with current/old B/A side counterparts
    for ob in olds_b:
        if ob != a2:
            candidates.add(_canonical_pair_key((a2, ob)))
        if ob != b2:
            candidates.add(_canonical_pair_key((b2, ob)))

    # Exclude the new key itself if it slipped in
    candidates.discard(new_key_canonical)

    return sorted(candidates)


def _normalize_compound_config(current: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    current = current or {}
    normalized: Dict[str, Any] = {}
    moved = False
    for k, v in current.items():
        try:
            nk = _canonical_pair_key(k)
        except Exception:
            nk = k  # keep unknown shapes as-is (will be pruned if not in target)
        if nk != k:
            moved = True
        normalized[nk] = v
    return normalized, moved


def _prune_compound_members(current: dict[str, Any], sample_set: Set[str]) -> dict[str, Any]:
    current = current or {}
    def members_ok(key: str) -> bool:
        try:
            return all([atomic in sample_set for atomic in _split_compound(key)])
        except Exception:
            return False
    return {k: v for k, v in current.items() if members_ok(k)}


def _migrate_compound_entry(
    *,
    current_cfg: dict[str, Any],
    new_compound_chan: str,
    rename_map: Optional[dict[str, str]],
) -> tuple[Optional[dict | Any], bool]:
    """
    Try to migrate overrides for compound key `new_compound_chan` from candidate ancestor keys
    derived via `rename_map` (old_atom -> new_atom). Payloads are carried verbatim.
    Returns (carried_cfg_or_None, carried_flag).
    """

    if not rename_map:
        return None, False

    candidates = _get_compound_key_candidate_ancestors(new_compound_chan, rename_map)

    # Deduplicate while preserving order
    ordered_candidates = []
    for k in candidates:
        if k not in ordered_candidates:
            ordered_candidates.append(k)

    for candidate_k in ordered_candidates:
        if candidate_k in current_cfg:
            return current_cfg[candidate_k], True

    return None, False


def _reconcile_section_channels(*, section_name: str, view: ConfigView,
                                target_channels: list[str],
                                build_entry: Callable[[str], Dict[str, Any]],
                                preserve_existing: bool = True,
                                compound: bool = False,
                                # Map of old_atomic_name -> new_atomic_name (for rename migration on compound keys)
                                rename_map: Optional[Dict[str, str]] = None,
                                # Optional: provide sample_set to drop compound entries that reference missing atoms
                                sample_set: Optional[Set[str]] = None) -> ConfigPatch:
    """
    Make <section>.channels match `target_channels` (add missing, keep overrides, drop removed).
    If compound=True, we:
      - normalize existing keys to canonical 'A-B'
      - migrate overrides across renames using `rename_map`
      - optionally drop entries whose atoms are not all in `sample_set`

    Parameters
    ----------
    section_name: str
        The section to modify (e.g. 'registration')
    view: ConfigView
        Current config view
    target_channels: list[str]
        Desired list of channel keys. Those can be:
        - atomic channel names (e.g. 'Ch488')
        - compound names (e.g. 'Ch488-Ch561') if compound=True
    build_entry: Callable[[str], Dict[str, Any]]
        Function to build a default entry for a given channel key.
    preserve_existing: bool
        If True, keep existing per-channel dicts as-is.
        If False, rebuild all entries via build_entry().
    compound: bool
        If True, treat channel keys as compound 'A-B' pairs.
    rename_map: Optional[Dict[str, str]]
        If compound=True, map of old_atomic_name -> new_atomic_name to help migrate overrides.
    sample_set: Optional[Set[str]]
        If compound=True, optional set of valid atomic channel names.
    """
    section = deepcopy(view.get(section_name) or {})
    current_cfg = deepcopy(section.get('channels') or {})

    moved = False
    if compound:
        current_cfg, moved = _normalize_compound_config(current_cfg)

    # # prune invalid compound members (atomic channels not in sample_set)
    # if compound and sample_set is not None:
    #     current_cfg = _prune_compound_members(current_cfg, sample_set)

    new_channels_sub_section: Dict[str, Any] = {}
    changed = moved or (set(current_cfg.keys()) != set(target_channels))
    for chan in target_channels:
        # keep existing entry as-is
        if preserve_existing and chan in current_cfg and current_cfg[chan]:
            new_channels_sub_section[chan] = current_cfg[chan]
        else:
            carried, carried_cfg = False, None
            if compound and rename_map: # try migrate (compound only)
                carried_cfg, carried = _migrate_compound_entry(current_cfg=current_cfg,
                                                               new_compound_chan=chan,
                                                               rename_map=rename_map)
            base = build_entry(chan)
            if compound and carried:
                merged = deep_merge(deepcopy(base), deepcopy(carried_cfg or {}))
                new_channels_sub_section[chan] = merged
                changed = True
            else:  # default to creating entry
                new_channels_sub_section[chan] = base
                if (not preserve_existing) or (chan not in current_cfg) or (not current_cfg[chan]):
                    changed = True

    if not changed and new_channels_sub_section == current_cfg:
        return {}

    return {section_name: {'channels': REPLACE(new_channels_sub_section)}}


@adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
          phase=Phase.PRE_VALIDATE,
          pipelines=('registration',),
          keys=('sample.channels', 'registration.channels'))
def reconcile_registration_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Build registration.channels for all sample channels using defaults/registration.yml:templates.
    - Channels with data_type == 'autofluorescence' use 'autofluorescence' template.
    - Others use 'regular' template with ${reference} = alignment reference channel.
    Also expands ${channel}/${reference} placeholders.
    """
    sample_channels = sm.channels
    if not sample_channels:
        return {}

    # alignment reference (for regular channels)
    ref = sm.alignment_reference_channel or (sample_channels[0] if sample_channels else None)

    # fetch templates from defaults
    templates = DEFAULTS_PROVIDER.get('registration')['templates']

    def build_entry(ch: str) -> Dict[str, Any]:
        if sm.data_type(ch) == 'autofluorescence':
            base = templates['autofluorescence']
            reference = 'atlas'  # template expects atlas fixed strings
        else:
            base = templates['regular']
            reference = ref
        return _swap_placeholders(base, channel=ch, reference=reference)

    return _reconcile_section_channels( section_name='registration', view=view,
                                        target_channels=sample_channels,
                                        build_entry=build_entry, preserve_existing=True)


# FIXME: add collocation_compatible=True if n> =2
@adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
          phase=Phase.PRE_VALIDATE,
          pipelines=('cell_map',),
          keys=('sample.channels','cell_map.channels'))
def reconcile_cell_map_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    cell_map_channels = sm.get_channels_by_pipeline('CellMap', as_list=True)
    if not cell_map_channels:
        return {}

    template_channel = DEFAULTS_PROVIDER.get('cell_map')['templates']['channel']

    def build_entry(ch: str) -> Dict[str, Any]:
        return _swap_placeholders(template_channel, channel=ch)

    return _reconcile_section_channels(section_name='cell_map', view=view,
                                       target_channels=cell_map_channels,
                                       build_entry=build_entry, preserve_existing=True)


@adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
          phase=Phase.PRE_VALIDATE,
          pipelines=('tract_map',),
          keys=('sample.channels','tract_map.channels'))
def reconcile_tract_map_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Keep tract_map.channels aligned with sample channels using defaults/tract_map.yml:template.channel
    (add missing, keep overrides, drop removed).
    """
    tract_channels = sm.get_channels_by_pipeline('TractMap', as_list=True)
    if not tract_channels:
        return {}

    template_channel = DEFAULTS_PROVIDER.get('tract_map')['templates']['channel']

    def build_entry(ch: str) -> Dict[str, Any]:
        return _swap_placeholders(template_channel, channel=ch)

    return _reconcile_section_channels(section_name='tract_map', view=view,
                                       target_channels=tract_channels,
                                       build_entry=build_entry, preserve_existing=True)


@adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
          phase=Phase.PRE_VALIDATE,
          pipelines=('colocalization',),
          keys=('sample.channels', 'colocalization.channels'))
def reconcile_colocalization_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Pairs = all 2-combinations of sm.channels_to_detect (when sm.is_colocalization_compatible).
    Canonical keys 'A-B'. Preserves overrides, migrates on rename, prunes removed.
    """
    if not sm.is_colocalization_compatible:
        return {}

    if len(sm.channels_to_detect) < 2:  # TODO: remove file altogether ?
        # nothing to keep → pop section if existed
        col = deepcopy(view.get('colocalization') or {})
        if not (col.get('channels') or {}):
            return {}
        col['channels'] = {}
        return {'colocalization': col}

    # Target canonical keys from combinations
    detect_src = _dedupe_preserve_order(sm.channels_to_detect)
    target_keys = [_canonical_pair_key((a, b)) for a, b in combinations(detect_src, 2)]

    template_channel = DEFAULTS_PROVIDER.get('colocalization')['templates']['channel']
    sample_set = set(sm.channels)

    def build_entry(key: str) -> Dict[str, Any]:
        base = _swap_placeholders(template_channel, channel=key)
        return {**base, 'channels': list(_split_compound(key))}

    return _reconcile_section_channels(
        section_name='colocalization', view=view,
        target_channels=target_keys, build_entry=build_entry,
        preserve_existing=True, compound=True,
        rename_map=sm.renamed_channels, sample_set=sample_set)


@adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
          phase=Phase.PRE_VALIDATE,
          pipelines=('vasculature',),
          keys=('sample.channels', 'vasculature.binarization'))
def reconcile_vasculature(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    vasculature.binarization has channel entries directly under the same dict.
     'combined' is a special channel and must always exist.
    """
    vasc_channels = sm.get_channels_by_pipeline('TubeMap', as_list=True)

    defaults = DEFAULTS_PROVIDER.get('vasculature')
    bin_defaults = defaults['binarization']
    vessels_tpl  = bin_defaults['vessels']
    large_vessels_tpl = bin_defaults['large_vessels']   # reused for veins
    combined_tpl = bin_defaults['combined']

    section = deepcopy(view.get('vasculature') or {})
    current_cfg = deepcopy(section.get('binarization') or {})

    new_bin_sub_section = {}

    # Ensure 'combined' channel exists (preserve overrides)
    current_combined = current_cfg.get('combined')
    new_bin_sub_section['combined'] = deepcopy(current_combined or combined_tpl)

    # Actual channels
    for ch in vasc_channels:
        if ch in current_cfg and current_cfg[ch]:  # carry over existing
            new_bin_sub_section[ch] = current_cfg[ch]
        else: # pick template and fill anything else than vessels uses arteries
            template = vessels_tpl if sm.data_type(ch) == 'vessels' else large_vessels_tpl
            new_bin_sub_section[ch] = deepcopy(template)

    # Prune removed channels (keep combined + profiles)
    changed = False
    # anything in current that should go away?
    to_keep = {'combined', *vasc_channels}
    if (set(current_cfg.keys()) != to_keep or
            any(new_bin_sub_section.get(k) != current_cfg.get(k)
                for k in to_keep if k in new_bin_sub_section)):
        changed = True

    if not changed:
        return {}

    # Seed other top-level blocks if absent (don’t overwrite user edits)
    for k in ('graph_construction', 'vessel_type_postprocessing', 'visualization'):
        if k not in section or section.get(k) is None:
            section[k] = deepcopy(defaults.get(k))

    # Write back binarization
    section['binarization'] = REPLACE(new_bin_sub_section)
    return {'vasculature': section}


@adjuster(step=Step.APPLY_RENAMES,
          phase=Phase.PRE_VALIDATE,
          pipelines=('vasculature',),
          keys=('sample.channels','vasculature.binarization'))
def apply_vasculature_renames(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    if not sm.renamed_channels:
        return {}

    section = deepcopy(view.get('vasculature') or {})
    current_bin_sub_sec = deepcopy(section.get('binarization') or {})
    changed = False
    for old, new in sm.renamed_channels.items():
        if old in current_bin_sub_sec and old != 'combined' and new not in current_bin_sub_sec:
            current_bin_sub_sec[new] = current_bin_sub_sec.pop(old)
            changed = True

    if not changed:
        return {}
    section['binarization'] = current_bin_sub_sec
    return {'vasculature': section}

# ########################### BATCH and GROUP ##########################
@adjuster(step=Step.CREATE_PIPELINE_SECTIONS,
          phase=Phase.PRE_VALIDATE,
          pipelines=('group_analysis',),
          keys=('group_analysis',),
          order=10)
def populate_group_analysis_defaults(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """Seed group_analysis with defaults (paths, groups) without overwriting user values."""
    defaults = DEFAULTS_PROVIDER.get('group_analysis') or {}
    section = deepcopy(view.get('group_analysis') or {})
    before = deepcopy(section)

    # carry over, fill if missing
    if 'paths' not in section or section['paths'] is None:
        section['paths'] = deepcopy(defaults.get('paths') or {})
    if 'groups' not in section or section['groups'] is None:
        section['groups'] = {}

    if section != before:
        return {'group_analysis': section}
    return {}


@adjuster(step=Step.ADJUST,
          phase=Phase.PRE_VALIDATE,
          pipelines=('group_analysis',),
          keys=('group_analysis.groups',),
          order=20)
def adjust_group_analysis_groups(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Normalize group lists: make lists, drop falsy items, and de-dupe while preserving order.
    """
    section = deepcopy(view.get('group_analysis') or {})
    groups = deepcopy(section.get('groups') or {})
    if not groups:
        return {}

    changed = False
    for g, items in groups.items():
        norm = [s for s in _ensure_list(items) if s]          # drop falsy
        norm = _dedupe_preserve_order(norm, key=lambda x: x)  # stable de-dupe
        if norm != items:
            groups[g] = norm
            changed = True

    if not changed:
        return {}
    section['groups'] = groups
    return {'group_analysis': section}

# REFACTOR: duplicate with group_analysis above
@adjuster(step=Step.CREATE_PIPELINE_SECTIONS,
          phase=Phase.PRE_VALIDATE,
          pipelines=('batch_processing',),
          keys=('batch_processing',),
          order=10)
def populate_batch_defaults(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """Seed batch_processing with defaults (paths, groups, comparisons) without overwriting user values."""
    defaults = DEFAULTS_PROVIDER.get('batch_processing') or {}
    section = deepcopy(view.get('batch_processing') or {})
    before = deepcopy(section)

    section.setdefault('paths', deepcopy(defaults.get('paths') or {}))
    section.setdefault('groups', {})
    section.setdefault('comparisons', [])

    if section != before:
        return {'batch_processing': section}
    return {}


@adjuster(step=Step.ADJUST,
          phase=Phase.PRE_VALIDATE,
          pipelines=('batch_processing',),
          keys=('batch_processing.groups', 'batch_processing.comparisons'),
          order=20)
def adjust_batch_groups_and_comparisons(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    - Normalize groups exactly like group_analysis
    - Validate comparisons:
        * each item must be a 2-sequence of distinct group names
        * both groups must exist in groups
        * de-duplicate comparisons preserving order
      (No lexicographic canonicalization; ["A","B"] and ["B","A"] remain distinct.)
    """
    section = deepcopy(view.get('batch_processing') or {})
    groups = deepcopy(section.get('groups') or {})
    comps  = deepcopy(section.get('comparisons') or [])
    if groups is None and comps is None:
        return {}

    # Normalize group members
    # REFACTOR: duplicate with group_analysis above
    changed = False
    for g, items in groups.items():
        norm = [s for s in _ensure_list(items) if s]
        norm = _dedupe_preserve_order(norm, key=lambda x: x)
        if norm != items:
            groups[g] = norm
            changed = True

    valid_group_names = set(groups.keys())

    # Normalize comparisons
    normalized_pairs = []
    for item in comps or []:
        # Accept tuples/lists; ignore other shapes
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        a, b = item[0], item[1]
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a == b:
            continue
        if a not in valid_group_names or b not in valid_group_names:
            continue
        normalized_pairs.append([a, b])  # keep original order

    deduped_pairs = _dedupe_preserve_order(normalized_pairs, key=lambda x: tuple(x))
    if deduped_pairs != comps:
        comps = deduped_pairs
        changed = True

    if not changed:
        return {}
    section['groups'] = groups
    section['comparisons'] = comps
    return {'batch_processing': section}


# ################################ MACHINE AND DISPLAY #########
@adjuster(step=Step.CREATE_PIPELINE_SECTIONS,
          phase=Phase.PRE_VALIDATE,
          pipelines=('machine',),
          keys=('machine',),
          order=10)
def populate_machine_defaults(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    defaults = DEFAULTS_PROVIDER.get('machine') or {}
    section = deepcopy(view.get('machine') or {})
    before = deepcopy(section)
    # Only fill missing keys from defaults
    _deep_merge_missing(section, defaults)
    return {'machine': section} if section != before else {}


@adjuster(step=Step.CREATE_PIPELINE_SECTIONS,
          phase=Phase.PRE_VALIDATE,
          pipelines=('display',),
          keys=('display',),
          order=10)
def populate_display_defaults(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    defaults = DEFAULTS_PROVIDER.get('display') or {}
    section = deepcopy(view.get('display') or {})
    before = deepcopy(section)
    _deep_merge_missing(section, defaults)
    return {'display': section} if section != before else {}

