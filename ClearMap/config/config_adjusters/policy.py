from __future__ import annotations

import warnings

"""
Adjusters are pure functions (view, sm) -> patch
that can inspect the current config view and sample manager
and return a patch to be merged into the working config.
They are defined in the ADJUSTERS DEFINITION section and registered
with the @adjuster decorator
"""
from copy import deepcopy
from typing import Any, Optional, Iterable, Dict, Sequence, List, Set, Tuple

from ClearMap.Utils.utilities import deep_merge, REPLACE, DELETE

from ClearMap.config.compound_keys import PairKey

from ClearMap.config.config_handler import ALTERNATIVES_REG
from ClearMap.config.config_adjusters.type_hints import ConfigView, ConfigPatch, SampleManagerProtocol, KeysPath
from ClearMap.config.config_adjusters.dict_ops import (merge_section_missing_only, normalize_json_compat,
                                                       normalise_dict, get_nested, deep_merge_missing, is_under)
from ClearMap.config.config_adjusters.patch_ops import (_strip_template_like_keys_from_container,
                                                        _stable_order_keys,
                                                        _restrict_entry_to_template_keys,
                                                        _strip_template_like_keys_from_entry, merge_patches,
                                                        _raise_if_rename_map_collides, set_patch_at_path)
from ClearMap.config.config_adjusters.resolver import get_current_resolver
from ClearMap.config.config_adjusters.adjusters_api import Phase, Step, patch_adjuster, AdjusterKind
from ClearMap.config.config_adjusters.policy_specs import (InstanceContainerSpec,
                                                           ContainerSpecRegistry, ContainerRole, ReconcileMode,
                                                           TemplateKind, ReconcileKind, Membership, ReconcileStepSpec,
                                                           SmellScanPolicy)

# ############ ADJUSTERS DEFINITION ############
INSTANCE_SPECS: tuple[InstanceContainerSpec, ...] = (
    InstanceContainerSpec(
        section='sample',
        container_path='.channels',
        defaults_templates_path='.templates',
        template_kind=TemplateKind.CHANNELS,
        membership=Membership(source='sample'),
        role=ContainerRole.SOURCE_OF_TRUTH,
        reconcile=ReconcileStepSpec(
            applies=True,
            mode=ReconcileMode.FILL_MISSING_ONLY,
            kind=ReconcileKind.GENERIC,
            restrict_to_template_keys=False,
        ),
    ),
    InstanceContainerSpec(
        section='stitching',
        container_path='.channels',
        defaults_templates_path='.templates',
        template_kind=TemplateKind.CHANNELS,
        membership=Membership(source='pipeline', pipeline='stitching'),
        reconcile=ReconcileStepSpec(
            applies=True,
            kind=ReconcileKind.SPECIAL,           # bespoke writer owns it
            restrict_to_template_keys=True,
        ),
    ),
    InstanceContainerSpec(
        section='registration',
        container_path='.channels',
        defaults_templates_path='.templates',
        template_kind=TemplateKind.CHANNELS,
        membership=Membership(source='sample'),
    ),
    InstanceContainerSpec(
        section='cell_map',
        container_path='.channels',
        defaults_templates_path='.templates',
        template_kind=TemplateKind.CHANNELS,
        membership=Membership(source='pipeline', pipeline='CellMap'),
    ),
    InstanceContainerSpec(
        section='cell_map',
        container_path='.performance.channels',
        defaults_templates_path='.performance.templates',
        template_kind=TemplateKind.PERF,  # you already have PERF kind; use it
        membership=Membership(source='pipeline', pipeline='CellMap'),
        # likely restrict to template keys; depends if you want to allow extra perf knobs
        reconcile=ReconcileStepSpec(
            applies=True,
            kind=ReconcileKind.GENERIC,
            restrict_to_template_keys=True,
        ),
    ),
    InstanceContainerSpec(
        section='tract_map',
        container_path='.channels',
        defaults_templates_path='.templates',
        template_kind=TemplateKind.CHANNELS,
        membership=Membership(source='pipeline', pipeline='TractMap'),
    ),
    InstanceContainerSpec(
        section='colocalization',
        container_path='.channels',
        defaults_templates_path='.templates',
        template_kind=TemplateKind.PAIRS,
        membership=Membership(source='sample', instance_kind='pairs', oriented=True),
        compound=True,
        compound_oriented=True,
        reconcile=ReconcileStepSpec(
            applies=True,
            kind=ReconcileKind.GENERIC,
            restrict_to_template_keys=True,
        ),
    ),
    InstanceContainerSpec(
        section='vasculature',
        container_path='.binarization.single_channels',
        defaults_templates_path='.binarization.templates',
        template_kind=TemplateKind.CHANNELS,
        membership=Membership(source='pipeline', pipeline='TubeMap'),
    ),
    InstanceContainerSpec(
        section='vasculature',
        container_path='.performance.binarization.single_channels',
        defaults_templates_path='.performance.binarization.templates',
        template_kind=TemplateKind.PERF,
        membership=Membership(source='pipeline', pipeline='TubeMap'),
    ),
)

INSTANCE_SPECS_REGISTRY = ContainerSpecRegistry(INSTANCE_SPECS)

RENAME_KEYS = tuple(spec.owns_prefix() for spec in INSTANCE_SPECS_REGISTRY.specs)
GENERIC_INSTANCE_KEYS = tuple(s.owns_prefix() for s in INSTANCE_SPECS_REGISTRY.generic_instance_specs())
static_candidates = ALTERNATIVES_REG.canonical_pipeline_config_names + ALTERNATIVES_REG.canonical_global_config_names
STATIC_SECTIONS = tuple(s for s in static_candidates if s != 'sample')
STATIC_WATCHED_KEYS = (('sample', 'channels'),) + tuple((s,) for s in STATIC_SECTIONS)
STATIC_OWNED_KEYS = tuple((s,) for s in STATIC_SECTIONS)


# ############ Global rename triggered by SampleManager ############
@patch_adjuster(step=Step.APPLY_RENAMES, phase=Phase.PRE_VALIDATE,
                watched_keys=None, kind=AdjusterKind.INSTANCE_OWNER,
                owned_keys=RENAME_KEYS)
def apply_channel_renames(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Rename section channels

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
    rename_map = sm.renamed_channels
    if not rename_map:
        return {}

    _raise_if_rename_map_collides(rename_map)

    patch: ConfigPatch = {}

    # rename targets are declared by spec.rename.applies
    for spec in INSTANCE_SPECS_REGISTRY.specs:
        if not spec.rename.applies:
            continue

        abs_path = spec.abs_container_path()
        current = normalise_dict(get_nested(view, abs_path))
        if not current:
            continue

        if spec.compound and spec.rename.migrate_payload:
            updated, changed = PairKey.rename_container_keys(current, rename_map, oriented=spec.compound_oriented)
        else:
            changed = False
            updated = deepcopy(current)
            for old, new in rename_map.items():
                if old in updated and new not in updated:
                    updated[new] = updated.pop(old)
                    changed = True

        if changed:
            set_patch_at_path(patch, abs_path, REPLACE(updated))

    return patch


# WARNING: this must not touch instance containers declared by InstanceContainerSpec
@patch_adjuster(step=Step.ENSURE_STATIC_BLOCKS, phase=Phase.PRE_VALIDATE,
                watched_keys=STATIC_WATCHED_KEYS, owned_keys=STATIC_OWNED_KEYS,
                kind=AdjusterKind.STATIC_FILLER, order=50)
def ensure_required_static_blocks(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Missing-only fill of required static blocks from defaults.

    .. warning::
        Must not touch instance containers declared by InstanceContainerSpec,
        because those are owned by reconcile/materialize adjusters and may be REPLACE'd.
    """
    resolver = get_current_resolver()
    patch: ConfigPatch = {}

    # TODO: decide for batch sections
    allowed_sections = sm.compute_required_sections() | set(ALTERNATIVES_REG.canonical_global_config_names)

    for sec in STATIC_SECTIONS:
        if sec not in allowed_sections:
            continue

        defaults = resolver.section_missing_only_defaults(sec)
        if not defaults:
            continue

        current = deepcopy(view.get(sec) or {})

        skip_paths = INSTANCE_SPECS_REGISTRY.skip_paths_for_missing_only_merge(sec)
        changed = merge_section_missing_only(current, defaults, skip_paths=skip_paths)

        if changed and current:
            patch[sec] = current

    return patch


def _stitching_ensure_layout_and_root(current: dict[str, Any], stitchable: list[str], ordered: list[str]) -> tuple[set[str], str, dict[str, Any], bool]:
    if not stitchable:
        return set(), '', current, False

    def _is_layout(channel: str, cfg: dict[str, Any]) -> bool:
        return cfg.get('layout_channel', channel) == channel

    changed = False
    layout_set = {ch for ch, cfg in current.items() if isinstance(cfg, dict) and _is_layout(ch, cfg)}

    if not layout_set:
        root = stitchable[0]
        cfg = current.get(root, {})
        current[root] = {**cfg, 'layout_channel': root}
        layout_set.add(root)
        changed = True

    index = {ch: i for i, ch in enumerate(ordered)}
    root_layout = min(layout_set, key=lambda ch: index.get(ch, 10**9))

    return layout_set, root_layout, current, changed


def _stitching_is_layout_and_repair(*, ch: str, cfg: dict[str, Any], layout_channels_set: set[str], root_layout: str) -> tuple[str, dict[str, Any], bool]:
    changed = False
    target = cfg.get('layout_channel', ch)

    if target == ch:
        # layout role; ensure explicit layout_channel
        if 'layout_channel' not in cfg:
            cfg = {**cfg, 'layout_channel': ch}
            changed = True
        return ch, cfg, changed

    # derived role; repair invalid reference
    if target not in layout_channels_set:
        cfg = {**cfg, 'layout_channel': root_layout}
        changed = True
        target = root_layout

    return target, cfg, changed


def _build_stitching_entry(*, ch: str, cfg: dict[str, Any] | None,
                           layout_channels_set: set[str],
                           root_layout: str) -> tuple[dict[str, Any], bool]:
    """
    Returns (entry, changed_vs_cfg).

    - Determines role from cfg.get('layout_channel', ch)
    - Repairs derived references to point to an existing layout (root_layout fallback)
    - Canonicalizes layout channels to explicitly carry layout_channel: ch (if absent)
    - Applies template + deep_merge(tpl_base, cfg)
    - Pops irrelevant keys (opposite-role keys, template-like keys)
    """
    resolver = get_current_resolver()

    cfg = normalise_dict(cfg)
    ref, cfg2, repaired = _stitching_is_layout_and_repair(ch=ch, cfg=cfg, layout_channels_set=layout_channels_set, root_layout=root_layout)

    tpl_base = resolver.stitching_entry_template(ch, root_layout=ref)  # role decided by ref==ch or not
    merged = deep_merge(deepcopy(tpl_base), deepcopy(cfg2))

    # keep-only template schema: this replaces “strip opposite-role keys”
    merged = _restrict_entry_to_template_keys(merged, tpl_base)

    _strip_template_like_keys_from_entry(merged)
    return merged, repaired

def _stitching_prune_to_membership(current: dict[str, Any], keep: list[str]) -> tuple[dict[str, Any], bool]:
    pruned = {k: current[k] for k in keep if k in current}
    return pruned, pruned != current


def reconcile_stitching_channels_step(*, stitching_section: dict[str, Any], sm: SampleManagerProtocol) -> tuple[dict[str, Any], bool]:
    stitchable = sm.stitchable_channels

    existing_channels_cfg = stitching_section.get('channels')
    current = normalise_dict(existing_channels_cfg)

    ordered = _stable_order_keys(current_keys=current.keys(), target_keys=stitchable)

    current, changed_prune = _stitching_prune_to_membership(current, ordered)
    layout_set, root_layout, current, changed_root = _stitching_ensure_layout_and_root(current, stitchable, ordered)
    changed = changed_prune or changed_root

    updated = {}
    for ch in stitchable:
        cfg = current.get(ch)
        if not isinstance(cfg, dict):
            cfg = {}
            changed = True

        entry, entry_changed = _build_stitching_entry(ch=ch, cfg=cfg, layout_channels_set=layout_set,
                                                      root_layout=root_layout)
        updated[ch] = entry
        if current.get(ch) != entry:
            changed = True
        changed = changed or entry_changed

    if not changed and updated == current:
        return stitching_section, False

    out = deepcopy(stitching_section)
    out['channels'] = REPLACE(updated)
    return out, True


@patch_adjuster(step=Step.CREATE_CHANNELS_RECONCILE, phase=Phase.PRE_VALIDATE,
                watched_keys=('sample.channels', 'stitching.channels'),
                owned_keys=('stitching.channels',),
                kind=AdjusterKind.INSTANCE_OWNER, order=50)
def reconcile_stitching_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """
    Single-writer for stitching.channels.

    Responsibilities (in one place):
      - prune to sm.stitchable_channels
      - ensure at least one layout channel (root)
      - repair derived layout_channel references (fallback to root layout)
      - materialize missing channels using defaults stitching.templates.{layout_template,derived_template}
      - apply role templates (layout vs derived) while preserving user overrides
      - strip opposite-role keys and template-leak keys

    Emits:
      {'stitching': {'channels': REPLACE(updated)}} when changed, else {}.
    """
    stitching_section = deepcopy(view.get('stitching') or {})
    stitching_section, changed = reconcile_stitching_channels_step(stitching_section=stitching_section, sm=sm)
    return {'stitching': stitching_section} if changed else {}


def _ctx_for_key(*, spec: InstanceContainerSpec, key: str, view: ConfigView, sm: SampleManagerProtocol) -> dict[str, Any]:
    return spec.ctx_builder(key, view, sm) if spec.ctx_builder else {}


def _bound_template_for_key(*, spec: InstanceContainerSpec, key: str, view: ConfigView, sm: SampleManagerProtocol, resolver) -> dict[str, Any]:
    ctx = _ctx_for_key(spec=spec, key=key, view=view, sm=sm)
    return resolver.entry_template(section=spec.section, kind=spec.template_kind, key=key, **ctx)


def _emit_container_patch(*, container_path: tuple[str, ...], new_map: dict[str, Any], replace: bool) -> ConfigPatch:
    patch: ConfigPatch = {}
    set_patch_at_path(patch, container_path, REPLACE(new_map) if replace else new_map)
    return patch


def _compound_preprocess(*, spec: InstanceContainerSpec, cur_map: dict[str, Any], sm: SampleManagerProtocol) -> tuple[dict[str, Any], bool]:
    """
    Canonicalize compound keys (if enabled) and optionally prune invalid atoms.
    Returns (updated_cur_map, moved_flag).
    """
    if not spec.compound:
        return cur_map, False

    cur_map, moved = PairKey.normalize_container_keys(cur_map, oriented=spec.compound_oriented)

    if spec.compound_prune_invalid_atoms:
        pruned = PairKey.prune_container_invalid_atoms(cur_map, allowed_atoms=set(sm.channels), oriented=spec.compound_oriented)
        moved = moved or (pruned != cur_map)
        cur_map = pruned

    return cur_map, moved


def _reconcile_missing_only(*, spec: InstanceContainerSpec, view: ConfigView, sm: SampleManagerProtocol,
                            resolver, cur_map: dict[str, Any], moved: bool) -> tuple[dict[str, Any], bool]:
    """
    Missing-only reconcile:
      - does NOT create/prune membership keys
      - fills missing fields within existing entries
      - does NOT REPLACE the container
    """
    changed = moved

    out = {}
    for k, v in (cur_map or {}).items():
        entry = v if isinstance(v, dict) else {}
        if entry is not v:
            changed = True

        before = deepcopy(entry)
        tpl = _bound_template_for_key(spec=spec, key=k, view=view, sm=sm, resolver=resolver)

        deep_merge_missing(entry, tpl)
        entry = normalize_json_compat(entry)
        out[k] = entry

        changed = changed or (entry != before)

    return out, changed


def _reconcile_replace_container(*, spec: InstanceContainerSpec, view: ConfigView, sm: SampleManagerProtocol,
                                 resolver, cur_map: dict[str, Any], moved: bool) -> tuple[dict[str, Any], bool]:
    """
    REPLACE reconcile:
      - establishes membership keys (resolve_keys)
      - materializes full entries from templates
      - optionally migrates payload for compound rename
      - optionally restricts to template schema
      - emits REPLACE(container)
    """
    changed = moved

    raw_keys = spec.resolve_keys(sm=sm, resolver=resolver)
    if spec.reconcile.preserve_existing_order:
        keys = _stable_order_keys(current_keys=cur_map.keys(), target_keys=raw_keys)
    else:
        keys = raw_keys

    new_map = {}
    for k in keys:
        if spec.reconcile.policy.preserve_existing and isinstance(cur_map.get(k), dict):
            new_map[k] = cur_map[k]
            continue

        carried_cfg = None
        carried = False
        if spec.compound and spec.rename.migrate_payload:
            carried_cfg, carried = PairKey.migrate_container_payload(
                container=cur_map, new_key=k, rename_map=sm.renamed_channels, oriented=spec.compound_oriented)

        base = _bound_template_for_key(spec=spec, key=k, view=view, sm=sm, resolver=resolver)

        entry = deep_merge(deepcopy(base), deepcopy(carried_cfg)) if carried else deepcopy(base)

        needs_materialize = (not spec.reconcile.policy.preserve_existing) or (k not in cur_map) or (not cur_map.get(k))
        changed = changed or (carried or needs_materialize)

        if spec.reconcile.restrict_to_template_keys and isinstance(entry, dict) and isinstance(base, dict):
            entry = _restrict_entry_to_template_keys(entry, base)

        _strip_template_like_keys_from_entry(entry)
        new_map[k] = entry

    if not spec.reconcile.policy.prune_removed:
        for k, v in cur_map.items():
            if k not in new_map:
                new_map[k] = v

    if not changed and cur_map == new_map:
        return new_map, False

    return new_map, True


def reconcile_instance_container(*, spec: InstanceContainerSpec,
                                 view: ConfigView, sm: SampleManagerProtocol,
                                 resolver) -> Optional[ConfigPatch]:
    if not spec.reconcile.applies:
        return None

    container_path = spec.abs_container_path()
    cur_map = spec.container_map(view)

    cur_map, moved = _compound_preprocess(spec=spec, cur_map=cur_map, sm=sm)

    # =========================================================
    # Missing-only reconcile mode
    # =========================================================
    if spec.reconcile.mode == ReconcileMode.FILL_MISSING_ONLY:
        # only fill per-entry defaults. (no resolve_keys(), no membership prunning, no channel creation
        new_map, changed = _reconcile_missing_only(spec=spec, view=view, sm=sm, resolver=resolver, cur_map=cur_map, moved=moved)
        if not changed:
            return None
        return _emit_container_patch(container_path=container_path, new_map=new_map, replace=False)

    # =========================================================
    # REPLACE-container mode
    # =========================================================
    new_map, changed = _reconcile_replace_container(spec=spec, view=view, sm=sm, resolver=resolver, cur_map=cur_map, moved=moved)
    if not changed:
        return None

    # Defensive: strip template-like keys from final container
    _strip_template_like_keys_from_container(new_map)

    return _emit_container_patch(container_path=container_path, new_map=new_map, replace=True)



# WARNING: all but stitching (SPECIAL writer)
@patch_adjuster(step=Step.CREATE_CHANNELS_RECONCILE, phase=Phase.PRE_VALIDATE,
                watched_keys=('sample.channels',), kind=AdjusterKind.INSTANCE_OWNER,
                owned_keys=GENERIC_INSTANCE_KEYS)
def reconcile_generic_instance_containers(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    resolver = get_current_resolver()
    patch: ConfigPatch = {}

    for spec in INSTANCE_SPECS_REGISTRY.ordered_for_reconcile(kind=ReconcileKind.GENERIC):
        p = reconcile_instance_container(spec=spec, view=view, sm=sm, resolver=resolver)
        if p:
            merge_patches(patch, p)

    return patch

########################## TEMPLATES CLEANUP ##############
TEMPLATE_ROOT_KEYS = tuple(spec.abs_templates_path() for spec in INSTANCE_SPECS_REGISTRY.specs
                           if spec.abs_templates_path())


@patch_adjuster(
    step=Step.ADJUST,               # keep it simple for the release
    phase=Phase.PRE_VALIDATE,
    watched_keys=None,              # must run regardless of changed_keys
    owned_keys=TEMPLATE_ROOT_KEYS,  # only allowed to touch canonical template roots
    kind=AdjusterKind.OTHER,
    order=10_000,                   # run late (after reconcile/materialize)
)
def sanitize_runtime_template_roots(view, sm) -> dict:
    """
    A1 sanitizer: delete template reservoirs located at canonical defaults_templates_path
    derived from InstanceContainerSpec.

    Scope:
      - only affects sections that exist in the current view
      - runner will additionally scope by active_sections based on owned_keys sections
    """
    patch = {}
    deleted: set[tuple[str, ...]] = set()

    for spec in INSTANCE_SPECS_REGISTRY.specs:
        if spec.section not in view:
            continue

        tpl_path = spec.abs_templates_path()
        if not tpl_path:
            continue

        if get_nested(view, tpl_path) is None:  # Nothing in instance cfg -> move on
            continue

        # Delete the entire templates reservoir subtree
        set_patch_at_path(patch, tpl_path, DELETE)
        deleted.add(tuple(tpl_path))

    # Warn once per path
    for p in sorted(deleted):
        warnings.warn(f'[policy-violation][runtime-templates] deleted {".".join(p)}',
                      RuntimeWarning, stacklevel=2)

    return patch



############## TEMPLATES SCORIES SCANNER ##############
def _any_under(path: KeysPath, roots: Sequence[KeysPath]) -> bool:
    return any(is_under(path, r) for r in roots)

def _iter_dict_paths(obj: Any, prefix: KeysPath = ()) -> Iterable[Tuple[KeysPath, Any]]:
    """Yield (path, node) for all dict nodes in a nested structure."""
    if not isinstance(obj, dict):
        return
    yield prefix, obj
    for k, v in obj.items():
        if isinstance(v, dict):
            yield from _iter_dict_paths(v, prefix + (str(k),))


def scan_config_smells(*, view: Dict[str, Any], sm: SampleManagerProtocol,
                       instance_specs_reg: ContainerSpecRegistry,
                       policy: SmellScanPolicy = SmellScanPolicy()) -> List[str]:
    """
    Returns a list of warning strings (does not emit warnings by itself).
    Callers may choose to warnings.warn(...) or log.

    Focus:
      - template-ish keys outside canonical templates roots (strong)
      - channel-ish keys under reserved namespaces like performance (light)
    """
    warnings_out: List[str] = []

    # Canonical template roots (section-rooted absolute paths)
    templates_roots: List[KeysPath] = []
    # Canonical instance containers
    container_roots: List[KeysPath] = []

    # For reserved namespace checks: collect parents that have a canonical ".<reserved>.channels" container
    reserved_parents_with_channels: Set[KeysPath] = set()

    for spec in instance_specs_reg:
        if spec.section not in view:
            continue
        # spec.abs_templates_path() -> Optional[KeysPath]
        tpath = spec.abs_templates_path()
        if tpath:
            templates_roots.append(tuple(tpath))

        cpath = spec.abs_container_path()
        if cpath:
            container_roots.append(tuple(cpath))

            # If this spec defines ".../<reserved>/channels", remember ".../<reserved>" as a reserved parent.
            # Example: ('cell_map','performance','channels') -> parent ('cell_map','performance')
            if len(cpath) >= 2 and cpath[-1] == "channels" and cpath[-2] in policy.reserved_namespace_keys:
                reserved_parents_with_channels.add(tuple(cpath[:-1]))

    # ----------------------------
    # Strong: template-ish keys outside template roots
    # ----------------------------
    if policy.strong_template_smells:
        for dict_path, node in _iter_dict_paths(view):
            # If we are already inside an allowed templates root, do not warn about template-ish keys here.
            # (Templates are expected under templates roots.)
            in_templates_root = _any_under(dict_path, templates_roots)

            if not isinstance(node, dict):
                continue

            for k in node.keys():
                ks = str(k)
                key_path = dict_path + (ks,)

                if ks == 'templates':  # FIXME: forbidden
                    warnings_out.append(f'[config-smell][templates-key-in-runtime] '
                                        f'Found forbidden key at {".".join(key_path)}')
                    continue

                is_templateish = (
                    ks in policy.templateish_exact or
                    ks.endswith(policy.templateish_suffix)
                )
                if is_templateish:
                    warnings_out.append(
                        f'[config-smell][template-key-outside-templates-root] '
                        f'Found template-ish key at {".".join(key_path)} (expected under one of: '
                        f'{", ".join(".".join(r) for r in templates_roots) or "<none>"})'
                    )

    # ----------------------------
    # Light: channel-ish keys under reserved namespaces (e.g. performance.<channel>)
    # ----------------------------
    if policy.light_channelish_smells and reserved_parents_with_channels:
        channels_set = set(getattr(sm, "channels", []) or [])

        for parent in reserved_parents_with_channels:
            if not parent or parent[0] not in view:
                continue
            # parent is like ('cell_map','performance')
            # locate node
            node: Any = view
            for p in parent:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(p)
            if not isinstance(node, dict):
                continue

            # if canonical children exist, channel-like siblings are suspicious
            for k in list(node.keys()):
                ks = str(k)
                if ks in ("channels", "templates"):
                    continue
                if ks in channels_set:
                    warnings_out.append(
                        f"[config-smell][channel-key-in-reserved-namespace] "
                        f"Found channel-like key at {'.'.join(parent + (ks,))}; "
                        f"canonical container exists at {'.'.join(parent + ('channels',))}"
                    )

    return warnings_out


def warn_config_smells(*, view: Dict[str, Any], sm: SampleManagerProtocol, instance_specs_reg: ContainerSpecRegistry,
                       policy: SmellScanPolicy = SmellScanPolicy()) -> None:
    for msg in scan_config_smells(view=view, sm=sm, instance_specs_reg=instance_specs_reg, policy=policy):
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

#
# @adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
#           phase=Phase.PRE_VALIDATE,
#           pipelines=('registration',),
#           keys=('sample.channels', 'registration.channels'))
# def reconcile_registration_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
#     """
#     Build registration.channels for all sample channels using defaults/registration.yml:templates.
#     - Channels with data_type == 'autofluorescence' use 'autofluorescence' template.
#     - Others use 'regular' template with ${reference} = alignment reference channel.
#     Also expands ${channel}/${reference} placeholders.
#     """
#     return _reconcile_simple_section_channels('registration', view, sm)
#
#
# # FIXME: add collocation_compatible=True if n> =2
# @adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
#           phase=Phase.PRE_VALIDATE,
#           pipelines=('cell_map',),
#           keys=('sample.channels','cell_map.channels'))
# def reconcile_cell_map_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
#     return _reconcile_simple_section_channels('cell_map', view, sm)
#
#
# @adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
#           phase=Phase.PRE_VALIDATE,
#           pipelines=('tract_map',),
#           keys=('sample.channels','tract_map.channels'))
# def reconcile_tract_map_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
#     """
#     Keep tract_map.channels aligned with sample channels using defaults/tract_map.yml:template.channel
#     (add missing, keep overrides, drop removed).
#     """
#     return _reconcile_simple_section_channels('tract_map', view, sm)


# @adjuster(step=Step.CREATE_CHANNELS_RECONCILE,
#           phase=Phase.PRE_VALIDATE,
#           pipelines=('colocalization',),
#           keys=('sample.channels', 'colocalization.channels'))
# def reconcile_colocalization_channels(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
#     """
#     Pairs = all 2-combinations of sm.channels_to_detect (when sm.is_colocalization_compatible).
#     Canonical keys 'A-B'. Preserves overrides, migrates on rename, prunes removed.
#     """
#     resolver = get_current_resolver()
#
#     target_keys = resolver.colocalization_target_keys()
#     if not target_keys:
#         col = deepcopy(view.get('colocalization') or {})
#         if not (col.get('channels') or {}):
#             return {}
#         col['channels'] = {}
#         return {'colocalization': col}
#
#     def build_entry(k: str) -> Dict[str, Any]:
#         return resolver.colocalization_entry_template(k)
#
#     return _reconcile_section_channels( section_name='colocalization', view=view, target_channels=target_keys,
#                                         build_entry=build_entry, preserve_existing=True, compound=True,
#                                         rename_map=sm.renamed_channels, sample_set=set(sm.channels))

# # ################################ GLOBAL SECTIONS #############

# TODO: ensure we can remove those

# @adjuster(step=Step.CREATE_PIPELINE_SECTIONS,
#           phase=Phase.PRE_VALIDATE,
#           pipelines=('machine',),
#           keys=('machine',),
#           order=10)
# def populate_machine_defaults(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
#     defaults = DEFAULTS_PROVIDER.get('machine')
#     section = deepcopy(view.get('machine') or {})
#     before = deepcopy(section)
#     # Only fill missing keys from defaults
#     _deep_merge_missing(section, defaults)
#     return {'machine': section} if section != before else {}
#
#
# @adjuster(step=Step.CREATE_PIPELINE_SECTIONS,
#           phase=Phase.PRE_VALIDATE,
#           pipelines=('display',),
#           keys=('display',),
#           order=10)
# def populate_display_defaults(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
#     defaults = DEFAULTS_PROVIDER.get('display')
#     section = deepcopy(view.get('display') or {})
#     before = deepcopy(section)
#     _deep_merge_missing(section, defaults)
#     return {'display': section} if section != before else {}
