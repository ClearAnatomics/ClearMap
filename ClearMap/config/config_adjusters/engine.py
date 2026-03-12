from __future__ import annotations
"""
config_adjusters.engine
======================

Execution engine for config adjusters: registry, selection, ordering, and patch aggregation.

This module orchestrates *adjusters* (pure, idempotent functions) over a config view and returns a
single merged patch for the coordinator to apply. It does not persist config, validate schema, or
interpret UI events.

Concepts and vocabulary
-----------------------

- **section**: top-level config key (e.g. 'sample', 'registration', 'vasculature').
- **block**: top-level subtree within a section
  (e.g. ('vasculature', 'visualization'), ('registration', 'performance')).
- **path**: tuple of keys locating a subtree
  (e.g. ('vasculature', 'performance', 'binarization')).
- **static block**: schema not keyed by runtime membership; filled missing-only from defaults.
- **instance block**: keyed by runtime membership (channels, pairs, etc.); owned by
  reconcile/materialize adjusters.
- **ownership**: adjuster-declared set of path prefixes it may modify.
- **missing-only merge**: recursively fill absent keys only.
- **container replace**: authoritative replacement of an entire mapping at a path
  (e.g. REPLACE(channels_dict)).

Operational model
-----------------

Adjusters are registered via a decorator that records phase, step, filters (pipelines/keys),
and ordering. `run_adjusters(...)` executes a fixed step plan per phase:

- structural renames,
- instance materialization / reconcile / prune,
- static defaults population,
- custom adjustments,
- final reconcile and shape normalization.

Adjusters run sequentially within each step. Each patch is merged into both the step aggregate and
the working view so later adjusters observe earlier results.

Adjuster contract
-----------------

An adjuster is:

    (view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch | None

Expectations:
- pure and idempotent,
- patch-only output (no side effects),
- bounded authority: either container owner (may replace) or missing-only filler.

Patch semantics
---------------

Patches are merged step-wise; the working view is updated via deep merge. Authors must be explicit
about container ownership to avoid downgrade of authoritative replaces into ordinary merges.

Defaults and tracing
--------------------

Defaults resolution is centralized via a DefaultsResolver bound for the run. Optional tracing
records plan, selection, key changes, and detects downgrade patterns on configured paths.

Style
-----

Prefer explicit behavior, small deterministic adjusters, and testability.
"""

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__licence__ = 'GPLv3 - GNU General Public License v3 (see LICENSE file)'
__copyright__ = 'Copyright © 2026 by Charly Rousseau and ClearAnatomics devs'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Tuple, Dict, Iterable, Callable, List, Any, Set

from ClearMap.Utils.utilities import deep_merge, DELETE, _REPLACE
from ClearMap.config.config_adjusters.type_hints import ConfigKeys, ConfigView, AdjustmentContext, ConfigPatch
from ClearMap.config.config_adjusters.dict_ops import is_under
from ClearMap.config.config_adjusters.patch_ops import merge_patches, _iter_patch_paths, iter_patch_items
from ClearMap.config.defaults_provider import get_defaults_provider
from ClearMap.config.config_adjusters.templates_resolver import (BindPolicy, set_current_resolver,
                                                                 ExperimentTemplatesResolver, GroupTemplatesResolver)
from ClearMap.config.config_adjusters.trace_logging import _get_path, Tracer, summarize_paths
from ClearMap.config.config_adjusters.adjusters_api import (Phase, Step, AdjusterSpec, _REGISTRY,
                                                            _config_keys_overlap, AdjusterKind,
                                                            AdjusterScope)
from ClearMap.config.config_adjusters.policy import INSTANCE_SPECS_REGISTRY, warn_config_smells
import ClearMap.config.config_adjusters.group_adjusters  # noqa: F401 (register adjusters)


PHASES_ORDER: Dict[Phase, Tuple[Step, ...]] = {
    Phase.PRE_VALIDATE: (
        Step.APPLY_RENAMES,
        Step.CREATE_PIPELINE_SECTIONS,
        Step.CREATE_CHANNELS_RECONCILE,
        Step.ENSURE_STATIC_BLOCKS,
        Step.POPULATE_DEFAULTS_MATERIALIZE,
        Step.ADJUST,
        # Step.NORMALIZE_SHAPES,
    ),
    Phase.POST_VALIDATE: (),
    Phase.PRE_COMMIT:    (),
    Phase.POST_COMMIT:   (),
}


def sections_from_keys(keys: Optional[Iterable[ConfigKeys]]) -> Optional[Set[str]]:
    if keys is None:
        return None
    secs = set()
    for k in keys:
        if isinstance(k, tuple):
            secs.add(k[0])
        elif isinstance(k, str):
            secs.add(k.split('.')[0])
        else:
            raise TypeError(f"Invalid config key type: {type(k).__name__}")
    return secs if secs else None


# Just for debug
@dataclass(frozen=True)
class RunnerConfig:
    trace: bool = True
    trace_paths: Tuple[str, ...] = ("vasculature.binarization", "vasculature.performance.binarization")
    trace_sink: Optional[Callable[[str], None]] = None


@contextmanager
def inject_sm_view(sm, *, view: dict, section_name: str = "sample"):
    """
    Temporarily make sm.config read from `view[section_name]` instead of the coordinator.
    We swap the instance's class to a dynamic subclass
    that only overrides the `config` property.
    """
    base_cls = sm.__class__

    class _Injected(base_cls):  # type: ignore[misc]
        @property
        def config(self):
            sample = view.get(section_name) or {}
            return MappingProxyType(sample)

    sm.__class__ = _Injected
    try:
        yield sm
    finally:
        sm.__class__ = base_cls



class AdjusterRunner:
    """
    Orchestrates adjuster selection, ordering, execution, patch aggregation,
    tracing, and (optionally) authority enforcement.

    Design goals:
      - keep run() readable
      - isolate tracing noise
      - isolate policy/enforcement hooks
      - keep behavior compatible with the prior run_adjusters() function
    """

    def __init__(self, *, phase: "Phase", active_sections: Optional[Iterable[str]],
                 changed_keys: Optional[Iterable[ConfigKeys]], config: RunnerConfig = RunnerConfig(),
                 schema_registry: Optional[Any] = None) -> None:
        self.phase = phase
        if active_sections is not None and not active_sections:
            raise ValueError("active_sections, if provided, must not be empty")
        self.active_sections = None if active_sections is None else set(active_sections)

        if changed_keys is not None and not changed_keys:
            raise ValueError("changed_keys, if provided, must not be empty")
        self.changed_keys = None if changed_keys is None else list(changed_keys)

        self.cfg = config
        self.t = Tracer(config.trace, config.trace_sink)

        # Optional: registry that can answer excluded_roots(section)->set[tuple[str,...]]
        # You can inject your ContainerSpecRegistry here once implemented.
        self.schema_registry = schema_registry

        self._resolver_policy = BindPolicy()
        self._working_view: Dict[str, Any] = {}
        self._global_patch: ConfigPatch = {}

    # ---------------- public API ----------------

    def run(self, *, view: ConfigView, ctx: AdjustmentContext) -> ConfigPatch:
        self._run_id = getattr(self, "_run_id", 0) + 1
        self.t(f"[run_adjusters] id={self._run_id} phase={self.phase} scope={ctx.scope} "
               f"active={self.active_sections} changed_keys={self.changed_keys}")

        steps_order = PHASES_ORDER.get(self.phase, ())
        if not steps_order:
            return {}

        self._working_view = view
        self._global_patch = {}

        self._install_template_resolver(ctx)

        self._trace_run_header(steps_order)

        by_step = self._select_specs(steps_order, scope=ctx.scope)

        for step in steps_order:
            step_specs = sorted(by_step.get(step, []), key=lambda s: (s.order, s.name))
            step_patch = self._run_step(step=step, specs=step_specs, ctx=ctx)

            if self.active_sections:  # clip to active sections
                step_patch = {k: v for k, v in step_patch.items() if k in self.active_sections}

            if step_patch:
                self.t("  [step_patch merge -> global patch]")
                merge_patches(self._global_patch, step_patch)

            if step == Step.APPLY_RENAMES and ctx.scope == AdjusterScope.EXPERIMENT:
                # Rename step can change facts, so resolver must be rebound.
                self._install_template_resolver(ctx)

        if self.phase == Phase.PRE_VALIDATE and ctx.scope == AdjusterScope.EXPERIMENT:
            # Only makes sense with real experiment SM + instance specs
            warn_config_smells(view=self._working_view, sm=ctx.sample_manager,
                               instance_specs_reg=INSTANCE_SPECS_REGISTRY)

        return self._global_patch

    # ---------------- selection ----------------

    def _select_specs(self, steps_order: Tuple["Step", ...], *, scope: AdjusterScope) -> Dict["Step", List["AdjusterSpec"]]:
        by_step: Dict["Step", List["AdjusterSpec"]] = {s: [] for s in steps_order}

        for spec in _REGISTRY:
            if spec.scope != scope:
                continue
            if spec.phase != self.phase:
                continue
            if spec.step not in by_step:
                continue

            spec_sections = sections_from_keys(spec.owned_keys)

            self.t(f'[select_specs] registry={len(_REGISTRY)} active_sections={self.active_sections} '
                   f'changed_keys={self.changed_keys}')
            if self.active_sections is not None:
                if not spec_sections:
                    continue
                if spec_sections.isdisjoint(self.active_sections):
                    continue

            if not _config_keys_overlap(self.changed_keys, spec.watched_keys):
                continue

            by_step[spec.step].append(spec)

        return by_step

    # ---------------- execution ----------------

    def _run_step(self, *, step: "Step", specs: List["AdjusterSpec"], ctx: AdjustmentContext) -> ConfigPatch:
        step_patch: ConfigPatch = {}

        self.t(f"\n[step] {step} specs={len(specs)}")

        for spec in specs:
            self._trace_spec_header(spec)
            self._summarize_trace_nodes(self._working_view, label='working_view before')

            if spec.requires_sample_manager:
                if ctx.sample_manager is None:
                    raise ValueError(f"{spec.name}: requires_sample_manager=True but ctx.sample_manager is None")
                if ctx.scope != AdjusterScope.EXPERIMENT:
                    raise ValueError(f"{spec.name}: requires_sample_manager=True but scope is {ctx.scope}")

            res = spec.fn(self._working_view, ctx) or {}
            if not res:
                self.t("    [result] empty")
            else:
                self._summarize_trace_nodes(res, label='result')

                self._enforce(spec, res)
                # Merge into step_patch (patch composition semantics)
                self._merge_step_patch_with_downgrade_detection(step_patch, res)
                # Sequential semantics: apply to working view
                self._working_view = deep_merge(self._working_view, res)

                self._summarize_trace_nodes(self._working_view, label='working_view after')

        return step_patch
    def _merge_step_patch_with_downgrade_detection(self, step_patch: ConfigPatch, res: ConfigPatch) -> None:
        trace_paths = self.cfg.trace_paths or ()
        if not trace_paths:
            merge_patches(step_patch, res)
            return

        agg_before = {p: _get_path(step_patch, p) for p in trace_paths}
        merge_patches(step_patch, res)
        agg_after = {p: _get_path(step_patch, p) for p in trace_paths}

        for p in trace_paths:
            b = agg_before[p]
            a = agg_after[p]
            # This is intentionally loose (keeps compatibility with your existing sentinel type name)
            if (b is not None and type(b).__name__ == "_REPLACE") and isinstance(a, dict):
                self.t(f"    [DOWNGRADE] step_patch {p}: _REPLACE -> dict after merging {type(res).__name__}")

    # ---------------- enforcement (hook points) ----------------

    def _enforce(self, spec: "AdjusterSpec", patch: ConfigPatch) -> None:
        """
        Hook point for authority gates.

        Recommended future behavior:
          - if spec.kind == STATIC_FILLER:
                forbid touching excluded roots (instance containers + templates reservoirs)
                forbid DELETE / REPLACE
          - if spec.kind == INSTANCE_OWNER:
                require all emitted paths under spec.owned_keys allowlist
        """
        if self.schema_registry is None:
            return

        self._enforce_ownership(spec, patch)

        if spec.kind == AdjusterKind.STATIC_FILLER:
            self._enforce_static_exclusions(spec, patch)
            self._enforce_no_sentinels_for_static(spec, patch)

    def _enforce_ownership(self, spec: AdjusterSpec, patch: ConfigPatch) -> None:
        if not spec.owned_keys:
            return
        for path in _iter_patch_paths(patch):
            if not any(is_under(owner, path) for owner in spec.owned_keys):
                dotted = '.'.join(path)
                owners = ', '.join('.'.join(o) for o in spec.owned_keys)
                raise ValueError(f'Adjuster {spec.name} emitted patch outside ownership: {dotted} not under [{owners}]')

    # FIXME: use
    def _enforce_static_exclusions(self, spec: "AdjusterSpec", patch: ConfigPatch) -> None:
        """
        Example enforcement: static fillers must not touch excluded roots.
        Requires schema_registry.excluded_roots(section) -> set[tuple[str,...]] (section-relative).
        """
        for p in _iter_patch_paths(patch):
            if not p:
                continue
            sec = p[0]
            rel = p[1:]
            excluded = self.schema_registry.excluded_roots(sec)
            if excluded and any(is_under(ex, rel) for ex in excluded):
                dotted = ".".join((sec,) + rel)
                raise ValueError(f"{spec.name}: touched excluded subtree: {dotted}")

    def _enforce_no_sentinels_for_static(self, spec: AdjusterSpec, patch: ConfigPatch) -> None:
        for path, val in iter_patch_items(patch):
            if val is DELETE or isinstance(val, _REPLACE):
                raise ValueError(f"{spec.name}: static filler emitted forbidden sentinel at {'.'.join(path)}")

    # ---------------- resolver lifecycle ----------------

    def _install_template_resolver(self, ctx: AdjustmentContext) -> None:
        if ctx.scope == AdjusterScope.EXPERIMENT:
            if ctx.sample_manager is None:
                raise ValueError("EXPERIMENT adjustment requires ctx.sample_manager to bind resolver")
            tpl_resolver = ExperimentTemplatesResolver(get_defaults_provider(), sm=ctx.sample_manager,
                                                       specs_registry=INSTANCE_SPECS_REGISTRY,
                                                       policy=self._resolver_policy)
        elif ctx.scope == AdjusterScope.GROUP:
            tpl_resolver = GroupTemplatesResolver(defaults=get_defaults_provider(), group_base_dir=ctx.group_base_dir,
                                                  policy=self._resolver_policy)
        else:
            raise ValueError(f"Unknown scope: {ctx.scope}")
        set_current_resolver(tpl_resolver)

    # ---------------- tracing helpers ----------------

    def _trace_run_header(self, steps_order: Tuple["Step", ...]) -> None:
        self.t(f"[run_adjusters] phase={self.phase} steps={list(steps_order)}")
        self.t(f"[run_adjusters] active_sections={list(self.active_sections) if self.active_sections else None}")
        self.t(f"[run_adjusters] changed_keys={list(self.changed_keys) if self.changed_keys else None}")

    def _trace_spec_header(self, spec: "AdjusterSpec") -> None:
        sections = sections_from_keys(spec.owned_keys)
        self.t(f"  [spec] {spec.name} order={spec.order} keys={spec.watched_keys} {sections=}")

    def _summarize_trace_nodes(self, obj: Any, *, label: str = 'nodes') -> str:
        trace_paths = self.cfg.trace_paths or ()
        if not trace_paths:
            return ""
        formatted_nodes = summarize_paths(obj, trace_paths)
        if formatted_nodes:
            self.t(f"    [{label}] {formatted_nodes}")
        return formatted_nodes

RUNS = 0

def run_adjusters(*, view: ConfigView, ctx: AdjustmentContext, phase: "Phase" = Phase.PRE_VALIDATE,
                  active_sections: Optional[Iterable[str]] = None,
                  changed_keys: Optional[Iterable[ConfigKeys]] = None,
                  trace: bool = True, trace_paths: Optional[Iterable[str]] = None,
                  trace_sink: Optional[Callable[[str], None]] = None, schema_registry: Optional[Any] = None,
                  ) -> ConfigPatch:
    """
    Wrapper over AdjusterRunner.
    """
    # global RUNS
    # print(f"=== run_adjusters() id={RUNS} phase={phase} active_sections={active_sections} changed_keys={changed_keys} ===")
    # RUNS += 1
    # if __debug__:
    #     traceback.print_stack(limit=8)
    cfg = RunnerConfig(
        trace=trace,
        trace_paths=tuple(trace_paths) if trace_paths is not None else RunnerConfig.trace_paths,
        trace_sink=trace_sink,
    )
    runner = AdjusterRunner(phase=phase, active_sections=active_sections, changed_keys=changed_keys, config=cfg,
                            schema_registry=schema_registry, )
    with inject_sm_view(ctx.sample_manager, view=view):
        result = runner.run(view=view, ctx=ctx)
    return result
