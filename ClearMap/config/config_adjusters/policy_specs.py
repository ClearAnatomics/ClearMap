from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Callable, Dict, Any, Iterable, Literal, Tuple

from ClearMap.config.config_adjusters.dict_ops import resolve_path, normalise_dict, get_nested
from ClearMap.config.config_adjusters.type_hints import SampleManagerProtocol, ConfigView, KeysPath


class TemplateKind(StrEnum):
    """
    The semantic kind of template to use for instance container materialization.
    1) CHANNELS: per-channel templates (e.g. registration.channels)
    2) PAIRS: per-pair templates (e.g. colocalization.pairs)
    4) PERF: per-channel performance templates (e.g. performance.binarization.channels)
    """
    CHANNELS = "channels"
    PAIRS = "pairs"
    PERF = "perf"


class ContainerRole(StrEnum):
    """
    The role of the instance container in the overall config structure.
    1) SOURCE_OF_TRUTH: authoritative source of truth for the given keys.
        (typically for sample.channels)
    2) INSTANCE_RUNTIME: runtime instance container, derived from source of truth.
    """
    SOURCE_OF_TRUTH = "source_of_truth"     # sample.channels
    INSTANCE_RUNTIME = "instance_runtime"   # registration.channels, etc.


class ReconcileMode(StrEnum):
    """
    How to apply changes to existing instance containers.
    1) REPLACE_CONTAINER: fully replace the container with the new materialized one
    (including removing keys no longer present).
    2) FILL_MISSING_ONLY: only fill in missing keys,
    do not remove existing keys or overwrite present ones.
    """
    REPLACE_CONTAINER = "replace_container"  # materialize + REPLACE(new_map)
    FILL_MISSING_ONLY = "fill_missing_only"  # mutate entries missing-only, no REPLACE


class ReconcileKind(StrEnum):
    GENERIC = "generic"
    SPECIAL = "special"   # needs bespoke writer (stitching today)


@dataclass(frozen=True)
class InstancePolicy:
    """
    Policy for handling existing keys in instance containers.
    Parameters
    ----------
    preserve_existing: bool
        If True, existing keys in the instance container are preserved (not overwritten).
        If False, existing keys are overwritten with new values from the template.
    prune_removed: bool
        If True, keys that are no longer present in the template
        are removed from the instance container
    """
    preserve_existing: bool = True
    prune_removed: bool = True


@dataclass(frozen=True)
class StepSpec:
    """
    Base class for step specifications.
    """
    applies: bool = True


@dataclass(frozen=True)
class RenameStepSpec(StepSpec):
    migrate_payload: bool = True


@dataclass(frozen=True)
class ReconcileStepSpec(StepSpec):
    mode: ReconcileMode = ReconcileMode.REPLACE_CONTAINER
    kind: ReconcileKind = ReconcileKind.GENERIC
    policy: InstancePolicy = InstancePolicy()
    preserve_existing_order: bool = True
    restrict_to_template_keys: bool = False


@dataclass(frozen=True)
class Membership:
    """
    Defines the *target keyspace* for an instance container.
    The spec is declarative; `keys()` is the single execution point.
    """
    source: Literal['sample', 'pipeline']
    pipeline: Optional[str] = None

    # for non-channel keyspaces produced by a pipeline (pairs, etc.)
    instance_kind: Optional[str] = None  # e.g. "pairs" (coloc), "triplets", etc.
    oriented: bool = False

    def keys(self, sm, resolver) -> list[str]:
        if self.source == 'sample':
            if self.instance_kind is None:
                return list(sm.channels)
            if self.instance_kind == 'pairs':
                return sm.colocalization_pair_keys(oriented=self.oriented)
            raise ValueError(f'Unsupported instance_kind for source=\'sample\': {self.instance_kind!r}')

        # pipeline-backed:
        if self.pipeline is None:
            raise ValueError('Membership.pipeline is required when source="pipeline"')

        if self.instance_kind:
            # new SM capability: pipeline may yield non-channel keys (pairs, etc.)
            return sm.get_instance_keys_by_pipeline(
                self.pipeline, instance_kind=self.instance_kind, oriented=self.oriented
            )

        return sm.get_channels_by_pipeline(self.pipeline, as_list=True)


@dataclass(frozen=True)
class InstanceContainerSpec:
    # topology / identity
    section: str
    container_path: str          # dotted, '.relative' or absolute
    defaults_templates_path: str  # Where does the template for that instance live in the defaults
    template_kind: TemplateKind  # How to pick the proper templates (semantically) for that instance. logical id, resolved by DefaultsResolver

    # declarative “keys”
    membership: Membership

    # semantics
    role: ContainerRole = ContainerRole.INSTANCE_RUNTIME

    # step-specific configs
    rename: RenameStepSpec = RenameStepSpec()
    reconcile: ReconcileStepSpec = ReconcileStepSpec()

    # compound semantics TODO: move into Membership/StepSpecs
    compound: bool = False
    compound_oriented: bool = False
    compound_prune_invalid_atoms: bool = True

    # Optional per-entry context (reference, root_layout, etc.)
    ctx_builder: Optional[
        Callable[[str, dict, "SampleManagerProtocol"], Dict[str, object]]
    ] = None

    def resolve_keys(self, *, sm, resolver) -> list[str]:
        # single place; no KeyspaceKind
        return self.membership.keys(sm, resolver)

    def abs_container_path(self) -> KeysPath:
        return resolve_path(self.section, self.container_path)

    def rel_container_path(self) -> KeysPath:
        p = self.abs_container_path()
        return p[1:]  # section-relative

    def abs_templates_path(self) -> Optional[KeysPath]:
        if not self.defaults_templates_path:
            return None
        return resolve_path(self.section, self.defaults_templates_path)

    def rel_templates_path(self) -> Optional[KeysPath]:
        p = self.abs_templates_path()
        return None if p is None else p[1:]

    def container_map(self, view: ConfigView) -> dict[str, Any]:
        return normalise_dict(get_nested(view, self.abs_container_path()))

    def owns_prefix(self) -> KeysPath: # Ownership is the container itself (not section root)
        return self.abs_container_path()


class ContainerSpecRegistry:
    def __init__(self, specs: Iterable[InstanceContainerSpec]) -> None:
        self._specs = tuple(specs)

    def __iter__(self):
        return iter(self._specs)

    @property
    def specs(self) -> tuple[InstanceContainerSpec, ...]:
        return self._specs

    def for_section(self, section: str) -> tuple[InstanceContainerSpec, ...]:
        return tuple(s for s in self._specs if s.section == section)

    # ---------------- roots / exclusions ----------------

    def instance_roots(self, section: str) -> set[KeysPath]:
        return {s.rel_container_path() for s in self.for_section(section)}

    def template_roots(self, section: str) -> set[KeysPath]:
        return {p for s in self.for_section(section) if (p := s.rel_templates_path()) is not None}

    def excluded_roots(self, section: str) -> set[KeysPath]:
        """
        Excluded subtrees for STATIC fillers within a section (section-relative paths):
        - instance containers (runtime)
        - templates reservoirs
        """
        return self.instance_roots(section) | self.template_roots(section)

    def skip_paths_for_missing_only_merge(self, section: str) -> set[KeysPath]:
        # Alias for potential later splitting of logic
        return self.excluded_roots(section)

    def all_instance_container_abs_paths(self) -> tuple[KeysPath, ...]:
        return tuple(s.abs_container_path() for s in self._specs)

    # ---------------- indexing ----------------

    def by_template_kind(self) -> dict[tuple[str, TemplateKind], InstanceContainerSpec]:
        """Map (section, template_kind) -> spec."""
        out: dict[tuple[str, TemplateKind], InstanceContainerSpec] = {}
        for s in self._specs:
            key = (s.section, s.template_kind)
            if key in out:
                raise ValueError(f'Duplicate template_kind mapping for {key}: {out[key]} vs {s}')
            out[key] = s
        return out

    # ---------------- step filtering helpers ----------------

    def _instance_specs_by_kind(self, kind: ReconcileKind) -> tuple[InstanceContainerSpec, ...]:
        """
        Filter specs by reconcile kind, respecting reconcile.applies.

        Semantics:
          - if reconcile.applies is False -> excluded from all reconcile lists
          - includes only matching kind (that apply)
        """
        out: list[InstanceContainerSpec] = []
        for s in self._specs:
            if not s.reconcile.applies:
                continue

            if s.reconcile.kind == kind:
                out.append(s)

        return tuple(out)

    def generic_instance_specs(self) -> tuple[InstanceContainerSpec, ...]:
        return self._instance_specs_by_kind(ReconcileKind.GENERIC)

    def special_instance_specs(self) -> tuple[InstanceContainerSpec, ...]:
        return self._instance_specs_by_kind(ReconcileKind.SPECIAL)

    def ordered_for_reconcile(self, *, kind: ReconcileKind = ReconcileKind.GENERIC) -> tuple[InstanceContainerSpec, ...]:
        """
        Ordering for reconcile execution:
          1) role: SOURCE_OF_TRUTH before INSTANCE_RUNTIME
          2) mode: FILL_MISSING_ONLY before REPLACE_CONTAINER
          3) stable tie-breakers: section, container_path
"""
        def rank(s: InstanceContainerSpec) -> tuple[int, int, str, str]:
            role_rank = 0 if s.role == ContainerRole.SOURCE_OF_TRUTH else 1
            mode_rank = 0 if s.reconcile.mode == ReconcileMode.FILL_MISSING_ONLY else 1
            # stable tie-breakers (avoid relying on tuple order)
            return role_rank, mode_rank, s.section, s.container_path

        specs = self._instance_specs_by_kind(kind)
        return tuple(sorted(specs, key=rank))


@dataclass(frozen=True)
class SmellScanPolicy:
    # Strong: template-ish key names outside templates roots
    strong_template_smells: bool = True

    # Light: channel-ish keys at reserved namespaces like performance
    light_channelish_smells: bool = True

    # Reserved namespaces where "performance.<channel>" is suspicious if performance.channels exists
    reserved_namespace_keys: Tuple[str, ...] = ("performance",)

    # Template-ish key names
    templateish_exact: Tuple[str, ...] = ("templates", "channel", "layout_template", "derived_template")

    # Suffix-based heuristic
    templateish_suffix: str = "_template"
