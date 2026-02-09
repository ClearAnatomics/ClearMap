from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from copy import deepcopy

from ClearMap.config.compound_keys import PairKey
from ClearMap.config.config_adjusters.type_hints import SampleManagerProtocol
from ClearMap.config.defaults_provider import DefaultsProvider


from ClearMap.config.config_adjusters.policy_specs import ContainerSpecRegistry, TemplateKind, InstanceContainerSpec


def require_dict(parent: Mapping[str, Any], key: str, *, path: tuple[str, ...] = (), allow_missing: bool = False) -> dict[str, Any]:
    """
    Extract a dict-valued child from `parent[key]`.

    Parameters
    ----------
    parent:
        Mapping to extract from.
    key:
        Key to extract.
    path:
        Cosmetic only; dotted path prefix used for error reporting.
    allow_missing:
        If True, missing key returns {}. If False (default), missing key raises.

    Returns
    -------
    dict[str, Any]
        The dict value at parent[key].

    Raises
    ------
    KeyError
        If key is missing and allow_missing is False.
    ValueError
        If key exists but is not a dict.
    """
    dotted = '.'.join(path + (key,)) if path else key

    if key not in parent:
        if allow_missing:
            return {}
        raise KeyError(f'Defaults invariant violated: missing key {dotted}')

    v = parent.get(key)
    if not isinstance(v, dict):
        raise ValueError(f'Defaults invariant violated: {dotted} must be a dict,'
                         f' got {type(v).__name__} with value: {v!r}')
    return v


@dataclass(frozen=True)
class RunFacts:
    """
    Snapshot of SM-derived facts used for binding templates in a *single* resolver instance.

    Notes
    -----
    - renamed_map provides ergonomic lookup old->new.
    - renamed_items provides deterministic iteration and hash-friendly representation (if needed).
    """
    sample_channels: Tuple[str, ...]
    alignment_reference: Optional[str]
    renamed_map: Mapping[str, str]
    renamed_items: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class BindPolicy:
    """
    Central binding policy knobs for the resolver.
    This is intentionally narrow: "binding" policy, not "reconcile" policy.
    """
    keep_templates_out_of_working_config: bool = True
    allow_entry_overwrite: bool = False  # templates are generally missing-only in adjusters


# ---------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------

class TemplatesResolver:
    """
    Binds defaults templates to a specific run context (sm + defaults) and offers
    *pure* template selection/expansion utilities.

    Design intent
    -------------
    - This object should NOT mutate config.
    - It should NOT decide reconciliation strategy (preserve vs overwrite vs prune).
    - It should provide:
        * target membership lists
        * fully-expanded template dicts for a given entry key
        * canonicalization helpers for compound keys
        * stable per-run facts (RunFacts)

    About "alignment_reference_channel changes"
    -------------------------------------------
    If you allow earlier steps to rename channels, then the "reference channel" may
    need to track those renames. This resolver addresses that in two ways:

    1) It snapshots the SM-reported reference and then normalizes it through the
       rename map (resolve_renamed_channel). This prevents a stale reference if SM
       reports the "old" name but the run applies renames.

    2) If your SM is itself config-dependent and can change during the run, you
       should instantiate a new resolver after the rename/reconcile step (or at least
       rebuild facts). This keeps binding coherent with the current working state.
    """

    PAIRS_ORIENTED = True

    _TEMPLATE_BINDERS = {
        ("sample", TemplateKind.CHANNELS): 'sample_channel_entry_template',
        ("registration", TemplateKind.CHANNELS): 'registration_entry_template',
        ("stitching", TemplateKind.CHANNELS): 'stitching_entry_template',
        ("cell_map", TemplateKind.CHANNELS): 'cell_map_entry_template',
        ('cell_map', TemplateKind.PERF): 'cell_map_perf_entry_template',
        ("tract_map", TemplateKind.CHANNELS): 'tract_map_entry_template',
        ("vasculature", TemplateKind.CHANNELS): 'vasculature_binarization_template',
        ("vasculature", TemplateKind.PERF): 'vasculature_perf_binarization_template',
        ("colocalization", TemplateKind.PAIRS): 'colocalization_entry_template',
    }

    def __init__(self, defaults: DefaultsProvider, sm: SampleManagerProtocol, *,
                 specs_registry: ContainerSpecRegistry, policy: BindPolicy = BindPolicy()) -> None:
        self._defaults = defaults
        self._sm = sm
        self._policy = policy
        self._specs = specs_registry.by_template_kind()

        self._validate_template_binders()

        # Snapshot, but normalize reference through rename map to avoid stale references.
        renamed_map: Mapping[str, str] = dict(sm.renamed_channels or {})
        renamed_items: Tuple[Tuple[str, str], ...] = tuple(renamed_map.items())

        sample_channels = tuple(sm.channels or ())
        ref = sm.alignment_reference_channel
        ref = self.resolve_renamed_channel(ref, renamed_map=renamed_map)

        self._facts = RunFacts( sample_channels=sample_channels, alignment_reference=ref,
                                renamed_map=renamed_map, renamed_items=renamed_items)

    def _validate_template_binders(self):
        for key, method_name in self._TEMPLATE_BINDERS.items():
            # if key not in self._specs:  # We're allowed not to have specs for all bindings
            #     raise KeyError(f"No InstanceContainerSpec for template binding {key}")
            _ = self.get_template_method(key)
        for (section, kind), spec in self._specs.items():
            if (section, kind) not in self._TEMPLATE_BINDERS:
                raise KeyError(f"No template binding for InstanceContainerSpec {(section, kind)}")

    def get_template_method(self, key):
        method_name = self._TEMPLATE_BINDERS.get(key)
        if method_name is None:
            raise KeyError(f"No template binding for {key}")

        fn = getattr(self, method_name, None)
        if not callable(fn):
            raise RuntimeError(f"Template binding {method_name!r} not found or not callable")

        return fn

    # def _templates_and_base(self, *, section: str, kind: TemplateKind, base_key: str) -> tuple[
    #     dict[str, Any], dict[str, Any]]:
    #     root = self._templates_root(section=section, kind=kind)
    #     spec = self._specs[(section, kind)]
    #     path = spec.abs_templates_path()
    #     base = require_dict(root, base_key, path=*path)
    #     return root, base

    @property
    def defaults(self) -> DefaultsProvider:
        return self._defaults

    @property
    def sm(self) -> SampleManagerProtocol:
        return self._sm

    @property
    def policy(self) -> BindPolicy:
        return self._policy

    def run_facts(self) -> RunFacts:
        return self._facts

    # ---- common utilities

    @staticmethod
    def resolve_renamed_channel(name: Optional[str], *, renamed_map: Mapping[str, str],
                                max_hops: int = 32) -> Optional[str]:
        """
        Follow rename chains old->new transitively to avoid stale references.
        Protects against cycles via max_hops.
        """
        if name is None:
            return None
        cur = name
        hops = 0
        while cur in renamed_map and hops < max_hops:
            nxt = renamed_map[cur]
            if nxt == cur:
                break
            cur = nxt
            hops += 1
        return cur

    def expand(self, obj: Any, *, channel: str, reference: Optional[str] = None) -> Any:
        """Recursively substitute ${channel} and ${reference} in strings within obj."""
        def sub_one(s: str) -> str:
            out = s.replace('${channel}', channel)
            if reference is not None:
                out = out.replace('${reference}', reference)
            return out

        if isinstance(obj, str):
            return sub_one(obj)
        if isinstance(obj, list):
            return [self.expand(x, channel=channel, reference=reference) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self.expand(x, channel=channel, reference=reference) for x in obj)
        if isinstance(obj, dict):
            return {k: self.expand(v, channel=channel, reference=reference) for k, v in obj.items()}
        return obj

    def entry_template(self, *, section: str, kind: 'TemplateKind', key: str, **ctx: Any) -> Dict[str, Any]:
        fn = self.get_template_method((section, kind))
        # Most bindings accept (key) or (channel), stitching accepts root_layout via ctx.
        return fn(key, **ctx)

    def _get_nested(self, root: Any, path: tuple[str, ...]) -> Any:
        node = root
        for p in path:
            if not isinstance(node, dict):
                return None
            node = node.get(p)
        return node

    def _templates_root_and_spec(self, *, section: str, kind: 'TemplateKind') -> tuple[dict[str, Any], InstanceContainerSpec]:
        spec = self._specs.get((section, kind))
        if spec is None:
            raise KeyError(f'No InstanceContainerSpec for {(section, kind)}')

        rel = spec.rel_templates_path()
        if rel is None:
            raise KeyError(f'{(section, kind)} has no defaults_templates_path')

        # defaults are section-rooted, and rel_templates_path is section-relative
        sec_defaults = self._defaults.get(section)
        node = self._get_nested(sec_defaults, rel)
        root = deepcopy(node) if isinstance(node, dict) else {}
        return root, spec

    def _require_template_dict(self, *, section: str, kind: TemplateKind, key: str) -> dict[str, Any]:
        root, spec = self._templates_root_and_spec(section=section, kind=kind)
        prefix = spec.abs_templates_path() or (section,)  # defensive fallback
        return require_dict(root, key, path=prefix)

    # -----------------------------------------------------------------
    # Sample-level defaults
    # -----------------------------------------------------------------

    def sample_channel_entry_template(self, channel: str) -> dict[str, Any]:
        return self._require_template_dict(section='sample', kind=TemplateKind.CHANNELS, key='channel')

    # -----------------------------------------------------------------
    # Stitching
    # -----------------------------------------------------------------

    def stitching_target_channels(self) -> list[str]:
        return self._sm.stitchable_channels

    def stitching_entry_template(self, channel: str, *, root_layout: Optional[str]) -> Dict[str, Any]:
        """
        Returns an expanded stitching template for `channel`.
        The caller decides role (layout vs derived) and any canonicalization.
        """
        templates, spec = self._templates_root_and_spec(section='stitching', kind=TemplateKind.CHANNELS)
        err_path = spec.abs_templates_path()
        layout_tpl = require_dict(templates, 'layout_template', path=err_path)
        derived_tpl = require_dict(templates, 'derived_template', path=err_path)

        # If root_layout is None or equals channel, we consider it a layout role template bind.
        if root_layout is None or root_layout == channel:
            return deepcopy(self.expand(layout_tpl, channel=channel, reference=channel))
        return deepcopy(self.expand(derived_tpl, channel=channel, reference=root_layout))

    @property
    def stitching_template_keys(self):
        return 'layout_template', 'derived_template'

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------

    def registration_target_channels(self) -> list[str]:
        return list(self._facts.sample_channels)

    def registration_entry_template(self, channel: str) -> Dict[str, Any]:
        if self._sm.data_type(channel) == 'autofluorescence':
            tpl_channel_key = 'autofluorescence'
            reference = 'atlas'
        else:
            tpl_channel_key = 'regular'
            reference = self._facts.alignment_reference or channel
        base = self._require_template_dict(section='registration', kind=TemplateKind.CHANNELS, key=tpl_channel_key)

        return deepcopy(self.expand(base, channel=channel, reference=reference))

    # -----------------------------------------------------------------
    # Cell map / tract map
    # -----------------------------------------------------------------

    def cell_map_target_channels(self) -> list[str]:
        return self._sm.get_channels_by_pipeline('CellMap', as_list=True)

    def cell_map_entry_template(self, channel: str) -> Dict[str, Any]:
        base = self._require_template_dict(section='cell_map', kind=TemplateKind.CHANNELS, key='channel')
        return deepcopy(self.expand(base, channel=channel, reference=self._facts.alignment_reference))

    def cell_map_perf_entry_template(self, channel: str) -> Dict[str, Any]:
        base = self._require_template_dict(section='cell_map', kind=TemplateKind.PERF, key='channel')
        return deepcopy(self.expand(base, channel=channel, reference=self._facts.alignment_reference))

    def tract_map_target_channels(self) -> list[str]:
        return self._sm.get_channels_by_pipeline('TractMap', as_list=True)

    def tract_map_entry_template(self, channel: str) -> Dict[str, Any]:
        base = self._require_template_dict(section='tract_map', kind=TemplateKind.CHANNELS, key='channel')
        return deepcopy(self.expand(base, channel=channel, reference=self._facts.alignment_reference))

    # -----------------------------------------------------------------
    # Colocalization (compound)
    # -----------------------------------------------------------------

    def colocalization_entry_template(self, canonical_pair_key: str) -> Dict[str, Any]:
        # templates_root = self._templates_root_and_spec(section='colocalization', kind=TemplateKind.PAIRS)
        # base = require_dict(templates_root, 'channel', path=('colocalization', 'templates'))
        base = self._require_template_dict(section='colocalization', kind=TemplateKind.PAIRS, key='channel')
        pk = PairKey.from_string(canonical_pair_key, oriented=False)
        expanded = self.expand(base, channel=str(pk), reference=self._facts.alignment_reference)

        if isinstance(expanded, dict):
            expanded = {**expanded, 'channels': [pk.a, pk.b]}
        return deepcopy(expanded)

    # -----------------------------------------------------------------
    # Vasculature
    # -----------------------------------------------------------------

    def vasculature_target_channels(self) -> list[str]:
        return self._sm.get_channels_by_pipeline('TubeMap', as_list=True)

    def vasculature_binarization_template(self, channel: str) -> Dict[str, Any]:
        # Select template by data type
        tpl_chan_k = 'vessels_template' if self._sm.data_type(channel) == 'vessels' else 'large_vessels_template'
        return self._require_template_dict(section='vasculature', kind=TemplateKind.CHANNELS, key=tpl_chan_k)

    def vasculature_perf_binarization_template(self, channel: str) -> Dict[str, Any]:
        # Select template by data type
        tpl_chan_k = 'vessels_template' if self._sm.data_type(channel) == 'vessels' else 'large_vessels_template'
        return self._require_template_dict(section='vasculature', kind=TemplateKind.PERF, key=tpl_chan_k)

    # -----------------------------------------------------------------
    # Missing-only section blocks (non-channel)
    # -----------------------------------------------------------------

    # FIXME: do we exclude templates here, or leave to caller?
    def section_missing_only_defaults(self, section_name: str) -> Dict[str, Any]:
        """
        Returns the raw defaults dict for a section (caller performs missing-only merge).
        If keep_templates_out_of_working_config is True, this function *does not* strip
        templates here (because the correct strip scope is section-specific).
        The adjuster can do template skipping in the deep-merge implementation.
        """
        d = self._defaults.get(section_name)
        return deepcopy(d) if isinstance(d, dict) else {}


# ----------------- Runner-managed resolver instance -----------------

_CURRENT_RESOLVER: Optional[TemplatesResolver] = None


def set_current_resolver(resolver: TemplatesResolver) -> Optional[TemplatesResolver]:
    global _CURRENT_RESOLVER
    old = _CURRENT_RESOLVER
    _CURRENT_RESOLVER = resolver
    return old


def get_current_resolver() -> TemplatesResolver:
    if _CURRENT_RESOLVER is None:
        raise RuntimeError('No TemplatesResolver has been installed for this run')
    return _CURRENT_RESOLVER
