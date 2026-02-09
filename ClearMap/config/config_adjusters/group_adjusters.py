from copy import deepcopy

from ClearMap.Utils.utilities import _ensure_list, _dedupe_preserve_order
from ClearMap.config.config_adjusters.policy import DEFAULTS_PROVIDER
from ClearMap.config.config_adjusters.adjusters_api import Phase, Step, patch_adjuster
from ClearMap.config.config_adjusters.type_hints import ConfigView, SampleManagerProtocol, ConfigPatch


@patch_adjuster(step=Step.CREATE_PIPELINE_SECTIONS, phase=Phase.PRE_VALIDATE, sections=('group_analysis',),
                watched_keys=('group_analysis',), owned_keys=None, order=10)
def populate_group_analysis_defaults(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """Seed group_analysis with defaults (paths, groups) without overwriting user values."""
    defaults = DEFAULTS_PROVIDER.get('group_analysis')
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


@patch_adjuster(step=Step.ADJUST, phase=Phase.PRE_VALIDATE, sections=('group_analysis',),
                watched_keys=('group_analysis.groups',), owned_keys=None, order=20)
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


@patch_adjuster(step=Step.CREATE_PIPELINE_SECTIONS, phase=Phase.PRE_VALIDATE, sections=('batch_processing',),
                watched_keys=('batch_processing',), owned_keys=None, order=10)
def populate_batch_defaults(view: ConfigView, sm: SampleManagerProtocol) -> ConfigPatch:
    """Seed batch_processing with defaults (paths, groups, comparisons) without overwriting user values."""
    defaults = DEFAULTS_PROVIDER.get('batch_processing')
    section = deepcopy(view.get('batch_processing') or {})
    before = deepcopy(section)

    section.setdefault('paths', deepcopy(defaults.get('paths') or {}))
    section.setdefault('groups', {})
    section.setdefault('comparisons', [])

    if section != before:
        return {'batch_processing': section}
    return {}


@patch_adjuster(step=Step.ADJUST, phase=Phase.PRE_VALIDATE, sections=('batch_processing',),
                watched_keys=('batch_processing.groups', 'batch_processing.comparisons'), owned_keys=None, order=20)
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
