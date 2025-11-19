from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Tuple, Dict, Union, Optional

from .tabs_interfaces import GenericTab, PreProcessingTab, PostProcessingTab, BatchTab

if TYPE_CHECKING:
    from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager

# Coarse (abstract) hierarchy group: 1..99 (0 and 100 reserved for absolute 1st and last)
BASE_GROUP: Dict[type[GenericTab], int] = {
    PreProcessingTab: 10,
    PostProcessingTab: 20,
    BatchTab:         30,
    GenericTab:       90,  # fallback
}

# Within-group score for concrete classes: 0..100
CONCRETE_TIER: Dict[Union[str, type[GenericTab]], int] = {
    'StitchingTab': 30,
    'RegistrationTab': 35,
    'ColocalizationTab': 80,
}

# Optional hard guards
ABSOLUTE_FIRST: set[Union[str, type[GenericTab]]] = {'SampleInfoTab'}
ABSOLUTE_LAST:  set[Union[str, type[GenericTab]]] = set()

DEFAULT_WITHIN = 50    # if a concrete class isn’t in CONCRETE_TIER
FALLBACK_GROUP = 90    # if no base in BASE_GROUP matches

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))

def _abs_priority(cls: type[GenericTab]) -> int | None:
    """Return 0 for absolute-first, 100 for absolute-last, or None for normal."""
    name = cls.__name__
    if name in ABSOLUTE_FIRST or cls in ABSOLUTE_FIRST:
        return 0
    if name in ABSOLUTE_LAST or cls in ABSOLUTE_LAST:
        return 100
    return None

def _resolve_group(cls: type[GenericTab]) -> int:
    for base in cls.__mro__[1:]:
        if base is object:
            break
        g = BASE_GROUP.get(base)
        if g is not None:
            return _clamp(g, 1, 99)
    return FALLBACK_GROUP

def _resolve_within(cls: type[GenericTab]) -> int:
    # concrete-class-specific within-group score (by name or class), else default
    priority = CONCRETE_TIER.get(cls, CONCRETE_TIER.get(cls.__name__))
    if priority is not None:
        return _clamp(priority, 0, 100)
    return DEFAULT_WITHIN

def _computed_score(cls: type[GenericTab]) -> float:
    group = _resolve_group(cls)
    within = _resolve_within(cls)
    return float(group) + (within / 100.0)

def global_score(cls: type[GenericTab]) -> float:
    """
    Absolute priorities override computed scores.
    0 = hard first, 100 = hard last, otherwise group+within/100.
    """
    abs_p = _abs_priority(cls)
    if abs_p is not None:
        return float(abs_p)
    return _computed_score(cls)

def _order_key(cls: type[GenericTab]) -> Tuple[float, str]:
    """
    Sort by:
      1) global score (absolute priority or computed)
      2) name for deterministic tiebreak.
    """
    return global_score(cls), cls.__name__


class TabRegistry:  # FIXME: ensure that colocalization tab existence is triggered when needed
    """
    Computes the set of tabs to display based on the current config view and sample state.
    """

    def __init__(self) -> None:
        # Start with declared keys and no loaded classes yet.
        self._tabs: dict[str, type[GenericTab] | None] = {
            'SampleInfoTab': None,
            'StitchingTab': None,
            'RegistrationTab': None,
            'GroupAnalysisTab': None,
            'BatchProcessingTab': None,
            'ColocalizationTab': None,
        }
        self._DATA_TYPE_TO_TAB_CLASS: dict[Any, type[GenericTab]] | None = None

    def __get_tabs(self, *tab_names: str) -> list[type[GenericTab]]:
        self._ensure_loaded(*tab_names)
        return [self._tabs[name] for name in tab_names if self._tabs[name] is not None]

    @property
    def _base_tabs(self):
        return self.__get_tabs('SampleInfoTab')

    @property
    def _compound_tabs(self):
        return self.__get_tabs('ColocalizationTab')

    @property
    def _batch_tabs(self):
        return self.__get_tabs('GroupAnalysisTab', 'BatchProcessingTab')

    @property
    def _preprocessing_tabs(self):
        return self.__get_tabs('StitchingTab', 'RegistrationTab')

    # ── Lazy import + caching ──────────────────────────────────────────
    def _ensure_loaded(self, *names: str) -> None:
        """
        Import and cache requested tabs by name if not already loaded.
        If called with no names, loads all known tabs.
        """
        to_load = [nm for nm in (names or self._tabs.keys()) if self._tabs[nm] is None]
        if not to_load and self._DATA_TYPE_TO_TAB_CLASS is not None:
            return

        from . import tabs  # heavy import, done once

        for cls_name in to_load:
            try:
                self._tabs[cls_name] = getattr(tabs, cls_name)
            except AttributeError:
                raise RuntimeError(f'Tab class {cls_name} not found in .tabs module')

        if self._DATA_TYPE_TO_TAB_CLASS is None:
            self._DATA_TYPE_TO_TAB_CLASS = getattr(tabs, 'DATA_TYPE_TO_TAB_CLASS')

    @staticmethod
    def _append_if(acc: list[type[GenericTab]], tab_cls: Optional[type[GenericTab]], sample_manager: SampleManager) -> None:
        if tab_cls and (tab_cls not in acc) and tab_cls.requirements_fulfilled(sample_manager):
            acc.append(tab_cls)

    @classmethod
    def _filter_by_requirements(cls, candidates: list[type[GenericTab]], sample_manager: SampleManager, *,
                                extra_requires: Callable[[Any], bool] | None = None,
                                existing: list[type[GenericTab]] | None = None) -> list[type[GenericTab]]:
        extra_ok = extra_requires(sample_manager) if extra_requires is not None else True
        existing_set = set(existing or [])
        return [tab_cls for tab_cls in candidates
                if (tab_cls not in existing_set) and tab_cls.requirements_fulfilled(sample_manager) and extra_ok
                ]

    def valid_tabs(self, sample_manager: "SampleManager") -> list[type[GenericTab]]:
        # self._ensure_loaded()  # lazy load on first real use

        required: list[type[GenericTab]] = self._base_tabs

        # regular pipeline tabs
        for ch in sample_manager.channels:
            data_cls = self._DATA_TYPE_TO_TAB_CLASS.get(sample_manager.data_type(ch))
            self._append_if(required, data_cls, sample_manager)

        # preprocessing
        for tab_cls in self._preprocessing_tabs:  # As soon as we have a sample
            self._append_if(required, tab_cls, sample_manager)

        # Compound tabs (e.g., colocalization)
        def coloc_compatible(sm: "SampleManager") -> bool:
            return sm.is_colocalization_compatible

        required += self._filter_by_requirements(self._compound_tabs, sample_manager,
                                                 extra_requires=coloc_compatible, existing=required)

        # Batch tabs
        required += self._filter_by_requirements(self._batch_tabs, sample_manager, existing=required)

        return self._sort_tabs(required)

    def _sort_tabs(self, classes: list[type[GenericTab]]) -> list[type[GenericTab]]:
        """Deduplicate and sort using global_score() helper."""
        ordered = []
        for c in classes:
            if c not in ordered:
                ordered.append(c)
        ordered.sort(key=_order_key)
        return ordered
