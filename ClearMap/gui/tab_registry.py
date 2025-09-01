from __future__ import annotations

from typing import List, Type

from .interfaces import PreProcessingTab, BatchTab, GenericTab, PostProcessingTab
from .tabs import (
    SampleInfoTab,
    StitchingTab, RegistrationTab,
    GroupAnalysisTab, BatchProcessingTab,
    ColocalizationTab,
    DATA_TYPE_TO_TAB_CLASS
)
from ..pipeline_orchestrators.sample_preparation import SampleManager


# Ensure a certain order for key tabs (anchors). Others will be sorted around them.
# The order here is from left to right in the UI.
TAB_ORDER = [
    SampleInfoTab,             # GenericTab
    StitchingTab,              # PreProcessingTab
    RegistrationTab,           # PreProcessingTab
    # <other tabs inserted dynamically here>
    ColocalizationTab,          # PostProcesingTab
    GroupAnalysisTab,           # BatchTab
    BatchProcessingTab,         # BatchTab
]

_ORDER_INDEX = {cls: i for i, cls in enumerate(TAB_ORDER)}


BUCKET_ORDER = {
    SampleInfoTab: 0,                             # handled by anchor, but keep for completeness
    PreProcessingTab: 1,                          # all other pre tabs
    PostProcessingTab: 2,                         # all other post tabs
    ColocalizationTab: 2,                         # also post (anchored above)
    GroupAnalysisTab: 3, BatchProcessingTab: 3,   # utilities
    BatchTab: 3,
    GenericTab: 99,                               # fallback
}
def _bucket_index(cls: Type[GenericTab]) -> int:
    # find the most specific matching base class from BUCKET_ORDER
    for base, idx in BUCKET_ORDER.items():
        if issubclass(cls, base):
            return idx
    return 99

def _order_key(cls: Type[GenericTab]) -> tuple[int, int, str]:
    """
    Sort tab classes with a two-tier key:
        - primary: explicit TAB_ORDER index (anchors first),
        - secondary: bucket precedence (pre before post before utilities),
        - tertiary: stable tie-breaker (class name) so it’s deterministic.

    Parameters
    ----------
    cls: Type[GenericTab]
        The tab class to evaluate.

    Returns
    -------
    tuple[int, int, str]
        The sorting key.
    """
    anchor = _ORDER_INDEX.get(cls, 10_000)     # huge number if not anchored
    bucket = _bucket_index(cls)
    return anchor, bucket, cls.__name__


class TabRegistry:
    """
    Computes the set of tabs to display based on the current config view and sample state.
    """

    def __init__(self) -> None:
        # base tabs always present
        self._base = [SampleInfoTab]
        self._preprocessing_tabs = [StitchingTab, RegistrationTab]
        self._batch_tabs = [GroupAnalysisTab, BatchProcessingTab]

    def valid_tabs(self, sample_manager: SampleManager) -> List[Type]:
        """
        Decide which tab classes should exist.

        .. warning::

            sample_manager is required to determine per-channel and compound tabs.
            It's config should be up-to-date with `view`.
        """
        classes: list[Type[GenericTab]] = self._base + self._preprocessing_tabs

        # Add per-channel tabs
        for ch in sample_manager.channels:
            cls = DATA_TYPE_TO_TAB_CLASS.get(sample_manager.data_type(ch))
            if cls and cls not in classes:
                classes.append(cls)

        self._handle_compound_tabs(classes, sample_manager)

        classes += self._batch_tabs

        # De-duplicate while preserving order
        ordered = self._sort_tabs(classes)
        return ordered

    def _sort_tabs(self, classes):
        seen = set()
        ordered = []
        for c in classes:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        ordered.sort(key=_order_key)
        return ordered

    def _handle_compound_tabs(self, classes: List[Type], sample_manager: SampleManager) -> None:
        """Add compound tabs (e.g., colocalization) when applicable"""
        if getattr(sample_manager, "is_colocalization_compatible", False):
            if ColocalizationTab not in classes:
                classes.append(ColocalizationTab)
