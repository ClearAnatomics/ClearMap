"""
change_detection
================

Functions that classify config change keys.
Typically used downstream of `CfgChanged` events
"""
from typing import Tuple

IRRELEVANT_DATA_TYPES = frozenset({'undefined', 'no-pipeline', None})



def whole_sample_changed(changed_keys: Tuple[str, ...]) -> bool:
    """Whether the entire sample section was replaced."""
    return 'sample' in changed_keys


def channels_added_or_removed(changed_keys: Tuple[str, ...]) -> bool:
    """Whether the channels list itself changed (add/remove, not just property edit)."""
    for k in changed_keys:
        if k == 'sample' or k == 'sample.channels':
            return True
        if k.startswith('sample.channels.') and k.count('.') == 2:
            return True
    return False


def any_channel_property_changed(changed_keys: Tuple[str, ...], suffix: str) -> bool:
    """Whether any channel had the given property suffix changed (e.g. '.path', '.data_type')."""
    if whole_sample_changed(changed_keys):
        return True
    return any(
        k.startswith('sample.channels.') and k.endswith(suffix)
        for k in changed_keys
    )


def channel_path_changed(changed_keys: Tuple[str, ...]) -> bool:
    """Whether any channel's raw path was set or changed."""
    return any_channel_property_changed(changed_keys, '.path')


def channel_data_type_changed(changed_keys: Tuple[str, ...]) -> bool:
    """Whether any channel's data_type was modified."""
    return any_channel_property_changed(changed_keys, '.data_type')


def extract_changed_channel_names(changed_keys: Tuple[str, ...]) -> set[str]:
    """Extract the set of channel names touched by the change."""
    names = set()
    for k in changed_keys:
        if k.startswith('sample.channels.'):
            parts = k.split('.')
            if len(parts) >= 3:
                names.add(parts[2])
    return names

def channels_with_changed_property(changed_keys: Tuple[str, ...], prop: str) -> set[str]:
    """
    Extract channel names where a specific property changed.

    Parameters
    ----------
    changed_keys : Tuple[str, ...]
        The tuple of changed config keys.
    prop : str
        The property to extract channel names from.
        (e.g. '.data_type', '.path')

    Returns
    -------
    set[str]
        The set of channel names that had the specified property changed.
    """
    names = set()
    for k in changed_keys:
        if k.startswith('sample.channels.') and k.endswith(prop):
            parts = k.split('.')
            if len(parts) >= 4:
                names.add(parts[2])
    return names


def _pipeline_for_dtype(data_type: str | None) -> str | None:
    """Map a data_type to its pipeline name, or None."""
    from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE
    return CONTENT_TYPE_TO_PIPELINE.get(data_type)
