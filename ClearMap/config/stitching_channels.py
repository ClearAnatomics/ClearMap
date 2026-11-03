"""
Which layout each channel is stitched with, read from the stitching config: each channel of ``channels`` is stitched
with the layout of its ``layout_channel``, its own when it names itself, else the one that channel is stitched with.
The channels which are not tiled are not stitched: they have no entry.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from ClearMap.Utils.exceptions import ChannelsNotPixelAlignedError, ClearMapConfigError

LAYOUT_NOT_CHOSEN = (None, '', 'undefined')  # layout_channel of a channel whose layout is not chosen yet


def layout_root(stitching_config: Mapping, channel: str) -> str:
    """
    The channel whose layout channel is stitched with: its layout_channel, followed until a channel computes its
    own layout (names itself).

    Raises
    ------
    ClearMapConfigError
        If a layout_channel along the way is not chosen, is not stitched, or the layout channels make a loop.
    """
    channels_config = stitching_config.get('channels', {})
    followed = [channel]
    while True:
        layout_channel = channels_config[followed[-1]].get('layout_channel')
        if layout_channel in LAYOUT_NOT_CHOSEN:
            raise ClearMapConfigError(f'The layout of the channel "{followed[-1]}" is not chosen '
                                      f'(stitching.channels.{followed[-1]}.layout_channel).')
        if layout_channel == followed[-1]:
            return layout_channel
        if layout_channel not in channels_config:
            raise ClearMapConfigError(f'The channel "{followed[-1]}" is stitched with the layout of '
                                      f'"{layout_channel}", which is not stitched.')
        if layout_channel in followed:
            raise ClearMapConfigError(f'The layout channels make a loop: {" -> ".join(followed + [layout_channel])}.')
        followed.append(layout_channel)


def check_stitched_together(stitching_config: Mapping, channels: Iterable[str]) -> None:
    """
    Check that the channels which are stitched are stitched with the same layout (that of the same channel), so
    that their tiles are placed the same way. The channels which are not stitched (not tiled) are not concerned.

    Raises
    ------
    ChannelsNotPixelAlignedError
        If they are stitched with different layouts, or the layout of one of them cannot be told.
    """
    channels = tuple(channels)
    stitched = [channel for channel in channels if channel in stitching_config.get('channels', {})]
    roots = {}
    unmet_requirements = []
    for channel in stitched:
        try:
            roots[channel] = layout_root(stitching_config, channel)
        except ClearMapConfigError as err:
            unmet_requirements.append(str(err))
    if len(set(roots.values())) > 1:
        unmet_requirements.append(f'they must be stitched with the same layout, got '
                                  f'{", ".join(f"{channel}: the layout of {root}" for channel, root in roots.items())}')
    if unmet_requirements:
        raise ChannelsNotPixelAlignedError(channels, unmet_requirements)
