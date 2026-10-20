"""
Which channels each registration aligns, read from the registration config: each channel of
``channels`` says what it is aligned with (``align_with``) and which of the two moves (``moving_channel``, a name,
or relative to the channel: ``self`` or ``other``, as in the dunder methods). The other one is fixed.
The results of a registration are in the folder ``<moving>_to_<fixed>``.
"""
from __future__ import annotations

from typing import Mapping

from ClearMap.Utils.exceptions import MissingRequirementException

ATLAS = 'atlas'  # align_with / moving_channel of the atlas: a partner which is not a channel of the sample
ALIGN_WITH_NOTHING = None  # align_with of a channel which is explicitly not registered
ALIGN_WITH_UNDECIDED = ''  # align_with of a channel whose partner is not chosen yet (in the GUI)
NOT_SELECTED_FOR_REGISTRATION = (ALIGN_WITH_NOTHING, ALIGN_WITH_UNDECIDED)
INTRINSICALLY_ALIGNED = 'intrinsically_aligned'  # moving_channel of a channel aligned by construction: needs no registration
MOVING_SELF = 'self'  # moving_channel: this channel moves (onto its align_with)
MOVING_OTHER = 'other'  # moving_channel: its align_with moves (onto this channel)


def is_selected_for_registration(channel_config: Mapping) -> bool:
    """Whether the channel is aligned with something (a partner is chosen, and it is not explicitly nothing)"""
    return channel_config.get('align_with') not in NOT_SELECTED_FOR_REGISTRATION


def is_explicitly_not_registered(channel_config: Mapping) -> bool:
    """Whether the channel is aligned with nothing, by choice (not: its partner is not chosen yet)"""
    return channel_config.get('align_with') is ALIGN_WITH_NOTHING


def may_be_registered(channel_config: Mapping) -> bool:
    """
    Whether the channel is not explicitly excluded from registration: it is selected, or its partner is not
    chosen yet. E.g. the channels which may be the partner of another one.
    """
    return not is_explicitly_not_registered(channel_config)


def is_intrinsically_aligned(channel_config: Mapping) -> bool:
    """Whether the channel is aligned by construction (e.g. acquired with the reference channel)"""
    return channel_config.get('moving_channel') == INTRINSICALLY_ALIGNED


def moving_channel_of(registration_config: Mapping, channel: str) -> str | None:
    """
    The moving channel of the registration of channel, by name: self and other resolved (other is None when the
    partner is not chosen, or explicitly nothing).
    """
    channel_config = registration_config['channels'][channel]
    moving_channel = channel_config.get('moving_channel')
    if moving_channel == MOVING_SELF:
        return channel
    if moving_channel == MOVING_OTHER:
        return channel_config.get('align_with') or None
    return moving_channel


def needs_registration(channel_config: Mapping) -> bool:
    """
    Whether a registration is computed for the channel: something moves, and it is not intrinsically aligned.
    (Whether the channel is selected for registration at all is :func:`is_selected_for_registration`.)
    """
    return channel_config.get('moving_channel') is not None and not is_intrinsically_aligned(channel_config)


def fixed_moving_channels(registration_config: Mapping, channel: str) -> tuple[str | None, str | None]:
    """
    The fixed and the moving channels of the registration of channel.

    Parameters
    ----------
    registration_config : Mapping
        The registration config (its ``channels`` section is read).
    channel : str
        The channel whose registration it is.

    Returns
    -------
    tuple[str | None, str | None]
        (fixed, moving), by name (see :func:`moving_channel_of`). The fixed channel is None if the channel is
        aligned with nothing. The fixed channel is whichever of (channel, align_with) is not the moving one.

    Raises
    ------
    MissingRequirementException
        If the partner of the channel is not chosen yet (align_with undecided).
    """
    channel_config = registration_config['channels'][channel]
    moving_channel = moving_channel_of(registration_config, channel)
    align_with = channel_config['align_with']
    if align_with is ALIGN_WITH_NOTHING:
        return None, moving_channel
    if align_with == ALIGN_WITH_UNDECIDED:
        raise MissingRequirementException(f'Channel {channel} missing align_with in registration config')
    fixed_channel = channel if align_with == moving_channel else align_with
    return fixed_channel, moving_channel


def registration_pairs(registration_config: Mapping) -> list[tuple[str, str]]:
    """
    The (moving, fixed) channels of every registration of the config which is computed, in the order of its
    channels. A channel which is not selected, whose partner is undecided, or which needs no registration has none.
    """
    pairs = []
    for channel, channel_config in registration_config['channels'].items():
        if not is_selected_for_registration(channel_config) or not needs_registration(channel_config):
            continue
        fixed_channel, moving_channel = fixed_moving_channels(registration_config, channel)
        pairs.append((moving_channel, fixed_channel))
    return pairs


def registration_sequence(registration_config: Mapping, first_channel: str, stop_channel: str = ATLAS) -> list[str]:
    """
    The channels from first_channel along the chain of align_with, until (excluded) stop_channel or a channel
    aligned with nothing.
    """
    channels_config = registration_config['channels']
    sequence = [first_channel]
    while True:
        next_channel = channels_config[sequence[-1]]['align_with']
        if next_channel in (None, stop_channel):
            break
        sequence.append(next_channel)
    return sequence
