"""
Which channels each registration aligns, read from the registration config: each channel of
``channels`` says what it is aligned with (``align_with``) and how (``moving_channel``, its direction: relative to the
channel, ``self`` or ``other`` moves, as in the dunder methods, or ``intrinsically_aligned``: nothing moves, the
channel was acquired with its partner). The other one of the two is fixed.
The results of a registration are in the folder ``<moving>_to_<fixed>``.

``moving_channel`` may also be a name, as written by older configs: the name of the channel is ``self``, the name of its
``align_with`` is ``other``. Only the names of channels (and ``atlas``) reach the paths: never ``self``, ``other`` nor
``intrinsically_aligned`` (see :func:`fixed_moving_channels`).
"""
from __future__ import annotations

from typing import Iterable, Mapping

from ClearMap.Utils.exceptions import ChannelsNotPixelAlignedError, ClearMapConfigError, MissingRequirementException

ATLAS = 'atlas'  # align_with / moving_channel of the atlas: a partner which is not a channel of the sample
ALIGN_WITH_NOTHING = None  # align_with of a channel which is explicitly not registered
ALIGN_WITH_UNDECIDED = ''  # align_with of a channel whose partner is not chosen yet (in the GUI)
NOT_SELECTED_FOR_REGISTRATION = (ALIGN_WITH_NOTHING, ALIGN_WITH_UNDECIDED)
INTRINSICALLY_ALIGNED = 'intrinsically_aligned'  # moving_channel of a channel aligned by construction: needs no registration
MOVING_SELF = 'self'  # moving_channel: this channel moves (onto its align_with)
MOVING_OTHER = 'other'  # moving_channel: its align_with moves (onto this channel)
REGISTRATION_DIRECTIONS = (MOVING_SELF, MOVING_OTHER, INTRINSICALLY_ALIGNED)  # moving_channel of a registered channel


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


def registration_direction(channel: str, channel_config: Mapping) -> str | None:
    """
    How the channel is registered with its partner: one of REGISTRATION_DIRECTIONS (the names of older configs read as
    self or other), or None if it has no partner (aligned with nothing, or partner not chosen yet) or no direction.
    A direction is kept while the channel has no partner, but it means nothing until it has one again.

    Raises
    ------
    ClearMapConfigError
        If moving_channel is a name which is neither the channel nor its align_with (e.g. left over from another
        align_with).
    """
    align_with = channel_config.get('align_with')
    moving_channel = channel_config.get('moving_channel')
    if align_with in NOT_SELECTED_FOR_REGISTRATION or moving_channel is None:
        return None
    if moving_channel in REGISTRATION_DIRECTIONS:
        return moving_channel
    if moving_channel == channel:
        return MOVING_SELF
    if moving_channel == align_with:
        return MOVING_OTHER
    raise ClearMapConfigError(f'The registration of the channel {channel!r} is aligned with {align_with!r} but its '
                              f'moving_channel is {moving_channel!r}: it must be one of {REGISTRATION_DIRECTIONS} '
                              f'(or the name of the channel, or of its align_with). Choose which of the two moves.')


def partner_change(channel: str, channel_config: Mapping, align_with: str | None) -> dict:
    """
    The changes of the registration config of channel when its partner becomes align_with (e.g. chosen in the GUI):
    the direction it had is kept, written relative to the channel (a name would not follow the partner), and a new
    partner moves onto the channel unless a direction was chosen (the default, see the templates).

    Returns
    -------
    dict
        The new ``align_with``, and the ``moving_channel`` if it changes.
    """
    moving_channel = channel_config.get('moving_channel')
    try:
        direction = registration_direction(channel, channel_config)
    except ClearMapConfigError:  # A name left over from another partner: it meant nothing, the default replaces it
        direction = None
    if direction is None and moving_channel in REGISTRATION_DIRECTIONS:  # Kept while it had no partner
        direction = moving_channel
    if direction is None and align_with not in NOT_SELECTED_FOR_REGISTRATION:
        direction = MOVING_OTHER
    changes = {'align_with': align_with}
    if direction is not None and direction != moving_channel:
        changes['moving_channel'] = direction
    return changes


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
        (fixed, moving): the names of channels or the atlas, which name the results (``<moving>_to_<fixed>``).
        (None, None) if no registration is computed for the channel: aligned with nothing, intrinsically aligned
        or without direction.

    Raises
    ------
    MissingRequirementException
        If the partner of the channel is not chosen yet (align_with undecided).
    ClearMapConfigError
        If moving_channel is inconsistent (see :func:`registration_direction`).
    """
    channel_config = registration_config['channels'][channel]
    align_with = channel_config['align_with']
    if align_with == ALIGN_WITH_UNDECIDED:
        raise MissingRequirementException(f'Channel {channel} missing align_with in registration config')
    direction = registration_direction(channel, channel_config)
    if direction == MOVING_SELF:
        return align_with, channel
    if direction == MOVING_OTHER:
        return channel, align_with
    return None, None


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


def reversed_registrations(registration_config: Mapping, channels: Iterable[str]) -> list[str]:
    """
    The channels among channels whose registration moves the channel itself (moving_channel: self). Transforming
    the points of a channel towards its partner needs the transform of its partner moving onto it (other): through
    the registration of one of these, the points need the inverse transform.
    """
    channels_config = registration_config['channels']
    return [channel for channel in channels if registration_direction(channel, channels_config[channel]) == MOVING_SELF]


def check_intrinsically_aligned(registration_config: Mapping, channels: Iterable[str]) -> None:
    """
    Check that the channels are aligned by construction (acquired together): exactly one of them is the root (it is
    registered on its own), every other one is intrinsically aligned with one of them, and following their align_with
    leads to the root.
    Registering them each on its own is not enough: their results would end at the same place, but not pixel for
    pixel (e.g. with a non rigid registration).

    Raises
    ------
    ChannelsNotPixelAlignedError
        Naming each requirement which is not met.
    """
    channels = tuple(channels)
    channels_config = registration_config.get('channels', {})
    root_channels = [channel for channel in channels
                     if not is_intrinsically_aligned(channels_config.get(channel, {}))]
    unmet_requirements = []
    if len(root_channels) != 1:
        unmet_requirements.append(f'exactly one channel must be the root (registered on its own), the others being '
                                  f'acquired together (moving_channel: {INTRINSICALLY_ALIGNED}), got {root_channels=}')
    for channel in channels:
        if channel in root_channels:
            continue
        align_with = channels_config[channel].get('align_with')
        if align_with not in channels:
            unmet_requirements.append(f'{channel} is acquired together with {align_with!r}, which is not one of them')
    if not unmet_requirements:  # Each one is aligned with one of them: they must all lead to the root
        for channel in channels:
            followed = [channel]
            for _ in channels:  # A root is reached within as many steps as there are channels, or it is a loop
                if followed[-1] in root_channels:
                    break
                followed.append(channels_config[followed[-1]]['align_with'])
            else:
                unmet_requirements.append(f'the channels they are acquired together with make a loop: {" -> ".join(followed)}')
                break
    if unmet_requirements:
        raise ChannelsNotPixelAlignedError(channels, unmet_requirements)


def registration_sequence(registration_config: Mapping, first_channel: str, stop_channel: str = ATLAS) -> list[str]:
    """
    The channels from first_channel along the chain of align_with, until (excluded) stop_channel or a channel
    aligned with nothing (or whose partner is not chosen yet).

    Raises
    ------
    ClearMapConfigError
        If the chain loops (e.g. a channel aligned with itself).
    """
    channels_config = registration_config['channels']
    sequence = [first_channel]
    while True:
        next_channel = channels_config[sequence[-1]]['align_with']
        if next_channel in NOT_SELECTED_FOR_REGISTRATION or next_channel == stop_channel:
            break
        if next_channel in sequence:
            raise ClearMapConfigError(f'The channels are aligned in a loop: {" -> ".join(sequence + [next_channel])}. '
                                      f'Align one of them with another channel or the atlas.')
        sequence.append(next_channel)
    return sequence
