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

from typing import Iterable, Mapping, NamedTuple

from ClearMap.IO.assets_constants import COMPOUND_REPRESENTATIVE_DATA_TYPES, CONTENT_TYPE_TO_PIPELINE
from ClearMap.Utils.exceptions import ChannelsNotPixelAlignedError, ClearMapConfigError, MissingRequirementException

ATLAS = 'atlas'  # align_with / moving_channel of the atlas: a partner which is not a channel of the sample
ALIGN_WITH_NOTHING = None  # align_with of a channel which is explicitly not registered
# align_with of a channel whose partner is not chosen yet: no default could be given (e.g. no reference channel). The
# user must choose (a channel, the atlas or ALIGN_WITH_NOTHING) before it is registered. A token, not a value: it can
# be told from an empty string written by accident, and it is a reserved channel name.
ALIGN_WITH_UNDECIDED = '__UNDECIDED__'
NOT_SELECTED_FOR_REGISTRATION = (ALIGN_WITH_NOTHING, ALIGN_WITH_UNDECIDED)
INTRINSICALLY_ALIGNED = 'intrinsically_aligned'  # moving_channel of a channel aligned by construction: needs no registration
MOVING_SELF = 'self'  # moving_channel: this channel moves (onto its align_with)
MOVING_OTHER = 'other'  # moving_channel: its align_with moves (onto this channel)
REGISTRATION_DIRECTIONS = (MOVING_SELF, MOVING_OTHER, INTRINSICALLY_ALIGNED)  # moving_channel of a registered channel
REFERENCE_DATA_TYPE = 'autofluorescence'  # The data type of the reference channel by default
# registration.reference_channel: the channel aligned with the atlas (the others are aligned with it by default),
# REFERENCE_UNDECIDED (to choose) or NO_REFERENCE (no channel is aligned with the atlas)
REFERENCE_UNDECIDED = ALIGN_WITH_UNDECIDED  # The same token: the reference is not chosen yet
NO_REFERENCE = None  # Explicitly: no channel is aligned with the atlas


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
    Whether a registration is computed for the channel: its partner is chosen (:func:`is_selected_for_registration`),
    something moves, and it is not intrinsically aligned. A direction kept while there is no partner (aligned with
    nothing, or undecided) plans nothing.
    """
    return (is_selected_for_registration(channel_config) and channel_config.get('moving_channel') is not None
            and not is_intrinsically_aligned(channel_config))


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
    check_partners_chosen(registration_config, [channel])
    channel_config = registration_config['channels'][channel]
    align_with = channel_config['align_with']
    direction = registration_direction(channel, channel_config)
    if direction == MOVING_SELF:
        return align_with, channel
    if direction == MOVING_OTHER:
        return channel, align_with
    return None, None


def check_partners_chosen(registration_config: Mapping, channels: Iterable[str]) -> None:
    """
    Check that the partner of each of channels is chosen (its align_with is not undecided).

    Raises
    ------
    MissingRequirementException
        Naming the channels whose partner is not chosen, and what to choose.
    """
    undecided = [channel for channel in channels
                 if registration_config['channels'][channel].get('align_with') == ALIGN_WITH_UNDECIDED]
    if undecided:
        raise MissingRequirementException(
            f'The partner of {", ".join(repr(channel) for channel in undecided)} is not chosen (no default could be '
            f'given, e.g. no single autofluorescence channel). Choose what each is aligned with (align_with): '
            f'another channel, the atlas, or None to not register it.')


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


def default_reference_channel(data_types: Mapping[str, str | None]) -> str:
    """
    The reference channel by default: the only channel of data type autofluorescence. With none (e.g. aligning a
    dapi channel with the atlas instead) or several, it is undecided (REFERENCE_UNDECIDED): the user chooses it.
    """
    references = [channel for channel, data_type in data_types.items() if data_type == REFERENCE_DATA_TYPE]
    return references[0] if len(references) == 1 else REFERENCE_UNDECIDED


def reference_channel_setting(registration_config: Mapping, data_types: Mapping[str, str | None]) -> str | None:
    """
    The reference channel of the registration config: a channel, REFERENCE_UNDECIDED or NO_REFERENCE. A config
    written before the setting existed (no key) has the channel its registrations already align with the atlas, if
    there is exactly one (what the user set, e.g. dapi), else the default of the data types (see
    default_reference_channel).
    """
    if 'reference_channel' in registration_config:
        return registration_config['reference_channel']
    aligned_with_the_atlas = [channel for channel, channel_config in registration_config.get('channels', {}).items()
                              if channel in data_types and (channel_config or {}).get('align_with') == ATLAS]
    if len(aligned_with_the_atlas) == 1:
        return aligned_with_the_atlas[0]
    return default_reference_channel(data_types)


def reference_channel_after_change(*, reference_before: str | None, reference_now: str | None,
                                   data_types_before: Mapping[str, str | None],
                                   data_types_after: Mapping[str, str | None]) -> str | None:
    """
    The reference channel once the data types or the channels change. Unedited (untouched by the change and still
    the default of before), it follows the default; a reference channel which is gone is replaced by the default;
    a reference chosen by hand stays.
    """
    unedited = reference_now == reference_before == default_reference_channel(data_types_before)
    gone = reference_now not in (REFERENCE_UNDECIDED, NO_REFERENCE) and reference_now not in data_types_after
    if unedited or gone:
        return default_reference_channel(data_types_after)
    return reference_now


def checked_reference_channel(registration_config: Mapping, reference_channel: str | None) -> str:
    """
    The reference channel, once it can give the atlas side of the registration: chosen, and aligned with the atlas
    (its resampled resolution is the resolution of the atlas space).

    Raises
    ------
    MissingRequirementException
        If the reference channel is not chosen, if there is none, or if it is not aligned with the atlas.
    """
    if reference_channel == REFERENCE_UNDECIDED:
        raise MissingRequirementException(f'The reference channel is not chosen (reference_channel, '
                                          f'no default could be given: no single autofluorescence channel).'
                                          f' Choose the channel aligned with the atlas, in the registration settings.')
    if reference_channel is NO_REFERENCE:
        raise MissingRequirementException(f'There is no reference channel (reference_channel: None): no channel '
                                          f'is aligned with the atlas. Choose one to use the atlas.')
    align_with = registration_config['channels'].get(reference_channel, {}).get('align_with')
    if align_with != ATLAS:
        raise MissingRequirementException(f'The reference channel {reference_channel!r} must be aligned with the atlas '
                                          f'(its align_with is {align_with!r}).')
    return reference_channel


def default_registration_partners_and_directions(
        data_types: Mapping[str, str | None], reference_channel: str | None) -> dict[str, tuple[str, str]]:
    """
    What each channel of the sample is aligned with by default, and how: ``(align_with, moving_channel)``.

    - The reference channel is aligned with the atlas.
    - The parts of the compound channels of a pipeline (e.g. the TubeMap vessels and arteries) other than their
      representative part (see COMPOUND_REPRESENTATIVE_DATA_TYPES) are acquired together with it: no registration
      of their own. This holds once the sample has exactly one representative part.
    - Every other channel is aligned with the reference channel, which moves onto it. Without a reference channel
      the partner of the others is undecided (REFERENCE_UNDECIDED: the user has to choose) or nothing
      (NO_REFERENCE: no channel is aligned with the atlas). A channel is never aligned with itself.

    These are defaults: the user may set any partner and direction (e.g. self) by hand.

    Parameters
    ----------
    data_types : Mapping[str, str | None]
        The data type of each channel of the sample.
    reference_channel : str or None
        The reference channel (see reference_channel_setting).

    Returns
    -------
    dict[str, tuple[str, str]]
        The defaults, by channel.
    """
    if reference_channel not in (REFERENCE_UNDECIDED, NO_REFERENCE) and reference_channel not in data_types:
        reference_channel = REFERENCE_UNDECIDED  # A reference which is not a channel of the sample (anymore)
    defaults = {channel: (ATLAS if channel == reference_channel else reference_channel, MOVING_OTHER)
                for channel in data_types}
    for pipeline, representative_data_type in COMPOUND_REPRESENTATIVE_DATA_TYPES.items():
        parts = [channel for channel, data_type in data_types.items()
                 if CONTENT_TYPE_TO_PIPELINE.get(data_type) == pipeline]
        representatives = [channel for channel in parts if data_types[channel] == representative_data_type]
        if len(representatives) == 1:
            defaults.update({channel: (representatives[0], INTRINSICALLY_ALIGNED)
                             for channel in parts if channel != representatives[0]})
    return defaults


def new_defaults_of_unedited_registrations(
        channels_config: Mapping, *, data_types_before: Mapping[str, str | None], reference_before: str | None,
        data_types_after: Mapping[str, str | None], reference_after: str | None) -> dict[str, dict[str, str]]:
    """
    The new defaults of the channels whose registration is unedited, when the defaults change with the data types
    or the reference channel (see default_registration_partners_and_directions). Unedited: still the default of
    before (a value set by hand which happens to be the default cannot be told apart, it follows too).
    A channel whose registration was set otherwise (by hand) is left alone, and so are the new channels
    (they are created with the defaults).

    Parameters
    ----------
    channels_config : Mapping
        The ``channels`` of the registration config.
    data_types_before, data_types_after : Mapping[str, str | None]
        The data type of each channel of the sample, before and after the change (the names of the channels after
        their renames).
    reference_before, reference_after : str or None
        The reference channel before and after the change.

    Returns
    -------
    dict[str, dict[str, str]]
        The new ``align_with`` and ``moving_channel`` of each channel to change.
    """
    defaults_before = default_registration_partners_and_directions(data_types_before, reference_before)
    defaults_after = default_registration_partners_and_directions(data_types_after, reference_after)
    changes = {}
    for channel, channel_config in channels_config.items():
        if channel not in defaults_before or channel not in defaults_after:
            continue
        current = (channel_config.get('align_with'), channel_config.get('moving_channel'))
        if current == defaults_before[channel] != defaults_after[channel]:
            changes[channel] = dict(zip(('align_with', 'moving_channel'), defaults_after[channel]))
    return changes


class RegistrationState(NamedTuple):
    """The registration config of a sample and the data types of its channels, at one point in time"""
    config: Mapping  # The registration config
    data_types: Mapping[str, str | None]  # The data type of each channel of the sample


def registration_changes_to_follow_defaults(now: RegistrationState, before: RegistrationState | None) -> dict:
    """
    The changes to the registration config which keep its defaults up to date once the data types or the reference
    channel changed from before to now: the unedited reference channel and registrations take the new defaults (see
    reference_channel_after_change and new_defaults_of_unedited_registrations).
    Without before (the config is loaded, not changed), only a missing reference channel is filled.

    Parameters
    ----------
    now : RegistrationState
        The registration after the change.
    before : RegistrationState or None
        The registration before the change, under the names of the channels after it (their renames applied).

    Returns
    -------
    dict
        The ``reference_channel`` and ``channels`` to change, or nothing.
    """
    reference_now = reference_channel_setting(now.config, now.data_types)
    if before is None:
        return {} if 'reference_channel' in now.config else {'reference_channel': reference_now}
    reference_before = reference_channel_setting(before.config, before.data_types)
    reference_after = reference_channel_after_change(reference_before=reference_before, reference_now=reference_now,
                                                     data_types_before=before.data_types, data_types_after=now.data_types)
    changes = {}
    if 'reference_channel' not in now.config or reference_after != reference_now:
        changes['reference_channel'] = reference_after
    channels_changes = new_defaults_of_unedited_registrations(now.config.get('channels', {}),
                                                              data_types_before=before.data_types, reference_before=reference_before,
                                                              data_types_after=now.data_types, reference_after=reference_after)
    if channels_changes:
        changes['channels'] = channels_changes
    return changes


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
