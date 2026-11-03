"""
The identity of the channels: a channel is told from the others by its id, whatever its name, so a rename moves
nothing; names are validated and unique in the registry only.
"""
import copy
import pickle

import pytest

from ClearMap.Utils.exceptions import (ChannelNameTakenError, ClearMapTypeError, InvalidChannelIdError,
                                       InvalidChannelNameError,
                                       UnknownChannelError)
from ClearMap.config import compound_keys
from ClearMap.config.channel import CHANNEL_NAME_RE, Channel, ChannelId, ChannelMap, ChannelRegistry

FIRST_NAME, SECOND_NAME, NEW_NAME = 'fos', 'auto', 'cfos'


@pytest.fixture()
def registry() -> ChannelRegistry:
    return ChannelRegistry()


@pytest.fixture()
def first(registry) -> Channel:
    return registry.add(FIRST_NAME)


@pytest.fixture()
def second(registry, first) -> Channel:
    return registry.add(SECOND_NAME)


# ----------------------------------------------------------------------------------- identity

def test_a_channel_is_the_same_channel_after_a_rename(registry, first):
    registry.rename_many({first: NEW_NAME})

    assert first.name == NEW_NAME
    assert registry.resolve(NEW_NAME) == first
    assert registry.channels == [first]


def test_the_old_name_is_not_a_channel_anymore_after_a_rename(registry, first):
    registry.rename_many({first: NEW_NAME})

    with pytest.raises(UnknownChannelError):
        registry.resolve(FIRST_NAME)


def test_channels_of_two_samples_with_the_same_name_are_different_channels():
    one, other = ChannelRegistry().add(FIRST_NAME), ChannelRegistry().add(FIRST_NAME)

    assert one != other
    assert one.name == other.name


def test_a_channel_is_not_comparable_with_its_name(first):
    with pytest.raises(ClearMapTypeError):
        first == FIRST_NAME


def test_a_channel_id_is_not_a_name_nor_comparable_with_one(first):
    assert not isinstance(first.id, str)
    with pytest.raises(ClearMapTypeError):
        first.id == first.id.value


def test_a_channel_id_wraps_the_string_of_a_uuid():
    channel_id = ChannelId.new()

    assert ChannelId(channel_id.value) == channel_id
    assert hash(ChannelId(channel_id.value)) == hash(channel_id)
    with pytest.raises(ClearMapTypeError):
        ChannelId(12)


@pytest.mark.parametrize('text', [
    'auto',                                        # a channel name: short, and not an id
    '',                                            # empty
    '0f8fad5b-d9cb-469f-a165-70867728950',         # one character short
    '0f8fad5b-d9cb-169f-a165-708677289506',        # not version 4
    '0f8fad5b-d9cb-469f-c165-708677289506',        # not the RFC 4122 variant
    '0f8fad5bd9cb469fa165708677289506',            # no hyphens
    '0f8fad5b-d9cb-469f-a165-708677289506 ',       # trailing space
])
def test_a_channel_id_refuses_text_which_is_not_a_uuid4(text):
    with pytest.raises(InvalidChannelIdError):
        ChannelId(text)


def test_a_channel_id_of_uppercase_text_is_the_same_id_as_lowercase():
    lower = ChannelId('0f8fad5b-d9cb-469f-a165-708677289506')
    upper = ChannelId('0F8FAD5B-D9CB-469F-A165-708677289506')

    assert upper == lower
    assert upper.value == lower.value


def test_a_channel_id_is_refused_as_a_type_error_when_not_text():
    with pytest.raises(ClearMapTypeError):
        ChannelId(None)


def test_a_registry_knows_the_ids_of_its_channels_only(registry, first):
    other = ChannelRegistry()
    stranger = other.add(first.name)

    assert registry.has_id(first.id)
    assert not registry.has_id(stranger.id)  # Same name, another registry (e.g. before a directory load)


def test_a_removed_channel_id_is_no_longer_known(registry, first):
    registry.remove(first)

    assert not registry.has_id(first.id)


def test_has_id_refuses_what_is_not_an_id(registry, first):
    with pytest.raises(ClearMapTypeError):
        registry.has_id(first.id.value)


def test_str_of_a_channel_is_its_name_for_display(registry, first):
    registry.rename_many({first: NEW_NAME})

    assert str(first) == NEW_NAME


# ----------------------------------------------------------------------------------- names

@pytest.mark.parametrize('name', ['', 'fos-dapi', 'my channel', 'é', 'a.b', None, 3])
def test_a_name_with_other_than_letters_digits_and_underscores_is_refused(registry, name):
    with pytest.raises(InvalidChannelNameError):
        registry.add(name)

    assert len(registry) == 0


def test_names_are_unique_case_insensitive_when_adding(registry, first):
    with pytest.raises(ChannelNameTakenError):
        registry.add(FIRST_NAME.upper())

    assert registry.names == [FIRST_NAME]


def test_names_are_unique_case_insensitive_when_renaming(registry, first, second):
    with pytest.raises(ChannelNameTakenError):
        registry.rename_many({second: FIRST_NAME.upper()})

    assert registry.names == [FIRST_NAME, SECOND_NAME]


def test_renaming_onto_another_channel_is_refused_and_changes_nothing(registry, first, second):
    with pytest.raises(ChannelNameTakenError):
        registry.rename_many({first: SECOND_NAME})

    assert registry.names == [FIRST_NAME, SECOND_NAME]
    assert registry.resolve(FIRST_NAME) == first


def test_a_refused_rename_of_several_channels_renames_none(registry, first, second):
    with pytest.raises(InvalidChannelNameError):
        registry.rename_many({first: NEW_NAME, second: 'not valid'})

    assert registry.names == [FIRST_NAME, SECOND_NAME]


def test_two_channels_can_swap_their_names(registry, first, second):
    registry.rename_many({first: SECOND_NAME, second: FIRST_NAME})

    assert (first.name, second.name) == (SECOND_NAME, FIRST_NAME)


def test_a_channel_can_change_the_case_of_its_own_name(registry, first):
    registry.rename_many({first: FIRST_NAME.upper()})

    assert first.name == FIRST_NAME.upper()


def test_a_name_is_resolved_exactly(registry, first):
    with pytest.raises(UnknownChannelError):
        registry.resolve(FIRST_NAME.upper())


def test_the_name_rule_is_the_one_of_the_compound_keys():
    assert compound_keys.CHANNEL_NAME_RE is CHANNEL_NAME_RE


# ----------------------------------------------------------------------------------- removal

def test_a_removed_channel_cannot_be_used_anymore(registry, first, second):
    registry.remove(first)

    assert registry.names == [SECOND_NAME]
    assert first not in registry
    with pytest.raises(UnknownChannelError):
        first.name
    with pytest.raises(UnknownChannelError):
        registry.rename_many({first: NEW_NAME})


def test_the_name_of_a_removed_channel_is_free_again(registry, first):
    registry.remove(first)

    assert registry.add(FIRST_NAME) != first


def test_a_channel_of_another_sample_is_refused(registry, first):
    foreign = ChannelRegistry().add(FIRST_NAME)

    assert foreign not in registry
    with pytest.raises(UnknownChannelError):
        registry.resolve(foreign)
    with pytest.raises(UnknownChannelError):
        registry.rename_many({foreign: NEW_NAME})


# ----------------------------------------------------------------------------------- ChannelMap

def test_what_is_kept_for_a_channel_follows_its_rename(registry, first):
    pages = ChannelMap()
    page = object()
    pages[first] = page  # P0005: it was kept under the name, which the rename left behind

    registry.rename_many({first: NEW_NAME})

    assert pages[first] is page
    assert pages.by_name() == {NEW_NAME: page}


@pytest.mark.parametrize('use', [lambda m, k: m[k], lambda m, k: m.__setitem__(k, 1),
                                 lambda m, k: m.__delitem__(k), lambda m, k: k in m, lambda m, k: m.get(k)],
                         ids=['get', 'set', 'delete', 'contains', 'get_default'])
def test_a_channel_map_refuses_a_name_as_key(first, use):
    channel_map = ChannelMap([(first, 'page')])

    with pytest.raises(ClearMapTypeError):
        use(channel_map, FIRST_NAME)


def test_a_channel_map_is_a_mapping_of_channels(registry, first, second):
    channel_map = ChannelMap([(first, 1), (second, 2)])
    del channel_map[first]
    channel_map.setdefault(first, 3)

    assert dict(channel_map.items()) == {second: 2, first: 3}
    assert len(channel_map) == 2
    assert first in channel_map


# ----------------------------------------------------------------------------------- sessions and processes

def test_nothing_session_local_can_be_pickled(first):
    for session_local in (first, first.id, ChannelMap([(first, 1)])):
        with pytest.raises(ClearMapTypeError, match='session-local'):
            pickle.dumps(session_local)


def test_values_by_name_can_be_pickled(first):
    assert pickle.loads(pickle.dumps(ChannelMap([(first, 1)]).by_name())) == {FIRST_NAME: 1}


def test_copying_a_channel_gives_the_same_handle(first):
    assert copy.copy(first) is first
    assert copy.deepcopy({'channel': first})['channel'] is first
