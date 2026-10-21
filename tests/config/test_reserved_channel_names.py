"""A channel cannot be named like a value of the config which is not a channel (moving_channel: other, ...)"""
import pytest

from ClearMap.Utils.exceptions import InvalidChannelNameError
from ClearMap.config.channel import ChannelRegistry

FIRST_NAME = 'fos'


@pytest.fixture()
def registry() -> ChannelRegistry:
    return ChannelRegistry()


@pytest.fixture()
def first(registry):
    return registry.add(FIRST_NAME)


@pytest.mark.parametrize('name', ['self', 'other', 'intrinsically_aligned', 'undefined', 'none', 'None', 'Other',
                                  '__UNDECIDED__', '__undecided__'])
def test_a_name_reserved_for_a_value_of_the_config_is_refused_whatever_the_case(registry, first, name):
    with pytest.raises(InvalidChannelNameError, match='reserved'):
        registry.add(name)
    with pytest.raises(InvalidChannelNameError, match='reserved'):
        registry.rename_many({first: name})

    assert registry.names == [FIRST_NAME]


def test_a_name_which_only_contains_a_reserved_word_is_valid(registry):
    registry.add('self_2')
    registry.add('other_fos')

    assert registry.names == ['self_2', 'other_fos']
