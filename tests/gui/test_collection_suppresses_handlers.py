"""
Setting a channel page up suppresses the handlers of the params of the channel (see GenericTab._set_up_channel_page).
The params of a channel are a single ``UiParameter`` or a ``UiParameterCollection`` (the stitching ones): both must
answer ``_suppress_handlers``, otherwise opening a sample with a stitchable channel fails.
"""
import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.gui.params import ChannelStitchingParams
from ClearMap.gui.params_interfaces import UiParameter, UiParameterCollection


class Child:
    """Stands for a sub-params: it suppresses its handlers as a real ``UiParameter`` does"""
    _suppress_handlers = UiParameter._suppress_handlers

    def __init__(self, painting=False):
        self._painting = painting


class Collection(UiParameterCollection):
    def __init__(self, children):
        super().__init__(tab=None, pipeline_name='Stitching', event_bus=EventBus())
        self._children = children

    @property
    def params(self):
        return self._children


def test_the_stitching_params_of_a_channel_suppress_their_handlers():
    assert hasattr(ChannelStitchingParams, '_suppress_handlers')


def test_a_collection_suppresses_the_handlers_of_every_sub_params_and_restores_them():
    children = [Child(), Child(painting=True)]
    collection = Collection(children)

    with collection._suppress_handlers():
        assert [child._painting for child in children] == [True, True]

    assert [child._painting for child in children] == [False, True]


def test_a_collection_restores_the_handlers_when_the_block_fails():
    children = [Child(), Child()]
    collection = Collection(children)

    with pytest.raises(RuntimeError):
        with collection._suppress_handlers():
            raise RuntimeError('boom')

    assert [child._painting for child in children] == [False, False]


def test_a_collection_skips_the_sub_params_not_made_yet():
    collection = Collection([Child(), None])

    with collection._suppress_handlers():
        pass
