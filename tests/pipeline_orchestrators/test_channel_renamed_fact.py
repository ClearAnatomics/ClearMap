"""
ChannelRenamed is a fact: it announces the renames the commit carried out, not the ones that were asked
(its subscribers re-key the workspace and the workers on it).
"""
from ClearMap.Utils.events import ChannelRenamed

CHANNEL, NEW_NAME = 'nuclei_a', 'nuclei_renamed'
UNKNOWN_CHANNEL, UNKNOWN_CHANNEL_NEW_NAME = 'not_a_channel', 'still_not_a_channel'


class RenameRecorder:
    def __init__(self):
        self.renames = []

    def on_renamed(self, event: ChannelRenamed):
        self.renames.append((event.old, event.new))


def _rename(experiment, old: str, new: str) -> RenameRecorder:
    recorder = RenameRecorder()
    experiment._bus.subscribe(ChannelRenamed, recorder.on_renamed)
    experiment.apply_ui_patch({'sample': {'$rename': {'channels': {old: new}}}})
    return recorder


def test_a_rename_is_announced(experiment, channel_editor):
    channel_editor.add(CHANNEL, 'nuclei', with_path=True)

    recorder = _rename(experiment, CHANNEL, NEW_NAME)

    assert recorder.renames == [(CHANNEL, NEW_NAME)]


def test_the_rename_of_an_unknown_channel_is_not_announced(experiment, channel_editor):
    channel_editor.add(CHANNEL, 'nuclei', with_path=True)

    recorder = _rename(experiment, UNKNOWN_CHANNEL, UNKNOWN_CHANNEL_NEW_NAME)

    assert recorder.renames == []
    assert list(experiment.cfg_coordinator.get_config_view('sample')['channels']) == [CHANNEL]
