"""
A subscriber that fails on a commit fact neither undoes the commit nor keeps the next facts from the other
subscribers (a GUI tab failing to show a rename used to stop the announcement of the rename, and the requester
then reverted a rename that was written).
"""
import pytest

from ClearMap.Utils.exceptions import SubscriberFailedWarning
from ClearMap.Utils.events import CfgChanged, ChannelRenamed

CHANNEL, NEW_NAME = 'nuclei_a', 'nuclei_renamed'


class FailingSubscriber:
    def on_cfg_changed(self, event):
        raise KeyError(CHANNEL)  # e.g. a tab still showing the old name


class RenameRecorder:
    def __init__(self):
        self.renames = []

    def on_renamed(self, event: ChannelRenamed):
        self.renames.append((event.old, event.new))


def test_a_subscriber_failing_on_the_commit_keeps_neither_the_rename_nor_its_announcement(experiment,
                                                                                          channel_editor):
    channel_editor.add(CHANNEL, 'nuclei', with_path=True)
    failing, recorder = FailingSubscriber(), RenameRecorder()
    experiment._bus.subscribe(CfgChanged, failing.on_cfg_changed)
    experiment._bus.subscribe(ChannelRenamed, recorder.on_renamed)

    with pytest.warns(SubscriberFailedWarning, match='FailingSubscriber.on_cfg_changed failed on CfgChanged'):
        experiment.apply_ui_patch({'sample': {'$rename': {'channels': {CHANNEL: NEW_NAME}}}})  # Does not raise

    assert list(experiment.cfg_coordinator.get_config_view('sample')['channels']) == [NEW_NAME]
    assert recorder.renames == [(CHANNEL, NEW_NAME)]
