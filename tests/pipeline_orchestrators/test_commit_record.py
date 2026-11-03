"""
The post-commit hooks are given the CommitRecord of the commit: the channels before and after it and the renames it
carried out, taken once right after the write. The hooks then do not have to guess what changed from the config.
"""
import pytest

from ClearMap.config.config_coordinator import CommitRecord, committed_channel_renames
from ClearMap.Utils.events import CfgChanged, ChannelRenamed, ChannelsChanged

CHANNEL, OTHER_CHANNEL, NEW_NAME = 'nuclei_a', 'nuclei_b', 'nuclei_renamed'
UNKNOWN_CHANNEL, UNKNOWN_CHANNEL_NEW_NAME = 'not_a_channel', 'still_not_a_channel'


class HookRecorder:
    def __init__(self):
        self.commit_records = []
        self.channels_in_config_when_run = []

    def attach_to(self, experiment):
        self._experiment = experiment
        experiment.cfg_coordinator.add_post_commit_hook(self.on_commit)

    def on_commit(self, commit_record: CommitRecord):
        self.commit_records.append(commit_record)
        self.channels_in_config_when_run.append(tuple(self._experiment.cfg_coordinator.current_channels))


class FactRecorder:
    """Subscriptions are weak and to bound methods: a recorder keeps the facts it is given"""

    def __init__(self):
        self.facts = []

    def on_fact(self, fact):
        self.facts.append(fact)


class NestedEditor(FactRecorder):
    """Edits the channels once, in reaction to the first commit announced (before its channel facts)"""

    def __init__(self, channel_editor, channel_to_add: str):
        super().__init__()
        self._channel_editor = channel_editor
        self._channel_to_add = channel_to_add
        self._done = False

    def on_cfg_changed(self, fact: CfgChanged):
        if not self._done:
            self._done = True
            self._channel_editor.add(self._channel_to_add)


@pytest.fixture()
def hook(experiment, channel_editor):
    channel_editor.add(CHANNEL, 'nuclei', with_path=True)
    recorder = HookRecorder()
    recorder.attach_to(experiment)
    return recorder


def _rename(experiment, old: str, new: str) -> None:
    experiment.apply_ui_patch({'sample': {'$rename': {'channels': {old: new}}}})


def test_the_record_of_an_added_channel_has_the_channels_before_and_after(hook, channel_editor):
    channel_editor.add(OTHER_CHANNEL)

    commit_record = hook.commit_records[0]
    assert commit_record.channels_before == (CHANNEL,)
    assert commit_record.channels_after == (CHANNEL, OTHER_CHANNEL)
    assert commit_record.renames == {}
    assert commit_record.channels_changed


def test_the_record_of_a_rename_has_the_rename(hook, experiment):
    _rename(experiment, CHANNEL, NEW_NAME)

    commit_record = hook.commit_records[0]
    assert commit_record.renames == {CHANNEL: NEW_NAME}
    assert commit_record.channels_before == (CHANNEL,)
    assert commit_record.channels_after == (NEW_NAME,)


def test_the_record_has_the_renames_carried_out_not_the_ones_asked(hook, experiment):
    _rename(experiment, UNKNOWN_CHANNEL, UNKNOWN_CHANNEL_NEW_NAME)

    assert all(commit_record.renames == {} for commit_record in hook.commit_records)


def test_a_commit_which_does_not_touch_the_channels_says_so(hook, channel_editor):
    channel_editor.set_data_type(CHANNEL, 'autofluorescence')

    commit_record = hook.commit_records[0]
    assert not commit_record.channels_changed
    assert commit_record.channels_before == commit_record.channels_after == (CHANNEL,)


def test_submit_gives_a_record_too_without_renames(hook, experiment):
    experiment.cfg_coordinator.submit(sample_manager=experiment.sample_manager)

    commit_record = hook.commit_records[0]
    assert commit_record.renames == {}
    assert commit_record.channels_before == commit_record.channels_after == (CHANNEL,)


def test_the_hooks_run_once_the_config_is_written(hook, channel_editor):
    channel_editor.add(OTHER_CHANNEL)

    assert hook.channels_in_config_when_run[0] == (CHANNEL, OTHER_CHANNEL)


def test_the_facts_of_a_commit_are_the_ones_of_its_record(hook, experiment):
    recorder = FactRecorder()
    experiment._bus.subscribe(ChannelsChanged, recorder.on_fact)
    experiment._bus.subscribe(ChannelRenamed, recorder.on_fact)

    _rename(experiment, CHANNEL, NEW_NAME)

    assert recorder.facts == [ChannelsChanged(before=[CHANNEL], after=[NEW_NAME]),
                              ChannelRenamed(old=CHANNEL, new=NEW_NAME)]


def test_a_commit_nested_in_a_fact_handler_does_not_change_the_facts_of_the_first(hook, experiment, channel_editor):
    """A subscriber which edits the channels in reaction to the first commit does not rewrite its history"""
    nested_editor = NestedEditor(channel_editor, OTHER_CHANNEL)
    experiment._bus.subscribe(CfgChanged, nested_editor.on_cfg_changed)
    experiment._bus.subscribe(ChannelsChanged, nested_editor.on_fact)

    _rename(experiment, CHANNEL, NEW_NAME)

    # WARNING: not asserting the order: the nested commit announces itself before the first one finishes
    #  doing so (the bus is not run-to-completion yet), which is a different matter
    assert len(nested_editor.facts) == 2
    assert ChannelsChanged(before=[CHANNEL], after=[NEW_NAME]) in nested_editor.facts
    assert ChannelsChanged(before=[NEW_NAME], after=[NEW_NAME, OTHER_CHANNEL]) in nested_editor.facts


def test_the_renames_carried_out_are_told_by_the_channels_before_and_after():
    assert committed_channel_renames({'a': 'b'}, ['a'], ['b']) == {'a': 'b'}
    assert committed_channel_renames({'a': 'b'}, ['c'], ['c']) == {}
