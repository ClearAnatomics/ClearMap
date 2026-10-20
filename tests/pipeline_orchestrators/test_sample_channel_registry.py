"""
The sample manager owns the registry of the identities of the channels, and changes it in the commits: a rename
keeps the identity of the channel, a removed channel is gone, a refused patch changes nothing, and the identities do
not outlive the experiment (they are minted again when it is opened).
"""
import pytest

from ClearMap.Utils.exceptions import ChannelNameTakenError, UnknownChannelError
from ClearMap.config.config_adjusters.patch_ops import DELETE
from ClearMap.config.channel import ChannelRegistry

CHANNEL, OTHER_CHANNEL, NEW_NAME = 'nuclei_a', 'nuclei_b', 'nuclei_renamed'
CHANNEL_IN_CAPITALS = CHANNEL.upper()


def _rename(experiment, old: str, new: str) -> None:
    experiment.apply_ui_patch({'sample': {'$rename': {'channels': {old: new}}}})


@pytest.fixture()
def registry(experiment, channel_editor) -> ChannelRegistry:
    channel_editor.add(CHANNEL, 'nuclei', with_path=True)
    channel_editor.add(OTHER_CHANNEL, 'nuclei', with_path=True)
    return experiment.sample_manager.channel_registry


def test_the_registry_has_the_channels_of_the_config_in_its_order(experiment, registry):
    assert registry.names == list(experiment.cfg_coordinator.get_config_view('sample')['channels'])
    assert registry.names == [CHANNEL, OTHER_CHANNEL]


def test_a_renamed_channel_keeps_its_identity(experiment, registry):
    channel = registry.resolve(CHANNEL)

    _rename(experiment, CHANNEL, NEW_NAME)

    assert channel.name == NEW_NAME
    assert registry.resolve(NEW_NAME) == channel
    assert registry.names == [NEW_NAME, OTHER_CHANNEL]
    with pytest.raises(UnknownChannelError):
        registry.resolve(CHANNEL)


def test_a_channel_which_changes_the_case_of_its_name_keeps_its_identity(experiment, registry):
    channel = registry.resolve(CHANNEL)

    _rename(experiment, CHANNEL, CHANNEL_IN_CAPITALS)

    assert channel.name == CHANNEL_IN_CAPITALS


def test_a_removed_channel_is_gone_and_the_others_are_untouched(experiment, registry):
    removed, kept = registry.resolve(CHANNEL), registry.resolve(OTHER_CHANNEL)

    experiment.apply_ui_patch({'sample': {'channels': {CHANNEL: DELETE}}})

    assert registry.names == [OTHER_CHANNEL]
    assert kept in registry and removed not in registry
    with pytest.raises(UnknownChannelError):
        removed.name


def test_an_added_channel_gets_an_identity_and_the_others_keep_theirs(experiment, registry, channel_editor):
    existing = registry.channels

    channel_editor.add('nuclei_c')

    assert registry.channels[:2] == existing
    assert registry.names == [CHANNEL, OTHER_CHANNEL, 'nuclei_c']


def test_a_refused_patch_does_not_change_the_registry(experiment, registry, channel_editor):
    channels_before = registry.channels

    with pytest.raises(ChannelNameTakenError):
        _rename(experiment, CHANNEL, OTHER_CHANNEL.upper())

    assert registry.channels == channels_before
    assert registry.names == [CHANNEL, OTHER_CHANNEL]


def test_the_registry_is_the_one_of_the_config_after_a_submit(experiment, registry):
    experiment.cfg_coordinator.submit(sample_manager=experiment.sample_manager)

    assert registry.names == [CHANNEL, OTHER_CHANNEL]


def test_reopening_the_experiment_mints_new_identities_and_the_old_handles_die(experiment, registry):
    old_channel = registry.resolve(CHANNEL)
    old_registry = experiment.sample_manager.channel_registry

    experiment.boot_open(experiment._exp_dir)

    new_registry = experiment.sample_manager.channel_registry
    assert new_registry is not old_registry
    assert new_registry.names == [CHANNEL, OTHER_CHANNEL]
    assert old_channel not in new_registry
    assert new_registry.resolve(CHANNEL).id != old_channel.id
    with pytest.raises(UnknownChannelError):
        new_registry.name_of(old_channel)
