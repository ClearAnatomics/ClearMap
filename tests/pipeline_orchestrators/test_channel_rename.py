"""
Renaming a channel carries what the user set for it over to the new name, in every section that holds
an entry for it (headless, with the config adjusters of the application).
"""
import pytest

from ClearMap.config.compound_keys import compound_is_oriented

PARTICLE_DIAMETER = 25  # Not the default, to tell a carried over value from a refilled one
OLD_NAME, NEW_NAME, OTHER_NAME = 'nuclei_a', 'nuclei_renamed', 'nuclei_b'


def _rename(experiment, old: str, new: str) -> None:
    experiment.apply_ui_patch({'sample': {'$rename': {'channels': {old: new}}}})


def _pair_keys(experiment) -> list[str]:
    return experiment.sample_manager.colocalization_pair_keys(oriented=compound_is_oriented('Colocalization'))


def _config(experiment, section: str) -> dict:
    return experiment.cfg_coordinator.get_config_view(section)


@pytest.fixture()
def two_nuclei_channels(experiment, channel_editor):
    for name in (OLD_NAME, OTHER_NAME):
        channel_editor.add(name, 'nuclei', with_path=True)


def test_renamed_channel_keeps_its_cell_map_settings(experiment, two_nuclei_channels):
    experiment.apply_ui_patch({'cell_map': {'channels': {OLD_NAME: {'detection': {'shape_detection':
                                                                                 {'threshold': 123}}}}}})

    _rename(experiment, OLD_NAME, NEW_NAME)

    channels = _config(experiment, 'cell_map')['channels']
    assert set(channels) == {NEW_NAME, OTHER_NAME}
    assert channels[NEW_NAME]['detection']['shape_detection']['threshold'] == 123


def test_renamed_channel_keeps_its_colocalization_pair_settings(experiment, two_nuclei_channels):
    (old_pair,) = _pair_keys(experiment)
    experiment.apply_ui_patch({'colocalization': {'channels': {old_pair: {'comparison':
                                                                         {'particle_diameter': PARTICLE_DIAMETER}}}}})

    _rename(experiment, OLD_NAME, NEW_NAME)

    channels = _config(experiment, 'colocalization')['channels']
    assert len(channels) == 1
    (pair_config,) = channels.values()
    assert pair_config['comparison']['particle_diameter'] == PARTICLE_DIAMETER


@pytest.mark.xfail(strict=True, reason='Pairs are oriented, so their key follows the order of the sample channels, '
                                       'and the renamed channel moves to the end. Right after the rename the '
                                       'config still has the key of the old order (nuclei_renamed-nuclei_b) while '
                                       'the sample manager says nuclei_b-nuclei_renamed. The next edit of the '
                                       'sample fixes it. Open: is that transient visible to the colocalization tab?')
def test_colocalization_pair_key_is_the_one_of_the_sample_manager_right_after_a_rename(experiment,
                                                                                       two_nuclei_channels):
    _rename(experiment, OLD_NAME, NEW_NAME)

    assert list(_config(experiment, 'colocalization')['channels']) == _pair_keys(experiment)


def test_renamed_channel_keeps_its_vasculature_settings(experiment, channel_editor):
    channel_editor.add('vessels', 'vessels', with_path=True)
    experiment.apply_ui_patch({'vasculature': {'binarization': {'single_channels': {'vessels': {'binarize': {
        'threshold': 321}}}}}})

    _rename(experiment, 'vessels', 'vessels_renamed')

    channels = _config(experiment, 'vasculature')['binarization']['single_channels']
    assert set(channels) == {'vessels_renamed'}
    assert channels['vessels_renamed']['binarize']['threshold'] == 321
