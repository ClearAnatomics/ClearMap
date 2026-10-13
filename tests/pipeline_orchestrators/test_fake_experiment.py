"""The fake experiment of tests/fake_experiment.py is a valid new experiment with a colocalization pair"""
from ClearMap.config.config_handler import ConfigHandler
from tests.fake_experiment import CHANNEL_NAMES, SAMPLE_FOLDER_NAME, make_fake_experiment


def test_fake_experiment_has_the_configs_the_channels_and_the_pair(tmp_path):
    folders = make_fake_experiment(tmp_path)

    sample_dir = tmp_path / SAMPLE_FOLDER_NAME
    for config_name in ('sample', 'cell_map', 'registration', 'colocalization'):
        assert (sample_dir / f'{config_name}.yml').exists(), config_name
    assert list(folders) == list(CHANNEL_NAMES)
    assert all(len(list(folder.glob('*.tif'))) > 0 and folder.parent == sample_dir for folder in folders.values())

    sample_cfg = ConfigHandler.get_cfg_from_path(sample_dir / 'sample.yml')
    assert {name: cfg['data_type'] for name, cfg in sample_cfg['channels'].items()} == {n: 'nuclei' for n in CHANNEL_NAMES}
    assert all(sample_cfg['channels'][name]['path'] == str(folder) for name, folder in folders.items())

    coloc_cfg = ConfigHandler.get_cfg_from_path(sample_dir / 'colocalization.yml')
    assert list(coloc_cfg['channels']) == ['-'.join(CHANNEL_NAMES)]
    assert 'n_processes' in coloc_cfg['performance']  # shared by the section, not per pair
