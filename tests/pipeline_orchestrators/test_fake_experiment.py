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


def test_fake_experiment_takes_the_data_types_the_sample_id_and_the_folder_name(tmp_path):
    folders = make_fake_experiment(tmp_path, ('cfos', 'lectin'), data_types={'lectin': 'vessels'},
                                   sample_id='sample_01', sample_folder_name='my_sample')

    sample_dir = tmp_path / 'my_sample'
    assert {folder.parent for folder in folders.values()} == {sample_dir}
    sample_cfg = ConfigHandler.get_cfg_from_path(sample_dir / 'sample.yml')
    assert sample_cfg['sample_id'] == 'sample_01'
    assert {name: cfg['data_type'] for name, cfg in sample_cfg['channels'].items()} == {'cfos': 'nuclei',
                                                                                         'lectin': 'vessels'}


def test_fake_experiment_has_no_sample_id_unless_asked(tmp_path):
    make_fake_experiment(tmp_path)

    sample_cfg = ConfigHandler.get_cfg_from_path(tmp_path / SAMPLE_FOLDER_NAME / 'sample.yml')
    assert not sample_cfg['sample_id']
