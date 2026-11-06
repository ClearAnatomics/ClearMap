import pytest

from ClearMap._version import CURRENT_CONFIG_VERSION
from ClearMap.config.config_handler import VERSION_SUFFIX, config_version_label
from ClearMap.config.defaults_provider import VERSION as DEFAULTS_FOLDER_NAME


@pytest.mark.parametrize('version', [(3, 1), (3, 1, 1), '3.1', '3.1.1', '3.1.0.dev0'])
def test_the_label_of_a_config_version_ignores_its_patch_number_and_dev_tag(version):
    assert config_version_label(version, separator='_') == 'v3_1'


def test_the_label_of_a_two_digit_minor_is_not_cut():
    assert config_version_label((3, 10, 2), separator='.') == 'v3.10'


def test_the_folder_and_file_names_of_the_current_config_come_from_the_current_config_version():
    major, minor = CURRENT_CONFIG_VERSION[:2]
    assert VERSION_SUFFIX == f'v{major}_{minor}'
    assert DEFAULTS_FOLDER_NAME == f'v{major}.{minor}'
