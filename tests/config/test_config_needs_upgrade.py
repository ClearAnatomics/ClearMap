"""
Whether the configs of an experiment must be converted depends on their format (major.minor), not on the package version.

ClearMap 3.1.1 offered to upgrade every experiment, because its configs say "3.1.0" (the defaults write it), and the
upgrade to "3.1.1" raised NotImplementedError: no experiment could be opened in the GUI.
"""
import warnings

import pytest
from packaging.version import Version

from ClearMap._version import CURRENT_CONFIG_VERSION
from ClearMap.Utils.exceptions import ClearMapConfigError
from ClearMap.config import config_handler
from ClearMap.config.config_handler import ConfigHandler, config_needs_upgrade
from ClearMap.config.convert_config_versions import convert_versions

SECTIONS_WITH_A_VERSION = ('sample', 'stitching', 'registration', 'cell_map', 'vasculature', 'machine')


def _version_written_by_the_defaults(section):
    path = ConfigHandler.get_default_path(section, must_exist=True, from_package=True)
    return ConfigHandler.get_cfg_from_path(path)['clearmap_version']


@pytest.mark.parametrize('section', SECTIONS_WITH_A_VERSION)
def test_the_configs_made_from_the_defaults_need_no_upgrade(section):
    assert not config_needs_upgrade(_version_written_by_the_defaults(section))


@pytest.mark.parametrize('patch', ['', '.0', '.1', '.7', '.0.dev0'])
def test_the_configs_of_any_patch_release_of_the_current_format_need_no_upgrade(patch):
    major, minor = CURRENT_CONFIG_VERSION

    assert not config_needs_upgrade(f'{major}.{minor}{patch}')


@pytest.mark.parametrize('config_version', ['3.0.0', '3.0', '2.1', Version('3.0.0'), (3, 0), '2.10'])
def test_the_configs_of_an_older_format_need_an_upgrade(config_version):
    assert config_needs_upgrade(config_version)


@pytest.mark.parametrize('config_version', [None, ''])
def test_configs_without_a_version_need_an_upgrade(config_version):
    assert config_needs_upgrade(config_version)


def test_the_configs_of_a_newer_version_are_refused_with_the_version_they_carry():
    major, minor = CURRENT_CONFIG_VERSION
    newer = f'{major}.{minor + 1}.0'

    with pytest.raises(ClearMapConfigError, match=newer.replace('.', r'\.')):
        config_needs_upgrade(newer)


def test_upgrading_the_configs_of_the_defaults_to_the_current_format_converts_nothing(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        convert_versions(_version_written_by_the_defaults('sample'), CURRENT_CONFIG_VERSION, exp_dir=tmp_path,
                         create_app=False)

    assert any('No conversion needed' in str(warning.message) for warning in caught)


def test_a_minor_version_with_two_digits_is_newer_not_older(monkeypatch):
    """As strings, "3.10" is before "3.9": the configs of 3.10 would be offered an upgrade to 3.9."""
    monkeypatch.setattr(config_handler, 'CURRENT_CONFIG_VERSION', (3, 9))

    with pytest.raises(ClearMapConfigError):
        config_needs_upgrade('3.10')
    assert config_needs_upgrade('3.8')
