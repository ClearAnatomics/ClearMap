"""
install_gui.sh records the elastix path it installed (macOS, Windows) in the machine configuration of the user
configuration folder (~/.clearmap unless CLEARMAP_CFG_DIR says otherwise), where Settings.elastix_path reads it.
The folder is read when ClearMap is imported, so each step runs in a new interpreter, as in the installation script.
"""
import os
import subprocess
import sys

import pytest

from ClearMap.config.config_handler import ConfigHandler

NEW_ELASTIX_PATH = '/new/elastix'


@pytest.fixture
def cfg_dir(tmp_path):
    folder = tmp_path / 'cfg_dir'
    folder.mkdir()
    return folder


def run_in_cfg_dir(cfg_dir, code):
    env = {**os.environ, 'CLEARMAP_CFG_DIR': str(cfg_dir)}
    env.pop('CLEARMAP_MACHINE_CFG', None)
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()[-1] if result.stdout else ''


def set_elastix_path(cfg_dir, elastix_path):
    run_in_cfg_dir(cfg_dir, f'from ClearMap.Utils.install_utils import set_elastix_path; '
                            f'set_elastix_path({elastix_path!r})')


def settings_elastix_path(cfg_dir):
    return run_in_cfg_dir(cfg_dir, 'from ClearMap import Settings; print(Settings.elastix_path)')


def test_the_elastix_path_set_by_the_installer_in_a_new_configuration_folder_is_the_one_of_settings(cfg_dir):
    set_elastix_path(cfg_dir, NEW_ELASTIX_PATH)

    assert settings_elastix_path(cfg_dir) == NEW_ELASTIX_PATH


def test_setting_the_elastix_path_changes_only_its_line_of_the_machine_configuration(cfg_dir):
    machine_yml = ConfigHandler.get_global_canonical_path('machine', root=cfg_dir / 'config')
    machine_yml.parent.mkdir(parents=True)
    machine_yml.write_text('# my machine\nverbosity: debug  # chatty\nelastix_path: null  # bundled\n', encoding='utf-8')

    set_elastix_path(cfg_dir, NEW_ELASTIX_PATH)

    assert machine_yml.read_text(encoding='utf-8') == (f"# my machine\nverbosity: debug  # chatty\n"
                                                       f"elastix_path: '{NEW_ELASTIX_PATH}'  # bundled\n")
    assert settings_elastix_path(cfg_dir) == NEW_ELASTIX_PATH
