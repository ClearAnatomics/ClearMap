"""
CLEARMAP_CFG_DIR sets the folder of the user configuration, instead of ~/.clearmap. Settings reads the machine
parameters when it is imported, so each check runs in a new interpreter.
"""
import json
import os
import subprocess
import sys

import pytest

PROBE = ('import json;'
         'from ClearMap import Settings;'
         'from ClearMap.config.config_handler import ConfigHandler, CLEARMAP_CFG_DIR;'
         'print(json.dumps({"settings": str(Settings.clearmap_cfg_dir), "elastix": Settings.elastix_path,'
         '                  "handler": str(CLEARMAP_CFG_DIR), "global_root": str(ConfigHandler.user_defaults_dir),'
         '                  "cache": str(Settings.clearmap_cache_dir)}))')


@pytest.fixture
def machine_cfg_dir(tmp_path):
    cfg_dir = tmp_path / 'cfg_dir'
    cfg_dir.mkdir()
    (cfg_dir / 'machine_params_v3_0.cfg').write_text("elastix_path = '/elastix/of/the/cfg/dir'\n")
    return cfg_dir


def read_configuration_folder(home, cfg_dir_env_var=None):
    env = {**os.environ, 'HOME': str(home)}
    env.pop('CLEARMAP_CFG_DIR', None)
    if cfg_dir_env_var is not None:
        env['CLEARMAP_CFG_DIR'] = cfg_dir_env_var
    result = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_the_configuration_folder_is_set_by_the_environment_variable(tmp_path, machine_cfg_dir):
    folders = read_configuration_folder(home=tmp_path / 'home', cfg_dir_env_var=str(machine_cfg_dir))

    assert folders['settings'] == folders['handler'] == folders['global_root'] == str(machine_cfg_dir)
    assert folders['elastix'] == '/elastix/of/the/cfg/dir'
    assert folders['cache'] == str(machine_cfg_dir / 'cache')


def test_the_configuration_folder_is_dot_clearmap_of_the_home_without_the_variable(tmp_path):
    home = tmp_path / 'home'

    folders = read_configuration_folder(home=home)

    assert folders['settings'] == folders['handler'] == folders['global_root'] == str(home / '.clearmap')
    assert folders['cache'] == str(home / '.clearmap' / 'cache')


def test_an_empty_variable_is_the_same_as_none(tmp_path):
    home = tmp_path / 'home'

    folders = read_configuration_folder(home=home, cfg_dir_env_var='')

    assert folders['settings'] == str(home / '.clearmap')
