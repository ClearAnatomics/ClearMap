"""
The first-run sentinels of early_boot are kept in the configuration folder (CLEARMAP_CFG_DIR, else ~/.clearmap), not
always in the home folder. The folder is read when ClearMap.config.config_handler is imported: new interpreter.
"""
import os
import subprocess
import sys

PROBE = ('from ClearMap.config import early_boot;'
         'early_boot._mark_first_run_done("v3_1");'
         'print(early_boot._is_first_run("v3_1"))')


def mark_first_run_done(home, cfg_dir_env_var=None):
    env = {**os.environ, 'HOME': str(home)}
    env.pop('CLEARMAP_CFG_DIR', None)
    if cfg_dir_env_var is not None:
        env['CLEARMAP_CFG_DIR'] = str(cfg_dir_env_var)
    result = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == 'False'  # Not the first run any more


def test_the_first_run_sentinel_is_written_in_the_configuration_folder(tmp_path):
    cfg_dir = tmp_path / 'cfg_dir'

    mark_first_run_done(home=tmp_path / 'home', cfg_dir_env_var=cfg_dir)

    assert (cfg_dir / '.first_run_sentinels').is_dir()
    assert not (tmp_path / 'home' / '.clearmap').exists()


def test_the_first_run_sentinel_is_in_dot_clearmap_of_the_home_without_the_variable(tmp_path):
    home = tmp_path / 'home'

    mark_first_run_done(home=home)

    assert (home / '.clearmap' / '.first_run_sentinels').is_dir()
