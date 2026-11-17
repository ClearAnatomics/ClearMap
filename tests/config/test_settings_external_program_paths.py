"""
The paths of ClearMap.Settings to external programs (elastix_path, tera_stitcher_path, imagej_path, ilastik_path)
come from the machine configuration of this version (the yml of the configuration folder), from the 3.0 machine
parameters only when that file does not exist. Without a value the elastix is the one bundled with ClearMap and the
others are None. They are read at their first access, so each check runs in a new interpreter.
"""
import json
import os
import subprocess
import sys

import pytest

from ClearMap import Settings
from ClearMap.config.config_handler import ConfigHandler

PROBE = ('import json, sys;'
         'from ClearMap import Settings;'
         'imported_at_import = sorted(m for m in ("configobj", "ClearMap.config.config_handler") if m in sys.modules);'
         'print(json.dumps({"elastix": Settings.elastix_path, "tera_stitcher": Settings.tera_stitcher_path,'
         '                  "imagej": Settings.imagej_path, "ilastik": Settings.ilastik_path,'
         '                  "imported_at_import": imported_at_import}))')

ELASTIX_OF_THE_YML = '/elastix/of/the/machine/yml'
ELASTIX_OF_THE_3_0_CFG = '/elastix/of/the/3.0/cfg'


@pytest.fixture
def cfg_dir(tmp_path):
    folder = tmp_path / 'cfg_dir'
    folder.mkdir()
    return folder


def write_machine_yml(cfg_dir, elastix_line):
    machine_yml = ConfigHandler.get_global_canonical_path('machine', root=cfg_dir / 'config')
    machine_yml.parent.mkdir(parents=True)
    machine_yml.write_text(f'verbosity: info\n{elastix_line}\n', encoding='utf-8')
    return machine_yml


def write_3_0_machine_parameters(cfg_dir):
    (cfg_dir / 'machine_params_v3_0.cfg').write_text(f"elastix_path = '{ELASTIX_OF_THE_3_0_CFG}'\n", encoding='utf-8')


def read_settings(cfg_dir):
    env = {**os.environ, 'CLEARMAP_CFG_DIR': str(cfg_dir)}
    env.pop('CLEARMAP_MACHINE_CFG', None)
    result = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_the_elastix_path_of_the_machine_configuration_of_this_version_is_used(cfg_dir):
    write_machine_yml(cfg_dir, f'elastix_path: "{ELASTIX_OF_THE_YML}"  # set by the user')

    assert read_settings(cfg_dir)['elastix'] == ELASTIX_OF_THE_YML


def test_the_machine_configuration_of_this_version_wins_over_the_3_0_machine_parameters(cfg_dir):
    write_machine_yml(cfg_dir, f'elastix_path: "{ELASTIX_OF_THE_YML}"')
    write_3_0_machine_parameters(cfg_dir)

    assert read_settings(cfg_dir)['elastix'] == ELASTIX_OF_THE_YML


def test_a_null_elastix_path_is_the_bundled_elastix_even_with_3_0_machine_parameters(cfg_dir):
    write_machine_yml(cfg_dir, 'elastix_path: null')
    write_3_0_machine_parameters(cfg_dir)

    assert read_settings(cfg_dir)['elastix'] == Settings.default_elastix_path


def test_the_3_0_machine_parameters_are_read_when_there_is_no_machine_configuration_of_this_version(cfg_dir):
    write_3_0_machine_parameters(cfg_dir)

    assert read_settings(cfg_dir)['elastix'] == ELASTIX_OF_THE_3_0_CFG


def test_without_any_machine_configuration_the_elastix_is_the_bundled_one_and_no_file_is_written(cfg_dir):
    assert read_settings(cfg_dir)['elastix'] == Settings.default_elastix_path
    assert not any(cfg_dir.iterdir())


def test_importing_settings_reads_no_configuration(cfg_dir):
    assert read_settings(cfg_dir)['imported_at_import'] == []


def test_the_other_external_program_paths_are_read_from_the_machine_configuration(cfg_dir):
    write_machine_yml(cfg_dir, "tera_stitcher_path: '/tera'\nimagej_path: '/fiji'\nilastik_path: '/ilastik'")

    settings = read_settings(cfg_dir)

    assert (settings['tera_stitcher'], settings['imagej'], settings['ilastik']) == ('/tera', '/fiji', '/ilastik')


def test_the_other_external_program_paths_are_read_from_the_3_0_machine_parameters_without_a_newer_file(cfg_dir):
    (cfg_dir / 'machine_params_v3_0.cfg').write_text("imagej_path = '/fiji/of/3.0'\n", encoding='utf-8')

    assert read_settings(cfg_dir)['imagej'] == '/fiji/of/3.0'


def test_the_other_external_program_paths_are_none_when_the_machine_configuration_has_none(cfg_dir):
    write_machine_yml(cfg_dir, 'elastix_path: null')

    settings = read_settings(cfg_dir)

    assert (settings['tera_stitcher'], settings['imagej'], settings['ilastik']) == (None, None, None)
