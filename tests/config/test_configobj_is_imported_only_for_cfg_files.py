"""
configobj only reads and writes the .cfg/.ini files of ClearMap 3.0 and earlier: the configuration modules and Settings
import, and read the yml configuration, without it.
"""
import subprocess
import sys

import pytest

CONFIGURATION_MODULE_NAMES = ('ClearMap.Settings', 'ClearMap.config.config_handler',
                              'ClearMap.config.config_adjusters.dict_ops', 'ClearMap.config.early_boot')

REFUSE_CONFIGOBJ = ('import importlib.abc, sys\n'
                    'class NoConfigObj(importlib.abc.MetaPathFinder):\n'
                    '    def find_spec(self, name, path, target=None):\n'
                    '        if name.split(".")[0] == "configobj":\n'
                    '            raise ModuleNotFoundError("configobj refused by the test", name=name)\n'
                    'sys.meta_path.insert(0, NoConfigObj())\n')


def run_without_configobj(code):
    return subprocess.run([sys.executable, '-c', REFUSE_CONFIGOBJ + code], capture_output=True, text=True, check=False)


@pytest.mark.parametrize('module_name', CONFIGURATION_MODULE_NAMES)
def test_the_configuration_module_imports_without_configobj(module_name):
    result = run_without_configobj(f'import {module_name}\n')

    assert result.returncode == 0, result.stderr


def test_a_yml_configuration_is_read_and_written_without_configobj(tmp_path):
    cfg_path = tmp_path / 'machine.yml'
    cfg_path.write_text('verbosity: info\n', encoding='utf-8')
    code = ('from ClearMap.config.config_handler import ConfigHandler\n'
            f'cfg = ConfigHandler.get_cfg_from_path({str(cfg_path)!r})\n'
            'cfg["verbosity"] = "debug"\n'
            'cfg.write()\n')

    result = run_without_configobj(code)

    assert result.returncode == 0, result.stderr
    assert 'verbosity: debug' in cfg_path.read_text(encoding='utf-8')
