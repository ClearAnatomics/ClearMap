"""
The theme of the GUI is a preference of the machine configuration: qdarkstyle (the look ClearMap is tested with, the
default) or system (no style sheet, the style of the desktop). The names are written once, in gui.style.THEMES; the
default of the machine configuration and its schema list the same.
"""
from pathlib import Path

import pytest
import yaml

import ClearMap.config
from ClearMap.Utils.exceptions import ClearMapConfigError
from ClearMap.config.early_boot import MachineConfig
from ClearMap.gui import style

CONFIG_FOLDER = Path(ClearMap.config.__file__).parent
MACHINE_DEFAULTS = CONFIG_FOLDER / 'defaults' / 'v3.1' / 'machine.yml'
MACHINE_SCHEMA = CONFIG_FOLDER / 'schemas' / 'v3.1' / 'machine.schema.yaml'
DARK_STYLE_SHEET = 'QWidget { background: dark }'


def test_the_qdarkstyle_theme_is_the_default_and_the_style_sheet_of_qdarkstyle(monkeypatch):
    monkeypatch.setattr(style, 'dark_style_sheet', lambda: DARK_STYLE_SHEET)

    assert style.DEFAULT_THEME == style.THEME_QDARKSTYLE
    assert style.application_style_sheet(style.THEME_QDARKSTYLE) == DARK_STYLE_SHEET


def test_the_system_theme_has_no_style_sheet():
    assert style.application_style_sheet(style.THEME_SYSTEM) == ''


def test_an_unknown_theme_is_refused_with_the_themes_and_the_key_to_set():
    with pytest.raises(ClearMapConfigError, match=r'adwaita.*qdarkstyle.*system.*"theme"'):
        style.application_style_sheet('adwaita')


def test_the_machine_defaults_and_the_schema_list_the_themes_of_the_style_module():
    defaults = yaml.safe_load(MACHINE_DEFAULTS.read_text())
    schema = yaml.safe_load(MACHINE_SCHEMA.read_text())

    assert defaults['theme'] == style.DEFAULT_THEME
    assert tuple(schema['properties']['theme']['enum']) == style.THEMES


@pytest.fixture()
def machine_cfg_path(tmp_path, monkeypatch):
    path = tmp_path / 'machine.yml'
    monkeypatch.setenv('CLEARMAP_MACHINE_CFG', str(path))
    return path


def test_the_theme_of_a_machine_config_without_one_is_qdarkstyle(machine_cfg_path):
    machine_cfg_path.write_text('verbosity: "info"\n')  # A config of before the theme existed

    assert MachineConfig().theme == style.THEME_QDARKSTYLE


@pytest.mark.parametrize('written, expected', [('system', 'system'), ('"System"', 'system'),
                                               ('"qdarkstyle"  # a comment', 'qdarkstyle')])
def test_the_theme_is_read_from_the_machine_config(machine_cfg_path, written, expected):
    machine_cfg_path.write_text(f'theme: {written}\n')

    assert MachineConfig().theme == expected
