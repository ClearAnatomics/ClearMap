"""
The theme of the GUI is a preference of the machine configuration: qdarkstyle (the look ClearMap is tested with, the
default), qdarkstyle-light (its light palette) or system (no style sheet, the style of the desktop). The names are written once, in gui.style.THEMES; the
default of the machine configuration and its schema list the same.
"""
import dataclasses
import functools
from pathlib import Path

import pytest
from matplotlib.colors import CSS4_COLORS
import yaml

import ClearMap.config
from ClearMap.Utils.exceptions import ClearMapConfigError
from ClearMap.config.early_boot import MachineConfig
from ClearMap.gui import style

CONFIG_FOLDER = Path(ClearMap.config.__file__).parent
MACHINE_DEFAULTS = CONFIG_FOLDER / 'defaults' / 'v3.1' / 'machine.yml'
MACHINE_SCHEMA = CONFIG_FOLDER / 'schemas' / 'v3.1' / 'machine.schema.yaml'
CSS_COLOUR_NAMES = frozenset(CSS4_COLORS)
DARK_STYLE_SHEET = 'QWidget { background: dark }'
LIGHT_STYLE_SHEET = 'QWidget { background: light }'


def test_the_qdarkstyle_theme_is_the_default_and_the_style_sheet_of_qdarkstyle(monkeypatch):
    monkeypatch.setattr(style, 'dark_style_sheet', lambda: DARK_STYLE_SHEET)

    assert style.DEFAULT_THEME == style.THEME_QDARKSTYLE
    assert style.application_style_sheet(style.THEME_QDARKSTYLE) == DARK_STYLE_SHEET


def test_the_qdarkstyle_light_theme_has_the_style_sheet_of_the_light_palette_of_qdarkstyle(monkeypatch):
    monkeypatch.setattr(style, 'light_style_sheet', lambda: LIGHT_STYLE_SHEET)

    assert style.application_style_sheet(style.THEME_QDARKSTYLE_LIGHT) == LIGHT_STYLE_SHEET


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


@pytest.mark.parametrize('background_is_dark', [True, False])
def test_the_qdarkstyle_theme_has_the_message_colours_made_for_a_dark_background(background_is_dark):
    assert (style.message_colours(style.THEME_QDARKSTYLE, background_is_dark=background_is_dark)
            == style.MESSAGE_COLOURS_ON_DARK)


@pytest.mark.parametrize('background_is_dark, expected', [(True, style.MESSAGE_COLOURS_ON_DARK),
                                                          (False, style.MESSAGE_COLOURS_ON_LIGHT)])
def test_the_system_theme_has_the_message_colours_of_the_background_of_the_desktop(background_is_dark, expected):
    assert style.message_colours(style.THEME_SYSTEM, background_is_dark=background_is_dark) == expected


@pytest.mark.parametrize('background_is_dark', [True, False])
def test_the_qdarkstyle_light_theme_has_the_message_colours_made_for_a_light_background(background_is_dark):
    assert (style.message_colours(style.THEME_QDARKSTYLE_LIGHT, background_is_dark=background_is_dark)
            == style.MESSAGE_COLOURS_ON_LIGHT)


def test_the_qdarkstyle_light_theme_gives_its_light_palette_to_the_tooltips():
    assert style.tooltip_colours(style.THEME_QDARKSTYLE_LIGHT) == (style.LightPalette.COLOR_BACKGROUND_2,
                                                                   style.LightPalette.COLOR_TEXT_2)


def test_the_qdarkstyle_theme_gives_its_palette_to_the_tooltips_and_the_system_theme_leaves_them_alone():
    assert style.tooltip_colours(style.THEME_QDARKSTYLE) == (style.DarkPalette.COLOR_BACKGROUND_2,
                                                             style.DarkPalette.COLOR_TEXT_2)
    assert style.tooltip_colours(style.THEME_SYSTEM) is None


@pytest.mark.parametrize('theme_function', [style.application_style_sheet, style.tooltip_colours,
                                            functools.partial(style.message_colours, background_is_dark=True)])
def test_every_theme_function_refuses_an_unknown_theme(theme_function):
    with pytest.raises(ClearMapConfigError, match=r'adwaita.*"theme"'):
        theme_function('adwaita')


@pytest.mark.parametrize('colours', [style.MESSAGE_COLOURS_ON_DARK, style.MESSAGE_COLOURS_ON_LIGHT],
                         ids=['dark', 'light'])
def test_every_message_colour_is_a_css_colour_name(colours):
    # An unknown name is silently ignored by a Qt style sheet: the message would keep the colour of the window text
    assert [name for name in dataclasses.astuple(colours) if name not in CSS_COLOUR_NAMES] == []
