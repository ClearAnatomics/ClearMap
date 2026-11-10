"""
The dialogs of an experiment upgrade use the running QApplication when there is one (the GUI), and leave its style
sheet alone: the upgrade replaced it with the dark one, whatever the theme of the GUI. A QApplication made for the
upgrade (a script) gets the style sheet of the theme of the machine configuration.

QApplication is a fake: the sandbox has no Qt.
"""
import pytest

from ClearMap.config import convert_config_versions
from ClearMap.gui import gui_utils_base, style

DARK_STYLE_SHEET = 'QWidget { background: dark }'
GUI_STYLE_SHEET = 'the style sheet of the theme of the GUI'


class FakeApplication:
    running = None

    def __init__(self, arguments=None):
        self.style_sheets = []
        FakeApplication.running = self

    @classmethod
    def instance(cls):
        return cls.running

    def setStyleSheet(self, style_sheet):
        self.style_sheets.append(style_sheet)


@pytest.fixture()
def fake_application(monkeypatch):
    monkeypatch.setattr(FakeApplication, 'running', None)
    monkeypatch.setattr(gui_utils_base, 'QApplication', FakeApplication)
    monkeypatch.setattr(style, 'dark_style_sheet', DARK_STYLE_SHEET.strip)
    return FakeApplication


@pytest.fixture()
def machine_theme(tmp_path, monkeypatch):
    def write_theme(theme):
        path = tmp_path / 'machine.yml'
        path.write_text(f'theme: "{theme}"\n')
        monkeypatch.setenv('CLEARMAP_MACHINE_CFG', str(path))
    return write_theme


def test_a_running_application_keeps_its_style_sheet(fake_application):
    gui = FakeApplication()
    gui.setStyleSheet(GUI_STYLE_SHEET)

    assert gui_utils_base.ensure_qapp(style_sheet=DARK_STYLE_SHEET) is gui
    assert gui.style_sheets == [GUI_STYLE_SHEET]


def test_a_new_application_gets_the_style_sheet_it_is_made_with(fake_application):
    app = gui_utils_base.ensure_qapp(style_sheet=DARK_STYLE_SHEET)

    assert app.style_sheets == [DARK_STYLE_SHEET]


def test_an_upgrade_started_from_the_gui_leaves_its_style_alone(fake_application, machine_theme):
    machine_theme(style.THEME_QDARKSTYLE)
    gui = FakeApplication()
    gui.setStyleSheet(GUI_STYLE_SHEET)

    convert_config_versions._prepare_the_dialogs()

    assert gui.style_sheets == [GUI_STYLE_SHEET]


@pytest.mark.parametrize('theme, style_sheets', [(style.THEME_QDARKSTYLE, [DARK_STYLE_SHEET]),
                                                 (style.THEME_SYSTEM, [])])
def test_an_upgrade_run_alone_has_the_theme_of_the_machine_configuration(fake_application, machine_theme,
                                                                           theme, style_sheets):
    machine_theme(theme)

    convert_config_versions._prepare_the_dialogs()

    assert FakeApplication.running.style_sheets == style_sheets
