"""
The root widget of a designer file is the class of the widget it is loaded into (gui_utils_base.create_clearmap_widget
builds the root class of the file). Only the main window is a QMainWindow: the other files are loaded into a QWidget or
a QDialog, which have no central widget, menu bar or status bar to receive those of a QMainWindow form.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import ClearMap.gui

CREATOR_FOLDER = Path(ClearMap.gui.__file__).parent / 'creator'
MAIN_WINDOW_UI_FILE_NAME = 'mainwindow.ui'
KNOWN_ROOT_CLASSES = ('QWidget', 'QDialog', 'QMainWindow')  # Exist in QtWidgets of PyQt5 and PyQt6


def _root_class(ui_file):
    return ET.parse(ui_file).getroot().find('widget').get('class')


def test_only_the_main_window_is_a_qmainwindow():
    ui_files = sorted(CREATOR_FOLDER.glob('*.ui'))
    main_windows = [ui_file.name for ui_file in ui_files if _root_class(ui_file) == 'QMainWindow']

    assert main_windows == [MAIN_WINDOW_UI_FILE_NAME]


def test_no_designer_file_sets_a_style_sheet_on_its_root_widget():
    """The loader sets every property of the root: a root style sheet made for the preview would apply in the GUI"""
    ui_files = sorted(CREATOR_FOLDER.glob('*.ui'))
    with_style_sheet = [ui_file.name for ui_file in ui_files if any(
        root_property.get('name') == 'styleSheet' and (root_property.findtext('string') or '').strip()
        for root_property in ET.parse(ui_file).getroot().find('widget').findall('property'))]

    assert not with_style_sheet, f'{with_style_sheet}: see gui/creator/README.rst for a dark preview in Designer'


def test_the_root_of_every_designer_file_is_a_class_of_qt_widgets():
    """The widget built from a file is of its root class (gui_utils_base.designer_root_class)"""
    unknown = {ui_file.name: _root_class(ui_file) for ui_file in CREATOR_FOLDER.glob('*.ui')
               if _root_class(ui_file) not in KNOWN_ROOT_CLASSES}

    assert not unknown


def test_the_assets_manager_is_a_plain_widget_as_it_is_shown_in_a_dock():
    """A QDialog in the dock of the main window hides on Escape (reject) and presses its default button on Enter"""
    assert _root_class(CREATOR_FOLDER / 'assets_manipulation.ui') == 'QWidget'
