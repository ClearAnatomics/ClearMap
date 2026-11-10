"""
The root widget of a designer file is the class of the widget it is loaded into. Only the main window is a
QMainWindow: the other files are loaded into a QWidget or a QDialog (gui_utils_base.create_clearmap_widget), which
have no central widget, menu bar or status bar to receive those of a QMainWindow form.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import ClearMap.gui

CREATOR_FOLDER = Path(ClearMap.gui.__file__).parent / 'creator'
MAIN_WINDOW_UI_FILE_NAME = 'mainwindow.ui'


def _root_class(ui_file):
    return ET.parse(ui_file).getroot().find('widget').get('class')


def test_only_the_main_window_is_a_qmainwindow():
    ui_files = sorted(CREATOR_FOLDER.glob('*.ui'))
    main_windows = [ui_file.name for ui_file in ui_files if _root_class(ui_file) == 'QMainWindow']

    assert main_windows == [MAIN_WINDOW_UI_FILE_NAME]
