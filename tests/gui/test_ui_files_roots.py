"""
The root widget of a designer file is the class of the widget it is loaded into. Only the main window is a
QMainWindow: the other files are loaded into a QWidget or a QDialog (gui_utils_base.create_clearmap_widget), which
have no central widget, menu bar or status bar to receive those of a QMainWindow form.
"""
import ast
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


def test_no_designer_file_sets_a_style_sheet_on_its_root_widget():
    """The loader sets every property of the root: a root style sheet made for the preview would apply in the GUI"""
    ui_files = sorted(CREATOR_FOLDER.glob('*.ui'))
    with_style_sheet = [ui_file.name for ui_file in ui_files if any(
        root_property.get('name') == 'styleSheet' and (root_property.findtext('string') or '').strip()
        for root_property in ET.parse(ui_file).getroot().find('widget').findall('property'))]

    assert not with_style_sheet, f'{with_style_sheet}: see gui/creator/README.rst for a dark preview in Designer'


QT_CLASS_ANCESTORS = {'QWidget': {'QWidget'}, 'QDialog': {'QDialog', 'QWidget'}, 'QTabWidget': {'QTabWidget', 'QWidget'},
                      'QMainWindow': {'QMainWindow', 'QWidget'}}
WIZARD_DEFAULT_WIDGET_CLASS = 'QDialog'  # widget_class_name of WizardWidget
WIZARD_WIDGET_CLASS_POSITION = 6  # Its position among the arguments of WizardWidget.__init__ (after ui_name)


def _constant(node):
    return node.value if isinstance(node, ast.Constant) else None


def _widget_classes_named_for_designer_files():
    """(file, line, ui file stem, class) of every literal call that builds a designer file into a widget of a class"""
    root_classes = {ui_file.stem: _root_class(ui_file) for ui_file in CREATOR_FOLDER.glob('*.ui')}
    named = []
    for source_file in sorted(Path(ClearMap.gui.__file__).parent.glob('*.py')):
        for node in ast.walk(ast.parse(source_file.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            args = [_constant(arg) for arg in node.args]
            keywords = {keyword.arg: _constant(keyword.value) for keyword in node.keywords}
            ui_name = Path(args[0]).stem if isinstance(args[0], str) else None
            if ui_name not in root_classes:
                continue
            if getattr(node.func, 'id', None) == 'create_clearmap_widget':
                widget_class = args[1] if len(args) > 1 else keywords.get('widget_class_name')
            elif ast.unparse(node.func) == 'super().__init__':  # A WizardWidget
                widget_class = (args[WIZARD_WIDGET_CLASS_POSITION] if len(args) > WIZARD_WIDGET_CLASS_POSITION
                                else keywords.get('widget_class_name', WIZARD_DEFAULT_WIDGET_CLASS))
            else:
                continue
            named.append((source_file.name, node.lineno, ui_name, widget_class, root_classes[ui_name]))
    return named


def test_the_widget_class_named_for_a_designer_file_is_its_root_class_or_a_subclass_of_it():
    """uic.loadUi refuses a widget which is not an instance of the root class ('Wrong base class of toplevel widget')"""
    named = _widget_classes_named_for_designer_files()
    wrong = [f'{file}:{line} {ui_name}.ui is a {root_class}, not built into a {widget_class}'
             for file, line, ui_name, widget_class, root_class in named
             if root_class not in QT_CLASS_ANCESTORS.get(widget_class, set())]

    assert len(named) >= 5, 'the calls were not found'
    assert not wrong
