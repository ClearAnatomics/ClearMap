"""
A designer file is built into a widget of its root class (qtpy.uic.loadUi), without its resource includes. The title given by the caller is set after the load, which sets the title written in the file.

qtpy's uic and widgets are replaced by fakes: the sandbox has no Qt. The load with a real Qt is checked by
test_designer_files_build_with_the_qt_binding.
"""
import types
import xml.etree.ElementTree as ET

import pytest
import qtpy

from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.gui import gui_utils_base

UI_FILE_TITLE = 'Dialog'  # The windowTitle of nested_progress_dialog.ui


class FakeDialog:
    def __init__(self):
        self.titles = []

    def setWindowTitle(self, title):
        self.titles.append(title)


class FakeUic:
    """loadUi: keeps the designer text it was given, and sets the title of the file, as the real one does"""
    def __init__(self):
        self.loaded = []

    def loadUi(self, ui_file, widget):
        self.loaded.append((ET.parse(ui_file).getroot(), widget))
        widget.setWindowTitle(UI_FILE_TITLE)


@pytest.fixture()
def fake_qt(monkeypatch):
    uic = FakeUic()
    monkeypatch.setattr(qtpy, 'uic', uic, raising=False)
    monkeypatch.setattr(qtpy.QtWidgets, 'QDialog', FakeDialog, raising=False)
    patched = []
    monkeypatch.setattr(gui_utils_base, 'apply_button_box_texts', patched.append)
    return uic, patched


def test_the_file_is_loaded_into_a_widget_of_the_root_class_of_the_file(fake_qt):
    uic, patched = fake_qt

    widget = gui_utils_base.create_clearmap_widget('nested_progress_dialog')

    assert isinstance(widget, FakeDialog)
    assert [loaded_widget for _, loaded_widget in uic.loaded] == [widget]
    assert patched == [widget]


def test_the_file_is_loaded_without_its_resource_includes(fake_qt):
    uic, _ = fake_qt

    gui_utils_base.create_clearmap_widget('nested_progress_dialog')

    loaded_root = uic.loaded[0][0]
    assert loaded_root.find('widget') is not None and loaded_root.find('resources') is None


def test_the_title_of_the_caller_replaces_the_title_of_the_file(fake_qt):
    widget = gui_utils_base.create_clearmap_widget('nested_progress_dialog', window_title='Progress')

    assert widget.titles == [UI_FILE_TITLE, 'Progress']


def test_a_root_which_is_not_a_qt_widgets_class_is_refused(fake_qt, monkeypatch):
    monkeypatch.setattr(gui_utils_base, 'QtWidgets', types.SimpleNamespace())  # No class: what a custom root is

    with pytest.raises(ClearMapValueError, match='nested_progress_dialog.ui is a QDialog'):
        gui_utils_base.create_clearmap_widget('nested_progress_dialog')


class FakeButton:
    def __init__(self):
        self.texts = []

    def setText(self, text):
        self.texts.append(text)


class FakeButtonBox:
    def __init__(self, **text_properties):
        self._properties = text_properties
        self.buttons = {}

    def property(self, name):
        return self._properties.get(name)

    def button(self, standard_button):
        return self.buttons.setdefault(standard_button, FakeButton())


class FakeContainer:
    def __init__(self, *button_boxes):
        self._button_boxes = button_boxes

    def findChildren(self, widget_class):
        return list(self._button_boxes)


def test_the_text_properties_of_a_button_box_give_the_text_of_its_standard_buttons():
    box = FakeButtonBox(applyText='Validate', okText='Go', openText='Plot')

    gui_utils_base.apply_button_box_texts(FakeContainer(box))

    buttons = {property_name: box.buttons[standard_button].texts for property_name, standard_button
               in gui_utils_base.BUTTON_TEXT_PROPERTY_TO_STANDARD_BUTTON.items()}
    assert buttons == {'applyText': ['Validate'], 'okText': ['Go'], 'openText': ['Plot']}


def test_a_button_box_without_text_properties_keeps_the_text_of_its_buttons():
    box = FakeButtonBox()

    gui_utils_base.apply_button_box_texts(FakeContainer(box))

    assert box.buttons == {}
