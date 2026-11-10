"""
A designer file is built into a widget of the Qt class that its caller names (qtpy.uic.loadUi), without its resource
includes. The title given by the caller is set after the load, which sets the title written in the file.

qtpy's uic and widgets are replaced by fakes: the sandbox has no Qt. The load with a real Qt is checked by
test_designer_files_build_with_the_qt_binding.
"""
import xml.etree.ElementTree as ET

import pytest
import qtpy

from ClearMap.gui import gui_utils_base, widget_monkeypatch_callbacks

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
    monkeypatch.setattr(widget_monkeypatch_callbacks, 'recursive_patch_widgets', patched.append)
    return uic, patched


def test_the_file_is_loaded_into_a_widget_of_the_class_named_by_the_caller(fake_qt):
    uic, patched = fake_qt

    widget = gui_utils_base.create_clearmap_widget('nested_progress_dialog', widget_class_name='QDialog')

    assert isinstance(widget, FakeDialog)
    assert [loaded_widget for _, loaded_widget in uic.loaded] == [widget]
    assert patched == [widget]


def test_the_file_is_loaded_without_its_resource_includes(fake_qt):
    uic, _ = fake_qt

    gui_utils_base.create_clearmap_widget('nested_progress_dialog', widget_class_name='QDialog')

    loaded_root = uic.loaded[0][0]
    assert loaded_root.find('widget') is not None and loaded_root.find('resources') is None


def test_the_title_of_the_caller_replaces_the_title_of_the_file(fake_qt):
    widget = gui_utils_base.create_clearmap_widget('nested_progress_dialog', widget_class_name='QDialog',
                                                   window_title='Progress')

    assert widget.titles == [UI_FILE_TITLE, 'Progress']
