"""
A line edit is read as the text typed in it, an empty field as the empty string. A field which holds something else
(the list of the tile extensions) reads its text with the casts of its ParamLink.
"""
from types import SimpleNamespace

import pytest

from ClearMap.gui import params_interfaces
from ClearMap.gui.params import SharedSampleParams
from ClearMap.gui.params_interfaces import QCheckBox, QLabel, QLineEdit


class FakeSignal:
    """A signal which keeps its slots: these widgets are never built by Qt (no QApplication), so the signals of the Qt
    class they stand for cannot be used"""
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot=None):
        self.slots = [] if slot is None else [kept for kept in self.slots if kept != slot]


def fake_signal(name):
    """A property which gives each widget its own signal. It comes before the signal of the Qt class in the MRO, which
    raises 'super-class __init__() was never called' with a real Qt"""
    def get_signal(widget):
        return widget.__dict__.setdefault(f'_signal_{name}', FakeSignal())
    return property(get_signal)


class FakeQtWidget:
    """What the parameter code asks any widget: the real Qt methods raise on a widget which Qt did not build"""
    def objectName(self):
        return ''

    def parentWidget(self):
        return None

    def blockSignals(self, blocked):
        """Return whether the signals were blocked before, like QObject.blockSignals"""
        previously_blocked = self.__dict__.get('_signals_blocked', False)
        self.__dict__['_signals_blocked'] = blocked
        return previously_blocked


class FakeTextWidget(FakeQtWidget):
    """The text of a line edit or a label of the sandbox (no Qt): the real getters and setters of the widget registry
    apply to it"""
    editingFinished = fake_signal('editingFinished')
    textChanged = fake_signal('textChanged')

    def __init__(self, text=''):
        self._text = text

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text


class FakeLineEdit(FakeTextWidget, QLineEdit):
    pass


class FakeLabel(FakeTextWidget, QLabel):
    pass


class FakeCheckBox(FakeQtWidget, QCheckBox):
    stateChanged = fake_signal('stateChanged')

    def __init__(self):
        self._checked = False

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        self._checked = checked


@pytest.fixture
def sample_params(monkeypatch):
    monkeypatch.setattr(params_interfaces, '_is_alive', lambda widget: widget is not None)
    tab = SimpleNamespace(sampleIdTxt=FakeLineEdit(), useIdAsPrefixCheckBox=FakeCheckBox(),
                          defaultTileExtensionLineEdit=FakeLineEdit(), srcFolderTxt=FakeLabel())
    config = {'sample': {'sample_id': 'mouse_1', 'use_id_as_prefix': False,
                         'default_tile_extension': ['.ome.tif', '.ome.npy']}}
    patches = []
    params = SharedSampleParams(tab, event_bus=None, get_view=lambda: config, apply_patch=patches.append)
    params.cfg_to_ui()
    return tab, params, patches


@pytest.mark.parametrize('text', ['mouse[1]', '/data/run[2]/tiles', '{x}', '[.ome.tif]'])
def test_a_text_with_brackets_is_written_as_it_was_typed(sample_params, text):
    tab, params, patches = sample_params
    tab.sampleIdTxt.setText(text)

    params.handle_widget_changed(attr_name='sample_id')  # Was a ValueError or SyntaxError of ast.literal_eval

    assert patches == [{'sample': {'sample_id': text}}]


def test_an_empty_field_is_written_as_the_empty_string(sample_params):
    tab, params, patches = sample_params
    tab.sampleIdTxt.setText('')

    params.handle_widget_changed(attr_name='sample_id')

    assert patches == [{'sample': {'sample_id': ''}}]  # Was None, which the schema of sample_id (a string) refuses


def test_none_is_shown_as_an_empty_field(sample_params):
    tab, params, _ = sample_params

    params.sample_id = None

    assert tab.sampleIdTxt.text() == ''  # Was the text "None"


def test_the_tile_extensions_are_shown_separated_by_commas(sample_params):
    tab, _, _ = sample_params

    assert tab.defaultTileExtensionLineEdit.text() == '.ome.tif, .ome.npy'  # Was the repr of the list


@pytest.mark.parametrize('text, tile_extensions', [
    ('.ome.tif, .ome.npy', ['.ome.tif', '.ome.npy']),
    ('.tif', ['.tif']),  # Was the string '.tif', which the schema refuses (an array)
    (' .tif ,, .npy,', ['.tif', '.npy']),
])
def test_the_tile_extensions_are_written_as_a_list(sample_params, text, tile_extensions):
    tab, params, patches = sample_params
    tab.defaultTileExtensionLineEdit.setText(text)

    params.handle_widget_changed(attr_name='default_tile_extension')

    assert patches == [{'sample': {'default_tile_extension': tile_extensions}}]
