"""
A line edit is read as the text typed in it, an empty field as the empty string. A field which holds something else
(the list of the tile extensions) reads its text with the casts of its ParamLink.
"""
from types import SimpleNamespace

import pytest

from ClearMap.gui import params_interfaces
from ClearMap.gui.params import SharedSampleParams
from ClearMap.gui.params_interfaces import QCheckBox, QLabel, QLineEdit


class FakeTextWidget:
    """The text of a line edit or a label of the sandbox (no Qt): the real getters and setters of the widget registry
    apply to it"""
    def __init__(self, text=''):
        self._text = text

    def text(self):
        return self._text

    def setText(self, text):
        self._text = text

    def objectName(self):
        return ''

    def parentWidget(self):
        return None


class FakeLineEdit(FakeTextWidget, QLineEdit):
    pass


class FakeLabel(FakeTextWidget, QLabel):
    pass


class FakeCheckBox(QCheckBox):
    def __init__(self):
        self._checked = False

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        self._checked = checked

    def objectName(self):
        return ''

    def parentWidget(self):
        return None


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
