"""A ParamLink with section level keys reads and writes relative to its section, not to the cfg_subtree of its parameter"""
import pytest

from ClearMap.gui import params_interfaces
from ClearMap.gui.params_interfaces import ParamLink, UiParameter


class FakeWidget:
    def __init__(self):
        self.value = None

    def objectName(self):
        return ''

    def parentWidget(self):
        return None


class FakeWidgetOps:
    def get(self, widget):
        return widget.value

    def set(self, widget, value, silent=False):
        widget.value = value

    def connect(self, widget, callback):
        return lambda: None


class PairParams(UiParameter):
    n_processes: int
    diameter: int

    def __init__(self, *args, **kwargs):
        self.widgets = {'n_processes': FakeWidget(), 'diameter': FakeWidget()}
        super().__init__(*args, **kwargs)

    @property
    def cfg_subtree(self):
        return ['colocalization', 'channels', 'a-b']

    def build_params_dict(self):
        return {
            'n_processes': ParamLink(['performance', 'n_processes'],
                                     self.widgets['n_processes'], section_level=True),
            'diameter': ParamLink(['comparison', 'particle_diameter'], self.widgets['diameter']),
        }


CONFIG = {'colocalization': {'performance': {'n_processes': 7},
                             'channels': {'a-b': {'comparison': {'particle_diameter': 30}}}}}


@pytest.fixture
def params(monkeypatch):
    monkeypatch.setattr(params_interfaces, 'WIDGET_OPS', FakeWidgetOps())
    monkeypatch.setattr(params_interfaces, '_is_alive', lambda widget: widget is not None)
    monkeypatch.setattr(params_interfaces, 'QFrame', FakeWidget.__mro__[1])  # No Qt needed: the fakes are no frames
    patches = []
    return PairParams(object(), event_bus=None, get_view=lambda: CONFIG, apply_patch=patches.append), patches


def test_section_level_keys_are_read_from_the_section(params):
    pair, _ = params
    pair.cfg_to_ui()  # Was KeyError: 'performance' (looked for in the subtree of the pair)

    assert pair.widgets['n_processes'].value == 7
    assert pair.widgets['diameter'].value == 30  # The relative keys still follow the cfg_subtree


def test_section_level_keys_are_written_to_the_section(params):
    pair, patches = params
    pair.widgets['n_processes'].value = 3
    pair.widgets['diameter'].value = 40

    pair.handle_widget_changed(attr_name='n_processes')
    pair.handle_widget_changed(attr_name='diameter')

    assert patches == [{'colocalization': {'performance': {'n_processes': 3}}},
                       {'colocalization': {'channels': {'a-b': {'comparison': {'particle_diameter': 40}}}}}]
