"""The direction radio buttons of a colocalization pair show the two comparisons: first → second"""
from types import SimpleNamespace

from ClearMap.gui.tabs import ColocalizationTab

CHANNEL_A, CHANNEL_B = 'fos', 'test'


class FakeRadioButton:
    def __init__(self):
        self.text = 'placeholder from the .ui file'

    def setText(self, text):
        self.text = text


def test_direction_radio_buttons_show_both_directions_of_the_pair():
    page = SimpleNamespace(colocalizationChannelAFirstRadioButton=FakeRadioButton(),
                           colocalizationChannelBFirstRadioButton=FakeRadioButton())

    ColocalizationTab._name_direction_radio_buttons(page, CHANNEL_A, CHANNEL_B)

    assert page.colocalizationChannelAFirstRadioButton.text == f'{CHANNEL_A} → {CHANNEL_B}'
    assert page.colocalizationChannelBFirstRadioButton.text == f'{CHANNEL_B} → {CHANNEL_A}'
