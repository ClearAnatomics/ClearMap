"""The 'First channel' radio buttons of a colocalization pair are labelled with the names of the channels"""
from types import SimpleNamespace

from ClearMap.gui.tabs import ColocalizationTab

CHANNEL_A, CHANNEL_B = 'fos', 'test'


class FakeRadioButton:
    def __init__(self):
        self.text = 'placeholder from the .ui file'

    def setText(self, text):
        self.text = text


def test_first_channel_radio_buttons_show_the_names_of_the_channels_of_the_pair():
    page = SimpleNamespace(colocalizationChannelAFirstRadioButton=FakeRadioButton(),
                           colocalizationChannelBFirstRadioButton=FakeRadioButton())

    ColocalizationTab._name_first_channel_radio_buttons(page, CHANNEL_A, CHANNEL_B)

    assert page.colocalizationChannelAFirstRadioButton.text == CHANNEL_A
    assert page.colocalizationChannelBFirstRadioButton.text == CHANNEL_B
