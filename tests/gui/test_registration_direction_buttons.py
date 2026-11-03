"""
The direction of the registration of a channel (moving_channel), chosen with radio buttons relative to the channel:
painted from the config, written relative to the channel, available once the channel has a partner.
"""
import pytest

from ClearMap.Utils.events import UiAlignWithChanged
from ClearMap.config.registration_channels import ALIGN_WITH_UNDECIDED, INTRINSICALLY_ALIGNED, MOVING_OTHER, MOVING_SELF
from ClearMap.gui import params
from ClearMap.gui.params import ChannelRegistrationParams

CHANNEL = 'fos'


class FakeRadioButton:
    def __init__(self):
        self.text, self.checked = '', False

    def setText(self, text):
        self.text = text

    def setChecked(self, checked):
        self.checked = checked


class FakeButtonGroup:
    def __init__(self):
        self.exclusive = True

    def setExclusive(self, exclusive):
        self.exclusive = exclusive


class FakeFrame:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled):
        self.enabled = enabled


class FakePage:
    def __init__(self):
        self.movingDirectionFrame = FakeFrame()


class FakeRegistrationParams:
    """The parts of ChannelRegistrationParams the direction reads and writes, with the real methods"""
    _paint_direction = ChannelRegistrationParams._paint_direction
    handle_direction_toggled = ChannelRegistrationParams.handle_direction_toggled
    handle_align_with_edited = ChannelRegistrationParams.handle_align_with_edited

    def __init__(self, align_with, moving_channel):
        self.name = CHANNEL
        self.view = {'align_with': align_with, 'moving_channel': moving_channel}
        self.align_with = align_with  # What the combo box says
        self.tab = FakePage()
        self._painting = False
        self._direction_buttons = FakeButtonGroup()
        self._direction_radio_buttons = {direction: FakeRadioButton()
                                         for direction in (MOVING_OTHER, INTRINSICALLY_ALIGNED, MOVING_SELF)}
        self.patches, self.events = [], []

    def _emit_patch(self, relative_path, value):
        self.patches.append((relative_path, value))

    def publish(self, event):
        self.events.append(event)

    def checked(self):
        return [direction for direction, button in self._direction_radio_buttons.items() if button.checked]


@pytest.fixture()
def warnings_shown(monkeypatch):
    shown = []

    class MessageBox:
        @staticmethod
        def warning(parent, title, text):
            shown.append(text)
    monkeypatch.setattr(params, 'QMessageBox', MessageBox)
    return shown


@pytest.mark.parametrize('align_with, moving_channel, checked', [
    ('auto', 'other', [MOVING_OTHER]), ('auto', 'auto', [MOVING_OTHER]),  # Relative, or the name of the partner
    ('auto', 'fos', [MOVING_SELF]), ('auto', 'intrinsically_aligned', [INTRINSICALLY_ALIGNED]),
    ('auto', None, []), ('atlas', 'auto', []),  # No direction, or a name left over from another partner
])
def test_the_button_of_the_direction_of_the_config_is_checked(align_with, moving_channel, checked):
    registration = FakeRegistrationParams(align_with, moving_channel)

    registration._paint_direction()

    assert registration.checked() == checked
    assert registration.tab.movingDirectionFrame.enabled
    assert registration._direction_buttons.exclusive and not registration._painting


def test_the_buttons_are_named_after_the_channel_and_its_partner():
    registration = FakeRegistrationParams('auto', 'other')

    registration._paint_direction()

    texts = {direction: button.text for direction, button in registration._direction_radio_buttons.items()}
    assert texts == {MOVING_OTHER: 'auto moves onto fos', INTRINSICALLY_ALIGNED: 'Acquired together with auto',
                     MOVING_SELF: 'fos moves onto auto'}


@pytest.mark.parametrize('align_with', [None, ALIGN_WITH_UNDECIDED])
def test_without_a_partner_the_buttons_are_disabled_and_show_the_direction_kept(align_with):
    registration = FakeRegistrationParams(align_with, 'self')

    registration._paint_direction()

    assert not registration.tab.movingDirectionFrame.enabled
    assert registration.checked() == [MOVING_SELF]


def test_choosing_a_direction_writes_it_relative_to_the_channel(warnings_shown):
    registration = FakeRegistrationParams('auto', 'auto')

    registration.handle_direction_toggled(INTRINSICALLY_ALIGNED, True)
    registration.handle_direction_toggled(MOVING_OTHER, False)  # The one unchecked by the choice

    assert registration.patches == [(['moving_channel'], INTRINSICALLY_ALIGNED)]
    assert warnings_shown == []


def test_choosing_that_the_channel_moves_warns_about_its_points(warnings_shown):
    registration = FakeRegistrationParams('auto', 'other')

    registration.handle_direction_toggled(MOVING_SELF, True)

    assert registration.patches == [(['moving_channel'], MOVING_SELF)]
    (text,) = warnings_shown
    assert 'fos will be registered onto auto' in text and 'inverse' in text


def test_painting_the_buttons_writes_nothing(warnings_shown):
    registration = FakeRegistrationParams('auto', 'other')
    registration._painting = True

    registration.handle_direction_toggled(MOVING_SELF, True)

    assert registration.patches == [] and warnings_shown == []


def test_changing_the_partner_writes_it_with_the_direction_relative_to_the_channel():
    registration = FakeRegistrationParams('auto', 'auto')  # The name of the old partner: it moved
    registration.align_with = 'atlas'  # Chosen in the combo box

    registration.handle_align_with_edited()

    assert registration.patches == [([], {'align_with': 'atlas', 'moving_channel': 'other'})]
    assert registration.events == [UiAlignWithChanged(channel_name=CHANNEL, align_with='atlas')]
