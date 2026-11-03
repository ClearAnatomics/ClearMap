"""
The "Go to sample orientation" button of the registration tab (below the reference channel) was connected to nothing,
and the jump of the sample tab it was meant for (go_to_orientation, unused) went to a section which no longer holds the
orientation: it is in the geometry section of each channel page. The button shows that of the reference channel.
"""
from ClearMap.Utils.event_bus import EventBus
from ClearMap.Utils.events import UiRequestShowChannelOrientation
from ClearMap.config.registration_channels import REFERENCE_UNDECIDED
from ClearMap.gui.tabs import RegistrationTab, SampleInfoTab

REFERENCE_CHANNEL = 'autofluorescence'
OTHER_CHANNEL = 'cfos'
GEOMETRY_PAGE = 'geometry page'


class FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class FakeButton:
    def __init__(self):
        self.clicked = FakeSignal()


class FakeRegistrationUi:
    """Any widget of the registration tab: a button which records what its click is connected to"""

    def __init__(self):
        self.buttons = {}

    def __getattr__(self, name):
        return self.buttons.setdefault(name, FakeButton())


class FakeSampleManager:
    channels = [REFERENCE_CHANNEL, OTHER_CHANNEL]

    def __init__(self, reference_channel_setting):
        self.reference_channel_setting = reference_channel_setting


class FakeRegistrationTab:
    _bind = RegistrationTab._bind
    go_to_reference_channel_orientation = RegistrationTab.go_to_reference_channel_orientation

    def __init__(self, bus, reference_channel_setting=REFERENCE_CHANNEL):
        self.ui = FakeRegistrationUi()
        self.bus = bus
        self.sample_manager = FakeSampleManager(reference_channel_setting)

    def _RegistrationTab__populate_atlas_comboboxes(self):
        pass

    def subscribe(self, event_type, handler):
        pass

    def publish(self, event):
        self.bus.publish(event)

    def run_registration(self):
        pass

    def plot_registration_results(self):
        pass

    def _on_bus_partner_defaults(self, event):
        pass

    def handle_align_with_changed(self, event):
        pass


class FakeToolBox:
    def __init__(self):
        self.current_widget = 'paths page'

    def setCurrentWidget(self, widget):
        self.current_widget = widget


class FakeChannelPage:
    def __init__(self):
        self.toolBox = FakeToolBox()
        self.geometryPage = GEOMETRY_PAGE


class FakeChannelsTabWidget:
    def __init__(self):
        self.pages = {REFERENCE_CHANNEL: FakeChannelPage(), OTHER_CHANNEL: FakeChannelPage()}
        self.current_page = self.pages[OTHER_CHANNEL]

    def get_channel_widget(self, name=None):
        return self.current_page if name is None else self.pages.get(name)

    def setCurrentWidget(self, page):
        self.current_page = page


class FakeTabWidget:
    def __init__(self):
        self.current_widget = 'registration tab'

    def setCurrentWidget(self, widget):
        self.current_widget = widget


class FakeMainWindow:
    def __init__(self):
        self.tabWidget = FakeTabWidget()


class FakeSampleUi:
    def __init__(self):
        self.channelsParamsTabWidget = FakeChannelsTabWidget()


class FakeSampleTab:
    go_to_orientation = SampleInfoTab.go_to_orientation

    def __init__(self, bus):
        self.ui = FakeSampleUi()
        self.main_window = FakeMainWindow()
        bus.subscribe(UiRequestShowChannelOrientation, self.go_to_orientation)


def click_go_to_sample_orientation(registration_tab):
    registration_tab._bind()
    for slot in registration_tab.ui.registrationGoToSampleOrientationPushButton.clicked.slots:
        slot()


def test_the_button_shows_the_orientation_of_the_reference_channel_in_the_sample_tab():
    bus = EventBus()
    sample_tab = FakeSampleTab(bus)

    click_go_to_sample_orientation(FakeRegistrationTab(bus))

    reference_page = sample_tab.ui.channelsParamsTabWidget.pages[REFERENCE_CHANNEL]
    assert sample_tab.main_window.tabWidget.current_widget is sample_tab.ui
    assert sample_tab.ui.channelsParamsTabWidget.current_page is reference_page
    assert reference_page.toolBox.current_widget == GEOMETRY_PAGE


def test_without_reference_channel_the_button_shows_the_orientation_of_the_channel_page_shown():
    bus = EventBus()
    sample_tab = FakeSampleTab(bus)

    click_go_to_sample_orientation(FakeRegistrationTab(bus, reference_channel_setting=REFERENCE_UNDECIDED))

    shown_page = sample_tab.ui.channelsParamsTabWidget.pages[OTHER_CHANNEL]
    assert sample_tab.ui.channelsParamsTabWidget.current_page is shown_page
    assert shown_page.toolBox.current_widget == GEOMETRY_PAGE
