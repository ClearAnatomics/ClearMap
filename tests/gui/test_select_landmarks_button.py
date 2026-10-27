"""
The "Select landmarks" button of the registration page of a channel opens the landmarks dialog of that channel.
Its lambda took the page index as a default argument, which the clicked signal replaced with its checked state
(UiRequestLandmarksDialog(page_index=False)), and the tab took the event for the channel: KeyError at every click.
"""
from ClearMap.Utils.event_bus import EventBus
from ClearMap.Utils.events import UiRequestLandmarksDialog
from ClearMap.gui import params, tabs
from ClearMap.gui.params import RegistrationParams
from ClearMap.gui.tabs import RegistrationTab

CHANNELS = ['autofluorescence', 'cfos']
CLICKED_CHANNEL = 'cfos'
NOT_CHECKED = False  # What clicked passes to its slots


class FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot, **kwargs):
        self.slots.append(slot)

    def emit(self, *values):
        for slot in self.slots:
            slot(*values)


class FakeButton:
    def __init__(self):
        self.clicked = FakeSignal()


class FakeChannelPage:
    def __init__(self):
        self.selectLandmarksPushButton = FakeButton()


class FakeChannelRegistrationParams:
    def __init__(self, tab, channel_name, **kwargs):
        self.tab = FakeChannelPage()
        self.page_index = CHANNELS.index(channel_name)  # Found when needed, as by the page id


class FakeChannelsTabWidget:
    def tabText(self, index):
        return CHANNELS[index]


class FakeRegistrationUi:
    channelsParamsTabWidget = FakeChannelsTabWidget()


class FakeRegistrationParams(dict):
    add_channel = RegistrationParams.add_channel
    request_landmarks_dialog = RegistrationParams.request_landmarks_dialog
    get_channel_name = RegistrationParams.get_channel_name

    def __init__(self, bus):
        super().__init__()
        self.tab = FakeRegistrationUi()
        self._bus = bus
        self._get_view = self._apply_patch = None

    @property
    def channels(self):
        return list(self)

    def publish(self, event):
        self._bus.publish(event)


class FakeImageAsset:
    def __init__(self, path):
        self.path = path


class FakeRegistrationWorker:
    def get_fixed_image(self, channel):
        return FakeImageAsset(f'{channel}_fixed.tif')

    def get_moving_image(self, channel):
        return FakeImageAsset(f'{channel}_moving.tif')

    def get_elx_asset(self, asset_type, channel):
        return FakeImageAsset(f'{channel}_{asset_type}.txt')


class FakeLandmarksSelectorDialog:
    def __init__(self, fixed_image_path, moving_image_path, fixed_image_landmarks_path, moving_image_landmarks_path):
        self.fixed_image_path = fixed_image_path
        self.data_viewers = {}

    def plot(self, lut=None, parent=None):
        pass


class FakePreferences:
    lut = 'flame'


class FakeMainWindow:
    def __init__(self):
        self.preference_editor = type('FakePreferenceEditor', (), {'params': FakePreferences()})()

    def centralWidget(self):
        return None

    def setup_plots(self, viewers):
        pass


class FakeRegistrationTab:
    launch_landmarks_dialog = RegistrationTab.launch_landmarks_dialog

    def __init__(self, registration_params):
        self.params = registration_params
        self.worker = FakeRegistrationWorker()
        self.main_window = FakeMainWindow()
        self.landmark_selector = None


def test_the_button_of_a_channel_opens_the_landmarks_dialog_of_that_channel(monkeypatch):
    monkeypatch.setattr(params, 'ChannelRegistrationParams', FakeChannelRegistrationParams)
    monkeypatch.setattr(tabs, 'LandmarksSelectorDialog', FakeLandmarksSelectorDialog)
    bus = EventBus()
    registration_params = FakeRegistrationParams(bus)
    tab = FakeRegistrationTab(registration_params)
    bus.subscribe(UiRequestLandmarksDialog, tab.launch_landmarks_dialog)
    for channel in CHANNELS:
        registration_params.add_channel(channel)

    registration_params[CLICKED_CHANNEL].tab.selectLandmarksPushButton.clicked.emit(NOT_CHECKED)

    assert tab.landmark_selector.fixed_image_path == f'{CLICKED_CHANNEL}_fixed.tif'
