"""
The config holds the landmarks weights as the sliders show them, in percent; the registration converts them to the
weights of elastix (elastix_landmarks_weight). The GUI converted them itself both ways, and the update of the
registration files of a channel (also run when its page is set up) took the converted weights for slider values:
it converted them again at each call, so a weight vanished or exploded (slider 50: 0.1, then 5e-5).
"""
from ClearMap.gui.params import ChannelRegistrationParams
from ClearMap.gui.widgets import FileDropListWidget

CHANNEL = 'cfos'
PARAMS_FILES = ['align_affine.txt', 'align_bspline.txt']
PERCENTS = [0, 50]


class FakeParamsFilesListWidget(FileDropListWidget):
    def __init__(self):  # Without Qt
        pass

    def get_items_text(self):
        return PARAMS_FILES


class FakeLandmarksWeightsPanel:
    def __init__(self):
        self.weights = None

    def set_items(self, names, weights=None):
        self.weights = weights


class FakeRegistrationPage:
    def __init__(self):
        self.paramsFilesListWidget = FakeParamsFilesListWidget()
        self.landmarksWeightsPanel = FakeLandmarksWeightsPanel()


class FakeChannelRegistrationParams:
    handle_params_files_changed = ChannelRegistrationParams.handle_params_files_changed
    handle_landmarks_weight_changed = ChannelRegistrationParams.handle_landmarks_weight_changed
    cfg_subtree = ['registration', 'channels', CHANNEL]
    name = CHANNEL

    def __init__(self, landmarks_weights):
        self.tab = FakeRegistrationPage()
        self.view = {'landmarks_weights': landmarks_weights, 'params_files': PARAMS_FILES}

    def _apply_patch(self, patch):
        self.view.update(patch['registration']['channels'][CHANNEL])

    def _update_value(self, keys, value):
        self.view[keys[-1]] = value


def test_updating_the_registration_files_keeps_the_percents_and_shows_them_on_the_sliders():
    channel_params = FakeChannelRegistrationParams(list(PERCENTS))

    channel_params.handle_params_files_changed()
    channel_params.handle_params_files_changed()

    assert channel_params.view['landmarks_weights'] == PERCENTS
    assert channel_params.tab.landmarksWeightsPanel.weights == PERCENTS


def test_a_slider_writes_its_percent_to_the_config():
    channel_params = FakeChannelRegistrationParams(list(PERCENTS))

    channel_params.handle_landmarks_weight_changed(0, 30)

    assert channel_params.view['landmarks_weights'] == [30, 50]
