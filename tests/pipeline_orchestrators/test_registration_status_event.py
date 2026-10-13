"""The end of the alignment is announced to the subscribers (the GUI updates the registered channels from it)"""
from ClearMap.Utils.events import RegistrationStatusChanged


class Recorder:
    def __init__(self):
        self.events = []

    def on_status(self, event):
        self.events.append(event)


def test_align_publishes_registration_status_changed(experiment, channel_editor):
    channel_editor.add('nuclei', 'nuclei', with_path=True)
    experiment.apply_ui_patch({'registration': {'channels': {'nuclei': {'align_with': None}}}})  # Not registered
    registration = experiment.get_worker('registration', None)
    assert registration.channels_to_register == []
    recorder = Recorder()
    experiment._bus.subscribe(RegistrationStatusChanged, recorder.on_status)

    registration.align()  # Nothing to register: only the announcement

    assert recorder.events == [RegistrationStatusChanged()]
