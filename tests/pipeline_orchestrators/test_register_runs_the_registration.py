"""
RegistrationProcessor.register is the registration the scripts and the batch processing run: the atlases of the
orientation and cropping of the sample, then the resampling of the channels, then their alignment.
"""
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor


class RecordingRegistrationProcessor:
    register = RegistrationProcessor.register

    def __init__(self):
        self.steps = []

    def _run(self, step):
        self.steps.append(step)

    def setup_atlases(self):
        self._run('setup_atlases')

    def resample_for_registration(self):
        self._run('resample_for_registration')

    def align(self):
        self._run('align')


def test_register_sets_the_atlases_up_resamples_then_aligns():
    registration = RecordingRegistrationProcessor()
    registration.register()
    assert registration.steps == ['setup_atlases', 'resample_for_registration', 'align']

