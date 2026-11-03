"""
An alignment canceled with the Abort button leaves its orchestrator stopped, like any other step: the next run clears
the flag when it starts (reset_stopped). align() cleared it at its end, so whoever looked after it saw no abort.
"""
from ClearMap.pipeline_orchestrators import registration_orchestrator
from ClearMap.pipeline_orchestrators.generic_orchestrators import CanceledProcessing
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor


class AbortedRegistrationProcessor:
    """The alignment of the first channel is canceled by the Abort button (stop_process terminates elastix)"""
    align = RegistrationProcessor.align
    config = {}
    channels_to_register = ['autofluorescence', 'cfos']

    def __init__(self):
        self.stopped = False
        self.events = []

    def align_channel(self, channel):
        self.stopped = True
        raise CanceledProcessing

    def update_watcher_main_progress(self):
        pass

    def publish(self, event):
        self.events.append(event)


def chosen(config, channels):
    pass


def test_a_canceled_alignment_leaves_the_orchestrator_stopped(monkeypatch):
    monkeypatch.setattr(registration_orchestrator, 'check_partners_chosen', chosen)
    registration = AbortedRegistrationProcessor()

    registration.align()

    assert registration.stopped
    assert len(registration.events) == 1  # The status changed (the first alignments may be done)
