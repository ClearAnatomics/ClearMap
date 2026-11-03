"""
The Abort button stops the step that runs (it sets ``stopped`` on its orchestrator and cancels its processes). A tab
that runs several steps in a row checks the orchestrator of the step that just ran before it starts the next one:
the user stopped the whole sequence. The registration loop and the vasculature tab went on with the next step.
"""
import functools
from types import SimpleNamespace

import pytest

from ClearMap.gui.tabs import RegistrationTab, VasculatureTab
from ClearMap.gui.tabs_interfaces import run_until_aborted
from ClearMap.pipeline_orchestrators.tube_map import BinaryVesselProcessorSteps

CHANNELS = ('arteries', 'veins')


class RecordingWorker:
    """An orchestrator whose steps record their calls; the step named abort_at is aborted by the user"""

    def __init__(self, calls, abort_at=None):
        self.stopped = False
        self._calls = calls
        self._abort_at = abort_at

    def stop_process(self):
        self.stopped = True

    def run_step(self, name, *args, **kwargs):
        self._calls.append(name)
        if name == self._abort_at:
            self.stop_process()


class BinaryWorker(RecordingWorker):
    channels_to_binarize = CHANNELS

    def __init__(self, calls, abort_at=None):
        super().__init__(calls, abort_at)
        steps = list(BinaryVesselProcessorSteps._default_steps)
        self.steps = {channel: SimpleNamespace(steps=steps) for channel in CHANNELS}

    def check_inputs_pixel_aligned(self, require_stitched):
        pass

    def binarize_channel(self, channel):
        self.run_step(f'binarize {channel}')

    def smooth_channel(self, channel):
        self.run_step(f'smooth {channel}')

    def deep_fill_channel(self, channel):
        self.run_step(f'deep_fill {channel}')

    def fill_channel(self, channel):
        self.run_step(f'fill {channel}')

    def combine_binary(self):
        self.run_step('combine')


class GraphWorker(RecordingWorker):
    def skeletonize_and_build_graph(self, binary_processor):
        self.run_step('build')

    def clean_graph(self):
        self.run_step('clean')

    def reduce_graph(self):
        self.run_step('reduce')

    def register(self):
        self.run_step('register')

    def post_process(self):
        self.run_step('post_process')


def run_the_step(task_name, func, step_args=None, step_kw_args=None, **kwargs):
    """wrap_step without the dialog and the thread"""
    func(*(step_args or []), **(step_kw_args or {}))


class FakeVasculatureTab:
    run_all = VasculatureTab.run_all
    build_graph = VasculatureTab.build_graph
    binarize_and_postprocess_channel = VasculatureTab.binarize_and_postprocess_channel
    _run_binarization_step = VasculatureTab._run_binarization_step
    combine = VasculatureTab.combine
    post_process_graph = VasculatureTab.post_process_graph

    def __init__(self, abort_at):
        self.calls = []
        self.workers = {'binary': BinaryWorker(self.calls, abort_at), 'graph': GraphWorker(self.calls, abort_at)}
        self.wrap_step = run_the_step

    def get_worker(self, substep):
        return self.workers[substep]

    def voxelize(self):
        self.calls.append('voxelize')


def binarization_of(channel):
    return [f'{step} {channel}' for step in ('binarize', 'smooth', 'deep_fill', 'fill')]  # The default order


WHOLE_VASCULATURE_PIPELINE = (binarization_of('arteries') + binarization_of('veins') +
                              ['combine', 'build', 'clean', 'reduce', 'register', 'post_process', 'voxelize'])


def test_the_whole_vasculature_pipeline_runs_without_an_abort():
    tab = FakeVasculatureTab(abort_at=None)
    tab.run_all()
    assert tab.calls == WHOLE_VASCULATURE_PIPELINE


@pytest.mark.parametrize('aborted_step', WHOLE_VASCULATURE_PIPELINE[:-1])
def test_aborting_any_step_stops_the_vasculature_pipeline_after_it(aborted_step):
    tab = FakeVasculatureTab(abort_at=aborted_step)
    tab.run_all()
    assert tab.calls == WHOLE_VASCULATURE_PIPELINE[:WHOLE_VASCULATURE_PIPELINE.index(aborted_step) + 1]


class ResamplingWorker(RecordingWorker):
    def reset_stopped(self):
        self.stopped = False

    def get(self, asset_type, channel):
        return SimpleNamespace(exists=True)

    def resample_channel(self, channel, increment_main):
        self.run_step(f'resample {channel}')

    def align(self):
        self.run_step('align')


class FakeProgressWatcher:
    def finish(self):
        pass

    def increment_main_progress(self):
        pass


class FakeMainWindow:
    def __init__(self):
        self.progress_watcher = FakeProgressWatcher()
        self.status_messages = []

    def make_progress_dialog(self, *args, **kwargs):
        pass

    def wrap_in_thread(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def print_status_msg(self, msg):
        self.status_messages.append(msg)


class FakeRegistrationTab:
    run_registration = RegistrationTab.run_registration

    def __init__(self, abort_at):
        self.calls = []
        self.worker = ResamplingWorker(self.calls, abort_at)
        self.params = {channel: SimpleNamespace(resample=True) for channel in CHANNELS}
        self.main_window = FakeMainWindow()
        self.wrap_step = run_the_step

    def setup_atlas(self):
        pass


def test_aborting_the_resampling_of_a_channel_stops_the_registration():
    tab = FakeRegistrationTab(abort_at='resample arteries')
    tab.run_registration()
    assert tab.calls == ['resample arteries']
    assert tab.main_window.status_messages == ['Registration aborted']


def test_the_registration_resamples_every_channel_then_aligns_without_an_abort():
    tab = FakeRegistrationTab(abort_at=None)
    tab.run_registration()
    assert tab.calls == ['resample arteries', 'resample veins', 'align']


class StoppableWorker:
    def __init__(self):
        self.stopped = False


def test_the_sequence_stops_after_the_step_whose_orchestrator_was_aborted():
    ran = []
    first, second = StoppableWorker(), StoppableWorker()

    def abort_the_second():
        ran.append('second')
        second.stopped = True

    steps = [(functools.partial(ran.append, 'first'), first), (abort_the_second, second),
             (functools.partial(ran.append, 'third'), first)]

    assert not run_until_aborted(steps)
    assert ran == ['first', 'second']


def test_the_sequence_runs_all_its_steps_when_nothing_is_aborted():
    ran = []
    steps = [(functools.partial(ran.append, 'first'), StoppableWorker())] * 2

    assert run_until_aborted(steps)
    assert ran == ['first', 'first']
