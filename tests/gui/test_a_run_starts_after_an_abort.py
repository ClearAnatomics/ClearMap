"""
The Abort button sets ``stopped`` on the orchestrator and nothing cleared it: after one abort the steps guarded by
``check_stopped`` returned at once and the tabs left their sequence after the first step, until the experiment was
reloaded. Starting a run (opening its progress dialog) clears it.
"""
from unittest.mock import MagicMock

from ClearMap.Utils.utilities import check_stopped
from ClearMap.gui.tabs import RegistrationTab
from ClearMap.gui.tabs_interfaces import GenericTab
from ClearMap.pipeline_orchestrators.generic_orchestrators import PipelineOrchestrator


class GuardedOrchestrator(PipelineOrchestrator):
    pipeline = 'stitching'

    def __init__(self):
        super().__init__(MagicMock())
        self.n_runs = 0

    @check_stopped
    def step(self):
        self.n_runs += 1

    def resample_channel(self, channel, increment_main=True):
        pass


class FakeMainWindow:
    def __init__(self):
        self.progress_watcher = MagicMock()
        self.n_dialogs_opened = 0
        self.stopped_when_the_dialog_opened = []

    def make_progress_dialog(self, *args, abort=None, **kwargs):
        self.n_dialogs_opened += 1
        self.stopped_when_the_dialog_opened.append(getattr(getattr(abort, '__self__', None), 'stopped', None))

    def wrap_in_thread(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def print_status_msg(self, msg):
        pass


class FakeTab:
    wrap_step = GenericTab.wrap_step
    name = 'tab'

    def __init__(self):
        self.main_window = FakeMainWindow()
        self.worker = GuardedOrchestrator()


def aborted_tab():
    tab = FakeTab()
    tab.worker.stop_process()
    return tab


def test_a_step_runs_after_the_abort_of_the_previous_run():
    tab = aborted_tab()

    tab.wrap_step('Step', tab.worker.step, abort_func=tab.worker.stop_process)

    assert tab.worker.n_runs == 1
    assert tab.main_window.stopped_when_the_dialog_opened == [False]


def test_a_step_without_a_title_does_not_start_a_run():
    tab = aborted_tab()

    tab.wrap_step('', tab.worker.step, abort_func=tab.worker.stop_process)

    assert tab.worker.n_runs == 0
    assert tab.worker.stopped


def test_an_abort_function_that_is_not_an_orchestrator_method_is_left_alone():
    tab = aborted_tab()

    tab.wrap_step('Step', tab.worker.step, abort_func=print)
    tab.wrap_step('Step', tab.worker.step)

    assert tab.worker.n_runs == 0
    assert tab.worker.stopped


class FakeRegistrationTab:
    resample_channel = RegistrationTab.resample_channel
    run_registration = RegistrationTab.run_registration

    def __init__(self):
        self.main_window = FakeMainWindow()
        self.worker = GuardedOrchestrator()
        self.worker.align = self.worker.step
        self.params = {}
        self.sample_manager = MagicMock()

    def setup_atlas(self):
        pass

    def wrap_step(self, *args, **kwargs):
        pass


def test_the_registration_runs_after_the_abort_of_the_previous_one():
    tab = FakeRegistrationTab()
    tab.worker.stop_process()

    tab.run_registration()

    assert tab.worker.n_runs == 1
    assert tab.main_window.stopped_when_the_dialog_opened == [False]


def test_resampling_a_channel_starts_a_run():
    tab = FakeRegistrationTab()
    tab.worker.stop_process()

    tab.resample_channel('main')

    assert tab.main_window.stopped_when_the_dialog_opened == [False]
