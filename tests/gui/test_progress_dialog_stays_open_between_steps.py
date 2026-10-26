"""
A run of several steps shows them in one progress dialog: wrap_step closes it only with close_when_done, which the
callers chaining steps set on their last one only. wrap_in_thread, which runs each step, closed it at the end of
every step (since the rename of run_gui to app), so the steps after the first one ran with no dialog.
(ClearMapAppBase.wrap_in_thread calls run_with_responsive_ui: app.py cannot be imported in a test, it
builds the application and its splash screen.)
"""
from ClearMap.gui.gui_utils_base import run_with_responsive_ui
from ClearMap.gui.tabs_interfaces import GenericTab


class RecordingProgressWatcher:
    def __init__(self):
        self.n_finished = 0
        self.main_step_name = 'Stitching'

    def start_polling(self):
        pass

    def stop_polling(self):
        pass

    def finish(self):
        self.n_finished += 1


class FakeApp:
    def processEvents(self):
        pass


class FakeMainWindow:
    """The parts of the main window wrap_step uses, its wrap_in_thread as in ClearMapAppBase"""

    def __init__(self):
        self.app = FakeApp()
        self.progress_watcher = RecordingProgressWatcher()
        self.n_dialogs_opened = 0

    def wrap_in_thread(self, func, *args, **kwargs):
        return run_with_responsive_ui(self.app, self.progress_watcher, func, *args, **kwargs)

    def make_progress_dialog(self, *args, **kwargs):
        self.n_dialogs_opened += 1

    def print_status_msg(self, msg):
        pass

    def log_progress(self, msg):
        pass


class FakeTab:
    wrap_step = GenericTab.wrap_step
    name = 'stitching'

    def __init__(self):
        self.main_window = FakeMainWindow()


def run_step():
    return None


def test_the_dialog_stays_open_until_the_last_step_of_a_run():
    tab = FakeTab()
    tab.wrap_step('Stitching', run_step, n_steps=3, close_when_done=False)
    tab.wrap_step('', run_step, n_steps=3, close_when_done=False)
    assert tab.main_window.progress_watcher.n_finished == 0
    tab.wrap_step('', run_step, n_steps=3)
    assert tab.main_window.progress_watcher.n_finished == 1
    assert tab.main_window.n_dialogs_opened == 1


def test_running_a_function_in_a_thread_leaves_the_dialog_to_its_caller():
    progress_watcher = RecordingProgressWatcher()
    run_with_responsive_ui(FakeApp(), progress_watcher, run_step)
    assert progress_watcher.n_finished == 0
