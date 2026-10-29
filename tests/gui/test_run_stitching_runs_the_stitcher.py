"""
The stitching tab runs StitchingProcessor.stitch in one progress dialog, sized with the steps of the channels set to
run (it ran the steps of each channel itself, with a copy of the logic of the processor: the batch and the scripts
differed).
The tiles to convert have a dialog of their own, which opens only when some are to convert.
"""
from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.gui.tabs import StitchingTab

N_STEPS = 5


class FakeStitcher:
    def __init__(self, channels_to_convert=()):
        self.channels_to_convert = channels_to_convert

    def channels_to_stitch(self):
        return ['main', 'derived']

    def should_convert_tiles(self, channel):
        return channel in self.channels_to_convert

    def n_steps_to_run(self, channels):
        assert channels == ['main', 'derived']
        return N_STEPS

    def stitch(self):
        pass

    def stop_process(self):
        pass


class FakeProgressWatcher:
    def __init__(self):
        self.n_finished = 0

    def finish(self):
        self.n_finished += 1


class FakeMainWindow:
    def __init__(self):
        self.progress_watcher = FakeProgressWatcher()
        self.status_messages = []

    def print_status_msg(self, msg):
        self.status_messages.append(msg)


class FakeStitchingTab:
    run_stitching = StitchingTab.run_stitching

    def __init__(self, stitcher, error=None):
        self.worker = stitcher
        self.main_window = FakeMainWindow()
        self.error = error
        self.steps = []
        self.n_conversions = 0
        self.n_plotable_updates = 0

    def wrap_step(self, task_name, func, **kwargs):
        self.steps.append((task_name, func, kwargs))
        if self.error:
            raise self.error

    def convert_tiles(self):
        self.n_conversions += 1

    def update_plotable_channels(self):
        self.n_plotable_updates += 1


def test_the_whole_run_is_one_dialog_sized_with_the_steps_to_run():
    stitcher = FakeStitcher()
    tab = FakeStitchingTab(stitcher)

    tab.run_stitching()

    [(task_name, func, kwargs)] = tab.steps
    assert (task_name, func, kwargs['n_steps']) == ('Stitching', stitcher.stitch, N_STEPS)
    assert kwargs['abort_func'] == stitcher.stop_process
    assert tab.main_window.progress_watcher.n_finished == 1
    assert tab.n_plotable_updates == 1


def test_the_tiles_are_converted_in_their_own_dialog_only_when_some_are_to_convert():
    tab = FakeStitchingTab(FakeStitcher())
    tab.run_stitching()
    assert tab.n_conversions == 0

    tab = FakeStitchingTab(FakeStitcher(channels_to_convert=['derived']))
    tab.run_stitching()
    assert tab.n_conversions == 1


def test_missing_requirements_are_reported_and_the_dialog_is_closed():
    tab = FakeStitchingTab(FakeStitcher(), error=MissingRequirementException('no layout'))

    tab.run_stitching()

    assert 'missing requirements' in tab.main_window.status_messages[0]
    assert tab.main_window.progress_watcher.n_finished == 1
