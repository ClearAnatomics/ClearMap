"""
The ProgressWatcher knows nothing of the front end: it publishes the progress on the event bus, the GUI dialog and
the log subscribe. It used to be a QWidget for its signals and its QTimer, so it could not be used without Qt.
"""
import subprocess
import sys
import threading
import time

import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.Utils.events import (ProgressMainStepNameChanged, ProgressSubStepNameChanged, ProgressMainValueChanged,
                                   ProgressMainMaximumChanged, ProgressSubValueChanged, ProgressSubMaximumChanged,
                                   ProgressFinished)
from ClearMap.Utils.progress import ProgressWatcher

PROGRESS_EVENTS = (ProgressMainStepNameChanged, ProgressSubStepNameChanged, ProgressMainValueChanged,
                   ProgressMainMaximumChanged, ProgressSubValueChanged, ProgressSubMaximumChanged, ProgressFinished)
POLL_INTERVAL_MS = 10
SECONDS_TO_WAIT_FOR_THE_POLLER = 5
MATCHING_LINE = 'Block 3/12 done\n'
OTHER_LINE = 'Reading the next block\n'


class ProgressRecorder:
    def __init__(self, bus):
        self.events = []
        self.last_sub_value = 0
        self.sub_value_reached = threading.Event()
        self.wanted_sub_value = None
        for event_type in PROGRESS_EVENTS:
            bus.subscribe(event_type, self.record)

    def record(self, event):
        self.events.append(event)
        if isinstance(event, ProgressSubValueChanged):
            self.last_sub_value = event.value
            if event.value == self.wanted_sub_value:
                self.sub_value_reached.set()

    def of_type(self, event_type):
        return [event for event in self.events if isinstance(event, event_type)]


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def recorder(bus):
    return ProgressRecorder(bus)


def test_importing_the_watcher_imports_no_qt():
    code = ('import sys, ClearMap.Utils.progress; '
            'print([name for name in sys.modules if name.split(".")[0] in ("PyQt5", "PyQt6", "PySide2", "PySide6")])')
    output = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    assert output.strip() == '[]'


def test_the_steps_and_their_progress_are_published(bus, recorder):
    watcher = ProgressWatcher(bus)
    watcher.setup(main_step_name='Cell map', main_step_length=3)
    watcher.prepare_for_substep(12, pattern=None, step_name='Detection')
    watcher.increment(4)
    watcher.increment(0.5)       # a fraction of the maximum: 6 of 12
    watcher.increment_main_progress()
    assert recorder.of_type(ProgressMainStepNameChanged)[-1] == ProgressMainStepNameChanged('Cell map')
    assert recorder.of_type(ProgressMainMaximumChanged)[-1] == ProgressMainMaximumChanged(3)
    assert recorder.of_type(ProgressSubStepNameChanged)[-1] == ProgressSubStepNameChanged('Detection')
    assert recorder.of_type(ProgressSubMaximumChanged)[-1] == ProgressSubMaximumChanged(12)
    assert [event.value for event in recorder.of_type(ProgressSubValueChanged)][-2:] == [4, 10]
    assert recorder.of_type(ProgressMainValueChanged)[-1] == ProgressMainValueChanged(2)
    assert not recorder.of_type(ProgressFinished)


def test_passing_the_last_main_step_publishes_the_end(bus, recorder):
    watcher = ProgressWatcher(bus)
    watcher.setup(main_step_name='Cell map', main_step_length=2)
    watcher.increment_main_progress()
    assert not recorder.of_type(ProgressFinished)
    watcher.increment_main_progress()
    assert recorder.of_type(ProgressFinished) == [ProgressFinished('Cell map')]


def test_finish_publishes_the_end(bus, recorder):
    watcher = ProgressWatcher(bus)
    watcher.main_step_name = 'Stitching'
    watcher.finish()
    assert recorder.of_type(ProgressFinished) == [ProgressFinished('Stitching')]


@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / 'clearmap.log'
    path.write_text(MATCHING_LINE)      # written before the substep: not counted
    return path


def test_the_progress_of_a_substep_is_counted_from_the_log_by_a_thread(bus, recorder, log_file):
    watcher = ProgressWatcher(bus, timer_interval_ms=POLL_INTERVAL_MS)
    watcher.log_path = str(log_file)
    watcher.prepare_for_substep(3, pattern='done', step_name='Detection')
    recorder.wanted_sub_value = 3
    watcher.start_polling()
    try:
        with open(log_file, 'a') as log:
            log.write(OTHER_LINE + MATCHING_LINE * 3 + OTHER_LINE)
        assert recorder.sub_value_reached.wait(SECONDS_TO_WAIT_FOR_THE_POLLER)
    finally:
        watcher.stop_polling()
    assert recorder.last_sub_value == 3


def test_stopping_the_polling_stops_its_thread(bus, log_file):
    watcher = ProgressWatcher(bus, timer_interval_ms=POLL_INTERVAL_MS)
    watcher.log_path = str(log_file)
    watcher.start_polling()
    poller = watcher._poller
    assert poller.is_alive()
    watcher.stop_polling()
    assert not poller.is_alive()
    watcher.stop_polling()      # twice: nothing to stop


def test_polling_again_after_a_stop_counts_again(bus, recorder, log_file):
    """The step after the first one is polled too (run_with_responsive_ui starts and stops the polling at each step)"""
    watcher = ProgressWatcher(bus, timer_interval_ms=POLL_INTERVAL_MS)
    watcher.log_path = str(log_file)
    watcher.start_polling()
    watcher.stop_polling()
    watcher.prepare_for_substep(2, pattern='done', step_name='Measurement')
    recorder.wanted_sub_value = 2
    watcher.start_polling()
    try:
        with open(log_file, 'a') as log:
            log.write(MATCHING_LINE * 2)
        assert recorder.sub_value_reached.wait(SECONDS_TO_WAIT_FOR_THE_POLLER)
    finally:
        watcher.stop_polling()


def test_starting_twice_leaves_one_poller(bus, log_file):
    watcher = ProgressWatcher(bus, timer_interval_ms=POLL_INTERVAL_MS)
    watcher.log_path = str(log_file)
    watcher.start_polling()
    first_poller = watcher._poller
    watcher.start_polling()
    try:
        assert not first_poller.is_alive()
    finally:
        watcher.stop_polling()


def test_nothing_is_polled_without_a_log(bus):
    watcher = ProgressWatcher(bus, timer_interval_ms=POLL_INTERVAL_MS)
    watcher.start_polling()
    assert watcher._poller is None
    watcher.stop_polling()


def test_a_new_substep_counts_from_the_end_of_the_log(bus, recorder, log_file):
    watcher = ProgressWatcher(bus, timer_interval_ms=POLL_INTERVAL_MS)
    watcher.log_path = str(log_file)
    watcher.prepare_for_substep(5, pattern='done', step_name='Detection')
    with open(log_file, 'a') as log:
        log.write(MATCHING_LINE * 2)
    assert watcher.count_dones() == 2
    watcher.prepare_for_substep(5, pattern='done', step_name='Measurement')
    time.sleep(POLL_INTERVAL_MS / 1000)
    assert watcher.count_dones() == 0
