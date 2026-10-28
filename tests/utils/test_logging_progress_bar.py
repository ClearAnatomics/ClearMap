"""
A headless run (a Slurm job) shows its progress as lines of text: the LoggingProgressBar subscribes to the events of
the ProgressWatcher, as the progress dialog of the GUI does.
"""
import io
import re

import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.Utils.progress import LoggingProgressBar, ProgressWatcher

N_MAIN_STEPS = 2
SUBSTEP_SIZE = 100
TIME_PREFIX = re.compile(r'^\d\d:\d\d:\d\d ')


@pytest.fixture
def watcher_and_output():
    bus = EventBus()
    output = io.StringIO()
    bar = LoggingProgressBar(bus, stream=output)
    yield ProgressWatcher(bus), output, bar


def lines_of(output):
    lines = output.getvalue().splitlines()
    assert all(TIME_PREFIX.match(line) for line in lines)
    return [TIME_PREFIX.sub('', line) for line in lines]


def test_a_run_is_a_line_per_step_per_tenth_of_a_substep_and_at_the_end(watcher_and_output):
    watcher, output, _bar = watcher_and_output
    watcher.setup(main_step_name='Cell map', main_step_length=N_MAIN_STEPS)
    watcher.prepare_for_substep(SUBSTEP_SIZE, pattern=None, step_name='Cell detection')
    for _ in range(SUBSTEP_SIZE):
        watcher.increment(1)
    watcher.increment_main_progress()
    watcher.prepare_for_substep(0, pattern=None, step_name='Cell filtering')
    watcher.increment_main_progress()
    assert lines_of(output) == (['Cell map', 'Step 1 of 2', '    Cell detection (100)']
                                + [f'    Cell detection: {percent}%' for percent in range(10, 101, 10)]
                                + ['Step 2 of 2', '    Cell filtering', 'Cell map finished in 0 s'])


def test_a_single_step_run_has_no_step_lines(watcher_and_output):
    watcher, output, _bar = watcher_and_output
    watcher.setup(main_step_name='Stitching', main_step_length=1)
    watcher.finish()
    assert lines_of(output) == ['Stitching', 'Stitching finished in 0 s']


def test_a_new_substep_starts_its_tenths_again(watcher_and_output):
    watcher, output, _bar = watcher_and_output
    watcher.prepare_for_substep(10, pattern=None, step_name='Binarization')
    watcher.increment(10)
    watcher.prepare_for_substep(10, pattern=None, step_name='Filling')
    watcher.increment(3)
    assert lines_of(output)[-2:] == ['    Filling (10)', '    Filling: 30%']
