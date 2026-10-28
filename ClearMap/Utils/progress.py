# -*- coding: utf-8 -*-
"""
progress
========

The progress of the processing, independent of the front end.

:class:`ProgressWatcher` publishes the progress on the event bus (the ``Progress...`` events of
:mod:`ClearMap.Utils.events`), and the displays subscribe to them: the progress dialog of the GUI, the log of a
headless run (:class:`LoggingProgressBar`). A watcher whose bus nobody listens to shows nothing.

.. warning:: Temporary workaround

    The progress inside a substep is not reported by the code that does the work: it is counted from the log file,
    as the lines that match a pattern, by a thread every ``timer_interval_ms``. When this was written, the modules
    only printed (thousands of prints, in code that could not be changed). It is to be replaced by progress reported
    by the code itself (e.g. the finished blocks, counted by the parent process).
"""
import re
import sys
import threading
import time

from ClearMap.Utils.event_bus import EventBus
from ClearMap.Utils.events import (ProgressMainStepNameChanged, ProgressSubStepNameChanged, ProgressMainValueChanged,
                                   ProgressMainMaximumChanged, ProgressSubValueChanged, ProgressSubMaximumChanged,
                                   ProgressFinished)

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ClearAnatomics/ClearMap'

DEFAULT_MAIN_STEP_NAME = 'Processing'
DEFAULT_MAX_PROGRESS = 100
DEFAULT_POLL_INTERVAL_MS = 250
PRINTED_FRACTION_STEP = 0.1          # The LoggingProgressBar writes a line every 10 % of a substep
LINE_TIME_FORMAT = '%H:%M:%S'


class ProgressWatcherBase:
    """
    What any code may call on a progress watcher, and the watcher that shows nothing: each method does nothing and
    each property gives its default (its setter ignores the value).

    The code that reports its progress is given one when no display was set (e.g. an orchestrator in a headless
    run), so it never tests whether it has a watcher. :class:`ProgressWatcher` is the one that publishes; each of
    its public members must be defined here too (tests/utils/test_progress_watcher_base.py checks it).
    """
    log_path = None
    pattern = None

    def reset(self):
        pass

    def setup(self, main_step_name, main_step_length, sub_step_length=0, pattern=None):
        pass

    def set_poll_interval(self, ms: int):
        pass

    def start_polling(self):
        pass

    def stop_polling(self):
        pass

    def prepare_for_substep(self, step_length, pattern, step_name):
        pass

    def get_progress(self):
        return 0

    def set_progress(self, value):
        pass

    def set_main_progress(self, value):
        pass

    def increment_main_progress(self, increment=1):
        pass

    def increment(self, increment):
        pass

    @property
    def max_progress(self):
        return DEFAULT_MAX_PROGRESS

    @max_progress.setter
    def max_progress(self, value):
        pass

    @property
    def main_max_progress(self):
        return 1

    @main_max_progress.setter
    def main_max_progress(self, value):
        pass

    @property
    def main_step_name(self):
        return DEFAULT_MAIN_STEP_NAME

    @main_step_name.setter
    def main_step_name(self, step_name):
        pass

    @property
    def sub_step_name(self):
        return None

    @sub_step_name.setter
    def sub_step_name(self, step_name):
        pass

    def count_dones(self):
        return 0

    def reset_log_length(self):
        pass

    def finish(self):
        pass


class ProgressWatcher(ProgressWatcherBase):
    """
    Watches the progress of a process and publishes it on the event bus.

    The main setup methods are :meth:`setup` and :meth:`prepare_for_substep`. The progress has two levels: the main
    steps and, inside the current one, its substep (whose progress may be counted from the log, see the warning of
    the module).
    """
    def __init__(self, bus: EventBus, max_progress=DEFAULT_MAX_PROGRESS, main_max_progress=1,
                 timer_interval_ms=DEFAULT_POLL_INTERVAL_MS):
        """
        Create a ProgressWatcher.

        Parameters
        ----------
        bus : EventBus
            Where the progress is published.
        max_progress : int
            The maximum progress value; when the progress reaches this value,
            the (sub-)operation is considered finished. Default is 100.
        main_max_progress : int
            The maximum progress value for the main operation.
            When the progress reaches this value, the main operation
            is considered finished. If all sub-operations are also finished, this is usually linked to the end of the
            whole process.
            Default is 1.
        timer_interval_ms : int
            Polling interval in milliseconds for log-based progress. Default is 250.
        """
        self.bus = bus
        self._main_step_name = DEFAULT_MAIN_STEP_NAME
        self._sub_step_name = None
        self._main_progress = 1
        self._main_max_progress = main_max_progress
        self._progress = 0
        self._max_progress = max_progress

        self._n_dones = 0
        self._previous_log_length = 0  # The log length at the end of the previous operation
        self.log_path = None
        self.pattern = None

        self._interval_ms = timer_interval_ms
        self._log_lock = threading.RLock()  # The poller reads the counters, the processing thread resets them
        self._polling_stop_requested = threading.Event()
        self._poller = None

    def reset(self):
        """Reset all the values to their initial state."""
        self.main_step_name = DEFAULT_MAIN_STEP_NAME
        self._main_progress = 1
        self._main_max_progress = 1
        self._progress = 0
        self._max_progress = DEFAULT_MAX_PROGRESS

        self._n_dones = 0
        self._previous_log_length = 0  # The log length at the end of the previous operation
        self.log_path = None
        self.pattern = None

    def setup(self, main_step_name, main_step_length, sub_step_length=0, pattern=None):
        """
        Post-initialisation setup.

        Parameters
        ----------
        main_step_name : str
            Title of the main processing step.
        main_step_length : int
            Total number of main steps.
        sub_step_length : int
            Total number of sub-steps within the current main step.
        pattern : str or re.Pattern or tuple or None
            Text pattern to search for in logs to detect progress increments.
        """
        self.main_step_name = main_step_name
        self.main_max_progress = main_step_length
        self.max_progress = sub_step_length
        self.pattern = pattern

        self.reset_log_length()
        self.set_main_progress(1)
        self.set_progress(0)
        # Force update
        self.bus.publish(ProgressMainValueChanged(self._main_progress))
        self.bus.publish(ProgressSubValueChanged(self._progress))

    def set_poll_interval(self, ms: int):
        """Set the log-polling interval in milliseconds (from the next poll on)."""
        self._interval_ms = ms

    def start_polling(self):
        """Start polling the log file for progress pattern matches, in a thread of its own."""
        if not self.log_path:  # If not setup
            return
        self.stop_polling()  # At most one poller
        self.reset_log_length()
        self._polling_stop_requested.clear()
        self._poller = threading.Thread(target=self._poll_until_stop_requested, name='progress log poller',
                                        daemon=True)
        self._poller.start()

    def stop_polling(self):
        """Stop log-file polling (returns once the poller has stopped)."""
        self._polling_stop_requested.set()
        if self._poller is not None:
            self._poller.join()
            self._poller = None

    def _poll_until_stop_requested(self):
        while not self._polling_stop_requested.wait(self._interval_ms / 1000):  # True once the stop is requested
            self._on_tick()

    def _on_tick(self):
        with self._log_lock:
            if not self.pattern or not self.log_path:
                return
            dones = self.count_dones()
            self.set_progress(dones)

    def prepare_for_substep(self, step_length, pattern, step_name):
        """
        Setup the watcher for a new substep.

        Parameters
        ----------
        step_length : int
            The number of steps in the operation.
        pattern : str or re.Pattern or tuple or None
            The text to look for in the logs to check for progress.
        step_name : str
            Name (title) of the substep.
        """
        with self._log_lock:
            self.max_progress = step_length
            self.pattern = pattern
            self.reset_log_length()
            self.set_progress(0)
        self.sub_step_name = step_name

    def get_progress(self):
        """
        Get the current progress.

        Returns
        -------
        int
        """
        return self._progress

    def set_progress(self, value):
        """
        Set the progress value of the current main or sub step.

        Parameters
        ----------
        value : int
            The progress value.
        """
        if self._progress == value:
            return
        self._progress = round(value)
        self.bus.publish(ProgressSubValueChanged(self._progress))

    def set_main_progress(self, value):
        """
        Set the progress value for the main step.

        Parameters
        ----------
        value : int
            The progress value.
        """
        if self._main_progress == value:
            return
        self._main_progress = round(value)
        self.reset_log_length()
        self.bus.publish(ProgressMainValueChanged(self._main_progress))
        if self._main_progress != 0 and self._main_progress == self.main_max_progress + 1:
            self.bus.publish(ProgressFinished(self.main_step_name))

    def increment_main_progress(self, increment=1):
        """
        Integer increment of the main progress.

        Parameters
        ----------
        increment : int
            The increment value (default is 1).
        """
        self.set_main_progress(self._main_progress + round(increment))

    def increment(self, increment):
        """
        Increment the progress value of the current main or sub step.

        Parameters
        ----------
        increment : int or float
            The increment value. If float, it is considered as a fraction
            of the maximum progress value.
        """
        if isinstance(increment, float):
            self.set_progress(self._progress + int(self.max_progress * increment))
        elif isinstance(increment, int):
            self.set_progress(self._progress + increment)

    @property
    def max_progress(self):
        """Current sub-step maximum."""
        return self._max_progress

    @max_progress.setter
    def max_progress(self, value):
        self._max_progress = round(value)
        self.bus.publish(ProgressSubMaximumChanged(self._max_progress))

    @property
    def main_max_progress(self):
        """Current main-step maximum."""
        return self._main_max_progress

    @main_max_progress.setter
    def main_max_progress(self, value):
        self._main_max_progress = round(value)
        self.bus.publish(ProgressMainMaximumChanged(self._main_max_progress))

    @property
    def main_step_name(self):
        """Current main step name."""
        return self._main_step_name

    @main_step_name.setter
    def main_step_name(self, step_name):
        self._main_step_name = step_name
        self.bus.publish(ProgressMainStepNameChanged(self._main_step_name))

    @property
    def sub_step_name(self):
        """Current sub-step name."""
        return self._sub_step_name

    @sub_step_name.setter
    def sub_step_name(self, step_name):
        self._sub_step_name = step_name
        self.bus.publish(ProgressSubStepNameChanged(self._sub_step_name))

    def _matches_pattern(self, line):
        if isinstance(self.pattern, tuple):  # TODO: cache
            return self.pattern[0] in line and self.pattern[1].match(line)  # Most efficient
        elif isinstance(self.pattern, str):
            return self.pattern in line
        elif isinstance(self.pattern, re.Pattern):
            return self.pattern.match(line)

    def count_dones(self):
        """
        Parse the logs to extract the number of completed operations
        (based on ``self.pattern``). For each match, the progress is
        incremented by 1. For efficiency, the logs are read from the
        last read position.

        Returns
        -------
        int
            The cumulative number of matched operations.
        """
        with self._log_lock:
            if self.pattern is None:
                return 0
            try:
                with open(self.log_path, 'rb') as log:      # binary: seek is byte-exact
                    log.seek(self._previous_log_length)
                    raw = log.read()
                    self._previous_log_length = log.tell()   # update to real end position
            except (FileNotFoundError, OSError):
                return self._n_dones

            text = raw.decode('utf-8', errors='replace')    # 0x9C → �, no crash
            new_lines = text.splitlines(keepends=True)

            n_dones = len([ln for ln in new_lines if self._matches_pattern(ln)])
            self._n_dones += n_dones
            return self._n_dones

    def reset_log_length(self):
        """Reset the done counter and seek to the end of the log file."""
        with self._log_lock:
            self._n_dones = 0
            if not self.log_path:
                self._previous_log_length = 0
                return
            try:
                with open(self.log_path, 'rb') as log:
                    log.seek(0, 2)                          # SEEK_END
                    self._previous_log_length = log.tell()   # real byte offset
            except FileNotFoundError:
                self._previous_log_length = 0

    def finish(self):
        """Publish that the processing is finished."""
        self.bus.publish(ProgressFinished(self.main_step_name))


class LoggingProgressBar:
    """
    The progress as lines of text, for a headless run (e.g. the output file of a Slurm job): the steps, each
    substep with its size, a line every 10 % of a substep, and the end with its duration.

    Subscribed without affinity: the handlers run in the thread that publishes (the processing thread, the log
    poller), so they share a lock.
    """
    def __init__(self, bus: EventBus, stream=None):
        """
        Parameters
        ----------
        bus : EventBus
            The bus the ProgressWatcher publishes on.
        stream : file-like or None
            Where the lines are written; standard output by default.
        """
        self.stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()
        self._main_maximum = 1
        self._printed_main_value = None
        self._sub_step_name = None
        self._sub_maximum = 0
        self._printed_fraction_steps = 0
        self._start_time = time.monotonic()
        for event_type, handler in ((ProgressMainStepNameChanged, self._on_main_step_name_changed),
                                    (ProgressMainMaximumChanged, self._on_main_maximum_changed),
                                    (ProgressMainValueChanged, self._on_main_value_changed),
                                    (ProgressSubStepNameChanged, self._on_sub_step_name_changed),
                                    (ProgressSubMaximumChanged, self._on_sub_maximum_changed),
                                    (ProgressSubValueChanged, self._on_sub_value_changed),
                                    (ProgressFinished, self._on_finished)):
            bus.subscribe(event_type, handler)

    def _write(self, line):
        print(f'{time.strftime(LINE_TIME_FORMAT)} {line}', file=self.stream, flush=True)

    def _on_main_step_name_changed(self, event: ProgressMainStepNameChanged):
        with self._lock:
            self._printed_main_value = None
            self._start_time = time.monotonic()
            if event.name:
                self._write(event.name)

    def _on_main_maximum_changed(self, event: ProgressMainMaximumChanged):
        with self._lock:
            self._main_maximum = event.maximum

    def _on_main_value_changed(self, event: ProgressMainValueChanged):
        with self._lock:
            is_a_step = 1 <= event.value <= self._main_maximum   # main maximum + 1 is the end
            if self._main_maximum > 1 and is_a_step and event.value != self._printed_main_value:
                self._printed_main_value = event.value
                self._write(f'Step {event.value} of {self._main_maximum}')

    def _on_sub_step_name_changed(self, event: ProgressSubStepNameChanged):
        with self._lock:
            self._sub_step_name = event.name
            self._printed_fraction_steps = 0
            if event.name:
                self._write(f'    {event.name}' + (f' ({self._sub_maximum})' if self._sub_maximum else ''))

    def _on_sub_maximum_changed(self, event: ProgressSubMaximumChanged):
        with self._lock:
            self._sub_maximum = event.maximum

    def _on_sub_value_changed(self, event: ProgressSubValueChanged):
        with self._lock:
            if self._sub_maximum <= 0:
                return
            fraction = min(event.value / self._sub_maximum, 1)
            fraction_steps = int(round(fraction / PRINTED_FRACTION_STEP, 6))   # 0.3 / 0.1 is 2.9999999999999996
            if fraction_steps > self._printed_fraction_steps:
                self._printed_fraction_steps = fraction_steps
                self._write(f'    {self._sub_step_name or ""}: {fraction:.0%}')

    def _on_finished(self, event: ProgressFinished):
        with self._lock:
            seconds = time.monotonic() - self._start_time
            self._write(f'{event.main_step_name or "Processing"} finished in {seconds:.0f} s')
