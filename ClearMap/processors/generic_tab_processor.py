"""
This module contains the generic processor classes that are used to define the processing steps and run the processing
This is inherited by all processors in ClearMap
"""
import os
import sys
import warnings
from concurrent.futures.process import BrokenProcessPool


class ProcessorSteps:
    def __init__(self, workspace, postfix=''):
        self.postfix = postfix
        self.workspace = workspace

    @property
    def steps(self):
        raise NotImplementedError

    def path_from_step_name(self, step_name):
        raise NotImplementedError

    @property
    def existing_steps(self):
        return [s for s in self.steps if self.step_exists(s)]

    @property
    def last_step(self):
        return self.existing_steps[-1]

    def get_next_steps(self, step_name):
        return self.steps[self.steps.index(step_name)+1:]

    def step_exists(self, step_name):
        return os.path.exists(self.path_from_step_name(step_name))

    def remove_next_steps_files(self, target_step_name):
        for step_name in self.get_next_steps(target_step_name):
            f_path = self.path_from_step_name(step_name)
            if os.path.exists(f_path):
                warnings.warn(f"WARNING: Remove previous step {step_name}, file {f_path}")
                os.remove(f_path)

    def path(self, step, step_back=False, n_before=0):
        if n_before:
            step = self.steps[self.steps.index(step) - n_before]
        f_path = self.path_from_step_name(step)
        if not os.path.exists(f_path):
            if step_back:  # FIXME: steps back only once ??
                f_path = self.path(self.steps[self.steps.index(step) - 1])
            else:
                raise IndexError(f'Could not find path "{f_path}" and not allowed to step back')
        return f_path


class TabProcessor:
    def __init__(self):
        self.stopped = False
        self.progress_watcher = None
        self.workspace = None
        self.machine_config = {}

    def set_progress_watcher(self, watcher):
        self.progress_watcher = watcher

    def update_watcher_progress(self, val):
        if self.progress_watcher is not None:
            self.progress_watcher.increment(val)

    def update_watcher_main_progress(self, val=1):
        if self.progress_watcher is not None:
            self.progress_watcher.increment_main_progress(val)

    def set_watcher_step(self, step_name):
        if self.progress_watcher is not None:
            self.progress_watcher.main_step_name = step_name

    def prepare_watcher_for_substep(self, counter_size, pattern, title, increment_main=False):
        """
        Prepare the progress watcher for the coming processing step. The watcher will in turn signal changes to the
        progress bar

        Arguments
        ---------
        counter_size: int
            The progress bar maximum
        pattern: str or re.Pattern or (str, re.Pattern)
            The string to search for in the log to signal an increment of 1
        title: str
            The title of the step for the progress bar
        increment_main: bool
            Whether a new step should be added to the main progress bar
        """
        if self.progress_watcher is not None:
            self.progress_watcher.prepare_for_substep(counter_size, pattern, title)
            if increment_main:
                self.update_watcher_main_progress()

    def stop_process(self):  # REFACTOR: put in parent class ??
        self.stopped = True
        if hasattr(self.workspace, 'executor') and self.workspace.executor is not None:
            if sys.version_info[:2] >= (3, 9):
                print('Canceling process')
                self.workspace.executor.shutdown(cancel_futures=True)  # The new clean version
            else:
                self.workspace.executor.immediate_shutdown()  # Dirty but we have no choice in python < 3.9
            self.workspace.executor = None
            # raise BrokenProcessPool
        elif hasattr(self.workspace, 'process') and self.workspace.process is not None:
            self.workspace.process.terminate()
            # self.workspace.process.wait()
            self.workspace.process = None
            raise CanceledProcessing

    @property
    def verbose(self):
        return self.machine_config['verbosity'] == 'debug'

    def run(self):
        raise NotImplementedError

    # def setup(self):
    #     pass


class CanceledProcessing(BrokenProcessPool):  # TODO: better inheritance
    pass
