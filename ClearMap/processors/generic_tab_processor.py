import sys
from concurrent.futures.process import BrokenProcessPool


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
