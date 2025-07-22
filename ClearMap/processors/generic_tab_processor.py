"""
This module contains the generic processor classes that are used to define the processing steps and run the processing
This is inherited by all processors in ClearMap
"""
import os
import sys
import warnings
from concurrent.futures.process import BrokenProcessPool

from ClearMap.Utils.utilities import handle_deprecated_args


class ProcessorSteps:
    def __init__(self, workspace, channel='', sub_step=''):
        self.channel = channel
        self.sub_step = sub_step
        self.workspace = workspace

    @property
    def steps(self):
        raise NotImplementedError

    def asset_from_step_name(self, step_name):
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
        return self.asset_from_step_name(step_name).exists

    def remove_next_steps_files(self, target_step_name):
        for step_name in self.get_next_steps(target_step_name):
            asset = self.asset_from_step_name(step_name)
            if asset.exists:
                warnings.warn(f"WARNING: Remove previous step {step_name}, file {asset.path}")
                asset.path.unlink(missing_ok=True)

    def get_asset(self, step, step_back=False, n_before=0):
        """
        Get the asset corresponding to the step name, optionally picking the nth previous step if `n_before` is set.
        If the asset does not exist, it will try to get the previous step if `step_back` is True.

        Parameters
        ----------
        step: str
            Name of the step to get the asset for.
        step_back: bool
            If True, will try to get the previous step's asset if the current step's asset does not exist.
        n_before: int
            If set, will return the asset of the nth previous step instead of the current step.
            Useful when you want to get the asset source to the current step for example.

        Returns
        -------
        Asset
            The asset corresponding to the step name.
        """
        if n_before:
            step = self.steps[self.steps.index(step) - n_before]
        asset = self.asset_from_step_name(step)
        if not asset.exists:
            if step_back:  # FIXME: steps back only once ??
                asset = self.get_asset(self.steps[self.steps.index(step) - 1])
            else:
                raise IndexError(f'Could not find path "{asset}" and not allowed to step back')
        return asset


class TabProcessor:
    def __init__(self):
        self.stopped = False
        self.progress_watcher = None
        self.workspace = None
        self.machine_config = {}

    @handle_deprecated_args(
        {'postfix': 'asset_sub_type',
         'prefix': 'sample_id'}
    )
    def get(self, asset_type, channel='current',
            asset_sub_type=None, **kwargs):   # channel and asset_sub_type defined for completion
        asset = self.workspace.get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs)
        return asset

    def get_path(self, asset_type, channel='current',
            asset_sub_type=None, **kwargs):   # channel and asset_sub_type defined for completion
        return self.get(asset_type, channel=channel, asset_sub_type=asset_sub_type, **kwargs).path

    def filename(self, *args, **kwargs):  # WARNING: deprecated
        """
        A shortcut to get the filename from the workspace

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        str
            The filename
        """
        return self.workspace.filename(*args, **kwargs)

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
        if executor := getattr(self.workspace, 'executor', None):
            if sys.version_info[:2] >= (3, 9):
                print('Canceling process')
                executor.shutdown(cancel_futures=True)  # The new clean version
            else:
                executor.immediate_shutdown()  # Dirty but we have no choice in python < 3.9
            self.workspace.executor = None
            # raise BrokenProcessPool
        elif process := getattr(self.workspace, 'process', None):
            process.terminate()
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


class ChannelTabProcessor(TabProcessor):
    """
    Tab processor that is processing a single channel.

    The config is expected to have a 'channels' section with the channel name as key.
    The config is stored in the `_processing_config` attribute, and accessed through the `processing_config` property
    which returns the section for the current channel.
    """
    def __init__(self):
        super().__init__()
        self._processing_config = None
        self.channel = ''

    @property
    def processing_config(self):
        return self._processing_config['channels'][self.channel]

    @processing_config.setter
    def processing_config(self, value):
        raise ValueError('Processing config is a property and cannot be set directly. '
                         'you should set the _processing_config attribute instead.')

    def reload_config(self):
        self._processing_config.reload()


class CanceledProcessing(BrokenProcessPool):  # TODO: better inheritance
    pass
