#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
TubeMap
=======

This module contains the classes to generate annotated graphs from vasculature
lightsheet data [Kirst2020]_.
"""
from __future__ import annotations

import os
import copy
import re
import functools
import platform
import time
import warnings
import gc
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QDialogButtonBox

import ClearMap.IO.IO as clearmap_io
from ClearMap.IO.Source import Source
from ClearMap.IO.workspace2 import Workspace2
from ClearMap.IO.workspace_asset import Asset

from ClearMap.config.config_coordinator import ConfigCoordinator

from ClearMap.pipeline_orchestrators.generic_orchestrators import PipelineOrchestrator, ProcessorSteps

import ClearMap.Alignment.Resampling as resampling_module
import ClearMap.Alignment.Elastix as elastix

from ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing import initialize_sink
import ClearMap.ParallelProcessing.BlockProcessing as block_processing

import ClearMap.ImageProcessing.Experts.Vasculature as vasculature
import ClearMap.ImageProcessing.machine_learning.vessel_filling.vessel_filling as vessel_filling
import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skeletonization
import ClearMap.ImageProcessing.Binary.Filling as binary_filling

import ClearMap.Analysis.Measurements.MeasureExpression as measure_expression
import ClearMap.Analysis.Measurements.radius_measurements as measure_radius
import ClearMap.Analysis.Measurements.Voxelization as voxelization

from ClearMap.Analysis.graphs import graph_processing
from ClearMap.Analysis.graphs.graph_filters import GraphFilter

from ClearMap.gui.dialog_helpers import warning_popup
from ClearMap.Utils.utilities import is_in_range, get_free_v_ram, clear_cuda_cache, sanitize_n_processes
from ClearMap.Utils.exceptions import (PlotGraphError, ClearMapVRamException,
                                       MissingRequirementException, MissingAssetError, AssetNotFoundError,
                                       ClearMapAssetError)

from .sample_info_management import SampleManager
from .registration_orchestrator import RegistrationProcessor

__author__ = ('Christoph Kirst <christoph.kirst.ck@gmail.com>,'
              ' Sophie Skriabine <sophie.skriabine@icm-institute.org>,'
              ' Charly Rousseau <charly.rousseau@icm-institute.org>')
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

MAX_PLOT_VERTICES = 300_000  # Empirical max number of vertices that can safely be plotted

USE_BINARY_POINTS_FILE = not platform.system().lower().startswith('darwin')
_SourcePath = Union[Path, str]


class VesselGraphProcessorSteps(ProcessorSteps):
    graph_raw       = 'raw'
    graph_cleaned   = 'cleaned'
    graph_reduced   = 'reduced'
    graph_annotated = 'annotated'

    _default_steps = (graph_raw, graph_cleaned, graph_reduced, graph_annotated)  # TODO: add traced

    def asset_from_step_name(self, step):
        return self.workspace.get('graph', channel=self.channel, asset_sub_type=step)


class BinaryVesselProcessorSteps(ProcessorSteps):
    stitched    = 'stitched'
    binary      = 'binary'
    smoothed    = 'smoothed'
    deep_filled = 'deep_filled'
    filled      = 'filled'
    combined    = 'combined'
    final       = 'final'

    _default_steps = (stitched, binary, smoothed, deep_filled, filled, combined, final)

    # Fixed anchors that cannot be reordered
    _prefix_steps = (stitched, binary)
    _suffix_steps = (combined, final)

    _GUI_STEP_TO_ASSET: dict[str, str] = {
        'binarize': 'binary',
        'smooth': 'smoothed',
        'binary_fill': 'filled',
        'deep_fill': 'deep_filled',
    }

    _BINARIZE_STEP_MAP: dict[str, tuple[str, bool]] = {
        'binarize':    ('binarize_channel',  False),
        'smooth':      ('smooth_channel',    False),
        'binary_fill': ('fill_channel',      True),   # prange — must be main thread
        'deep_fill':   ('deep_fill_channel', False),
    }

    _lifecycle_steps: frozenset[str] = frozenset({stitched, combined, final})

    def __init__(self, workspace, channel, config_provider: Callable[[], dict] | None = None):
        self._config_provider = config_provider or (lambda: {})
        super().__init__(workspace, channel)
        self._outputs: Dict[str, _SourcePath] = {}
        self._pending_cleanup: list[str] = []  # canonical paths to delete

    @classmethod
    def _default_middle_steps(cls) -> tuple[str, ...]:
        """Steps between prefix and suffix, derived from _default_steps."""
        anchors = set(cls._prefix_steps) | set(cls._suffix_steps)
        return tuple(s for s in cls._default_steps if s not in anchors)

    def asset_from_step_name(self, step):  # FIXME: split step and substep
        if step in (self.stitched, self.binary):
            asset = self.workspace.get(step, channel=self.channel)
        else:
            asset = self.workspace.get('binary', channel=self.channel, asset_sub_type=step)
        return asset

    @property
    def steps(self) -> tuple[str, ...]:
        """
        Compute step order fresh on each access — reflects any GUI
        reordering committed to config since construction.
        """
        cfg_order = self._config_provider().get('step_order')

        prefix = list(self._prefix_steps)
        suffix = list(self._suffix_steps)
        anchor_assets = set(self._prefix_steps) | set(self._suffix_steps)

        if cfg_order:
            middle = [self._GUI_STEP_TO_ASSET[name]
                      for name in cfg_order
                      if name in self._GUI_STEP_TO_ASSET
                      and self._GUI_STEP_TO_ASSET[name] not in anchor_assets]
        else:
            middle = list(self._default_middle_steps())

        return tuple(prefix + middle + suffix)


    #   #################################  output tracking ####################################

    @staticmethod
    def _extract_backing_path(output: Any) -> _SourcePath:
        """
        Extract a plain path from whatever a step produces.
        Asset → .path, Source → .location, Path/str → as-is.
        Never stores a live Source or array — clearmap_io.as_source()
        reconstructs on demand.
        """
        if isinstance(output, Asset):
            return output.path
        if isinstance(output, Source):  # covers MMP.Source, npy.Source …
            return output.location
        if isinstance(output, (Path, str)):
            return output
        return output  # unexpected — caller will fail loudly

    def record_output(self, asset: str, output: Any,
                      temp_path: str = '', keep: bool = False) -> None:
        """
        Store the backing path of a completed step.

        Source objects are serialisable path handles — we extract .path so
        _outputs holds only plain Paths, never live Source instances.
        clearmap_io.as_source(path) reconstructs on demand.
        """
        output_path = self._extract_backing_path(output)
        self._outputs[asset] = output_path
        if output_path and not keep:
            self._pending_cleanup.append(str(output_path))

    def consume_and_cleanup(self) -> None:
        """Delete the pending temp file once the next step has consumed its input."""
        for to_clean in self._pending_cleanup:
            try:
                clearmap_io.delete_file(to_clean)
            except Exception:
                pass
        self._pending_cleanup = []

    def get_source(self, current_asset: str) -> _SourcePath:
        """
        Walk backward from current_asset through the configured (non-lifecycle)
        step order and return the first available backing path.

        Precedence per previous step:
          1. In-memory result stored by record_output  (is-not-None guard)
          2. On-disk asset
          3. Raw binary (fallback)

        Callers reconstruct the Source with clearmap_io.as_source(result).
        """
        order = [s for s in self.steps if s not in self._lifecycle_steps]

        try:
            current_idx = order.index(current_asset)
        except ValueError:
            return self.asset_from_step_name(self.binary).path

        for prev_asset in reversed(order[:current_idx]):
            stored = self._outputs.get(prev_asset)
            if stored is not None:
                return stored
            try:
                asset = self.get_asset(prev_asset)
                if asset.exists:
                    return asset.path
            except (KeyError, AttributeError, FileNotFoundError,
                    AssetNotFoundError, ClearMapAssetError):  # For skipped steps
                continue

        return self.asset_from_step_name(self.binary).path

    def get_last_output(self) -> _SourcePath:
        """
        Return the last available output path, used by combine_binary.
        """
        order = [s for s in self.steps if s not in self._lifecycle_steps]

        for step in reversed(order):
            stored = self._outputs.get(step)
            if stored is not None:
                return stored
            try:
                asset = self.get_asset(step)
                if asset.exists:
                    return asset.path
            except (KeyError, AttributeError, FileNotFoundError):
                continue

        raise FileNotFoundError(f'No binary output found for channel "{self.channel}"')


class BinaryVesselProcessor(PipelineOrchestrator):
    config_name = 'vasculature'

    def __init__(self, sample_manager: Optional[SampleManager] = None,
                 config_coordinator: Optional[ConfigCoordinator] = None):
        super().__init__(config_coordinator)
        self.sample_manager: Optional[SampleManager] = None
        self.workspace: Optional[Workspace2] = None

        self.inputs_match = False
        self.inputs_shapes: tuple = (None, None)

        self.all_vessels_channel: str = ''
        self.arteries_channel: str = ''
        # TODO: add veins too
        self.steps: Dict[str, BinaryVesselProcessorSteps] = {}
        self.block_re = ('Processing block',
                         re.compile(r'.*?Processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))
        self.vessel_filling_re = ('Vessel filling',
                                  re.compile(r'.*?Vessel filling: processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))

        self.setup(sample_manager)

    def setup(self, sample_manager=None):
        self.sample_manager = sample_manager if sample_manager is not None else self.sample_manager
        if self.sample_manager is not None and self.sample_manager.setup_complete:
            self.workspace = self.sample_manager.workspace

            self.all_vessels_channel = self.sample_manager.get_channels_by_type(channel_type='vessels')
            if not self.all_vessels_channel:
                warnings.warn('Vessels channel not set')
                return

            # noinspection PyTypeChecker
            self.arteries_channel = self.sample_manager.get_channels_by_type(channel_type='arteries',
                                                                             multiple_found_action='warn')

            self.assert_input_shapes_match()

            all_channels = self.sample_manager.channels
            obsolete_channels = [k for k in self.steps if k not in all_channels]
            for k in obsolete_channels:
                del self.steps[k]

            for channel_name in self.channels_to_binarize():
                if channel_name:
                    self.steps[channel_name] = BinaryVesselProcessorSteps(
                        self.workspace, channel=channel_name,
                        config_provider=lambda ch=channel_name: (
                            self.config.get('binarization', {}).get('single_channels', {}).get(ch, {})))

            compound_channel = tuple(self.channels_to_binarize())  # FIXME: old keys not cleared
            sample_id = self.sample_manager.prefix
            self.workspace.ensure_pipeline('TubeMap', compound_channel, sample_id=sample_id,
                                           channel_content_type='compound', create_channel=True)

    # ############################### INPUTS ###############################

    def assert_input_shapes_match(self):
        """
        Ensure that the input shapes (stitched images of the different color channels) match
        before starting the binarization process since they must overlap.

        .. warning::
            stitched may not exist when processor is created.
            if not, check again in the run/binarize method
        """
        try:
            assets_to_binarize = self.assets_to_binarize()
            shapes = [asset.shape() for asset in assets_to_binarize]
        except FileNotFoundError:
            warnings.warn('Stitched images not found. Cannot check shapes.')
            self.inputs_shapes = None, None
            return
        if not all([s == shapes[0] for s in shapes]):
            self.inputs_match = False
            self.inputs_shapes = shapes
            raise ValueError(f'Channels to binarize have different shapes. This is not supported yet.'
                             f'Got shapes: {shapes}')
        self.inputs_match = True  # WARNING: may need to be reset when changing channels to binarize
        self.inputs_shapes = shapes

    def channels_to_binarize(self):
        return self.sample_manager.get_channels_by_pipeline('TubeMap', as_list=True)

    def assets_to_binarize(self) -> list[Any]:
        channels_to_binarize = self.channels_to_binarize()
        assets_to_binarize = [self.workspace.get('stitched', channel=c) for c in channels_to_binarize]
        return assets_to_binarize

    # ############################# PUBLIC API ######################################

    def run(self):
        self.binarize()
        for channel in self.channels_to_binarize():
            self.postprocess(channel)
        self.combine_binary()

    def binarize(self):
        if not self.inputs_match:
            self.assert_input_shapes_match()
        if self.inputs_match:
            for channel in self.channels_to_binarize():
                self.binarize_channel(channel)
        else:
            raise ValueError('Channels to binarize have different shapes. This is not supported yet.')

    def postprocess(self, channel: str) -> None:
        """
        Run post-processing steps (smooth, fill, deep_fill) for *channel* in
        the order declared by step_order in the config.

        Binarize is always the first step and is excluded here — it runs via
        binarize() / binarize_channel() before postprocess is called.
        """
        dispatch: Dict[str, Callable] = {
            BinaryVesselProcessorSteps.smoothed: self.smooth_channel,
            BinaryVesselProcessorSteps.filled: self.fill_channel,
            BinaryVesselProcessorSteps.deep_filled: self.deep_fill_channel,
        }
        for step in self.steps[channel].steps:
            if step in BinaryVesselProcessorSteps._lifecycle_steps or step == BinaryVesselProcessorSteps.binary:
                continue
            fn = dispatch.get(step)
            if fn is not None:
                fn(channel)

    def __get_n_blocks(self, channel):
        # TODO: use actual processing params to get real n blocks
        dim_size = self.get('stitched', channel=channel).shape()[2]
        blk_size = vasculature.default_binarization_processing_parameter['size_max']
        overlap = vasculature.default_binarization_processing_parameter['overlap']
        n_blocks = int(np.ceil((dim_size - blk_size) / (blk_size - overlap) + 1))
        return n_blocks

    @staticmethod
    def setup_channel_operation(operation):
        @functools.wraps(operation)
        def wrapper(self, channel, *args, **kwargs):
            operation_type = operation.__name__.replace('_channel', '')
            if operation_type == 'deep_fill':
                n_blocks = 1200
            else:
                n_blocks = self.__get_n_blocks(channel)

            asset_to_gui = {v: k for k, v in BinaryVesselProcessorSteps._GUI_STEP_TO_ASSET.items()}
            gui_order = [asset_to_gui[stp] for stp in self.steps[channel].steps
                         if stp not in BinaryVesselProcessorSteps._lifecycle_steps]
                         # and stp in asset_to_gui]  # TODO: check guard against unknown steps
            first_op = gui_order[0] if gui_order else operation_type
            first_step = channel == self.channels_to_binarize()[0] and operation_type == first_op
            increment_main = not first_step
            self.prepare_watcher_for_substep(n_blocks, self.block_re,
                                             f'{operation_type} {channel.title()}', increment_main)

            return operation(self, channel, *args, **kwargs)

        return wrapper

    @setup_channel_operation
    def binarize_channel(self, channel):
        self._binarize(channel)  # TODO: update watcher

    @setup_channel_operation
    def smooth_channel(self, channel):
        self._smooth(channel)  # TODO: update watcher

    @setup_channel_operation
    def fill_channel(self, channel):  # WARNING: should run from main thread
        self._fill(channel)  # TODO: update watcher

    @setup_channel_operation
    def deep_fill_channel(self, channel):  # n_blocks because of decorator
        self._deep_fill_channel(channel)  # TODO: update watcher

    def _binarize(self, channel):
        """

        postfix str
            empty for raw
        """
        binarization_cfg = self.config['binarization']['single_channels'][channel]
        if not binarization_cfg['binarize']['run']:
            return
        self.steps[channel].remove_next_steps_files(self.steps[channel].binary)

        source = self.workspace.source('stitched', channel=channel)
        sink = self.get_path('binary', channel=channel)

        binarization_parameter = copy.deepcopy(vasculature.default_binarization_parameter)
        binarization_parameter['clip']['clip_range'] = binarization_cfg['binarize']['clip_range']
        deconvolve_threshold = binarization_cfg['binarize']['threshold']
        if deconvolve_threshold is not None:
            binarization_parameter['deconvolve']['threshold'] = deconvolve_threshold

        if channel != self.all_vessels_channel:  # For arteries or veins
            binarization_parameter.update(equalize=None, vesselize=None)

        processing_parameter = copy.deepcopy(vasculature.default_binarization_processing_parameter)
        channel_perf = self.config['performance']['binarization']['single_channels'][channel]
        block_params = channel_perf['binarize']['block_processing']
        processing_parameter.update(
            processes=sanitize_n_processes(block_params['n_processes']),
            size_min=block_params['size_min'], size_max=block_params['size_max'],
            overlap=block_params['overlap'], as_memory=True, verbose=True)

        vasculature.binarize(source, sink,
                             binarization_parameter=binarization_parameter,
                             processing_parameter=processing_parameter)
        keep = binarization_cfg['binarize'].get('save', True)
        self.steps[channel].record_output(BinaryVesselProcessorSteps.binary, sink, keep=keep)

    def plot_binarization_result(self, parent=None, channel='', arrange=False):
        """
        channel str:
            The channel to plot
        """
        from ClearMap.Visualization.Qt import Plot3d as q_p3d
        images = [(self.get_path('stitched', asset_sub_type=channel)),
                  (self.get_path('binary', asset_sub_type=channel))]
        dvs = q_p3d.plot(images, title=[img.name for img in images],
                         arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        return dvs

    def _smooth(self, channel):
        binarization_cfg = self.config['binarization']['single_channels'][channel]
        if not binarization_cfg['smooth']['run']:
            return

        self.steps[channel].remove_next_steps_files(self.steps[channel].smoothed)

        source = self.steps[channel].get_source(BinaryVesselProcessorSteps.smoothed)
        source = clearmap_io.as_source(source)
        sink_path = self.get_path('binary', channel=channel, asset_sub_type='smoothed')
        smoothing_parameters = copy.deepcopy(vasculature.default_postprocessing_parameter['smooth'])

        channel_perf = self.config['performance']['binarization']['single_channels'][channel]
        perf_cfg = channel_perf['smooth']['block_processing']
        perf_params = copy.deepcopy(vasculature.default_postprocessing_processing_parameter)
        perf_params.update(size_max=perf_cfg['size_max'])

        result = vasculature.apply_smoothing(source, sink_path, smoothing_parameters, perf_params,
                                             processes=sanitize_n_processes(perf_cfg['n_processes']),
                                             verbose=True)

        self.steps[channel].consume_and_cleanup()
        keep = binarization_cfg['smooth'].get('save', True)
        self.steps[channel].record_output(BinaryVesselProcessorSteps.smoothed, result, keep=keep)

    def _fill(self, channel):
        if not self.config['binarization']['single_channels'][channel]['binary_fill']['run']:
            return

        self.steps[channel].remove_next_steps_files(self.steps[channel].filled)

        source = self.steps[channel].get_source(BinaryVesselProcessorSteps.filled)
        source = clearmap_io.as_source(source)
        sink = self.get_path('binary', channel=channel, asset_sub_type='filled')
        sink = initialize_sink(sink, shape=source.shape, dtype=source.dtype, order=source.order, return_buffer=False)

        perf_cfg = self.config['performance']['binarization']['single_channels'][channel]['binary_fill']
        binary_filling.fill(source, sink=sink, processes=sanitize_n_processes(perf_cfg['n_processes']),
                            verbose=True)

        self.steps[channel].consume_and_cleanup()
        keep = self.config['binarization']['single_channels'][channel]['binary_fill'].get('save', True)
        self.steps[channel].record_output(BinaryVesselProcessorSteps.filled, sink, keep=keep)

    def _deep_fill_channel(self, channel, size_max=None, overlap=None, resample_factor=None):
        binary_cfg = self.config['binarization']['single_channels'][channel]
        if not binary_cfg['deep_fill']['run']:
            return

        perf_params = self.config['performance']['binarization']['single_channels'][channel]['deep_fill']['block_processing']
        REQUIRED_V_RAM = 22000  # REFACTOR: put in config or at top of module
        if size_max is None:
            size_max = perf_params['size_max']
        if overlap is None:
            overlap = perf_params['overlap']
        if resample_factor is None:
            resample_factor = binary_cfg['deep_fill']['resample_factor']

        if not get_free_v_ram() > REQUIRED_V_RAM:
            btn = warning_popup(f'Insufficient VRAM',
                                f'You do not have enough free memory on your graphics card to '
                                f'run this operation. This step needs 22GB VRAM, {get_free_v_ram() / 1000} were found. '
                                f'Please free some or upgrade your hardware.')
            if btn == QDialogButtonBox.Abort:
                raise ClearMapVRamException(f'Insufficient VRAM, found only {get_free_v_ram()} < {REQUIRED_V_RAM}')
            elif btn == QDialogButtonBox.Retry:
                self._deep_fill_channel(channel, size_max, overlap, resample_factor)

        self.steps[channel].remove_next_steps_files(self.steps[channel].deep_filled)

        # TODO: check how source casts to bool
        source = self.steps[channel].get_source(BinaryVesselProcessorSteps.deep_filled)
        sink = self.get_path('binary', channel=channel, asset_sub_type='deep_filled')

        processing_parameter = copy.deepcopy(vessel_filling.default_fill_vessels_processing_parameter)
        processing_parameter.update(size_max=size_max, size_min='fixed', axes='all', overlap=overlap)

        vessel_filling.fill_vessels(source, sink, resample=resample_factor, threshold=0.5,
                                    cuda=True, processing_parameter=processing_parameter, verbose=True)
        gc.collect(); clear_cuda_cache()

        self.steps[channel].consume_and_cleanup()
        keep = binary_cfg['deep_fill'].get('save', True)
        self.steps[channel].record_output(BinaryVesselProcessorSteps.deep_filled, sink, keep=keep)

    def combine_binary(self):
        """Merge the binary images of the different vascular network components into a single mask"""
        # FIXME: probably missing the call to workspace.add_channel(self.channels_to_binarize())
        sink_asset = self.get('binary', channel=self.channels_to_binarize(), asset_sub_type='combined')  # Temporary
        if len(self.channels_to_binarize()) > 1:
            sources = [self.steps[ch].get_last_output() for ch in self.channels_to_binarize()]
            perf_params = self.config['performance']['binarization']['combine']['block_processing']
            block_processing.process(np.logical_or, sources, sink_asset.path,
                                     size_max=perf_params['size_max'], overlap=perf_params['overlap'],
                                     processes=sanitize_n_processes(perf_params['n_processes']), verbose=True)
        else:  # We expect to have at least all_vessels_channel
            source = self.steps[self.all_vessels_channel].get_last_output()
            clearmap_io.copy_file(source, sink_asset.path)

        self.post_process_binary_combined()
        if self.config['binarization']['combined'].get('compress'):
            with ProcessPoolExecutor(max_workers=1) as executor:  # Send to separate process to avoid blocking
                executor.submit(sink_asset.compress, algorithm='bz2')

    def post_process_binary_combined(self):
        """Postprocess the combined binary image (typically smooth and fill)"""
        source = self.get_path('binary', self.channels_to_binarize(), asset_sub_type='combined')
        sink = self.get_path('binary', self.channels_to_binarize(), asset_sub_type='final')
        if self.config['binarization']['combined']['binary_fill']:
            postprocessing_parameter = copy.deepcopy(vasculature.default_postprocessing_parameter)
            block_params = copy.deepcopy(vasculature.default_postprocessing_processing_parameter)
            block_params['size_max'] = 50
            vasculature.postprocess(source, sink, postprocessing_parameter=postprocessing_parameter,
                                    processing_parameter=block_params,
                                    processes=sanitize_n_processes(-1), verbose=True)  # TODO: n_processes in config?
        else:
            clearmap_io.link_file(source, sink)

    def plot_vessel_filling_results(self, parent=None, channel='', arrange=False):
        from ClearMap.Visualization.Qt import Plot3d as q_p3d
        channel = channel if channel else self.all_vessels_channel
        images = [(self.steps[self.all_vessels_channel].get_asset(
            self.steps[self.all_vessels_channel].filled, step_back=True).path),  # FIXME: check if we really want filled here
                  (self.get_path('binary', channel=channel, asset_sub_type='filled'))]
        titles = [img.stem for img in images]
        images = [str(img) for img in images]
        lut_ = self.machine_config['default_lut']
        return q_p3d.plot(images, title=titles, arrange=arrange, lut=lut_, parent=parent)

    def plot_combined(self, parent=None, arrange=False):  # TODO: final or not option
        from ClearMap.Visualization.Qt import Plot3d as q_p3d
        all_vessels = self.steps[self.all_vessels_channel].get_asset(self.steps[self.all_vessels_channel].filled,
                                                                     step_back=True)
        combined = self.get_path('binary', channel=self.channels_to_binarize(), asset_sub_type='combined')
        if self.config['binarization']['single_channels'][self.arteries_channel]['binarize']['run']:
            arteries_filled = self.get_path('binary', channel=self.arteries_channel, asset_sub_type='filled')
            dvs = q_p3d.plot([all_vessels, arteries_filled, combined], title=['all vessels', 'arteries', 'combined'],
                             arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        else:
            dvs = q_p3d.plot([all_vessels, combined], title=['all vessels', 'combined'],
                             arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        return dvs

    def plot_results(self, steps, channels=None, side_by_side=True, arrange=True, parent=None):
        from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
        from ClearMap.Visualization.Qt import Plot3d as q_p3d
        if channels is None:
            channels = [self.all_vessels_channel, ]
        images = [self.steps[channels[i]].get_asset(steps[i], step_back=True) for i in range(len(steps))]
        images = [img.path for img in images]
        for img in images:
            if not img.exists:
                raise MissingRequirementException(f'File {img} not found')
        titles = [os.path.basename(img) for img in images]
        if not side_by_side:  # overlay
            images = [images, ]
            titles = ' vs '.join(titles)
        dvs = q_p3d.plot(images, title=titles, arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        if len(dvs) > 1:
            link_dataviewers_cursors(dvs)
        return dvs, titles


class VesselGraphProcessor(PipelineOrchestrator):
    """
    The graph contains the following edge properties:
        * artery_raw
        * artery_binary
        * artery
        * vein
        * radii
        * distance_to_surface
    """
    config_name = 'vasculature'

    # Legacy voxel-space thresholds — kept for old graphs without spacing/radius_units
    # New graphs use µm equivalents from config
    _LEGACY_THRESHOLDS = {
        # artery channel expression measurement
        'artery_search_shift_vx': 0.0,  # _set_artery_binary
        'arteriness_search_shift_vx': 10.0,  # _set_arteriness
        # vein channel expression measurement
        'vein_search_shift_vx':      0.0,
        'veinness_search_shift_vx':  10.0,
        # post-process filters
        'restrictive_vein_radius_vx': 6.5,
        'permissive_vein_radius_vx': 6.5,
        'final_vein_radius_vx': 6.5,
        'artery_trace_radius_vx': 4.0,
        'vein_trace_radius_vx': 5.0,
    }

    # Support for legacy graphs without physical units — determines how radii are measured and thresholds applied
    class RadiusLevel(Enum):
        FULL = 'full'  # spacing + radius_units + radius_units_axial  (current)
        SCALAR_UM = 'scalar_um'  # spacing + radius_units, no axial             (intermediate)
        VOXELS = 'voxels'  # only radii in voxels

    def __init__(self, sample_manager: Optional[SampleManager] = None,
                 config_coordinator: Optional[ConfigCoordinator] = None,
                 registration_processor: Optional[RegistrationProcessor] = None):
        super().__init__(config_coordinator)
        self.sample_manager: Optional[SampleManager] = sample_manager
        self.registration_processor: Optional[RegistrationProcessor] = registration_processor
        self.workspace: Optional[Workspace2] = None  # set in setup

        self.build_graph_re = 'Graph'  # TBD:
        self.skel_re = 'Iteration'  # TBD:
        self.__graphs = {
            'raw': None,
            'cleaned': None,
            'reduced': None,
            'annotated': None,
            'traced': None
        }
        self.branch_density = None
        self.steps: VesselGraphProcessorSteps = VesselGraphProcessorSteps(self.workspace)  # FIXME: handle skeleton
        self.setup(sample_manager, registration_processor)
        self.parent_channels = tuple(self.config['binarization']['single_channels'].keys())
        self.steps.channel = self.parent_channels

    def setup(self, sample_manager=None, registration_processor=None):
        self.sample_manager = sample_manager if sample_manager is not None else self.sample_manager
        self.registration_processor = registration_processor or self.registration_processor
        if self.sample_manager is not None and self.sample_manager.setup_complete:
            self.workspace = self.sample_manager.workspace
            self.steps.workspace = self.workspace

            self.parent_channels = tuple(self.config['binarization']['single_channels'].keys())
            self.steps.channel = self.parent_channels

            sample_id = self.sample_manager.prefix
            self.workspace.ensure_pipeline('TubeMap', self.parent_channels, channel_content_type='compound',
                                           sample_id=sample_id, create_channel=True)

    def __get_graph(self, step):
        if step not in self.__graphs:
            raise ValueError(f'Unknown graph step "{step}"')
        g = self.__graphs[step]
        if g is None:
            g = self.get('graph', channel=self.parent_channels, asset_sub_type=step).read()
            self.__graphs[step] = g
        return g

    def __set_graph(self, step, graph):
        if step not in self.__graphs:
            raise ValueError(f'Unknown graph step "{step}"')
        self.__graphs[step] = graph

    def save_graph(self, base_name):
        graph = self.__graphs[base_name]  # We do not use the getter here to avoid loading the graph
        self.get('graph', channel=self.parent_channels, asset_sub_type=base_name).write(graph)

    @staticmethod
    def _graph_radius_level(graph) -> 'VesselGraphProcessor.RadiusLevel':
        has_spacing = 'spacing' in graph.graph_properties
        has_um = 'radius_units' in graph.vertex_properties
        has_um_axial = 'radius_units_axial' in graph.vertex_properties
        if has_spacing and has_um and has_um_axial:
            return VesselGraphProcessor.RadiusLevel.FULL
        if has_spacing and has_um:
            return VesselGraphProcessor.RadiusLevel.SCALAR_UM
        return VesselGraphProcessor.RadiusLevel.VOXELS

    def _legacy_warn(self, method: str) -> None:
        """Warn user if graphs are outdated and miss the units aware radii"""
        warnings.warn(f"{method}: graph predates unit-aware radii ('radius_units_axial' / 'spacing' absent). "
                      f"Falling back to voxel-space computation with legacy thresholds. "
                      f"Rebuild the graph (re-run clean + reduce) for physical accuracy.",
                      DeprecationWarning, stacklevel=3)

    @property
    def graph_raw(self):
        return self.__get_graph('raw')

    @graph_raw.setter
    def graph_raw(self, graph):
        self.__set_graph('raw', graph)

    @property
    def graph_cleaned(self):
        return self.__get_graph('cleaned')

    @graph_cleaned.setter
    def graph_cleaned(self, graph):
        self.__set_graph('cleaned', graph)

    @property
    def graph_reduced(self):
        return self.__get_graph('reduced')

    @graph_reduced.setter
    def graph_reduced(self, graph):
        self.__set_graph('reduced', graph)

    @property
    def graph_annotated(self):
        return self.__get_graph('annotated')

    @graph_annotated.setter
    def graph_annotated(self, graph):
        self.__set_graph('annotated', graph)

    @property
    def graph_traced(self):
        return self.__get_graph('traced')

    @graph_traced.setter
    def graph_traced(self, graph):
        self.__set_graph('traced', graph)

    def unload_temporary_graphs(self):
        """
        To free up memory

        Returns
        -------

        """
        self.graph_raw = None
        self.graph_cleaned = None
        self.graph_reduced = None

    @property
    def arteries_channel(self) -> str:
        return self.sample_manager.get_channels_by_type('arteries', missing_action='ignore',
                                                        multiple_found_action='error')

    @property
    def veins_channel(self) -> str:
        return self.sample_manager.get_channels_by_type('veins', missing_action='ignore',
                                                        multiple_found_action='error')

    @property
    def use_arteries_for_graph(self):  # TODO: see if improve
        return bool(self.arteries_channel)

    def run(self):
        self.pre_process()
        self.post_process()

    @staticmethod
    def reload_processing_config(operation):  # FIXME: rename
        @functools.wraps(operation)
        def wrapper(self, *args, **kwargs):
            graph_cfg = self.config['graph_construction']
            return operation(self, *args, graph_cfg=graph_cfg, **kwargs)

        return wrapper

    @staticmethod
    def requires_graph(step):
        def decorator(operation):
            @functools.wraps(operation)
            def wrapper(self, *args, **kwargs):
                try:
                    self.__get_graph(step)
                except FileNotFoundError:
                    raise MissingRequirementException(f"Graph for step '{step}' is missing.")
                return operation(self, *args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def requires_binary(asset_sub_type):
        def decorator(operation):
            @functools.wraps(operation)
            def wrapper(self, *args, **kwargs):
                try:
                    self.get('binary', channel=self.parent_channels, asset_sub_type=asset_sub_type)
                except FileNotFoundError:
                    raise MissingRequirementException(f"Binary asset '{asset_sub_type}' is missing.")
                return operation(self, *args, **kwargs)

            return wrapper

        return decorator

    # ################################ perf config accessor #####################################

    @property
    def _graph_perf_cfg(self) -> dict:
        """
        Read graph construction performance config (fresh each call)
        """
        return self.config.get('performance', {}).get('graph_construction', {})

    def _n_processes(self, step: str) -> int | None:
        """
        Resolve n_processes for a graph construction step.
        (Returns None when unset)
        """
        raw = self._graph_perf_cfg.get(step, {}).get('n_processes')
        return sanitize_n_processes(raw) if raw is not None else None

    ##################################### ACTUAL COMPUTATIONS ###################################

    def pre_process(self):
        self.skeletonize_and_build_graph()
        self.clean_graph()
        self.reduce_graph()
        self.register()

    @reload_processing_config
    def skeletonize_and_build_graph(self, graph_cfg=None):
        self.skeletonize()  # WARNING: main thread (prange)
        if graph_cfg['build']:
            self._build_graph_from_skeleton()  # WARNING: main thread (prange)

    @reload_processing_config
    def clean_graph(self, graph_cfg=None):
        if graph_cfg['clean']:
            self.__clean_graph()

    @reload_processing_config
    def reduce_graph(self, vertex_to_edge_mapping=None, edge_to_edge_mappings=None, graph_cfg=None):
        if graph_cfg['reduce']:
            self.__reduce_graph(vertex_to_edge_mappings=vertex_to_edge_mapping,
                                edge_to_edge_mappings=edge_to_edge_mappings)

    @reload_processing_config
    def register(self, graph_cfg=None):
        if not self.registration_processor.was_registered:
            raise MissingRequirementException('To register the graph, the underlying image needs to be registered.'
                                              ' Please run registration first.')
        if graph_cfg['transform'] or graph_cfg['annotate']:
            self.__register()

    @requires_binary('final')
    def skeletonize(self):
        if self.config['graph_construction']['skeletonize']:
            n_blocks = 100  # TODO: TBD
            self.prepare_watcher_for_substep(n_blocks, self.skel_re, f'Skeletonization', True)
            binary = self.get_path('binary', channel=self.parent_channels, asset_sub_type='final')
            skeletonization.skeletonize(binary, sink=self.get_path('skeleton', channel=self.parent_channels),  # WARNING: prange
                                        delete_border=True, processes=self._n_processes('skeletonize'), verbose=True)

    def _measure_radii(self):  # FIXME: do on the clean graph to avoid measuring cliques ?
        coordinates = self.graph_raw.vertex_coordinates()
        source = self.get_path('binary', channel=self.parent_channels, asset_sub_type='final')
        spacing = np.array(self.sample_manager.get_channel_resolution(self.parent_channels[0])) # µm/vox, shape (3,)

        # Distances in all 3 directions
        radii_um_axial = measure_radius.measure_radius(source, coordinates,
                                                       value=0, fraction=None, max_radius=150,
                                                       return_indices=False, default=-1,
                                                       return_radii_as_scalar=False, scale=spacing)  # WARNING: prange

        radii_vx_axial = radii_um_axial / spacing[None, :]  # (n, 3) voxels per axis

        self.graph_raw.define_vertex_property('radius_units_axial', radii_um_axial)  # µm (n,3)
        self.graph_raw.define_vertex_property('radii_axial', radii_vx_axial)  # vox (n,3)

        # Euclidean norms
        radii_um_scalar = np.linalg.norm(radii_um_axial, axis=1)
        radii_vx_scalar = np.linalg.norm(radii_vx_axial, axis=1)

        self.graph_raw.set_vertex_radii(radii_vx_scalar)
        self.graph_raw.define_vertex_property('radius_units', radii_um_scalar)

        if not self.graph_raw.has_graph_property('spacing'):
            self.graph_raw.add_graph_property('spacing', spacing)

    def _set_vertex_vessel_type_expression(self, asset_type: str, channel: str, asset_sub_type=None,
                                           radius_shift_um: float = 0.0,
                                           _legacy_radius_shift_vx: float = 0.0):
        """
        Parameters
        ----------
        radius_shift_um : float
            Extra search margin beyond the measured vessel wall, in µm.
        """
        dtype = self.sample_manager.data_type(channel)
        dtype_singular = f'{dtype[:-3]}y' if dtype.endswith('ies') else dtype[:-1]
        property_name = f'{dtype_singular}_{"raw" if asset_type == "stitched" else asset_type}'

        if not isinstance(asset_sub_type, (list, tuple)):
            asset_sub_type = [asset_sub_type]
        for sub_type in asset_sub_type:
            source = self.get_path(asset_type, channel=channel, asset_sub_type=sub_type)
            if source.exists():
                break

        coordinates = self.graph_raw.vertex_coordinates()

        level = self._graph_radius_level(self.graph_raw)
        if level == VesselGraphProcessor.RadiusLevel.FULL:
            spacing = np.array(self.graph_raw.graph_property('spacing'))   # µm/vox (3,)
            # Per-axis physical radii + margin
            radii_um_axial = self.graph_raw.vertex_property('radius_units_axial') + radius_shift_um  # Conservative: search far enough to cover vessel boundary in every axis
            search_radius_vx = np.max(radii_um_axial / spacing, axis=1)
        elif level == VesselGraphProcessor.RadiusLevel.SCALAR_UM:  # Intermediate level
            self._legacy_warn(f'_set_vertex_channel_expression({property_name}) scalar µm fallback')
            spacing          = np.array(self.graph_raw.graph_property('spacing'))
            radii_um         = self.graph_raw.vertex_property('radius_units') + radius_shift_um
            search_radius_vx = radii_um / np.mean(spacing)   # isotropic approx
        else:  # full legacy level
            self._legacy_warn(f'_set_vertex_channel_expression({property_name})')
            search_radius_vx = self.graph_raw.vertex_radii_voxels() + _legacy_radius_shift_vx

        res = measure_expression.measure_expression(source, coordinates, search_radius_vx, method='max',
                                                    n_processes=self._n_processes('build'))  # WARNING: prange
        prop = res if asset_type == 'binary' else np.asarray(res, dtype=float)  # TODO: do as f(source.dtype)
        self.graph_raw.define_vertex_property(property_name, prop)

    def _set_artery_binary(self):
        """Define if vertex is artery from binary labeling"""
        # FIXME: should use last step of BinaryVesselProcessor
        self._set_vertex_vessel_type_expression(asset_type='binary', channel=self.arteries_channel,
                                                asset_sub_type=['filled', 'postprocessed'],
                                                radius_shift_um=0.0,
                                                _legacy_radius_shift_vx=self._LEGACY_THRESHOLDS[
                                                    'artery_search_shift_vx'])

    def _set_arteriness(self):
        """Assign 'arteriness' from signal intensity."""
        level = self._graph_radius_level(self.graph_raw)
        if level != self.RadiusLevel.VOXELS:
            spacing = np.array(self.graph_raw.graph_property('spacing'))
            radius_shift_um = 10.0 * np.mean(spacing)  # average enclosing
        else:
            # Legacy path — radius_shift_um is ignored inside
            # _set_vertex_vessel_type_expression because it takes the
            # voxel branch using _legacy_radius_shift_vx instead
            radius_shift_um = 0.0

        self._set_vertex_vessel_type_expression(asset_type='stitched', channel=self.arteries_channel,
                                                radius_shift_um=radius_shift_um,
                                                _legacy_radius_shift_vx=self._LEGACY_THRESHOLDS['arteriness_search_shift_vx'])

    def _set_vein_binary(self):
        """Define if vertex is vein from binary labeling"""
        self._set_vertex_vessel_type_expression(asset_type='binary', channel=self.veins_channel,
                                                asset_sub_type=['filled', 'postprocessed'],
                                                radius_shift_um=0.0,
                                                _legacy_radius_shift_vx=self._LEGACY_THRESHOLDS['vein_search_shift_vx'])

    def _set_veinness(self):
        """Assign 'veinness' from signal intensity."""
        level = self._graph_radius_level(self.graph_raw)
        if level != self.RadiusLevel.VOXELS:
            spacing = np.array(self.graph_raw.graph_property('spacing'))
            radius_shift_um = 10.0 * np.mean(spacing)
        else:
            radius_shift_um = 0.0

        self._set_vertex_vessel_type_expression(asset_type='stitched', channel=self.veins_channel,
                                                radius_shift_um=radius_shift_um,
                                                _legacy_radius_shift_vx=self._LEGACY_THRESHOLDS['veinness_search_shift_vx'])

    def _build_graph_from_skeleton(self):  # TODO: split for requirements
        if self.config['graph_construction']['build']:
            n_blocks = 100  # TBD:
            self.prepare_watcher_for_substep(n_blocks, self.build_graph_re, 'Building graph', True)
            self.steps.remove_next_steps_files(self.steps.graph_raw)
            skeleton_path = self.get_path('skeleton', channel=self.parent_channels)
            spacing = self.sample_manager.get_channel_resolution(self.parent_channels[0])
            self.graph_raw = graph_processing.graph_from_skeleton(skeleton_path, spacing=spacing, physical_units='µm',
                                                                  # check_border=False, verbose=True)  # WARNING: main thread (prange)
                                                                  check_border=False,
                                                                  n_processes=1, verbose=True)  # WARNING: main thread (prange)
            self._measure_radii()  # WARNING: main thread (prange)
            if self.use_arteries_for_graph:  # TODO: do same for veins if exists
                self._set_artery_binary()  # WARNING: main thread (prange)
                self._set_arteriness()  # WARNING: main thread (prange)
            if self.veins_channel:
                self._set_vein_binary()
                self._set_veinness()
            self.save_graph('raw')

    @requires_graph('raw')
    def __clean_graph(self):
        """
        Remove spurious data e.g. outliers ...

        Returns graph_cleaned
        -------

        """
        vertex_mappings = copy.copy(graph_processing.DEFAULT_VERTEX_TO_VERTEX)
        if self.use_arteries_for_graph:
            vertex_mappings.update({'artery_binary': np.max, 'artery_raw': np.max})
        if self.veins_channel:
            vertex_mappings.update({'vein_binary': np.max, 'vein_raw': np.max})
        self.steps.remove_next_steps_files(self.steps.graph_cleaned)
        self.graph_cleaned = graph_processing.clean_graph(
            self.graph_raw, vertex_mappings=vertex_mappings,
            processes=self._n_processes('clean'), verbose=True)
        self.save_graph('cleaned')

    @requires_graph('cleaned')
    def __reduce_graph(self, vertex_to_edge_mappings=None, edge_to_edge_mappings=None):
        """
        Simplify straight segments between branches
        Returns
        -------

        """

        def vote(expression):
            return np.sum(expression) >= len(expression) / 1.5

        vertex_to_edge_mappings = vertex_to_edge_mappings or graph_processing.DEFAULT_VERTEX_TO_EDGE
        edge_to_edge_mappings = edge_to_edge_mappings
        edge_geometry_vertex_properties = ['coordinates', 'coordinates_units', 'radii', 'radius_units',
                                           'length', 'chain_id', '_vertex_id_']
        # add conditionally on new graphs
        if 'radii_axial' in self.graph_cleaned.vertex_properties:
            edge_geometry_vertex_properties.append('radii_axial')
        if 'radius_units_axial' in self.graph_cleaned.vertex_properties:
            edge_geometry_vertex_properties.append('radius_units_axial')
        if self.use_arteries_for_graph:
            vertex_to_edge_mappings.update({'artery_binary': vote, 'artery_raw': np.max})
            edge_geometry_vertex_properties.extend(['artery_binary', 'artery_raw'])
        if self.veins_channel:
            vertex_to_edge_mappings.update({'vein_binary': vote, 'vein_raw': np.max})
            edge_geometry_vertex_properties.extend(['vein_binary', 'vein_raw'])
        self.steps.remove_next_steps_files(self.steps.graph_reduced)
        self.graph_reduced = graph_processing.reduce_graph(self.graph_cleaned,
                                                           vertex_to_edge_mappings=vertex_to_edge_mappings,
                                                           edge_to_edge_mappings=edge_to_edge_mappings,
                                                           compute_edge_length=True,
                                                           edge_geometry_vertex_properties=edge_geometry_vertex_properties,
                                                           return_maps=False, processes=self._n_processes('reduce'),
                                                           verbose=True)
        self.save_graph('reduced')

    @property
    def resampled_shape(self):  # Can be any of the parent channels because they ought to have the same shape
        return self.get('resampled', channel=self.parent_channels[0]).shape()

    @property
    def binary_shape(self):
        return self.get('binary', channel=self.parent_channels, asset_sub_type='final').shape()

    # Atlas registration and annotation
    def _transform(self):
        def transformation(coordinates):
            coordinates = resampling_module.resample_points(
                coordinates,
                original_shape=self.binary_shape,
                resampled_shape=self.resampled_shape)

            if self.registration_processor.was_registered:
                for channel in self.get_registration_sequence_channels():
                    results_dir = self.get_path('aligned', channel=channel).parent
                    coordinates = elastix.transform_points(coordinates, transform_directory=results_dir,
                                                           binary=USE_BINARY_POINTS_FILE, indices=False)
            return coordinates

        self.graph_reduced.transform_properties(transformation=transformation,
                                                vertex_properties={'coordinates': 'coordinates_atlas'},
                                                edge_geometry_properties={'coordinates': 'coordinates_atlas'},
                                                verbose=True)
        self.save_graph('reduced')

    def _scale(self):
        """
        Convert radius properties from original space to atlas-voxel space.

        New graphs (FULL/SCALAR_UM): µm → atlas voxels
            scalar:  r_atlas = r_um / mean(atlas_spacing)
            axial:   r_atlas[:, i] = r_um[:, i] / atlas_spacing[i]

        Legacy graphs (VOXELS): original voxels → atlas voxels
            scalar:  r_atlas = r_vx * mean(spacing) / mean(atlas_spacing)
            axial:   r_atlas[:, i] = r_vx[:, i] * resample_factor[i]
        """
        resample_factor = np.array(resampling_module.resample_factor(
            original_shape=self.binary_shape,
            resampled_shape=self.resampled_shape))                    # (3,)

        graph = self.graph_reduced
        level = self._graph_radius_level(graph)

        if 'spacing' in graph.graph_properties:
            spacing = np.array(graph.graph_property('spacing'))
        else:
            self._legacy_warn('_scale (no spacing)')
            spacing = np.array(
                self.sample_manager.get_channel_resolution(self.parent_channels[0]))

        atlas_spacing      = spacing / resample_factor                # µm/atlas-vox (3,)
        atlas_spacing_mean = float(np.mean(atlas_spacing))
        spacing_mean       = float(np.mean(spacing))

        # ── scan all radius-like properties ──────────────────────────────
        radius_props: dict[str, tuple[bool, bool]] = {}   # name → (is_um, is_axial)
        for prop_name in list(graph.vertex_properties):
            if not ('radii' in prop_name or 'radius' in prop_name):
                continue
            if prop_name.endswith('_atlas'):
                continue
            prop   = graph.vertex_property(prop_name)
            is_um    = any(prop_name.endswith(sfx) for sfx in ('_units', 'um'))
            is_axial = prop.ndim == 2 and prop.shape[1] == 3
            radius_props[prop_name] = (is_um, is_axial)

        # ── define scaling functions ─────────────────────────────────────
        def scale_um_scalar(r):
            return r / atlas_spacing_mean

        def scale_um_axial(r):
            return r / atlas_spacing  # (n,3) / (3,)

        def scale_vx_scalar(r):
            return r * (spacing_mean / atlas_spacing_mean)

        def scale_vx_axial(r):
            return r * resample_factor  # (n,3) * (3,)

        dispatch = {  # key tuple (is_um, us_axial)
            (True,  True):  scale_um_axial,
            (True,  False): scale_um_scalar,
            (False, True):  scale_vx_axial,
            (False, False): scale_vx_scalar,
        }

        # ── apply per group ──────────────────────────────────────────────
        groups: dict[tuple[bool, bool], list[str]] = {}
        for prop_name, key in radius_props.items():
            groups.setdefault(key, []).append(prop_name)

        for key, prop_names in groups.items():
            fn      = dispatch[key]
            mapping = {p: f'{p}_atlas' for p in prop_names}

            e_mapping = {k: v for k, v in mapping.items()
                         if k in graph.edge_properties}
            eg_mapping = {k: v for k, v in mapping.items()
                          if k in graph.edge_geometry_properties}

            graph.transform_properties(transformation=fn, vertex_properties=mapping,
                                         edge_properties=e_mapping or None,
                                         edge_geometry_properties=eg_mapping or None)

    def _annotate(self):
        """Atlas annotation of the graph (i.e. add property 'region' to vertices)"""
        annotator = self.registration_processor.annotators[
            self.parent_channels[0]]  # warning: assuming same annotator for all channels
        self.graph_reduced.annotate_properties(functools.partial(annotator.label_points),
                                               vertex_properties={'coordinates_atlas': 'annotation'},
                                               edge_geometry_properties={'coordinates_atlas': 'annotation'})

        self.graph_reduced.annotate_properties(functools.partial(annotator.label_points_hemispheres),
                                               vertex_properties={'coordinates_atlas': 'hemisphere'},
                                               edge_geometry_properties={'coordinates_atlas': 'hemisphere'})

    def _compute_distance_to_surface(self):
        """add distance to brain surface as vertices properties"""
        distance_atlas = self.get('atlas', channel=self.parent_channels[0],
                                  asset_sub_type='distance_to_surface').read()
        atlas_shape = distance_atlas.shape

        def distance(coordinates):
            c = np.round(coordinates).astype(int)
            x, y, z = [np.clip(c[:, ax], 0, atlas_shape[ax] - 1) for ax in range(3)]
            return distance_atlas[x, y, z]

        graph = self.graph_reduced
        graph.transform_properties(distance,
                                   vertex_properties={'coordinates_atlas': 'distance_to_surface'},
                                   edge_geometry_properties={'coordinates_atlas': 'distance_to_surface'})

        distance_to_surface = graph.edge_geometry('distance_to_surface', as_list=True)
        distance_to_surface_edge = np.array([np.min(d) for d in distance_to_surface])
        graph.define_edge_property('distance_to_surface', distance_to_surface_edge)
        self.save_graph('reduced')

    @requires_graph('reduced')
    def __register(self):
        if self.config['graph_construction']['transform']:
            self._transform()
        self._scale()
        if self.registration_processor.was_registered and self.config['graph_construction']['annotate']:
            self._annotate()
            self._compute_distance_to_surface()
        self.steps.remove_next_steps_files(self.steps.graph_annotated)

        # discard non-connected graph components
        self.graph_annotated = self.graph_reduced.largest_component()
        self.save_graph('annotated')

    def get_filter(self, graph_step, filter_type, property_name, value):
        """
        Get a filter object for the graph step

        Parameters
        ----------
        graph_step: str
            The graph step to filter (e.g. 'annotated', 'reduced')
        filter_type: str
            The type of filter to apply (e.g. 'vertex', 'edge')
        property_name: str
            The property name to filter on (e.g. 'artery', 'vein', 'radii')
        value: int | float | bool | str | tuple
            The value to filter by. Can be:
                - int/float: exact match
                - str: string match (e.g. 'artery', 'vein')
                - bool: 'True' or 'False' for boolean properties
                - tuple: range (min, max) for numerical properties. None means open-ended range.

        Returns
        -------
        GraphFilter
            A filter object for the graph step
        """
        graph = self.__get_graph(graph_step)
        return GraphFilter(graph, filter_type, property_name, value)

    # POST PROCESS
    @requires_graph('annotated')
    def _pre_filter_veins(self, vein_intensity_range_on_arteries_channel: tuple[float, float], min_vein_radius_um: float):
        """
        Filter veins based on radius and intensity in arteries channel

        Parameters
        ----------
        vein_intensity_range_on_arteries_channel : (tuple)
            Above max (second val) on artery channel, this is an artery
        min_vein_radius: (int)

        Returns
        -------

        """
        is_in_vein_range = is_in_range(self.graph_annotated.edge_property('artery_raw'),
                                       vein_intensity_range_on_arteries_channel)

        level = self._graph_radius_level(self.graph_annotated)
        if level != VesselGraphProcessor.RadiusLevel.VOXELS:
            radii = self.graph_annotated.edge_radii_um()
            threshold = min_vein_radius_um
        else:
            self._legacy_warn('_pre_filter_veins')
            radii = self.graph_annotated.edge_radii_voxels()
            threshold = self._LEGACY_THRESHOLDS['restrictive_vein_radius_vx']

        return np.logical_and(radii >= threshold, is_in_vein_range)

    @requires_graph('annotated')
    def _pre_filter_arteries(self, arteries_min_noise_edges: int):
        """
        Remove components where too few edges are arteries

        Parameters
        ----------
        arteries_min_noise_edges : (int)
            below is capillary

        Returns
        -------

        """
        artery = self.graph_annotated.edge_property('artery_binary')

        artery_graph = self.graph_annotated.sub_graph(edge_filter=artery, view=True)
        artery_graph_edge, edge_map = artery_graph.edge_graph(return_edge_map=True)
        artery_components, artery_size = artery_graph_edge.label_components(return_vertex_counts=True)
        too_small = edge_map[np.in1d(artery_components, np.where(artery_size < arteries_min_noise_edges)[0])]
        artery[too_small] = False
        return artery

    @requires_graph('annotated')
    def _post_filter_veins(self, restrictive_veins, min_vein_radius_um: float | int):
        artery = self.graph_annotated.edge_property('artery')

        level = self._graph_radius_level(self.graph_annotated)
        if level != VesselGraphProcessor.RadiusLevel.VOXELS:
            radii = self.graph_annotated.edge_radii_um()
            threshold = min_vein_radius_um
        else:
            self._legacy_warn('_post_filter_veins')
            radii = self.graph_annotated.edge_radii_voxels()
            threshold = self._LEGACY_THRESHOLDS['permissive_vein_radius_vx']

        large_vessels = radii >= threshold
        permissive_veins = np.logical_and(np.logical_or(restrictive_veins, large_vessels), np.logical_not(artery))
        return permissive_veins

    # TRACING
    @requires_graph('annotated')
    def _trace_arteries(self, veins, max_tracing_iterations: int = 5):
        """
        Trace arteries by hysteresis thresholding
        Keeps small arteries that are too weakly immuno-positive but still too big to be capillaries
        stop at surface, vein or low artery expression

        Parameters
        ----------
        veins
        """
        artery = self.graph_annotated.edge_property('artery')

        level = self._graph_radius_level(self.graph_annotated)
        if level != VesselGraphProcessor.RadiusLevel.VOXELS:
            radii = self.graph_annotated.edge_radii_um()
            trace_radius = self.config['vessel_type_postprocessing']['tracing']['artery_trace_radius_um']
        else:
            self._legacy_warn('_trace_arteries')
            radii = self.graph_annotated.edge_radii_voxels()
            trace_radius = self._LEGACY_THRESHOLDS['artery_trace_radius_vx']

        condition_args = {
            'distance_to_surface': self.graph_annotated.edge_property('distance_to_surface'),
            'distance_threshold': self.config['vessel_type_postprocessing']['tracing']['distance_to_surface_min'],
            'vein': veins,
            'radii': radii,
            'artery_trace_radius': trace_radius,
            'artery_intensity': self.graph_annotated.edge_property('artery_raw'),
            'artery_intensity_min': self.config['vessel_type_postprocessing']['tracing']['artery_intensity_min']
        }

        def continue_edge(graph, edge, **kwargs):
            if kwargs['distance_to_surface'][edge] < kwargs['distance_threshold'] or kwargs['vein'][edge]:
                return False
            else:
                return (kwargs['radii'][edge] >= kwargs['artery_trace_radius'] and
                        kwargs['artery_intensity'][edge] >= kwargs['artery_intensity_min'])

        artery_traced = graph_processing.trace_edge_label(self.graph_annotated, artery,
                                                          condition=continue_edge,
                                                          max_iterations=max_tracing_iterations,
                                                          **condition_args)

        self.graph_annotated.define_edge_property('artery', artery_traced)

    @requires_graph('annotated')
    def _trace_veins(self, max_tracing_iterations: int = 5):
        """
        Trace veins by hysteresis thresholding - stop before arteries
        """
        min_distance_to_artery = 1
        artery = self.graph_annotated.edge_property('artery')
        artery_expanded = self.graph_annotated.edge_dilate_binary(artery, steps=min_distance_to_artery)

        trace_cfg = self.config['vessel_type_postprocessing']['tracing']
        pre_filt_cfg = self.config['vessel_type_postprocessing']['pre_filtering']

        level = self._graph_radius_level(self.graph_annotated)
        if level != VesselGraphProcessor.RadiusLevel.VOXELS:
            radii = self.graph_annotated.edge_radii_um()
            trace_radius = trace_cfg['vein_trace_radius_um']
        else:
            self._legacy_warn('_trace_veins')
            radii = self.graph_annotated.edge_radii_voxels()
            trace_radius = self._LEGACY_THRESHOLDS['vein_trace_radius_vx']

        artery_intensity = self.graph_annotated.edge_property('artery_raw') if self.use_arteries_for_graph else None
        vein_intensity = self.graph_annotated.edge_property('vein_raw') if self.veins_channel else None

        condition_args = {
            'artery_expanded': artery_expanded,
            'radii': radii,
            'artery_intensity': artery_intensity,
            'vein_trace_radius': trace_radius,
            'vein_intensity': vein_intensity,
            'vein_intensity_range': tuple(pre_filt_cfg['vein_intensity_range_on_arteries_ch']),
            'vein_intensity_min': trace_cfg['vein_intensity_min']
        }

        def continue_edge(graph, edge, **kwargs):
            if kwargs['artery_expanded'][edge]:  # too close to artery
                return False
            else:
                radius_ok = kwargs['radii'][edge] >= kwargs['vein_trace_radius']
                if not radius_ok:
                    return False
                else:
                    # If we have vein signal: must be positive
                    if kwargs['vein_intensity'] is not None:
                        if kwargs['vein_intensity'][edge] < kwargs['vein_intensity_min']:
                            return False
                    # If we have artery signal: must be low
                    if kwargs['artery_intensity'] is not None:
                        lo, hi = kwargs['vein_intensity_range_on_arteries_ch']
                        if not (lo <= kwargs['artery_intensity'][edge] <= hi):
                            return False
                    return True

        vein_traced = graph_processing.trace_edge_label(
            self.graph_annotated, self.graph_annotated.edge_property('vein'),
            condition=continue_edge, max_iterations=max_tracing_iterations,
            **condition_args)

        self.graph_annotated.define_edge_property('vein', vein_traced)

    @requires_graph('annotated')
    def _remove_small_vessel_components(self, vessel_name, min_vessel_size: int = 30):
        """
        Filter out small components that will become capillaries
        """
        vessel = self.graph_annotated.edge_property(vessel_name)
        graph_vessel = self.graph_annotated.sub_graph(edge_filter=vessel, view=True)
        graph_vessel_edge, edge_map = graph_vessel.edge_graph(return_edge_map=True)

        vessel_components, vessel_size = graph_vessel_edge.label_components(return_vertex_counts=True)
        remove = edge_map[np.in1d(vessel_components, np.where(vessel_size < min_vessel_size)[0])]
        vessel[remove] = False

        self.graph_annotated.define_edge_property(vessel_name, vessel)

    @requires_graph('annotated')
    def post_process(self):  # TODO: progress
        """
        Iteratively refine arteries and veins based on one another

        Returns
        -------

        """
        if self.use_arteries_for_graph:
            cfg = self.config['vessel_type_postprocessing']

            artery = self._pre_filter_arteries(cfg['pre_filtering']['arteries_min_noise_edges'])

            # Definitely a vein because too big
            restrictive_veins = self._pre_filter_veins(
                cfg['pre_filtering']['vein_intensity_range_on_arteries_ch'],
                min_vein_radius_um=cfg['pre_filtering']['restrictive_vein_radius_um'])
            artery[restrictive_veins] = False

            self.graph_annotated.define_edge_property('artery', artery)

            # Not huge vein but not an artery so still a vein (with temporary radius for artery tracing)
            tmp_veins = self._post_filter_veins(restrictive_veins,
                                                min_vein_radius_um=cfg['pre_filtering']['permissive_vein_radius_um'])
            self._trace_arteries(tmp_veins, max_tracing_iterations=cfg['tracing']['max_arteries_iterations'])

            # The real vein size filtering
            vein = self._post_filter_veins(restrictive_veins,
                                           min_vein_radius_um=cfg['pre_filtering']['final_vein_radius_um'])
            self.graph_annotated.define_edge_property('vein', vein)

            self._trace_veins(max_tracing_iterations=cfg['tracing']['max_veins_iterations'])

            #  Clean up fragments (veins or arteries) that tracing extended but not enough to be biologically meaningful.
            self._remove_small_vessel_components('artery',
                                                 min_vessel_size=cfg['capillaries_removal']['min_artery_component_edges'])
            self._remove_small_vessel_components('vein',
                                                 min_vessel_size=cfg['capillaries_removal']['min_vein_component_edges'])

            self.graph_annotated.save(self.get_path('graph', channel=self.parent_channels))
            self.graph_traced = self.graph_annotated

    def __get_branch_voxelization_params(self):
        voxelize_branch_parameter = {
            'method': 'sphere',
            'radius': tuple(self.config['visualization']['voxelization']['size']),
            'weights': None,
            'shape': self.get('atlas', channel=self.parent_channels[0], asset_sub_type='reference').shape(),
            'verbose': True
        }
        return voxelize_branch_parameter

    def __voxelize(self, vertices, voxelize_branch_parameter: dict[str, Any]):
        density_path = self.get_path('density', channel=self.parent_channels, asset_sub_type='branches')
        clearmap_io.delete_file(density_path)
        self.branch_density = voxelization.voxelize(vertices,
                                                    sink=density_path,
                                                    dtype='float32',
                                                    **voxelize_branch_parameter)  # WARNING: prange

    # @requires_graph('traced')
    def voxelize(self, weight_by_radius=False, vertex_degrees=None, filters=None, operators=None):
        try:
            graph = self.graph_traced
        except (KeyError, FileNotFoundError):
            graph = self.graph_annotated
        vertices = graph.vertex_property('coordinates_atlas')
        voxelize_branch_parameter = self.__get_branch_voxelization_params()

        if vertex_degrees:
            if filters is None:
                filters = []
                operators = []
            if not isinstance(vertex_degrees, (list, tuple)):
                vertex_degrees = (vertex_degrees, vertex_degrees)
            filters += [GraphFilter(graph, 'vertex', 'degree', vertex_degrees)]

        if filters:
            if len(operators) != len(filters) - 1:
                raise ValueError("Number of operators must be len(filters) - 1")

                # Start with the first atomic filter, then fold left
            combined = filters[0]
            for op_str, nxt in zip(operators, filters[1:]):
                combined = combined.combine_with(nxt, op_str)
            vertices = vertices[combined.as_mask('vertex')]

        if weight_by_radius:
            voxelize_branch_parameter.update(weights=graph.vertex_radii_units())

        self.__voxelize(vertices, voxelize_branch_parameter)

    def plot_voxelization(self, parent):
        from ClearMap.Visualization.Qt import Plot3d as q_p3d
        return q_p3d.plot(self.get_path('density', channel=self.parent_channels, asset_sub_type='branches'),
                          arrange=False, parent=parent, lut='flame')

    @requires_graph('traced')
    def write_vertex_table(self):
        """
        Write a table with vertex coordinates and properties
        """
        coordinates = self.graph_traced.vertex_property('coordinates')
        df = pd.DataFrame({'x': coordinates[:, 0], 'y': coordinates[:, 1], 'z': coordinates[:, 2]})
        df['degree'] = self.graph_traced.vertex_degrees()

        df['radius_vx'] = self.graph_traced.vertex_radii_voxels()
        if 'radius_units' in self.graph_traced.vertex_properties:
            df['radius_um'] = self.graph_traced.vertex_radii_units()

        if self.registration_processor.was_registered:
            annotator = self.registration_processor.annotators[self.parent_channels[0]]
            coordinates_transformed = self.graph_traced.vertex_property('coordinates_atlas')
            atlas_resolution = self.get_alignment_ref_channel_reg_cfg()['resampled_resolution']
            extra_columns = annotator.get_columns(coordinates_transformed, atlas_resolution,
                                                  self.graph_traced.vertex_property('annotation'))
            df = pd.concat([df, extra_columns], axis=1)

        df.to_feather(self.get_path('vertices', channel=self.parent_channels, extension='.feather'))

    @requires_graph('annotated')
    def get_structure_sub_graph(self, structure_id):
        # Assign label of requested structure to all its children
        annotator = self.registration_processor.annotators[self.parent_channels[0]]
        level = annotator.find(structure_id)['level']
        try:
            graph = self.graph_traced
        except MissingAssetError:
            graph = self.graph_annotated
        label_leveled = annotator.convert_label(graph.vertex_annotation(), value='id', level=level)

        vertex_filter = label_leveled == structure_id
        # if get_neighbours:
        #     vertex_filter = graph.expand_vertex_filter(vertex_filter, steps=2)
        if vertex_filter is None:
            return
        return graph.sub_graph(vertex_filter=vertex_filter)

    def plot_graph_structure(self, structure_id, plot_type):
        annotator = self.registration_processor.annotators[self.parent_channels[0]]
        structure_name = annotator.get_names_map()[structure_id]
        graph_chunk = self.get_structure_sub_graph(structure_id)
        if not graph_chunk:
            return
        region_color = annotator.label_to_color(graph_chunk.vertex_annotation(), key='id', alpha=True, as_int=False)
        return self.plot_graph_chunk(graph_chunk,
                                     title=f'Structure {structure_name} graph',
                                     plot_type=plot_type, region_color=region_color)

    def plot_graph_chunk(self, graph_chunk, plot_type='mesh', title='sub graph', region_color=None,
                         show=True, n_max_vertices=MAX_PLOT_VERTICES):
        from ClearMap.Visualization.Vispy import plot_graph_3d  # WARNING: vispy dependency
        if plot_type == 'line':
            scene = plot_graph_3d.plot_graph_line(graph_chunk, vertex_colors=region_color, title=title,
                                                  show=show, bg_color=self.machine_config['three_d_plot_bg'])
        elif plot_type == 'mesh':
            if graph_chunk.n_vertices > n_max_vertices:
                raise PlotGraphError(f'Cannot plot graph with more than {n_max_vertices},'
                                     f'got {graph_chunk.n_vertices}')
            if region_color is not None and region_color.ndim == 1:
                region_color = np.broadcast_to(region_color, (graph_chunk.n_vertices, 3))
            scene = plot_graph_3d.plot_graph_mesh(graph_chunk, vertex_colors=region_color,
                                                  title=title, show=show,
                                                  bg_color=self.machine_config['three_d_plot_bg'])
        elif plot_type == 'edge_property':
            scene = plot_graph_3d.plot_graph_edge_property(graph_chunk, edge_property='artery_raw', title=title,
                                                           percentiles=[2, 98], normalize=True, mesh=True,
                                                           show=show, bg_color=self.machine_config['three_d_plot_bg'])
        else:
            raise ValueError(f'Unrecognised plot type  "{plot_type}"')
        # scene.canvas.bgcolor = vispy.color.color_array.Color(self.machine_config['three_d_plot_bg'])
        return [scene.canvas.native]

    def visualize_graph_annotations(self, chunk_range, plot_type='mesh', graph_step='reduced', show=True):
        if graph_step in self.steps.existing_steps:
            graph = getattr(self, f'graph_{graph_step}')
        else:
            raise ValueError(f'graph step "{graph_step}" not recognised, '
                             f'available steps are "{self.steps.existing_steps}"')
        graph_chunk = graph.sub_slice(chunk_range)
        title = f'{graph_step.title()} Graph'
        annotator = self.registration_processor.annotators[self.parent_channels[0]]
        if graph_step == 'annotated':
            region_color = annotator.label_to_color(graph_chunk.vertex_annotation(), key='id', alpha=True, as_int=False)
        else:
            region_color = None
        return self.plot_graph_chunk(graph_chunk, plot_type, title, region_color, show)

    def get_registration_sequence_channels(self):
        return (self.registration_processor.
                get_registration_sequence_channels(self.parent_channels[0], 'atlas'))
