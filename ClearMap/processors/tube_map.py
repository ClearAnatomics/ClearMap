#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
TubeMap
=======

This module contains the classes to generate annotated graphs from vasculature
lightsheet data [Kirst2020]_.
"""
import os
import copy
import re
import functools
import platform
import warnings
import gc
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ClearMap.Analysis.graphs.graph_filters import GraphFilter
from ClearMap.IO.assets_specs import ChannelSpec
from PyQt5.QtWidgets import QDialogButtonBox

from ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing import initialize_sink
from ClearMap.Utils.exceptions import PlotGraphError, ClearMapVRamException, MissingRequirementException
from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
from ClearMap.processors.generic_tab_processor import TabProcessor, ProcessorSteps

import ClearMap.IO.IO as clearmap_io

import ClearMap.Alignment.Resampling as resampling_module
import ClearMap.Alignment.Elastix as elastix

import ClearMap.ImageProcessing.Experts.Vasculature as vasculature
import ClearMap.ImageProcessing.machine_learning.vessel_filling.vessel_filling as vessel_filling
import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skeletonization
import ClearMap.ImageProcessing.Binary.Filling as binary_filling

import ClearMap.Analysis.Measurements.MeasureExpression as measure_expression
import ClearMap.Analysis.Measurements.radius_measurements as measure_radius
from ClearMap.Analysis.graphs import graph_processing
import ClearMap.Analysis.Measurements.Voxelization as voxelization

import ClearMap.ParallelProcessing.BlockProcessing as block_processing

from ClearMap.Visualization.Qt import Plot3d as q_p3d
from ClearMap.Visualization.Vispy import plot_graph_3d  # WARNING: vispy dependency

from ClearMap.gui.dialogs import warning_popup
from ClearMap.Utils.utilities import is_in_range, get_free_v_ram, clear_cuda_cache

__author__ = ('Christoph Kirst <christoph.kirst.ck@gmail.com>,'
              ' Sophie Skriabine <sophie.skriabine@icm-institute.org>,'
              ' Charly Rousseau <charly.rousseau@icm-institute.org>')
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


USE_BINARY_POINTS_FILE = not platform.system().lower().startswith('darwin')


class VesselGraphProcessorSteps(ProcessorSteps):
    def __init__(self, workspace, channel='', sub_step=''):
        super().__init__(workspace, channel=channel, sub_step=sub_step)
        self.graph_raw = 'raw'
        self.graph_cleaned = 'cleaned'
        self.graph_reduced = 'reduced'
        self.graph_annotated = 'annotated'

    @property
    def steps(self):
        return self.graph_raw, self.graph_cleaned, self.graph_reduced, self.graph_annotated  # TODO: add traced

    def asset_from_step_name(self, step):
        return self.workspace.get('graph', channel=self.channel, asset_sub_type=step)


class BinaryVesselProcessorSteps(ProcessorSteps):
    def __init__(self, workspace, channel='', sub_step=''):
        super().__init__(workspace, channel=channel, sub_step=sub_step)
        self.stitched = 'stitched'
        self.binary = 'binary'
        self.postprocessed = 'postprocessed'
        self.filled = 'filled'
        self.combined = 'combined'
        self.final = 'final'

    @property
    def steps(self):
        return self.stitched, self.binary, self.postprocessed, self.filled, self.combined, self.final

    def asset_from_step_name(self, step):  # FIXME: split step and substep
        if step in (self.stitched, self.binary):
            asset = self.workspace.get(step, channel=self.channel)
        else:
            asset = self.workspace.get('binary', channel=self.channel, asset_sub_type=step)
        return asset

    # def last_path(self, arteries=False):
    #     return self.path(self.last_step, arteries=arteries)
    #
    # def previous_path(self, arteries=False):
    #     if self.previous_step is not None:
    #         return self.path(self.previous_step, arteries=arteries)


class BinaryVesselProcessor(TabProcessor):
    steps: dict[str: BinaryVesselProcessorSteps]

    def __init__(self, sample_manager=None):
        super().__init__()
        self.inputs_match = False
        self.postprocessing_tmp_params = None
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.sample_manager= None
        self.workspace = None
        self.all_vessels_channel = ''
        self.arteries_channel = ''
        # TODO: add veins too
        self.steps = {}
        self.block_re = ('Processing block',
                         re.compile(r'.*?Processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))
        self.vessel_filling_re = ('Vessel filling',
                                  re.compile(r'.*?Vessel filling: processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))

        self.setup(sample_manager)

    def setup(self, sample_manager):
        self.sample_manager = sample_manager
        if sample_manager is not None:
            self.workspace = sample_manager.workspace
            configs = sample_manager.get_configs()
            self.sample_config = configs['sample']
            self.machine_config = configs['machine']
            self.processing_config = self.sample_manager.config_loader.get_cfg('vasculature')

            self.set_progress_watcher(self.sample_manager.progress_watcher)

            self.all_vessels_channel = self.sample_manager.get_channels_by_type(channel_type='vessels')
            if not self.all_vessels_channel:
                warnings.warn('Vessels channel not set')
                return

            self.arteries_channel = self.sample_manager.get_channels_by_type(channel_type='arteries',
                                                                             multiple_found_action='warn')

            self.assert_input_shapes_match()

            all_channels = self.sample_config['channels'].keys()
            for k in self.steps.keys():
                if k not in all_channels:
                    self.steps.pop(k)

            for channel_name in (self.all_vessels_channel, self.arteries_channel):  # TODO: add veins here too
                if channel_name:
                    self.steps[channel_name] = BinaryVesselProcessorSteps(self.workspace, channel=channel_name)

            compound_channel = tuple(self.channels_to_binarize())
            if compound_channel not in self.workspace.asset_collections.keys():  # FIXME: could be string version
                sample_id = self.sample_manager.config['sample_id'] if self.sample_config['use_id_as_prefix'] else None
                self.workspace.add_channel(ChannelSpec(compound_channel, 'compound'), sample_id=sample_id)
                self.workspace.add_pipeline('TubeMap', compound_channel, sample_id=sample_id)

    def assert_input_shapes_match(self):
        """
        Ensure that the input shapes (stitched images of the different color channels) match
        before starting the binarization process since they must overlap.

        .. warning::
            stitched may not exist when processor is created.
            if not, check again in the run/binarize method
        """
        try:
            shapes = [self.workspace.get('stitched', channel=c).shape() for c in self.channels_to_binarize()]
        except FileNotFoundError:
            warnings.warn('Stitched images not found. Cannot check shapes.')
            return
        if not all([s == shapes[0] for s in shapes]):
            raise ValueError('Channels to binarize have different shapes. This is not supported yet.')
        self.inputs_match = True  # WARNING: may need to be reset when changing channels to binarize

    def run(self):
        self.binarize()
        self.combine_binary()

    def binarize(self):
        if not self.inputs_match:
            self.assert_input_shapes_match()
        if self.inputs_match:
            for channel in self.channels_to_binarize():
                self.binarize_channel(channel)
        else:
            raise ValueError('Channels to binarize have different shapes. This is not supported yet.')

    def channels_to_binarize(self):
        return tuple([c for c in self.processing_config['binarization'].keys() if c != 'combined'])

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
            self.processing_config.reload()
            cfg = self.processing_config['binarization']
            operation_type = operation.__name__.replace('_channel', '')
            operations = list(cfg[channel].keys())
            if operation_type == 'deep_fill':
                n_blocks = 1200
            else:
                n_blocks = self.__get_n_blocks(channel)

            # FIXME: find other way to increment
            increment_main = not( channel == self.channels_to_binarize()[0] and operation_type == operations[0])
            self.prepare_watcher_for_substep(n_blocks, self.block_re,
                                             f'{operation_type} {channel.title()}', increment_main)

            return operation(self, channel, *args, **kwargs)

        return wrapper

    @setup_channel_operation
    def binarize_channel(self, channel):
        self._binarize(channel)  # TODO: update watcher

    @setup_channel_operation
    def smooth_channel(self, channel):
        self.smooth(channel)  # TODO: update watcher

    @setup_channel_operation
    def fill_channel(self, channel):  # WARNING: should run from main thread
        self.fill(channel)  # TODO: update watcher

    @setup_channel_operation
    def deep_fill_channel(self, channel):  # n_blocks because of decorator
        self.__deep_fill_channel(channel)  # TODO: update watcher

    def _binarize(self, channel):
        """

        postfix str
            empty for raw
        """
        self.steps[channel].remove_next_steps_files(self.steps[channel].binary)

        source = self.workspace.source('stitched', channel=channel)
        sink = self.get_path('binary', channel=channel)

        binarization_parameter = copy.deepcopy(vasculature.default_binarization_parameter)
        binarization_cfg = self.processing_config['binarization'][channel]
        binarization_parameter['clip']['clip_range'] = binarization_cfg['binarize']['clip_range']
        deconvolve_threshold = binarization_cfg['binarize']['threshold']
        if deconvolve_threshold is not None:
            binarization_parameter['deconvolve']['threshold'] = deconvolve_threshold

        if channel != self.all_vessels_channel:  # For arteries or veins
            binarization_parameter.update(equalize=None, vesselize=None)

        processing_parameter = copy.deepcopy(vasculature.default_binarization_processing_parameter)
        processing_parameter.update(processes=self.machine_config['n_processes_binarization'],
                                    overlap=self.machine_config['detection_chunk_overlap'],
                                    size_max=self.machine_config['detection_chunk_size_max'],
                                    size_min=self.machine_config['detection_chunk_size_min'],
                                    as_memory=True, verbose=True)

        vasculature.binarize(source, sink,
                             binarization_parameter=binarization_parameter,
                             processing_parameter=processing_parameter)

    def plot_binarization_result(self, parent=None, channel='', arrange=False):
        """
        channel str:
            The channel to plot
        """
        images = [(self.get_path('stitched', asset_sub_type=channel)),
                  (self.get_path('binary', asset_sub_type=channel))]
        dvs = q_p3d.plot(images, title=[img.name for img in images],
                         arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        return dvs

    def smooth(self, channel):
        binarization_cfg = self.processing_config['binarization'][channel]
        run_smoothing = binarization_cfg['smooth']['run']
        run_filling = binarization_cfg['binary_fill']['run']

        self.steps[channel].remove_next_steps_files(self.steps[channel].postprocessed)  # TODO: do we keep ?

        source = self.workspace.source('binary', channel=channel)
        sink = self.get_path('binary', channel=channel, asset_sub_type='postprocessed')
        sink = initialize_sink(sink, shape=source.shape, dtype=source.dtype, order=source.order, return_buffer=False)

        params = copy.deepcopy(vasculature.default_postprocessing_processing_parameter)
        params.update(size_max=50)

        postprocessing_parameter = copy.deepcopy(vasculature.default_postprocessing_parameter)
        if not run_smoothing:
            postprocessing_parameter.update(smooth=False)
        postprocessing_parameter.update(fill=run_filling)

        fill_source, tmp_f_path, save = vasculature.apply_smoothing(source, sink, params, postprocessing_parameter,
                                                                    processes=None, verbose=True)
        self.postprocessing_tmp_params = {'fill_source': fill_source, 'tmp_path': tmp_f_path, 'save': save}

        # vasculature.postprocess(source, sink, processing_parameter=params,
        #                         postprocessing_parameter=postprocessing_parameter,
        #                         processes=None, verbose=True)  # WARNING: prange if filling

    def fill(self, channel):
        binarization_cfg = self.processing_config['binarization'][channel]
        run_smoothing = binarization_cfg['smooth']['run']
        sink = self.get_path('binary', channel=channel, asset_sub_type='postprocessed')
        if not run_smoothing and os.path.exists(sink):  # FIXME: cannot assume, should prompt (or use separate files)
            source = sink  # We assume the smoothing ran previously, hence source is previous postprocessed
        else:
            if self.postprocessing_tmp_params is not None:
                source = self.postprocessing_tmp_params['fill_source']
            else:
                source = self.get_path('binary', channel=channel, asset_sub_type='postprocessed')

        source = clearmap_io.as_source(source)
        sink = initialize_sink(sink, shape=source.shape, dtype=source.dtype, order=source.order, return_buffer=False)

        binary_filling.fill(source, sink=sink, processes=None, verbose=True)  # WARNING: prange if filling
        if run_smoothing and not self.postprocessing_tmp_params['save']:
            clearmap_io.delete_file(self.postprocessing_tmp_params['tmp_path'])

    def __deep_fill_channel(self, channel, size_max=None, overlap=None, resample_factor=None):
        REQUIRED_V_RAM = 22000  # TODO: put in config at top of module
        if size_max is None:
            size_max = self.processing_config['binarization'][channel]['deep_fill']['size_max']
        if overlap is None:
            overlap = self.processing_config['binarization'][channel]['deep_fill']['overlap']
        if resample_factor is None:
            resample_factor = self.processing_config['binarization'][channel]['deep_fill']['resample_factor']

        if not get_free_v_ram() > REQUIRED_V_RAM:
            btn = warning_popup(f'Insufficient VRAM',
                                f'You do not have enough free memory on your graphics card to '
                                f'run this operation. This step needs 22GB VRAM, {get_free_v_ram()/1000} were found. '
                                f'Please free some or upgrade your hardware.')
            if btn == QDialogButtonBox.Abort:
                raise ClearMapVRamException(f'Insufficient VRAM, found only {get_free_v_ram()} < {REQUIRED_V_RAM}')
            elif btn == QDialogButtonBox.Retry:
                self.__deep_fill_channel(channel, size_max, overlap, resample_factor)

        self.steps[channel].remove_next_steps_files(self.steps[channel].filled)

        source = self.get_path('binary', channel=channel, asset_sub_type='postprocessed')
        sink = self.get_path('binary', channel=channel, asset_sub_type='filled')

        processing_parameter = copy.deepcopy(vessel_filling.default_fill_vessels_processing_parameter)
        processing_parameter.update(size_max=size_max, size_min='fixed', axes='all', overlap=overlap)

        vessel_filling.fill_vessels(source, sink, resample=resample_factor, threshold=0.5,
                                    cuda=True, processing_parameter=processing_parameter, verbose=True)
        gc.collect()
        clear_cuda_cache()

    def combine_binary(self):
        """Merge the binary images of the different vascular network components into a single mask"""
        # FIXME: probably missing the call to workspace.add_channel(self.channels_to_binarize())
        sink_asset = self.get('binary', channel=self.channels_to_binarize(), asset_sub_type='combined')  # Temporary
        if len(self.channels_to_binarize()) > 1:
            sources = []
            for channel in self.channels_to_binarize():
                asset = self.steps[channel].get_asset(self.steps[channel].filled, step_back=True)
                if asset.exists:
                    sources.append(asset.path)
                else:
                    raise FileNotFoundError(f'File {asset.path} not found')
            block_processing.process(np.logical_or, sources, sink_asset.path,
                                     size_max=500, overlap=0, processes=None, verbose=True)
        else:  # We expect to have at least all_vessels_channel
            source = self.steps[self.all_vessels_channel].get_asset(self.steps[self.all_vessels_channel].filled,
                                                                    step_back=True).path
            clearmap_io.copy_file(source, sink_asset.path)

        self.post_process_binary_combined()
        if self.processing_config['binarization']['combined'].get('compress'):
            with ProcessPoolExecutor(max_workers=1) as executor:  # Send to separate process to avoid blocking
                executor.submit(sink_asset.compress, algorithm='bz2')

    def post_process_binary_combined(self):
        """Postprocess the combined binary image (typically smooth and fill)"""
        source = self.get_path('binary', self.channels_to_binarize(), asset_sub_type='combined')
        sink = self.get_path('binary', self.channels_to_binarize(), asset_sub_type='final')
        if self.processing_config['binarization']['combined']['binary_fill']:
            postprocessing_parameter = copy.deepcopy(vasculature.default_postprocessing_parameter)
            postprocessing_processing_parameter = copy.deepcopy(vasculature.default_postprocessing_processing_parameter)
            postprocessing_processing_parameter['size_max'] = 50
            vasculature.postprocess(source, sink, postprocessing_parameter=postprocessing_parameter,
                                    processing_parameter=postprocessing_processing_parameter,
                                    processes=None, verbose=True)
        else:
            clearmap_io.copy_file(source, sink)  # FIXME: could be a symlink

    def plot_vessel_filling_results(self, parent=None, channel='', arrange=False):
        channel = channel if channel else self.all_vessels_channel
        images = [(self.steps[self.all_vessels_channel].get_asset(
            self.steps[self.all_vessels_channel].postprocessed, step_back=True)),
                  (self.get_path('binary', channel=channel, asset_sub_type='filled'))]
        titles = [os.path.basename(img) for img in images]
        return q_p3d.plot(images, title=titles, arrange=arrange,
                          lut=self.machine_config['default_lut'], parent=parent)

    def plot_combined(self, parent=None, arrange=False):  # TODO: final or not option
        all_vessels = self.steps[self.all_vessels_channel].get_asset(self.steps[self.all_vessels_channel].filled, step_back=True)
        combined = self.get_path('binary', channel=self.channels_to_binarize(), asset_sub_type='combined')
        if self.processing_config['binarization'][self.arteries_channel]['binarize']['run']:
            arteries_filled = self.get_path('binary', channel=self.arteries_channel, asset_sub_type='filled')
            dvs = q_p3d.plot([all_vessels, arteries_filled, combined], title=['all vessels', 'arteries', 'combined'],
                             arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        else:
            dvs = q_p3d.plot([all_vessels, combined], title=['all vessels', 'combined'],
                             arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        return dvs

    def plot_results(self, steps, channels=None, side_by_side=True, arrange=True, parent=None):
        if channels is None:
            channels = [self.all_vessels_channel, ]
        images = [self.steps[channels[i]].get_asset(steps[i], step_back=True) for i in range(len(steps))]
        for img in images:
            if not img.exists():
                raise MissingRequirementException(f'File {img} not found')
        titles = [os.path.basename(img) for img in images]
        if not side_by_side:  # overlay
            images = [images, ]
            titles = ' vs '.join(titles)
        dvs = q_p3d.plot(images, title=titles, arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        if len(dvs) > 1:
            link_dataviewers_cursors(dvs)
        return dvs, titles


class VesselGraphProcessor(TabProcessor):
    """
    The graph contains the following edge properties:
        * artery_raw
        * artery_binary
        * artery
        * vein
        * radii
        * distance_to_surface
    """
    def __init__(self, sample_manager=None, registration_processor=None):
        super().__init__()
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
        self.sample_config = {}
        self.processing_config = {}
        self.machine_config = {}
        self.sample_manager = sample_manager
        self.registration_processor = registration_processor
        self.workspace = None
        self.steps = VesselGraphProcessorSteps(self.workspace)  # FIXME: handle skeleton
        self.setup(sample_manager, registration_processor)
        self.parent_channels = tuple([k for k in self.processing_config['binarization'].keys() if k != 'combined'])
        self.steps.channel = self.parent_channels
        self.arteries_channel = self.sample_manager.get_channels_by_type('arteries', missing_action='ignore',
                                                                         multiple_found_action='error')

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

    def setup(self, sample_manager, registration_processor):
        self.sample_manager = sample_manager
        self.registration_processor = registration_processor
        if sample_manager is not None:
            self.workspace = sample_manager.workspace
            self.steps.workspace = self.workspace
            configs = sample_manager.get_configs()
            self.sample_config = configs['sample']
            self.machine_config = configs['machine']
            self.processing_config = self.sample_manager.config_loader.get_cfg('vasculature')

            self.set_progress_watcher(self.sample_manager.progress_watcher)

            self.parent_channels = tuple([k for k in self.processing_config['binarization'].keys() if k != 'combined'])
            self.steps.channel = self.parent_channels

            if self.parent_channels not in self.workspace.asset_collections.keys():
                sample_id = self.sample_manager.config['sample_id']
                self.workspace.add_channel(ChannelSpec(self.parent_channels, 'compound'), sample_id=sample_id)
                self.workspace.add_pipeline('TubeMap', self.parent_channels, sample_id=sample_id)

    @property
    def use_arteries_for_graph(self):  # TODO: see if improve
        return bool(self.arteries_channel)

    def run(self):
        self.pre_process()
        self.post_process()

    @staticmethod
    def reload_processing_config(operation):
        @functools.wraps(operation)
        def wrapper(self, *args, **kwargs):
            self.processing_config.reload()
            graph_cfg = self.processing_config['graph_construction']
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
        if graph_cfg['transform'] or graph_cfg['annotate']:
            self.__register()

    @requires_binary('final')
    def skeletonize(self):
        if self.processing_config['graph_construction']['skeletonize']:
            n_blocks = 100  # TODO: TBD
            self.prepare_watcher_for_substep(n_blocks, self.skel_re, f'Skeletonization', True)
            binary = self.get_path('binary', channel=self.parent_channels, asset_sub_type='final')
            skeletonization.skeletonize(binary, sink=self.get_path('skeleton', channel=self.parent_channels),  # WARNING: prange
                                        delete_border=True, verbose=True)

    def _measure_radii(self):
        coordinates = self.graph_raw.vertex_coordinates()
        source = self.get_path('binary', channel=self.parent_channels, asset_sub_type='final')
        spacing = np.array(self.sample_manager.get_channel_resolution(self.parent_channels[0]))
        radii, indices = measure_radius.measure_radius(source, coordinates,
                                                       value=0, fraction=None, max_radius=150,
                                                       return_indices=True, default=-1, scale=spacing)  # WARNING: prange
        self.graph_raw.define_vertex_property('radius_units', radii)
        # TODO: call measure_radius with return_radii_as_scalar=False and store vectors

    def _set_graph_artery_property(self, asset_type, asset_sub_type=None, suffix='', radius_shift=0):
        suffix = suffix or asset_type
        if not isinstance(asset_sub_type, (list, tuple)):
            asset_sub_type = [asset_sub_type]
        for sub_type in asset_sub_type:
            source = self.get_path(asset_type, channel=self.arteries_channel, asset_sub_type=sub_type)
            if source.exists():
                break
        coordinates = self.graph_raw.vertex_coordinates()  # OPTIMISE: cache ?
        radii = self.graph_raw.vertex_radii() + radius_shift
        expression = measure_expression.measure_expression(source, coordinates, radii, method='max')  # WARNING: prange
        prop = expression if asset_type == 'binary' else np.asarray(expression.array, dtype=float)  # TODO: do as f(source.dtype)
        self.graph_raw.define_vertex_property(f'artery_{suffix}', prop)

    def _set_artery_binary(self):
        """Define if vertex is artery from binary labelling"""
        self._set_graph_artery_property('binary', asset_sub_type=['filled', 'postprocessed'])

    def _set_arteriness(self):
        """'arteriness' from signal intensity"""
        self._set_graph_artery_property('stitched', suffix='raw', radius_shift=10)

    def _build_graph_from_skeleton(self):  # TODO: split for requirements
        if self.processing_config['graph_construction']['build']:
            n_blocks = 100  # TBD:
            self.prepare_watcher_for_substep(n_blocks, self.build_graph_re, 'Building graph', True)
            self.steps.remove_next_steps_files(self.steps.graph_raw)
            skeleton_path = self.get_path('skeleton', channel=self.parent_channels)
            spacing = self.sample_manager.get_channel_resolution(self.parent_channels[0])
            self.graph_raw = graph_processing.graph_from_skeleton(skeleton_path, spacing=spacing, distance_unit='µm',
                                                                  verbose=True)  # WARNING: main thread (prange)
            self._measure_radii()  # WARNING: main thread (prange)
            if self.use_arteries_for_graph:  # TODO: do same for veins if exists
                self._set_artery_binary()  # WARNING: main thread (prange)
                self._set_arteriness()  # WARNING: main thread (prange)
            self.save_graph('raw')

    @requires_graph('raw')
    def __clean_graph(self):
        """
        Remove spurious data e.g. outliers ...

        Returns graph_cleaned
        -------

        """
        vertex_mappings = {
            'coordinates': graph_processing.mean_vertex_coordinates,
            'radii': np.max
        }
        if self.use_arteries_for_graph:
            vertex_mappings.update({
                'artery_binary': np.max,
                'artery_raw': np.max
            })  # TODO: do same for veins if exists
        self.steps.remove_next_steps_files(self.steps.graph_cleaned)
        self.graph_cleaned = graph_processing.clean_graph(self.graph_raw, vertex_mappings=vertex_mappings, verbose=True)
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

        vertex_to_edge_mappings = vertex_to_edge_mappings or {'radii': np.max}
        edge_to_edge_mappings = edge_to_edge_mappings or {'length': np.sum}
        edge_geometry_vertex_properties = ['coordinates', 'radii']
        if self.use_arteries_for_graph:
            vertex_to_edge_mappings.update({
                'artery_binary': vote,
                'artery_raw': np.max})  # TODO: do same for veins if exists
            edge_geometry_vertex_properties.extend(['artery_binary', 'artery_raw'])
        self.steps.remove_next_steps_files(self.steps.graph_reduced)
        self.graph_reduced = graph_processing.reduce_graph(self.graph_cleaned, compute_edge_length=True,
                                                           edge_to_edge_mappings=edge_to_edge_mappings,
                                                           vertex_to_edge_mappings=vertex_to_edge_mappings,
                                                           edge_geometry_vertex_properties=edge_geometry_vertex_properties,
                                                           edge_geometry_edge_properties=None,
                                                           return_maps=False, verbose=True)
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
        """Apply transform to graph properties"""
        def scaling(radii):
            resample_factor = resampling_module.resample_factor(
                original_shape=self.binary_shape,
                resampled_shape=self.resampled_shape)
            return radii * np.mean(resample_factor)

        self.graph_reduced.transform_properties(transformation=scaling,
                                                vertex_properties={'radii': 'radii_atlas'},
                                                edge_properties={'radii': 'radii_atlas'},
                                                edge_geometry_properties={'radii': 'radii_atlas'})

    def _annotate(self):
        """Atlas annotation of the graph (i.e. add property 'region' to vertices)"""
        annotator = self.registration_processor.annotators[self.parent_channels[0]]  # warning: assuming same annotator for all channels
        self.graph_reduced.annotate_properties(functools.partial(annotator.label_points),
                                               vertex_properties={'coordinates_atlas': 'annotation'},
                                               edge_geometry_properties={'coordinates_atlas': 'annotation'})

        self.graph_reduced.annotate_properties(functools.partial(annotator.label_points_hemispheres),
                                               vertex_properties={'coordinates_atlas': 'hemisphere'},
                                               edge_geometry_properties={'coordinates_atlas': 'hemisphere'})

    def _compute_distance_to_surface(self):
        """add distance to brain surface as vertices properties"""
        # %% Distance to surface
        distance_atlas = self.get('atlas', channel=self.parent_channels[0], asset_sub_type='distance_to_surface').read()
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
        if self.processing_config['graph_construction']['transform']:
            self._transform()
        self._scale()
        if self.registration_processor.was_registered and self.processing_config['graph_construction']['annotate']:
            self._annotate()
            self._compute_distance_to_surface()
        self.steps.remove_next_steps_files(self.steps.graph_annotated)

        # discard non connected graph components
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
                - tuple: range (min, max) for numerical properties. None means open ended range.

        Returns
        -------
        GraphFilter
            A filter object for the graph step
        """
        graph = self.__get_graph(graph_step)
        return GraphFilter(graph, filter_type, property_name, value)

    # POST PROCESS
    @requires_graph('annotated')
    def _pre_filter_veins(self, vein_intensity_range_on_arteries_channel, min_vein_radius):
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
        radii = self.graph_annotated.edge_property('radii')
        restrictive_vein = np.logical_and(radii >= min_vein_radius, is_in_vein_range)
        return restrictive_vein

    @requires_graph('annotated')
    def _pre_filter_arteries(self, huge_vein, min_size):
        """
        Remove capillaries and veins from arteries

        Parameters
        ----------
        min_size : (int)  below is capillary

        Returns
        -------

        """
        artery = self.graph_annotated.edge_property('artery_binary')

        artery_graph = self.graph_annotated.sub_graph(edge_filter=artery, view=True)
        artery_graph_edge, edge_map = artery_graph.edge_graph(return_edge_map=True)
        artery_components, artery_size = artery_graph_edge.label_components(return_vertex_counts=True)
        too_small = edge_map[np.in1d(artery_components, np.where(artery_size < min_size)[0])]
        artery[too_small] = False
        artery[huge_vein] = False
        return artery

    @requires_graph('annotated')
    def _post_filter_veins(self, restrictive_veins, min_vein_radius=6.5):
        radii = self.graph_annotated.edge_property('radii')
        artery = self.graph_annotated.edge_property('artery')

        large_vessels = radii >= min_vein_radius
        permissive_veins = np.logical_and(np.logical_or(restrictive_veins, large_vessels), np.logical_not(artery))
        return permissive_veins

    # TRACING
    @requires_graph('annotated')
    def _trace_arteries(self, veins, max_tracing_iterations=5):
        """
        Trace arteries by hysteresis thresholding
        Keeps small arteries that are too weakly immuno-positive but still too big to be capillaries
        stop at surface, vein or low artery expression

        Parameters
        ----------
        veins
        """
        artery = self.graph_annotated.edge_property('artery')
        condition_args = {
            'distance_to_surface': self.graph_annotated.edge_property('distance_to_surface'),
            'distance_threshold': 15,
            'vein': veins,
            'radii': self.graph_annotated.edge_property('radii'),
            'artery_trace_radius': 4,  # FIXME: param
            'artery_intensity': self.graph_annotated.edge_property('artery_raw'),
            'artery_intensity_min': 200  # FIXME: param
        }

        def continue_edge(graph, edge, **kwargs):
            if kwargs['distance_to_surface'][edge] < kwargs['distance_threshold'] or kwargs['vein'][edge]:
                return False
            else:
                return kwargs['radii'][edge] >= kwargs['artery_trace_radius'] and \
                       kwargs['artery_intensity'][edge] >= kwargs['artery_intensity_min']

        artery_traced = graph_processing.trace_edge_label(self.graph_annotated, artery,
                                                          condition=continue_edge, max_iterations=max_tracing_iterations,
                                                          **condition_args)
        # artery_traced = graph.edge_open_binary(graph.edge_close_binary(artery_traced, steps=1), steps=1)
        self.graph_annotated.define_edge_property('artery', artery_traced)

    @requires_graph('annotated')
    def _trace_veins(self, max_tracing_iterations=5):
        """
        Trace veins by hysteresis thresholding - stop before arteries
        """
        min_distance_to_artery = 1

        artery = self.graph_annotated.edge_property('artery')
        radii = self.graph_annotated.edge_property('radii')
        condition_args = {
            'artery_expanded': self.graph_annotated.edge_dilate_binary(artery, steps=min_distance_to_artery),
            'radii': radii,
            'vein_trace_radius': 5  # FIXME: param
        }

        def continue_edge(graph, edge, **kwargs):
            if kwargs['artery_expanded'][edge]:
                return False
            else:
                return kwargs['radii'][edge] >= kwargs['vein_trace_radius']

        vein_traced = graph_processing.trace_edge_label(self.graph_annotated, self.graph_annotated.edge_property('vein'),
                                                        condition=continue_edge, max_iterations=max_tracing_iterations,
                                                        **condition_args)
        # vein_traced = graph.edge_open_binary(graph.edge_close_binary(vein_traced, steps=1), steps=1)

        self.graph_annotated.define_edge_property('vein', vein_traced)

    @requires_graph('annotated')
    def _remove_small_vessel_components(self, vessel_name, min_vessel_size=30):
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
        self.processing_config.reload()
        if self.use_arteries_for_graph:
            cfg = self.processing_config['vessel_type_postprocessing']
            # Definitely a vein because too big
            restrictive_veins = self._pre_filter_veins(cfg['pre_filtering']['vein_intensity_range_on_arteries_ch'],
                                                       min_vein_radius=cfg['pre_filtering']['restrictive_vein_radius'])

            artery = self._pre_filter_arteries(restrictive_veins, min_size=cfg['pre_filtering']['arteries_min_radius'])
            self.graph_annotated.define_edge_property('artery', artery)

            # Not huge vein but not an artery so still a vein (with temporary radius for artery tracing)
            tmp_veins = self._post_filter_veins(restrictive_veins,
                                                min_vein_radius=cfg['pre_filtering']['permissive_vein_radius'])
            self._trace_arteries(tmp_veins, max_tracing_iterations=cfg['tracing']['max_arteries_iterations'])

            # The real vein size filtering
            vein = self._post_filter_veins(restrictive_veins, min_vein_radius=cfg['pre_filtering']['final_vein_radius'])
            self.graph_annotated.define_edge_property('vein', vein)

            self._trace_veins(max_tracing_iterations=cfg['tracing']['max_veins_iterations'])

            self._remove_small_vessel_components('artery', min_vessel_size=cfg['capillaries_removal']['min_artery_size'])
            self._remove_small_vessel_components('vein', min_vessel_size=cfg['capillaries_removal']['min_vein_size'])

            self.graph_annotated.save(self.get_path('graph', channel=self.parent_channels))
            self.graph_traced = self.graph_annotated

    def __get_branch_voxelization_params(self):
        voxelize_branch_parameter = {
            'method': 'sphere',
            'radius': tuple(self.processing_config['visualization']['voxelization']['size']),
            'weights': None,
            'shape': self.get('atlas', channel=self.parent_channels[0], asset_sub_type='reference').shape(),
            'verbose': True
        }
        return voxelize_branch_parameter

    def __voxelize(self, vertices, voxelize_branch_parameter):
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
            voxelize_branch_parameter.update(weights=graph.vertex_radii())

        self.__voxelize(vertices, voxelize_branch_parameter)

    def plot_voxelization(self, parent):
        return q_p3d.plot(self.get_path('density', channel=self.parent_channels, asset_sub_type='branches'),
                          arrange=False, parent=parent, lut='flame')

    @requires_graph('traced')
    def write_vertex_table(self):
        """
        Write a table with vertex coordinates and properties
        """
        coordinates = self.graph_traced.vertex_property('coordinates')
        df = pd.DataFrame({'x': coordinates[:, 0], 'y': coordinates[:, 1], 'z': coordinates[:, 2]})
        df['radius'] = self.graph_traced.vertex_property('radii')
        df['degree'] = self.graph_traced.vertex_degrees()

        if self.registration_processor.was_registered:
            annotator = self.registration_processor.annotators[self.parent_channels[0]]
            coordinates_transformed = self.graph_traced.vertex_property('coordinates_atlas')
            atlas_resolution = self.registration_processor.config['channels'][self.sample_manager.alignment_reference_channel]['resampled_resolution']
            extra_columns = annotator.get_columns(coordinates_transformed, atlas_resolution,
                                                  self.graph_traced.vertex_property('annotation'))
            df = pd.concat([df, extra_columns], axis=1)

        df.to_feather(self.get_path('vertices', channel=self.parent_channels, extension='.feather'))

    @requires_graph('traced')
    def get_structure_sub_graph(self, structure_id):
        # Assign label of requested structure to all its children
        annotator = self.registration_processor.annotators[self.parent_channels[0]]
        level = annotator.find(structure_id)['level']
        label_leveled = annotator.convert_label(self.graph_traced.vertex_annotation(), value='id', level=level)

        vertex_filter = label_leveled == structure_id
        # if get_neighbours:
        #     vertex_filter = graph.expand_vertex_filter(vertex_filter, steps=2)
        if vertex_filter is None:
            return
        return self.graph_traced.sub_graph(vertex_filter=vertex_filter)

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

    def plot_graph_chunk(self, graph_chunk, plot_type='mesh', title='sub graph', region_color=None, show=True, n_max_vertices=300000):
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
        out = [self.parent_channels[0]]
        registration_cfg = self.registration_processor.config['channels']
        while True:
            next_channel = registration_cfg[out[-1]]['align_with']
            if next_channel in (None, 'atlas'):
                break
            out.append(next_channel)
        return out
