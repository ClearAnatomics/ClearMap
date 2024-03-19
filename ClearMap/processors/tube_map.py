#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
TubeMap
=======

This module contains the classes to generate annotated graphs from vasculature
lightsheet data [Kirst2020]_.
"""
import gc
import os
import copy
import re

import numpy as np
from PyQt5.QtWidgets import QDialogButtonBox

from ClearMap.Utils.exceptions import PlotGraphError, ClearMapVRamException
from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
from ClearMap.processors.generic_tab_processor import TabProcessor, ProcessorSteps

import ClearMap.IO.IO as clearmap_io

import ClearMap.Alignment.Annotation as annotation_module
import ClearMap.Alignment.Resampling as resampling_module
import ClearMap.Alignment.Elastix as elastix

import ClearMap.ImageProcessing.Experts.Vasculature as vasculature
import ClearMap.ImageProcessing.machine_learning.vessel_filling.vessel_filling as vessel_filling
import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skeletonization
import ClearMap.ImageProcessing.Binary.Filling as binary_filling

import ClearMap.Analysis.Measurements.MeasureExpression as measure_expression
import ClearMap.Analysis.Measurements.MeasureRadius as measure_radius
import ClearMap.Analysis.Graphs.GraphProcessing as graph_processing
import ClearMap.Analysis.Measurements.Voxelization as voxelization

import ClearMap.ParallelProcessing.BlockProcessing as block_processing

from ClearMap.Visualization.Qt import Plot3d as q_p3d
from ClearMap.Visualization.Vispy import PlotGraph3d as plot_graph_3d  # WARNING: vispy dependency

from ClearMap.gui.dialogs import warning_popup
from ClearMap.Utils.utilities import is_in_range, get_free_v_ram, requires_files, FilePath

__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Sophie Skriabine <sophie.skriabine@icm-institute.org>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


class VesselGraphProcessorSteps(ProcessorSteps):
    def __init__(self, workspace, postfix=''):
        super().__init__(workspace, postfix=postfix)
        self.graph_raw = 'raw'
        self.graph_cleaned = 'cleaned'
        self.graph_reduced = 'reduced'
        self.graph_annotated = 'annotated'

    @property
    def steps(self):
        return self.graph_raw, self.graph_cleaned, self.graph_reduced, self.graph_annotated  # FIXME: add traced

    def path_from_step_name(self, step):
        f_path = self.workspace.filename('graph', postfix=step)
        return f_path


class BinaryVesselProcessorSteps(ProcessorSteps):
    def __init__(self, workspace, postfix=''):
        super().__init__(workspace, postfix)
        self.stitched = 'stitched'
        self.binary = 'binary'
        self.postprocessed = 'postprocessed'
        self.filled = 'filled'
        self.combined = 'combined'
        self.final = 'final'

    @property
    def steps(self):
        return self.stitched, self.binary, self.postprocessed, self.filled, self.combined, self.final

    def path_from_step_name(self, step):
        if step in (self.stitched, self.binary):
            f_path = self.workspace.filename(step, postfix=self.postfix)
        else:
            postfix_base = ''
            if self.postfix and step in (self.postprocessed, self.filled):
                postfix_base = f'{self.postfix}_'
            f_path = self.workspace.filename('binary', postfix=f'{postfix_base}{step}')
        return f_path

    # def last_path(self, arteries=False):
    #     return self.path(self.last_step, arteries=arteries)
    #
    # def previous_path(self, arteries=False):
    #     if self.previous_step is not None:
    #         return self.path(self.previous_step, arteries=arteries)


class BinaryVesselProcessor(TabProcessor):
    def __init__(self, preprocessor=None):
        super().__init__()
        self.postprocessing_tmp_params = None
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.preprocessor = None
        self.workspace = None
        self.steps = {
            'raw': BinaryVesselProcessorSteps(self.workspace),
            'arteries': BinaryVesselProcessorSteps(self.workspace, postfix='arteries'),
        }
        self.block_re = ('Processing block',
                         re.compile(r'.*?Processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))
        self.vessel_filling_re = ('Vessel filling',
                                  re.compile(r'.*?Vessel filling: processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))

        self.setup(preprocessor)

    def setup(self, preprocessor):
        self.preprocessor = preprocessor
        if preprocessor is not None:
            self.workspace = preprocessor.workspace
            for steps in self.steps.values():
                steps.workspace = self.workspace
            configs = preprocessor.get_configs()
            self.sample_config = configs['sample']
            self.machine_config = configs['machine']
            self.processing_config = self.preprocessor.config_loader.get_cfg('vasculature')

            self.set_progress_watcher(self.preprocessor.progress_watcher)

    def run(self):
        self.binarize()
        self.combine_binary()

    def get_n_blocks(self, dim_size):
        blk_size = vasculature.default_binarization_processing_parameter['size_max']
        overlap = vasculature.default_binarization_processing_parameter['overlap']
        n_blocks = int(np.ceil((dim_size - blk_size) / (blk_size - overlap) + 1))
        return n_blocks

    def binarize(self):  # TODO: check real n blocks for post_processing
        # Raw
        self.binarize_channel('raw')
        self.binarize_channel('arteries')

    def binarize_channel(self, channel):
        self.processing_config.reload()
        binarization_cfg = self.processing_config['binarization'][channel]
        postfix = channel if channel == 'arteries' else None

        n_blocks = self.get_n_blocks(self.workspace.source('stitched', postfix=postfix).shape[2])
        if binarization_cfg['binarization']['run']:
            self.prepare_watcher_for_substep(n_blocks, self.block_re, f'{channel.title()} binarization',
                                             channel == 'arteries')
            self._binarize(self.machine_config['n_processes_binarization'],
                           binarization_cfg['binarization']['clip_range'],
                           binarization_cfg['binarization']['threshold'],
                           channel)  # FIXME: update watcher

    def smooth_channel(self, channel):
        self.processing_config.reload()
        binarization_cfg = self.processing_config['binarization'][channel]
        postfix = channel if channel == 'arteries' else None

        n_blocks = self.get_n_blocks(self.workspace.source('stitched', postfix=postfix).shape[2])
        if binarization_cfg['smoothing']['run']:
            self.prepare_watcher_for_substep(n_blocks, self.block_re, f'Smoothing {channel.title()}', True)
            self.smooth(channel)  # FIXME: update watcher

    def fill_channel(self, channel):  # WARNING: should run from main thread
        self.processing_config.reload()
        binarization_cfg = self.processing_config['binarization'][channel]
        postfix = channel if channel == 'arteries' else None

        n_blocks = self.get_n_blocks(self.workspace.source('stitched', postfix=postfix).shape[2])
        if binarization_cfg['binary_filling']['run']:
            self.prepare_watcher_for_substep(n_blocks, self.block_re, f'Binary filling {channel.title()}', True)
            self.fill(channel)  # FIXME: update watcher

    def deep_fill_channel(self, channel):
        self.processing_config.reload()
        binarization_cfg = self.processing_config['binarization'][channel]
        postfix = channel if channel == 'arteries' else None

        n_blocks = self.get_n_blocks(self.workspace.source('stitched', postfix=postfix).shape[2])
        if binarization_cfg['deep_filling']['run']:
            self.prepare_watcher_for_substep(1200, self.vessel_filling_re, f'Deep filling {channel} channel', True)  # FIXME: compute max
            if channel == 'raw':  # FIXME: update watcher
                self._fill_vessels(500, 50, channel)   # FIXME: number  literals
            elif channel == 'arteries':  # FIXME: update watcher
                self._fill_vessels(1000, 100, channel, resample_factor=2)   # FIXME: number  literals

    def _binarize(self, n_processes, clip_range, deconvolve_threshold, channel):
        """

        postfix str
            empty for raw
        """

        postfix = channel if channel == 'arteries' else None
        self.steps[channel].remove_next_steps_files(self.steps[channel].binary)

        source = self.workspace.filename('stitched', postfix=postfix)
        sink = self.workspace.filename('binary', postfix=postfix)

        binarization_parameter = copy.deepcopy(vasculature.default_binarization_parameter)
        binarization_parameter['clip']['clip_range'] = clip_range
        if deconvolve_threshold is not None:
            binarization_parameter['deconvolve']['threshold'] = deconvolve_threshold

        if postfix == 'arteries':  # FIXME: more flexible, do not use name
            binarization_parameter.update(equalize=None, vesselize=None)

        processing_parameter = copy.deepcopy(vasculature.default_binarization_processing_parameter)
        processing_parameter.update(processes=n_processes, as_memory=True, verbose=True)

        vasculature.binarize(source, sink,
                             binarization_parameter=binarization_parameter,
                             processing_parameter=processing_parameter)

    def plot_binarization_result(self, parent=None, postfix='', arrange=False):
        """
        postfix str:
            empty for raw
        """
        images = [(self.workspace.filename('stitched', postfix=postfix)),
                  (self.workspace.filename('binary', postfix=postfix))]
        dvs = q_p3d.plot(images, title=[os.path.basename(img) for img in images],
                         arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        return dvs

    def smooth(self, channel):
        postfix = channel if channel == 'arteries' else None
        binarization_cfg = self.processing_config['binarization'][channel]
        run_smoothing = binarization_cfg['smoothing']['run']
        run_filling = binarization_cfg['binary_filling']['run']

        self.steps[channel].remove_next_steps_files(self.steps[channel].postprocessed)

        source = self.workspace.filename('binary', postfix=postfix)
        sink_postfix = f'{postfix}_postprocessed' if postfix else 'postprocessed'
        sink = self.workspace.filename('binary', postfix=sink_postfix)

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
        run_smoothing = binarization_cfg['smoothing']['run']
        postfix = channel if channel == 'arteries' else None
        sink_postfix = f'{postfix}_postprocessed' if postfix else 'postprocessed'
        sink = self.workspace.filename('binary', postfix=sink_postfix)
        if not run_smoothing and os.path.exists(sink):  # FIXME: cannot assume, should prompt (or use separate files)
            source = sink  # We assume the smoothing ran previously, hence source is previous postprocessed
        else:
            if self.postprocessing_tmp_params is not None:
                source = self.postprocessing_tmp_params['fill_source']
            else:
                source = self.workspace.filename('binary', postfix=postfix)

        binary_filling.fill(source, sink=sink, processes=None, verbose=True)  # WARNING: prange if filling
        if run_smoothing and not self.postprocessing_tmp_params['save']:
            clearmap_io.delete_file(self.postprocessing_tmp_params['tmp_path'])

    def plot_vessel_filling_results(self, parent=None, postfix_base='', arrange=False):
        if postfix_base:
            postfix_base += '_'
        images = [(self.steps['raw'].path(self.steps['raw'].postprocessed, step_back=True)),
                  (self.workspace.filename('binary', postfix=f'{postfix_base}filled'))]
        titles = [os.path.basename(img) for img in images]
        return q_p3d.plot(images, title=titles, arrange=arrange,
                          lut=self.machine_config['default_lut'], parent=parent)

    def _fill_vessels(self, size_max, overlap, channel, resample_factor=1):
        REQUIRED_V_RAM = 22000
        if not get_free_v_ram() > REQUIRED_V_RAM:
            btn = warning_popup(f'Insufficient VRAM',
                                f'You do not have enough free memory on your graphics card to '
                                f'run this operation. This step needs 22GB VRAM, {get_free_v_ram()/1000} were found. '
                                f'Please free some or upgrade your hardware.')
            if btn == QDialogButtonBox.Abort:
                raise ClearMapVRamException(f'Insufficient VRAM, found only {get_free_v_ram()} < {REQUIRED_V_RAM}')
            elif btn == QDialogButtonBox.Retry:
                self._fill_vessels(size_max, overlap, channel, resample_factor)

        self.steps[channel].remove_next_steps_files(self.steps[channel].filled)

        postfix_base = ''
        if channel == 'arteries':
            postfix_base = 'arteries_'
        source = self.workspace.filename('binary', postfix=f'{postfix_base}postprocessed')
        sink = self.workspace.filename('binary', postfix=f'{postfix_base}filled')

        processing_parameter = copy.deepcopy(vessel_filling.default_fill_vessels_processing_parameter)
        processing_parameter.update(size_max=size_max, size_min='fixed', axes='all', overlap=overlap)

        vessel_filling.fill_vessels(source, sink, resample=resample_factor, threshold=0.5,
                                    cuda=True, processing_parameter=processing_parameter, verbose=True)
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    def plot_combined(self, parent=None, arrange=False):  # TODO: final or not option
        raw = self.steps['raw'].path(self.steps['raw'].filled, step_back=True)
        combined = self.steps['raw'].path(self.steps['raw'].combined)
        if self.processing_config['binarization']['arteries']['binarization']['run']:
            arteries_filled = self.workspace.filename('binary', postfix='arteries_filled')
            dvs = q_p3d.plot([raw, arteries_filled, combined], title=['Raw', 'arteries', 'combined'],
                             arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        else:
            dvs = q_p3d.plot([raw, combined], title=['Raw', 'combined'],
                             arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        return dvs

    def combine_binary(self):
        # MERGE
        sink = self.workspace.filename('binary', postfix='combined')  # Temporary
        if not self.processing_config['binarization']['arteries']['binarization']['run']:
            source = self.workspace.filename('binary', postfix='filled')
            source_arteries = self.workspace.filename('binary', postfix='arteries_filled')
            block_processing.process(np.logical_or, [source, source_arteries], sink,
                                     size_max=500, overlap=0, processes=None, verbose=True)
        else:
            source = self.steps['raw'].path(self.steps['raw'].filled, step_back=True)
            clearmap_io.copy_file(source, sink)

        # POST_PROCESS
        postprocessing_parameter = copy.deepcopy(vasculature.default_postprocessing_parameter)
        postprocessing_processing_parameter = copy.deepcopy(vasculature.default_postprocessing_processing_parameter)
        postprocessing_processing_parameter['size_max'] = 50
        source = self.workspace.filename('binary', postfix='combined')
        sink = self.workspace.filename('binary', postfix='final')
        vasculature.postprocess(source, sink, postprocessing_parameter=postprocessing_parameter,
                                processing_parameter=postprocessing_processing_parameter,
                                processes=None, verbose=True)
        # clearmap_io.delete_file(workspace.filename('binary', postfix='combined')  # TODO: check since temporary
        # if plot:
        #     return q_p3d.plot([source, sink], arrange=False, parent=parent)

    def plot_results(self, steps, channels=('raw',), side_by_side=True, arrange=True, parent=None):
        images = [self.steps[channels[i]].path(steps[i], step_back=True) for i in range(len(steps))]
        titles = [os.path.basename(img) for img in images]
        if not side_by_side:  # overlay
            images = [images, ]
            titles = ' vs '.join(titles)
        dvs = q_p3d.plot(images, title=titles, arrange=arrange, lut=self.machine_config['default_lut'], parent=parent)
        if len(dvs) > 1:
            link_dataviewers_cursors(dvs)
        return dvs


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
    def __init__(self, preprocessor=None):
        super().__init__()
        self.build_graph_re = 'Graph'  # TBD:
        self.skel_re = 'Iteration'  # TBD:
        self.__graph_raw = None
        self.__graph_cleaned = None
        self.__graph_reduced = None
        self.__graph_annotated = None
        self.__graph_traced = None
        self.branch_density = None
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.preprocessor = None
        self.workspace = None
        self.steps = VesselGraphProcessorSteps(self.workspace)  # FIXME: handle skeleton
        self.setup(preprocessor)

    @property
    def graph_raw(self):
        if self.__graph_raw is None:
            self.__graph_raw = clearmap_io.read(self.workspace.filename('graph', postfix='raw'))  # FIXME: handle missing
        return self.__graph_raw

    @graph_raw.setter
    def graph_raw(self, graph):
        self.__graph_raw = graph

    @property
    def graph_cleaned(self):
        if self.__graph_cleaned is None:
            self.__graph_cleaned = clearmap_io.read(self.workspace.filename('graph', postfix='cleaned'))  # FIXME: handle missing
        return self.__graph_cleaned

    @graph_cleaned.setter
    def graph_cleaned(self, graph):
        self.__graph_cleaned = graph

    @property
    def graph_reduced(self):
        if self.__graph_reduced is None:
            self.__graph_reduced = clearmap_io.read(self.workspace.filename('graph', postfix='reduced'))
        return self.__graph_reduced

    @graph_reduced.setter
    def graph_reduced(self, graph):
        self.__graph_reduced = graph

    @property
    def graph_annotated(self):
        if self.__graph_annotated is None:
            self.__graph_annotated = clearmap_io.read(self.workspace.filename('graph', postfix='annotated'))
        return self.__graph_annotated

    @graph_annotated.setter
    def graph_annotated(self, graph):
        self.__graph_annotated = graph

    @property
    def graph_traced(self):
        if self.__graph_traced is None:
            self.__graph_traced = clearmap_io.read(self.workspace.filename('graph'))
        return self.__graph_traced

    @graph_traced.setter
    def graph_traced(self, graph):
        self.__graph_traced = graph

    def unload_temporary_graphs(self):
        """
        To free up memory
        
        Returns
        -------

        """
        self.graph_raw = None
        self.graph_cleaned = None
        self.graph_reduced = None

    def setup(self, preprocessor):
        self.preprocessor = preprocessor
        if preprocessor is not None:
            self.workspace = preprocessor.workspace
            self.steps.workspace = self.workspace
            configs = preprocessor.get_configs()
            self.sample_config = configs['sample']
            self.machine_config = configs['machine']
            self.processing_config = self.preprocessor.config_loader.get_cfg('vasculature')

            self.set_progress_watcher(self.preprocessor.progress_watcher)

    def run(self):
        self.pre_process()
        self.post_process()

    @property
    def use_arteries_for_graph(self):
        return self.processing_config['graph_construction']['use_arteries']

    def pre_process(self):
        self.skeletonize_and_build_graph()
        self.clean_graph()
        self.reduce_graph()
        self.register()

    def skeletonize_and_build_graph(self):
        self.processing_config.reload()
        graph_cfg = self.processing_config['graph_construction']
        self.skeletonize(self.workspace.filename('skeleton'))# WARNING: main thread (prange)
        if graph_cfg['build'] or graph_cfg['skeletonize']:
            self._build_graph_from_skeleton()  # WARNING: main thread (prange)

    def clean_graph(self):
        self.processing_config.reload()
        graph_cfg = self.processing_config['graph_construction']
        if graph_cfg['clean']:
            self.__clean_graph()

    def reduce_graph(self, vertex_to_edge_mapping=None, edge_to_edge_mappings=None):
        self.processing_config.reload()
        graph_cfg = self.processing_config['graph_construction']
        if graph_cfg['reduce']:
            self.__reduce_graph(vertex_to_edge_mappings=vertex_to_edge_mapping, edge_to_edge_mappings=edge_to_edge_mappings)

    def register(self):
        self.processing_config.reload()
        graph_cfg = self.processing_config['graph_construction']
        if graph_cfg['transform'] or graph_cfg['annotate']:
            self.__register()

    @requires_files([FilePath('binary', postfix='final')])
    def skeletonize(self, skeleton):
        if self.processing_config['graph_construction']['skeletonize']:
            n_blocks = 100  # FIXME: TBD
            self.prepare_watcher_for_substep(n_blocks, self.skel_re, f'Skeletonization', True)
            binary = self.workspace.filename('binary', postfix='final')
            skeletonization.skeletonize(binary, sink=skeleton,  # WARNING: prange
                                        delete_border=True, verbose=True)

    def _measure_radii(self):
        coordinates = self.graph_raw.vertex_coordinates()
        radii, indices = measure_radius.measure_radius(self.workspace.filename('binary', postfix='final'),
                                                       coordinates,
                                                       value=0, fraction=None, max_radius=150,
                                                       return_indices=True, default=-1)  # WARNING: prange
        self.graph_raw.set_vertex_radii(radii)

    def _set_artery_binary(self):
        """check if vertex is artery from binary labelling"""
        binary_arteries = self.workspace.filename('binary', postfix='arteries_filled')
        coordinates = self.graph_raw.vertex_coordinates()  # OPTIMISE: is this necessary again (set as attribute ?)
        radii = self.graph_raw.vertex_radii()
        expression = measure_expression.measure_expression(binary_arteries, coordinates,
                                                           radii, method='max')   # WARNING: prange
        self.graph_raw.define_vertex_property('artery_binary', expression)

    def _set_arteriness(self):
        """'arteriness' from signal intensity"""
        artery_raw = self.workspace.filename('stitched', postfix='arteries')
        coordinates = self.graph_raw.vertex_coordinates()
        radii = self.graph_raw.vertex_radii()
        radii_measure = radii + 10
        expression = measure_expression.measure_expression(artery_raw, coordinates,
                                                           radii_measure, method='max')   # WARNING: prange
        self.graph_raw.define_vertex_property('artery_raw', np.asarray(expression.array, dtype=float))

    def _build_graph_from_skeleton(self):  # TODO: split for requirements
        if self.processing_config['graph_construction']['build']:
            n_blocks = 100  # TBD:
            self.prepare_watcher_for_substep(n_blocks, self.build_graph_re, f'Building graph', True)
            self.steps.remove_next_steps_files(self.steps.graph_raw)
            self.graph_raw = graph_processing.graph_from_skeleton(self.workspace.filename('skeleton'),
                                                                  verbose=True)  # WARNING: main thread (prange)
            # p3d.plot_graph_line(graph_raw)
            self._measure_radii()  # WARNING: main thread (prange)
            if self.use_arteries_for_graph:
                self._set_artery_binary()  # WARNING: main thread (prange)
                self._set_arteriness()  # WARNING: main thread (prange)
            self.save_graph('raw')

    @requires_files([FilePath('graph', postfix='raw')])
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
            })
        self.steps.remove_next_steps_files(self.steps.graph_cleaned)
        self.graph_cleaned = graph_processing.clean_graph(self.graph_raw, vertex_mappings=vertex_mappings, verbose=True)
        self.save_graph('cleaned')

    @requires_files([FilePath('graph', postfix='cleaned')])
    def __reduce_graph(self, vertex_to_edge_mappings=None, edge_to_edge_mappings=None):
        """
        Simplify straight segments between branches
        Returns
        -------

        """
        def vote(expression):
            return np.sum(expression) >= len(expression) / 1.5

        if vertex_to_edge_mappings is None:
            vertex_edge_mappings = {'radii': np.max}
        if edge_to_edge_mappings is None:
            edge_to_edge_mappings = {'length': np.sum}
        edge_geometry_vertex_properties = ['coordinates', 'radii']
        if self.use_arteries_for_graph:
            vertex_edge_mappings.update({
                'artery_binary': vote,
                'artery_raw': np.max})
            edge_geometry_vertex_properties.extend(['artery_binary', 'artery_raw'])
        self.steps.remove_next_steps_files(self.steps.graph_reduced)
        self.graph_reduced = graph_processing.reduce_graph(self.graph_cleaned, edge_length=True,
                                                           edge_to_edge_mappings=edge_to_edge_mappings,
                                                           vertex_to_edge_mappings=vertex_edge_mappings,
                                                           edge_geometry_vertex_properties=edge_geometry_vertex_properties,
                                                           edge_geometry_edge_properties=None,
                                                           return_maps=False, verbose=True)
        self.save_graph('reduced')

    # Atlas registration and annotation
    def _transform(self):
        def transformation(coordinates):
            coordinates = resampling_module.resample_points(
                coordinates, sink=None,
                source_shape=clearmap_io.shape(self.workspace.filename('binary', postfix='final')),
                sink_shape=clearmap_io.shape(self.workspace.filename('resampled')))

            if self.preprocessor.was_registered:
                coordinates = elastix.transform_points(
                    coordinates, sink=None,
                    transform_directory=self.workspace.filename('resampled_to_auto'),
                    binary=True, indices=False)

                coordinates = elastix.transform_points(
                    coordinates, sink=None,
                    transform_directory=self.workspace.filename('auto_to_reference'),
                    binary=True, indices=False)

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
                source_shape=clearmap_io.shape(self.workspace.filename('binary', postfix='final')),
                sink_shape=clearmap_io.shape(self.workspace.filename('resampled')))
            return radii * np.mean(resample_factor)

        self.graph_reduced.transform_properties(transformation=scaling,
                                                vertex_properties={'radii': 'radii_atlas'},
                                                edge_properties={'radii': 'radii_atlas'},
                                                edge_geometry_properties={'radii': 'radii_atlas'})

    def _annotate(self):
        """Atlas annotation of the graph (i.e. add property 'region' to vertices)"""
        annotation_module.set_annotation_file(self.preprocessor.annotation_file_path)

        def annotation(coordinates):
            label = annotation_module.label_points(coordinates,
                                                   annotation_file=self.preprocessor.annotation_file_path,
                                                   key='id')
            return label

        def annotation_hemisphere(coordinates):
            hemisphere_labels = annotation_module.label_points(coordinates,
                                                               annotation_file=self.preprocessor.hemispheres_file_path,
                                                               key='id')
            return hemisphere_labels

        self.graph_reduced.annotate_properties(annotation,
                                               vertex_properties={'coordinates_atlas': 'annotation'},
                                               edge_geometry_properties={'coordinates_atlas': 'annotation'})

        self.graph_reduced.annotate_properties(annotation_hemisphere,
                                               vertex_properties={'coordinates_atlas': 'hemisphere'},
                                               edge_geometry_properties={'coordinates_atlas': 'hemisphere'})

    def _compute_distance_to_surface(self):
        """add distance to brain surface as vertices properties"""
        # %% Distance to surface
        distance_atlas = clearmap_io.read(self.preprocessor.distance_file_path)
        distance_atlas_shape = distance_atlas.shape

        def distance(coordinates):
            c = np.round(coordinates).astype(int).clip(0, None)
            x, y, z = [c[:, i] for i in range(3)]
            x[x >= distance_atlas_shape[0]] = distance_atlas_shape[0] - 1
            y[y >= distance_atlas_shape[1]] = distance_atlas_shape[1] - 1
            z[z >= distance_atlas_shape[2]] = distance_atlas_shape[2] - 1
            d = distance_atlas[x, y, z]
            return d

        graph = self.graph_reduced
        graph.transform_properties(distance,
                                   vertex_properties={'coordinates_atlas': 'distance_to_surface'},
                                   edge_geometry_properties={'coordinates_atlas': 'distance_to_surface'})

        distance_to_surface = graph.edge_geometry('distance_to_surface', as_list=True)
        distance_to_surface_edge = np.array([np.min(d) for d in distance_to_surface])
        graph.define_edge_property('distance_to_surface', distance_to_surface_edge)
        self.save_graph('reduced')

    @requires_files([FilePath('graph', postfix='reduced')])
    def __register(self):
        if self.processing_config['graph_construction']['transform']:
            self._transform()
        self._scale()
        if self.preprocessor.was_registered and self.processing_config['graph_construction']['annotate']:
            self._annotate()
            self._compute_distance_to_surface()
        self.steps.remove_next_steps_files(self.steps.graph_annotated)

        # discard non connected graph components
        self.graph_annotated = self.graph_reduced.largest_component()
        self.save_graph('annotated')

    # POST PROCESS
    @requires_files([FilePath('graph', postfix='annotated')])
    def _pre_filter_veins(self, vein_intensity_range_on_arteries_channel, min_vein_radius):
        """
        Filter veins based on radius and intensity in arteries channel

        Parameters
        ----------
        vein_intensity_range_on_arteries_channel : (tuple)     Above max (second val) on artery channel, this is an artery
        min_vein_radius: (int)

        Returns
        -------

        """
        is_in_vein_range = is_in_range(self.graph_annotated.edge_property('artery_raw'),
                                       vein_intensity_range_on_arteries_channel)
        radii = self.graph_annotated.edge_property('radii')
        restrictive_vein = np.logical_and(radii >= min_vein_radius, is_in_vein_range)
        return restrictive_vein

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

    def _post_filter_veins(self, restrictive_veins, min_vein_radius=6.5):
        radii = self.graph_annotated.edge_property('radii')
        artery = self.graph_annotated.edge_property('artery')

        large_vessels = radii >= min_vein_radius
        permissive_veins = np.logical_and(np.logical_or(restrictive_veins, large_vessels), np.logical_not(artery))
        return permissive_veins

    # TRACING
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

            self.graph_annotated.save(self.workspace.filename('graph'))  # WARNING: no postfix
            self.graph_traced = self.graph_annotated

    def __get_branch_voxelization_params(self):
        voxelize_branch_parameter = {
            'method': 'sphere',
            'radius': tuple(self.processing_config['visualization']['voxelization']['size']),
            'weights': None,
            'shape': clearmap_io.shape(self.preprocessor.reference_file_path),
            'verbose': True
        }
        return voxelize_branch_parameter

    def __voxelize(self, vertices, voxelize_branch_parameter):
        clearmap_io.delete_file(self.workspace.filename('density', postfix='branches'))
        self.branch_density = voxelization.voxelize(vertices,
                                                    sink=self.workspace.filename('density', postfix='branches'),
                                                    dtype='float32',
                                                    **voxelize_branch_parameter)  # WARNING: prange

    def voxelize(self, weight_by_radius=False, vertex_degrees=None):
        vertices = self.graph_traced.vertex_property('coordinates_atlas')
        voxelize_branch_parameter = self.__get_branch_voxelization_params()

        if vertex_degrees and vertex_degrees >= 1:
            degrees = self.graph_traced.vertex_degrees() == vertex_degrees
            vertices = vertices[degrees]

        if weight_by_radius:
            voxelize_branch_parameter.update(weights=self.graph_traced.vertex_radii())

        self.__voxelize(vertices, voxelize_branch_parameter)

    def plot_voxelization(self, parent):
        return q_p3d.plot(self.workspace.filename('density', postfix='branches'),
                          arrange=False, parent=parent, lut=self.machine_config['default_lut'])  # FIXME: fire

    def get_structure_sub_graph(self, structure_id):
        vertex_labels = self.graph_traced.vertex_annotation()

        # Assign label of requested structure to all its children
        level = annotation_module.find(structure_id)['level']
        label_leveled = annotation_module.convert_label(vertex_labels, value='id', level=level)

        vertex_filter = label_leveled == structure_id
        # if get_neighbours:
        #     vertex_filter = graph.expand_vertex_filter(vertex_filter, steps=2)
        if vertex_filter is None:
            return
        return self.graph_traced.sub_graph(vertex_filter=vertex_filter)

    def plot_graph_structure(self, structure_id, plot_type):
        structure_name = annotation_module.get_names_map()[structure_id]
        graph_chunk = self.get_structure_sub_graph(structure_id)
        if not graph_chunk:
            return
        region_color = annotation_module.convert_label(graph_chunk.vertex_annotation(), key='id', value='rgba')
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
            raise ValueError(f'graph step {graph_step} not recognised, '
                             f'available steps are {self.steps.existing_steps}')
        graph_chunk = graph.sub_slice(chunk_range)
        title = f'{graph_step.title()} Graph'
        if graph_step == 'annotated':
            region_color = annotation_module.convert_label(graph_chunk.vertex_annotation(), key='id', value='rgba')
        else:
            region_color = None
        return self.plot_graph_chunk(graph_chunk, plot_type, title, region_color, show)

    def save_graph(self, base_name):
        graph = getattr(self, f'graph_{base_name}')
        graph.save(self.workspace.filename('graph', postfix=base_name))
