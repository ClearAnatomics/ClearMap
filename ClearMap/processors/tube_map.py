#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
TubeMap
=======

This module contains the classes to generate annotated graphs from vasculature
lightsheet data [Kirst2020]_.
"""
import copy
import os
import re

import numpy as np
import vispy


from ClearMap.processors.sample_preparation import TabProcessor

import ClearMap.IO.IO as clearmap_io

import ClearMap.Alignment.Annotation as annotation_module
import ClearMap.Alignment.Resampling as resampling_module
import ClearMap.Alignment.Elastix as elastix

import ClearMap.ImageProcessing.Experts.Vasculature as vasculature
import ClearMap.ImageProcessing.MachineLearning.VesselFilling.VesselFilling as vessel_filling
import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skeletonization

import ClearMap.Analysis.Measurements.MeasureExpression as measure_expression
import ClearMap.Analysis.Measurements.MeasureRadius as measure_radius
import ClearMap.Analysis.Graphs.GraphProcessing as graph_processing
import ClearMap.Analysis.Graphs.GraphGt as graph_gt
import ClearMap.Analysis.Measurements.Voxelization as voxelization

import ClearMap.ParallelProcessing.BlockProcessing as block_processing

from ClearMap.Visualization.Qt import Plot3d as q_p3d
from ClearMap.Visualization.Vispy import PlotGraph3d as plot_graph_3d  # WARNING: vispy dependency

from ClearMap.gui.dialogs import warning_popup
from ClearMap.Utils.utilities import is_in_range, get_free_v_ram


__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Sophie Skriabine <sophie.skriabine@icm-institute.org>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


class BinaryVesselProcessorSteps:
    def __init__(self, workspace):
        self.workspace = workspace
        self.stitched = 'stitched'
        self.binary = 'binary'
        self.postprocessed = 'postprocessed'
        self.filled = 'filled'
        self.combined = 'combined'
        self.final = 'final'

        self._last_step = self.stitched
        self.previous_step = None

    @property
    def last_step(self):
        return self._last_step

    @last_step.setter
    def last_step(self, step):
        self.previous_step = self.last_step
        self._last_step = step

    @property
    def steps(self):
        return self.stitched, self.binary, self.postprocessed, self.filled, self.combined, self.final

    def path(self, step, step_back=False, n_before=0, arteries=False):
        if n_before:
            step = self.steps[self.steps.index(step) - n_before]
        postfix = 'arteries' if arteries else ''
        if step in (self.stitched, self.binary):
            f_path = self.workspace.filename(step, postfix=postfix)
        else:
            postfix_base = ''
            if postfix and step in (self.postprocessed, self.filled):
                postfix_base = '{}_'.format(postfix)
            f_path = self.workspace.filename('binary', postfix='{}{}'.format(postfix_base, step))
        if not os.path.exists(f_path):
            if step_back:
                f_path = self.path(self.steps[self.steps.index(step) - 1])
            else:
                raise IndexError('Could not find path {} and not allowed to step back')
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
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.preprocessor = None
        self.workspace = None
        self.steps = BinaryVesselProcessorSteps(self.workspace)
        self.block_re = ('Processing block',
                         re.compile(r'.*?Processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))
        self.vessel_filling_re = ('Vessel filling',
                                  re.compile(r'.*?Vessel filling: processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))

        self.setup(preprocessor)

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

    def get_n_blocks(self, dim_size):
        blk_size = vasculature.default_binarization_processing_parameter['size_max']
        overlap = vasculature.default_binarization_processing_parameter['overlap']
        n_blocks = int(np.ceil((dim_size - blk_size) / (blk_size - overlap) + 1))
        return n_blocks

    def _binarize(self, n_processes, clip_range, deconvolve_threshold, postfix=''):
        """

        postfix str
            empty for raw
        """

        source = self.workspace.filename('stitched', postfix=postfix)
        sink = self.workspace.filename('binary', postfix=postfix)

        binarization_parameter = copy.deepcopy(vasculature.default_binarization_parameter)
        binarization_parameter['clip']['clip_range'] = clip_range
        if deconvolve_threshold is not None:
            binarization_parameter['deconvolve']['threshold'] = 450
            binarization_parameter.update(equalize=None, vesselize=None)

        processing_parameter = copy.deepcopy(vasculature.default_binarization_processing_parameter)
        processing_parameter.update(processes=n_processes, as_memory=True, verbose=True)

        vasculature.binarize(source, sink,
                             binarization_parameter=binarization_parameter,
                             processing_parameter=processing_parameter)

    def plot_binarization_result(self, parent=None, postfix=''):
        """
        postfix str:
            empty for raw
        """
        images = [(self.workspace.filename('stitched', postfix=postfix)),
                  (self.workspace.filename('binary', postfix=postfix))]
        dvs = q_p3d.plot(images, title=[os.path.basename(img) for img in images],
                         arange=False, lut=self.machine_config['default_lut'], parent=parent)
        return dvs

    def _smooth_and_fill(self, postfix=''):
        source = self.workspace.filename('binary', postfix=postfix)
        sink_postfix = '{}_postprocessed'.format(postfix) if postfix else 'postprocessed'
        sink = self.workspace.filename('binary', postfix=sink_postfix)

        postprocessing_parameter = copy.deepcopy(vasculature.default_postprocessing_parameter)
        if not postfix:  # FIXME: for both for Elisa
            postprocessing_parameter['fill'] = None  # Dilate erode
        postprocessing_processing_parameter = copy.deepcopy(vasculature.default_postprocessing_processing_parameter)
        postprocessing_processing_parameter.update(size_max=50)

        vasculature.postprocess(source, sink, postprocessing_parameter=postprocessing_parameter,
                                processing_parameter=postprocessing_processing_parameter,
                                processes=None, verbose=True)

        # q_p3d.plot([[source, sink]])  FIXME:

    def binarize(self):  # TODO: check real n blocks for post_processing
        # Raw
        binarization_cfg = self.processing_config['binarization']['binarization']
        n_blocks = self.get_n_blocks(self.workspace.source('stitched').shape[2])
        self.prepare_watcher_for_substep(n_blocks, self.block_re, 'Raw binarization', False)
        self._binarize(self.machine_config['n_processes_binarization'],
                       binarization_cfg['raw']['clip_range'],
                       binarization_cfg['raw']['threshold'])
        self.steps.last_step = self.steps.binary
        self.prepare_watcher_for_substep(n_blocks, self.block_re, 'Raw postprocessing', True)
        if binarization_cfg['raw']['post_process']:
            self._smooth_and_fill()
            self.steps.last_step = self.steps.postprocessed

        if not binarization_cfg['arteries']['skip']:
            n_blocks = self.get_n_blocks(self.workspace.source('stitched', postfix='arteries').shape[2])
            self.prepare_watcher_for_substep(n_blocks, self.block_re, 'Arteries binarization', True)
            self._binarize(self.machine_config['n_processes_binarization'],
                           binarization_cfg['arteries']['clip_range'],
                           binarization_cfg['arteries']['threshold'], postfix='arteries')
            self.prepare_watcher_for_substep(n_blocks, self.block_re, 'Arteries postprocessing', True)
            if binarization_cfg['arteries']['post_process']:
                self._smooth_and_fill('arteries')

    def _fill_vessels(self, size_max, overlap, postfix_base='', resample_factor=1):
        if postfix_base:
            postfix_base += '_'
        source = self.workspace.filename('binary', postfix='{}postprocessed'.format(postfix_base))
        sink = self.workspace.filename('binary', postfix='{}filled'.format(postfix_base))
        # clearmap_io.delete_file(sink)

        processing_parameter = copy.deepcopy(vessel_filling.default_fill_vessels_processing_parameter)
        processing_parameter.update(size_max=size_max,
                                    size_min='fixed',
                                    axes=all,
                                    overlap=overlap)

        vessel_filling.fill_vessels(source, sink, resample=resample_factor, threshold=0.5,
                                    cuda=True, processing_parameter=processing_parameter, verbose=True)

    def plot_vessel_filling_results(self, parent=None, postfix_base=''):
        if postfix_base:
            postfix_base += '_'
        images = [(self.steps.path(self.steps.postprocessed, step_back=True)),
                  (self.workspace.filename('binary', postfix='{}filled'.format(postfix_base)))]
        titles = [os.path.basename(img) for img in images]
        return q_p3d.plot(images, title=titles, arange=False,
                          lut=self.machine_config['default_lut'], parent=parent)

    def fill_vessels(self):
        if not get_free_v_ram() > 22000:
            warning_popup('Insufficient VRAM',
                          'You do not have enough free memory on your graphics card to'
                          'run this operation. This step needs 22GB VRAM, {} were found.'
                          'Please free some or upgrade your hardware.'.format(get_free_v_ram()))
            return
        if self.processing_config['binarization']['vessel_filling']['main']:
            self.prepare_watcher_for_substep(1200, self.vessel_filling_re, 'Filling main channel', True)  # FIXME: compute max
            self._fill_vessels(500, 50)  # FIXME: extact numbers
            self.steps.last_step = self.steps.filled
        if not self.processing_config['binarization']['binarization']['arteries']['skip'] and \
                self.processing_config['binarization']['vessel_filling']['secondary']:
            self.prepare_watcher_for_substep(1200, self.vessel_filling_re, 'Filling secondary channel', True)  # FIXME: compute max
            self._fill_vessels(1000, 100, 'arteries', resample_factor=2)   # FIXME: extact numbers

    def plot_combined(self, parent=None):  # TODO: final or not option
        raw = self.steps.path(self.steps.filled, step_back=True)
        combined = self.steps.path(self.steps.combined)
        if not self.processing_config['binarization']['binarization']['arteries']['skip']:
            arteries_filled = self.workspace.filename('binary', postfix='arteries_filled')
            dvs = q_p3d.plot([raw, arteries_filled, combined], title=['Raw', 'arteries', 'combined'],
                             arange=False, lut=self.machine_config['default_lut'], parent=parent)
        else:
            dvs = q_p3d.plot([raw, combined], title=['Raw', 'combined'],
                             arange=False, lut=self.machine_config['default_lut'], parent=parent)
        return dvs

    def combine_binary(self):
        # MERGE
        sink = self.workspace.filename('binary', postfix='combined')  # Temporary
        if not self.processing_config['binarization']['binarization']['arteries']['skip']:
            source = self.workspace.filename('binary', postfix='filled')
            source_arteries = self.workspace.filename('binary', postfix='arteries_filled')
            block_processing.process(np.logical_or, [source, source_arteries], sink,
                                     size_max=500, overlap=0, processes=None, verbose=True)
        else:
            source = self.steps.path(self.steps.filled, step_back=True)
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
        #     return q_p3d.plot([source, sink], arange=False, parent=parent)


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
        self.graph_cleaned = None
        self.graph_reduced = None
        self.annotated_graph = None
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.preprocessor = None
        self.workspace = None
        self.setup(preprocessor)

    def setup(self, preprocessor):
        self.preprocessor = preprocessor
        if preprocessor is not None:
            self.workspace = preprocessor.workspace
            configs = preprocessor.get_configs()
            self.sample_config = configs['sample']
            self.machine_config = configs['machine']
            self.processing_config = self.preprocessor.config_loader.get_cfg('vasculature')

            self.set_progress_watcher(self.preprocessor.progress_watcher)

    def run(self):
        self.pre_process()
        self.post_process()

    def pre_process(self):
        self.build_graph()  # WARNING: optional steps
        self.clean_graph()  # WARNING: optional steps
        self.reduce_graph()  # WARNING: optional steps
        self.register()

    def _measure_radii(self):
        coordinates = self.graph_raw.vertex_coordinates()
        radii, indices = measure_radius.measure_radius(self.workspace.filename('binary', postfix='final'),
                                                       coordinates,
                                                       value=0, fraction=None, max_radius=150,
                                                       return_indices=True, default=-1)
        self.graph_raw.set_vertex_radii(radii)

    def _set_artery_binary(self):
        """check if vertex is artery from binary labelling"""
        binary_arteries = self.workspace.filename('binary', postfix='arteries_filled')
        coordinates = self.graph_raw.vertex_coordinates()  # OPTIMISE: is this necessary again (set as attribute ?)
        radii = self.graph_raw.vertex_radii()
        expression = measure_expression.measure_expression(binary_arteries, coordinates, radii, method='max')
        self.graph_raw.define_vertex_property('artery_binary', expression)

    def _set_arteriness(self):
        """'arteriness' from signal intensity"""
        artery_raw = self.workspace.filename('stitched', postfix='arteries')
        coordinates = self.graph_raw.vertex_coordinates()
        radii = self.graph_raw.vertex_radii()
        radii_measure = radii + 10
        expression = measure_expression.measure_expression(artery_raw, coordinates,
                                                           radii_measure, method='max')
        self.graph_raw.define_vertex_property('artery_raw', np.asarray(expression.array, dtype=float))

    def build_graph(self):
        binary = self.workspace.filename('binary', postfix='final')
        skeleton = self.workspace.filename('skeleton')
        skeletonization.skeletonize(binary, sink=skeleton, delete_border=True, verbose=True)
        graph_raw = graph_processing.graph_from_skeleton(self.workspace.filename('skeleton'), verbose=True)
        self.graph_raw = graph_raw
        # p3d.plot_graph_line(graph_raw)
        self._measure_radii()
        if not self.processing_config['binarization']['binarization']['arteries']['skip']:
            self._set_artery_binary()
            self._set_arteriness()
        graph_raw.save(self.workspace.filename('graph', postfix='raw'))

    def clean_graph(self):
        """
        Remove spurious data e.g. outliers ...

        Returns graph_cleaned
        -------

        """
        vertex_mappings = {
            'coordinates': graph_processing.mean_vertex_coordinates,
            'radii': np.max
        }
        if not self.processing_config['binarization']['binarization']['arteries']['skip']:
            vertex_mappings.update({
                'artery_binary': np.max,
                'artery_raw': np.max
            })
        graph_cleaned = graph_processing.clean_graph(self.graph_raw, vertex_mappings=vertex_mappings, verbose=True)
        graph_cleaned.save(self.workspace.filename('graph', postfix='cleaned'))
        self.graph_cleaned = graph_cleaned  # OPTIMISE: check if necessary to have attribute or better to reload

    def reduce_graph(self):
        """
        Simplify straight segments between branches
        Returns
        -------

        """
        def vote(expression):
            return np.sum(expression) >= len(expression) / 1.5

        vertex_edge_mappings = {'radii': np.max}
        edge_geometry_vertex_properties = ['coordinates', 'radii']
        if not self.processing_config['binarization']['binarization']['arteries']['skip']:
            vertex_edge_mappings.update({
                'artery_binary': vote,
                'artery_raw': np.max})
            edge_geometry_vertex_properties.extend(['artery_binary', 'artery_raw'])
        graph_reduced = graph_processing.reduce_graph(self.graph_cleaned, edge_length=True,
                                                      edge_to_edge_mappings={'length': np.sum},
                                                      vertex_to_edge_mappings=vertex_edge_mappings,
                                                      edge_geometry_vertex_properties=edge_geometry_vertex_properties,
                                                      edge_geometry_edge_properties=None,
                                                      return_maps=False, verbose=True)
        graph_reduced.save(self.workspace.filename('graph', postfix='reduced'))
        self.graph_reduced = graph_reduced
        # graph_reduced = graph_gt.load(self.workspace.filename('graph', postfix='reduced'))

    def visualize_graph_annotations(self, chunk_range, plot_type='mesh', graph_step='reduced', show=True):
        graph_steps = self.get_graph_steps()
        try:
            graph_chunk = graph_steps[graph_step].sub_slice(chunk_range)
        except KeyError:
            raise ValueError('graph step {} not recognised, available steps are {}'
                             .format(graph_step, graph_steps.keys()))

        # region_label = self.graph_reduced.vertex_properties('annotation')
        # region_color = np.array([[1, 0, 0, 1], [0, 0, 1, 1]])[region_label]
        title = f'{graph_step.title()} Graph'
        if graph_step == 'annotated':
            region_color = annotation_module.convert_label(graph_chunk.vertex_annotation(), key='order', value='rgba')
        else:
            region_color = None
        if plot_type == 'line':
            scene = plot_graph_3d.plot_graph_line(graph_chunk, vertex_colors=region_color, title=title,
                                                  show=show, bg_color=self.machine_config['three_d_plot_bg'])
        elif plot_type == 'mesh':
            scene = plot_graph_3d.plot_graph_mesh(graph_chunk, vertex_colors=region_color, title=title,
                                                  show=show, bg_color=self.machine_config['three_d_plot_bg'])
        elif plot_type == 'edge_property':
            scene = plot_graph_3d.plot_graph_edge_property(graph_chunk, edge_property='artery_raw', title=title,
                                                           percentiles=[2, 98], normalize=True, mesh=True,
                                                           show=show, bg_color=self.machine_config['three_d_plot_bg'])
        else:
            raise ValueError(f'Unrecognised plot type  "{plot_type}"')
        # scene.canvas.bgcolor = vispy.color.color_array.Color(self.machine_config['three_d_plot_bg'])
        return [scene.canvas.native]

    def get_graph_steps(self):  # FIXME: make dynamic w/ lazy loading
        graph_steps = {
            'cleaned': self.graph_cleaned,
            'reduced': self.graph_reduced,
            'annotated': self.annotated_graph
        }
        for k, v in graph_steps.items():
            if v is None and self.workspace.exists('graph', postfix=k):
                graph_steps[k] = graph_gt.load(self.workspace.filename('graph', postfix=k))
        return graph_steps

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
            label = annotation_module.label_points(coordinates, key='order')
            return label

        self.graph_reduced.annotate_properties(annotation,
                                               vertex_properties={'coordinates_atlas': 'annotation'},
                                               edge_geometry_properties={'coordinates_atlas': 'annotation'})

    def _compute_distance_to_surface(self):
        """add distance to brain surface as vertices properties"""
        # %% Distance to surface
        distance_atlas = clearmap_io.read(self.preprocessor.distance_file_path)
        distance_atlas_shape = distance_atlas.shape

        def distance(coordinates):
            c = (np.asarray(np.round(coordinates), dtype=int)).clip(0, None)
            x, y, z = [c[:, i] for i in range(3)]
            x[x >= distance_atlas_shape[0]] = distance_atlas_shape[0] - 1
            y[y >= distance_atlas_shape[1]] = distance_atlas_shape[1] - 1
            z[z >= distance_atlas_shape[2]] = distance_atlas_shape[2] - 1
            d = distance_atlas[x, y, z]
            return d

        self.graph_reduced.transform_properties(distance,
                                                vertex_properties={'coordinates_atlas': 'distance_to_surface'},
                                                edge_geometry_properties={'coordinates_atlas': 'distance_to_surface'})

        distance_to_surface = self.graph_reduced.edge_geometry('distance_to_surface', as_list=True)
        distance_to_surface_edge = np.array([np.min(d) for d in distance_to_surface])
        self.graph_reduced.define_edge_property('distance_to_surface', distance_to_surface_edge)

    def register(self):
        self._transform()
        self._scale()
        if self.preprocessor.was_registered:
            self._annotate()
            self._compute_distance_to_surface()
        annotated_graph = self.graph_reduced.largest_component()  # TODO: explanation
        annotated_graph.save(self.workspace.filename('graph', postfix='annotated'))
        self.annotated_graph = annotated_graph

    # POST PROCESS
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
        if self.annotated_graph is None:  # FIXME: do similar loading wherever required
            self.annotated_graph = graph_gt.load(self.workspace.filename('graph', postfix='annotated'))
        is_in_vein_range = is_in_range(self.annotated_graph.edge_property('artery_raw'),
                                       vein_intensity_range_on_arteries_channel)
        radii = self.annotated_graph.edge_property('radii')
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
        artery = self.annotated_graph.edge_property('artery_binary')

        artery_graph = self.annotated_graph.sub_graph(edge_filter=artery, view=True)
        artery_graph_edge, edge_map = artery_graph.edge_graph(return_edge_map=True)
        artery_components, artery_size = artery_graph_edge.label_components(return_vertex_counts=True)
        too_small = edge_map[np.in1d(artery_components, np.where(artery_size < min_size)[0])]
        artery[too_small] = False
        artery[huge_vein] = False
        return artery

    def _post_filter_veins(self, restrictive_veins, min_vein_radius=6.5):
        radii = self.annotated_graph.edge_property('radii')
        artery = self.annotated_graph.edge_property('artery')

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
        artery = self.annotated_graph.edge_property('artery')
        condition_args = {
            'distance_to_surface': self.annotated_graph.edge_property('distance_to_surface'),
            'distance_threshold': 15,
            'vein': veins,
            'radii': self.annotated_graph.edge_property('radii'),
            'artery_trace_radius': 4,  # FIXME: param
            'artery_intensity': self.annotated_graph.edge_property('artery_raw'),
            'artery_intensity_min': 200  # FIXME: param
        }

        def continue_edge(graph, edge, **kwargs):
            if kwargs['distance_to_surface'][edge] < kwargs['distance_threshold'] or kwargs['vein'][edge]:
                return False
            else:
                return kwargs['radii'][edge] >= kwargs['artery_trace_radius'] and \
                       kwargs['artery_intensity'][edge] >= kwargs['artery_intensity_min']

        artery_traced = graph_processing.trace_edge_label(self.annotated_graph, artery,
                                                          condition=continue_edge, max_iterations=max_tracing_iterations,
                                                          **condition_args)
        # artery_traced = graph.edge_open_binary(graph.edge_close_binary(artery_traced, steps=1), steps=1)
        self.annotated_graph.define_edge_property('artery', artery_traced)

    def _trace_veins(self, max_tracing_iterations=5):
        """
        Trace veins by hysteresis thresholding - stop before arteries
        """
        min_distance_to_artery = 1

        artery = self.annotated_graph.edge_property('artery')
        radii = self.annotated_graph.edge_property('radii')
        condition_args = {
            'artery_expanded': self.annotated_graph.edge_dilate_binary(artery, steps=min_distance_to_artery),
            'radii': radii,
            'vein_trace_radius': 5  # FIXME: param
        }

        def continue_edge(graph, edge, **kwargs):
            if kwargs['artery_expanded'][edge]:
                return False
            else:
                return kwargs['radii'][edge] >= kwargs['vein_trace_radius']

        vein_traced = graph_processing.trace_edge_label(self.annotated_graph, self.annotated_graph.edge_property('vein'),
                                                        condition=continue_edge, max_iterations=max_tracing_iterations,
                                                        **condition_args)
        # vein_traced = graph.edge_open_binary(graph.edge_close_binary(vein_traced, steps=1), steps=1)

        self.annotated_graph.define_edge_property('vein', vein_traced)

    def _remove_small_vessel_components(self, vessel_name, min_vessel_size=30):
        """
        Filter out small components that will become capillaries
        """
        vessel = self.annotated_graph.edge_property(vessel_name)
        graph_vessel = self.annotated_graph.sub_graph(edge_filter=vessel, view=True)
        graph_vessel_edge, edge_map = graph_vessel.edge_graph(return_edge_map=True)

        vessel_components, vessel_size = graph_vessel_edge.label_components(return_vertex_counts=True)
        remove = edge_map[np.in1d(vessel_components, np.where(vessel_size < min_vessel_size)[0])]
        vessel[remove] = False

        self.annotated_graph.define_edge_property(vessel_name, vessel)

    def post_process(self):  # FIXME: progress
        """
        Iteratively refine arteries and veins based on one another

        Returns
        -------

        """
        if not self.processing_config['binarization']['binarization']['arteries']['skip']:
            cfg = self.processing_config['vessel_type_postprocessing']
            # Definitely a vein because too big
            restrictive_veins = self._pre_filter_veins(cfg['pre_filtering']['vein_intensity_range_on_arteries_ch'],
                                                       min_vein_radius=cfg['pre_filtering']['restrictive_vein_radius'])

            artery = self._pre_filter_arteries(restrictive_veins, min_size=cfg['pre_filtering']['arteries_min_radius'])
            self.annotated_graph.define_edge_property('artery', artery)

            # Not huge vein but not an artery so still a vein (with temporary radius for artery tracing)
            tmp_veins = self._post_filter_veins(restrictive_veins,
                                                min_vein_radius=cfg['pre_filtering']['permissive_vein_radius'])
            self._trace_arteries(tmp_veins, max_tracing_iterations=cfg['tracing']['max_arteries_iterations'])

            # The real vein size filtering
            vein = self._post_filter_veins(restrictive_veins, min_vein_radius=cfg['pre_filtering']['final_vein_radius'])
            self.annotated_graph.define_edge_property('vein', vein)

            self._trace_veins(max_tracing_iterations=cfg['tracing']['max_veins_iterations'])

            self._remove_small_vessel_components('artery', min_vessel_size=cfg['capillaries_removal']['min_artery_size'])
            self._remove_small_vessel_components('vein', min_vessel_size=cfg['capillaries_removal']['min_vein_size'])

            self.annotated_graph.save(self.workspace.filename('graph'))

    def voxelize(self):
        voxelize_branch_parameter = {
            "method": 'sphere',
            "radius": self.processing_config['voxelization']['size'],
            "weights": None,
            "shape": clearmap_io.shape(self.preprocessor.reference_file),
            "verbose": True
        }

        vertices = self.annotated_graph.vertex_coordinates()

        self.branch_density = voxelization.voxelize(vertices,
                                                    sink=self.workspace.filename('density', postfix='branches'),
                                                    dtype='float32',
                                                    **voxelize_branch_parameter)

    def plot_voxelization(self, parent):
        return q_p3d.plot(self.branch_density, arange=False, parent=parent)
