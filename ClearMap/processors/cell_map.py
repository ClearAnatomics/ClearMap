#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellMap
=======

This module contains the class to analyze (detect) individual cells,
e.g. to analyze immediate early gene expression data from iDISCO+ cleared tissue [Renier2016]_.


.. image:: ../Static/cell_abstract_2016.jpg
   :target: https://doi.org/10.1016/j.cell.2020.01.028
   :width: 300

.. figure:: ../Static/CellMap_pipeline.png

  iDISCO+ and ClearMap: A Pipeline for Cell Detection, Registration, and 
  Mapping in Intact Samples Using Light Sheet Microscopy.


References
----------
.. [Renier2016] `Mapping of brain activity by automated volume analysis of immediate early genes. Renier* N, Adams* EL, Kirst* C, Wu* Z, et al. Cell. 2016 165(7):1789-802 <https://doi.org/10.1016/j.cell.2016.05.007>`_
"""


import copy
import importlib
import os
import platform
import re
import warnings
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from PyQt5.QtGui import QColor
from matplotlib import pyplot as plt
import pyqtgraph as pg

# noinspection PyPep8Naming
import ClearMap.Alignment.Elastix as elastix
# noinspection PyPep8Naming
import ClearMap.IO.IO as clearmap_io
# noinspection PyPep8Naming
import ClearMap.Visualization.Plot3d as plot_3d
import ClearMap.Visualization.Qt.Plot3d as qplot_3d
# noinspection PyPep8Naming
import ClearMap.Alignment.Resampling as resampling
# noinspection PyPep8Naming
import ClearMap.ImageProcessing.Experts.Cells as cell_detection
# noinspection PyPep8Naming
import ClearMap.Analysis.Measurements.Voxelization as voxelization
# noinspection PyPep8Naming
import ClearMap.Alignment.Annotation as annotation
from ClearMap.processors.sample_preparation import PreProcessor
from ClearMap.processors.generic_tab_processor import TabProcessor
from ClearMap.Utils.utilities import runs_on_ui
from ClearMap.Visualization.Qt.widgets import Scatter3D

__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

if platform.system().lower().startswith('darwin'):
    USE_BINARY_POINTS_FILE = False
else:
    USE_BINARY_POINTS_FILE = True


class CellDetector(TabProcessor):
    def __init__(self, preprocessor=None):
        super().__init__()
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.preprocessor = None
        self.workspace = None
        self.cell_detection_re = ('Processing block',
                                  re.compile(r'.*?Processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))
        self.setup(preprocessor)

    def setup(self, preprocessor):
        self.preprocessor = preprocessor
        if preprocessor is not None:
            self.workspace = preprocessor.workspace
            configs = preprocessor.get_configs()
            self.sample_config = configs['sample']
            self.machine_config = configs['machine']
            self.processing_config = self.preprocessor.config_loader.get_cfg('cell_map')

            self.set_progress_watcher(self.preprocessor.progress_watcher)

    @property
    def detected(self):
        return os.path.exists(self.workspace.filename('cells', postfix='raw'))

    def run(self):
        # select sub-slice for testing the pipeline
        slicing = (
            slice(*self.processing_config['test_set_slicing']['dim_0']),
            slice(*self.processing_config['test_set_slicing']['dim_1']),
            slice(*self.processing_config['test_set_slicing']['dim_2'])
        )
        self.create_test_dataset(slicing)
        self.run_cell_detection(tuning=True)
        if self.processing_config['detection']['preview']:
            self.preview_cell_detection()

        self.run_cell_detection()
        # print(f"Number of cells detected: {self.get_n_detected_cells()}")

        self.post_process_cells()

    def post_process_cells(self):
        self.processing_config.reload()
        if self.processing_config['detection']['plot_cells'] and not runs_on_ui():
            self.plot_cells()
        self.filter_cells()
        if self.processing_config['cell_filtration']['preview'] and not runs_on_ui():
            self.plot_filtered_cells()
        self.atlas_align()
        self.export_collapsed_stats()

    def voxelize(self, postfix=''):
        self.processing_config.reload()
        coordinates, cells, voxelization_parameter = self.get_voxelization_params(postfix=postfix)
        # %% Unweighted
        coordinates, counts_file_path = self.voxelize_unweighted(coordinates, voxelization_parameter)
        if self.processing_config['voxelization']['preview']['counts'] and not runs_on_ui():
            self.plot_voxelized_counts()
        # %% Weighted
        # intensities_file_path = self.voxelize_weighted(coordinates, cells, voxelization_parameter)  # WARNING: Currently causing issues
        # if self.processing_config['voxelization']['preview']['densities']:
        #     self.plot_voxelized_intensities()

    def plot_voxelized_counts(self, arrange=True, parent=None):
        scale = self.preprocessor.processing_config['registration']['resampling']['raw_sink_resolution']
        return plot_3d.plot(self.workspace.filename('density', postfix='counts'),
                            scale=scale, title='Cell density (voxelized)', lut='flame',
                            arrange=arrange, parent=parent)

    def create_test_dataset(self, slicing):
        self.workspace.create_debug('stitched', slicing=slicing)
        self.update_watcher_main_progress()

    def get_voxelization_params(self, postfix=''):
        voxelization_parameter = {
            'radius': self.processing_config['voxelization']['radii'],
            'verbose': True
        }
        if self.workspace.debug:  # Path will use debug
            voxelization_parameter['shape'] = clearmap_io.shape(self.workspace.filename('cells', postfix='shape'))
        elif self.preprocessor.was_registered:
            voxelization_parameter['shape'] = clearmap_io.shape(self.preprocessor.annotation_file_path)
        else:
            voxelization_parameter['shape'] = self.preprocessor.resampled_shape
        if postfix:  # Hack to compensate for the fact that the realigned makes no sense in
            cells, coordinates = self.get_coords(coord_type=postfix, aligned=False)
        else:
            cells, coordinates = self.get_coords(coord_type=None, aligned=True)
        return coordinates, cells, voxelization_parameter

    def get_coords(self, coord_type='filtered', aligned=False):
        if coord_type not in ('filtered', 'raw', None):
            raise ValueError(f'Coordinate type "{coord_type}" not recognised')
        if coord_type is None:
            dataframe_path = self.workspace.filename('cells', extension='.feather')
            if os.path.exists(dataframe_path):
                table = pd.read_feather(dataframe_path)
            else:
                table = np.load(self.workspace.filename('cells')).T
        else:
            table = np.load(self.workspace.filename('cells', postfix=coord_type))
        if aligned:
            coordinates = np.array([table[axis] for axis in ['xt', 'yt', 'zt']]).T
        else:
            coordinates = np.array([table[axis] for axis in ['x', 'y', 'z']]).T
        return table, coordinates

    def voxelize_unweighted(self, coordinates, voxelization_parameter):
        """
        Voxelize un weighted i.e. for cell counts

        Parameters
        ----------
        coordinates
            str, array or Source
            Source of point of nxd coordinates.

        voxelization_parameter
            dict

        Returns
        -------

        """
        counts_file_path = self.workspace.filename('density', postfix='counts')  # TODO: improve var name
        clearmap_io.delete_file(counts_file_path)
        self.set_watcher_step('Unweighted voxelisation')
        voxelization.voxelize(coordinates, sink=counts_file_path, **voxelization_parameter)  # WARNING: prange
        self.update_watcher_main_progress()
        # self.remove_crust(coordinates, voxelization_parameter)  # WARNING: currently causing issues
        return coordinates, counts_file_path

    def voxelize_weighted(self, coordinates, source, voxelization_parameter):
        """
        Voxelize weighted i.e. for cell intensities

        Parameters
        ----------
        coordinates
            np.array
        source
            Source.Source
        voxelization_parameter
            dict

        Returns
        -------

        """
        intensities_file_path = self.workspace.filename('density', postfix='intensities')
        intensities = source['source']
        voxelization.voxelize(coordinates, sink=intensities_file_path, weights=intensities, **voxelization_parameter)   # WARNING: prange
        return intensities_file_path

    def atlas_align(self):
        """Atlas alignment and annotation """
        table, coordinates = self.get_coords(coord_type='filtered')
        df = pd.DataFrame({'x': coordinates[:, 0], 'y': coordinates[:, 1], 'z': coordinates[:, 2]})
        df['size'] = table['size']
        df['source'] = table['source']

        if self.preprocessor.was_registered:
            coordinates_transformed = self.transform_coordinates(coordinates)
            df['xt'] = coordinates_transformed[:, 0]
            df['yt'] = coordinates_transformed[:, 1]
            df['zt'] = coordinates_transformed[:, 2]

            structure_ids = annotation.label_points(coordinates_transformed,
                                                    annotation_file=self.preprocessor.annotation_file_path,
                                                    key='id')
            df['id'] = structure_ids

            hemisphere_labels = annotation.label_points(coordinates_transformed,
                                                        annotation_file=self.preprocessor.hemispheres_file_path,
                                                        key='id')
            df['hemisphere'] = hemisphere_labels

            names = annotation.convert_label(structure_ids, key='id', value='name')
            df['name'] = names

            unique_ids = np.sort(df['id'].unique())

            order_map = {id_: annotation.find(id_, key='id')['order'] for id_ in unique_ids}
            df['order'] = df['id'].map(order_map)

            color_map = {id_: annotation.find(id_, key='id')['rgb'] for id_ in unique_ids}  # WARNING RGB upper case should give integer but does not work
            df['color'] = df['id'].map(color_map)

            volumes = annotation.annotation.get_lateralised_volume_map(
                self.preprocessor.processing_config['registration']['resampling']['autofluo_sink_resolution'],
                self.preprocessor.hemispheres_file_path
            )
            df['volume'] = df.set_index(['id', 'hemisphere']).index.map(volumes.get)

        df.to_feather(self.workspace.filename('cells', extension='.feather'))

    def transform_coordinates(self, coords):
        coords = resampling.resample_points(
            coords, sink=None,
            source_shape=self.preprocessor.raw_stitched_shape,
            sink_shape=self.preprocessor.resampled_shape)

        if self.preprocessor.was_registered:
            coords = elastix.transform_points(
                coords, sink=None,
                transform_directory=self.workspace.filename('resampled_to_auto'),
                binary=USE_BINARY_POINTS_FILE, indices=False)

            coords = elastix.transform_points(
                coords, sink=None,
                transform_directory=self.workspace.filename('auto_to_reference'),
                binary=USE_BINARY_POINTS_FILE, indices=False)

        return coords

    def filter_cells(self):
        self.processing_config.reload()
        thresholds = {
            'source': None,
            'size': self.processing_config['cell_filtration']['thresholds']['size']
        }
        if self.processing_config['cell_filtration']['thresholds']['intensity'] is not None:
            thresholds['source'] = self.processing_config['cell_filtration']['thresholds']['intensity']
        cell_detection.filter_cells(source=self.workspace.filename('cells', postfix='raw'),
                                    sink=self.workspace.filename('cells', postfix='filtered'),
                                    thresholds=thresholds)

    def run_cell_detection(self, tuning=False):
        self.processing_config.reload()
        self.workspace.debug = tuning  # TODO: use context manager
        cell_detection_param = copy.deepcopy(cell_detection.default_cell_detection_parameter)
        cell_detection_param['illumination_correction'] = None  # WARNING: illumination or illumination_correction
        cell_detection_param['background_correction']['shape'] = self.processing_config['detection']['background_correction']['diameter']
        cell_detection_param['maxima_detection']['shape'] = self.processing_config['detection']['maxima_detection']['shape']
        cell_detection_param['intensity_detection']['measure'] = ['source']
        cell_detection_param['shape_detection']['threshold'] = self.processing_config['detection']['shape_detection']['threshold']
        if tuning:
            clearmap_io.delete_file(self.workspace.filename('cells', postfix='bkg'))
            cell_detection_param['background_correction']['save'] = self.workspace.filename('cells', postfix='bkg')
            clearmap_io.delete_file(self.workspace.filename('cells', postfix='shape'))
            cell_detection_param['shape_detection']['save'] = self.workspace.filename('cells', postfix='shape')

            # clearmap_io.delete_file(workspace.filename('cells', postfix='maxima'))
            # cell_detection_param['maxima_detection']['save'] = workspace.filename('cells', postfix='maxima')

        processing_parameter = copy.deepcopy(cell_detection.default_cell_detection_processing_parameter)
        processing_parameter.update(  # TODO: store as other dict and run .update(**self.extra_detection_params)
            processes=self.machine_config['n_processes_cell_detection'],
            size_min=self.machine_config['detection_chunk_size_min'],
            size_max=self.machine_config['detection_chunk_size_max'],
            overlap=self.machine_config['detection_chunk_overlap'],
            verbose=True
        )

        n_steps = self.get_n_blocks(self.workspace.source('stitched').shape[2])  # OPTIMISE: read metadata w/out load  # TODO: round to processors
        self.prepare_watcher_for_substep(n_steps, self.cell_detection_re, 'Detecting cells')
        try:
            cell_detection.detect_cells(self.workspace.filename('stitched'),
                                        self.workspace.filename('cells', postfix='raw'),
                                        cell_detection_parameter=cell_detection_param,
                                        processing_parameter=processing_parameter,
                                        workspace=self.workspace)  # WARNING: prange inside multiprocess (including arrayprocessing and devolvepoints for vox)
        except BrokenProcessPool as err:
            print('Cell detection canceled')
            return
        finally:
            self.workspace.debug = False
            self.update_watcher_main_progress()

    def export_as_csv(self):
        """
        Export the cell coordinates to csv

        .. deprecated:: 2.1
            Use :func:`atlas_align` and `export_collapsed_stats` instead.
        """
        warnings.warn("export_as_csv is deprecated and will be removed in future versions;"
                      "please use the new formats from atlas_align and export_collapsed_stats", DeprecationWarning, 2)

        csv_file_path = self.workspace.filename('cells', extension='.csv')
        self.get_cells_df().to_csv(csv_file_path)

    def export_collapsed_stats(self, all_regions=True):
        df = self.get_cells_df()

        collapsed = pd.DataFrame()
        relevant_columns = ['id', 'order', 'name', 'hemisphere', 'volume', 'size']
        for i in (0, 255):  # Split by hemisphere to group by structure and reconcatenate hemispheres after
            grouped = df[df['hemisphere'] == i][relevant_columns].groupby(['id'], as_index=False)

            tmp = pd.DataFrame()
            tmp['Structure ID'] = grouped['id'].first()
            tmp['Structure order'] = grouped['order'].first()
            tmp['Structure name'] = grouped['name'].first()
            tmp['Hemisphere'] = grouped['hemisphere'].first()
            tmp['Structure volume'] = grouped['volume'].first()
            tmp['Cell counts'] = grouped['name'].count()
            tmp['Average cell size'] = grouped['size'].mean()

            collapsed = pd.concat((collapsed, tmp))

        if all_regions:  # Add regions even if they are empty
            uniq_ids = np.unique(annotation.annotation.atlas)
            tmp = pd.DataFrame({'Structure ID': uniq_ids, 'mock': ''})
            tmp['Structure name'] = annotation.convert_label(uniq_ids, key='id', value='name')
            df_mock = pd.DataFrame({'Hemisphere': [0, 255], 'mock': ''})
            tmp = tmp.merge(df_mock, on='mock').drop(columns='mock')
            vol_map = annotation.annotation.get_lateralised_volume_map(
                self.preprocessor.processing_config['registration']['resampling']['autofluo_sink_resolution'],
                self.preprocessor.hemispheres_file_path
            )
            tmp['Structure volume'] = tmp.set_index(['Structure ID', 'Hemisphere']).index.map(vol_map.get)
            order_map = {id_: annotation.find(id_, key='id')['order'] for id_ in uniq_ids}
            tmp['Structure order'] = tmp['Structure ID'].map(order_map)
            collapsed = tmp.merge(collapsed[['Structure ID', 'Hemisphere', 'Cell counts', 'Average cell size']],
                                  how='left', on=['Structure ID', 'Hemisphere'])

        collapsed = collapsed.sort_values(by='Structure ID')

        csv_file_path = self.workspace.filename('cells', postfix='stats', extension='.csv')
        collapsed.to_csv(csv_file_path, index=False)

    def plot_cells(self):  # For non GUI
        source = self.workspace.source('cells', postfix='raw')
        plt.figure(1)
        plt.clf()
        names = source.dtype.names
        nx, ny = plot_3d.subplot_tiling(len(names))
        for i, name in enumerate(names):
            plt.subplot(nx, ny, i + 1)
            plt.hist(source[name])
            plt.title(name)
        plt.tight_layout()

    def plot_cells_3d_scatter_w_atlas_colors(self, raw=False, parent=None):
        if raw:
            dv = qplot_3d.plot(self.workspace.filename('stitched'), title='Stitched and cells',
                               # scale=self.preprocessor.sample_config['resolutions']['raw'],# FIXME: correct scaling for anisotropic
                               arrange=False, lut='white', parent=parent)[0]
        else:
            if self.preprocessor.was_registered:  # REFACTORING: could extract
                dv = qplot_3d.plot(clearmap_io.source(self.preprocessor.reference_file_path),
                                   title='Reference and cells',
                                   arrange=False, lut='white', parent=parent)[0]
            else:
                dv = qplot_3d.plot(self.workspace.filename('resampled'), title='Resampled and cells',
                                   arrange=False, lut='white', parent=parent)[0]

        scatter = pg.ScatterPlotItem()

        dv.view.addItem(scatter)
        dv.scatter = scatter

        df = self.get_cells_df()

        if raw:
            coordinates = df[['x', 'y', 'z']].values.astype(int)
            # coordinates = coordinates * np.array(self.preprocessor.sample_config['resolutions']['raw'])
            # coordinates = coordinates.astype(int)  # required to match integer z  # FIXME: correct scaling for anisotropic
        else:
            coordinates = df[['xt', 'yt', 'zt']].values.astype(int)  # required to match integer z
            dv.atlas = clearmap_io.read(self.preprocessor.annotation_file_path)
            dv.structure_names = annotation.get_names_map()
        if 'hemisphere' in df.columns:
            hemispheres = df['hemisphere']
        else:
            hemispheres = None
        dv.scatter_coords = Scatter3D(coordinates, colors=df['color'].values,
                                      hemispheres=hemispheres, half_slice_thickness=0)
        dv.refresh()
        return [dv]

    @property
    def df_path(self):
        feather_path = self.workspace.filename('cells', extension='.feather')
        if os.path.exists:
            return feather_path
        else:
            return self.workspace.filename('cells')

    def get_cells_df(self):
        if self.df_path.endswith('.feather'):
            return pd.read_feather(self.df_path)
        else:
            return pd.DataFrame(np.load(self.df_path))

    def plot_filtered_cells(self, parent=None, smarties=False):
        _, coordinates = self.get_coords('filtered')
        stitched_path = self.workspace.filename('stitched')
        dv = qplot_3d.plot(stitched_path, title='Stitched and filtered cells', arrange=False,
                           lut='white', parent=parent)[0]
        scatter = pg.ScatterPlotItem()

        dv.view.addItem(scatter)
        dv.scatter = scatter

        dv.scatter_coords = Scatter3D(coordinates, smarties=smarties, half_slice_thickness=3)
        dv.refresh()
        return [dv]

    def plot_background_subtracted_img(self):
        coordinates = np.hstack([self.workspace.source('cells', postfix='raw')[c][:, None] for c in 'xyz'])
        p = plot_3d.list_plot_3d(coordinates)
        return plot_3d.plot_3d(self.workspace.filename('stitched'), view=p, cmap=plot_3d.grays_alpha(alpha=1))

    def remove_crust(self, coordinates,voxelization_parameter):
        dist2surf = clearmap_io.read(self.preprocessor.distance_file_path)
        threshold = 3
        shape = dist2surf.shape

        good_coordinates = np.logical_and(np.logical_and(coordinates[:, 0] < shape[0],
                                                         coordinates[:, 1] < shape[1]),
                                          coordinates[:, 2] < shape[2]).nonzero()[0]
        coordinates = coordinates[good_coordinates]
        coordinates_wcrust = coordinates[np.asarray(
            [dist2surf[tuple(np.floor(coordinates[i]).astype(int))] > threshold for i in
             range(coordinates.shape[0])]).nonzero()[0]]

        voxelization.voxelize(coordinates_wcrust, sink=self.workspace.filename('density', postfix='counts_wcrust'),
                              **voxelization_parameter)   # WARNING: prange

    def preview_cell_detection(self, parent=None, arrange=True, sync=True):
        sources = [self.workspace.filename('stitched'),
                   self.workspace.filename('cells', postfix='bkg'),
                   self.workspace.filename('cells', postfix='shape')
                   ]
        sources = [s for s in sources if os.path.exists(s)]  # Remove missing files (if not tuning)
        titles = [os.path.basename(s) for s in sources]
        luts = ['white', 'white', 'random']
        return plot_3d.plot(sources, title=titles, arrange=arrange, sync=sync, lut=luts, parent=parent)

    def get_n_detected_cells(self):
        if os.path.exists(self.workspace.filename('cells', postfix='raw')):
            _, coords = self.get_coords(coord_type='raw')
            return np.max(coords.shape)  # TODO: check dimension instead
        else:
            return 0

    def get_n_filtered_cells(self):
        if os.path.exists(self.workspace.filename('cells', postfix='filtered')):
            _, coords = self.get_coords(coord_type='filtered')
            return np.max(coords.shape)  # TODO: check dimension instead
        else:
            return 0

    def plot_voxelized_intensities(self, arrange=True):
        return plot_3d.plot(self.workspace.filename('density', postfix='intensities'), arrange=arrange)

    def get_n_blocks(self, dim_size):
        blk_size = self.machine_config['detection_chunk_size_max']
        overlap = self.machine_config['detection_chunk_overlap']
        n_blocks = int(np.ceil((dim_size - blk_size) / (blk_size - overlap) + 1))
        return n_blocks

    def export_to_clearmap1_fmt(self):
        """
        ClearMap 1.0 export (will generate the files cells_ClearMap1_intensities, cells_ClearMap1_points_transformed,
        cells_ClearMap1_points necessaries to use the analysis script of ClearMap1.
        In ClearMap2 the 'cells' file contains already all this information)
        In order to align the coordinates when we have right and left hemispheres,
        if the orientation of the brain is left, will calculate the new coordinates for the Y axes,
        this change will not affect the orientation of the heatmaps, since these are generated from
         the ClearMap2 file 'cells'

        .. deprecated:: 2.1
            Use :func:`atlas_align` and `export_collapsed_stats` instead.
        """
        warnings.warn("export_to_clearmap1_fmt is deprecated and will be removed in future versions;"
                      "please use the new formats from atlas_align and export_collapsed_stats", DeprecationWarning, 2)
        source = self.workspace.source('cells')
        clearmap1_format = {'points': ['x', 'y', 'z'],
                            'points_transformed': ['xt', 'yt', 'zt'],
                            'intensities': ['source', 'dog', 'background', 'size']}
        for filename, names in clearmap1_format.items():
            sink = self.workspace.filename('cells', postfix=['ClearMap1', filename])
            print(filename, sink)
            data = np.array(
                [source[name] if name in source.dtype.names else np.full(source.shape[0], np.nan) for name in names]
            )
            data = data.T
            clearmap_io.write(sink, data)

    def convert_cm2_to_cm2_1_fmt(self):
        """Atlas alignment and annotation """
        cells = np.load(self.workspace.filename('cells'))
        df = pd.DataFrame({ax: cells[ax] for ax in 'xyz'})
        df['size'] = cells['size']
        df['source'] = cells['source']
        for ax in 'xyz':
            df[f'{ax}t'] = cells[f'{ax}t']
        df['order'] = cells['order']
        df['name'] = cells['name']

        coordinates_transformed = np.vstack([cells[f'{ax}t'] for ax in 'xyz']).T

        # FIXME: Put key ID and get ID directly
        hemisphere_label = annotation.label_points(coordinates_transformed,
                                                   annotation_file=self.preprocessor.hemispheres_file_path,
                                                   key='id')
        unique_labels = np.sort(df['order'].unique())
        color_map = {lbl: annotation.find(lbl, key='order')['rgb'] for lbl in
                     unique_labels}  # WARNING RGB upper case should give integer but does not work
        id_map = {lbl: annotation.find(lbl, key='order')['id'] for lbl in unique_labels}

        atlas = clearmap_io.read(self.preprocessor.annotation_file_path)
        atlas_scale = self.preprocessor.processing_config['registration']['resampling']['autofluo_sink_resolution']
        atlas_scale = np.prod(atlas_scale)
        volumes = {_id: (atlas == _id).sum() * atlas_scale for _id in
                   id_map.values()}  # Volumes need a lookup on ID since the atlas is in ID space

        df['id'] = df['order'].map(id_map)
        df['hemisphere'] = hemisphere_label
        df['color'] = df['order'].map(color_map)
        df['volume'] = df['id'].map(volumes)

        df.to_feather(self.workspace.filename('cells', extension='.feather'))


if __name__ == "__main__":
    import sys
    preprocessor = PreProcessor()
    preprocessor.setup(sys.argv[1:3])
    preprocessor.setup_atlases()
    # preprocessor.run()

    detector = CellDetector(preprocessor)
