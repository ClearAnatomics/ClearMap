#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellMap
=======

This script is the main pipeline to analyze immediate early gene expression 
data from iDISCO+ cleared tissue [Renier2016]_.

See the :ref:`CellMap tutorial </CellMap.ipynb>` for a tutorial and usage.


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
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

import os

import numpy as np
from numpy.lib import recfunctions
from matplotlib import pyplot as plt

# noinspection PyPep8Naming
import ClearMap.Alignment.Elastix as elastix
# noinspection PyPep8Naming
import ClearMap.IO.IO as clearmap_io
# noinspection PyPep8Naming
import ClearMap.Visualization.Plot3d as plot_3d
# noinspection PyPep8Naming
import ClearMap.Alignment.Resampling as resampling
# noinspection PyPep8Naming
import ClearMap.ImageProcessing.Experts.Cells as cell_detection
# noinspection PyPep8Naming
import ClearMap.Analysis.Measurements.Voxelization as voxelization
# noinspection PyPep8Naming
import ClearMap.Alignment.Annotation as annotation
from ClearMap.Scripts.sample_preparation import PreProcessor
from ClearMap.config.config_loader import get_cfg


class CellDetector(object):
    def __init__(self, preprocessor=None):
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.preprocessor = None
        self.workspace = None
        self.annotation_file = None
        self.distance_file = None
        self.setup(preprocessor)

    def setup(self, preprocessor):
        self.preprocessor = preprocessor
        if preprocessor is not None:
            self.workspace = preprocessor.workspace
            atlas_files = preprocessor.get_atlas_files()
            self.annotation_file = atlas_files['annotation']
            self.distance_file = atlas_files['distance']
            configs = preprocessor.get_configs()
            self.sample_config = configs['sample']
            self.machine_config = configs['machine']
            self.processing_config = get_cfg(os.path.join(self.sample_config['base_directory'], 'cell_map_params.cfg'))

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
        # print("Number of cells detected: {}".format(self.get_n_detected_cells()))

        self.post_process_cells()

    def post_process_cells(self):
        if self.processing_config['detection']['plot_cells']:
            self.plot_cells()
        self.filter_cells()
        if self.processing_config['cell_filtration']['preview']:
            self.plot_filtered_cells()
        self.atlas_align()
        self.export_as_csv()
        self.export_to_clearmap1_fmt()
        self.voxelize()

    def voxelize(self, postfix=''):
        coordinates, source, voxelization_parameter = self.get_voxelization_params(postfix=postfix)
        # %% Unweighted
        coordinates, counts_file_path = self.voxelize_unweighted(coordinates, source, voxelization_parameter)
        if self.processing_config['voxelization']['preview']['counts']:
            self.plot_voxelized_counts()
        # %% Weighted
        intensities_file_path = self.voxelize_weighted(coordinates, source, voxelization_parameter)
        if self.processing_config['voxelization']['preview']['densities']:
            self.plot_voxelized_intensities()

    def plot_voxelized_counts(self, arange=True, parent=None):
        return plot_3d.plot(self.workspace.filename('density', postfix='counts'), arange=arange, parent=parent)

    def create_test_dataset(self, slicing):
        self.workspace.create_debug('stitched', slicing=slicing)

    def get_voxelization_params(self, postfix=''):
        voxelization_parameter = {
            'shape': clearmap_io.shape(self.annotation_file),
            'radius': self.processing_config['voxelization']['radii'],
            'verbose': True
        }
        if postfix:
            source = self.workspace.source('cells', postfix=postfix)  # Hack to compensate for the fact that the realigned makes no sense in
            coordinates = np.array([source[axis] for axis in 'xyz']).T
        else:
            source = self.workspace.source('cells')
            coordinates = np.array([source[n] for n in ['xt', 'yt', 'zt']]).T
        return coordinates, source, voxelization_parameter

    # def voxelize_chunk(self):
    #     self.workspace.debug = True
    #     coordinates = self.get_coords()
    #
    #     shape = clearmap_io.shape(self.annotation_file),
    #     # %% Unweighted
    #     counts_file_path = self.workspace.filename('density', postfix='counts_upperlayers')
    #     clearmap_io.delete_file(self.workspace.filename('density', postfix='counts'))  # WARNING: deletes different file
    #     voxelization.voxelize(coordinates, sink=counts_file_path, radius=(2, 2, 2), shape=shape)
    #     self.workspace.debug = False

    def get_coords(self, coord_type='filtered'):
        if not coord_type in ('filtered', 'raw'):
            raise ValueError('Coordinate type "{}" not recognised'.format(coord_type))
        table = np.load(self.workspace.filename('cells', postfix=coord_type))
        coordinates = np.array([table[axis] for axis in ['x', 'y', 'z']]).T
        return coordinates

    def voxelize_unweighted(self, coordinates, source, voxelization_parameter):
        """
        Voxelize un weighted i.e. for cell counts

        Parameters
        ----------
        source
            Source.Source
        voxelization_parameter
            dict

        Returns
        -------

        """
        counts_file_path = self.workspace.filename('density', postfix='counts')  # TODO: improve var name
        clearmap_io.delete_file(counts_file_path)
        voxelization.voxelize(coordinates, sink=counts_file_path, **voxelization_parameter)
        self.remove_crust(coordinates, voxelization_parameter)
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
        voxelization.voxelize(coordinates, sink=intensities_file_path, weights=intensities, **voxelization_parameter)
        return intensities_file_path

    def atlas_align(self):
        """Atlas alignment and annotation """
        # Cell alignment
        source = self.workspace.source('cells', postfix='filtered')
        coordinates = np.array([source[c] for c in 'xyz']).T
        coordinates_transformed = self.transform_coordinates(coordinates)
        # Cell annotation
        label = annotation.label_points(coordinates_transformed, key='order')
        names = annotation.convert_label(label, key='order', value='name')
        # Save results
        coordinates_transformed.dtype = [(t, float) for t in ('xt', 'yt', 'zt')]
        label = np.array(label, dtype=[('order', int)])
        names = np.array(names, dtype=[('name', 'U256')])
        cells_data = recfunctions.merge_arrays([source[:], coordinates_transformed, label, names],
                                               flatten=True, usemask=False)
        clearmap_io.write(self.workspace.filename('cells'), cells_data)

    def transform_coordinates(self, coords):
        coords = resampling.resample_points(
            coords, sink=None, orientation=None,
            source_shape=clearmap_io.shape(self.workspace.filename('stitched')),
            sink_shape=clearmap_io.shape(self.workspace.filename('resampled')))

        coords = elastix.transform_points(
            coords, sink=None,
            transform_directory=self.workspace.filename('resampled_to_auto'),
            binary=True, indices=False)

        coords = elastix.transform_points(
            coords, sink=None,
            transform_directory=self.workspace.filename('auto_to_reference'),
            binary=True, indices=False)

        return coords

    def filter_cells(self):
        cell_detection.filter_cells(source=self.workspace.filename('cells', postfix='raw'),
                                    sink=self.workspace.filename('cells', postfix='filtered'),
                                    thresholds={
                                        'source': None,
                                        'size': self.processing_config['cell_filtration']['thresholds']['size']
                                    })

    def run_cell_detection(self, tuning=False):
        self.workspace.debug = tuning
        self.processing_config.reload()
        cell_detection_param = cell_detection.default_cell_detection_parameter.copy()
        cell_detection_param['illumination'] = None  # WARNING: illumination or illumination_correction
        cell_detection_param['background_correction']['shape'] = self.processing_config['detection']['background_correction']['diameter']
        cell_detection_param['intensity_detection']['measure'] = ['source']
        cell_detection_param['shape_detection']['threshold'] = self.processing_config['detection']['shape_detection']['threshold']
        if tuning:
            clearmap_io.delete_file(self.workspace.filename('cells', postfix='bkg'))
            cell_detection_param['background_correction']['save'] = self.workspace.filename('cells', postfix='bkg')
            clearmap_io.delete_file(self.workspace.filename('cells', postfix='shape'))
            cell_detection_param['shape_detection']['save'] = self.workspace.filename('cells', postfix='shape')

            # clearmap_io.delete_file(workspace.filename('cells', postfix='maxima'))
            # cell_detection_param['maxima_detection']['save'] = workspace.filename('cells', postfix='maxima')

        processing_parameter = cell_detection.default_cell_detection_processing_parameter.copy()
        processing_parameter.update(  # TODO: store as other dict and run .update(**self.extra_detection_params)
            processes=self.machine_config['n_processes_cell_detection'],  # FIXME: add machine_config
            size_max=self.machine_config['detection_chunk_size_max'],
            size_min=self.machine_config['detection_chunk_size_min'],
            overlap=self.machine_config['detection_chunk_overlap'],
            verbose=True
        )

        try:
            cell_detection.detect_cells(self.workspace.filename('stitched'),
                                        self.workspace.filename('cells', postfix='raw'),
                                        cell_detection_parameter=cell_detection_param,
                                        processing_parameter=processing_parameter)
        finally:
            self.workspace.debug = False

    @property
    def detected(self):
        return os.path.exists(self.workspace.filename('cells', postfix='raw'))

    def export_as_csv(self):  # FIXME: use pandas feather
        source = self.workspace.source('cells')
        header = ', '.join([h[0] for h in source.dtype.names])
        np.savetxt(self.workspace.filename('cells', extension='csv'), source[:], header=header, delimiter=',', fmt='%s')

    def export_to_clearmap1_fmt(self):
        """ClearMap 1.0 export (will generate the files cells_ClearMap1_intensities, cells_ClearMap1_points_transformed,
        cells_ClearMap1_points necessaries to use the analysis script of ClearMap1.
        In ClearMap2 the 'cells' file contains already all this information)
        In order to align the coordinates when we have right and left hemispheres, if the orientation of the brain is left,
         will calculate the new coordinates for the Y axes (resta a lonxitude do eixo Y),
         this change will not affect the orientation of the heatmaps,
        since these are generated from the ClearMap2 file 'cells'
        """

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
            data = data.T   # FIXME: seems hacky
            # if self.sample_config['orientation'] == (1, -2, 3):  # WARNING: seems hacky, why that particular orientation
            #     if filename == 'points_transformed':
            #         data[:, 1] = 528 - data[:, 1]  # WARNING: why 528
            clearmap_io.write(sink, data)

    def plot_cells(self):
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

    def plot_filtered_cells(self):
        coordinates = self.get_coords('filtered')
        p = plot_3d.list_plot_3d(coordinates, color=(1, 0, 0, 0.5), size=10)
        return plot_3d.plot_3d(self.workspace.filename('stitched'), view=p, cmap=plot_3d.grays_alpha(alpha=1))

    def plot_background_substracted_img(self):
        coordinates = np.hstack([self.workspace.source('cells', postfix='raw')[c][:, None] for c in 'xyz'])
        p = plot_3d.list_plot_3d(coordinates)
        return plot_3d.plot_3d(self.workspace.filename('stitched'), view=p, cmap=plot_3d.grays_alpha(alpha=1))

    def remove_crust(self, coordinates,voxelization_parameter):
        dist2surf = clearmap_io.read(self.distance_file)
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
                              **voxelization_parameter)

    def preview_cell_detection(self, parent=None, arange=True, sync=True):
        sources = [self.workspace.filename('stitched'),
                   self.workspace.filename('cells', postfix='bkg'),
                   self.workspace.filename('cells', postfix='shape')
                   ]
        sources = [s for s in sources if os.path.exists(s)]  # Remove missing files (if not tuning)
        return plot_3d.plot(sources, arange=arange, sync=sync, lut='white', parent=parent)

    def get_n_detected_cells(self):
        coords = self.get_coords(coord_type='raw')
        return np.max(coords.shape)  # TODO: check dimension instead

    def get_n_fitlered_cells(self):
        coords = self.get_coords(coord_type='filtered')
        return np.max(coords.shape)  # TODO: check dimension instead

    def plot_voxelized_intensities(self, arange=True):
        return plot_3d.plot(self.workspace.filename('density', postfix='intensities'), arange=arange)


if __name__ == "__main__":
    import sys
    preprocessor = PreProcessor()
    preprocessor.setup(sys.argv[1:3])
    preprocessor.setup_atlases()
    # preprocessor.run()

    detector = CellDetector(preprocessor)
