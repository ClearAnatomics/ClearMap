#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellMap
=======

This module contains the class to analyze (detect) individual cells,
e.g. to analyze immediate early gene expression data from iDISCO+ cleared tissue [Renier2016]_.


.. image:: ../static/cell_abstract_2016.jpg
   :target: https://doi.org/10.1016/j.cell.2020.01.028
   :width: 300

.. figure:: ../static/CellMap_pipeline.png

  iDISCO+ and ClearMap: A Pipeline for Cell Detection, Registration, and 
  Mapping in Intact Samples Using Light Sheet Microscopy.
"""


import copy
import re
import platform
import warnings
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

import pyqtgraph as pg
from matplotlib.colors import to_hex

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
from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.Utils.utilities import requires_assets, FilePath
from ClearMap.processors.generic_tab_processor import TabProcessor
from ClearMap.Visualization.Qt.widgets import Scatter3D

__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


USE_BINARY_POINTS_FILE = not platform.system().lower().startswith('darwin')


class CellDetector(TabProcessor):
    def __init__(self, sample_manager=None, channel=None, registration_processor=None):
        super().__init__()
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.sample_manager = None
        self.registration_processor = None
        self.workspace = None
        self.channel = None
        self.cell_detection_re = ('Processing block',
                                  re.compile(r'.*?Processing block \d+/\d+.*?\selapsed time:\s\d+:\d+:\d+\.\d+'))
        if channel is None:
            raise ValueError(f'No channel specified. Please provide a channel name '
                             f'that matches the one in the sample_params file.')
        self.setup(sample_manager, channel, registration_processor)

    def setup(self, sample_manager, channel_name, registration_processor):
        self.channel = channel_name
        self.sample_config = None
        if sample_manager is not None:
            self.sample_manager = sample_manager
            self.workspace = sample_manager.workspace
            configs = sample_manager.get_configs()
            self.sample_config = configs['sample']
            self.machine_config = configs['machine']
            # FIXME: potential issue of config duplication if several instances are called
            self.processing_config = self.sample_manager.config_loader.get_cfg('cell_map')['channels'][self.channel]

            self.set_progress_watcher(self.sample_manager.progress_watcher)
        self.registration_processor = registration_processor

    @property
    def detected(self):
        return self.get('cells', channel=self.channel, asset_sub_type='raw').exists

    # WARNING: required because we pass a section of the config to the processor, not the whole config
    #   Maybe we should pass the whole config to the processor and let it handle the section (with self.channel)
    #   make sure this works with other config backends.
    def reload_config(self, max_iter=10):
        p = self.processing_config
        for i in range(max_iter):
            if hasattr(p, 'reload'):
                p.reload()
                return
            else:
                p = p.parent
        else:
            raise ValueError(f'Could not find a reload method in the config, after {max_iter} iterations')

    def post_process_cells(self):
        self.reload_config()
        self.filter_cells()
        self.atlas_align()
        self.export_collapsed_stats()

    def voxelize(self, sub_step=''):
        self.reload_config()
        coordinates, cells, voxelization_parameter = self.get_voxelization_params(sub_step=sub_step)
        # %% Unweighted
        _ = self.voxelize_unweighted(coordinates, voxelization_parameter)

    @requires_assets([FilePath('density', postfix='counts')])
    def plot_voxelized_counts(self, arrange=True, parent=None):
        scale = self.registration_processor.config['channels'][self.channel]['resampled_resolution']
        return plot_3d.plot(self.get_path('density', channel=self.channel, asset_sub_type='counts'),
                            scale=scale, title='Cell density (voxelized)', lut='flame',
                            arrange=arrange, parent=parent)

    def create_test_dataset(self, slicing):
        self.workspace.create_debug('stitched', channel=self.channel, slicing=slicing)
        self.update_watcher_main_progress()

    def get_voxelization_params(self, sub_step=''):
        voxelization_parameter = {
            'radius': self.processing_config['voxelization']['radii'],
            'verbose': True
        }
        if self.workspace.debug:  # Path will use debug
            voxelization_parameter['shape'] = self.get('cells', channel=self.channel, asset_sub_type='shape').shape()
        elif self.registration_processor.was_registered:
            voxelization_parameter['shape'] = self.get('atlas', channel=self.channel, asset_sub_type='annotation').shape()
        else:
            voxelization_parameter['shape'] = self.sample_manager.resampled_shape(self.channel)
        if sub_step:  # Hack to compensate for the fact that the realigned makes no sense in
            cells, coordinates = self.get_coords(coord_type=sub_step, aligned=False)
        else:
            cells, coordinates = self.get_coords(coord_type=None, aligned=True)
        return coordinates, cells, voxelization_parameter

    def get_coords(self, coord_type='filtered', aligned=False):
        if coord_type not in ('filtered', 'raw', None):
            raise ValueError(f'Coordinate type "{coord_type}" not recognised')

        kwargs = {'asset_type': 'cells', 'channel': self.channel, 'asset_sub_type': coord_type}

        table_path = self.get_path(**kwargs, extension='.feather')
        if not table_path.exists:
            table_path = self.get_path(**kwargs)

        loaders = {'.feather': pd.read_feather, '.npy': np.load}
        table = loaders[table_path.suffix](table_path)

        axes = ['xt', 'yt', 'zt'] if aligned else ['x', 'y', 'z']
        coordinates = np.array([table[axis] for axis in axes]).T  # .T for (n, axes)
        return table, coordinates

    def voxelize_unweighted(self, coordinates, voxelization_parameter):
        """
        Voxelize un weighted i.e. for cell counts

        Parameters
        ----------
        coordinates: str, array or Source
            Source of point of nxd coordinates.
        voxelization_parameter:  dict
            Dictionary to be passed to voxelization.voxelise (i.e. with these optional keys:
                shape, dtype, weights, method, radius, kernel, processes, verbose

        Returns
        -------
        coordinates, counts_file_path: np.array, str
        """
        counts_file_path = self.get_path('density', channel=self.channel, asset_sub_type='counts')
        clearmap_io.delete_file(counts_file_path)
        self.set_watcher_step('Unweighted voxelisation')
        voxelization.voxelize(coordinates, sink=counts_file_path, **voxelization_parameter)  # WARNING: prange
        self.update_watcher_main_progress()
        # uncrusted_coordinates = self.remove_crust(coordinates)  # WARNING: currently causing issues
        #         density_path = self.get_path('density', channel=self.channel, asset_sub_type='counts_wcrust')
        #         voxelization.voxelize(uncrusted_coordinates, sink=density_path, **voxelization_parameter)   # WARNING: prange
        return coordinates, counts_file_path

    def voxelize_weighted(self, coordinates, source, voxelization_parameter):
        """
        Voxelize weighted i.e. for cell intensities

        Parameters
        ----------
        coordinates: np.array
        source: Source.Source
        voxelization_parameter: dict
        """
        intensities_file_path = self.get_path('density', channel=self.channel, asset_sub_type='intensities')
        intensities = source['source']
        voxelization.voxelize(coordinates, sink=intensities_file_path, weights=intensities, **voxelization_parameter)   # WARNING: prange
        return intensities_file_path

    def atlas_align(self):
        """Atlas alignment and annotation """
        table, coordinates = self.get_coords(coord_type='filtered')
        df = pd.DataFrame({'x': coordinates[:, 0], 'y': coordinates[:, 1], 'z': coordinates[:, 2]})
        df['size'] = table['size']
        df['source'] = table['source']

        if self.registration_processor.was_registered:  # FIXME: check if should be registered and raise error
            coordinates_transformed = self.transform_coordinates(coordinates)
            annotator = self.registration_processor.annotators[self.channel]
            channel_cfg = self.registration_processor.config['channels'][self.sample_manager.alignment_reference_channel]
            atlas_resolution = channel_cfg['resampled_resolution']
            extra_columns = annotator.get_columns(coordinates_transformed, atlas_resolution)  # OPTIMISE: parallel
            df = pd.concat([df, extra_columns], axis=1)
        else:
            warnings.warn('Atlas alignment requires a registered sample.'
                          'Skipping alignment and using resampled coordinates.')


        df.to_feather(self.get_path('cells', channel=self.channel, extension='.feather'))

    def transform_coordinates(self, coords):
        target_channel = 'atlas'  # FIXME: add control for target channel
        resampled_shape = self.sample_manager.resampled_shape(channel=self.channel)
        if resampled_shape is None:
            if target_channel == 'atlas':
                resampled_shape = self.get('atlas', channel=self.channel, asset_sub_type='reference').shape()
            else:
                raise ValueError(f'Resampled shape not found for channel {self.channel}')
        coords = resampling.resample_points(
            coords,
            original_shape=self.sample_manager.stitched_shape(channel=self.channel),
            resampled_shape=resampled_shape)

        if self.registration_processor.was_registered:
            for i, channel in enumerate(self.get_registration_sequence_channels(stop_channel=target_channel)):
                if self.registration_processor.config['channels'][channel]['moving_channel'] in (None, 'intrinsically aligned'):
                    continue
                results_dir = self.get_path('aligned', channel=channel).parent
                coords = elastix.transform_points(coords, transform_directory=results_dir, binary=USE_BINARY_POINTS_FILE)

        return coords

    def filter_cells(self):
        self.reload_config()
        thresholds = {
            'source': self.processing_config['cell_filtration']['thresholds']['intensity'],
            'size': self.processing_config['cell_filtration']['thresholds']['size']
        }
        cell_detection.filter_cells(source=self.get_path('cells', channel=self.channel, asset_sub_type='raw'),
                                    sink=self.get_path('cells', channel=self.channel, asset_sub_type='filtered'),
                                    thresholds=thresholds)

    def run_cell_detection(self, tuning=False, save_maxima=False, save_shape=False, save_as_binary_mask=False):
        self.reload_config()
        self.workspace.debug = tuning  # TODO: use context manager
        cell_detection_param = copy.deepcopy(cell_detection.default_cell_detection_parameter)
        cell_detection_param['illumination_correction'] = None  # WARNING: illumination or illumination_correction
        cell_detection_param['background_correction']['shape'] = self.processing_config['detection']['background_correction']['diameter']
        cell_detection_param['maxima_detection']['shape'] = self.processing_config['detection']['maxima_detection']['shape']
        cell_detection_param['intensity_detection']['measure'] = ['source']
        cell_detection_param['shape_detection']['threshold'] = self.processing_config['detection']['shape_detection']['threshold']
        shape_path = self.get_path('cells', channel=self.channel, asset_sub_type='shape')
        if tuning:
            bkg_path = self.get_path('cells', channel=self.channel, asset_sub_type='bkg')
            clearmap_io.delete_file(bkg_path)
            cell_detection_param['background_correction']['save'] = bkg_path
            clearmap_io.delete_file(shape_path)
            cell_detection_param['shape_detection']['save'] = shape_path

        if save_shape:
            clearmap_io.delete_file(shape_path)
            cell_detection_param['shape_detection']['save'] = shape_path
            # erase any prior existing file to prevent confusion of sink with source in IO.initialize
            # Path(shape_path).unlink(missing_ok=True)
            # if save_as_binary_mask:
                # cell_detection_param['shape_detection']['save_dtype'] = 'bool'

        if save_maxima:
            maxima_path = self.get_path('cells', channel=self.channel, asset_sub_type='maxima')
            clearmap_io.delete_file(maxima_path)
            cell_detection_param['maxima_detection']['save'] = maxima_path

        processing_parameter = copy.deepcopy(cell_detection.default_cell_detection_processing_parameter)
        processing_parameter.update(  # TODO: store as other dict and run .update(**self.extra_detection_params)
            processes=self.machine_config['n_processes_cell_detection'],
            size_min=self.machine_config['detection_chunk_size_min'],
            size_max=self.machine_config['detection_chunk_size_max'],
            overlap=self.machine_config['detection_chunk_overlap'],
            verbose=True
        )

        # TODO: round to processors
        n_steps = self.get_n_blocks(self.get('stitched', channel=self.channel).shape()[2])
        self.prepare_watcher_for_substep(n_steps, self.cell_detection_re, 'Detecting cells')
        try:
            cell_detection.detect_cells(self.get_path('stitched', channel=self.channel),
                                        self.get_path('cells', channel=self.channel, asset_sub_type='raw'),
                                        cell_detection_parameter=cell_detection_param,
                                        processing_parameter=processing_parameter,
                                        workspace=self.workspace)  # WARNING: prange inside multiprocess (including arrayprocessing and devolvepoints for vox)
            if save_shape and save_as_binary_mask and clearmap_io.dtype(shape_path) != 'bool':
                clearmap_io.write(shape_path, np.array(clearmap_io.read(shape_path).astype('bool')))
        except BrokenProcessPool as err:
            print(f'Cell detection canceled, see: {err}')
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
                      "please use the new formats from atlas_align and export_collapsed_stats",
                      DeprecationWarning, 2)

        csv_file_path = self.get_path('cells', channel=self.channel, extension='.csv')
        self.get_cells_df().to_csv(csv_file_path)

    def export_collapsed_stats(self, all_regions=True):
        df = self.get_cells_df()

        collapsed = pd.DataFrame()
        relevant_columns = ['id', 'order', 'name', 'hemisphere', 'volume', 'size']
        for i in (0, 255):  # Split by hemisphere to group by structure and reconcatenate hemispheres after
            grouped = df[df['hemisphere'] == i][relevant_columns].groupby(['id'], as_index=False)

            tmp = pd.DataFrame()
            first = grouped.first()
            tmp['Structure ID'] = first['id']
            tmp['Structure order'] = first['order']
            tmp['Structure name'] = first['name']
            tmp['Hemisphere'] = first['hemisphere']
            tmp['Structure volume'] = first['volume']
            tmp['Cell counts'] = grouped.count()['name']
            tmp['Average cell size'] = grouped['size'].mean()

            collapsed = pd.concat((collapsed, tmp))

        annotator = self.registration_processor.annotators[self.channel]
        if all_regions:  # Add regions even if they are empty
            uniq_ids = np.unique(annotator.atlas)
            tmp = pd.DataFrame({'Structure ID': uniq_ids, 'mock': ''})
            tmp['Structure name'] = annotator.convert_label(uniq_ids, key='id', value='name')
            df_mock = pd.DataFrame({'Hemisphere': [0, 255], 'mock': ''})
            tmp = tmp.merge(df_mock, on='mock').drop(columns='mock')
            vol_map = annotator.get_lateralised_volume_map(
                self.registration_processor.config['channels'][self.sample_manager.alignment_reference_channel]['resampled_resolution'],
                self.get_path('atlas', channel=self.channel, asset_sub_type='hemispheres')
            )
            tmp['Structure volume'] = tmp.set_index(['Structure ID', 'Hemisphere']).index.map(vol_map.get)
            order_map = {id_: annotator.find(id_, key='id')['order'] for id_ in uniq_ids}
            tmp['Structure order'] = tmp['Structure ID'].map(order_map)
            collapsed = tmp.merge(collapsed[['Structure ID', 'Hemisphere', 'Cell counts', 'Average cell size']],
                                  how='left', on=['Structure ID', 'Hemisphere'])

        collapsed = collapsed.sort_values(by='Structure ID')

        csv_file_path = self.get_path('cells', channel=self.channel, asset_sub_type='stats', extension='.csv')
        collapsed.to_csv(csv_file_path, index=False)

    def plot_cells_3d_scatter_w_atlas_colors(self, raw=False, parent=None):
        asset_properties = {'channel': self.channel}
        if raw:
            asset_properties['asset_type'] = 'stitched'
        else:
            if self.registration_processor.was_registered:
                asset_properties['asset_type'] = 'atlas'
                asset_properties['asset_sub_type'] = 'reference'
            else:
                asset_properties['asset_type'] = 'resampled'

        asset = self.get(**asset_properties)
        if not asset.exists or not self.df_path.exists:
            raise MissingRequirementException(f'plot_cells_3d_scatter_w_atlas_colors missing files:'
                                              f'image: {asset.path} {"not" if not asset.exists else ""} found'
                                              f'cells data frame {"not" if not self.df_path.exists else ""} found')
        dv = qplot_3d.plot(asset.path, title=f'{asset_properties["asset_type"].title()} and cells',  # FIXME: correct scaling for anisotropic if raw
                           arrange=False, lut='white', parent=parent)[0]

        scatter = pg.ScatterPlotItem()

        dv.view.addItem(scatter)
        dv.scatter = scatter

        df = self.get_cells_df()

        if raw:
            coordinates = df[['x', 'y', 'z']].values.astype(int)
            # coordinates = coordinates * np.array(self.sample_manager.config['resolutions']['raw'])
            # coordinates = coordinates.astype(int)  # required to match integer z  # FIXME: correct scaling for anisotropic
        else:
            coordinates = df[['xt', 'yt', 'zt']].values.astype(int)  # required to match integer z
            dv.atlas = self.get('atlas', channel=self.channel, asset_sub_type='annotation').read()
            dv.structure_names = self.registration_processor.annotators[self.channel].get_names_map()
        if 'hemisphere' in df.columns:
            hemispheres = df['hemisphere']
        else:
            hemispheres = None

        unique_ids = np.sort(np.unique(df['id']))
        annotator = self.registration_processor.annotators[self.channel]
        # color_map = {id_: annotator.find(id_, key='id')['rgb'] for id_ in unique_ids}
        color_map = {id_: annotator.convert_label(id_, key='id', value='color_hex_triplet') for id_ in unique_ids}
        color_map[0] = np.array(to_hex((1, 0, 0)))  # default to red
        df['color'] = df['id'].map(color_map)
        dv.scatter_coords = Scatter3D(coordinates, colors=df['color'].to_list(),
                                      hemispheres=hemispheres, half_slice_thickness=0)
        dv.refresh()
        return [dv]

    @property
    def df_path(self):
        feather_path = self.get_path('cells', channel=self.channel, extension='.feather')
        if feather_path.exists:
            return feather_path
        else:
            return self.get_path('cells', channel=self.channel)

    def get_cells_df(self):
        df_path = self.df_path
        if df_path.suffix == '.feather':
            return pd.read_feather(df_path)
        else:
            return pd.DataFrame(np.load(df_path))

    @requires_assets([FilePath('cells', asset_sub_type='filtered'), FilePath('stitched')])
    def plot_filtered_cells(self, parent=None, smarties=False):
        _, coordinates = self.get_coords('filtered')
        stitched_path = self.get_path('stitched', channel=self.channel)
        dv = qplot_3d.plot(stitched_path, title='Stitched and filtered cells', arrange=False,
                           lut='white', parent=parent)[0]
        scatter = pg.ScatterPlotItem()

        dv.view.addItem(scatter)
        dv.scatter = scatter

        dv.scatter_coords = Scatter3D(coordinates, smarties=smarties, half_slice_thickness=3)
        dv.refresh()
        return [dv]

    def plot_background_subtracted_img(self):
        src = self.get('cells', channel=self.channel, asset_sub_type='raw').as_source()
        coordinates = np.hstack([src[c][:, None] for c in 'xyz'])
        p = plot_3d.list_plot_3d(coordinates)
        return plot_3d.plot_3d(self.get_path('stitched', channel=self.channel),
                               view=p, cmap=plot_3d.grays_alpha(alpha=1))

    def remove_crust(self, coordinates, threshold=3):
        distance_to_surface = self.get('atlas', channel=self.channel, asset_sub_type='distance').read()

        # Convert coordinates to integer and insure they are within the distance_to_surface array bounds
        int_coordinates = np.floor(coordinates).astype(int)  # TODO: check if floor required
        xs, ys, zs = int_coordinates.T
        xmax, ymax, zmax = distance_to_surface.shape
        within_atlas = (xs >= 0) & (xs < xmax) & (ys >= 0) & (ys < ymax) & (zs >= 0) & (zs < zmax)

        # Get the distance_to_surface values at the valid coordinates
        valid_coordinates = int_coordinates[within_atlas]
        dist_values = distance_to_surface[tuple(valid_coordinates.T)]

        uncrusted_coordinates = valid_coordinates[dist_values > threshold]
        return uncrusted_coordinates

    def preview_cell_detection(self, parent=None, arrange=True, sync=True):
        sources = [
            self.get_path('stitched', channel=self.channel),
            self.get_path('cells', channel=self.channel, asset_sub_type='bkg'),
            self.get_path('cells', channel=self.channel, asset_sub_type='shape')
        ]
        sources = [s for s in sources if s.exists]  # Remove missing files (if not tuning)
        if not sources:
            raise MissingRequirementException('No files found for preview')
        titles = [s.name for s in sources]
        luts = ['white', 'white', 'random']
        return plot_3d.plot(sources, title=titles, arrange=arrange, sync=sync, lut=luts, parent=parent)

    def get_n_detected_cells(self):
        if self.get('cells', channel=self.channel, asset_sub_type='raw').exists:
            _, coords = self.get_coords(coord_type='raw')
            return np.max(coords.shape)  # TODO: check dimension instead
        else:
            return 0

    def get_n_filtered_cells(self):
        if self.get('cells', channel=self.channel, asset_sub_type='filtered').exists:
            _, coords = self.get_coords(coord_type='filtered')
            return np.max(coords.shape)  # TODO: check dimension instead
        else:
            return 0

    def plot_voxelized_intensities(self, arrange=True):
        density_path = self.get_path('density', channel=self.channel, asset_sub_type='intensities')
        return plot_3d.plot(density_path, arrange=arrange)

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
        source = self.get('cells', channel=self.channel).as_source()
        clearmap1_format = {'points': ['x', 'y', 'z'],
                            'points_transformed': ['xt', 'yt', 'zt'],
                            'intensities': ['source', 'dog', 'background', 'size']}
        for sub_type, names in clearmap1_format.items():
            sink = self.get_path('cells', channel=self.channel, asset_sub_type=f'ClearMap1{sub_type}')
            print(sub_type, sink)
            data = np.array(
                [source[name] if name in source.dtype.names else np.full(source.shape[0], np.nan) for name in names]
            )
            data = data.T
            clearmap_io.write(sink, data)

    def convert_cm2_to_cm2_1_fmt(self):
        """Atlas alignment and annotation """
        cells = np.load(self.get_path('cells', channel=self.channel))
        df = pd.DataFrame({ax: cells[ax] for ax in 'xyz'})
        df['size'] = cells['size']
        df['source'] = cells['source']
        for ax in 'xyz':
            df[f'{ax}t'] = cells[f'{ax}t']
        df['order'] = cells['order']
        df['name'] = cells['name']

        coordinates_transformed = np.vstack([cells[f'{ax}t'] for ax in 'xyz']).T

        annotator = self.registration_processor.annotators[self.channel]
        hemisphere_label = annotator.label_points_hemispheres(coordinates_transformed)

        unique_labels = np.sort(df['order'].unique())  # WARNING: ClearMap2 used order as key (now deprecated)
        color_map = {lbl: annotator.find(lbl, key='order')['rgb'] for lbl in
                     unique_labels}  # WARNING RGB upper case should give integer but does not work
        id_map = {lbl: annotator.find(lbl, key='order')['id'] for lbl in unique_labels}

        atlas = self.get('atlas', channel=self.channel, asset_sub_type='annotation').read()
        atlas_scale = self.registration_processor.config['channels'][self.sample_manager.alignment_reference_channel]['resampled_resolution']
        atlas_scale = np.prod(atlas_scale)
        volumes = {_id: (atlas == _id).sum() * atlas_scale for _id in
                   id_map.values()}  # Volumes need a lookup on ID since the atlas is in ID space

        df['id'] = df['order'].map(id_map)
        df['hemisphere'] = hemisphere_label
        df['color'] = df['order'].map(color_map)
        df['volume'] = df['id'].map(volumes)

        df.to_feather(self.get_path('cells', channel=self.channel, extension='.feather'))

    def get_registration_sequence_channels(self, stop_channel='atlas'):
        out = [self.channel]
        registration_cfg = self.registration_processor.config['channels']
        while True:
            next_channel = registration_cfg[out[-1]]['align_with']
            if next_channel in (None, stop_channel):
                break
            out.append(next_channel)
        return out
