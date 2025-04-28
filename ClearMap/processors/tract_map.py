import functools
import os
import platform
import shutil
import tempfile
from multiprocessing.managers import BaseManager
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from matplotlib.colors import to_hex

from ClearMap.IO import IO as cmp_io
from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.Utils.utilities import sanitize_n_processes
from ClearMap.processors.generic_tab_processor import TabProcessor
from ClearMap.Alignment import Elastix as elastix
from ClearMap.Alignment.Resampling import resample_points
from ClearMap.ParallelProcessing.DataProcessing import ArrayProcessing as array_processing
from ClearMap.ParallelProcessing import BlockProcessing as block_processing
from ClearMap.ImageProcessing.Experts import Vasculature as vasculature
from ClearMap.Analysis.Measurements.Voxelization import voxelize

from ClearMap.Visualization.Qt.widgets import Scatter3D
from ClearMap.Visualization.Qt import Plot3d as q_plot_3d

USE_BINARY_POINTS_FILE = not platform.system().lower().startswith('darwin')  # i.e. binary is available in elastix


# WARNING: this has to be top level to be pickleable (because of the way multiprocessing works)
def label_points_wrapper(annotator, coords):
    return np.expand_dims(annotator.label_points(coords), axis=-1)  # Add empty dim to match shape of coords


class TractMapProcessor(TabProcessor):
    def __init__(self, sample_manager=None, channel=None, registration_processor=None):
        super().__init__()
        self.save_intermediate_binarization_results = True
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.sample_manager = None
        self.registration_processor = None
        self.workspace = None
        self.channel = None

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
            self.processing_config = self.sample_manager.config_loader.get_cfg('tract_map')['channels'][self.channel]

            self.set_progress_watcher(self.sample_manager.progress_watcher)
        self.registration_processor = registration_processor

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

    def create_test_dataset(self, slicing):
        self.workspace.create_debug('stitched', channel=self.channel, slicing=slicing)
        self.update_watcher_main_progress()

    def compute_clip_range(self, array_path, pixel_percents=(0.7, 0.99999)):  # TODO: display in UI and set in params
        array = cmp_io.read(array_path)
        print(f'{array.shape=}, {array.size=}')
        # Sample every 10th pixel in each dimension, to have a smaller array to compute the quantiles
        ratio = self.processing_config['detection']['decimation_ratio']
        decimated_array = array[::ratio, ::ratio, ::ratio]

        min_quantile, max_quantile = [np.quantile(decimated_array, p) for p in pixel_percents]
        print(f'Intensity values that contain {pixel_percents} percents respectively '
              f'of the pixels: {min_quantile=}, {max_quantile=}')
        return min_quantile, max_quantile  # TODO: add display in UI

    def binarize(self, clip_low, clip_high):
        binarization_parameter = vasculature.default_binarization_parameter.copy()

        binarization_parameter['clip']['clip_range'] = (clip_low, clip_high)
        binarization_parameter['equalize'] = None
        if self.save_intermediate_binarization_results:
            if binarization_parameter['equalize'] is not None:
                binarization_parameter['equalize']['save'] = str(
                    self.get_path('binary', channel=self.channel, asset_sub_type='equalize')
                )
            # binarization_parameter['vesselize']['save'] = str(
            #     self.get_path('binary', channel=self.channel, asset_sub_type='vesselize')
            # )
            # binarization_parameter['median']['save'] = str(
            #     self.get_path('binary', channel=self.channel, asset_sub_type='median')
            # )
        binarization_parameter['vesselize']['threshold'] = 1
        binarization_parameter['vesselize']['tubeness']['sigma'] = 1
        binarization_parameter['deconvolve'] = None
        binarization_parameter['adaptive'] = None

        processing_parameter = vasculature.default_binarization_processing_parameter.copy()
        processing_parameter.update(processes=self.processing_config['parallel_params']['n_processes_binarization'],
                                    as_memory=False,
                                    verbose=True)

        vasculature.binarize(self.get_path('stitched', channel=self.channel),
                      self.get_path('binary', channel=self.channel),
                      binarization_parameter=binarization_parameter,
                      processing_parameter=processing_parameter)
        print('TractMap binarization finished')
        self.update_watcher_main_progress()

    def mask_to_coordinates(self, as_memmap=False):
        mask = str(self.get_path('binary', channel=self.channel))
        output_asset = self.get('binary', asset_sub_type='pixels_raw', channel=self.channel)
        if output_asset.exists:
            output_asset.delete()
        if as_memmap:
            return array_processing.where(mask, output_asset.path,
                                          processes=self.processing_config['parallel_params']['n_processes_where'],
                                          verbose=True)
            self.update_watcher_main_progress()
            print('TractMap coordinates extraction finished')
        else:
            raise NotImplementedError('Output to file not implemented yet')

    def get_registration_sequence_channels(self, stop_channel='atlas'):
        out = [self.channel]
        registration_cfg = self.registration_processor.config['channels']
        while True:
            next_channel = registration_cfg[out[-1]]['align_with']
            if next_channel in (None, stop_channel):
                break
            out.append(next_channel)
        return out

    @staticmethod
    def transformation(coords, source_shape, resampled_shape, results_directories):
        coords = resample_points(
            coords, sink=None, orientation=None,
            source_shape=source_shape,
            sink_shape=resampled_shape
        )

        for results_dir in results_directories:
            # We copy the files to temporary directories to avoid conflicts during parallel processing
            with (tempfile.TemporaryDirectory(suffix='_elastix_output') as temp_dir,
                  tempfile.TemporaryDirectory(suffix='_elastix_params') as temp_params_dir,
                  tempfile.NamedTemporaryFile(suffix='_elastix_input.bin', delete=False) as temp_file):


                # Copy the params files to avoid a race condition because they get edited by each process
                file_names = list(results_dir.glob('TransformParameters.*.txt'))
                if not file_names:
                    raise FileNotFoundError(f'No parameter files found in {results_dir}')
                # print(f'Copying {file_names} to {temp_params_dir}')
                for f_name in file_names:
                    f_name = Path(f_name).relative_to(results_dir)
                    f_path = results_dir / f_name
                    dest_path = Path(temp_params_dir) / f_name
                    shutil.copyfile(f_path, dest_path)

                coords = elastix.transform_points(
                    coords, sink=None,
                    transform_directory=temp_params_dir,
                    result_directory=temp_dir,
                    temp_file=temp_file.name,
                    binary=True, indices=False)

                Path(temp_file.name).unlink(missing_ok=True)

        return coords

    def parallel_transform(self, processes=-1):
        coords = self.get('binary', channel=self.channel, asset_sub_type='pixels_raw').as_source()
        coordinates_transformed_path = self.get_path('binary', asset_sub_type='coordinates_transformed', channel=self.channel)
        coordinates_transformed_path.unlink(missing_ok=True)
        transformed_coords = array_processing.initialize_sink(coordinates_transformed_path,
            dtype='float64', shape=coords.shape, return_buffer=False
        )
        perf_params = self.processing_config['parallel_params']

        target_channel = 'atlas'  # FIXME: add control for target channel
        status_bcp = self.workspace.debug
        self.workspace.debug = False
        resampled_shape = self.sample_manager.resampled_shape(channel=self.channel)
        self.workspace.debug = status_bcp
        if resampled_shape is None:
            # FIXME: in this case compute from scale differences
            raise ValueError(f'Resampled shape not found for channel {self.channel}')

        results_directories = []
        for channel in self.get_registration_sequence_channels(stop_channel=target_channel):
            if self.registration_processor.config['channels'][channel]['moving_channel'] in (None, 'intrinsically aligned'):
                continue
            else:
                results_directories.append(self.get_path('aligned', channel=channel).parent)

        debug_bcp = self.workspace.debug
        self.workspace.debug = False
        source_shape = self.sample_manager.stitched_shape(channel=self.channel)
        self.workspace.debug = debug_bcp
        transfo = functools.partial(TractMapProcessor.transformation,
                                    source_shape=source_shape,
                                    resampled_shape=resampled_shape,
                                    results_directories=results_directories
                                    )
        transfo.__name__ = 'parallel_transform'

        processes = sanitize_n_processes(processes)

        if processes == 1:
            transformed_coords[:, :] = transfo(coords)
        else:
            block_processing.process(transfo, coords, transformed_coords,
                                     axes=[0], processes=perf_params['n_processes_transform'],
                                     size_min=perf_params['min_point_list_size'],
                                     size_max=perf_params['max_point_list_size'],
                                     verbose=True)
        self.update_watcher_main_progress()
        print('TractMap coordinates transformed')
        return transformed_coords

    def label(self):
        class AnnotationProxy:
            def __init__(self, annotator):
                self._annotator = annotator

            def label_points(self, coords):
                return self._annotator.label_points(coords)

        class AnnotationManager(BaseManager):
            pass

        AnnotationManager.register('Annotation', AnnotationProxy)  # added 3.11 shutdown_timeout

        coordinates_transformed = self.get('binary', channel=self.channel,
                                           asset_sub_type='coordinates_transformed').as_source()
        labels = array_processing.initialize_sink(self.get_path('binary', channel=self.channel,
                                                                asset_sub_type='labels'),
                                 dtype='int64', shape=(coordinates_transformed.shape[0], 1),
                                 return_buffer=False)

        with AnnotationManager() as manager:
            annotator = manager.Annotation(self.registration_processor.annotators[self.channel])
            labeling_fn = functools.partial(label_points_wrapper, annotator)
            labeling_fn.__name__ = 'label_points'  # for block_processing prints
            perf_params = self.processing_config['parallel_params']
            block_processing.process(labeling_fn, coordinates_transformed, labels,
                                     axes=[0], processes=perf_params['n_processes_label'],
                                     size_min=perf_params['min_point_list_size'],
                                     size_max=perf_params['max_point_list_size'],
                                     verbose=True)
        self.update_watcher_main_progress()
        print('TractMap coordinates labeled')
        return labels

    def shift_coordinates(self):
        """Shift the coordinates by the cropping amount to get the values in whole sample reference frame"""
        coords_asset = self.get('binary', asset_sub_type='pixels_raw', channel=self.channel)
        coordinates = coords_asset.as_source()
        for i in range(3):
            shift = self.processing_config['test_set_slicing'][f'dim_{i}'][0]
            coordinates[:, i] += shift
        cmp_io.write(coords_asset.path, coordinates)
        print('TractMap coordinates shifted')

    def run_pipeline(self, tuning=False):
        self.workspace.debug = tuning
        self.reload_config()

        self.binarize(*self.processing_config['binarization']['clip_range'])
        self.mask_to_coordinates(as_memmap=USE_BINARY_POINTS_FILE)

        if tuning:
            self.shift_coordinates()

        self.parallel_transform()
        self.label()
        self.voxelize()

        self.export_df(asset_sub_type=None)

        self.workspace.debug = False

    def export_df(self, asset_sub_type=None):
        ratio = self.processing_config['display']['decimation_ratio']
        decimated_coordinates_raw = self.get(
            'binary', channel=self.channel, asset_sub_type='pixels_raw').as_source()[::ratio, :]

        decimated_coordinates_transformed = self.get(
            'binary', channel=self.channel,
            asset_sub_type='coordinates_transformed').as_source()[::ratio, :]

        decimated_labels = self.get('binary', channel=self.channel,
                                    asset_sub_type='labels').as_source()[::ratio, :]

        # Build the DataFrame
        df = pd.DataFrame({'id': decimated_labels[:, 0]})
        for i, l in enumerate('xyz'):
            df[l] = decimated_coordinates_raw[:, i]

        for i, l in enumerate('xyz'):
            df[f'{l}t'] = decimated_coordinates_transformed[:, i]

        unique_ids = np.sort(np.unique(decimated_labels))
        annotator = self.registration_processor.annotators[self.channel]
        color_map = {id_: annotator.find(id_, key='id')['rgb'] for id_ in unique_ids}
        color_map[0] = np.array((1, 0, 0))  # default to red
        df['color'] = df['id'].map(color_map)

        df.to_feather(self.get_path('tract_voxels', channel=self.channel,
                                    asset_sub_type=asset_sub_type, extension='.feather'))
        self.update_watcher_main_progress()
        print('TractMap DF exported')
        return df

    def voxelize(self):
        voxelization_parameter = dict(
            shape=cmp_io.shape(self.registration_processor.annotators[self.channel].annotation_file),
            dtype=None,
            weights=None,
            method='sphere',
            radius=self.processing_config['voxelization']['radii'],
            kernel=None,
            processes=None,
            verbose=True
        )
        voxelize(self.get('binary', asset_sub_type='coordinates_transformed', channel=self.channel).as_source(),
                 sink=self.get_path('density', channel=self.channel, asset_sub_type='counts'),
                 **voxelization_parameter
        )
        self.update_watcher_main_progress()
        print('TractMap voxelization finished')

    def plot_tracts_3d_scatter_w_atlas_colors(self, raw=False, coordinates_from_debug=False, plot_onto_debug=False, parent=None):
        asset_properties = {'channel': self.channel}
        if raw:
            asset_properties['asset_type'] = 'stitched'  # FIXME: select based on range
            # asset_properties['asset_type'] = 'resampled'
        else:
            if self.registration_processor.was_registered:
                asset_properties['asset_type'] = 'atlas'
                asset_properties['asset_sub_type'] = 'reference'
            else:
                asset_properties['asset_type'] = 'resampled'

        ws_debug_backup = self.workspace.debug
        self.workspace.debug = plot_onto_debug
        asset = self.get(**asset_properties)
        asset_path = asset.path

        self.workspace.debug = coordinates_from_debug
        df_path = self.get_path('tract_voxels', channel=self.channel, extension='.feather')
        self.workspace.debug = ws_debug_backup
        if not asset_path.exists() or not df_path.exists():
            raise MissingRequirementException(f'plot_3d_scatter_w_atlas_colors missing files:'
                                              f'image: {asset_path} {"not" if not asset_path.exists() else ""} found'
                                              f'tract voxels data frame {"not" if not df_path.exists() else ""} found')
        # FIXME: correct scaling for anisotropic if raw
        dv = q_plot_3d.plot(asset_path, title=f'{asset_properties["asset_type"].title()} and tracts coordinates',
                           arrange=False, lut='white', parent=parent)[0]

        scatter = pg.ScatterPlotItem()

        dv.view.addItem(scatter)
        dv.scatter = scatter

        df = pd.read_feather(df_path)

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

    def plot_voxelized_counts(self):
        asset = self.get('density', channel=self.channel, asset_sub_type='counts')
        if not asset.exists:
            raise MissingRequirementException(f'plot_voxelized_counts missing file: {asset.path} not found')
        dv = q_plot_3d.plot(asset.path, title=f'Voxelized counts', arrange=False, lut='flame')[0]
        return [dv]
