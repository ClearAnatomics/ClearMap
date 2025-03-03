import functools
import glob
import platform
import shutil
import tempfile
from multiprocessing.managers import BaseManager
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg

from ClearMap.IO import IO as cmp_io
from ClearMap.Utils.exceptions import MissingRequirementException
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
        iter = 0
        while not hasattr(p, 'reload'):
            p = p.parent
            iter += 1
            if iter > max_iter:
                raise ValueError('Could not find a reload method in the config')
        p.reload()

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
            binarization_parameter['vesselize']['save'] = str(
                self.get_path('binary', channel=self.channel, asset_sub_type='vesselize')
            )
            binarization_parameter['median']['save'] = str(
                self.get_path('binary', channel=self.channel, asset_sub_type='median')
            )
        binarization_parameter['vesselize']['threshold'] = 1
        binarization_parameter['vesselize']['tubeness']['sigma'] = 1
        binarization_parameter['deconvolve'] = None
        binarization_parameter['adaptive'] = None

        processing_parameter = vasculature.default_binarization_processing_parameter.copy()
        processing_parameter.update(processes=self.machine_config['n_processes_binarization'],
                                    as_memory=False,
                                    verbose=True)

        vasculature.binarize(self.get_path('stitched', channel=self.channel),
                      self.get_path('binary', channel=self.channel),
                      binarization_parameter=binarization_parameter,
                      processing_parameter=processing_parameter)

    def mask_to_coordinates(self, as_memmap=False):
        mask = str(self.get_path('binary', channel=self.channel))
        output_path = self.get_path('binary', asset_sub_type='pixels_raw', channel=self.channel)
        if as_memmap:
            return array_processing.where(mask, output_path,
                                          processes=self.machine_config['n_processes_tract_map'],
                                          verbose=True)
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

    def transformation(self, coords):
        target_channel = 'atlas'  # FIXME: add control for target channel
        resampled_shape = self.sample_manager.resampled_shape(channel=self.channel)
        if resampled_shape is None:
            if target_channel == 'atlas':
                resampled_shape = self.get('atlas', channel=self.channel, asset_sub_type='reference').shape()
            else:
                raise ValueError(f'Resampled shape not found for channel {self.channel}')
        coords = resample_points(
            coords, sink=None, orientation=None,
            source_shape=self.sample_manager.stitched_shape(channel=self.channel),
            sink_shape=resampled_shape
        )

        if self.registration_processor.was_registered:
            for i, channel in enumerate(self.get_registration_sequence_channels(stop_channel=target_channel)):
                if self.registration_processor.config['channels'][channel]['moving_channel'] in (None, 'intrinsically aligned'):
                    continue
                results_dir = self.get_path('aligned', channel=channel).parent
                with (tempfile.TemporaryDirectory(suffix='elastix_output') as temp_dir,
                      tempfile.TemporaryDirectory(suffix='elastix_params') as temp_params_dir,
                      tempfile.NamedTemporaryFile(suffix='elastix_input.bin', delte=False) as temp_file):

                    # _, transform_parameter_file = transform_directory_and_file(transform_directory=ws.filename('resampled_to_auto'))

                    # Copy the params files to avoid a race condition because they get edited by each process
                    for f_name in glob.glob(results_dir / 'TransformParameters.*.txt'):
                        f_path = results_dir / f_name
                        shutil.copy(f_path, temp_params_dir)

                    coords = elastix.transform_points(
                        coords, sink=None,
                        transform_directory=temp_params_dir,
                        result_directory=temp_dir,
                        temp_file=temp_file.name,
                        binary=True, indices=False)

                    tmp_f_path = Path(temp_file.name)
                tmp_f_path.unlink(missing_ok=True)

        return coords

    def parallel_transform(self):
        coords = self.get('binary', channel=self.channel, asset_sub_type='pixels_raw').as_source
        transformed_coords = array_processing.initialize_sink(
            self.get_path('binary', asset_sub_type='coordinates_transformed', channel=self.channel),
            dtype='float64', shape=coords.shape, return_buffer=False
        )
        # FIXME: see if works with method or if we should use a function and pass params here
        perf_params = self.processing_config['parallel_params']
        block_processing.process(self.transformation, coords, transformed_coords,
                                 axes=[0], processes=perf_params['n_processes_transform'],
                                 size_min=perf_params['min_point_list_size'],
                                 size_max=perf_params['max_point_list_size'],
                                 verbose=True)

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
                                           asset_sub_type='coordinates_transformed').as_source
        labels = array_processing.initialize_sink(self.get_path('binary', channel=self.channel,
                                                                asset_sub_type='labels'),
                                 dtype='int64', shape=(coordinates_transformed.shape[0], 1),
                                 return_buffer=False)
        # labeling_fn = functools.partial(ano.label_points, key='id')
        def label_points_wrapper(annotator, coords):
            return np.expand_dims(annotator.label_points(coords), axis=-1)  # Add empty dim to match shape of coords

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
        return labels

    def run_pipeline(self, tuning=False):
        self.reload_config()

        self.update_watcher_main_progress()
        self.binarize(*self.processing_config['binarization']['clip_range'])
        self.update_watcher_main_progress()

        self.mask_to_coordinates(as_memmap=USE_BINARY_POINTS_FILE)
        self.update_watcher_main_progress()

        self.parallel_transform()
        self.update_watcher_main_progress()

        self.label()
        self.update_watcher_main_progress()

        self.voxelize()
        self.update_watcher_main_progress()

        self.export_df(asset_sub_type=None)
        self.update_watcher_main_progress()

        self.plot_cells_3d_scatter_w_atlas_colors(raw=False)

    def export_df(self, asset_sub_type='fake'):  # FIXME: needs dynamic asset_sub_type
        ratio = self.processing_config['display']['decimation_ratio']
        decimated_coordinates_raw = self.get(
            'binary', channel=self.channel, asset_sub_type='pixels_raw').as_source[::ratio, :]
        cmp_io.write(self.get_path(
            'binary', channel=self.channel,
            asset_sub_type=f'coordinates_raw_decimated_{ratio}'),
            decimated_coordinates_raw)

        decimated_coordinates_transformed = self.get(
            'binary', channel=self.channel,
            asset_sub_type='coordinates_transformed').as_source[::ratio, :]
        cmp_io.write(self.get_path(
            'binary', channel=self.channel,
            asset_sub_type=f'coordinates_transformed_decimated_{ratio}'),
            decimated_coordinates_transformed)

        decimated_labels = self.get('binary', channel=self.channel,
                                    asset_sub_type='labels').as_source[::ratio, :]
        cmp_io.write(self.get_path(
            'binary', channel=self.channel,
            asset_sub_type=f'labels_decimated_{ratio}'),
            decimated_labels)

        # Build the DataFrame
        df = pd.DataFrame({'id': decimated_labels[:, 0]})
        for i, l in enumerate('xyz'):
            df[l] = decimated_coordinates_raw[:, i]

        for i, l in enumerate('xyz'):
            df[f'{l}t'] = decimated_coordinates_transformed[:, i]

        unique_ids = np.sort(np.unique(decimated_labels))
        annotator = self.registration_processor.annotators[self.channel]
        color_map = {id_: annotator.find(id_, key='id')['rgb'] for id_ in unique_ids}
        df['color'] = df['id'].map(color_map)

        # WARNING: asset_sub_type is "fake" because it is not really cells.
        #  Rename to tract_coords or something
        #  and make sure to use it in the plots
        df.to_feather(self.get_path('cells', channel=self.channel,
                                    asset_sub_type=asset_sub_type, extension='.feather'))

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
        voxelize(self.get('binary', asset_sub_type='coordinates_transformed', channel=self.channel).source,
                 sink=self.get_path('density', channel=self.channel, asset_sub_type='counts'),
                 **voxelization_parameter
        )


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
        df_path = self.get_path('cells', channel=self.channel, extension='.feather')
        if not asset.exists or not df_path.exists():
            raise MissingRequirementException(f'plot_cells_3d_scatter_w_atlas_colors missing files:'
                                              f'image: {asset.path} {"not" if not asset.exists else ""} found'
                                              f'cells data frame {"not" if not df_path.exists() else ""} found')
        dv = q_plot_3d.plot(asset.path, title=f'{asset_properties["asset_type"].title()} and cells',  # FIXME: correct scaling for anisotropic if raw
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
        dv.scatter_coords = Scatter3D(coordinates, colors=df['color'].values,
                                      hemispheres=hemispheres, half_slice_thickness=0)
        dv.refresh()
        return [dv]

    def plot_voxelized_counts(self):
        asset = self.get('density', channel=self.channel, asset_sub_type='counts')
        if not asset.exists:
            raise MissingRequirementException(f'plot_voxelized_counts missing file: {asset.path} not found')
        dv = q_plot_3d.plot(asset.path, title=f'Voxelized counts', arrange=False, lut='flame')[0]
        return [dv]

# Required params:
# channel
# ['voxelization']['radii']
# ['clip_low']
# ['clip_high']
# ['decimation_ratio']
# ['slicing']  # For debug