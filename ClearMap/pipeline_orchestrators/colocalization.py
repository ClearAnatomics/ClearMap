import math
from typing import List, Optional

import numpy as np
import pandas as pd

from matplotlib.colors import to_hex
import pyqtgraph as pg

from ClearMap.IO.workspace2 import Workspace2
from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.Utils.utilities import sanitize_n_processes
from ClearMap.config.config_coordinator import ConfigCoordinator

from ClearMap.Analysis.Measurements import Voxelization as voxelization
from ClearMap.Analysis.colocalization.channel import Channel as ColocalizationChannel

from ClearMap.pipeline_orchestrators.generic_orchestrators import CompoundChannelPipelineOrchestrator
from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager
from ClearMap.pipeline_orchestrators.registration_orchestrator import RegistrationProcessor

from ClearMap.Visualization.Qt import Plot3d as q_plot_3d
from ClearMap.Visualization.Qt.widgets import Scatter3D


class ColocalizationProcessor(CompoundChannelPipelineOrchestrator):
    colocalization_channels: dict[ColocalizationChannel]
    config_name = 'colocalization'

    def __init__(self, sample_manager: Optional[SampleManager] = None,
                 config_coordinator: Optional[ConfigCoordinator] = None,
                 channels : Optional[List[str]] = None,
                 registration_processor: Optional[RegistrationProcessor] = None):
        super().__init__(config_coordinator)
        self.sample_manager = sample_manager
        self.channels: List[str] = channels
        self.registration_processor: Optional[RegistrationProcessor] = registration_processor
        self.workspace: Workspace2 | None = None

        self.filtered_table: Optional[pd.DataFrame] = None
        if channels is None:
            raise ClearMapValueError(f'No channels specified. Please provide a pair of channels to compare. '
                                     f'They must match the ones in the sample_params file.')
        if len(channels) != 2:
            raise ClearMapValueError(f'Please provide exactly two channels to compare. '
                                     f'They must match the ones in the sample_params file.')
        self.colocalization_channels: dict[str, ColocalizationChannel] = {}  # The objects that compute the colocalization from the colocalization package
        self.setup_finalised: bool = False
        self.setup(sample_manager, channels, registration_processor)

    def setup(self, sample_manager: Optional[SampleManager], channel_names: tuple[str],
              registration_processor: Optional[RegistrationProcessor] = None):
        self.channels = tuple(channel_names)
        if registration_processor is not None:
            self.registration_processor = registration_processor
        if sample_manager is not None:
            self.sample_manager = sample_manager
            self.workspace = sample_manager.workspace

            sample_id = self.sample_manager.sample_id
            self.workspace.ensure_pipeline('Colocalization', channel_id=self.channels, sample_id=sample_id,
                                           permute_channels=True, create_channel=True)

            self.finalise_setup()

    def finalise_setup(self):
        if self.setup_finalised:
            return
        finalised_channels = {chan: False for chan in self.channels}
        for channel in self.channels:
            resolution = self.sample_manager.get_channel_resolution(channel)
            try:
                self.colocalization_channels[channel] = ColocalizationChannel(
                    self.get('cells', channel=channel, asset_sub_type='shape').existing_path,
                    self.get_cells_df(channel),
                    voxel_dims=resolution)
                finalised_channels[channel] = True
            except FileNotFoundError as err:
                print(f'Colocalization {err}')
                self.setup_finalised = False
        self.setup_finalised = all(finalised_channels.values())

    def get_cells_df(self, channel):
        return pd.read_feather(self.get_path('cells', channel=channel))

    def compute_colocalization(self, channel_a, channel_b):
        # voxel_blob_diameter will also be used to compute the overlap
        voxel_blob_diameter = self.config['comparison']['particle_diameter']
        n_processes = sanitize_n_processes(self.config['performance']['n_processes'])

        report = self.colocalization_channels[channel_a].compare(self.colocalization_channels[channel_b],
                                                                 blob_diameter=voxel_blob_diameter,
                                                                 size_min=4*voxel_blob_diameter,
                                                                 size_max=6*voxel_blob_diameter, #  FIXME: add control for size_max
                                                                 processes=n_processes)
        report_path = self.get_path('colocalization', channel=(channel_a, channel_b),
                                    asset_sub_type='report')
        report.reset_index(inplace=True)  # WARNING: extract the index (no drop) to a separate column to allow saving to feather
        report.to_feather(report_path)

    def save_filtered_table(self, channel_a, channel_b):
        if self.filtered_table is not None:
            self.filtered_table.to_feather(self.get_path('colocalization', channel=(channel_a, channel_b),
                                                         asset_sub_type='filtered_report'))

    def plot_nearest_neighbors(self, channel_a, channel_b, parent=None):  # TODO: improve with line between particles
        channel_a_particle_coordinates, channel_b_particle_coordinates, channel_a_no_neighbour_coordinates = self.filter_table(
            channel_a, channel_b)

        # if physical coordinates
        # channel_a_particle_coordinates *= self.sample_manager.config['channels'][channel_a]['resolution']
        # channel_a_particle_coordinates *= self.sample_manager.config['channels'][channel_b]['resolution']

        lut = ['red', 'blue', 'orange', 'yellow', 'brown', 'pink', 'cyan', 'olive', 'grey']
        lut = [to_hex(col) for col in lut]
        n_colocalized_particles = len(channel_a_particle_coordinates)
        n_repeats = n_colocalized_particles / len(lut)
        colours = np.tile(lut, math.ceil(n_repeats))[:n_colocalized_particles]

        channel_a_particle_coordinates['colour'] = colours
        channel_a_particle_coordinates['symbol'] = '+'
        channel_b_particle_coordinates['colour'] = colours
        channel_b_particle_coordinates['symbol'] = 'd'
        channel_a_no_neighbour_coordinates['colour'] = to_hex('grey')
        channel_a_no_neighbour_coordinates['symbol'] = 'o'

        scatter_df = pd.concat([channel_a_particle_coordinates,
                                channel_b_particle_coordinates,
                                channel_a_no_neighbour_coordinates])

        self.filtered_table = scatter_df

        channel_a_stitched = self.get_path('stitched', channel=channel_a)
        channel_b_stitched = self.get_path('stitched', channel=channel_b)
        dv = q_plot_3d.plot([[channel_a_stitched, channel_b_stitched]],
                            title=f'Nearest Neighbours {channel_a} - {channel_b}',
                            arrange=False, parent=parent)[0]

        scatter = pg.ScatterPlotItem()
        dv.view.addItem(scatter)
        dv.scatter = scatter
        dv.scatter_coords = Scatter3D(scatter_df, half_slice_thickness=3,
                                      marker_size=self.config['comparison']['particle_diameter'] // 2)
        dv.refresh()

        return [dv]

    def filter_table(self, channel_a, channel_b):  # TODO: add options for which criteria ?/
        report = pd.read_feather(self.get_path('colocalization', channel=(channel_a, channel_b),
                                               asset_sub_type='report'))  # TODO: see if we cache
        maximum_distance = self.config['analysis']['max_particle_distance']
        within_distance_mask = report['closest blob distance'].values < maximum_distance
        chan_a = self.colocalization_channels[channel_a]
        channel_a_particle_coordinates = report.loc[within_distance_mask, [f'center of bounding box {ax}'
                                                                           for ax in chan_a.coord_names]]
        channel_a_particle_coordinates.columns = list(chan_a.coord_names)
        channel_b_particle_coordinates = report.loc[within_distance_mask, [f'closest blob center {ax}'
                                                                           for ax in chan_a.coord_names]]
        channel_b_particle_coordinates.columns = list(chan_a.coord_names)
        # outside_distance_particles = report[report['closest blob distance'] >= maximum_distance]
        channel_a_no_neighbour_coordinates = report.loc[~within_distance_mask, [f'center of bounding box {ax}'
                                                                                for ax in chan_a.coord_names]]
        channel_a_no_neighbour_coordinates.columns = list(chan_a.coord_names)
        return channel_a_particle_coordinates, channel_b_particle_coordinates, channel_a_no_neighbour_coordinates

    def plot_overlaps(self):
        pass

    def voxelize_filtered_table(self, channel_a, channel_b):
        coordinates, voxelization_parameter = self.get_voxelization_params(channel_a, channel_b)
        _ = self.voxelize_unweighted(channel_a, channel_b, coordinates, voxelization_parameter)

    def get_voxelization_params(self, channel_a, channel_b):
        voxelization_parameter = {
            'radius': self.config['voxelization']['radii'],
            'verbose': True
        }
        if self.workspace.debug:  # Path will use debug
            voxelization_parameter['shape'] = self.get('cells', channel=(channel_a, channel_b),
                                                       asset_sub_type='shape').shape()
        elif self.registration_processor.was_registered:
            voxelization_parameter['shape'] = self.get('atlas', channel=channel_a,
                                                       asset_sub_type='annotation').shape()
        else:
            voxelization_parameter['shape'] = self.sample_manager.resampled_shape(channel_a)
        channel_a_particle_coordinates, _, _ = self.filter_table(channel_a, channel_b)
        coordinates = channel_a_particle_coordinates[list(self.colocalization_channels[channel_a].coord_names)].values
        return coordinates, voxelization_parameter

    def voxelize_unweighted(self, channel_a, channel_b, coordinates, voxelization_parameter):
        """
        Voxelize un weighted i.e. for cell counts

        Parameters
        ----------
        channel_a: str
            Name of the first channel
        channel_b: str
            Name of the second channel
        coordinates: str, array or Source
            Source of point of nxd coordinates.
        voxelization_parameter:  dict
            Dictionary to be passed to voxelization.voxelise (i.e. with these optional keys:
                shape, dtype, weights, method, radius, kernel, processes, verbose

        Returns
        -------
        coordinates, counts_file_path: np.array, str
        """
        counts_asset = self.get('density', channel=(channel_a, channel_b), asset_sub_type='counts')
        counts_asset.delete(missing_ok=True)
        self.set_watcher_step('Unweighted voxelisation')
        voxelization.voxelize(coordinates, sink=counts_asset.path, **voxelization_parameter)  # WARNING: prange
        self.update_watcher_main_progress()
        # uncrusted_coordinates = self.remove_crust(coordinates)  # WARNING: currently causing issues
        #         density_path = self.get_path('density', channel=self.channel, asset_sub_type='counts_wcrust')
        #         voxelization.voxelize(uncrusted_coordinates, sink=density_path, **voxelization_parameter)   # WARNING: prange
        return coordinates, counts_asset.path
