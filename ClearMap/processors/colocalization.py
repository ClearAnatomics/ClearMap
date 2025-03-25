import math

import numpy as np
import pandas as pd

from matplotlib.colors import to_hex
import pyqtgraph as pg

from ClearMap.Analysis.colocalization.channel import Channel as ColocalizationChannel
from ClearMap.IO.assets_specs import ChannelSpec
from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.Visualization.Qt.widgets import Scatter3D
from ClearMap.processors.generic_tab_processor import TabProcessor

from ClearMap.Visualization.Qt import Plot3d as q_plot_3d


class ColocalizationProcessor(TabProcessor):
    colocalization_channels: dict[ColocalizationChannel]

    def __init__(self, sample_manager=None, channels=None):
        super().__init__()
        self.filtered_table = None
        self.sample_config = None
        self.processing_config = None
        self.machine_config = None
        self.sample_manager = None
        self.registration_processor = None
        self.workspace = None
        if channels is None:
            raise ClearMapValueError(f'No channels specified. Please provide a pair of channels to compare. '
                                     f'They must match the ones in the sample_params file.')
        if len(channels) != 2:
            raise ClearMapValueError(f'Please provide exactly two channels to compare. '
                                     f'They must match the ones in the sample_params file.')
        self.channels = channels
        self.colocalization_channels = {}  # The objects that compute the colocalization from the colocalization package
        self.setup(sample_manager, channels)

    def setup(self, sample_manager, channel_names):
        self.channels = channel_names
        self.sample_config = None
        if sample_manager is not None:
            self.sample_manager = sample_manager
            self.workspace = sample_manager.workspace
            configs = sample_manager.get_configs()
            self.sample_config = configs['sample']
            self.machine_config = configs['machine']
            channel_name = '-'.join(self.channels).lower()
            self.processing_config = self.sample_manager.config_loader.get_cfg('colocalization')['channels'][channel_name]

            if self.channels not in self.workspace.asset_collections.keys():
                # FIXME: ugly. should be handled by add_pipeline with missing_ok=True
                sample_id = self.sample_manager.config['sample_id']
                self.workspace.add_channel(ChannelSpec(self.channels, 'colocalization'),
                                           sample_id=sample_id)
                self.workspace.add_pipeline('Colocalization', self.channels, sample_id=sample_id)

            self.set_progress_watcher(self.sample_manager.progress_watcher)


        for channel in self.channels:
            resolution = self.sample_manager.config['channels'][channel]['resolution']
            self.colocalization_channels[channel] = ColocalizationChannel(
                self.get('cells', channel=channel, asset_sub_type='shape').existing_path,
                self.get_cells_df(channel), voxel_dims=resolution)

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

    def get_cells_df(self, channel):
        return pd.read_feather(self.get_path('cells', channel=channel))

    def compute_colocalization(self):
        # size_min = 10
        # size_max = 16
        # overlap = 5
        # n_processes = 10
        self.reload_config()
        compound_channel_name = ('-'.join(self.channels).lower())
        channel_a, channel_b = self.channels
        # voxel_blob_diameter will also be used to compute the overlap
        voxel_blob_diameter = self.processing_config['comparison']['particle_diameter']
        report = self.colocalization_channels[channel_a].compare(self.colocalization_channels[channel_b],
                                                                 blob_diameter=voxel_blob_diameter,
                                                                 size_min=4*voxel_blob_diameter,
                                                                 size_max=8*voxel_blob_diameter)
        report_path = self.get_path('colocalization', channel=(channel_a, channel_b),
                                    asset_sub_type='report')
        report.reset_index(inplace=True)  # WARNING: extract the index (no drop) to a separate column to allow saving to feather
        report.to_feather(report_path)

    def save_filtered_table(self):
        if self.filtered_table is not None:
            self.filtered_table.to_feather(self.get_path('colocalization', channel=self.channels,
                                                         asset_sub_type='filtered_report'))

    def plot_nearest_neighbors(self, channel_a, channel_b, parent=None):  # TODO: improve with line between particles
        report = pd.read_feather(self.get_path('colocalization', channel=(channel_a, channel_b),
                                               asset_sub_type='report'))   # TODO: see if we cache
        maximum_distance = self.processing_config['analysis']['max_particle_distance']
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
        dv.scatter_coords = Scatter3D(scatter_df, half_slice_thickness=3)
        dv.refresh()

        return [dv]


    def plot_overlaps(self):
        pass
