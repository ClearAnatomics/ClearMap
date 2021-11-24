"""
This is the part that is common to both pipelines (includes file conversion, stitching and registration)
"""
import os
import sys

import numpy as np

# noinspection PyPep8Naming
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import ClearMap.Settings as settings
# noinspection PyPep8Naming
import ClearMap.Alignment.Elastix as elastix
# noinspection PyPep8Naming
import ClearMap.Alignment.Annotation as annotation
# noinspection PyPep8Naming
import ClearMap.IO.Workspace as workspace
# noinspection PyPep8Naming
import ClearMap.IO.IO as clearmap_io
# noinspection PyPep8Naming
import ClearMap.Visualization.Qt.Plot3d as plot_3d
# noinspection PyPep8Naming
import ClearMap.Alignment.Resampling as resampling
# noinspection PyPep8Naming
import ClearMap.Alignment.Stitching.StitchingRigid as stitching_rigid
# noinspection PyPep8Naming
import ClearMap.Alignment.Stitching.StitchingWobbly as stitching_wobbly
from ClearMap.IO.metadata import define_auto_stitching_params, define_auto_resolution, get_file_path
from ClearMap.config.config_loader import get_configs


class PreProcessor(object):
    def __init__(self):
        self.resources_directory = None
        self.workspace = None
        self.machine_config = {}
        self.sample_config = {}
        self.processing_config = {}
        self.align_channels_affine_file = ''
        self.align_reference_affine_file = ''
        self.align_reference_bspline_file = ''
        self.annotation_file_path = ''
        self.reference_file_path = ''
        self.distance_file_path = ''
        # if not any('SPYDER' in name for name in os.environ):
        #     pyqtgraph.mkQApp()
        # self.setup(cfg_paths)
        # self.setup_atlases()

    def setup(self, cfg_paths):
        self.resources_directory = settings.resources_path
        self.set_configs(cfg_paths)
        src_directory = os.path.expanduser(self.sample_config['base_directory'])

        self.workspace = workspace.Workspace(self.processing_config['pipeline_name'], directory=src_directory)
        src_paths = {k: v for k, v in self.sample_config['src_paths'].items() if v is not None}
        self.workspace.update(**src_paths)
        self.workspace.info()

    def set_configs(self, cfg_paths):
        cfg_paths = [os.path.expanduser(p) for p in cfg_paths]
        self.machine_config, self.sample_config, self.processing_config = get_configs(*cfg_paths)

    def setup_atlases(self):  # TODO: add possibility to load custom reference file (i.e. defaults to None in cfg)
        results = annotation.prepare_annotation_files(
            slicing=(self.sample_config['slice_x'], self.sample_config['slice_y'], self.sample_config['slice_z']),
            orientation=self.sample_config['orientation'],
            overwrite=False, verbose=True)
        self.annotation_file_path, self.reference_file_path, self.distance_file_path = results
        atlas_cfg = self.processing_config['registration']['atlas']
        align_dir = os.path.join(self.resources_directory, atlas_cfg['align_files_folder'])
        self.align_channels_affine_file = os.path.join(align_dir, atlas_cfg['align_channels_affine_file'])
        self.align_reference_affine_file = os.path.join(align_dir, atlas_cfg['align_reference_affine_file'])
        self.align_reference_bspline_file = os.path.join(align_dir, atlas_cfg['align_reference_bspline_file'])

    def run(self):
        self.convert_data()  # TODO: make optional
        self.stitch()
        self.resample_for_registration()
        self.align()
        return self.workspace, self.get_configs(), self.get_atlas_files()

    def convert_data(self):
        """Convert raw data to npy file"""
        source = self.workspace.source('raw')
        sink = self.workspace.filename('stitched')
        clearmap_io.delete_file(sink)
        clearmap_io.convert(source, sink, processes=self.machine_config['n_processes_file_conv'], verbose=True)

    def stitch(self):
        stitching_cfg = self.processing_config['stitching']
        if not stitching_cfg['rigid']['skip']:
            layout = self._stitch_rigid()

        if not stitching_cfg['wobbly']['skip']:
            self._stitch_wobbly(layout)

        self.plot_stitching_results()

        if not stitching_cfg['output_conversion']['skip']:
            self.convert_to_image_format()

    def convert_to_image_format(self):
        fmt = self.processing_config['stitching']['output_conversion']['format'].strip('.')
        clearmap_io.convert_files(self.workspace.file_list('stitched', extension='npy'),  # FIXME: implement raw and arteries
                                  extension=fmt, processes=12, verbose=True)

    def _stitch_rigid(self):
        stitching_cfg = self.processing_config['stitching']
        overlaps, projection_thickness = define_auto_stitching_params(self.workspace.source('raw').file_list[0],
                                                                      stitching_cfg)
        layout = stitching_wobbly.WobblyLayout(expression=self.workspace.filename('raw'), tile_axes=['X', 'Y'],
                                               overlaps=overlaps)
        if stitching_cfg['rigid']['background_pixels'] is None:
            background_params = stitching_cfg['rigid']['background_level']
        else:
            background_params = (stitching_cfg['rigid']['background_level'],
                                 stitching_cfg['rigid']['background_pixels'])
        max_shifts = [stitching_cfg['rigid']['max_shifts_{}'.format(ax)] for ax in ('x', 'y', 'z')]
        stitching_rigid.align_layout_rigid_mip(layout, depth=projection_thickness, max_shifts=max_shifts,
                                               ranges=[None, None, None], background=background_params,
                                               clip=25000, processes='!serial', verbose=True)  # TODO: check processes
        # stitching_rigid.place_layout(layout, method='optimization', min_quality=-np.inf, lower_to_origin=True,
        #                              verbose=True)
        layout.place(method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)

        # lyt = TiledLayout(tiling, overlaps=overlaps)
        # print(lyt.tile_positions)
        # lyt.center_tile_source()
        # lyt.source_positions()
        #
        # lyt.align_on_tiling(max_shifts=max_shifts[0], verbose=True)  # TODO: chekc why axis 0
        # lyt.place(method='optimization', lower_to_origin=True, verbose=True)
        # lyt.plot_alignments()

        # slc = layout.slice_along_axis(50, axis=2)
        #
        # slc.align(max_shifts=max_shifts[2][1], verbose=True, processes=None)  # TODO: check max_shifts
        # slc.place(verbose=True)
        # slc.plot_alignments()

        # layout.plot_alignments()  # TODO: TEST
        # plt.show()

        stitching_rigid.save_layout(self.workspace.filename('layout', postfix='aligned_axis'), layout)
        return layout

    def _stitch_wobbly(self, layout):
        stitching_cfg = self.processing_config['stitching']
        # layout = st.load_layout(_workspace.filename('layout', postfix='aligned_axis'))  # TODO: check if can be used to avoid passing arg
        max_shifts = [stitching_cfg['wobbly']['max_shifts_{}'.format(ax)] for ax in ('x', 'y', 'z')]
        stitching_wobbly.align_layout(layout, axis_range=(None, None, 3), max_shifts=max_shifts, axis_mip=None,
                                      validate=dict(method='foreground', valid_range=(200, None), size=None),
                                      # FIXME: replace valid_range
                                      prepare=dict(method='normalization', clip=None, normalize=True),
                                      validate_slice=dict(method='foreground', valid_range=(200, 20000), size=1500),
                                      # FIXME: replace valid_range
                                      prepare_slice=None,
                                      find_shifts=dict(method='tracing', cutoff=3 * np.sqrt(2)),
                                      processes=None, verbose=True)
        # noinspection Duplicates
        stitching_rigid.save_layout(self.workspace.filename('layout', postfix='aligned'), layout)
        # %% Wobbly placement
        # layout = st.load_layout(_workspace.filename('layout', postfix='aligned'));
        stitching_wobbly.place_layout(layout, min_quality=-np.inf,
                                      method='optimization',
                                      smooth=dict(method='window', window='bartlett', window_length=100, binary=None),
                                      smooth_optimized=dict(method='window', window='bartlett',
                                                            window_length=20, binary=10),
                                      fix_isolated=False, lower_to_origin=True,
                                      processes=None, verbose=True)
        stitching_rigid.save_layout(self.workspace.filename('layout', postfix='placed'), layout)
        # %% Wobbly stitching
        layout = stitching_rigid.load_layout(self.workspace.filename('layout', postfix='placed'))
        stitching_wobbly.stitch_layout(layout, sink=self.workspace.filename('stitched'),
                                       method='interpolation', processes='!serial', verbose=True)

    def resample_for_registration(self):
        resampling_cfg = self.processing_config['registration']['resampling']
        if not resampling_cfg['skip']:
            default_resample_parameter = {
                "processes": resampling_cfg['processes'],
                # "verbose": False  # FIXME:
                "verbose": resampling_cfg['verbose']
            }

            # Raw
            clearmap_io.delete_file(self.workspace.filename('resampled'))
            src_res = define_auto_resolution(self.workspace.source('raw').file_list[0],
                                             self.sample_config['resolutions']['raw'])

            result = resampling.resample(self.workspace.filename('stitched'),
                                     source_resolution=src_res,
                                     sink=self.workspace.filename('resampled'),
                                     sink_resolution=resampling_cfg['raw_sink_resolution'],
                                     **default_resample_parameter)
            assert result.array.max() != 0, 'Resampled raw has no data'
            if resampling_cfg['plot_raw']:  # FIXME: probably need to change params for UI
                plot_3d.plot(self.workspace.filename('resampled'))

            # Autofluorescence
            auto_fluo_path = self.workspace.source('autofluorescence').file_list[0]
            auto_res = define_auto_resolution(auto_fluo_path, self.sample_config['resolutions']['autofluorescence'])
            result = resampling.resample(self.workspace.filename('autofluorescence'),
                                         source_resolution=auto_res,
                                         sink=self.workspace.filename('resampled', postfix='autofluorescence'),
                                         sink_resolution=resampling_cfg['autofluo_sink_resolution'],
                                         **default_resample_parameter)
            assert result.array.max() != 0, 'Resampled autofluorescence has no data'
            if resampling_cfg['plot_autofluo']:
                plot_3d.plot([self.workspace.filename('resampled'),
                              self.workspace.filename('resampled', postfix='autofluorescence')])

    def align(self):
        self.__align_resampled_to_auto()
        self.__align_auto_to_ref()

    def __align_resampled_to_auto(self):
        align_channels_parameter = {
            # moving and reference images
            "moving_image": self.workspace.filename('resampled', postfix='autofluorescence'),
            "fixed_image": self.workspace.filename('resampled'),

            # elastix parameter files for alignment
            "affine_parameter_file": self.align_channels_affine_file,
            "bspline_parameter_file": None,

            # directory of the alignment result '/home/nicolas.renier/Documents/ClearMap_Resources/Par0000affine.txt'
            "result_directory": self.workspace.filename('resampled_to_auto')
        }
        elastix.align(**align_channels_parameter)

    def __align_auto_to_ref(self):
        align_reference_parameter = {
            # moving and reference images
            "moving_image": self.reference_file_path,
            "fixed_image": self.workspace.filename('resampled', postfix='autofluorescence'),

            # elastix parameter files for alignment
            "affine_parameter_file": self.align_reference_affine_file,
            "bspline_parameter_file": self.align_reference_bspline_file,
            # directory of the alignment result
            "result_directory": self.workspace.filename('auto_to_reference')
        }
        for k, v in align_reference_parameter.items():
            if not v:
                raise ValueError('Registration missing parameter "{}"'.format(k))
        elastix.align(**align_reference_parameter)

    def get_configs(self):
        cfg = {
            'machine': self.machine_config,
            'sample': self.sample_config,
            'processing': self.processing_config
        }
        return cfg

    def get_atlas_files(self):
        if not self.annotation_file_path:
            self.setup_atlases()
        atlas_files = {
            'annotation': self.annotation_file_path,
            'reference': self.reference_file_path,
            'distance': self.distance_file_path
        }
        return atlas_files

    def plot_stitching_results(self, parent=None):
        cfg = self.processing_config['stitching']['preview']
        paths = []
        title = ''
        if cfg['raw']:
            paths.append(self.workspace.filename('stitched'))
            title += 'Raw stitched'
        if cfg['arteries']:
            paths.append(self.workspace.filename('stitched', postfix='arteries'))  # FIXME: hard coded postfix
            title += ' and arteries stitched'
        if len(paths) != 2:
            paths = paths[0]
        dvs = plot_3d.plot(paths, arange=False, lut='white', parent=parent)
        return dvs

    def reload_processing_cfg(self):
        self.processing_config.reload()


if __name__ == '__main__':
    preprocessor = PreProcessor()
    preprocessor.setup(sys.argv[1:3])
    preprocessor.setup_atlases()
    preprocessor.run()
