"""
sample_preparation
==================

This is the part that is common to both pipelines to process the raw images.
It includes file conversion, stitching and registration
"""
import os
import re
import sys
from concurrent.futures.process import BrokenProcessPool

import numpy as np

# noinspection PyPep8Naming
import matplotlib
import tifffile

from ClearMap.Utils.utilities import runs_on_ui
from ClearMap.gui.gui_utils import TmpDebug

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
from ClearMap.IO.metadata import define_auto_stitching_params, define_auto_resolution, pattern_finders_from_base_dir
from ClearMap.config.config_loader import get_configs, ConfigLoader


__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


class CanceledProcessing(BrokenProcessPool):  # TODO: better inheritance
    pass


class TabProcessor:
    def __init__(self):
        self.stopped = False
        self.progress_watcher = None
        self.workspace = None
        self.machine_config = {}

    def set_progress_watcher(self, watcher):
        self.progress_watcher = watcher

    def update_watcher_progress(self, val):
        if self.progress_watcher is not None:
            self.progress_watcher.increment(val)

    def update_watcher_main_progress(self, val=1):
        if self.progress_watcher is not None:
            self.progress_watcher.increment_main_progress(val)

    def prepare_watcher_for_substep(self, counter_size, pattern, title, increment_main=False):
        """
        Prepare the progress watcher for the coming processing step. The watcher will in turn signal changes to the
        progress bar

        Arguments
        ---------
        counter_size: int
            The progress bar maximum
        pattern: str or re.Pattern or (str, re.Pattern)
            The string to search for in the log to signal an increment of 1
        title: str
            The title of the step for the progress bar
        increment_main: bool
            Whether a new step should be added to the main progress bar
        """
        if self.progress_watcher is not None:
            self.progress_watcher.prepare_for_substep(counter_size, pattern, title)
            if increment_main:
                self.progress_watcher.increment_main_progress()

    def stop_process(self):  # REFACTOR: put in parent class ??
        self.stopped = True
        if hasattr(self.workspace, 'executor') and self.workspace.executor is not None:
            if sys.version_info[:2] >= (3, 9):
                self.workspace.executor.shutdown(cancel_futures=True)  # The new clean version
            else:
                self.workspace.executor.immediate_shutdown()  # Dirty but we have no choice in python < 3.9
            self.workspace.executor = None
            # raise BrokenProcessPool
        elif hasattr(self.workspace, 'process') and self.workspace.process is not None:
            self.workspace.process.terminate()
            # self.workspace.process.wait()
            raise CanceledProcessing

    @property
    def verbose(self):
        return self.machine_config['verbosity'] == 'debug'

    def run(self):
        raise NotImplementedError

    # def setup(self):
    #     pass


class PreProcessor(TabProcessor):
    def __init__(self):
        super().__init__()
        self.config_loader = None
        self.src_directory = None
        self.resources_directory = None
        self.sample_config = {}
        self.processing_config = {}
        self.align_channels_affine_file = ''
        self.align_reference_affine_file = ''
        self.align_reference_bspline_file = ''
        self.default_annotation_file_path = None
        self.default_hemispheres_file_path = None
        self.default_reference_file_path = None
        self.default_distance_file_path = None
        self.annotation_file_path = ''
        self.hemispheres_file_path = ''
        self.reference_file_path = ''
        self.distance_file_path = ''
        self.__align_auto_to_ref_re = re.compile(r"\d+\s-?\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+")
        self.__align_resampled_to_auto_re = re.compile(r"\d+\s-\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+\s\d+\.\d+")
        self.__resample_re = ('Resampling: resampling',
                              re.compile(r".*?Resampling:\sresampling\saxes\s.+\s?,\sslice\s.+\s/\s\d+"))
        self.__wobbly_stitching_place_re = 'done constructing constraints for component'
        self.__wobbly_stitching_algin_lyt_re = ('Alignment: Wobbly alignment',
                                                re.compile(r"Alignment:\sWobbly alignment \(\d+, \d+\)->\(\d+, \d+\) "
                                                           r"along axis [0-3] done: elapsed time: \d+:\d{2}:\d{2}.\d+"))
        self.__wobbly_stitching_stitch_re = ('Stitching: stitching',
                                             re.compile(r'Stitching: stitching wobbly slice \d+/\d+'))
        self.__rigid_stitching_align_re = ('done',
                                           re.compile(r"Alignment: aligning \(\d+, \d+\) with \(\d+, \d+\), alignment"
                                                      r" pair \d+/\d+ done, shift = \(-?\d+, -?\d+, -?\d+\),"
                                                      r" quality = -\d+\.\d+e\+\d+!"))
        # if not runs_on_spyder():
        #     pyqtgraph.mkQApp()

    def setup(self, cfgs, watcher=None, convert_tiles=True):
        """

        Parameters
        ----------
        cfgs tuple of (machine_cfg_path, sample_cfg_path, processing_fg_path) or
            (machine_cfg, sample_cfg, processing_cfg)

        Returns
        -------

        """
        self.resources_directory = settings.resources_path
        if all([isinstance(cfg, str) for cfg in cfgs]):
            self.set_configs(cfgs)
        else:  # Directly the config
            self.machine_config, self.sample_config, self.processing_config = cfgs
        self.src_directory = os.path.dirname(self.sample_config.filename)
        self.config_loader = ConfigLoader(self.src_directory)

        if watcher is not None:
            self.progress_watcher = watcher

        self.workspace = workspace.Workspace(self.processing_config['pipeline_name'], directory=self.src_directory)
        self.workspace.tmp_debug = TmpDebug(self.workspace)
        src_paths = {k: v for k, v in self.sample_config['src_paths'].items() if v is not None}
        self.workspace.update(**src_paths)
        self.workspace.info()
        if convert_tiles:
            self.convert_tiles()
        # FIXME: check if setup_atlas should go here

    def unpack_atlas(self, atlas_base_name):
        res = annotation.uncompress_atlases(atlas_base_name)
        self.default_annotation_file_path, self.default_hemispheres_file_path, \
            self.default_reference_file_path, self.default_distance_file_path = res

    def stack_columns(self):
        pattern_finders = pattern_finders_from_base_dir(self.src_directory)
        for pat_finder in pattern_finders:
            for y in pat_finder.y_values:
                for x in pat_finder.x_values:
                    stack = np.vstack([tifffile.imread(f_path) for f_path in pat_finder.get_sub_tiff_paths(x=x, y=y)])
                    new_path = pat_finder.sub_pattern_str(x=x, y=y)
                    tifffile.imsave(new_path, stack)

    @property
    def aligned_autofluo_path(self):
        return os.path.join(self.workspace.filename('auto_to_reference'), 'result.1.mhd')
    
    @property
    def raw_stitched_shape(self):
        if self.resampled_shape is not None:
            raw_resampled_res_from_cfg = np.array(self.processing_config['registration']['resampling']['raw_sink_resolution'])
            raw_res_from_cfg = np.array(self.sample_config['resolutions']['raw'])
            return self.resampled_shape * (raw_resampled_res_from_cfg / raw_res_from_cfg)
        else:
            return clearmap_io.shape(self.workspace.filename('stitched'))

    @property
    def resampled_shape(self):
        if os.path.exists(self.workspace.filename('resampled')):
            return clearmap_io.shape(self.workspace.filename('resampled'))

    def convert_tiles(self):
        """
        Convert list of input files to numpy files for efficiency reasons

        Returns
        -------

        """
        if self.stopped:
            return
        if self.use_npy():
            file_list = self.workspace.file_list('raw')
            if not file_list or os.path.splitext(file_list[0])[-1] == '.tif':
                try:
                    clearmap_io.convert_files(self.workspace.file_list('raw', extension='tif'), extension='npy',
                                              processes=self.machine_config['n_processes_file_conv'],
                                              workspace=self.workspace, verbose=self.verbose)
                except BrokenProcessPool:
                    print('File conversion canceled')
                    return
                if self.sample_config['src_paths']['arteries']:
                    try:
                        clearmap_io.convert_files(self.workspace.file_list('arteries', extension='tif'),
                                                  extension='npy',
                                                  processes=self.machine_config['n_processes_file_conv'],
                                                  workspace=self.workspace, verbose=self.verbose)
                    except BrokenProcessPool:
                        print('File conversion canceled')
                        return

    def use_npy(self):
        return self.processing_config['conversion']['use_npy'] or \
               self.workspace.filename('raw').endswith('.npy') or \
               os.path.exists(self.workspace.filename('raw', extension='npy'))

    def set_configs(self, cfg_paths):
        cfg_paths = [os.path.expanduser(p) for p in cfg_paths]
        self.machine_config, self.sample_config, self.processing_config = get_configs(*cfg_paths)

    def setup_atlases(self):  # TODO: add possibility to load custom reference file (i.e. defaults to None in cfg)
        x_slice = slice(None) if self.sample_config['slice_x'] is None else slice(*self.sample_config['slice_x'])
        y_slice = slice(None) if self.sample_config['slice_y'] is None else slice(*self.sample_config['slice_y'])
        z_slice = slice(None) if self.sample_config['slice_z'] is None else slice(*self.sample_config['slice_z'])
        xyz_slicing = (x_slice, y_slice, z_slice)
        res = annotation.prepare_annotation_files(
            slicing=xyz_slicing,
            orientation=self.sample_config['orientation'],
            annotation_file=self.default_annotation_file_path, hemispheres_file=self.default_hemispheres_file_path,
            reference_file=self.default_reference_file_path, distance_to_surface_file=self.default_distance_file_path,
            hemispheres=True,
            overwrite=False, verbose=True)
        self.annotation_file_path, self.hemispheres_file_path, self.reference_file_path, self.distance_file_path = res

        self.update_watcher_main_progress()
        atlas_cfg = self.processing_config['registration']['atlas']
        align_dir = os.path.join(self.resources_directory, atlas_cfg['align_files_folder'])
        self.align_channels_affine_file = os.path.join(align_dir, atlas_cfg['align_channels_affine_file'])
        self.align_reference_affine_file = os.path.join(align_dir, atlas_cfg['align_reference_affine_file'])
        self.align_reference_bspline_file = os.path.join(align_dir, atlas_cfg['align_reference_bspline_file'])

    def run(self):
        self.stitch()
        self.resample_for_registration()
        self.align()
        return self.workspace, self.get_configs(), self.get_atlas_files()

    def stitch(self):
        if self.stopped:
            return
        stitching_cfg = self.processing_config['stitching']
        if not stitching_cfg['rigid']['skip']:
            self.stitch_rigid()

        if not stitching_cfg['wobbly']['skip']:
            self.stitch_wobbly()

        if self.stopped:
            return
        self.plot_stitching_results()

        if not stitching_cfg['output_conversion']['skip']:
            self.convert_to_image_format()

    def convert_to_image_format(self):  # TODO: support progress
        """
        Convert (optionally) to image formats readable by e.g. Fiji
        -------

        """
        if self.stopped:
            return
        fmt = self.processing_config['stitching']['output_conversion']['format'].strip('.')
        if self.processing_config['stitching']['output_conversion']['raw']:
            try:
                clearmap_io.convert_files(self.workspace.file_list('stitched', extension='npy'),
                                          extension=fmt, processes=self.machine_config['n_processes_file_conv'],
                                          workspace=self.workspace, verbose=True)
            except BrokenProcessPool:
                print('File conversion canceled')
                return
        if self.processing_config['stitching']['output_conversion']['arteries']:
            try:
                clearmap_io.convert_files(self.workspace.file_list('stitched', postfix='arteries', extension='npy'),
                                          extension=fmt, processes=self.machine_config['n_processes_file_conv'],
                                          workspace=self.workspace, verbose=True)
            except BrokenProcessPool:
                print('File conversion canceled')
                return

    @property
    def was_stitched_rigid(self):
        return os.path.exists(self.workspace.filename('layout', postfix='aligned_axis'))

    @property
    def was_registered(self):
        # return os.path.exists(self.workspace.filename('resampled_to_auto'))
        return os.path.exists(self.aligned_autofluo_path)

    @property
    def n_rigid_steps_to_run(self):
        return int(not self.processing_config['stitching']['rigid']['skip'])

    def stitch_rigid(self, force=False):
        if force:
            self.stopped = False
        if self.stopped:
            return
        stitching_cfg = self.processing_config['stitching']
        overlaps, projection_thickness = define_auto_stitching_params(self.workspace.source('raw').file_list[0],
                                                                      stitching_cfg)
        layout = self.get_wobbly_layout(overlaps)
        if stitching_cfg['rigid']['background_pixels'] is None:
            background_params = stitching_cfg['rigid']['background_level']
        else:
            background_params = (stitching_cfg['rigid']['background_level'],
                                 stitching_cfg['rigid']['background_pixels'])
        max_shifts = [stitching_cfg['rigid']['max_shifts_{}'.format(ax)] for ax in ('x', 'y', 'z')]
        self.prepare_watcher_for_substep(len(layout.alignments), self.__rigid_stitching_align_re, 'Align layout rigid')
        try:
            stitching_rigid.align_layout_rigid_mip(layout, depth=projection_thickness, max_shifts=max_shifts,
                                                   ranges=[None, None, None], background=background_params,
                                                   clip=25000, processes=self.machine_config['n_processes_stitching'],
                                                   workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Stitching canceled')
            return
        layout.place(method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)
        self.update_watcher_main_progress()

        # layout.plot_alignments()  # TODO: TEST
        # plt.show()

        stitching_rigid.save_layout(self.workspace.filename('layout', postfix='aligned_axis'), layout)
        self.layout = layout

    def get_wobbly_layout(self, overlaps=None):
        if overlaps is None:
            overlaps, projection_thickness = define_auto_stitching_params(self.workspace.source('raw').file_list[0],
                                                                          self.processing_config['stitching'])
        if self.use_npy():
            raw_path = self.workspace.filename('raw', extension='npy')
        else:
            raw_path = self.workspace.filename('raw')
        layout = stitching_wobbly.WobblyLayout(expression=raw_path, tile_axes=['X', 'Y'], overlaps=overlaps)
        return layout

    @property
    def n_wobbly_steps_to_run(self):
        return int(not self.processing_config['stitching']['wobbly']['skip']) * 3

    def __align_layout_wobbly(self, layout):
        stitching_cfg = self.processing_config['stitching']
        max_shifts = [stitching_cfg['wobbly']['max_shifts_{}'.format(ax)] for ax in ('x', 'y', 'z')]

        n_pairs = len(layout.alignments)
        self.prepare_watcher_for_substep(n_pairs, self.__wobbly_stitching_algin_lyt_re, 'Align layout wobbly')
        try:
            stitching_wobbly.align_layout(layout, axis_range=(None, None, 3), max_shifts=max_shifts, axis_mip=None,
                                          validate=dict(method='foreground', valid_range=(200, None), size=None),
                                          # FIXME: replace valid_range
                                          prepare=dict(method='normalization', clip=None, normalize=True),
                                          validate_slice=dict(method='foreground', valid_range=(200, 20000), size=1500),
                                          # FIXME: replace valid_range
                                          prepare_slice=None,
                                          find_shifts=dict(method='tracing', cutoff=3 * np.sqrt(2)),
                                          processes=self.machine_config['n_processes_stitching'],
                                          workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return
        self.update_watcher_main_progress()

    def __place_layout_wobbly(self, layout):
        self.prepare_watcher_for_substep(len(layout.alignments) / 2,  # WARNING: bad estimation
                                         self.__wobbly_stitching_place_re, 'Place layout wobbly')
        try:
            stitching_wobbly.place_layout(layout, min_quality=-np.inf,
                                          method='optimization',
                                          smooth=dict(method='window', window='bartlett', window_length=100,
                                                      binary=None),
                                          smooth_optimized=dict(method='window', window='bartlett',
                                                                window_length=20, binary=10),
                                          fix_isolated=False, lower_to_origin=True,
                                          processes=self.machine_config['n_processes_stitching'],
                                          workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return
        self.update_watcher_main_progress()

    def __stitch_layout_wobbly(self):
        layout = stitching_rigid.load_layout(self.workspace.filename('layout', postfix='placed'))
        n_slices = len(self.workspace.file_list('autofluorescence'))  # TODO: find better proxy
        self.prepare_watcher_for_substep(n_slices, self.__wobbly_stitching_stitch_re, 'Stitch layout wobbly', True)
        try:
            stitching_wobbly.stitch_layout(layout, sink=self.workspace.filename('stitched'), method='interpolation',
                                           processes=self.machine_config['n_processes_stitching'],
                                           workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return

        if self.processing_config['stitching']['run']['arteries']:
            self.prepare_watcher_for_substep(n_slices, self.__wobbly_stitching_stitch_re,
                                             'Stitch layout wobbly arteries', True)
            try:
                if self.use_npy():
                    layout.replace_source_location(self.workspace.filename('raw', extension='npy'),
                                                   self.workspace.filename('arteries', extension='npy'))
                else:
                    layout.replace_source_location(self.workspace.filename('raw'), self.workspace.filename('arteries'))
                stitching_wobbly.stitch_layout(layout, sink=self.workspace.filename('stitched', postfix='arteries'),
                                               method='interpolation',
                                               processes=self.machine_config['n_processes_stitching'],
                                               workspace=self.workspace, verbose=True)
            except BrokenProcessPool:
                print('Wobbly stitching arteries canceled')
                return
        self.update_watcher_main_progress()

    def stitch_wobbly(self, force=False):
        if force:
            self.stopped = False
        if self.stopped:
            return
        layout = stitching_rigid.load_layout(self.workspace.filename('layout', postfix='aligned_axis'))
        self.__align_layout_wobbly(layout)
        if self.stopped:
            return
        stitching_rigid.save_layout(self.workspace.filename('layout', postfix='aligned'), layout)

        # layout = st.load_layout(_workspace.filename('layout', postfix='aligned'))  # FIXME: check if required
        self.__place_layout_wobbly(layout)
        if self.stopped:
            return
        stitching_rigid.save_layout(self.workspace.filename('layout', postfix='placed'), layout)

        self.__stitch_layout_wobbly()
        if self.stopped:
            return

    def __resample_raw(self):
        resampling_cfg = self.processing_config['registration']['resampling']
        default_resample_parameter = {
            "processes": resampling_cfg['processes'],
            "verbose": resampling_cfg['verbose']
        }  # WARNING: duplicate (use method ??)
        clearmap_io.delete_file(self.workspace.filename('resampled'))  # FIXME:
        f_list = self.workspace.source('raw').file_list
        if f_list:
            src_res = define_auto_resolution(f_list[0], self.sample_config['resolutions']['raw'])
        else:
            src_res = self.sample_config['resolutions']['raw']

        n_planes = len(self.workspace.file_list('autofluorescence'))  # TODO: find more elegant solution for counter
        self.prepare_watcher_for_substep(n_planes, self.__resample_re, 'Resampling raw')
        try:
            result = resampling.resample(self.workspace.filename('stitched'),
                                         source_resolution=src_res,
                                         sink=self.workspace.filename('resampled'),
                                         sink_resolution=resampling_cfg['raw_sink_resolution'],
                                         workspace=self.workspace,
                                         **default_resample_parameter)
        except BrokenProcessPool:
            print('Resampling canceled')
            return
        assert result.array.max() != 0, 'Resampled raw has no data'

    def __resample_autofluorescence(self):
        resampling_cfg = self.processing_config['registration']['resampling']
        default_resample_parameter = {
            "processes": resampling_cfg['processes'],
            "verbose": resampling_cfg['verbose']
        }  # WARNING: duplicate (use method ??)
        try:
            auto_fluo_path = self.workspace.source('autofluorescence').file_list[0]
        except IndexError:
            print('Could not resample autofluorescence, file not found')
            return
        auto_res = define_auto_resolution(auto_fluo_path, self.sample_config['resolutions']['autofluorescence'])
        n_planes = len(self.workspace.file_list('autofluorescence'))  # TODO: find more elegant solution for counter
        self.prepare_watcher_for_substep(n_planes, self.__resample_re, 'Resampling autofluorescence', True)
        try:
            result = resampling.resample(self.workspace.filename('autofluorescence'),
                                         source_resolution=auto_res,
                                         sink=self.workspace.filename('resampled', postfix='autofluorescence'),
                                         sink_resolution=resampling_cfg['autofluo_sink_resolution'],
                                         workspace=self.workspace,
                                         **default_resample_parameter)
        except BrokenProcessPool:
            print('Resampling canceled')
            return
        assert result.array.max() != 0, 'Resampled autofluorescence has no data'

    @property
    def n_registration_steps(self):
        resampling_cfg = self.processing_config['registration']['resampling']  # WARNING: probably 1 more when arteries included
        n_steps_atlas_setup = 1
        n_steps_align = 2  # WARNING: probably 1 more when arteries included
        return n_steps_atlas_setup + int(not resampling_cfg['skip'])*2 + n_steps_align

    def resample_for_registration(self, force=False):
        if force:
            self.stopped = False
        if self.stopped:
            return
        resampling_cfg = self.processing_config['registration']['resampling']
        if not resampling_cfg['skip']:
            # Raw
            self.__resample_raw()
            if self.stopped:
                return
            if resampling_cfg['plot_raw'] and not runs_on_ui():
                plot_3d.plot(self.workspace.filename('resampled'))

            # Autofluorescence
            self.__resample_autofluorescence()
            if self.stopped:
                return
            self.update_watcher_main_progress()
            if resampling_cfg['plot_autofluo'] and not runs_on_ui():
                plot_3d.plot([self.workspace.filename('resampled'),
                              self.workspace.filename('resampled', postfix='autofluorescence')])

    def align(self, force=False):
        if force:
            self.stopped = False
        if self.stopped:
            return
        try:
            self.__align_resampled_to_auto()
            self.update_watcher_main_progress()
            self.__align_auto_to_ref()
            self.update_watcher_main_progress()
        except CanceledProcessing:
            print('Alignment canceled')
        self.stopped = False

    def __align_resampled_to_auto(self):
        self.prepare_watcher_for_substep(2000, self.__align_resampled_to_auto_re, 'Align res to auto')
        align_channels_parameter = {
            # moving and reference images
            "moving_image": self.workspace.filename('resampled', postfix='autofluorescence'),
            "fixed_image": self.workspace.filename('resampled'),

            # elastix parameter files for alignment
            "affine_parameter_file": self.align_channels_affine_file,
            "bspline_parameter_file": None,

            # directory of the alignment result '/home/nicolas.renier/Documents/ClearMap_Resources/Par0000affine.txt'
            "result_directory": self.workspace.filename('resampled_to_auto'),
            'workspace': self.workspace
        }
        elastix.align(**align_channels_parameter)
        self.__check_elastix_success('resampled_to_auto')

    def __align_auto_to_ref(self):
        self.prepare_watcher_for_substep(17000, self.__align_auto_to_ref_re, 'Align auto to ref')
        align_reference_parameter = {
            # moving and reference images
            "moving_image": self.reference_file_path,
            "fixed_image": self.workspace.filename('resampled', postfix='autofluorescence'),

            # elastix parameter files for alignment
            "affine_parameter_file": self.align_reference_affine_file,
            "bspline_parameter_file": self.align_reference_bspline_file,
            # directory of the alignment result
            "result_directory": self.workspace.filename('auto_to_reference'),
            'workspace': self.workspace
        }
        for k, v in align_reference_parameter.items():
            if not v:
                raise ValueError('Registration missing parameter "{}"'.format(k))
        elastix.align(**align_reference_parameter)
        self.__check_elastix_success('auto_to_reference')

    def __check_elastix_success(self, results_dir_name):
        with open(os.path.join(self.workspace.filename(results_dir_name), 'elastix.log'), 'r') as logfile:
            if 'fail' in logfile.read():
                results_msg = results_dir_name.replace('_', ' ')
                raise ValueError(f'Alignment {results_msg} failed')  # TODO: change exception type

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
            paths.append(self.workspace.filename('stitched', postfix='arteries'))  # WARNING: hard coded postfix
            title += ' and arteries stitched'
        if len(paths) != 2:
            paths = paths[0]
        dvs = plot_3d.plot(paths, arange=False, lut='white', parent=parent)
        return dvs


def main():
    preprocessor = PreProcessor()
    preprocessor.setup(sys.argv[1:3])
    preprocessor.setup_atlases()
    preprocessor.run()


if __name__ == '__main__':
    main()
