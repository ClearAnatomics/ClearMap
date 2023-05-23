"""
sample_preparation
==================

This is the part that is common to both pipelines to process the raw images.
It includes file conversion, stitching and registration
"""
import os
import platform
import re
import sys
from concurrent.futures.process import BrokenProcessPool

import numpy as np

# noinspection PyPep8Naming
import matplotlib
import tifffile

matplotlib.use('Qt5Agg')


import ClearMap.Settings as settings
from ClearMap.Utils.utilities import runs_on_ui, requires_files, FilePath
from ClearMap.config.atlas import ATLAS_NAMES_MAP, STRUCTURE_TREE_NAMES_MAP
from ClearMap.gui.gui_utils import TmpDebug  # REFACTOR: move to workspace object
from ClearMap.processors.generic_tab_processor import TabProcessor, CanceledProcessing
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
from ClearMap.IO.elastix_config import ElastixParser
from ClearMap.config.config_loader import get_configs, ConfigLoader, CLEARMAP_CFG_DIR
from ClearMap.config.update_config import update_default_config
import ClearMap.Visualization.Plot3d as q_plot_3d



__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


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
        if not os.path.exists(CLEARMAP_CFG_DIR):
            update_default_config()

    @property
    def prefix(self):
        return self.sample_config['sample_id'] if self.sample_config['use_id_as_prefix'] else None

    def filename(self, *args, **kwargs):
        return self.workspace.filename(*args, **kwargs)

    def z_only(self, channel='raw'):
        tags = self.workspace.expression(channel, prefix=self.prefix).tags
        axes = [tag.name for tag in tags]
        return axes == ['Z']

    @property
    def is_tiled(self):
        return self.__is_tiled('raw')

    def __is_tiled(self, channel):
        tags = self.workspace.expression(channel, prefix=self.prefix).tags
        if not tags:
            return False
        else:
            return not self.z_only(channel)

    @property
    def autofluorescence_is_tiled(self):
        return self.__is_tiled('autofluorescence')

    def __has_tiles(self, channel):
        # extension = 'npy' if self.use_npy() else None
        # return len(clearmap_io.file_list(self.filename(channel, prefix=self.prefix, extension=extension)))
        # noinspection PyTypeChecker
        return len(clearmap_io.file_list(self.filename(channel, prefix=self.prefix)))

    @property
    def has_tiles(self):
        return self.__has_tiles('raw')

    def check_has_all_tiles(self, channel):
        extension = 'npy' if self.use_npy() else None
        return self.workspace.all_tiles_exist(channel, extension=extension)

    @property
    def has_npy(self):
        # noinspection PyTypeChecker
        return len(clearmap_io.file_list(self.filename('raw', prefix=self.prefix, extension='.npy')))

    def get_autofluo_pts_path(self, direction='resampled_to_auto'):
        elastix_folder = self.filename(direction)
        return os.path.join(elastix_folder, 'autolfuorescence_landmarks.pts')  # TODO: use workspace

    def clear_landmarks(self):
        for f_path in (self.ref_pts_path, self.resampled_pts_path,
                       self.get_autofluo_pts_path('resampled_to_auto'),
                       self.get_autofluo_pts_path('auto_to_reference')):
            if os.path.exists(f_path):
                os.remove(f_path)

    @property
    def resampled_pts_path(self):
        return os.path.join(self.filename('resampled_to_auto'), 'resampled_landmarks.pts')

    @property
    def ref_pts_path(self):
        return os.path.join(self.filename('auto_to_reference'), 'reference_landmarks.pts')  # TODO: use workspace

    def setup(self, cfgs, watcher=None, convert_tiles=True):
        """

        Parameters
        ----------
        cfgs tuple of (machine_cfg_path, sample_cfg_path, processing_cfg_path) or
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

        self.workspace = workspace.Workspace(self.processing_config['pipeline_name'], directory=self.src_directory,
                                             prefix=self.prefix)
        self.workspace.tmp_debug = TmpDebug(self.workspace)
        src_paths = {k: v for k, v in self.sample_config['src_paths'].items() if v is not None}
        self.workspace.update(**src_paths)
        self.workspace.info()
        if convert_tiles:
            self.convert_tiles()
        # FIXME: check if setup_atlas should go here

    def unpack_atlas(self, atlas_base_name):
        res = annotation.decompress_atlases(atlas_base_name)
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
        return os.path.join(self.filename('auto_to_reference'), 'result.1.mhd')
    
    @property
    def raw_stitched_shape(self):
        if self.resampled_shape is not None:
            raw_resampled_res_from_cfg = np.array(self.processing_config['registration']['resampling']['raw_sink_resolution'])
            raw_res_from_cfg = np.array(self.sample_config['resolutions']['raw'])
            return self.resampled_shape * (raw_resampled_res_from_cfg / raw_res_from_cfg)
        else:
            return clearmap_io.shape(self.filename('stitched'))

    @property
    def resampled_shape(self):
        if os.path.exists(self.filename('resampled')):
            return clearmap_io.shape(self.filename('resampled'))

    def convert_tiles(self, force=False):
        """
        Convert list of input files to numpy files for efficiency reasons

        Returns
        -------

        """
        if self.stopped:
            return
        if force or self.use_npy():
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
            self.update_watcher_main_progress()

    def use_npy(self):
        return self.processing_config['conversion']['use_npy'] or \
               self.filename('raw').endswith('.npy') or \
               os.path.exists(self.filename('raw', extension='npy'))

    def set_configs(self, cfg_paths):
        cfg_paths = [os.path.expanduser(p) for p in cfg_paths]
        self.machine_config, self.sample_config, self.processing_config = get_configs(*cfg_paths)

    def setup_atlases(self):  # TODO: add possibility to load custom reference file (i.e. defaults to None in cfg)
        if not self.processing_config:
            return  # Not setup yet. FIXME: find better way around
        self.prepare_watcher_for_substep(0, None, 'Initialising atlases')
        atlas_base_name = ATLAS_NAMES_MAP[self.processing_config['registration']['atlas']['id']]['base_name']
        self.unpack_atlas(atlas_base_name)
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
        annotation.set_annotation_file(self.annotation_file_path)

        structure_tree_id = self.processing_config['registration']['atlas']['structure_tree_id']
        structure_file_name = STRUCTURE_TREE_NAMES_MAP[structure_tree_id]
        annotation.set_label_file(os.path.join(settings.atlas_folder, structure_file_name))

        self.update_watcher_main_progress()
        atlas_cfg = self.processing_config['registration']['atlas']
        align_dir = os.path.join(self.resources_directory, atlas_cfg['align_files_folder'])
        self.align_channels_affine_file = os.path.join(align_dir, atlas_cfg['align_channels_affine_file'])
        self.align_reference_affine_file = os.path.join(align_dir, atlas_cfg['align_reference_affine_file'])
        self.align_reference_bspline_file = os.path.join(align_dir, atlas_cfg['align_reference_bspline_file'])

    def plot_atlas(self):
        return q_plot_3d.plot(self.reference_file_path, lut=self.machine_config['default_lut'])

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

    @property
    def n_channels_convert(self):
        return [self.processing_config['stitching']['output_conversion']['raw'],
                self.processing_config['stitching']['output_conversion']['arteries']].count(True)

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
                clearmap_io.convert_files(self.workspace.file_list('stitched', extension='npy', prefix=self.prefix),
                                          extension=fmt, processes=self.machine_config['n_processes_file_conv'],
                                          workspace=self.workspace, verbose=True)
                self.update_watcher_main_progress()
            except BrokenProcessPool:
                print('File conversion canceled')
                return
        if self.processing_config['stitching']['output_conversion']['arteries']:
            try:
                clearmap_io.convert_files(self.workspace.file_list('stitched', postfix='arteries',
                                                                   prefix=self.prefix, extension='npy'),
                                          extension=fmt, processes=self.machine_config['n_processes_file_conv'],
                                          workspace=self.workspace, verbose=True)
                self.update_watcher_main_progress()
            except BrokenProcessPool:
                print('File conversion canceled')
                return

    @property
    def was_stitched_rigid(self):
        return os.path.exists(self.filename('layout', postfix='aligned_axis'))

    @property
    def was_registered(self):
        # return os.path.exists(self.filename('resampled_to_auto'))
        return os.path.exists(self.aligned_autofluo_path)

    @property
    def n_rigid_steps_to_run(self):
        return int(not self.processing_config['stitching']['rigid']['skip'])

    @requires_files([FilePath('raw')])
    def stitch_rigid(self, force=False):
        if force:
            self.stopped = False
        if self.stopped:
            return
        self.set_watcher_step('Stitching rigid')
        stitching_cfg = self.processing_config['stitching']
        overlaps, projection_thickness = define_auto_stitching_params(self.workspace.source('raw').file_list[0],
                                                                      stitching_cfg)
        layout = self.get_wobbly_layout(overlaps)
        if stitching_cfg['rigid']['background_pixels'] is None:
            background_params = stitching_cfg['rigid']['background_level']
        else:
            background_params = (stitching_cfg['rigid']['background_level'],
                                 stitching_cfg['rigid']['background_pixels'])
        max_shifts = [stitching_cfg['rigid'][f'max_shifts_{ax}'] for ax in 'xyz']
        self.prepare_watcher_for_substep(len(layout.alignments), self.__rigid_stitching_align_re, 'Align layout rigid')
        try:
            stitching_rigid.align_layout_rigid_mip(layout, depth=projection_thickness, max_shifts=max_shifts,
                                                   ranges=[None, None, None], background=background_params,
                                                   clip=25000, processes=self.machine_config['n_processes_stitching'],
                                                   workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Stitching canceled')
            self.stopped = True  # FIXME: see if appropriate solution
            return  # FIXME: do not run stiching_wobbly if rigid failed
        layout.place(method='optimization', min_quality=-np.inf, lower_to_origin=True, verbose=True)
        self.update_watcher_main_progress()

        # layout.plot_alignments()  # TODO: TEST
        # plt.show()

        stitching_rigid.save_layout(self.filename('layout', postfix='aligned_axis'), layout)
        self.layout = layout

    @requires_files([FilePath('raw')])  # TODO: optional requires npy
    def get_wobbly_layout(self, overlaps=None):
        if overlaps is None:
            overlaps, projection_thickness = define_auto_stitching_params(self.workspace.source('raw').file_list[0],
                                                                          self.processing_config['stitching'])
        extension = 'npy' if self.use_npy() else None  # TODO: optional requires
        raw_path = self.filename('raw', extension=extension)
        layout = stitching_wobbly.WobblyLayout(expression=raw_path, tile_axes=['X', 'Y'], overlaps=overlaps)
        return layout

    @property
    def n_wobbly_steps_to_run(self):
        return int(not self.processing_config['stitching']['wobbly']['skip']) * 3

    def __align_layout_wobbly(self, layout):
        wobbly_cfg = self.processing_config['stitching']['wobbly']
        max_shifts = [
            wobbly_cfg['max_shifts_x'],
            wobbly_cfg['max_shifts_y'],
            wobbly_cfg['max_shifts_z'],
        ]
        stack_validation_params = dict(
            method='foreground',
            valid_range=wobbly_cfg["stack_valid_range"],
            size=wobbly_cfg["stack_pixel_size"]
        )
        slice_validation_params = dict(
            method='foreground',
            valid_range=wobbly_cfg["slice_valid_range"],
            size=wobbly_cfg["slice_pixel_size"]
        )

        n_pairs = len(layout.alignments)
        self.prepare_watcher_for_substep(n_pairs, self.__wobbly_stitching_algin_lyt_re, 'Align layout wobbly')
        try:
            stitching_wobbly.align_layout(layout, axis_range=(None, None, 3), max_shifts=max_shifts, axis_mip=None,
                                          stack_validation_params=stack_validation_params,
                                          prepare=dict(method='normalization', clip=None, normalize=True),
                                          slice_validation_params=slice_validation_params,
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
            n_processes = self.machine_config['n_processes_stitching']
            if platform.system().lower().startswith('darwin'):
                n_processes = 1
            stitching_wobbly.place_layout(layout, min_quality=-np.inf,
                                          method='optimization',
                                          smooth=dict(method='window', window='bartlett', window_length=100,
                                                      binary=None),
                                          smooth_optimized=dict(method='window', window='bartlett',
                                                                window_length=20, binary=10),
                                          fix_isolated=False, lower_to_origin=True,
                                          processes=n_processes,
                                          workspace=self.workspace, verbose=True)
        except BrokenProcessPool:
            print('Wobbly stitching canceled')
            return
        self.update_watcher_main_progress()

    def __stitch_layout_wobbly(self):
        layout = stitching_rigid.load_layout(self.filename('layout', postfix='placed'))
        n_slices = len(self.workspace.file_list('autofluorescence'))  # TODO: find better proxy
        self.prepare_watcher_for_substep(n_slices, self.__wobbly_stitching_stitch_re, 'Stitch layout wobbly', True)
        try:
            stitching_wobbly.stitch_layout(layout, sink=self.filename('stitched'), method='interpolation',
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
                    layout.replace_source_location(self.filename('raw', extension='npy'),
                                                   self.filename('arteries', extension='npy'))
                else:
                    layout.replace_source_location(self.filename('raw'), self.filename('arteries'))
                stitching_wobbly.stitch_layout(layout, sink=self.filename('stitched', postfix='arteries'),
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
        self.set_watcher_step('Stitching wobbly')
        layout = stitching_rigid.load_layout(self.filename('layout', postfix='aligned_axis'))
        self.__align_layout_wobbly(layout)
        if self.stopped:
            return
        stitching_rigid.save_layout(self.filename('layout', postfix='aligned'), layout)

        # layout = st.load_layout(self.filename('layout', postfix='aligned'))  # FIXME: check if required
        self.__place_layout_wobbly(layout)
        if self.stopped:
            return
        stitching_rigid.save_layout(self.filename('layout', postfix='placed'), layout)

        self.__stitch_layout_wobbly()
        if self.stopped:
            return

    def __resample_raw(self):
        resampling_cfg = self.processing_config['registration']['resampling']
        default_resample_parameter = {
            'processes': self.machine_config['n_processes_resampling'],
            'verbose': resampling_cfg['verbose']
        }  # WARNING: duplicate (use method ??)
        clearmap_io.delete_file(self.filename('resampled'))  # FIXME:
        try:
            f_list = self.workspace.source('raw').file_list
        except AttributeError:  # e.g. single file, force use of config
            f_list = None
        if f_list:
            src_res = define_auto_resolution(f_list[0], self.sample_config['resolutions']['raw'])
        else:
            src_res = self.sample_config['resolutions']['raw']

        n_planes = len(self.workspace.file_list('autofluorescence'))  # FIXME: use uimg metadata or z pattern of raw instead
        if n_planes < 2:  # e.g. 1 file
            n_planes = clearmap_io.shape(self.workspace.filename('autofluorescence'))[-1]
        self.prepare_watcher_for_substep(n_planes, self.__resample_re, 'Resampling raw')
        try:
            result = resampling.resample(self.filename('stitched'),
                                         source_resolution=src_res,
                                         sink=self.filename('resampled'),
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
            'processes': self.machine_config['n_processes_resampling'],
            'verbose': resampling_cfg['verbose']
        }  # WARNING: duplicate (use method ??)
        try:
            auto_fluo_path = self.workspace.file_list('autofluorescence')[0]
        except IndexError:
            print('Could not resample autofluorescence, file not found')
            return
        auto_res = define_auto_resolution(auto_fluo_path, self.sample_config['resolutions']['autofluorescence'])
        n_planes = len(self.workspace.file_list('autofluorescence'))  # TODO: find more elegant solution for counter
        if n_planes < 2:  # e.g. 1 file
            n_planes = clearmap_io.shape(self.workspace.filename('autofluorescence'))[-1]
        self.prepare_watcher_for_substep(n_planes, self.__resample_re, 'Resampling autofluorescence', True)
        try:
            result = resampling.resample(self.filename('autofluorescence'),
                                         source_resolution=auto_res,
                                         sink=self.filename('resampled', postfix='autofluorescence'),
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
                plot_3d.plot(self.filename('resampled'))

            # Autofluorescence
            self.__resample_autofluorescence()
            if self.stopped:
                return
            self.update_watcher_main_progress()
            if resampling_cfg['plot_autofluo'] and not runs_on_ui():
                plot_3d.plot([self.filename('resampled'),
                              self.filename('resampled', postfix='autofluorescence')])

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
            "moving_image": self.filename('resampled', postfix='autofluorescence'),
            "fixed_image": self.filename('resampled'),

            # elastix parameter files for alignment
            "affine_parameter_file": self.align_channels_affine_file,
            "bspline_parameter_file": None,

            "result_directory": self.filename('resampled_to_auto'),
            'workspace': self.workspace
        }
        use_landmarks = os.path.exists(self.get_autofluo_pts_path('resampled_to_auto')) and os.path.exists(
            self.resampled_pts_path)
        if use_landmarks:  # FIXME: add use_landmarks to config
            align_channels_parameter.update(moving_landmarks_path=self.resampled_pts_path,
                                            fixed_landmarks_path=self.get_autofluo_pts_path('resampled_to_auto'))
            self.patch_elastix_parameter_files([self.align_channels_affine_file])
        elastix.align(**align_channels_parameter)
        self.restore_elastix_parameter_files([self.align_channels_affine_file])  # FIXME: do in try except
        self.__check_elastix_success('resampled_to_auto')

    def __align_auto_to_ref(self):
        self.prepare_watcher_for_substep(17000, self.__align_auto_to_ref_re, 'Align auto to ref')
        align_reference_parameter = {
            # moving and reference images
            "moving_image": self.reference_file_path,
            "fixed_image": self.filename('resampled', postfix='autofluorescence'),

            # elastix parameter files for alignment
            "affine_parameter_file": self.align_reference_affine_file,
            "bspline_parameter_file": self.align_reference_bspline_file,
            # directory of the alignment result
            "result_directory": self.filename('auto_to_reference'),
            'workspace': self.workspace
        }
        use_landmarks = os.path.exists(self.get_autofluo_pts_path('auto_to_reference')) and os.path.exists(self.ref_pts_path)
        if use_landmarks:  # FIXME: add use_landmarks to config
            align_reference_parameter.update(moving_landmarks_path=self.ref_pts_path,
                                             fixed_landmarks_path=self.get_autofluo_pts_path('auto_to_reference'))
            self.patch_elastix_parameter_files([self.align_reference_affine_file, self.align_reference_bspline_file])
        for k, v in align_reference_parameter.items():
            if not v:
                raise ValueError(f'Registration missing parameter "{k}"')
        elastix.align(**align_reference_parameter)
        self.restore_elastix_parameter_files([self.align_reference_affine_file, self.align_reference_bspline_file])  # FIXME: do in try except
        self.__check_elastix_success('auto_to_reference')

    def __check_elastix_success(self, results_dir_name):
        with open(os.path.join(self.filename(results_dir_name), 'elastix.log'), 'r') as logfile:
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
        titles = []
        if cfg['raw']:
            paths.append(self.filename('stitched'))
            titles.append('Raw stitched')
        if cfg['arteries']:
            paths.append(self.filename('stitched', postfix='arteries'))  # WARNING: hard coded postfix
            titles.append('Arteries stitched')
        if len(paths) != 2:
            paths = paths[0]
        dvs = plot_3d.plot(paths, title=titles, arrange=False, lut='white', parent=parent)
        return dvs

    @staticmethod
    def patch_elastix_parameter_files(elastix_files):
        for f_path in elastix_files:
            cfg = ElastixParser(f_path)
            cfg['Registration'] = ['MultiMetricMultiResolutionRegistration']
            cfg['Metric'] = ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"]
            cfg.write()

    @staticmethod
    def restore_elastix_parameter_files(elastix_files):
        for f_path in elastix_files:
            cfg = ElastixParser(f_path)
            cfg['Registration'] = ['MultiResolutionRegistration']
            cfg['Metric'] = ["AdvancedMattesMutualInformation"]
            cfg.write()

    def stitch_overlay(self, channel, color=True):
        """
        This creates a *dumb* overlay of the tiles
        i.e. only using the fixed guess overlap
        Parameters
        ----------
        channel
        color

        Returns
        -------

        """
        positions = self.workspace.get_positions(channel)
        mosaic_shape = {ax: max([p[ax] for p in positions]) + 1 for ax in 'XY'}  # +1 because 0 indexing
        files = self.workspace.file_list(channel)
        tile_shape = {k: v for k, v in zip('XYZ', clearmap_io.shape(files[0]))}
        middle_z = int(tile_shape['Z'] / 2)
        overlaps = {k: self.processing_config['stitching']['rigid'][f'overlap_{k.lower()}'] for k in 'XY'}
        output_shape = [tile_shape[ax] * mosaic_shape[ax] - overlaps[ax] * (mosaic_shape[ax] - 1) for ax in 'XY']
        cyan_image = np.zeros(output_shape, dtype=int)
        magenta_image = np.zeros(output_shape, dtype=int)
        for tile_path, pos in zip(files, positions):
            starts = {ax: pos[ax] * tile_shape[ax] - pos[ax] * overlaps[ax] for ax in 'XY'}
            ends = {ax: starts[ax] + tile_shape[ax] for ax in starts.keys()}
            if (pos['Y'] + pos['X']) % 2:  # Alternate colors
                layer = cyan_image
            else:
                layer = magenta_image
            tile = clearmap_io.read(tile_path)[:, :, middle_z]  # TODO: see if can seek
            layer[starts['X']: ends['X'], starts['Y']: ends['Y']] = tile
        blank = np.zeros(output_shape, dtype=cyan_image.dtype)
        if color:
            high_intensity = (cyan_image.mean() + 4 * cyan_image.std())
            cyan_image = cyan_image / high_intensity * 128
            cyan_image = np.dstack((blank, cyan_image, cyan_image))  # To Cyan RGB
            high_intensity = (magenta_image.mean() + 4 * magenta_image.std())
            magenta_image = magenta_image / high_intensity * 128
            magenta_image = np.dstack((magenta_image, blank, magenta_image))  # To Magenta RGB
        output_image = cyan_image + magenta_image  # TODO: normalise
        if color:
            output_image = output_image.clip(0, 255).astype(np.uint8)
        return output_image

    def overlay_layout_plane(self, layout):
        """Overlays the sources to check their placement.

        Arguments
        ---------
        layout : Layout class
          The layout with the sources to overlay.

        Returns
        -------
        image : array
          A color image.
        """
        sources = layout.sources

        dest_shape = tuple(layout.extent[:-1])
        full_lower = layout.lower
        middle_z = round(dest_shape[-1] / 2)

        cyan_image = np.zeros(dest_shape, dtype=int)
        magenta_image = np.zeros(dest_shape, dtype=int)
        # construct full image
        for s in sources:
            l = s.lower
            u = s.upper
            tile = clearmap_io.read(s.location)[:, :,
                   middle_z]  # So as not to load the data into the list for memory efficiency
            current_slicing = tuple(slice(ll - fl, uu - fl) for ll, uu, fl in zip(l, u, full_lower))[:2]

            is_odd = sum(s.tile_position) % 2
            if is_odd:  # Alternate colors
                layer = cyan_image
            else:
                layer = magenta_image

            layer[current_slicing] = tile
        blank = np.zeros(dest_shape, dtype=cyan_image.dtype)

        high_intensity = (cyan_image.mean() + 4 * cyan_image.std())
        cyan_image = cyan_image / high_intensity * 128
        cyan_image = np.dstack((blank, cyan_image, cyan_image))  # To Cyan RGB

        high_intensity = (magenta_image.mean() + 4 * magenta_image.std())
        magenta_image = magenta_image / high_intensity * 128
        magenta_image = np.dstack((magenta_image, blank, magenta_image))  # To Magenta RGB

        output_image = cyan_image + magenta_image  # TODO: normalise
        output_image = output_image.clip(0, 255).astype(np.uint8)

        return output_image

    def plot_layout(self, postfix='aligned_axis'):
        if postfix not in ('aligned_axis', 'aligned', 'placed'):
            raise ValueError(f'Expected on of ("aligned_axis", "aligned", "placed") for postfix, got "{postfix}"')
        layout = stitching_rigid.load_layout(self.filename('layout', postfix=postfix))
        overlay = self.overlay_layout_plane(layout)
        return overlay


def main():
    preprocessor = PreProcessor()
    preprocessor.setup(sys.argv[1:3])
    preprocessor.setup_atlases()
    preprocessor.run()


if __name__ == '__main__':
    main()


def init_preprocessor(folder, atlas_base_name=None, convert_tiles=False):
    cfg_loader = ConfigLoader(folder)
    configs = get_configs(cfg_loader.get_cfg_path('sample'), cfg_loader.get_cfg_path('processing'))
    pre_proc = PreProcessor()
    if atlas_base_name is None:
        atlas_id = configs[2]['registration']['atlas']['id']
        atlas_base_name = ATLAS_NAMES_MAP[atlas_id]['base_name']
    json_file = os.path.join(settings.atlas_folder, STRUCTURE_TREE_NAMES_MAP[configs[2]['registration']['atlas']['structure_tree_id']])
    pre_proc.unpack_atlas(atlas_base_name)
    pre_proc.setup(configs, convert_tiles=convert_tiles)
    pre_proc.setup_atlases()
    annotation.initialize(annotation_file=pre_proc.annotation_file_path, label_file=json_file)
    return pre_proc
