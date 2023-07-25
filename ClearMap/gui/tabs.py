# -*- coding: utf-8 -*-
"""
tabs
====

The different tabs that correspond to different functionalities of the GUI
"""
import functools
import os.path
import copy
from shutil import copyfile

import numpy as np
import pandas as pd

import mpld3

import pyqtgraph as pg
from ClearMap.config.config_loader import ConfigLoader
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QDialogButtonBox
from qdarkstyle import DarkPalette

import ClearMap.IO.IO as clearmap_io
from ClearMap.IO.MHD import mhd_read
from ClearMap.Alignment import Annotation as annotation
from ClearMap.Analysis.Statistics.group_statistics import make_summary, density_files_are_comparable, compare_groups
from ClearMap.Utils.exceptions import PlotGraphError, ClearMapVRamException, GroupStatsError
from ClearMap.Visualization.Matplotlib.PlotUtils import plot_sample_stats_histogram, plot_volcano
from ClearMap.Visualization.atlas import create_color_annotation

from ClearMap.gui.dialogs import prompt_dialog, get_directory_dlg
from ClearMap.gui.gui_utils import format_long_nb_to_str, surface_project, np_to_qpixmap
from ClearMap.gui.interfaces import GenericTab, PostProcessingTab
from ClearMap.gui.params import ParamsOrientationError, VesselParams, SampleParameters, \
    AlignmentParams, CellMapParams, GroupAnalysisParams, BatchProcessingParams
from ClearMap.gui.widgets import PatternDialog, SamplePickerDialog, DataFrameWidget, LandmarksSelectorDialog
from ClearMap.Visualization.Qt.widgets import Scatter3D

from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
from ClearMap.Visualization.Qt import Plot3d as plot_3d

from ClearMap.processors.sample_preparation import PreProcessor, init_preprocessor
from ClearMap.processors.cell_map import CellDetector
from ClearMap.processors.tube_map import BinaryVesselProcessor, VesselGraphProcessor
from ClearMap.processors.batch_process import BatchProcessor

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


class SampleTab(GenericTab):
    """
    The tab manager to define the parameters of the sample
    This refers to values that are intrinsic to the sample and the acquisition
    like resolution, orientation ...
    """
    def __init__(self, main_window, tab_idx=0):
        super().__init__(main_window, 'Sample', tab_idx, 'sample_tab')
        self.mini_brain_scaling = None
        self.mini_brain = None

    def set_params(self, *args):
        """
        Set the params object which links the UI and the configuration file
        Parameters
        ----------
        args

        Returns
        -------

        """
        self.params = SampleParameters(self.ui, self.main_window.src_folder)

    def setup(self):
        """
        Setup the UI elements, notably the signal/slot connections which are not
        automatically set through the params object attribute

        Returns
        -------

        """
        self.init_ui()

        self.ui.srcFolderBtn.clicked.connect(self.main_window.prompt_experiment_folder)
        self.connect_whats_this(self.ui.srcFolderInfoToolButton, self.ui.srcFolderBtn)
        self.ui.sampleIdPushButton.clicked.connect(self.main_window.load_config_and_setup_ui)
        self.connect_whats_this(self.ui.sampleIdInfoToolButton, self.ui.sampleIdLabel)

        self.ui.launchPatternWizzardPushButton.clicked.connect(self.launch_pattern_wizard)
        self.connect_whats_this(self.ui.patternWizzardBtnInfoToolButton, self.ui.launchPatternWizzardPushButton)

        self.connect_whats_this(self.ui.mainChannelPathInfoToolButton, self.ui.mainChannelPathLbl)
        self.connect_whats_this(self.ui.autofluoChannelPathInfoToolButton, self.ui.autofluoChannelPathLbl)
        self.connect_whats_this(self.ui.secondaryChannelPathInfoToolButton, self.ui.secondaryChannelPathLbl)

        self.connect_whats_this(self.ui.sampleOrientationInfoToolButton, self.ui.sampleOrientationLbl)
        self.connect_whats_this(self.ui.miniBrainFrameInfoToolButton, self.ui.miniBrainFrame)
        self.connect_whats_this(self.ui.sampleCropXLblInfoToolButton, self.ui.sampleCropXLbl)
        self.connect_whats_this(self.ui.sampleCropYLblInfoToolButton, self.ui.sampleCropYLbl)
        self.connect_whats_this(self.ui.sampleCropZLblInfoToolButton, self.ui.sampleCropZLbl)
        self.ui.plotMiniBrainPushButton.clicked.connect(self.plot_mini_brain)

        self.ui.sampleViewAtlasPushButton.clicked.connect(self.display_atlas)

        self.ui.applyBox.connectApply(self.main_window.alignment_tab_mgr.setup_workers)  # FIXME: check if risks overwrite if still same sample
        self.ui.applyBox.connectSave(self.save_cfg)

        self.ui.sampleIdPushButton.setIcon(self.main_window._reload_icon)

    def save_cfg(self):  # REFACTOR: use this instead of direct calls to ui_to_cfg
        """
        Save the config to file

        Returns
        -------

        """
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Sample config saved')

    def load_config_to_gui(self):
        """
        Set every control on the UI to the value in the params
        Returns
        -------

        """
        try:
            self.params.cfg_to_ui()
        except ParamsOrientationError as err:
            self.main_window.popup(str(err), 'Invalid orientation. Defaulting')
            self.params.orientation = (1, 2, 3)

    @property
    def src_folder(self):
        return self.ui.srcFolderTxt.text()

    @src_folder.setter
    def src_folder(self, folder):
        self.ui.srcFolderTxt.setText(folder)

    def display_sample_id(self, sample_id):
        """
        Display the sample ID to the corresponding UI widget
        Parameters
        ----------
        sample_id : str
            The unique ID for that sample

        Returns
        -------

        """
        self.ui.sampleIdTxt.setText(sample_id)

    def display_use_id_as_prefix(self, use_id):
        """
        Displays whether to use the ID as prefix in the corresponding
        widget of the UI

        Parameters
        ----------
        use_id : bool
            Whether to se the sample ID as prefix in the file names

        Returns
        -------

        """
        self.ui.useIdAsPrefixCheckBox.setChecked(use_id)

    def get_sample_id(self):
        """
        Get the sample ID from the GUI widget
        Returns
        -------

        """
        return self.ui.sampleIdTxt.text()

    def go_to_orientation(self):
        """
        Jump to the sample orientation (space info) tab

        Returns
        -------

        """
        self.ui.toolBox.setCurrentIndex(2)
        self.main_window.tabWidget.setCurrentIndex(0)

    def launch_pattern_wizard(self):
        """
        Start the pattern selection wizard. This wizard helps create the
        pattern strings for the individual tiles, with specific characters
        representing the digits for the different axes.

        Returns
        -------

        """
        dlg = PatternDialog(self.src_folder, self.params,
                            min_file_number=self.main_window.preference_editor.params.pattern_finder_min_n_files,
                            tile_extension=self.params.tile_extension)
        dlg.exec()

    def plot_mini_brain(self):
        """
        Plot the brain icon which represents the acquisition sample orientation graphically
        to help users pick the right orientation.

        Returns
        -------

        """
        img = self.__transform_mini_brain()
        mask, proj = surface_project(img)
        img = np_to_qpixmap(proj, mask)
        self.ui.miniBrainLabel.setPixmap(img)

    def __transform_mini_brain(self):  # REFACTOR: extract
        """
        Apply the set of transforms to the mini brain as defined by the crop and
        orientation parameters input by the user.

        Returns
        -------

        """
        def scale_range(rng, scale):
            for i in range(len(rng)):
                if rng[i] is not None:
                    rng[i] = round(rng[i] / scale)
            return rng

        def range_or_default(rng, scale):
            if rng is not None:
                return scale_range(rng, scale)
            else:
                return 0, None

        orientation = self.params.orientation
        x_scale, y_scale, z_scale = self.mini_brain_scaling
        img = self.mini_brain.copy()
        axes_to_flip = [abs(axis) - 1 for axis in orientation if axis < 0]
        if axes_to_flip:
            img = np.flip(img, axes_to_flip)
        img = img.transpose([abs(axis) - 1 for axis in orientation])
        x_min, x_max = range_or_default(self.params.slice_x, x_scale)
        y_min, y_max = range_or_default(self.params.slice_y, y_scale)
        z_min, z_max = range_or_default(self.params.slice_z, z_scale)
        img = img[x_min:x_max, y_min:y_max:, z_min:z_max]
        return img

    def display_atlas(self):
        """
        Plot the atlas as a FIXME:
        Returns
        -------

        """
        self.main_window.alignment_tab_mgr.setup_workers()
        dvs = self.main_window.alignment_tab_mgr.preprocessor.plot_atlas()   # REFACTOR:
        self.main_window.setup_plots(dvs)


class AlignmentTab(GenericTab):
    """
    The tab responsible for all the alignments, including the stitching and
    aligning to the atlas.
    """
    def __init__(self, main_window, tab_idx=1):
        super().__init__(main_window, 'Alignment', tab_idx, 'alignment_tab')

        self.processing_type = 'pre'
        self.sample_params = None
        self.preprocessor = PreProcessor()

        self.advanced_control_names = ['advancedAtlasSettingsGroupBox']

    def set_params(self, sample_params):
        """
        Set the params object which links the UI and the configuration file
        Parameters
        ----------
        sample_params : SampleParameters
            The params object that links the UI to the config for all the properties intrinsic to the sample

        Returns
        -------

        """
        self.sample_params = sample_params
        self.params = AlignmentParams(self.ui)
        self.params.registration.atlas_id_changed.connect(self.preprocessor.setup_atlases)
        self.params.registration.atlas_structure_tree_id_changed.connect(self.preprocessor.setup_atlases)

    def setup_workers(self):
        """
        Setup the worker (PreProcessor) which handle the computations associated with this tab
        Returns
        -------

        """
        self.sample_params.ui_to_cfg()
        self.preprocessor.setup((self.main_window.preference_editor.params.config,
                                 self.sample_params.config, self.params.config),
                                convert_tiles=False)

        if self.preprocessor.has_tiles and not self.preprocessor.has_npy and\
                prompt_dialog('Tile conversion', 'Convert individual tiles to npy for efficiency'):
            self.convert_tiles()
        self.wrap_step('Setting up atlas', self.setup_atlas, n_steps=1, save_cfg=False, nested=False)  # TODO: abort_func=self.preprocessor.stop_process

    def convert_tiles(self):
        if not self.preprocessor.has_tiles:
            return
        self.wrap_step('Converting tiles', self.preprocessor.convert_tiles, step_kw_args={'force': True}, n_steps=0,
                       abort_func=self.preprocessor.stop_process, save_cfg=False, nested=False)

    def setup(self):
        """
        Setup the UI elements, notably the signal/slot connections which are not
        automatically set through the params object attribute

        Returns
        -------

        """
        self.init_ui()

        self.ui.convertTilesToNpyPushButton.clicked.connect(self.convert_tiles)

        self.ui.previewStitchingPushButton.clicked.connect(functools.partial(self.preview_stitching_dumb, color=True))
        self.connect_whats_this(self.ui.rigidProjectionThicknessLblInfoToolButton, self.ui.rigidProjectionThicknessLbl)
        self.ui.stitchingPreviewLevelsPushButton.clicked.connect(functools.partial(self.preview_stitching_dumb, color=False))
        self.ui.stitchingPreviewRigidPushButton.clicked.connect(functools.partial(self.preview_stitching_smart, postfix='aligned_axis'))

        self.ui.runStitchingPushButton.clicked.connect(self.run_stitching)
        self.ui.displayStitchingPushButton.clicked.connect(self.plot_stitching_results)
        self.ui.displayStitchingClearPlots.clicked.connect(self.main_window.clear_plots)
        self.ui.convertOutputPushButton.clicked.connect(self.convert_output)

        self.ui.registrationGoToSampleOrientationPushButton.clicked.connect(self.main_window.sample_tab_mgr.go_to_orientation)
        self.ui.registerPushButton.clicked.connect(self.run_registration)
        self.ui.clearLandmarksFilesPushButton.clicked.connect(self.clear_landmarks)
        self.ui.plotRegistrationResultsSideBySidePushButton.clicked.connect(self.plot_registration_results_side_by_side)
        self.ui.plotRegistrationResultsCompositePushButton.clicked.connect(self.plot_registration_results_composite)
        self.ui.plotRegistrationResultsRawSideBySidePushButton.clicked.connect(self.plot_registration_results_side_by_side_raw)
        self.ui.plotRegistrationResultsRawCompositePushButton.clicked.connect(self.plot_registration_results_composite_raw)

        # TODO: ?? connect alignment folder button

        self.ui.useAutoToRefLandmarksPushButton.clicked.connect(self.display_auto_to_ref_landmarks_dialog)
        self.ui.useResampledToAutoLandmarksPushButton.clicked.connect(self.display_resampled_to_auto_landmarks_dialog)
        self.connect_whats_this(self.ui.resampledLandmarksInfoToolButton, self.ui.useResampledToAutoLandmarksPushButton)
        self.connect_whats_this(self.ui.referenceLandmarksInfoToolButton, self.ui.useAutoToRefLandmarksPushButton)

    def set_progress_watcher(self, watcher):
        """
        Setup the watcher object that will handle the progress in the computation for this tab

        Parameters
        ----------
        watcher : ProgressWatcher
            The object that tracks the progress

        Returns
        -------

        """
        self.preprocessor.set_progress_watcher(watcher)

    def setup_atlas(self):
        """
        Setup the atlas that corresponds to the orientation and crop

        Returns
        -------

        """
        self.sample_params.ui_to_cfg()  # To make sure we have the slicing up to date
        self.params.registration.ui_to_cfg()
        self.preprocessor.setup_atlases()

    def preview_stitching_dumb(self, color):
        """
        Preview the stitching based only on a *dumb* overlay of the tiles
        i.e. only using the fixed guess overlap

        Parameters
        ----------
        color : bool
            Whether to stitch in chessboard or continuous grayscale

        Returns
        -------

        """
        if color:
            overlay = [pg.image(self.preprocessor.stitch_overlay('raw', color))]
        else:  # TODO: make DataViewer work with 2D color
            overlay = plot_3d.plot(self.preprocessor.stitch_overlay('raw', color), lut='flame', min_max=(100, 5000))
        self.main_window.setup_plots(overlay)

    def preview_stitching_smart(self, postfix='aligned_axis'):
        """
        Preview the stitching based on the actual stitching variable, rigid by default.

        Parameters
        ----------
        postfix : str
            One of ('aligned_axis', 'aligned', 'placed')

        Returns
        -------

        """
        n_steps = self.preprocessor.n_rigid_steps_to_run
        self.wrap_step('Stitching', self.preprocessor.stitch_rigid, step_kw_args={'force': True},
                       n_steps=n_steps, abort_func=self.preprocessor.stop_process)
        overlay = [pg.image(self.preprocessor.plot_layout(postfix=postfix))]
        self.main_window.setup_plots(overlay)

    def run_stitching(self):
        """
        Run the actual stitching steps based on the values in the config file (set from the UI).
        Returns
        -------

        """
        if not self.preprocessor.is_tiled:  # BYPASS stitching, just copy or stack
            self.wrap_step('Stitching', clearmap_io.convert,
                           step_args=[self.preprocessor.filename('raw'), self.preprocessor.filename('stitched')])
        else:
            n_steps = self.preprocessor.n_rigid_steps_to_run + self.preprocessor.n_wobbly_steps_to_run
            skip_wobbly = self.params.stitching_wobbly.skip
            if not self.params.stitching_rigid.skip:
                if not self.preprocessor.check_has_all_tiles('raw'):
                    self.progress_watcher.finish()
                    self.main_window.popup('Missing tiles, stitching aborted')
                    return
                self.wrap_step('Stitching', self.preprocessor.stitch_rigid, step_kw_args={'force': True},
                               n_steps=n_steps, abort_func=self.preprocessor.stop_process, close_when_done=skip_wobbly)
            if not skip_wobbly:
                if self.preprocessor.was_stitched_rigid:
                    self.wrap_step(task_name=None, func=self.preprocessor.stitch_wobbly,
                                   step_kw_args={'force': self.params.stitching_rigid.skip}, n_steps=n_steps)

                else:
                    self.main_window.popup('Could not run wobbly stitching <br>without rigid stitching first')
        self.progress_watcher.finish()

    def plot_stitching_results(self):
        """
        Plot the stitched image in 3D in the viewer

        Returns
        -------

        """
        self.params.stitching_general.ui_to_cfg()
        if not self.step_exists('stitching', (self.preprocessor.filename('stitched'))):  # TODO: add arteries option
            return
        dvs = self.preprocessor.plot_stitching_results(parent=self.main_window.centralWidget())
        self.main_window.setup_plots(dvs)

    def convert_output(self):
        """
        Convert the stitched output to the file type specified by the drop down
        menu in the UI

        Returns
        -------

        """
        fmt = self.params.stitching_general.conversion_fmt
        if not self.step_exists('stitching', (self.preprocessor.filename('stitched'))):  # TODO: add arteries option
            return
        self.params.stitching_general.ui_to_cfg()
        self.wrap_step(f'Converting stitched image to {fmt}', self.preprocessor.convert_to_image_format,
                       n_steps=self.preprocessor.n_channels_convert, abort_func=self.preprocessor.stop_process,
                       save_cfg=False, nested=False)

    def clear_landmarks(self):
        self.preprocessor.clear_landmarks()

    def display_auto_to_ref_landmarks_dialog(self):
        """
        Display the dialog to use landmarks for optimising the registration
        of the autofluorescence to the reference

        Returns
        -------

        """
        images = [self.preprocessor.filename('resampled', postfix='autofluorescence'),
                  self.preprocessor.reference_file_path]
        self.__display_landmarks_dialog(images, 'auto_to_reference')

    def display_resampled_to_auto_landmarks_dialog(self):
        """
        Display the dialog to use landmarks for optimising the registration
        of the resampled main channel to autofluorescence

        Returns
        -------

        """
        images = [self.preprocessor.filename('resampled', postfix='autofluorescence'),
                  self.preprocessor.filename('resampled')]
        self.__display_landmarks_dialog(images, 'resampled_to_auto')

    def __display_landmarks_dialog(self, images, direction):
        titles = [os.path.basename(img) for img in images]
        dvs = plot_3d.plot(images, title=titles, arrange=False, sync=False,
                           lut=self.main_window.preference_editor.params.lut,
                           parent=self.main_window.centralWidget())
        self.main_window.setup_plots(dvs)

        landmark_selector = LandmarksSelectorDialog('', params=self.params)
        landmark_selector.data_viewers = dvs
        for i in range(2):
            scatter = pg.ScatterPlotItem()
            dvs[i].enable_mouse_clicks()
            dvs[i].view.addItem(scatter)
            dvs[i].scatter = scatter
            coords = [landmark_selector.fixed_coords(), landmark_selector.moving_coords()][i]  # FIXME: check order (A to B)
            dvs[i].scatter_coords = Scatter3D(coords, colors=np.array(landmark_selector.colors),
                                              half_slice_thickness=3)
            callback = [landmark_selector.set_fixed_coords, landmark_selector.set_moving_coords][i]
            dvs[i].mouse_clicked.connect(callback)
        callbacks = {
            'auto_to_reference': self.write_auto_to_ref_registration_landmark_coords,
            'resampled_to_auto': self.write_resampled_to_auto_registration_landmark_coords
        }
        landmark_selector.dlg.buttonBox.accepted.connect(callbacks[direction])
        self.landmark_selector = landmark_selector  # REFACTOR: find better way to keep in scope
        # return landmark_selector

    def write_auto_to_ref_registration_landmark_coords(self):
        """
        Write the corresponding landmarks to file for use in landmark optimised registration

        Returns
        -------

        """
        self.__write_registration_landmark_coords('auto_to_reference')

    def write_resampled_to_auto_registration_landmark_coords(self):
        """
        Write the corresponding landmarks to file for use in landmark optimised registration

        Returns
        -------

        """
        self.__write_registration_landmark_coords('resampled_to_auto')

    def __write_registration_landmark_coords(self, direction):
        landmarks_file_paths = [self.preprocessor.get_autofluo_pts_path(direction)]
        if direction == 'auto_to_reference':
            landmarks_file_paths.append(self.preprocessor.ref_pts_path)
        elif direction == 'resampled_to_auto':
            landmarks_file_paths.append(self.preprocessor.resampled_pts_path)
        else:
            raise ValueError(f'Direction must be one of ("auto_to_reference", "resampled_to_auto"), got {direction}')
        markers = [mrkr for mrkr in self.landmark_selector.coords if all(mrkr)]
        for i, f_path in enumerate(landmarks_file_paths):
            if not os.path.exists(os.path.dirname(f_path)):
                os.mkdir(os.path.dirname(f_path))
            with open(f_path, 'w') as landmarks_file:
                landmarks_file.write(f'point\n{len(markers)}\n')
                for marker in markers:
                    x, y, z = marker[i]
                    landmarks_file.write(f'{x} {y} {z}\n')
        self.landmark_selector.dlg.close()
        self.landmark_selector = None

    def run_registration(self):
        """
        Run the actual registration between the sample and the reference atlas.

        Returns
        -------

        """
        self.main_window.make_progress_dialog('Registering', n_steps=4, abort=self.preprocessor.stop_process,
                                              parent=self.main_window)  # FIXME: compute n_steps (par of processor)
        self.setup_atlas()
        if not self.params.registration.skip_resampling:
            if self.preprocessor.autofluorescence_is_tiled and \
                    not self.preprocessor.check_has_all_tiles('autofluorescence'):
                self.main_window.progress_watcher.finish()
                self.main_window.print_status_msg('Registration skipped because of missing tiles')
                return
            self.main_window.wrap_in_thread(self.preprocessor.resample_for_registration, force=True)
            # self.main_window.print_status_msg('Resampled')
        self.main_window.wrap_in_thread(self.preprocessor.align)

    def __prepare_registration_results_graph(self, step='ref_to_auto'):
        if step == 'ref_to_auto':
            img_paths = (self.preprocessor.filename('resampled', postfix='autofluorescence'),
                         self.preprocessor.aligned_autofluo_path)
        elif step == 'auto_to_raw':
            img_paths = (self.preprocessor.filename('resampled'),  # TODO: check direction
                         os.path.join(self.preprocessor.filename('resampled_to_auto'), 'result.0.mhd'))
        else:
            raise ValueError(f'Unrecognised step option {step}, should be one of ["ref_to_auto", "auto_to_raw"]')
        if not self.step_exists('registration', img_paths):
            raise ValueError(f'Missing requirements {img_paths}')
        image_sources = copy.deepcopy(list(img_paths))
        for i, im_path in enumerate(image_sources):
            if im_path.endswith('.mhd'):
                image_sources[i] = mhd_read(im_path)
        titles = [os.path.basename(img) for img in img_paths]
        return image_sources, titles

    def plot_registration_results_side_by_side_raw(self):
        """
        Plot the result of the registration between raw and autofluorescence
        side-by-side in synchronised plots

        Returns
        -------

        """
        self.main_window.clear_plots()
        image_sources, titles = self.__prepare_registration_results_graph('auto_to_raw')
        dvs = plot_3d.plot(image_sources, title=titles, arrange=False, sync=True,
                           lut=self.main_window.preference_editor.params.lut,
                           parent=self.main_window.centralWidget())
        link_dataviewers_cursors(dvs)
        self.main_window.setup_plots(dvs, ['autofluo', 'aligned'])

    def plot_registration_results_side_by_side(self):
        """
        Plot the result of the registration between autofluorescence and reference atlas
        side-by-side in synchronised plots

        Returns
        -------

        """
        self.main_window.clear_plots()
        image_sources, titles = self.__prepare_registration_results_graph()
        dvs = plot_3d.plot(image_sources, title=titles, arrange=False, sync=True,
                           lut=self.main_window.preference_editor.params.lut,
                           parent=self.main_window.centralWidget())
        link_dataviewers_cursors(dvs)
        self.main_window.setup_plots(dvs, ['autofluo', 'aligned'])

    def plot_registration_results_composite_raw(self):
        """
        Plot the result of the registration between raw and autofluorescence
        as an overlay plot

        Returns
        -------

        """
        self.main_window.clear_plots()
        image_sources, titles = self.__prepare_registration_results_graph('auto_to_raw')
        dvs = plot_3d.plot([image_sources, ], title=' '.join(titles), arrange=False, sync=True,
                           lut=self.main_window.preference_editor.params.lut,
                           parent=self.main_window.centralWidget())
        self.main_window.setup_plots(dvs)

    def plot_registration_results_composite(self):
        """
        Plot the result of the registration between autofluorescence and reference atlas
        as an overlay plot

        Returns
        -------

        """
        self.main_window.clear_plots()
        image_sources, titles = self.__prepare_registration_results_graph()
        dvs = plot_3d.plot([image_sources, ], title=' '.join(titles), arrange=False, sync=True,
                           lut=self.main_window.preference_editor.params.lut,
                           parent=self.main_window.centralWidget())
        self.main_window.setup_plots(dvs)


class CellCounterTab(PostProcessingTab):
    """
    The tab responsible for the cell detection and cell coordinates alignment
    """
    def __init__(self, main_window, tab_idx=2):
        super().__init__(main_window, 'CellMap', tab_idx, 'cell_map_tab')

        self.cell_intensity_histogram = None
        self.cell_size_histogram = None
        self.preprocessor = None
        self.cell_detector = None

        self.advanced_control_names = ['detectionShapeGroupBox']

    def set_params(self, sample_params, alignment_params):
        """
        Set the params object which links the UI and the configuration file
        Parameters
        ----------
        sample_params : SampleParameters
            The params object that links the UI to the config for all the properties intrinsic to the sample
        alignment_params: AlignmentParams
            The params object that links the UI to the config for things related to sample alignment

        Returns
        -------
        """
        self.params = CellMapParams(self.ui, sample_params, alignment_params, self.main_window.src_folder)

    def setup_workers(self):
        """
         Setup the cell detection worker, which handle the computations associated with this tab

         Returns
         -------

         """
        if self.preprocessor.workspace is not None:
            self.params.ui_to_cfg()
            self.cell_detector = CellDetector(self.preprocessor)
        else:
            self.main_window.print_warning_msg('Workspace not initialised')

    def setup_cell_detector(self):
        """
        Post configuration of the CellDetector object  # FIXME: check redundancy with above

        Returns
        -------

        """
        if self.cell_detector.preprocessor is None and self.preprocessor.workspace is not None:  # preproc initialised
            self.params.ui_to_cfg()
            self.cell_detector.setup(self.preprocessor)
            self.update_cell_number()

    def setup(self):
        """
        Setup the UI elements, notably the signal/slot connections which are not
        automatically set through the params object attribute

        Returns
        -------

        """
        self.init_ui()

        self.ui.toolBox.currentChanged.connect(self.handle_tool_tab_changed)

        self.ui.detectionPreviewTuningOpenPushButton.clicked.connect(self.plot_debug_cropping_interface)
        # FIXME: add load icon
        self.ui.detectionPreviewTuningCropPushButton.clicked.connect(self.create_cell_detection_tuning_sample)
        self.ui.detectionPreviewPushButton.clicked.connect(self.run_tuning_cell_detection)

        self.ui.previewCellFiltersPushButton.clicked.connect(self.preview_cell_filter)

        self.ui.runCellMapPushButton.clicked.connect(self.run_cell_map)

        self.ui.cellMapPlotVoxelizationPushButton.clicked.connect(self.plot_cell_map_results)
        self.ui.cellMap3dScatterOnRefPushButton.clicked.connect(self.plot_cells_scatter_w_atlas_colors)
        self.ui.cellMap3dScatterOnStitchedPushButton.clicked.connect(self.plot_cells_scatter_w_atlas_colors_raw)

    def setup_cell_param_histogram(self, cells, plot_item, key='size', x_log=False):
        """
        Plots the histogram of the cell parameter defined by key. This is used to display the
        distribution of cell sizes or intensities

        Parameters
        ----------
        cells : pd.DataFrame
            The Cells dataframe containing one row per detected cell
        plot_item : QWidget or None
            The Plot element to plot into. If None, creates a new one
        key : str
            The key (cell attribute) in the dataframe to plot.
             One of 'size' or 'source'
        x_log : bool
            X axis is log

        Returns
        -------

        """
        values = cells[key].values
        hist, bin_edges = np.histogram(values, bins=20)
        if plot_item is None:
            widget = pg.plot(hist, bin_edges[:-1], pen=pg.mkPen(DarkPalette.COLOR_ACCENT_2))
        else:
            widget = plot_item
            widget.plot(hist, bin_edges[:-1], pen=pg.mkPen(DarkPalette.COLOR_ACCENT_2), clear=True)
        widget.setBackground(DarkPalette.COLOR_BACKGROUND_2)
        widget.setLogMode(x=x_log)
        return widget

    def voxelize(self):
        """
        Creates the cell density plot
        Returns
        -------

        """
        if os.path.exists(self.preprocessor.filename('cells', postfix='filtered')):
            self.wrap_step('Voxelization', self.cell_detector.voxelize,
                           abort_func=self.cell_detector.stop_process, nested=False)#, main_thread=True)
        else:
            self.main_window.popup('Could not run voxelization, missing filtered cells table. '
                                   'Please ensure that cell filtering has been run.', base_msg='Missing file')

    def set_progress_watcher(self, watcher):
        """
        Setup the watcher object that will handle the progress in the computation for this tab

        Parameters
        ----------
        watcher : ProgressWatcher
            The object that tracks the progress

        Returns
        -------

        """
        if self.cell_detector is not None and self.cell_detector.preprocessor is not None:  # If initialised
            self.cell_detector.set_progress_watcher(watcher)

    def plot_debug_cropping_interface(self):
        """
        Plot the orthoslicer to select a subset of the sample to perform cell detections
        tests on

        Returns
        -------

        """
        self.plot_slicer('detectionSubset', self.ui, self.params)

    def handle_tool_tab_changed(self, tab_idx):
        """
        Triggered when a new sub tab (tooltab) of the cell detection tab is selected.
        It will either plot the cell parameter distributions or update the cell count display.

        Parameters
        ----------
        tab_idx

        Returns
        -------

        """
        if tab_idx == 1:
            for sample_type in ('normal', 'debug'):
                try:
                    self.preprocessor.workspace.debug = sample_type == 'debug'
                    cells_df = pd.DataFrame(np.load(self.preprocessor.workspace.filename('cells', postfix='raw')))
                    self.__plot_histograms(cells_df)
                    break
                except FileNotFoundError:
                    end = 'skipping' if sample_type == 'debug' else 'trying debug'
                    print(f'Could not find {sample_type} cells dataframe file, {end}')
                finally:
                    self.preprocessor.workspace.debug = False
            else:
                self.main_window.popup('No cells file found, cannot display histograms yet')
        elif tab_idx == 3:
            self.update_cell_number()

    def __plot_histograms(self, cells_df):
        self.cell_size_histogram = self.setup_cell_param_histogram(cells_df, self.cell_size_histogram, 'size')
        self.cell_intensity_histogram = self.setup_cell_param_histogram(cells_df, self.cell_intensity_histogram,
                                                                        'source')
        layout = self.ui.cellDetectionThresholdsLayout
        if layout.count() <= 4:
            label = layout.takeAt(2).widget()
            controls = layout.takeAt(2).widget()
            graph_width = label.width() + controls.width()
            graph_height = 50
            self.cell_size_histogram.resize(graph_width, graph_height)
            self.cell_size_histogram.setMaximumSize(graph_width, graph_height)
            layout.addWidget(self.cell_size_histogram, 1, 0, 1, 2)
            layout.addWidget(label, 2, 0, 1, 1)
            layout.addWidget(controls, 2, 1, 1, 1)
            self.cell_intensity_histogram.resize(graph_width, graph_height)
            self.cell_intensity_histogram.setMaximumSize(graph_width, graph_height)
            layout.addWidget(self.cell_intensity_histogram, 4, 0, 1, 2)

            lyt = self.ui.cellDetectionThresholdsLayout
            group_box = lyt.parent()
            # widgets = [lyt.itemAt(i).widget() for i in range(lyt.count())]
            page = group_box.parent()
            # heights = [lyt.itemAt(i).widget().height() for i in range(lyt.count())]
            container = page.parent().parent()
            container.setMinimumHeight(container.parent().height() - container.height()
                                       + group_box.height())

    def create_cell_detection_tuning_sample(self):
        """
        Create an array from a subset of the sample to perform tests on
        Returns
        -------

        """
        self.wrap_step('Creating tuning sample', self.cell_detector.create_test_dataset,
                       step_kw_args={'slicing': self.params.slicing}, nested=False)

    def run_tuning_cell_detection(self):
        """
        Run the cell detection on a subset of the sample which was previously selected

        Returns
        -------

        """
        self.wrap_step('Cell detection preview', self.cell_detector.run_cell_detection, step_kw_args={'tuning': True})
        if self.cell_detector.stopped:
            return
        with self.cell_detector.workspace.tmp_debug:
            self.plot_detection_results()

    def detect_cells(self):  # TODO: merge w/ above w/ tuning option
        """
        Run the cell detection on the whole sample

        Returns
        -------

        """
        self.wrap_step('Detecting cells', self.cell_detector.run_cell_detection,
                       step_kw_args={'tuning': False, 'save_shape': self.params.save_shape},
                       abort_func=self.cell_detector.stop_process)
        if self.cell_detector.stopped:
            return
        self.update_cell_number()

    def update_cell_number(self):  # FIXME: try except or check that cells and cells filtered exist
        """
        Update the cell count number displayed based on the size of the raw and filtered cell detection files
        Returns
        -------

        """
        self.ui.nDetectedCellsLabel.setText(
            format_long_nb_to_str(self.cell_detector.get_n_detected_cells()))
        self.ui.nDetectedCellsAfterFilterLabel.setText(
            format_long_nb_to_str(self.cell_detector.get_n_filtered_cells()))

    # def reset_detected(self):
    #     self.cell_detector.detected = False

    def plot_detection_results(self):  # FIXME: check only steps
        """
        Display the different steps of the cell detection in a grid to evaluate the filters

        Returns
        -------

        """
        self.main_window.clear_plots()
        if not self.step_exists('cell detection', [self.preprocessor.filename('stitched')]):
            return
        dvs = self.cell_detector.preview_cell_detection(parent=self.main_window.centralWidget(),
                                                        arrange=False, sync=True)  # TODO: add close
        if len(dvs) == 1:
            self.main_window.print_warning_msg('Preview not run, '
                                               'will only display stitched image for memory space reasons')
        else:
            link_dataviewers_cursors(dvs)
        self.main_window.setup_plots(dvs)

    def plot_cell_filter_results(self):
        """
        Plot the cells as colored dots on top of the raw image fraction used for tests
        Returns
        -------

        """
        self.main_window.clear_plots()
        if not self.step_exists('cell filtering', [self.preprocessor.filename('stitched'),
                                                   self.preprocessor.filename('cells', postfix='filtered')]):
            return
        dvs = self.cell_detector.plot_filtered_cells(smarties=True)
        self.main_window.setup_plots(dvs)

    def plot_cells_scatter_w_atlas_colors(self):
        """
        Plot the cells as colored symbols on top of the resampled (aligned) image
        Returns
        -------

        """
        self.main_window.clear_plots()
        if self.preprocessor.was_registered:
            required_paths = [self.preprocessor.reference_file_path]
        else:
            required_paths = [self.preprocessor.filename('resampled')]
        required_paths.append(self.cell_detector.df_path)
        if not self.step_exists('cell count', required_paths):
            return
        dvs = self.cell_detector.plot_cells_3d_scatter_w_atlas_colors(parent=self.main_window)  # FIXME: add progress
        self.main_window.setup_plots(dvs)

    def plot_cells_scatter_w_atlas_colors_raw(self):
        """
        Plot the cells as colored symbols on top of the raw stitched (not aligned) image
        Returns
        -------

        """
        self.main_window.clear_plots()
        if not self.step_exists('cell count', [self.preprocessor.filename('stitched'),
                                               self.cell_detector.df_path]):
            return
        dvs = self.cell_detector.plot_cells_3d_scatter_w_atlas_colors(raw=True, parent=self.main_window)
        self.main_window.setup_plots(dvs)

    def __filter_cells(self, is_last_step=True):
        raw_cells_path = self.preprocessor.filename('cells', postfix='raw')
        if os.path.exists(raw_cells_path):
            self.wrap_step('Filtering cells', self.cell_detector.filter_cells, n_steps=2+(1 - is_last_step),
                           abort_func=self.cell_detector.stop_process, close_when_done=False)
            self.wrap_step('Voxelizing', self.cell_detector.voxelize, step_args=['filtered'], save_cfg=False,
                           close_when_done=is_last_step)#, main_thread=True)
        self.plot_cell_filter_results()

    def preview_cell_filter(self):  # FIXME: circular calls TEST
        with self.cell_detector.workspace.tmp_debug:
            self.__filter_cells()

    def filter_cells(self):
        self.__filter_cells(is_last_step=False)
        self.wrap_step('Aligning', self.cell_detector.atlas_align, abort_func=self.cell_detector.stop_process,
                       save_cfg=False)
        self.cell_detector.export_collapsed_stats()

    def run_cell_map(self):
        """
        Run the whole pipeline at once
        Returns
        -------

        """
        self.params.ui_to_cfg()
        self.update_cell_number()
        if self.params.detect_cells:
            self.detect_cells()
            self.update_cell_number()
        if self.params.filter_cells:
            self.cell_detector.post_process_cells()  # FIXME: save cfg and use progress
            self.update_cell_number()
        if self.params.voxelize:
            self.voxelize()
        if self.params.plot_when_finished:
            self.plot_cell_map_results()
        # WARNING: some plots in .post_process_cells() without UI params

    def plot_cell_map_results(self):
        """
        Plot the voxelisation (density map) result
        Returns
        -------

        """
        self.main_window.clear_plots()
        if not self.step_exists('voxelization', [self.preprocessor.filename('density', postfix='counts')]):
            return
        dvs = self.cell_detector.plot_voxelized_counts(arrange=False)
        self.main_window.setup_plots(dvs)


class VasculatureTab(PostProcessingTab):
    """
    The tab responsible for the vasculature tracts detection, graph extraction and analysis
    """
    def __init__(self, main_window, tab_idx=3):
        super().__init__(main_window, 'Vasculature', tab_idx, 'vasculature_tab')

        self.params = None
        self.preprocessor = None

        self.binary_vessel_processor = None
        self.vessel_graph_processor = None

    def set_params(self, sample_params, alignment_params):
        """
        Set the params object which links the UI and the configuration file
        Parameters
        ----------
        sample_params : SampleParameters
            The params object that links the UI to the config for all the properties intrinsic to the sample
        alignment_params: AlignmentParams
            The params object that links the UI to the config for things related to sample alignment

        Returns
        -------
        """
        self.params = VesselParams(self.ui, sample_params, alignment_params, self.main_window.src_folder)

    def setup_preproc(self, pre_processor):
        """
        Associate the preprocessor to the current tab

        Parameters
        ----------
        pre_processor : PreProcessor

        Returns
        -------

        """
        self.preprocessor = pre_processor

    def setup_workers(self):
        """
         Setup the BinaryVesselProcessor and VesselGraphProcessor workers,
          which handle the computations associated with this tab

         Returns
         -------

         """
        if self.preprocessor.workspace is not None:
            self.params.ui_to_cfg()
            self.binary_vessel_processor = BinaryVesselProcessor(self.preprocessor)
            self.vessel_graph_processor = VesselGraphProcessor(self.preprocessor)
        else:
            self.main_window.print_warning_msg('Workspace not initialised')

    def setup_vessel_processors(self):
        """
        Post configuration of the BinaryVesselProcessor and VesselGraphProcessor objects  # FIXME: check redundancy with above

        Returns
        -------

        """
        if self.preprocessor.workspace is not None:  # Inited
            if self.binary_vessel_processor.preprocessor is None:
                self.params.ui_to_cfg()
                self.binary_vessel_processor.setup(self.preprocessor)
            if self.vessel_graph_processor.preprocessor is None:
                self.params.ui_to_cfg()
                self.vessel_graph_processor.setup(self.preprocessor)

    def setup(self):
        """
        Setup the UI elements, notably the signal/slot connections which are not
        automatically set through the params object attribute

        Returns
        -------

        """
        self.init_ui()

        # ######################################## BINARIZATION ##############################
        self.connect_whats_this(self.ui.binarizationRawClippingRangeInfoToolButton, self.ui.binarizationRawClippingRangeLbl)
        self.connect_whats_this(self.ui.binarizationRawThresholdInfoToolButton, self.ui.binarizationRawThresholdLbl)
        self.connect_whats_this(self.ui.binarizationRawDeepFillingInfoToolButton, self.ui.binarizationRawDeepFillingCheckBox)
        self.ui.binarizeVesselsPushButton.clicked.connect(functools.partial(self.binarize_channel, channel='raw'))
        self.connect_whats_this(self.ui.binarizationArteriesClippingRangeInfoToolButton, self.ui.binarizationArteriesClippingRangeLbl)
        self.connect_whats_this(self.ui.binarizationArteriesThresholdInfoToolButton, self.ui.binarizationArteriesThresholdLbl)
        self.connect_whats_this(self.ui.binarizationArteriesDeepFillingInfoToolButton, self.ui.binarizationArteriesDeepFillingCheckBox)
        self.ui.binarizeArteriesPushButton.clicked.connect(functools.partial(self.binarize_channel, channel='arteries'))
        self.ui.binarizationCombinePushButton.clicked.connect(self.combine)

        self.ui.binarizationPlotSideBySidePushButton.clicked.connect(
            functools.partial(self.plot_binarization_results, plot_side_by_side=True))
        self.ui.binarizationPlotOverlayPushButton.clicked.connect(
            functools.partial(self.plot_binarization_results, plot_side_by_side=False))

        # ######################################## GRAPH ##############################
        self.ui.buildGraphSelectAllCheckBox.stateChanged.connect(self.__select_all_graph_steps)
        self.ui.buildGraphPushButton.clicked.connect(self.build_graph)
        self.ui.unloadGraphsPushButton.clicked.connect(self.unload_temporary_graphs)
        self.connect_whats_this(self.ui.buildGraphPushButtonInfoToolButton, self.ui.buildGraphPushButton)

        self.connect_whats_this(self.ui.maxArteriesTracingIterationsInfoToolButton, self.ui.maxArteriesTracingIterationsLbl)  # FIXME: try to automatise
        self.connect_whats_this(self.ui.minArterySizeInfoToolButton, self.ui.minArterySizeLbl)
        self.connect_whats_this(self.ui.veinIntensityRangeOnArteriesChannelInfoToolButton, self.ui.veinIntensityRangeOnArteriesChannelLbl)
        self.connect_whats_this(self.ui.restrictiveMinVeinRadiusInfoToolButton, self.ui.restrictiveMinVeinRadiusLbl)
        self.connect_whats_this(self.ui.permissiveMinVeinRadiusInfoToolButton, self.ui.permissiveMinVeinRadiusLbl)
        self.connect_whats_this(self.ui.finalMinVeinRadiusInfoToolButton, self.ui.finalMinVeinRadiusLbl)
        self.connect_whats_this(self.ui.maxVeinsTracingIterationsInfoToolButton, self.ui.maxVeinsTracingIterationsLbl)
        self.connect_whats_this(self.ui.minVeinSizeInfoToolButton, self.ui.minVeinSizeLbl)
        self.ui.postProcessVesselTypesPushButton.clicked.connect(self.post_process_graph)

        # ######################################## DISPLAY ##############################
        # slicer
        self.ui.graphSlicerButtonBox.connectOpen(self.plot_graph_type_processing_chunk_slicer)
        self.connect_whats_this(self.ui.graphSlicerGroupBoxInfoToolButton, self.ui.graphSlicerGroupBox)

        self.ui.plotGraphPickRegionPushButton.clicked.connect(self.pick_region)
        self.ui.plotGraphChunkPushButton.clicked.connect(self.display_graph_chunk_from_cfg)
        self.ui.plotGraphClearPlotPushButton.clicked.connect(self.main_window.clear_plots)

        self.ui.voxelizeGraphPushButton.clicked.connect(self.voxelize)
        self.ui.plotGraphVoxelizationPushButton.clicked.connect(self.plot_voxelization)
        self.ui.runAllVasculaturePushButton.clicked.connect(self.run_all)

    def unload_temporary_graphs(self):
        """
        Unload the temporary vasculature graph objects to free up RAM
        """
        self.vessel_graph_processor.unload_temporary_graphs()

    def set_progress_watcher(self, watcher):
        """
        Setup the watcher object that will handle the progress in the computation for this tab

        Parameters
        ----------
        watcher : ProgressWatcher
            The object that tracks the progress

        Returns
        -------

        """
        if self.binary_vessel_processor is not None and self.binary_vessel_processor.preprocessor is not None:
            self.binary_vessel_processor.set_progress_watcher(watcher)
        if self.vessel_graph_processor is not None and self.vessel_graph_processor.preprocessor is not None:
            self.vessel_graph_processor.set_progress_watcher(watcher)

    def __select_all_graph_steps(self, state):
        for chk_bx in (self.ui.buildGraphSkeletonizeCheckBox, self.ui.buildGraphBuildCheckBox,
                       self.ui.buildGraphCleanCheckBox, self.ui.buildGraphReduceCheckBox,
                       self.ui.buildGraphTransformCheckBox, self.ui.buildGraphRegisterCheckBox):
            chk_bx.setCheckState(state)  # TODO: check that not tristate

    # ####################### BINARY  #######################

    def binarize_channel(self, channel, stop_on_error=False):
        """
        Perform all the selected binarization steps on the given channel

        Parameters
        ----------
        channel
        stop_on_error

        Returns
        -------

        """
        # FIXME: n_steps = self.params.binarization_params.n_steps
        try:
            self.wrap_step('Vessel binarization', self.binary_vessel_processor.binarize_channel,
                           step_args=[channel], abort_func=self.binary_vessel_processor.stop_process)
            self.wrap_step('Vessel binarization', self.binary_vessel_processor.smooth_channel,
                           step_args=[channel], abort_func=self.binary_vessel_processor.stop_process)
            self.wrap_step('Vessel binarization', self.binary_vessel_processor.fill_channel,
                           step_args=[channel], abort_func=self.binary_vessel_processor.stop_process,
                           main_thread=True)  # WARNING: The parallel cython loops inside cannot run from child thread
            self.wrap_step('Vessel binarization', self.binary_vessel_processor.deep_fill_channel,
                           step_args=[channel], abort_func=self.binary_vessel_processor.stop_process)
        except ClearMapVRamException as err:
            if stop_on_error:
                raise err

    def combine(self):
        """
        Combine the binarized (thresholded) version of the different channels.
        Returns
        -------

        """
        self.wrap_step('Combining channels', self.binary_vessel_processor.combine_binary,
                       abort_func=self.binary_vessel_processor.stop_process)

    # FIXME: channel
    def plot_binarization_results(self, plot_side_by_side=True):
        """
        Plot the thresholded images resulting from the binarization at the steps specified
        by the comboboxes in the UI.

        Parameters
        ----------
        plot_side_by_side

        Returns
        -------

        """
        self.main_window.clear_plots()
        binarization_params = self.params.binarization_params
        steps = (binarization_params.plot_step_1,
                 binarization_params.plot_step_2)
        channels = (binarization_params.plot_channel_1,
                    binarization_params.plot_channel_2)
        channels = [c.replace('all vessels', 'raw') for c in channels]
        channels = [c for s, c in zip(steps, channels) if s is not None]
        steps = [s for s in steps if s is not None]
        files = [self.binary_vessel_processor.steps[c].path_from_step_name(s) for s, c in zip(steps, channels)]
        for f in files:
            if not os.path.exists(f):
                self.main_window.popup(f'Missing file {f}')
                return

        dvs = self.binary_vessel_processor.plot_results(steps, channels=channels,
                                                        side_by_side=plot_side_by_side,
                                                        arrange=False, parent=self.main_window)
        self.main_window.setup_plots(dvs, steps)

    # ###########################  GRAPH  #############################

    def run_all(self):
        """
        Run the whole vasculature pipeline at once

        Returns
        -------

        """
        try:
            self.binarize_channel('raw', stop_on_error=True)
            self.binarize_channel('arteries', stop_on_error=True)
        except ClearMapVRamException:
            return
        self.combine()
        self.build_graph()
        self.post_process_graph()
        self.voxelize()

    def build_graph(self):
        """
        Run the pipeline to build the vasculature graph
        Returns
        -------

        """
        # FIXME: n_steps = 4
        self.wrap_step('Building vessel graph', self.vessel_graph_processor.build_graph,
                       abort_func=self.vessel_graph_processor.stop_process, main_thread=True)
        self.wrap_step('Building vessel graph', self.vessel_graph_processor.clean_graph,
                       abort_func=self.vessel_graph_processor.stop_process)
        self.wrap_step('Building vessel graph', self.vessel_graph_processor.reduce_graph,
                       abort_func=self.vessel_graph_processor.stop_process)
        self.wrap_step('Building vessel graph', self.vessel_graph_processor.register,
                       abort_func=self.vessel_graph_processor.stop_process)

    def plot_graph_type_processing_chunk_slicer(self):  # Refactor: rename
        """
        Plot the orthoslicer to pick a sub part of the graph to display because
        depending on the display options, the whole graph may not fit in memory

        Returns
        -------

        """
        self.plot_slicer('graphConstructionSlicer', self.ui, self.params.visualization_params)

    def __get_tube_map_slicing(self):
        self.params.visualization_params.ui_to_cfg()  # Fix for lack of binding between 2 sets of range interfaces
        return self.params.visualization_params.slicing

    def display_graph_chunk(self, graph_step):
        """
        Display a chunk of the graph selected with the slicer

        Parameters
        ----------
        graph_step : str
            The name of the step to display (from 'raw', 'cleaned', 'reduced', 'annotated')

        Returns
        -------

        """
        self.main_window.clear_plots()
        slicing = self.__get_tube_map_slicing()
        try:
            dvs = self.vessel_graph_processor.visualize_graph_annotations(slicing, plot_type='mesh',
                                                                          graph_step=graph_step, show=False)
        except PlotGraphError as err:
            self.main_window.popup(str(err), base_msg='PlotGraphError')
            return
        self.main_window.setup_plots(dvs)
        self.main_window.perf_monitor.stop()

    def display_graph_chunk_from_cfg(self):  # REFACTOR: split ?
        self.display_graph_chunk(self.params.visualization_params.graph_step)

    def plot_graph_structure(self):
        """
        Plot a subregion of the vasculature graph corresponding to a structure
        using the atlas registration results
        Returns
        -------

        """
        structure_id = self.params.visualization_params.structure_id
        if structure_id is not None:
            self._plot_graph_structure(structure_id, annotation.find(structure_id, key='id')['rgb'])
        else:
            print('No structure ID')
        self.main_window.structure_selector.close()

    def post_process_graph(self):
        """
        Post process the graph by filtering, tracing and removing capillaries
        Returns
        -------

        """
        self.wrap_step('Post processing vasculature graph', self.vessel_graph_processor.post_process,
                       abort_func=self.vessel_graph_processor.stop_process)  # FIXME: n_steps = 8

    def pick_region(self):
        """
        Open a dialog to select a brain region and plot it
        Returns
        -------

        """
        self.main_window.structure_selector.structure_selected.connect(
            self.params.visualization_params.set_structure_id)
        self.main_window.structure_selector.onAccepted(self.plot_graph_structure)
        self.main_window.structure_selector.onRejected(self.main_window.structure_selector.close)
        self.main_window.structure_selector.show()

    def _plot_graph_structure(self, structure_id, structure_color):
        self.main_window.clear_plots()
        try:
            dvs = self.vessel_graph_processor.plot_graph_structure(structure_id,
                                                                   self.params.visualization_params.plot_type)
        except PlotGraphError as err:
            self.main_window.popup(str(err), base_msg='PlotGraphError')
            return
        if dvs:
            self.main_window.setup_plots(dvs)
            self.main_window.perf_monitor.stop()

    def voxelize(self):
        """
        Run the voxelisation (density map) on the vasculature graph
        Returns
        -------

        """
        voxelization_params = {
            'weight_by_radius': self.params.visualization_params.weight_by_radius,
            'vertex_degrees': self.params.visualization_params.vertex_degrees
        }
        self.wrap_step('Running voxelization', self.vessel_graph_processor.voxelize,
                       step_kw_args=voxelization_params)#, main_thread=True)

    def plot_voxelization(self):
        """
        Plot the density map
        Returns
        -------

        """
        self.main_window.clear_plots()
        dvs = self.vessel_graph_processor.plot_voxelization(self.main_window.centralWidget())
        self.main_window.setup_plots(dvs)


################################################################################################

class GroupAnalysisProcessor:
    def __init__(self, progress_watcher, results_folder=None):
        self.results_folder = results_folder
        self.progress_watcher = progress_watcher

    def plot_p_vals(self, selected_comparisons, groups, parent=None):
        p_vals_imgs = []
        for pair in selected_comparisons:  # TODO: Move to processor object to be wrapped
            gp1_name, gp2_name = pair
            # Reread because of cm_io orientation
            p_val_path = os.path.join(self.results_folder, f'p_val_colors_{gp1_name}_{gp2_name}.tif')

            p_vals_imgs.append(clearmap_io.read(p_val_path))
        pre_proc = init_preprocessor(os.path.join(self.results_folder, groups[selected_comparisons[0][0]][0]))
        atlas = clearmap_io.read(pre_proc.annotation_file_path)
        if len(p_vals_imgs) == 1:
            gp1_name, gp2_name = selected_comparisons[0]
            gp1_img = clearmap_io.read(os.path.join(self.results_folder, f'avg_density_{gp1_name}.tif'))
            gp2_img = clearmap_io.read(os.path.join(self.results_folder, f'avg_density_{gp2_name}.tif'))
            colored_atlas = create_color_annotation(pre_proc.annotation_file_path)
            images = [gp1_img, gp2_img, p_vals_imgs[0], colored_atlas]
            titles = [gp1_name, gp2_name, 'P values', 'colored_atlas']
            luts = ['flame', 'flame', None, None]
            min_maxes = [None, None, None, (0, 255)]
        else:
            images = p_vals_imgs
            titles = [f'{gp1_name} vs {gp2_name} p values' for gp1_name, gp2_name in selected_comparisons]
            luts = None
            min_maxes = None
        dvs = plot_3d.plot(images, title=titles, arrange=False, sync=True,
                           lut=luts, min_max=min_maxes,
                           parent=parent)
        names_map = annotation.get_names_map()
        for dv in dvs:
            dv.atlas = atlas
            dv.structure_names = names_map
        link_dataviewers_cursors(dvs)
        return dvs

    def compute_p_vals(self, selected_comparisons, groups, wraping_func):
        for pair in selected_comparisons:  # TODO: Move to processor object to be wrapped
            gp1_name, gp2_name = pair
            gp1, gp2 = [groups[gp_name] for gp_name in pair]
            if not density_files_are_comparable(self.results_folder, gp1, gp2):
                raise GroupStatsError('Could not compare files, sizes differ')
            ids = []
            for gp_dir in gp1 + gp2:
                loader = ConfigLoader(gp_dir)
                ids.append(loader.get_cfg('sample')['sample_id'])
            if len(ids) != len(set(ids)):
                raise GroupStatsError('Analysis impossible, some IDs are not unique. please check and start again')
            wraping_func(compare_groups, self.results_folder, gp1_name, gp2_name, gp1, gp2)
            self.progress_watcher.increment_main_progress()

    def run_plots(self, plot_function, selected_comparisons, plot_kw_args):
        dvs = []
        for pair in selected_comparisons:
            gp1_name, gp2_name = pair
            if 'group_names' in plot_kw_args.keys() and plot_kw_args['group_names'] is None:
                plot_kw_args['group_names'] = pair
            stats_df_path = os.path.join(self.results_folder, f'statistics_{gp1_name}_{gp2_name}.csv')
            fig = plot_function(pd.read_csv(stats_df_path), **plot_kw_args)
            browser = QWebEngineView()
            browser.setHtml(mpld3.fig_to_html(fig))  # WARNING: won't work if external objects. Then
            #   saving the html to URl (file///) and then
            #   .load(Url) would be required
            dvs.append(browser)
        return dvs


class BatchTab(GenericTab):
    @property
    def initialised(self):
        return self.params is not None

    def setup(self):
        self.init_ui()

        self.ui.resultsFolderPushButton.clicked.connect(self.setup_results_folder)

        self.ui.folderPickerHelperPushButton.clicked.connect(self.create_wizard)
        self.connect_whats_this(self.ui.folderPickerHelperInfoToolButton, self.ui.folderPickerHelperPushButton)
        self.ui.batchToolBox.setCurrentIndex(0)

    def setup_results_folder(self):
        results_folder = get_directory_dlg(self.main_window.preference_editor.params.start_folder,
                                           'Select the folder where results will be written')
        self.config_loader = ConfigLoader(results_folder)
        cfg_path = self.config_loader.get_cfg_path('batch', must_exist=False)
        if not os.path.exists(cfg_path):
            try:
                default_cfg_file_path = self.config_loader.get_default_path('batch')
                copyfile(default_cfg_file_path, cfg_path)
                self.params.fix_cfg_file(cfg_path)
            except FileNotFoundError as err:
                self.main_window.print_error_msg(f'Could not locate file for "batch"')
                raise err

        self.main_window.logger.set_file(os.path.join(results_folder, 'info.log'))  # WARNING: set logs to global results folder
        self.main_window.error_logger.set_file(os.path.join(results_folder, 'errors.html'))
        self.main_window.progress_watcher.log_path = self.main_window.logger.file.name

        self.params.read_configs(cfg_path)

        self.load_config_to_gui()
        self.params.results_folder = results_folder
        self.params.ui_to_cfg()
        self.setup_workers()

    def create_wizard(self):
        self.params.ui_to_cfg()
        return SamplePickerDialog(self.params.results_folder, params=self.params)  # FIXME: check if results_folder or make both equal with self.params.src_folder


class GroupAnalysisTab(BatchTab):
    def __init__(self, main_window, tab_idx=4):  # REFACTOR: offload computations to BatchProcessor object
        super().__init__(main_window, 'Group analysis', tab_idx, 'group_analysis_tab')

        self.params = None
        self.processor = GroupAnalysisProcessor(self.main_window.progress_watcher)

        self.processing_type = 'batch'

    def set_params(self):
        self.params = GroupAnalysisParams(self.ui, src_folder='', preferences=self.main_window.preference_editor.params)

    def setup(self):
        """
        Setup the UI elements, notably the signal/slot connections which are not
        automatically set through the params object attribute

        Returns
        -------

        """
        super().setup()
        self.ui.runPValsPushButton.clicked.connect(self.run_p_vals)
        self.ui.plotPValsPushButton.clicked.connect(self.plot_p_vals)
        self.ui.batchStatsPushButton.clicked.connect(self.make_group_stats_tables)

    def setup_workers(self):
        self.processor.results_folder = self.params.results_folder
        self.processor.progress_watcher = self.main_window.progress_watcher

    # def setup_workers(self):
    #     self.processor = BatchProcessor(self.params.config)

    def make_group_stats_tables(self):
        self.main_window.print_status_msg('Computing stats table')
        self.main_window.clear_plots()
        # TODO: set abort callback
        self.main_window.make_progress_dialog('Group stats', n_steps=len(self.params.selected_comparisons))
        dvs = []
        for gp1_name, gp2_name in self.params.selected_comparisons:
            df = self.main_window.wrap_in_thread(make_summary, self.params.results_folder,
                                                 gp1_name, gp2_name,
                                                 self.params.groups[gp1_name], self.params.groups[gp2_name],
                                                 output_path=None, save=True)
            self.main_window.progress_watcher.increment_main_progress()
            dvs.append(DataFrameWidget(df).table)
        self.main_window.setup_plots(dvs)
        self.main_window.signal_process_finished()

    def run_p_vals(self):  # FIXME: split compute and display
        self.params.ui_to_cfg()

        self.main_window.print_status_msg('Computing p_val maps')
        # TODO: set abort callback
        self.main_window.make_progress_dialog('P value maps', n_steps=len(self.params.selected_comparisons))
        try:
            self.processor.compute_p_vals(self.params.selected_comparisons, self.params.groups,
                                          self.main_window.wrap_in_thread)
        except GroupStatsError as err:
            self.main_window.popup(str(err), base_msg='Cannot proceed with analysis')
        self.main_window.signal_process_finished()

    def plot_p_vals(self):
        self.main_window.clear_plots()
        self.main_window.print_status_msg('Plotting p_val maps')
        dvs = self.processor.plot_p_vals(self.params.selected_comparisons, self.params.groups,
                                         parent=self.main_window.centralWidget())
        self.main_window.setup_plots(dvs)

    def run_plots(self, plot_function, plot_kw_args):
        self.main_window.clear_plots()
        dvs = self.processor.run_plots(plot_function, self.params.selected_comparisons, plot_kw_args)
        self.main_window.setup_plots(dvs)

    def plot_volcanoes(self):
        self.run_plots(plot_volcano, {'group_names': None, 'p_cutoff': 0.05, 'show': False, 'save_path': ''})

    def plot_histograms(self, fold_threshold=2):
        aba_json_df_path = annotation.default_label_file  # FIXME: aba_json needs fold levels
        self.run_plots(plot_sample_stats_histogram, {'aba_df': pd.read_csv(aba_json_df_path),
                                                     'sort_by_order': True, 'value_cutoff': 0,
                                                     'fold_threshold': fold_threshold, 'fold_regions': True,
                                                     'show': False})


class BatchProcessingTab(BatchTab):
    def __init__(self, main_window, tab_idx=4):
        super().__init__(main_window, 'Batch', tab_idx, 'batch_processing_tab')

        self.params = None
        self.processor = BatchProcessor(self.main_window.progress_watcher)

        self.processing_type = 'batch'

    def set_params(self):
        self.params = BatchProcessingParams(self.ui, src_folder='', preferences=self.main_window.preference_editor.params)
        self.processor.params = self.params

    def setup(self):
        """
        Setup the UI elements, notably the signal/slot connections which are not
        automatically set through the params object attribute

        Returns
        -------

        """
        super().setup()
        self.ui.batchRunPushButton.clicked.connect(self.run_batch_process)

    def run_batch_process(self):
        self.params.ui_to_cfg()
        self.main_window.make_progress_dialog('Analysing samples', n_steps=0, maximum=0)  # TODO: see abort callback
        self.main_window.wrap_in_thread(self.processor.process_folders)
