# -*- coding: utf-8 -*-
"""
tabs
====

The different tabs that correspond to different functionalities of the GUI
"""

import os.path

import numpy as np
from PyQt5.QtWidgets import QDialogButtonBox
import pyqtgraph as pg

from ClearMap.IO.MHD import mhd_read
from ClearMap.Analysis.Statistics.group_statistics import make_summary, density_files_are_comparable, compare_groups
from ClearMap.gui.dialogs import prompt_dialog
from ClearMap.processors.batch_process import process_folders
from ClearMap.processors.cell_map import CellDetector
from ClearMap.processors.sample_preparation import PreProcessor
from ClearMap.processors.tube_map import BinaryVesselProcessor, VesselGraphProcessor
from ClearMap.gui.gui_utils import format_long_nb_to_str, surface_project, np_to_qpixmap, create_clearmap_widget
from ClearMap.Visualization.Qt.DataViewer import link_dataviewers_cursors
from ClearMap.gui.params import ParamsOrientationError, VesselParams, PreferencesParams, SampleParameters, \
    AlignmentParams, CellMapParams, BatchParams
from ClearMap.gui.widgets import PatternDialog, SamplePickerDialog, DataFrameWidget, RedCross, LandmarksSelectorDialog, \
    Scatter3D
from ClearMap.Visualization.Qt import Plot3d as plot_3d


__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


# ############################################ INTERFACES ##########################################

class GenericUi:
    def __init__(self, main_window, name, ui_file_name, widget_class_name):
        """

        Parameters
        ----------
        main_window: ClearMapGui
        name: str
        ui_file_name: str
        widget_class_name: str
        """
        self.main_window = main_window
        self.name = name
        self.ui_file_name = ui_file_name
        self.widget_class_name = widget_class_name
        self.ui = None
        self.params = None
        self.progress_watcher = None

    def init_ui(self):
        self.ui = create_clearmap_widget(f'{self.ui_file_name}.ui', patch_parent_class=self.widget_class_name)
        self.ui.setupUi()
        self.patch_button_boxes()

    def initial_cfg_load(self):
        self.params.cfg_to_ui()

    def set_progress_watcher(self, watcher):
        pass

    def patch_button_boxes(self):
        self.main_window.patch_button_boxes(self.ui)

    def set_params(self, *args):
        raise NotImplementedError()

    def load_params(self):
        self.params.cfg_to_ui()


class GenericTab(GenericUi):
    def __init__(self, main_window, name, tab_idx, ui_file_name):
        """

        Parameters
        ----------
        main_window: ClearMapGui
        name: str
        tab_idx: int
        ui_file_name: str
        """
        
        super().__init__(main_window, name, ui_file_name, 'QTabWidget')

        self.processing_type = None
        self.tab_idx = tab_idx

        self.minimum_width = 200  # REFACTOR:

    def init_ui(self):
        super().init_ui()
        self.ui.setMinimumWidth(self.minimum_width)
        self.main_window.tabWidget.removeTab(self.tab_idx)
        self.main_window.tabWidget.insertTab(self.tab_idx, self.ui, self.name.title())

    def set_params(self, *args):
        raise NotImplementedError()

    def step_exists(self, step_name, file_list):
        for f_path in file_list:
            if not os.path.exists(f_path):
                self.main_window.print_error_msg(f'Missing {step_name} file {f_path}.'
                                                 f'Please ensure {step_name} is run first.')
                return False
        return True

    def setup_workers(self):
        pass


class PostProcessingTab(GenericTab):
    def __init__(self, main_window, name, tab_idx, ui_file_name):
        super().__init__(main_window, name, tab_idx, ui_file_name)

        self.preprocessor = None
        self.processing_type = 'post'

    def set_params(self, sample_params, alignment_params):
        raise NotImplementedError()

    def setup_preproc(self, pre_processor):
        self.preprocessor = pre_processor

    def plot_slicer(self, slicer_prefix, tab, params):
        # if self.preprocessor.was_registered:
        #     img = mhd_read(self.preprocessor.annotation_file_path)  # FIXME: does not work (probably compressed format)
        # else:
        img = self.preprocessor.workspace.source('resampled')
        self.main_window.ortho_viewer.setup(img, params, parent=self.main_window)
        dvs = self.main_window.ortho_viewer.plot_orthogonal_views()
        self.main_window.ortho_viewer.add_cropping_bars()
        self.main_window.setup_plots(dvs, ['x', 'y', 'z'])

        # WARNING: needs to be done after setup
        for axis, ax_max in zip('XYZ', self.preprocessor.raw_stitched_shape):
            getattr(tab, f'{slicer_prefix}{axis}RangeMax').setMaximum(ax_max)
        self.main_window.ortho_viewer.update_ranges()


class GenericDialog(GenericUi):
    def __init__(self, main_window, name, file_name):
        super().__init__(main_window, name, file_name, 'QDialog')

    def init_ui(self):
        super().init_ui()
        self.ui.setWindowTitle(self.name.title())

    def set_params(self, *args):
        raise NotImplementedError()

# ############################################ IMPLEMENTATIONS #######################################


class PreferenceUi(GenericDialog):
    def __init__(self, main_window):
        super().__init__(main_window, 'Preferences', 'preferences_editor')

    def setup(self, font_size):
        self.init_ui()

        self.setup_preferences()

        self.ui.buttonBox.connectApply(self.params.ui_to_cfg)
        self.ui.buttonBox.connectOk(self.apply_prefs_and_close)
        self.ui.buttonBox.connectCancel(self.ui.close)

        self.params.font_size = font_size

        self.ui.fontComboBox.currentFontChanged.connect(self.main_window.set_font)

    def set_params(self, *args):
        self.params = PreferencesParams(self.ui, self.main_window.src_folder)

    def setup_preferences(self):
        self.set_params()
        machine_cfg_path = self.main_window.config_loader.get_default_path('machine')
        if self.main_window.file_exists(machine_cfg_path):
            self.params.get_config(machine_cfg_path)
            self.params.cfg_to_ui()
        else:
            msg = 'Missing machine config file. Please ensure a machine_params.cfg file ' \
                  'is available at {}. This should be done at installation'.format(machine_cfg_path)
            self.main_window.print_error_msg(msg)
            raise FileNotFoundError(msg)

    def open(self):
        return self.ui.exec()

    def apply_prefs_and_close(self):
        self.params.ui_to_cfg()
        self.ui.close()
        self.main_window.reload_prefs()


class SampleTab(GenericTab):
    def __init__(self, main_window, tab_idx=0):
        super().__init__(main_window, 'Sample', tab_idx, 'sample_tab')
        self.mini_brain_scaling = None
        self.mini_brain = None

    def set_params(self, *args):
        self.params = SampleParameters(self.ui, self.main_window.src_folder)

    def setup(self):
        self.init_ui()

        self.ui.srcFolderBtn.clicked.connect(self.main_window.set_src_folder)
        self.ui.sampleIdButtonBox.connectApply(self.main_window.parse_cfg)

        self.ui.launchPatternWizzardPushButton.clicked.connect(self.launch_pattern_wizard)

        self.ui.plotMiniBrainPushButton.clicked.connect(self.plot_mini_brain)

        self.ui.applyBox.connectApply(self.main_window.alignment_tab_mgr.setup_workers)
        self.ui.applyBox.connectSave(self.save_cfg)

        self.ui.advancedCheckBox.stateChanged.connect(self.swap_tab_advanced)

        self.ui.sampleIdButtonBox.button(QDialogButtonBox.Apply).setIcon(self.main_window._reload_icon)  # REFACTOR: put inside style that overrides ?

    def save_cfg(self):  # REFACTOR: use this instead of direct calls to ui_to_cfg
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Sample config saved')

    def initial_cfg_load(self):
        try:
            self.save_cfg()
        except ParamsOrientationError as err:
            self.main_window.popup(str(err), 'Invalid orientation. Defaulting')
            self.params.orientation = (1, 2, 3)

    @property
    def src_folder(self):
        return self.ui.srcFolderTxt.text()

    @src_folder.setter
    def src_folder(self, folder):
        self.ui.srcFolderTxt.setText(folder)

    def swap_tab_advanced(self):  # TODO: implement
        checked = self.ui.advancedCheckBox.isChecked()

    def launch_pattern_wizard(self):
        dlg = PatternDialog(self.src_folder, self.params)
        dlg.exec()

    def plot_mini_brain(self):
        img = self.__transform_mini_brain()
        mask, proj = surface_project(img)
        img = np_to_qpixmap(proj, mask)
        self.ui.miniBrainLabel.setPixmap(img)

    def __transform_mini_brain(self):  # REFACTOR: extract
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


class AlignmentTab(GenericTab):
    def __init__(self, main_window, tab_idx=1):
        super().__init__(main_window, 'Alignment', tab_idx, 'alignment_tab')

        self.processing_type = 'pre'
        self.sample_params = None
        self.preprocessor = PreProcessor()

    def set_params(self, sample_params):
        self.sample_params = sample_params
        self.params = AlignmentParams(self.ui)
        self.params.registration.atlas_id_changed.connect(self.preprocessor.unpack_atlas)

    def setup_workers(self):
        self.sample_params.ui_to_cfg()
        self.preprocessor.setup((self.main_window.preference_editor.params.config,
                                 self.sample_params.config, self.params.config),
                                convert_tiles=False)
        if prompt_dialog('Tile conversion', 'Convert individual tiles to npy for efficiency'):
            self.main_window.make_progress_dialog('Converting tiles', maximum=0, canceled_callback=self.preprocessor.stop_process)
            self.main_window.wrap_in_thread(self.preprocessor.convert_tiles)
            self.main_window.progress_dialog.done(1)
        self.setup_atlas()

    def setup(self):
        self.init_ui()

        self.ui.runStitchingButtonBox.connectApply(self.run_stitching)
        self.ui.displayStitchingButtonBox.connectApply(self.plot_stitching_results)
        self.ui.displayStitchingButtonBox.connectClose(self.main_window.remove_old_plots)
        self.ui.convertOutputButtonBox.connectApply(self.convert_output)
        self.ui.registerButtonBox.connectApply(self.run_registration)
        self.ui.plotRegistrationResultsButtonBox.connectApply(self.plot_registration_results)

        # TODO: ?? connect alignment folder button

        self.ui.advancedAtlasSettingsGroupBox.setVisible(False)
        self.ui.advancedCheckBox.stateChanged.connect(self.swap_tab_advanced)
        self.ui.useAutoToRefLandmarksPushButton.clicked.connect(self.display_auto_to_ref_landmarks_dialog)
        self.ui.useResampledToAutoLandmarksPushButton.clicked.connect(self.display_resampled_to_auto_landmarks_dialog)

    def set_progress_watcher(self, watcher):
        self.preprocessor.set_progress_watcher(watcher)

    def swap_tab_advanced(self):
        checked = self.ui.advancedCheckBox.isChecked()
        self.ui.advancedAtlasSettingsGroupBox.setVisible(checked)

    def run_stitching(self):
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Stitching')
        n_steps = self.preprocessor.n_rigid_steps_to_run + self.preprocessor.n_wobbly_steps_to_run
        self.main_window.make_nested_progress_dialog('Stitching', n_steps=n_steps, sub_maximum=0,
                                                     sub_process_name='Getting layout',
                                                     abort_callback=self.preprocessor.stop_process,
                                                     parent=self.main_window)
        self.main_window.logger.n_lines = 0
        if not self.params.stitching_rigid.skip:
            self.main_window.wrap_in_thread(self.preprocessor.stitch_rigid, force=True)
            self.main_window.print_status_msg('Stitched rigid')
        if not self.params.stitching_wobbly.skip:
            if self.preprocessor.was_stitched_rigid:
                self.main_window.wrap_in_thread(self.preprocessor.stitch_wobbly, force=self.params.stitching_rigid.skip)
                self.main_window.print_status_msg('Stitched wobbly')
            else:
                self.main_window.popup('Could not run wobbly stitching <br>without rigid stitching first')
        self.main_window.progress_dialog.done(1)

    def plot_stitching_results(self):
        self.params.stitching_general.ui_to_cfg()
        if not self.step_exists('stitching', (self.preprocessor.workspace.filename('stitched'))):  # TODO: add arteries option
            return
        dvs = self.preprocessor.plot_stitching_results(parent=self.main_window.centralwidget)
        self.main_window.setup_plots(dvs)

    def convert_output(self):
        fmt = self.params.stitching_general.conversion_fmt
        if not self.step_exists('stitching', (self.preprocessor.workspace.filename('stitched'))):  # TODO: add arteries option
            return
        self.main_window.print_status_msg(f'Converting stitched image to {fmt}')
        self.main_window.make_progress_dialog('Converting files', canceled_callback=self.preprocessor.stop_process)
        self.params.stitching_general.ui_to_cfg()
        self.preprocessor.convert_to_image_format()  # TODO: check if use checkbox state
        self.main_window.progress_dialog.done(1)
        self.main_window.print_status_msg('Conversion finished')

    def setup_atlas(self):  # FIXME: call when value changed in atlas settings
        self.sample_params.ui_to_cfg()  # To make sure we have the slicing up to date
        self.params.registration.ui_to_cfg()
        # self.preprocessor.unpack_atlas(self.params.registration.atlas_info[self.params.registration.atlas_id]['base_name'])
        self.preprocessor.unpack_atlas(self.params.registration.atlas_base_name)
        self.preprocessor.setup_atlases()

    def display_auto_to_ref_landmarks_dialog(self):
        images = [self.preprocessor.workspace.filename('resampled', postfix='autofluorescence'),
                  self.preprocessor.reference_file_path]
        self.__display_landmarks_dialog(images, 'auto_to_reference')

    def display_resampled_to_auto_landmarks_dialog(self):
        images = [self.preprocessor.workspace.filename('resampled', postfix='autofluorescence'),
                  self.preprocessor.workspace.filename('resampled')]
        self.__display_landmarks_dialog(images, 'resampled_to_auto')

    def __display_landmarks_dialog(self, images, direction):
        dvs = plot_3d.plot(images, arange=False, sync=False, lut=self.main_window.preference_editor.params.lut,
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
            dvs[i].scatter_coords = Scatter3D(coords, colors=np.array(landmark_selector.colors), half_slice_thickness=5)
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
        self.__write_registration_landmark_coords('auto_to_reference')

    def write_resampled_to_auto_registration_landmark_coords(self):
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
            with open(f_path, 'w') as landmarks_file:
                landmarks_file.write(f'point\n{len(markers)}\n')
                for marker in markers:
                    x, y, z = marker[i]
                    landmarks_file.write(f'{x} {y} {z}\n')
        self.landmark_selector.dlg.close()
        self.landmark_selector = None

    def run_registration(self):
        self.main_window.print_status_msg('Registering')
        self.main_window.make_nested_progress_dialog('Registering', n_steps=self.preprocessor.n_registration_steps,
                                                     sub_maximum=0, abort_callback=self.preprocessor.stop_process,
                                                     parent=self.main_window)
        self.setup_atlas()
        self.main_window.print_status_msg('Resampling for registering')
        self.main_window.wrap_in_thread(self.preprocessor.resample_for_registration, force=True)
        self.main_window.print_status_msg('Aligning')
        self.main_window.wrap_in_thread(self.preprocessor.align)
        self.main_window.progress_dialog.done(1)
        self.main_window.print_status_msg('Registered')

    def plot_registration_results(self):
        if not self.step_exists('registration',
                                (self.preprocessor.workspace.filename('resampled', postfix='autofluorescence'),
                                 self.preprocessor.aligned_autofluo_path)):
            return
        image_sources = [
            self.preprocessor.workspace.filename('resampled', postfix='autofluorescence'),
            mhd_read(self.preprocessor.aligned_autofluo_path)
        ]
        dvs = plot_3d.plot(image_sources, arange=False, sync=True, lut=self.main_window.preference_editor.params.lut,
                           parent=self.main_window.centralWidget())  # TODO: why () to centralWidget required here only
        link_dataviewers_cursors(dvs, RedCross)
        self.main_window.setup_plots(dvs, ['autofluo', 'aligned'])


class CellCounterTab(PostProcessingTab):
    def __init__(self, main_window, tab_idx=2):
        super().__init__(main_window, 'CellMap', tab_idx, 'cell_map_tab')

        self.preprocessor = None
        self.cell_detector = None

    def set_params(self, sample_params, alignment_params):
        self.params = CellMapParams(self.ui, sample_params, alignment_params, self.main_window.src_folder)

    def setup_workers(self):
        if self.preprocessor.workspace is not None:
            self.params.ui_to_cfg()
            self.cell_detector = CellDetector(self.preprocessor)
        else:
            self.main_window.print_warning_msg('Workspace not initialised')

    def setup_cell_detector(self):
        if self.cell_detector.preprocessor is None and self.preprocessor.workspace is not None:  # preproc initialised
            self.params.ui_to_cfg()
            self.cell_detector.setup(self.preprocessor)

    def setup(self):
        self.init_ui()

        self.ui.toolBox.currentChanged.connect(self.handle_tool_tab_changed)

        self.ui.detectionPreviewTuningButtonBox.connectOpen(self.plot_debug_cropping_interface)
        self.ui.detectionPreviewTuningSampleButtonBox.connectApply(self.create_cell_detection_tuning_sample)
        self.ui.detectionPreviewButtonBox.connectApply(self.run_tuning_cell_detection)
        self.ui.detectionSubsetXRangeMin.valueChanged.connect(self.main_window.ortho_viewer.update_x_min)
        self.ui.detectionSubsetXRangeMax.valueChanged.connect(self.main_window.ortho_viewer.update_x_max)
        self.ui.detectionSubsetYRangeMin.valueChanged.connect(self.main_window.ortho_viewer.update_y_min)
        self.ui.detectionSubsetYRangeMax.valueChanged.connect(self.main_window.ortho_viewer.update_y_max)
        self.ui.detectionSubsetZRangeMin.valueChanged.connect(self.main_window.ortho_viewer.update_z_min)
        self.ui.detectionSubsetZRangeMax.valueChanged.connect(self.main_window.ortho_viewer.update_z_max)

        # for ctrl in (self.cell_map_tab.backgroundCorrectionDiameter, self.cell_map_tab.detectionThreshold):
        #     ctrl.valueChanged.connect(self.reset_detected)  FIXME: find better way

        self.ui.runCellDetectionButtonBox.connectApply(self.detect_cells)
        self.ui.runCellDetectionPlotButtonBox.connectApply(self.plot_detection_results)
        self.ui.previewCellFiltersButtonBox.connectApply(self.preview_cell_filter)
        self.ui.previewCellFiltersButtonBox.connectOk(self.filter_cells)

        self.ui.cellMapVoxelizeButtonBox.connectApply(self.voxelize)

        self.ui.runCellMapButtonBox.connectApply(self.run_cell_map)
        self.ui.runCellMapPlotButtonBox.connectApply(self.plot_cell_map_results)

        self.ui.runCellMapPlot3DScatterButtonBox.connectApply(self.plot_cells_scatter_w_atlas_colors)
        self.ui.runCellMapPlot3DScatterButtonBox.connectOk(self.plot_cells_scatter_w_atlas_colors_raw)  # REFACTOR: find less hacky way than apply & OK

    def voxelize(self):
        self.params.ui_to_cfg()
        if os.path.exists(self.preprocessor.workspace.filename('cells', postfix='filtered')):
            self.cell_detector.voxelize()
        else:
            self.main_window.popup('Could not run voxelization, missing filtered cells table. '
                                   'Please ensure that cell filtering has been run.', base_msg='Missing file')

    def set_progress_watcher(self, watcher):
        if self.cell_detector is not None and self.cell_detector.preprocessor is not None:  # If initialised
            self.cell_detector.set_progress_watcher(watcher)

    def plot_debug_cropping_interface(self):
        self.plot_slicer('detectionSubset', self.ui, self.params)

    def handle_tool_tab_changed(self, tab_idx):
        if tab_idx == 3:
            self.update_cell_number()

    def create_cell_detection_tuning_sample(self):
        slicing = (slice(self.params.crop_x_min, self.params.crop_x_max),
                   slice(self.params.crop_y_min, self.params.crop_y_max),
                   slice(self.params.crop_z_min, self.params.crop_z_max))
        self.cell_detector.create_test_dataset(slicing=slicing)
        self.main_window.print_status_msg('Tuning sample created')  # TODO: progress bar

    def run_tuning_cell_detection(self):
        self.params.ui_to_cfg()
        self.main_window.make_progress_dialog('Cell detection preview')
        self.main_window.wrap_in_thread(self.cell_detector.run_cell_detection, tuning=True)
        self.main_window.progress_dialog.done(1)
        if self.cell_detector.stopped:
            return
        with self.cell_detector.workspace.tmp_debug:
            self.plot_detection_results()

    def detect_cells(self):  # TODO: merge w/ above w/ tuning option
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Starting cell detection')
        self.main_window.make_nested_progress_dialog(title='Detecting cells', n_steps=0,
                                                     abort_callback=self.cell_detector.stop_process)
        self.main_window.wrap_in_thread(self.cell_detector.run_cell_detection, tuning=False)
        if self.cell_detector.stopped:
            return
        if self.params.plot_detected_cells:
            self.cell_detector.plot_cells()  # TODO: integrate into UI
        self.update_cell_number()
        self.main_window.progress_dialog.done(1)
        self.main_window.print_status_msg('Cell detection done')

    def update_cell_number(self):  # FIXME: try except or check that cells and cells filtered exist
        self.ui.nDetectedCellsLabel.setText(
            format_long_nb_to_str(self.cell_detector.get_n_detected_cells()))
        self.ui.nDetectedCellsAfterFilterLabel.setText(
            format_long_nb_to_str(self.cell_detector.get_n_filtered_cells()))

    # def reset_detected(self):
    #     self.cell_detector.detected = False

    def plot_detection_results(self):
        if not self.step_exists('cell detection', [self.preprocessor.workspace.filename('stitched'),
                                                   self.preprocessor.workspace.filename('cells', postfix='bkg'),
                                                   self.preprocessor.workspace.filename('cells', postfix='shape')]):
            return
        dvs = self.cell_detector.preview_cell_detection(parent=self.main_window.centralWidget(),
                                                        arange=False, sync=True)  # TODO: add close
        if len(dvs) == 1:
            self.main_window.print_warning_msg('Preview not run, '
                                               'will only display stitched image for memory space reasons')
        else:
            link_dataviewers_cursors(dvs, RedCross)
        self.main_window.setup_plots(dvs)

    def plot_cell_filter_results(self):
        if not self.step_exists('cell filtering', [self.preprocessor.workspace.filename('stitched'),
                                                   self.workspace.filename('cells', postfix='filtered')]):
            return
        dvs = self.cell_detector.plot_filtered_cells(smarties=True)
        self.main_window.setup_plots(dvs)

    def plot_cells_scatter_w_atlas_colors(self):
        if not self.step_exists('cell count', [self.preprocessor.workspace.filename('stitched'),
                                               self.cell_detector.df_path]):
            return
        dvs = self.cell_detector.plot_cells_3d_scatter_w_atlas_colors(parent=self.main_window)
        self.main_window.setup_plots(dvs)

    def plot_cells_scatter_w_atlas_colors_raw(self):
        if not self.step_exists('cell count', [self.preprocessor.workspace.filename('stitched'),
                                               self.cell_detector.df_path]):
            return
        dvs = self.cell_detector.plot_cells_3d_scatter_w_atlas_colors(raw=True, parent=self.main_window)
        self.main_window.setup_plots(dvs)

    def __filter_cells(self):
        debug_raw_cells_path = self.cell_detector.workspace.filename('cells', postfix='raw')
        if os.path.exists(debug_raw_cells_path):
            self.cell_detector.filter_cells()
            self.cell_detector.voxelize('filtered')
        self.plot_cell_filter_results()

    def preview_cell_filter(self):
        self.params.ui_to_cfg()
        with self.cell_detector.workspace.tmp_debug:
            self.__filter_cells()

    def filter_cells(self):
        self.params.ui_to_cfg()
        self.__filter_cells()
        self.cell_detector.atlas_align()
        self.cell_detector.export_collapsed_stats()

    def run_cell_map(self):
        self.params.ui_to_cfg()
        if not self.cell_detector.detected:
            self.detect_cells()
        self.update_cell_number()
        self.cell_detector.post_process_cells()
        self.update_cell_number()
        if self.params.plot_when_finished:
            self.plot_cell_map_results()
        # WARNING: some plots in .post_process_cells() without UI params

    def plot_cell_map_results(self):
        if not self.step_exists('voxelization', [self.preprocessor.workspace.filename('density', postfix='counts')]):
            return
        dvs = self.cell_detector.plot_voxelized_counts(arange=False)
        self.main_window.setup_plots(dvs)


class VasculatureTab(PostProcessingTab):
    def __init__(self, main_window, tab_idx=3):
        super().__init__(main_window, 'Vasculature', tab_idx, 'vasculature_tab')

        self.params = None
        self.preprocessor = None

        self.binary_vessel_processor = None
        self.vessel_graph_processor = None

    def set_params(self, sample_params, alignment_params):
        self.params = VesselParams(self.ui, sample_params, alignment_params, self.main_window.src_folder)

    def setup_preproc(self, pre_processor):
        self.preprocessor = pre_processor

    def setup_workers(self):
        if self.preprocessor.workspace is not None:
            self.params.ui_to_cfg()
            self.binary_vessel_processor = BinaryVesselProcessor(self.preprocessor)
            self.vessel_graph_processor = VesselGraphProcessor(self.preprocessor)
        else:
            self.main_window.print_warning_msg('Workspace not initialised')

    def setup_vessel_processors(self):
        if self.preprocessor.workspace is not None:  # Inited
            if self.binary_vessel_processor.preprocessor is None:
                self.params.ui_to_cfg()
                self.binary_vessel_processor.setup(self.preprocessor)
            if self.vessel_graph_processor.preprocessor is None:
                self.params.ui_to_cfg()
                self.vessel_graph_processor.setup(self.preprocessor)

    def setup(self):
        self.init_ui()

        self.ui.binarizationButtonBox.connectApply(self.binarize_vessels)
        self.ui.plotBinarizationButtonBox.connectApply(self.plot_binarization_results)
        self.ui.fillVesselsButtonBox.connectApply(self.fill_vessels)
        self.ui.plotFillVesselsButtonBox.connectApply(self.plot_vessel_filling_results)
        self.ui.plotFillVesselsButtonBox.connectClose(self.main_window.remove_old_plots)
        self.ui.buildGraphButtonBox.connectApply(self.build_graph)

        self.ui.graphConstructionSlicerButtonBox.connectOpen(self.plot_graph_construction_chunk_slicer)
        # self.display_cleaned_graph_chunk
        self.ui.graphConstructionPlotGraphButtonBox.connectApply(self.display_reduced_graph_chunk)
        # TODO: clean use other buttons
        self.ui.graphConstructionPlotGraphButtonBox.connectOk(self.display_annotated_graph_chunk)
        self.ui.graphConstructionPlotGraphButtonBox.connectClose(self.main_window.remove_old_plots)

        self.ui.postProcessVesselTypesButtonBox.connectApply(self.post_process_graph)
        self.ui.postProcessVesselTypesSlicerButtonBox.connectOpen(self.plot_graph_type_processing_chunk_slicer)
        self.ui.postProcessVesselTypesPlotButtonBox.connectApply(self.display_annotated_graph_chunk)
        self.ui.postProcessVesselTypesPlotButtonBox.connectClose(self.main_window.remove_old_plots)

    def set_progress_watcher(self, watcher):
        if self.binary_vessel_processor is not None and self.binary_vessel_processor.preprocessor is not None:
            self.binary_vessel_processor.set_progress_watcher(watcher)
        if self.vessel_graph_processor is not None and self.vessel_graph_processor.preprocessor is not None:
            self.vessel_graph_processor.set_progress_watcher(watcher)

    def _get_n_binarize_steps(self):
        n_steps = 1
        n_steps += self.params.binarization_params.post_process_raw
        n_steps += self.params.binarization_params.run_arteries_binarization
        n_steps += self.params.binarization_params.post_process_arteries
        return n_steps

    def binarize_vessels(self):  # FIXME: factorise (wrap in sub_call function)
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Starting vessel binarization')
        self.main_window.make_nested_progress_dialog(title='Binarizing vessels', n_steps=self._get_n_binarize_steps(),
                                                     abort_callback=self.binary_vessel_processor.stop_process)
        self.main_window.wrap_in_thread(self.binary_vessel_processor.binarize)
        self.main_window.progress_dialog.done(1)
        self.main_window.print_status_msg('Vessels binarized')

    def plot_binarization_results(self):
        if not self.step_exists('binarization', [self.preprocessor.workspace.filename('stitched'),
                                                 self.preprocessor.workspace.filename('binary')]):
            return
        dvs = self.binary_vessel_processor.plot_binarization_result(parent=self.main_window)
        link_dataviewers_cursors(dvs, RedCross)
        self.main_window.setup_plots(dvs, ['stitched', 'binary'])

    def fill_vessels(self):
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Starting vessel filling')
        bin_params = self.params.binarization_params
        n_steps = bin_params.fill_main_channel + bin_params.fill_secondary_channel
        self.main_window.make_nested_progress_dialog(title='Filling vessels', n_steps=n_steps,
                                                     abort_callback=self.binary_vessel_processor.stop_process)
        self.main_window.wrap_in_thread(self.binary_vessel_processor.fill_vessels)
        self.main_window.wrap_in_thread(self.binary_vessel_processor.combine_binary)  # REFACTOR: not great location
        self.main_window.progress_dialog.done(1)
        self.main_window.print_status_msg('Vessel filling done')

    def plot_vessel_filling_results(self):  # TODO: add step_exists check
        dvs = self.binary_vessel_processor.plot_vessel_filling_results()
        link_dataviewers_cursors(dvs, RedCross)
        self.main_window.setup_plots(dvs)

    def build_graph(self):
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Building vessel graph')
        self.main_window.make_nested_progress_dialog(title='Building vessel graph', n_steps=4,
                                                     abort_callback=self.vessel_graph_processor.stop_process)
        self.main_window.wrap_in_thread(self.vessel_graph_processor.pre_process)
        self.main_window.progress_dialog.done(1)
        self.main_window.print_status_msg('Building vessel graph done')

    def plot_graph_construction_chunk_slicer(self):
        self.params.graph_params._crop_values_from_cfg()  # Fix for lack of binding between 2 sets of range interfaces
        self.plot_slicer('graphConstructionSlicer', self.ui, self.params.graph_params)

    def plot_graph_type_processing_chunk_slicer(self):
        self.params.graph_params._crop_values_from_cfg()  # Fix for lack of binding between 2 sets of range interfaces
        self.plot_slicer('vesselProcessingSlicer', self.ui, self.params.graph_params)

    def __get_tube_map_slicing(self):
        self.params.graph_params._crop_values_from_cfg()  # Fix for lack of binding between 2 sets of range interfaces
        slicing = (slice(self.params.graph_params.crop_x_min, self.params.graph_params.crop_x_max),
                   slice(self.params.graph_params.crop_y_min, self.params.graph_params.crop_y_max),
                   slice(self.params.graph_params.crop_z_min, self.params.graph_params.crop_z_max))
        return slicing

    def display_cleaned_graph_chunk(self):
        slicing = self.__get_tube_map_slicing()
        dvs = self.vessel_graph_processor.visualize_graph_annotations(slicing, plot_type='mesh', graph_step='cleaned')
        self.main_window.setup_plots(dvs)

    def display_reduced_graph_chunk(self):
        slicing = self.__get_tube_map_slicing()
        dvs = self.vessel_graph_processor.visualize_graph_annotations(slicing, plot_type='mesh', graph_step='reduced')
        self.main_window.setup_plots(dvs)

    def display_annotated_graph_chunk(self):
        slicing = self.__get_tube_map_slicing()
        dvs = self.vessel_graph_processor.visualize_graph_annotations(slicing, plot_type='mesh', graph_step='annotated')
        self.main_window.setup_plots(dvs)

    def post_process_graph(self):
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Post processing vasculature graph')
        self.main_window.make_nested_progress_dialog(title='Post processing graph', n_steps=8,
                                                     abort_callback=self.vessel_graph_processor.stop_process)
        self.main_window.wrap_in_thread(self.vessel_graph_processor.post_process)
        self.main_window.progress_dialog.done(1)
        self.main_window.print_status_msg('Vasculature graph post-processing DONE')

################################################################################################


class BatchTab(GenericTab):
    def __init__(self, main_window, tab_idx=4):  # FIXME: offload computations to BatchProcessor object
        super().__init__(main_window, 'Batch', tab_idx, 'batch_tab')

        self.params = None

        self.processing_type = 'batch'

    @property
    def initialised(self):
        return self.params is not None

    def set_params(self):
        self.params = BatchParams(self.ui, '', preferences=self.main_window.preference_editor.params)
        # self.params = BatchParams(self.ui, self.main_window.src_folder)

    def setup(self):
        self.init_ui()

        self.ui.folderPickerHelperPushButton.clicked.connect(self.create_wizard)
        self.ui.runPValsButtonBox.connectApply(self.run_p_vals)
        self.ui.batchRunButtonBox.connectApply(self.run_batch_process)
        self.ui.batchStatsButtonBox.connectApply(self.make_group_stats_tables)

        self.ui.batchToolBox.setCurrentIndex(0)

    def create_wizard(self):
        return SamplePickerDialog('', params=self.params)

    def make_group_stats_tables(self):
        self.main_window.print_status_msg('Computing stats table')
        dvs = []
        groups = self.params.groups
        self.main_window.make_nested_progress_dialog(title='Group stats',
                                                     n_steps=len(self.params.selected_comparisons))  # TODO: see abort callback
        for pair in self.params.selected_comparisons:
            gp1_name, gp2_name = pair
            df = self.main_window.wrap_in_thread(make_summary, self.params.results_folder,
                                                 gp1_name, gp2_name, groups[gp1_name], groups[gp2_name],
                                                 output_path=None, save=True)
            self.main_window.progress_watcher.increment_main_progress()
            dvs.append(DataFrameWidget(df).table)
        self.main_window.setup_plots(dvs)
        self.main_window.print_status_msg('Idle, waiting for input')

    def run_p_vals(self):
        self.params.ui_to_cfg()
        groups = self.params.groups
        p_vals_imgs = []
        self.main_window.print_status_msg('Computing p_val maps')
        self.main_window.make_nested_progress_dialog(title='P value maps',
                                                     n_steps=len(self.params.selected_comparisons))  # TODO: see abort callback
        for pair in self.params.selected_comparisons:
            gp1_name, gp2_name = pair
            if not density_files_are_comparable(self.params.results_folder, groups[gp1_name], groups[gp2_name]):
                self.main_window.popup('Could not compare files, sizes differ', base_msg='Cannot compare files')
            res = self.main_window.wrap_in_thread(compare_groups, self.params.results_folder,
                                                  gp1_name, gp2_name, groups[gp1_name], groups[gp2_name])
            self.main_window.progress_watcher.increment_main_progress()
            p_vals_imgs.append(res)

        self.main_window.progress_dialog.done(1)
        # self.main_window.print_status_msg('P vals computed')
        self.main_window.print_status_msg('Idle, waiting for input')

        return  # FIXME: add support for RGB images in DataViewer
        # if len(self.params.selected_comparisons) == 1:
        #     gp1_img = clearmap_io.read(os.path.join(self.params.results_folder, f'condensed_{gp1_name}.tif'))
        #     gp2_img = clearmap_io.read(os.path.join(self.params.results_folder, f'condensed_{gp2_name}.tif'))
        #     dvs = plot_3d.plot([gp1_img, gp2_img, p_vals_imgs[0]])
        # else:
        #     dvs = plot_3d.plot(p_vals_imgs, arange=False, sync=True,
        #                        lut=self.main_window.preference_editor.params.lut,
        #                        parent=self.main_window.centralWidget())
        #     link_dataviewers_cursors(dvs, RedCross)
        # self.main_window.setup_plots(dvs)

    def run_batch_process(self):
        self.params.ui_to_cfg()
        paths = [p for ps in self.params.get_all_paths() for p in ps]  # flatten list
        self.main_window.make_progress_dialog('Analysing samples', maximum=0)  # TODO: see abort callback
        # TODO: pass progress watcher to increment from within
        self.main_window.wrap_in_thread(process_folders, paths,
                                        self.params.align, self.params.count_cells, self.params.run_vaculature)
        self.main_window.progress_dialog.done(1)
