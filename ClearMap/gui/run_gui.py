import os
import sys

from multiprocessing.pool import ThreadPool
from shutil import copyfile
import traceback
import types

import numpy as np

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QSpinBox, QDoubleSpinBox, QFrame, \
    QDialogButtonBox, QComboBox, QLineEdit, QStyle, QWidget, QMessageBox, QToolBox

# ########################################### SPLASH SCREEN ###########################################################
from ClearMap.gui.dialogs import make_splash, update_pbar

# To show splash before slow imports
app = QApplication([])

splash, progress_bar = make_splash()
splash.show()
update_pbar(app, progress_bar, 10)

# ############################################  SLOW IMPORTS #########################################################

import pygments
from pygments.lexers.python import PythonTracebackLexer  # noqa
from pygments.formatters.html import HtmlFormatter

import qdarkstyle

import pyqtgraph as pg
pg.setConfigOption('background', '#1A1D1E')

from skimage import transform as sk_transform  # Slowish

from ClearMap.IO import TIF
from ClearMap.IO.MHD import mhd_read
from ClearMap.Settings import resources_path
from ClearMap.gui.gui_logging import Printer
from ClearMap.config.config_loader import ConfigLoader
from ClearMap.gui.params import SampleParameters, ConfigNotFoundError, CellMapParams, \
    PreferencesParams, PreprocessingParams, ParamsOrientationError, VesselParams
from ClearMap.gui.widget_monkeypatch_callbacks import get_value, set_value, controls_enabled, get_check_box, \
    enable_controls, disable_controls, set_text, get_text, connect_apply, connect_close, connect_save, connect_open, \
    connect_ok, connect_cancel, connect_value_changed, connect_text_changed
from ClearMap.gui.pyuic_utils import loadUiType
from ClearMap.gui.dialogs import get_directory_dlg, warning_popup, make_progress_dialog, make_nested_progress_dialog

from ClearMap.gui.gui_utils import QDARKSTYLE_BACKGROUND, DARK_BACKGROUND, np_to_qpixmap, \
    html_to_ansi, html_to_plain_text, compute_grid, surface_project, format_long_nb_to_str, \
    link_dataviewers_cursors  # needs plot_3d

from ClearMap.gui.widgets import OrthoViewer, PbarWatcher  # needs plot_3d
from ClearMap.Visualization import Plot3d as plot_3d
update_pbar(app, progress_bar, 20)
from ClearMap.Scripts.sample_preparation import PreProcessor
update_pbar(app, progress_bar, 40)
from ClearMap.Scripts.cell_map import CellDetector
update_pbar(app, progress_bar, 60)
from ClearMap.Scripts.tube_map import BinaryVesselProcessor, VesselGraphProcessor
update_pbar(app, progress_bar, 80)

# TODO
"""
Handle reset detected correctly
Fix qrc resources (ui files should not be coded by path)
Test and check that works with secondary channel
Delete intermediate files
Ensure all machine config params are in the preferences UI

Previews:
    - Add rigid alignment : plane in middle of stack from each column + stitch with different colours

    
Analysis:
    
LATER:
Auto modes:
    - Run all
    - Batch run
"""

Ui_ClearMapGui, _ = loadUiType('ClearMap/gui/mainwindow.ui', patch_parent_class=False)


class ClearMapGuiBase(QMainWindow, Ui_ClearMapGui):
    def __init__(self):
        super().__init__()
        self.graph_names = {}
        self.__reload_icon = self.style().standardIcon(QStyle.SP_BrowserReload)
        self.logger = None

    def find_child_by_name(self, child_name, child_type, parent=None):
        if parent is None:
            parent = self
        for child in parent.findChildren(child_type):
            if child.objectName() == child_name:
                return child

    def print_error_msg(self, msg):
        self.statusbar.setStyleSheet("color: red")
        self.statusbar.showMessage(msg)

    def print_warning_msg(self, msg):
        self.statusbar.setStyleSheet("color: yellow")
        self.statusbar.showMessage(msg)

    def print_status_msg(self, msg):
        self.statusbar.setStyleSheet("color: white")
        self.statusbar.showMessage(msg)

    def fix_btn_boxes_text(self):
        for btn_box in self.findChildren(QDialogButtonBox):
            if btn_box.property('applyText'):
                btn_box.button(QDialogButtonBox.Apply).setText(btn_box.property('applyText'))

    def fix_styles(self):
        self.sample_tab.sampleIdButtonBox.button(QDialogButtonBox.Apply).setIcon(self.__reload_icon)

        self.fix_btn_boxes_text()

        self.fix_btns_stylesheet()
        self.fix_widgets_backgrounds()

    def fix_btns_stylesheet(self):
        for btn in self.findChildren(QPushButton):
            btn.setStyleSheet('QPushButton {'
                              'background-color: #455364; '
                              'color: #E0E1E3;'
                              'border-radius: 4px;'
                              'padding: 2px;'
                              # 'outline: none;'
                              '}'
                              'QPushButton:pressed {'
                              'background-color: #259AE9; '
                              # 'background-color: #60798B; '
                              # 'border: 2px #259AE9;'
                              '}'
                              )

    def fix_widgets_backgrounds(self):
        for widget_type in (QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit):
            for btn in self.findChildren(widget_type):
                btn.setStyleSheet('background-color: {}; '.format(DARK_BACKGROUND))

    def popup(self, msg, base_msg='Missing configuration file'):
        self.print_warning_msg(html_to_plain_text(msg))
        return warning_popup(base_msg, msg)

    def file_exists(self, f_path):
        if os.path.exists(f_path):
            return True
        else:
            msg = 'File "{}" not found'.format(f_path)
            self.print_error_msg(msg)
            return False

    def setup_plots(self, dvs, graph_names=None):
        self._remove_old_plots()

        n_rows, n_cols = compute_grid(len(dvs))
        grid_size = (n_rows * n_cols)
        n_spacers = grid_size - len(dvs)
        for i in range(n_spacers):
            if graph_names:
                graph_names.append('spacer_{}'.format(i))
            dvs.append(QWidget(parent=self))
        for i, dv in enumerate(dvs):
            graph_name = 'graph_{}'.format(i)
            setattr(self, graph_name, dv)
            dv.setObjectName(graph_name)
            row = i // n_cols
            col = i % n_cols
            self.graphLayout.addWidget(dv, row, col, 1, 1)
            self.__resize_graph(dv, n_cols, n_rows)
            if graph_names:
                self.graph_names[graph_names[i]] = graph_name

    def __resize_graph(self, dv, n_cols, n_rows, margin=20):
        size = round((self.graphDock.width() - margin) / n_cols), round((self.graphDock.height() - margin) / n_rows)
        dv.resize(*size)
        dv.setMinimumSize(*size)  # required to avoid wobbly dv
        # dv.setMaximumSize(*size)

    def _remove_old_plots(self):
        for i in range(self.graphLayout.count(), -1, -1):
            graph = self.graphLayout.takeAt(i)
            if graph is not None:
                widg = graph.widget()
                widg.setParent(None)
                widg.deleteLater()
                delattr(self, widg.objectName())
        self.graph_names = {}

    def setupIcons(self):
        self.__reload_icon = self.style().standardIcon(QStyle.SP_BrowserReload)

    def patch_compound_boxes(self):
        for bx in self.findChildren(QFrame):
            bx_name = bx.objectName().lower()
            if bx_name.startswith('triplet') or bx_name.endswith('let') or bx_name.endswith('optionallineedit'):
                bx.controlsEnabled = types.MethodType(controls_enabled, bx)
                bx.getCheckBox = types.MethodType(get_check_box, bx)
                bx.enableControls = types.MethodType(enable_controls, bx)
                bx.disableControls = types.MethodType(disable_controls, bx)
                if bx_name.startswith('triplet') or bx_name.endswith('let'):  # singlet double triplet
                    bx.getValue = types.MethodType(get_value, bx)
                    bx.setValue = types.MethodType(set_value, bx)
                    bx.valueChangedConnect = types.MethodType(connect_value_changed, bx)
                elif bx_name.endswith('optionallineedit'):
                    bx.setText = types.MethodType(set_text, bx)
                    bx.text = types.MethodType(get_text, bx)
                    bx.textChangedConnect = types.MethodType(connect_text_changed, bx)
                else:
                    print('Skipping box "{}", type not recognised'.format(bx_name))

    def patch_button_boxes(self, parent=None):
        if parent is None:
            parent = self
        for bx in parent.findChildren(QDialogButtonBox):
            bx.connectApply = types.MethodType(connect_apply, bx)
            bx.connectClose = types.MethodType(connect_close, bx)
            bx.connectSave = types.MethodType(connect_save, bx)
            bx.connectOpen = types.MethodType(connect_open, bx)
            bx.connectOk = types.MethodType(connect_ok, bx)
            bx.connectCancel = types.MethodType(connect_cancel, bx)

    def patch_tool_boxes(self):
        for tb in self.findChildren(QToolBox):
            tb.setCurrentIndex(0)

    def monkey_patch(self):
        self.patch_compound_boxes()
        self.patch_button_boxes()
        self.patch_tool_boxes()
        self.fix_styles()

    @staticmethod
    def create_missing_file_msg(f_type, f_path, default_f_path):
        base_msg = 'No {} file found at:<br>  <nobr><em>"{}"</em></nobr>.'.format(f_type, f_path)
        msg = '{} <br><br>Do you want to load a default one from:<br>  <nobr><em>"{}"</em></nobr>' \
            .format(base_msg, default_f_path)
        return base_msg, msg

    def make_progress_dialog(self, msg, maximum=100, canceled_callback=None):
        dialog = make_progress_dialog(msg, maximum, canceled_callback, self)
        self.progress_dialog = dialog
        self.progress_watcher.progress_changed.connect(self.progress_dialog.setValue)
        self.preprocessor.set_progress_watcher(self.progress_watcher)
        # self.cell_detector.set_progress_watcher(self.progress_watcher)

    def make_nested_progress_dialog(self, title='Processing', n_steps=1, sub_maximum=100,
                                    sub_process_name='', abort_callback=None, parent=None):
        if n_steps:
            n_steps += 1  # To avoid range shrinking because starting from 1 not 0
        dialog = make_nested_progress_dialog(title=title, overall_maximum=n_steps, sub_maximum=sub_maximum,
                                             sub_process_name=sub_process_name, abort_callback=abort_callback,
                                             parent=parent)
        self.progress_dialog = dialog
        self.progress_watcher.progress_changed.connect(self.progress_dialog.subProgressBar.setValue)
        self.progress_watcher.main_progress_changed.connect(self.progress_dialog.mainProgressBar.setValue)
        self.progress_watcher.main_progress_changed.connect(self.progress_watcher.reset_log_length)
        self.progress_watcher.max_changed.connect(self.progress_dialog.subProgressBar.setMaximum)
        self.progress_watcher.main_max_changed.connect(self.progress_dialog.mainProgressBar.setMaximum)
        self.progress_watcher.progress_name_changed.connect(self.progress_dialog.subProgressLabel.setText)

        self.progress_watcher.main_max_progress = n_steps

        self.preprocessor.set_progress_watcher(self.progress_watcher)  # FIXME: specific of subclass
        if self.cell_detector is not None and self.cell_detector.preprocessor is not None:  # If initialised
            self.cell_detector.set_progress_watcher(self.progress_watcher)
        if self.binary_vessel_processor is not None and self.binary_vessel_processor.preprocessor is not None:
            self.binary_vessel_processor.set_progress_watcher(self.progress_watcher)
        if self.vessel_graph_processor is not None and self.vessel_graph_processor.preprocessor is not None:
            self.vessel_graph_processor.set_progress_watcher(self.progress_watcher)


class ClearMapGui(ClearMapGuiBase):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor
        self.config_loader = ConfigLoader('')
        self.ortho_viewer = OrthoViewer()
        self.machine_cfg_path = None
        self.sample_cfg_path = None
        self.processing_cfg_path = None
        self.sample_params = None
        self.processing_params = None
        self.cell_map_params = None
        self.vessel_params = None

        self.cell_detector = None
        self.binary_vessel_processor = None
        self.vessel_graph_processor = None

        self.mini_brain_scaling, self.mini_brain = self.setup_mini_brain()

        self.setWindowIcon(QtGui.QIcon('ClearMap/gui/icons/logo_cyber.png'))  # REFACTOR: use qrc

        self.setupUi(self)
        self.amendUi()

        self.actionPreferences.triggered.connect(self.preferences_editor.exec)  # TODO: check if move to setup_preferences

        self.progress_watcher = PbarWatcher()
        self.app = QApplication.instance()

    def patch_stdout(self):
        sys.stdout = self.logger
        sys.stderr = self.error_logger

    def setup_mini_brain(self):
        atlas_path = os.path.join(resources_path, 'Atlas',
                                  'ABA_25um_annotation.tif')  # FIXME: function of chosen atlas
        arr = TIF.Source(atlas_path).array
        mini_brain_scaling = (5, 5, 5)  # TODO: prefs
        return mini_brain_scaling, sk_transform.downscale_local_mean(arr, mini_brain_scaling)

    def swap_sample_tab_advanced(self):  # TODO: implement
        checked = self.sample_tab.advancedCheckBox.isChecked()

    def swap_preprocessing_tab_advanced(self):
        checked = self.preprocessing_tab.advancedCheckBox.isChecked()
        self.preprocessing_tab.atlasSettingsPage.setVisible(checked)

    def amendUi(self):
        self.logger = Printer()
        self.logger.text_updated.connect(self.textBrowser.append)
        self.error_logger = Printer(color='red', logger_type='error')
        self.error_logger.text_updated.connect(self.textBrowser.append)

        self.setupIcons()
        self.setup_tabs()
        self.setup_preferences_editor()

        self.monkey_patch()
        self.logoLabel.setPixmap(QtGui.QPixmap('icons/logo_cyber.png'))

        self.graphLayout.removeWidget(self.frame)

        self.print_status_msg('Idle, waiting for input')

    def setup_tabs(self):
        self.setup_sample_tab()
        self.setup_preprocessing_tab()
        self.setup_cell_map_tab()
        self.setup_vasculature_tab()
        self.tabWidget.tabBarClicked.connect(self.handle_tab_click)
        self.tabWidget.setCurrentIndex(0)

    def setup_sample_tab(self):
        cls, _ = loadUiType('ClearMap/gui/sample_tab.ui', patch_parent_class='QTabWidget')
        self.sample_tab = cls()
        self.sample_tab.setupUi()
        self.patch_button_boxes(self.sample_tab)
        self.tabWidget.removeTab(0)
        self.tabWidget.insertTab(0, self.sample_tab, 'Sample')

        self.sample_tab.srcFolderBtn.clicked.connect(self.set_src_folder)
        self.sample_tab.sampleIdButtonBox.connectApply(self.parse_cfg)

        self.sample_tab.plotMiniBrainPushButton.clicked.connect(self.plot_mini_brain)

        self.sample_tab.applyBox.connectApply(self.setup_preprocessor)
        self.sample_tab.applyBox.connectSave(self.save_sample_cfg)

        self.sample_tab.advancedCheckBox.stateChanged.connect(self.swap_sample_tab_advanced)

    def setup_preprocessing_tab(self):
        cls, _ = loadUiType('ClearMap/gui/preprocessing_tab.ui', patch_parent_class='QTabWidget')
        self.preprocessing_tab = cls()
        self.preprocessing_tab.setupUi()
        self.patch_button_boxes(self.preprocessing_tab)
        self.tabWidget.removeTab(1)
        self.tabWidget.insertTab(1, self.preprocessing_tab, 'Alignments')

        self.preprocessing_tab.runStitchingButtonBox.connectApply(self.run_stitching)
        self.preprocessing_tab.displayStitchingButtonBox.connectApply(self.plot_stitching_results)
        self.preprocessing_tab.displayStitchingButtonBox.connectClose(self._remove_old_plots)
        self.preprocessing_tab.convertOutputButtonBox.connectApply(self.convert_output)
        self.preprocessing_tab.registerButtonBox.connectApply(self.run_registration)
        self.preprocessing_tab.plotRegistrationResultsButtonBox.connectApply(self.plot_registration_results)

        # FIXME: connect alignemnt folder button

        self.preprocessing_tab.atlasSettingsPage.setVisible(False)
        self.preprocessing_tab.advancedCheckBox.stateChanged.connect(self.swap_preprocessing_tab_advanced)

    def setup_cell_map_tab(self):
        cls, _ = loadUiType('ClearMap/gui/cell_map_tab.ui', patch_parent_class='QTabWidget')
        self.cell_map_tab = cls()
        self.cell_map_tab.setupUi()
        self.patch_button_boxes(self.cell_map_tab)
        self.tabWidget.removeTab(2)
        self.tabWidget.insertTab(2, self.cell_map_tab, 'CellMap')

        self.cell_map_tab.detectionPreviewTuningButtonBox.connectOpen(self.plot_debug_cropping_interface)
        self.cell_map_tab.detectionPreviewTuningSampleButtonBox.connectApply(self.create_cell_detection_tuning_sample)
        self.cell_map_tab.detectionPreviewButtonBox.connectApply(self.run_tuning_cell_detection)
        self.cell_map_tab.detectionSubsetXRangeMin.valueChanged.connect(self.ortho_viewer.update_x_min)
        self.cell_map_tab.detectionSubsetXRangeMax.valueChanged.connect(self.ortho_viewer.update_x_max)
        self.cell_map_tab.detectionSubsetYRangeMin.valueChanged.connect(self.ortho_viewer.update_y_min)
        self.cell_map_tab.detectionSubsetYRangeMax.valueChanged.connect(self.ortho_viewer.update_y_max)
        self.cell_map_tab.detectionSubsetZRangeMin.valueChanged.connect(self.ortho_viewer.update_z_min)
        self.cell_map_tab.detectionSubsetZRangeMax.valueChanged.connect(self.ortho_viewer.update_z_max)

        # for ctrl in (self.cell_map_tab.backgroundCorrectionDiameter, self.cell_map_tab.detectionThreshold):
        #     ctrl.valueChanged.connect(self.reset_detected)  FIXME: find better way

        self.cell_map_tab.runCellDetectionButtonBox.connectApply(self.detect_cells)
        self.cell_map_tab.runCellDetectionPlotButtonBox.connectApply(self.plot_detection_results)
        self.cell_map_tab.previewCellFiltersButtonBox.connectApply(self.preview_cell_filter)

        self.cell_map_tab.runCellMapButtonBox.connectApply(self.run_cell_map)
        self.cell_map_tab.runCellMapPlotButtonBox.connectApply(self.plot_cell_map_results)

    def setup_vasculature_tab(self):
        cls, _ = loadUiType('ClearMap/gui/vasculature_tab.ui', patch_parent_class='QTabWidget')
        self.vasculature_tab = cls()
        self.vasculature_tab.setupUi()
        self.patch_button_boxes(self.vasculature_tab)
        self.tabWidget.removeTab(3)
        self.tabWidget.insertTab(3, self.vasculature_tab, 'Vasculature')

        self.vasculature_tab.binarizationButtonBox.connectApply(self.binarize_vessels)
        self.vasculature_tab.fillVesselsButtonBox.connectApply(self.fill_vessels)
        # self.vasculature_tab.fillVesselsButtonBox.connectClose()  # FIXME:
        self.vasculature_tab.buildGraphButtonBox.connectApply(self.build_graph)
        self.vasculature_tab.postProcessVesselTypesButtonBox.connectApply(self.post_process_graph)

    def setup_preferences_editor(self):
        cls, _ = loadUiType('ClearMap/gui/preferences_editor.ui', patch_parent_class='QDialog')
        self.preferences_editor = cls()
        self.preferences_editor.setWindowTitle('Preferences')
        self.preferences_editor.setupUi()
        self.patch_button_boxes(self.preferences_editor)

        self.setup_preferences()

        self.preferences_editor.buttonBox.connectApply(self.preferences.ui_to_cfg)
        self.preferences_editor.buttonBox.connectCancel(self.preferences_editor.close)
        self.preferences_editor.buttonBox.connectOk(self.apply_prefs_and_close)

    def handle_tab_click(self, tab_index):
        if tab_index in (1, 2) and self.preprocessor.workspace is None:
            self.popup('WARNING', 'Workspace not initialised, '
                                  'cannot proceed to alignment')
            self.tabWidget.setCurrentIndex(0)
        if tab_index == 2:
            if not os.path.exists(self.preprocessor.aligned_autofluo_path):
                ok = self.popup('WARNING', 'Alignment not performed, please run first') == QMessageBox.Ok
                self.tabWidget.setCurrentIndex(1)  # WARNING: does not work
            else:
                self.setup_cell_detector()
        if tab_index == 3:
            if not os.path.exists(self.preprocessor.aligned_autofluo_path):
                ok = self.popup('WARNING', 'Alignment not performed, please run first') == QMessageBox.Ok
                self.tabWidget.setCurrentIndex(1)  # WARNING: does not work
            else:
                self.setup_vessel_processors()

    def apply_prefs_and_close(self):
        self.preferences.ui_to_cfg()
        self.preferences_editor.close()

    def save_sample_cfg(self):  # REFACTOR: use this instead of direct calls to ui_to_cfg
        self.sample_params.ui_to_cfg()
        self.print_status_msg('Sample config saved')

    def setup_preprocessor(self):
        self.save_sample_cfg()
        self.preprocessor.setup((self.preferences.config, self.sample_params.config, self.processing_params.config))

    def setup_cell_detector(self):
        if self.cell_detector.preprocessor is None and self.preprocessor.workspace is not None:  # preproc initialised
            self.processing_params.ui_to_cfg()
            self.cell_detector.setup(self.preprocessor)

    def setup_vessel_processors(self):
        if self.preprocessor.workspace is not None:  # Initied
            if self.binary_vessel_processor.preprocessor is None:
                self.processing_params.ui_to_cfg()
                self.binary_vessel_processor.setup(self.preprocessor)
            if self.vessel_graph_processor.preprocessor is None:
                self.processing_params.ui_to_cfg()
                self.vessel_graph_processor.setup(self.preprocessor)

    def __get_cfg_path(self, cfg_name):
        cfg_path = self.config_loader.get_cfg_path(cfg_name, must_exist=False)
        was_copied = False
        if not self.file_exists(cfg_path):
            default_cfg_file_path = self.config_loader.get_default_path(cfg_name)
            base_msg, msg = self.create_missing_file_msg(cfg_name.title().replace('_', ''),
                                                         cfg_path, default_cfg_file_path)
            ret = self.popup(msg)
            if ret == QMessageBox.Ok:
                copyfile(default_cfg_file_path, cfg_path)
                was_copied = True
            else:
                raise FileNotFoundError(html_to_ansi(base_msg))
        return was_copied, cfg_path

    def get_cfg_paths(self):  # REFACTOR: move to config module
        if not self.src_folder:
            msg = 'Missing source folder, please define first'
            self.print_error_msg(msg)
            raise FileNotFoundError(msg)
        was_copied, sample_cfg_path = self.__get_cfg_path('sample')
        if was_copied:
            self.sample_params.fix_sample_cfg_file(sample_cfg_path)
        self.sample_cfg_path = sample_cfg_path
        was_copied, self.processing_cfg_path = self.__get_cfg_path('processing')
        if was_copied:
            self.processing_params.fix_pipeline_name(self.processing_cfg_path)

    def setup_preferences(self):
        self.preferences = PreferencesParams(self.preferences_editor, self.src_folder)
        machine_cfg_path = self.config_loader.get_default_path('machine')
        if self.file_exists(machine_cfg_path):
            self.machine_cfg_path = machine_cfg_path
            self.preferences.get_config(self.machine_cfg_path)
            self.preferences.cfg_to_ui()
        else:
            msg = 'Missing machine config file. Please ensure a machine_params.cfg file ' \
                  'is available at {}. This should be done at installation'.format(machine_cfg_path)
            self.print_error_msg(msg)
            raise FileNotFoundError(msg)

    def parse_cfg(self):
        self.print_status_msg('Parsing configuration')
        try:
            self.get_cfg_paths()
        except FileNotFoundError as err:
            self.error_logger.write(str(err))  # TODO: see if let escalate
            return

        error = False
        try:
            self.sample_params.get_config(self.sample_cfg_path)
        except ConfigNotFoundError:
            self.print_error_msg('Loading sample config file failed')
            error = True
        try:
            self.processing_params.get_config(self.processing_cfg_path)
        except ConfigNotFoundError:
            self.print_error_msg('Loading preprocessing config file failed')
            error = True
        if self.processing_params.pipeline_is_cell_map:
            self.cell_map_params = CellMapParams(self.cell_map_tab, self.sample_params, self.processing_params)
            try:
                self.cell_map_params.get_config(self.__get_cfg_path('cell_map')[1])
            except ConfigNotFoundError:
                self.print_error_msg('Loading Cell Map config file failed')
                error = True
            self.cell_detector = CellDetector()
        if self.processing_params.pipeline_is_tube_map:  # WARNING: should not be exclusive
            self.vessel_params = VesselParams(self.vasculature_tab)
            try:
                self.vessel_params.get_config(self.__get_cfg_path('tube_map')[1])
            except ConfigNotFoundError:
                self.print_error_msg('Loading Tube Map config file failed')
                error = True
            self.binary_vessel_processor = BinaryVesselProcessor()
            self.vessel_graph_processor = VesselGraphProcessor()
        if not error:
            self.print_status_msg('Config loaded')
            try:
                self.sample_params.cfg_to_ui()
            except ParamsOrientationError as err:
                self.popup(str(err), 'Invalid orientation. Defaulting')
                self.sample_params.orientation = (1, 2, 3)
            self.processing_params.cfg_to_ui()
            if self.cell_map_params is not None:
                self.cell_map_params.cfg_to_ui()
            if self.vessel_params is not None:
                self.vessel_params.cfg_to_ui()

    def set_src_folder(self):
        self.src_folder = get_directory_dlg(self.preferences.start_folder)
        self.config_loader.src_dir = self.src_folder
        self.sample_params = SampleParameters(self.sample_tab, self.src_folder)
        self.processing_params = PreprocessingParams(self.preprocessing_tab)

    @property
    def src_folder(self):
        return self.sample_tab.srcFolderTxt.text()

    @src_folder.setter
    def src_folder(self, src_folder):
        self.logger.set_file(os.path.join(src_folder, 'info.log'))
        self.progress_watcher.log_path = self.logger.file.name
        self.error_logger.set_file(os.path.join(src_folder, 'errors.html'))
        self.sample_tab.srcFolderTxt.setText(src_folder)

    def get_graphs(self):
        return sorted([getattr(self, attr) for attr in dir(self) if attr.startswith('graph_')])

    def resize_graphs(self):
        n_rows, n_cols = compute_grid(len(self.get_graphs()))  # WARNING: take care of placeholders
        for i in range(self.graphLayout.count(), -1, -1):  # Necessary to count backwards to get all graphs
            graph = self.graphLayout.itemAt(i)
            if graph is not None:
                widg = graph.widget()
                self.__resize_graph(widg, n_cols, n_rows)

    def plot_mini_brain(self):
        img = self.__transform_mini_brain()
        mask, proj = surface_project(img)
        img = np_to_qpixmap(proj, mask)
        self.sample_tab.miniBrainLabel.setPixmap(img)

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

        orientation = self.sample_params.orientation
        x_scale, y_scale, z_scale = self.mini_brain_scaling
        img = self.mini_brain.copy()
        axes_to_flip = [abs(axis) - 1 for axis in orientation if axis < 0]
        if axes_to_flip:
            img = np.flip(img, axes_to_flip)
        img = img.transpose([abs(axis) - 1 for axis in orientation])
        x_min, x_max = range_or_default(self.sample_params.slice_x, x_scale)
        y_min, y_max = range_or_default(self.sample_params.slice_y, y_scale)
        z_min, z_max = range_or_default(self.sample_params.slice_z, z_scale)
        img = img[x_min:x_max, y_min:y_max:, z_min:z_max]
        return img

    # ###################################### PREPROCESSING ####################################

    def wrap_in_thread(self, func, *args, **kwargs):
        pool = ThreadPool(processes=1)
        result = pool.apply_async(func, args, kwargs)
        while not result.ready():
            result.wait(0.25)
            self.progress_watcher.set_progress(self.progress_watcher.count_dones())
            self.app.processEvents()
        return result.get()

    def run_stitching(self):
        self.processing_params.ui_to_cfg()
        self.print_status_msg('Stitching')
        n_steps = self.preprocessor.n_rigid_steps_to_run + self.preprocessor.n_wobbly_steps_to_run
        self.make_nested_progress_dialog('Stitching', n_steps=n_steps, sub_maximum=0, sub_process_name='Getting layout',
                                         abort_callback=self.preprocessor.stop_process, parent=self)
        self.logger.n_lines = 0
        if not self.processing_params.stitching_rigid.skip:
            self.wrap_in_thread(self.preprocessor._stitch_rigid, force=True)
            self.print_status_msg('Stitched rigid')
        if not self.processing_params.stitching_wobbly.skip:
            if self.preprocessor.was_stitched_rigid:
                self.wrap_in_thread(self.preprocessor._stitch_wobbly, force=self.processing_params.stitching_rigid.skip)
                self.print_status_msg('Stitched wobbly')
            else:
                self.popup('Could not run wobbly stitching <br>without rigid stitching first')
        self.progress_dialog.done(1)

    def plot_stitching_results(self):
        self.processing_params.stitching_general.ui_to_cfg()
        dvs = self.preprocessor.plot_stitching_results(parent=self.centralwidget)
        self.setup_plots(dvs)

    def convert_output(self):
        fmt = self.processing_params.stitching_general.conversion_fmt
        self.print_status_msg('Converting stitched image to {}'.format(fmt))
        self.make_progress_dialog('Converting files')
        self.processing_params.stitching_general.ui_to_cfg()
        self.preprocessor.convert_to_image_format()  # TODO: check if use checkbox state
        self.progress_dialog.done(1)
        self.print_status_msg('Conversion finished')

    def setup_atlas(self):  # TODO: call when value changed in atlas settings
        self.save_sample_cfg()  # To make sure we have the slicing up to date
        self.processing_params.registration.ui_to_cfg()
        self.preprocessor.setup_atlases()

    def run_registration(self):
        self.print_status_msg('Registering')
        self.make_nested_progress_dialog('Registering', n_steps=self.preprocessor.n_registration_steps,
                                         sub_maximum=0, abort_callback=self.preprocessor.stop_process,
                                         parent=self)
        self.setup_atlas()
        self.print_status_msg('Resampling for registering')
        self.wrap_in_thread(self.preprocessor.resample_for_registration, force=True)
        self.print_status_msg('Aligning')
        self.wrap_in_thread(self.preprocessor.align)
        self.progress_dialog.done(1)
        self.print_status_msg('Registered')

    def plot_registration_results(self):
        image_sources = [
            self.preprocessor.workspace.filename('resampled', postfix='autofluorescence'),
            mhd_read(self.preprocessor.aligned_autofluo_path)
        ]
        dvs = plot_3d.plot(image_sources, arange=False, sync=True, lut=self.preferences.lut,
                           parent=self.centralWidget())  # FIXME: why parenthesis to centralWidget requierd here only
        link_dataviewers_cursors(dvs)
        self.setup_plots(dvs, ['autofluo', 'aligned'])

    # ################################  CELL MAP  #################################

    def plot_debug_cropping_interface(self):
        img = self.cell_detector.workspace.source('resampled')
        self.ortho_viewer.setup(img, self.cell_map_params, parent=self)
        dvs = self.ortho_viewer.plot_orthogonal_views()
        self.ortho_viewer.add_cropping_bars()
        self.setup_plots(dvs, ['x', 'y', 'z'])

        # WARNING: needs to be done after setup
        # OPTIMISE: try clearmap_io.shape(self.preprocessor.workspace('stitched'))
        shape = self.preprocessor.workspace.source('stitched').shape
        self.cell_map_tab.detectionSubsetXRangeMax.setMaximum(shape[0])
        self.cell_map_tab.detectionSubsetYRangeMax.setMaximum(shape[1])
        self.cell_map_tab.detectionSubsetZRangeMax.setMaximum(shape[2])

        self.ortho_viewer.update_ranges()

    def create_cell_detection_tuning_sample(self):
        slicing = (slice(self.cell_map_params.crop_x_min, self.cell_map_params.crop_x_max),
                   slice(self.cell_map_params.crop_y_min, self.cell_map_params.crop_y_max),
                   slice(self.cell_map_params.crop_z_min, self.cell_map_params.crop_z_max))
        self.cell_detector.create_test_dataset(slicing=slicing)
        self.print_status_msg('Tuning sample created')  # TODO: progress bar

    def run_tuning_cell_detection(self):
        self.cell_map_params.ui_to_cfg()
        self.make_progress_dialog('Cell detection preview')
        self.wrap_in_thread(self.cell_detector.run_cell_detection, tuning=True)
        self.progress_dialog.done(1)
        if self.cell_detector.stopped:
            return
        with self.cell_detector.workspace.tmp_debug:
            self.plot_detection_results()

    def detect_cells(self):  # TODO: merge w/ above w/ tuning option
        self.cell_map_params.ui_to_cfg()
        self.print_status_msg('Starting cell detection')
        self.make_nested_progress_dialog(title='Detecting cells', n_steps=0,
                                         abort_callback=self.cell_detector.stop_process)
        self.wrap_in_thread(self.cell_detector.run_cell_detection, tuning=False)
        if self.cell_detector.stopped:
            return
        if self.cell_map_params.plot_detected_cells:
            self.cell_detector.plot_cells()  # TODO: integrate into UI
        self.cell_map_tab.nDetectedCellsLabel.setText(
            format_long_nb_to_str(self.cell_detector.get_n_detected_cells()))
        self.progress_dialog.done(1)
        self.print_status_msg('Cell detection done')

    # def reset_detected(self):
    #     self.cell_detector.detected = False

    def plot_detection_results(self):
        dvs = self.cell_detector.preview_cell_detection(parent=self.centralWidget(), arange=False, sync=True)  # TODO: add close
        if len(dvs) == 1:
            self.print_warning_msg('Preview not run, will only display stitched image for memory space reasons')
        else:
            link_dataviewers_cursors(dvs)
        self.setup_plots(dvs)

    def plot_cell_filter_results(self):
        dvs = self.cell_detector.plot_filtered_cells(smarties=True)
        self.setup_plots(dvs)

    def preview_cell_filter(self):
        self.cell_map_params.ui_to_cfg()
        with self.cell_detector.workspace.tmp_debug:
            debug_raw_cells_path = self.cell_detector.workspace.filename('cells', postfix='raw')
            if os.path.exists(debug_raw_cells_path):
                self.cell_detector.filter_cells()
                self.cell_detector.voxelize('filtered')
            self.plot_cell_filter_results()

    def run_cell_map(self):
        self.cell_map_params.ui_to_cfg()
        if not self.cell_detector.detected:
            self.detect_cells()
        self.cell_map_tab.nDetectedCellsLabel.setText(
            format_long_nb_to_str(self.cell_detector.get_n_detected_cells()))
        self.cell_detector.post_process_cells()
        self.cell_map_tab.nDetectedCellsAfterFilterLabel.setText(
            format_long_nb_to_str(self.cell_detector.get_n_fitlered_cells()))
        if self.cell_map_params.plot_when_finished:
            self.plot_cell_map_results()
        # WARNING: some plots in .post_process_cells() without UI params

    def plot_cell_map_results(self):
        dvs = self.cell_detector.plot_voxelized_counts(arange=False)
        self.setup_plots(dvs)

# TUBE MAP
    def _get_n_binarize_steps(self):
        n_steps = 1
        n_steps += self.vessel_params.binarization_params.post_process_raw
        n_steps += self.vessel_params.binarization_params.run_arteries_binarization
        n_steps += self.vessel_params.binarization_params.post_process_arteries
        return n_steps

    def binarize_vessels(self):
        self.vessel_params.ui_to_cfg()
        self.print_status_msg('Starting vessel binarization')
        self.make_nested_progress_dialog(title='Binarizing vessels', n_steps=self._get_n_binarize_steps(),
                                         abort_callback=self.binary_vessel_processor.stop_process)
        self.wrap_in_thread(self.binary_vessel_processor.binarize)
        self.progress_dialog.done(1)
        self.print_status_msg('Vessels binarized')

    def fill_vessels(self):
        self.vessel_params.ui_to_cfg()
        self.print_status_msg('Starting vessel filling')
        n_steps = self.vessel_params.binarization_params.fill_main_channel +\
                  self.vessel_params.binarization_params.fill_secondary_channel
        self.make_nested_progress_dialog(title='Filling vessels', n_steps=n_steps,
                                         abort_callback=self.binary_vessel_processor.stop_process)
        self.wrap_in_thread(self.binary_vessel_processor.fill_vessels)
        self.wrap_in_thread(self.binary_vessel_processor.combine_binary)  # REFACTOR: not great location
        self.progress_dialog.done(1)
        self.print_status_msg('Vessel filling done')

    def build_graph(self):
        self.vessel_params.ui_to_cfg()
        self.print_status_msg('Building vessel graph')
        self.make_nested_progress_dialog(title='Building vessel graph', n_steps=4,
                                         abort_callback=self.vessel_graph_processor.stop_process)
        self.wrap_in_thread(self.vessel_graph_processor.pre_process)
        self.progress_dialog.done(1)
        self.print_status_msg('Building vessel graph done')

    def post_process_graph(self):
        self.vessel_params.ui_to_cfg()
        self.print_status_msg('Post processing vasculature graph')
        self.make_nested_progress_dialog(title='Post processing graph', n_steps=8,
                                         abort_callback=self.vessel_graph_processor.stop_process)
        self.wrap_in_thread(self.vessel_graph_processor.post_process)
        self.progress_dialog.done(1)
        self.print_status_msg('Vasculature graph post-processing DONE')


def create_main_window(app):
    preprocessor = PreProcessor()
    clearmap_main_win = ClearMapGui(preprocessor)
    if clearmap_main_win.preferences.start_full_screen:
        clearmap_main_win.showMaximized()  # TODO: check if redundant with show
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    return clearmap_main_win


def main(app, splash):
    # app.setAttribute(QtCore.Qt.AA_DontCreateNativeWidgetSiblings)

    clearmap_main_win = create_main_window(app)

    def except_hook(exc_type, exc_value, exc_tb):
        lexer = PythonTracebackLexer()
        formatter = HtmlFormatter(full=True, style='native', lineos='table', wrapcode=True, noclasses=True)
        formatter.style.background_color = QDARKSTYLE_BACKGROUND
        raw_traceback = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        formatted_traceback = pygments.highlight(raw_traceback, lexer, formatter)
        clearmap_main_win.error_logger.write(formatted_traceback)

    clearmap_main_win.show()
    splash.finish(clearmap_main_win)
    if clearmap_main_win.preferences.verbosity != 'trace':  # WARNING: will disable progress bars
        clearmap_main_win.patch_stdout()
        sys.excepthook = except_hook
    sys.exit(app.exec())


if __name__ == "__main__":
    main(app, splash)
