import os
import sys

from shutil import copyfile
import traceback
import types

import numpy as np
import pygments
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonTracebackLexer
from skimage import transform as sk_transform

from PyQt5 import QtGui
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QSpinBox, \
    QDoubleSpinBox, QFrame, QDialogButtonBox, QComboBox, QLineEdit, QStyle, QWidget, QMessageBox, QToolBox, \
    QProgressDialog

import qdarkstyle

import ClearMap.Visualization.Plot3d as plot_3d
from ClearMap.IO import TIF
from ClearMap.IO.MHD import mhd_read
from ClearMap.Settings import resources_path

from ClearMap.Scripts.cell_map import CellDetector
from ClearMap.Scripts.sample_preparation import PreProcessor

from ClearMap.config.config_loader import get_cfg

from ClearMap.gui.gui_utils import Printer, QDARKSTYLE_BACKGROUND, DARK_BACKGROUND, np_to_qpixmap, clean_path, \
    html_to_ansi, html_to_plain_text, compute_grid, BLUE_COLOR_TABLE, runs_from_pycharm, surface_project, \
    format_long_nb_to_str, link_dataviewers_cursors
from ClearMap.gui.params import SampleParameters, ConfigNotFoundError, GeneralStitchingParams, RigidStitchingParams, \
    WobblyStitchingParams, RegistrationParams, CellMapParams, PreferencesParams
from ClearMap.gui.pyuic_utils import loadUiType
from ClearMap.gui.widget_monkeypatch_callbacks import get_value, set_value, controls_enabled, get_check_box, \
    enable_controls, disable_controls, set_text, get_text, connect_apply
from ClearMap.gui.widgets import RectItem

# TODO
"""
Previews:
    - Add rigid alignment : plane in middle of stack from each column + stitch with different colours
    - CellMap Filtered cells (plot voxelize_unweighted from chunk (single dataviewer)
    - CellMap Run: plot checkbox + plot button
Delete intermediate files
Analysis:
(X)Check with Christophe slice range ...

Progress bar


preferences:
    (-) number of processes for all
    (X) chunk size
    
LATER:
Auto modes:
    - Run all
    - Batch run


DONE:
CellMap:
    (X) reset detected upon value changed
Atlas space info:
    (X) Add cropping to preview
(X) Scale dataviewer 
"""

Ui_ClearMapGui, _ = loadUiType('ClearMap/gui/mainwindow.ui', patch_parent_class=False)


class ClearMapGui(QMainWindow, Ui_ClearMapGui):
    def __init__(self, preprocessor):
        super(ClearMapGui, self).__init__()

        self.preprocessor = preprocessor
        self.logger = None
        self.machine_cfg_path = None
        self.sample_cfg_path = None
        self.processing_cfg_path = None
        self.cell_map_params = None
        self.graph_names = {}

        atlas_path = os.path.join(resources_path, 'Atlas', 'ABA_25um_annotation.tif')  # WARNING: function of chosen atlas
        arr = TIF.Source(atlas_path).array
        self.mini_brain_scaling = (5, 5, 5)
        self.mini_brain = sk_transform.downscale_local_mean(arr, self.mini_brain_scaling)

        self.setWindowIcon(QtGui.QIcon('icons/logo_cyber.png'))  # REFACTOR: use qrc

        self.setupUi(self)
        self.amendUi()

        self.actionPreferences.triggered.connect(self.config_window.exec)

    def find_child_by_name(self, child_name, child_type, parent=None):
        if parent is None:
            parent = self
        for child in parent.findChildren(child_type):
            if child.objectName() == child_name:
                return child

    def swap_resolutions_group_box(self):
        self.sample_tab.resolutionsGroupBox.setVisible(self.sample_tab.advancedCheckBox.isChecked())

    def swap_preprocessing_tab_advanced(self):
        checked = self.preprocessing_tab.advancedCheckBox.isChecked()
        self.preprocessing_tab.atlasSettingsPage.setVisible(checked)

    def print_error_msg(self, msg):
        self.statusbar.setStyleSheet("color: red")
        self.statusbar.showMessage(msg)

    def print_warning_msg(self, msg):
        self.statusbar.setStyleSheet("color: yellow")
        self.statusbar.showMessage(msg)

    def print_status_msg(self, msg):
        self.statusbar.setStyleSheet("color: white")
        self.statusbar.showMessage(msg)

    def patch_stdout(self):
        sys.stdout = self.logger
        sys.stderr = self.error_logger

    def amendUi(self):
        self.logger = Printer(self.textBrowser)
        self.error_logger = Printer(self.textBrowser, 'red')

        self.setupIcons()

        self.setup_sample_tab()
        self.setup_preprocessing_tab()
        self.setup_cell_map_tab()
        self.tabWidget.setCurrentIndex(0)

        self.setup_preferences_editor()

        self.patch_compound_boxes()
        self.patch_button_boxes()
        self.patch_tool_boxes()
        self.fix_styles()
        self.logoLabel.setPixmap(QtGui.QPixmap('icons/logo_cyber.png'))

        self.graphLayout.removeWidget(self.frame)  # FIXME:
        # dvs = plot_3d.plot([os.path.expanduser('~/Desktop/cell_map_test_images/auto.tif'),
        #                     os.path.expanduser('~/Desktop/cell_map_test_images/resampled.tif')],
        #                    arange=False, lut='white', parent=self.centralwidget)
        # link_dataviewers_cursors(dvs)
        # self.setup_plots(dvs)

        # self.graphDock.resized = types.ClassMethodDescriptorType(pyqtSignal()
        # self.graphDock.resizeEvent = types.MethodType(dock_resize_event, self.graphDock)
        # self.graphDock.resized.connect(self.resize_graphs)

        self.print_status_msg('Idle, waiting for input')

    # def resizeEvent(self, event):
    #     # dock_size = self.graphDock.size()
    #     self.resize_graphs()
    #     QMainWindow.resizeEvent(self, event)

    def setupIcons(self):
        self.__reload_icon = self.style().standardIcon(QStyle.SP_BrowserReload)

    def fix_styles(self):
        self.sample_tab.sampleIdButtonBox.button(QDialogButtonBox.Apply).setIcon(self.__reload_icon)

        self.fix_btn_boxes_text()

        self.fix_btns_stylesheet()
        self.fix_widgets_backgrounds()

    def fix_btn_boxes_text(self):
        for btn_box in self.findChildren(QDialogButtonBox):
            if btn_box.property('applyText'):
                btn_box.button(QDialogButtonBox.Apply).setText(btn_box.property('applyText'))

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
                elif bx_name.endswith('optionallineedit'):
                    bx.setText = types.MethodType(set_text, bx)
                    bx.text = types.MethodType(get_text, bx)
                else:
                    print('Skipping box "{}", type not recognised'.format(bx_name))

    def patch_button_boxes(self, parent=None):
        if parent is None:
            parent = self
        for bx in parent.findChildren(QDialogButtonBox):
            bx.connectApply = types.MethodType(connect_apply, bx)

    def patch_tool_boxes(self):
        for tb in self.findChildren(QToolBox):
            tb.setCurrentIndex(0)

    def setup_sample_tab(self):
        cls, _ = loadUiType('ClearMap/gui/sample_tab.ui', patch_parent_class='QTabWidget')
        self.sample_tab = cls()
        self.sample_tab.setupUi()
        self.patch_button_boxes(self.sample_tab)
        self.tabWidget.removeTab(0)
        self.tabWidget.insertTab(0, self.sample_tab, 'Sample')

        self.sample_tab.plotMiniBrainPushButton.clicked.connect(self.plot_mini_brain)

        self.sample_tab.advancedCheckBox.stateChanged.connect(self.swap_resolutions_group_box)
        self.sample_tab.srcFolderBtn.clicked.connect(self.set_src_folder)

        # src_folder = os.path.expanduser('~/Desktop/cell_map_test_images')
        # self.src_folder = src_folder
        # self.sample_params = SampleParameters(self.sample_tab, src_folder)  # FIXME: do upon folder selection

        self.sample_tab.sampleIdButtonBox.connectApply(self.parse_cfg)
        self.sample_tab.applyBox.connectApply(self.setup_preprocessor)
        self.sample_tab.applyBox.button(QDialogButtonBox.Save).clicked.connect(self.save_sample_cfg)  # TODO: check if could just be self.sample_cfg.ui_to_cfg

    def setup_preprocessing_tab(self):
        cls, _ = loadUiType('ClearMap/gui/preprocessing_tab.ui', patch_parent_class='QTabWidget')
        self.preprocessing_tab = cls()
        self.preprocessing_tab.setupUi()
        self.patch_button_boxes(self.preprocessing_tab)
        self.tabWidget.removeTab(1)
        self.tabWidget.insertTab(1, self.preprocessing_tab, 'Alignments')

        self.preprocessing_tab.runStitchingButtonBox.connectApply(self.run_stitching)
        self.preprocessing_tab.displayStitchingButtonBox.connectApply(self.plot_stitching_results)
        self.preprocessing_tab.displayStitchingButtonBox.button(QDialogButtonBox.Close). \
            clicked.connect(self._remove_old_plots)
        self.preprocessing_tab.convertOutputButtonBox.connectApply(self.convert_output)
        self.preprocessing_tab.registerButtonBox.connectApply(self.run_registration)
        self.preprocessing_tab.plotRegistrationResultsButtonBox.connectApply(self.plot_registration_results)

        self.preprocessing_tab.atlasSettingsPage.setVisible(False)
        self.preprocessing_tab.advancedCheckBox.stateChanged.connect(self.swap_preprocessing_tab_advanced)

    def setup_cell_map_tab(self):
        cls, _ = loadUiType('ClearMap/gui/cell_map_tab.ui', patch_parent_class='QTabWidget')
        self.cell_map_tab = cls()
        self.cell_map_tab.setupUi()
        self.patch_button_boxes(self.cell_map_tab)
        self.tabWidget.removeTab(2)
        self.tabWidget.insertTab(2, self.cell_map_tab, 'CellMap')
        self.tabWidget.tabBarClicked.connect(self.handle_tab_click)

        self.cell_map_tab.detectionPreviewTuningButtonBox.button(QDialogButtonBox.Open). \
            clicked.connect(self.plot_debug_cropping_interface)
        self.cell_map_tab.detectionPreviewTuningSampleButtonBox.connectApply(self.create_cell_detection_tuning_sample)
        self.cell_map_tab.detectionPreviewButtonBox.connectApply(self.run_tuning_cell_detection)
        self.cell_map_tab.detectionSubsetXRangeMin.valueChanged.connect(self.update_x_min)
        self.cell_map_tab.detectionSubsetXRangeMax.valueChanged.connect(self.update_x_max)
        self.cell_map_tab.detectionSubsetYRangeMin.valueChanged.connect(self.update_y_min)
        self.cell_map_tab.detectionSubsetYRangeMax.valueChanged.connect(self.update_y_max)
        self.cell_map_tab.detectionSubsetZRangeMin.valueChanged.connect(self.update_z_min)
        self.cell_map_tab.detectionSubsetZRangeMax.valueChanged.connect(self.update_z_max)

        # for ctrl in (self.cell_map_tab.backgroundCorrectionDiameter, self.cell_map_tab.detectionThreshold):
        #     ctrl.valueChanged.connect(self.reset_detected)  FIXME: find better way

        self.cell_map_tab.runCellDetectionButtonBox.connectApply(self.detect_cells)
        self.cell_map_tab.runCellDetectionPlotButtonBox.connectApply(self.plot_detection_results)
        self.cell_map_tab.previewCellFiltersButtonBox.connectApply(self.preview_cell_filter)

        self.cell_map_tab.runCellMapButtonBox.connectApply(self.run_cell_map)
        self.cell_map_tab.runCellMapPlotButtonBox.connectApply(self.plot_cell_map_results)

    def setup_preferences_editor(self):
        cls, _ = loadUiType('ClearMap/gui/preferences_editor.ui', patch_parent_class='QDialog')
        self.config_window = cls()
        self.config_window.setWindowTitle('Preferences')
        self.config_window.setupUi()
        self.patch_button_boxes(self.config_window)

        self.setup_preferences()

        self.config_window.buttonBox.connectApply(self.preferences.ui_to_cfg)
        self.config_window.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(self.config_window.close)
        self.config_window.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self.apply_prefs_and_close)

    def handle_tab_click(self, tab_index):
        if tab_index == 2:  # FIXME: handle other tabs (e.g. message if workspace not init when tab_index==1)
            self.setup_cell_detector()

    def apply_prefs_and_close(self):
        self.preferences.ui_to_cfg()
        self.config_window.close()

    def save_sample_cfg(self):
        self.sample_params.ui_to_cfg()
        self.print_status_msg('Sample config saved')

    def setup_preprocessor(self):
        self.save_sample_cfg()
        self.preprocessor.setup((self.sample_cfg_path, self.processing_cfg_path, self.machine_cfg_path))

    def setup_cell_detector(self):
        if self.cell_detector.preprocessor is None and self.preprocessor.workspace is not None:  # preproc initialised
            self.update_preprocessing_cfg()
            self.cell_detector.setup(self.preprocessor)

    def update_preprocessing_cfg(self):
        for param in self.processing_params.values():
            param.ui_to_cfg()

    def get_cell_map_cfg_path(self):
        cfg_path = clean_path(os.path.join(self.src_folder, 'cell_map_params.cfg'))
        if not self.file_exists(cfg_path):
            default_cfg_file_path = clean_path('~/.clearmap/default_cell_map_params.cfg')
            base_msg, msg = self.create_missing_file_msg('CellMap', cfg_path, default_cfg_file_path)
            ret = self.popup(msg)
            if ret == QMessageBox.Ok:
                copyfile(default_cfg_file_path, cfg_path)
            else:
                raise FileNotFoundError(html_to_ansi(base_msg))
        self.cell_map_cfg_path = cfg_path
        return cfg_path

    @staticmethod
    def create_missing_file_msg(f_type, f_path, default_f_path):
        base_msg = 'No {} file found at:<br>  <nobr><em>"{}"</em></nobr>.'.format(f_type, f_path)
        msg = '{} <br><br>Do you want to load a default one from:<br>  <nobr><em>"{}"</em></nobr>' \
            .format(base_msg, default_f_path)
        return base_msg, msg

    def popup(self, msg):
        self.print_warning_msg(html_to_plain_text(msg))
        dlg = QMessageBox()
        dlg.setIcon(QMessageBox.Warning)
        dlg.setWindowTitle('Warning')
        dlg.setText('<b>Missing configuration file</b>')
        dlg.setInformativeText(msg)
        dlg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        dlg.setDefaultButton(QMessageBox.Ok)
        return dlg.exec()

    def get_cfg_paths(self):
        if not self.src_folder:
            msg = 'Missing source folder, please define first'
            self.print_error_msg(msg)
            raise FileNotFoundError(msg)
        sample_cfg_path = clean_path(os.path.join(self.src_folder, 'sample.cfg'))
        if not self.file_exists(sample_cfg_path):
            default_sample_cfg_path = clean_path('~/.clearmap/default_sample.cfg')
            base_msg, msg = self.create_missing_file_msg('sample', sample_cfg_path, default_sample_cfg_path)
            ret = self.popup(msg)
            if ret == QMessageBox.Ok:
                copyfile(default_sample_cfg_path, sample_cfg_path)
            else:
                raise FileNotFoundError(html_to_ansi(base_msg))
            self.sample_params.fix_sample_cfg_file(sample_cfg_path)
        self.sample_cfg_path = sample_cfg_path
        processing_cfg_path = clean_path(os.path.join(self.src_folder, 'processing_params.cfg'))
        if not self.file_exists(processing_cfg_path):
            default_processing_cfg_path = clean_path('~/.clearmap/default_processing_params.cfg')
            base_msg, msg = self.create_missing_file_msg('processing params',
                                                         processing_cfg_path, default_processing_cfg_path)
            ret = self.popup(msg)
            if ret == QMessageBox.Ok:
                copyfile(default_processing_cfg_path, processing_cfg_path)
            else:
                raise FileNotFoundError(html_to_ansi(base_msg))
        self.processing_cfg_path = processing_cfg_path

    def setup_preferences(self):
        self.preferences = PreferencesParams(self.config_window, self.src_folder)
        machine_cfg_path = clean_path('~/.clearmap/machine_params.cfg')
        if self.file_exists(machine_cfg_path):
            self.machine_cfg_path = machine_cfg_path
            self.preferences.get_config(self.machine_cfg_path)
            self.preferences.cfg_to_ui()
        else:
            self.print_error_msg('Missing machine config file. Please ensure a machine_params.cfg file '
                                 'is available at {}. This should be done at installation'.format(machine_cfg_path))
            raise FileNotFoundError('Missing machine config file. Please ensure a machine_params.cfg file '
                                    'is available at {}. This should be done at installation'.format(machine_cfg_path))

    def file_exists(self, f_path):
        if os.path.exists(f_path):
            return True
        else:
            msg = 'File "{}" not found'.format(f_path)
            self.print_error_msg(msg)
            return False

    def parse_cfg(self):
        self.print_status_msg('Parsing configuration')
        try:
            self.get_cfg_paths()
        except FileNotFoundError as err:
            self.error_logger.write(str(err))  # TODO: see if let escalate
            return

        error = False
        self.machine_config = get_cfg(self.machine_cfg_path)
        if not self.machine_config:
            self.print_error_msg('Loading machine config file failed')
            error = True
        try:
            self.sample_params.get_config(self.sample_cfg_path)
        except ConfigNotFoundError as err:
            self.print_error_msg('Loading sample config file failed')
            error = True
        self.processing_params = {
            'stitching': GeneralStitchingParams(self.preprocessing_tab),
            'rigid_stitching': RigidStitchingParams(self.preprocessing_tab),
            'wobbly_stitching': WobblyStitchingParams(self.preprocessing_tab),
            'registration': RegistrationParams(self.preprocessing_tab)
        }
        for param in self.processing_params.values():
            try:
                param.get_config(self.processing_cfg_path)
            except ConfigNotFoundError as err:
                self.print_error_msg('Loading preprocessing config file failed')
                error = True
        if self.processing_params['stitching'].config['pipeline_name'].lower() == 'cellmap':
            self.cell_map_params = CellMapParams(self.cell_map_tab)
            try:
                self.cell_map_params.get_config(self.get_cell_map_cfg_path())
            except ConfigNotFoundError as err:
                self.print_error_msg('Loading Cell Map config file failed')
                error = True
            self.cell_detector = CellDetector()
        if not error:
            self.print_status_msg('Config loaded')
            self.sample_params.cfg_to_ui()
            for param in self.processing_params.values():
                param.cfg_to_ui()
            if self.cell_map_params is not None:
                self.cell_map_params.cfg_to_ui()

    def set_src_folder(self):
        diag = QFileDialog()
        if sys.platform == 'win32' or runs_from_pycharm():  # avoids bug with windows COM object init failed
            opt = QFileDialog.Options(QFileDialog.DontUseNativeDialog)
        else:
            opt = QFileDialog.Options()
        src_folder = diag.getExistingDirectory(parent=diag, caption="Choose the source directory",
                                               directory=self.preferences.start_folder, options=opt)
        diag.close()
        self.src_folder = src_folder
        self.sample_params = SampleParameters(self.sample_tab, src_folder)

    @property
    def src_folder(self):
        return self.sample_tab.srcFolderTxt.text()

    @src_folder.setter
    def src_folder(self, src_folder):
        self.sample_tab.srcFolderTxt.setText(src_folder)

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

    def get_graphs(self):
        return [getattr(self, attr) for attr in dir(self) if attr.startswith('graph_')]

    def resize_graphs(self):
        n_rows, n_cols = compute_grid(len(self.get_graphs()))  # WARNING: take care of placeholders
        for i in range(self.graphLayout.count(), -1, -1):
            graph = self.graphLayout.itemAt(i)
            if graph is not None:
                widg = graph.widget()
                self.__resize_graph(widg, n_cols, n_rows)

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

        dlg = QProgressDialog(msg, 'Abort', 0, maximum, parent=self)  # TODO: see if can have a notnativestyle on unity
        dlg.setMinimumDuration(0)  # TODO: add image
        dlg.setValue(0)  # To force update
        self.progress_dialog = dlg  # TODO: bind was canceled

    def run_stitching(self):
        stitched_rigid = False
        for param_name, param in self.processing_params.items():
            if param_name.endswith('stitching'):
                param.ui_to_cfg()
                for param_name, param in self.processing_params.items():  # FIXME: make less hacky. required because otherwise the other dicts that overlap will not have been updated
                    if param_name.endswith('stitching'):
                        param.config.reload()  # FIXME: probably need to do that for registration too
        self.preprocessor.reload_processing_cfg()
        self.make_progress_dialog('Stitching')
        self.print_status_msg('Stitching')
        if not self.processing_params['rigid_stitching'].skip:
            layout = self.preprocessor._stitch_rigid()
            stitched_rigid = True
            self.progress_dialog.setValue(30)
            self.print_status_msg('Stitched rigid')
        if not self.processing_params['wobbly_stitching'].skip:
            if stitched_rigid:
                self.preprocessor._stitch_wobbly(layout)
                self.print_status_msg('Stitched wobbly')
            else:
                self.popup('Could not run wobbly stitching <br>without rigid stitching first')
        self.progress_dialog.setValue(self.progress_dialog.maximum())  # TODO: make more progressive

    def plot_stitching_results(self):
        self.processing_params['stitching'].ui_to_cfg()
        dvs = self.preprocessor.plot_stitching_results(parent=self.centralwidget)
        self.setup_plots(dvs)

    def convert_output(self):
        fmt = self.processing_params['stitching'].conversion_fmt
        self.print_status_msg('Convertng stitched image to {}'.format(fmt))
        self.processing_params['stitching'].ui_to_cfg()
        self.preprocessor.convert_to_image_format()  # TODO: check if use checkbox state
        self.print_status_msg('Conversion finished')

    def plot_registration_results(self):
        image_sources = [
            self.preprocessor.workspace.filename('resampled', postfix='autofluorescence'),
            mhd_read(os.path.join(self.preprocessor.workspace.filename('auto_to_reference'), 'result.1.mhd'))
        ]
        dvs = plot_3d.plot(image_sources, arange=False, sync=False, lut='white',
                           parent=self.centralWidget())  # TODO: lut as part of preferences # FIXME: why parenthesis
        link_dataviewers_cursors(dvs)
        self.setup_plots(dvs, ['autofluo', 'aligned'])

    def plot_orthogonal_views(self, img):
        x = np.copy(img)
        y = np.copy(img).swapaxes(0, 1)
        z = np.copy(img).swapaxes(0, 2)
        return plot_3d.plot([x, y, z], arange=False, lut='white', parent=self.centralwidget, sync=False)

    def plot_debug_cropping_interface(self):
        img = TIF.Source(self.preprocessor.workspace.filename('resampled'))
        shape = img.shape
        dvs = self.plot_orthogonal_views(img.array)

        self.x_rect_min = RectItem(QRectF(0, 0, 0, shape[1]))  # REFACTOR:
        dvs[0].view.addItem(self.x_rect_min)
        self.x_rect_max = RectItem(QRectF(shape[0], 0, 0, shape[1]))
        dvs[0].view.addItem(self.x_rect_max)
        # dvs[0].jumpFrames(round(shape[2] / 2))

        self.y_rect_min = RectItem(QRectF(0, 0, 0, shape[0]))
        dvs[1].view.addItem(self.y_rect_min)
        self.y_rect_max = RectItem(QRectF(shape[1], 0, 0, shape[0]))
        dvs[1].view.addItem(self.y_rect_max)

        self.z_rect_min = RectItem(QRectF(0, 0, 0, shape[1]))
        dvs[2].view.addItem(self.z_rect_min)
        self.z_rect_max = RectItem(QRectF(shape[2], 0, 0, shape[1]))
        dvs[2].view.addItem(self.z_rect_max)

        self.setup_plots(dvs, list('xyz'))

        # After setup
        self.cell_map_tab.detectionSubsetXRangeMax.setMaximum(shape[0])  # TODO: check if value resets if set at more than max
        self.cell_map_tab.detectionSubsetYRangeMax.setMaximum(shape[1])
        self.cell_map_tab.detectionSubsetZRangeMax.setMaximum(shape[2])

    def _update_rect(self, axis, val, min_or_max='min'):
        rect_item_name = '{}_rect_{}'.format(axis, min_or_max)
        if not hasattr(self, rect_item_name):
            return
        rect_itm = getattr(self, rect_item_name)
        if min_or_max == 'min':
            rect_itm.rect.setWidth(val)
        else:
            rect_itm.rect.setLeft(val)
        rect_itm._generate_picture()
        graph = getattr(self, self.graph_names[axis])
        graph.view.update()

    def update_x_min(self, val):  # TODO: move to attribute object
        self._update_rect('x', val, 'min')

    def update_x_max(self, val):
        self._update_rect('x', val, 'max')

    def update_y_min(self, val):
        self._update_rect('y', val, 'min')

    def update_y_max(self, val):
        self._update_rect('y', val, 'max')

    def update_z_min(self, val):
        self._update_rect('z', val, 'min')

    def update_z_max(self, val):
        self._update_rect('z', val, 'max')

    def create_cell_detection_tuning_sample(self):  # TODO add messages
        ratios = self.get_cell_map_scaling_ratios(direction='to_resampled')
        crops = self.cell_map_params.scale_crop_values(ratios)

        # TODO: verify order
        slicing = (
            slice(crops[0], crops[1]),  # TODO: check if dict better
            slice(crops[2], crops[3]),
            slice(crops[4], crops[5])
        )
        self.cell_detector.create_test_dataset(slicing=slicing)
        self.cell_map_params.crop_values_to_cfg(self.get_cell_map_scaling_ratios(direction='to_original'))

    def get_cell_map_scaling_ratios(self, direction='to_original'):
        raw_res = np.array(self.sample_params.raw_resolution)
        atlas_res = np.array(self.processing_params['registration'].raw_atlas_resolution)
        if direction == 'to_original':
            ratios = raw_res / atlas_res
        elif direction == 'to_resampled':
            ratios = atlas_res / raw_res
        else:
            raise ValueError('Valid values for direction are to_original and to_resampled, got "{}" instead'
                             .format(direction))
        return ratios

    def run_tuning_cell_detection(self):
        self.cell_map_params.ui_to_cfg()
        self.cell_map_params.crop_values_to_cfg(ratios=self.get_cell_map_scaling_ratios())
        self.cell_detector.run_cell_detection(tuning=True)

    def plot_mini_brain(self):
        img = self.__transform_mini_brain()
        mask, proj = surface_project(img)
        img = np_to_qpixmap(proj, mask, BLUE_COLOR_TABLE)
        self.sample_tab.miniBrainLabel.setPixmap(img)

    def __transform_mini_brain(self):
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

    def detect_cells(self):
        self.cell_map_params.ui_to_cfg()
        self.make_progress_dialog('Detecting cells')
        self.cell_detector.run_cell_detection(tuning=False)
        # self.progress_dialog.setValue(self.progress_dialog.maximum())  # TODO: see why doesn't work
        if self.cell_map_params.plot_detected_cells:
            self.cell_detector.plot_cells()  # TODO: integrate into UI
        self.cell_map_tab.nDetectedCellsLabel.setText(
            format_long_nb_to_str(self.cell_detector.get_n_detected_cells()))

    # def reset_detected(self):
    #     self.cell_detector.detected = False

    def plot_detection_results(self):
        dvs = self.cell_detector.preview_cell_detection(parent=self.centralWidget(), arange=False, sync=False)  # TODO: add close
        if len(dvs) == 1:
            self.print_warning_msg('Preview not run, will only display stitched image for memory space reasons')
        self.setup_plots(dvs)

    def plot_cell_filter_results(self):  # TODO: FIX, does not integrate as widget
        dvs = self.cell_detector.plot_filtered_cells()
        self.setup_plots(dvs)

    def preview_cell_filter(self):
        self.cell_detector.workspace.debug = True
        debug_raw_cells_path = self.cell_detector.workspace.filename('cells', postfix='raw')
        if os.path.exists(debug_raw_cells_path):
            debug_filtered_cells_path = self.cell_detector.workspace.filename('cells', postfix='filtered')
            if not os.path.exists(debug_filtered_cells_path):
                self.cell_detector.filter_cells()
            self.cell_detector.voxelize('filtered')
        # self.plot_cell_map_results()
        # self.plot_cell_filter_results()  # WARNING:
        dvs = self.cell_detector.plot_voxelized_counts(arange=False, parent=self.centralWidget())
        self.setup_plots(dvs)
        self.cell_detector.workspace.debug = False

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

    def setup_atlas(self):  # TODO: call when value changed in atlas settings
        self.sample_params.ui_to_cfg()  # To make sure we have the slicing up to date
        self.processing_params['registration'].ui_to_cfg()
        self.preprocessor.setup_atlases()

    def run_registration(self):
        self.print_status_msg('Registering')
        self.make_progress_dialog('Registering')
        self.setup_atlas()
        self.progress_dialog.setValue(10)
        self.print_status_msg('Resampling for registering')
        self.preprocessor.resample_for_registration()
        self.progress_dialog.setValue(30)
        self.preprocessor.align()  # TODO: update value from within align
        self.progress_dialog.setValue(self.progress_dialog.maximum())
        self.print_status_msg('Registered')


def main():
    app = QApplication([])
    # app.setAttribute(QtCore.Qt.AA_DontCreateNativeWidgetSiblings)
    preprocessor = PreProcessor()
    clearmap_main_win = ClearMapGui(preprocessor)
    if clearmap_main_win.preferences.start_full_screen:
        clearmap_main_win.showMaximized()  # TODO: check if redundant with show
    app.setStyleSheet(qdarkstyle.load_stylesheet())

    def except_hook(exc_type, exc_value, exc_tb):
        lexer = PythonTracebackLexer()
        formatter = HtmlFormatter(full=True, style='native', lineos='table', wrapcode=True, noclasses=True)
        formatter.style.background_color = QDARKSTYLE_BACKGROUND
        raw_traceback = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        formatted_traceback = pygments.highlight(raw_traceback, lexer, formatter)
        clearmap_main_win.error_logger.write(formatted_traceback)

    clearmap_main_win.show()
    clearmap_main_win.patch_stdout()
    sys.excepthook = except_hook
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
