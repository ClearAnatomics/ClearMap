# -*- coding: utf-8 -*-
"""
run_gui
=======

The main file for the GUI that contains the entry point to start the software
"""

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

import math
import os
import sys
import time
from datetime import datetime

from multiprocessing.pool import ThreadPool
from shutil import copyfile
import traceback
import types

import psutil
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWebEngineWidgets import QWebEngineView  # WARNING: import required before app creation
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QSpinBox, QDoubleSpinBox, QFrame, \
    QDialogButtonBox, QComboBox, QLineEdit, QStyle, QWidget, QMessageBox, QToolBox, QProgressBar, QLabel
from qdarkstyle import DarkPalette

os.environ['CLEARMAP_GUI_HOSTED'] = "1"
# ########################################### SPLASH SCREEN ###########################################################
from ClearMap.gui.dialogs import make_splash, update_pbar, make_simple_progress_dialog

# To show splash before slow imports
ICONS_FOLDER = 'ClearMap/gui/creator/icons/'   # REFACTOR: use qrc

app = QApplication([])
app.setApplicationName('ClearMap')
app.setApplicationDisplayName('ClearMap')
app.setApplicationVersion('2.1')
palette = app.palette()  # WARNING: necessary because QWhatsThis does not follow stylesheets
palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(DarkPalette.COLOR_BACKGROUND_2))
palette.setColor(QPalette.ColorRole.ToolTipText, QColor(DarkPalette.COLOR_TEXT_2))
app.setPalette(palette)


from ClearMap.gui.gui_utils import get_current_res, UI_FOLDER

CURRENT_RES = get_current_res(app)

splash, progress_bar = make_splash(res=CURRENT_RES)
splash.show()
update_pbar(app, progress_bar, 10)

# ############################################  SLOW IMPORTS #########################################################

import pygments
import pygments.styles
from pygments.lexers.python import PythonTracebackLexer  # noqa
from pygments.formatters.html import HtmlFormatter

import qdarkstyle

import pyqtgraph as pg

update_pbar(app, progress_bar, 20)
from ClearMap.Utils.utilities import title_to_snake, get_percent_v_ram_use, gpu_util
from ClearMap.gui.gui_logging import Printer
from ClearMap.config.config_loader import ConfigLoader
from ClearMap.gui.params import ConfigNotFoundError, UiParameterCollection, UiParameter
from ClearMap.gui.widget_monkeypatch_callbacks import get_value, set_value, controls_enabled, get_check_box, \
    enable_controls, disable_controls, set_text, get_text, connect_apply, connect_close, connect_save, connect_open, \
    connect_ok, connect_cancel, connect_value_changed, connect_text_changed
update_pbar(app, progress_bar, 40)
from ClearMap.gui.pyuic_utils import loadUiType
from ClearMap.gui.dialogs import get_directory_dlg, warning_popup, make_nested_progress_dialog, DISPLAY_CONFIG
from ClearMap.gui.gui_utils import html_to_ansi, html_to_plain_text, compute_grid
from ClearMap.gui.style import DARK_BACKGROUND, PLOT_3D_BG, \
    BTN_STYLE_SHEET, TOOLTIP_STYLE_SHEET, COMBOBOX_STYLE_SHEET, WARNING_YELLOW

from ClearMap.gui.widgets import OrthoViewer, ProgressWatcher, setup_mini_brain  # needs plot_3d
update_pbar(app, progress_bar, 60)
from ClearMap.gui.tabs import SampleTab, AlignmentTab, CellCounterTab, VasculatureTab, PreferenceUi, BatchTab

update_pbar(app, progress_bar, 80)

pg.setConfigOption('background', PLOT_3D_BG)

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

Ui_ClearMapGui, _ = loadUiType(os.path.join(UI_FOLDER, 'creator', 'mainwindow.ui'), from_imports=True,
                               import_from='ClearMap.gui.creator', patch_parent_class=False)


class ClearMapGuiBase(QMainWindow, Ui_ClearMapGui):
    def __init__(self):
        super().__init__()
        self.graphs = []
        self._reload_icon = self.style().standardIcon(QStyle.SP_BrowserReload)
        self.logger = None
        self.progress_logger = Printer(logger_type='progress')
        self.error_logger = None
        self.progress_dialog = None
        self.progress_watcher = ProgressWatcher()

        self.cpu_bar = QProgressBar()
        self.ram_bar = QProgressBar()
        self.gpu_bar = QProgressBar()
        self.vram_bar = QProgressBar()

        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update_monitoring_bars)

    def find_child_by_name(self, child_name, child_type, parent=None):
        if parent is None:
            parent = self
        for child in parent.findChildren(child_type):
            if child.objectName() == child_name:
                return child

    def __print_status_msg(self, msg, color, n_blinks=1, period=1):
        for i in range(n_blinks):
            self.statusbar.setStyleSheet(f'color: {color}')
            self.statusbar.showMessage(msg)
            if i < (n_blinks - 1):
                self.app.processEvents()
                time.sleep(period/2)
                self.statusbar.clearMessage()
                self.app.processEvents()
                time.sleep(period/2)

    def print_error_msg(self, msg, n_blinks=1, period=1):
        self.__print_status_msg(msg, 'red', n_blinks=n_blinks, period=period)

    def print_warning_msg(self, msg, n_blinks=1, period=1):
        self.__print_status_msg(msg, 'yellow', n_blinks=n_blinks, period=period)

    def print_status_msg(self, msg, n_blinks=1, period=1):
        self.__print_status_msg(msg, 'green', n_blinks=n_blinks, period=period)

    def fix_btn_boxes_text(self):
        for btn_box in self.findChildren(QDialogButtonBox):
            if btn_box.property('applyText'):
                btn_box.button(QDialogButtonBox.Apply).setText(btn_box.property('applyText'))
            if btn_box.property('okText'):
                btn_box.button(QDialogButtonBox.Ok).setText(btn_box.property('okText'))
            if btn_box.property('openText'):
                btn_box.button(QDialogButtonBox.Open).setText(btn_box.property('openText'))

    def set_font_size(self, target_font_size=DISPLAY_CONFIG[CURRENT_RES]['font_size']):
        font_sizes = self.__get_font_sizes()
        if len(font_sizes) > 4:  # WARNING: Hacky
            font_sizes = font_sizes[:4]
        small, regular, big, huge = font_sizes
        if target_font_size == regular:
            return

        font_swap = {
            'small': target_font_size - 3,
            'regular': target_font_size,
            'big': target_font_size + 2,
            'huge': target_font_size + 10
        }

        for widget in self.findChildren(QWidget):
            font = widget.property("font")
            try:
                font.setPointSize(font_swap[widget.property('font_size_name')])
            except KeyError:
                print(f'Skipping widget {widget.objectName()}')
            widget.setFont(font)

    def set_font(self):
        for widget in self.findChildren(QWidget):
            font = widget.property("font")
            try:
                font.setFamily(self.preference_editor.params.font_family)  # FIXME: part of child class
            except KeyError:
                print(f'Skipping widget {widget.objectName()}')
            widget.setFont(font)

    def fix_sizes(self):
        # self.set_font_size()
        self.tabWidget.setMinimumWidth(200)
        self.tabWidget.setMinimumHeight(600)

    def fix_styles(self):
        self.fix_btn_boxes_text()
        self.setStyleSheet(BTN_STYLE_SHEET)  # Makes it look qdarkstyle
        # self.fix_btns_stylesheet()
        self.fix_widgets_backgrounds()
        self.fix_sizes()
        self.fix_tootips_stylesheet()
        self.setup_monitoring_bars()  # FIXME: find better location

    def fix_tootips_stylesheet(self):
        for widg in self.findChildren(QWidget):
            if hasattr(widg, 'toolTip') and widg.toolTip():
                widg.setStyleSheet(TOOLTIP_STYLE_SHEET)

    def __get_font_sizes(self):
        point_sizes = set()
        for widg in self.findChildren(QWidget):
            point_sizes.add(widg.property("font").pointSize())
        return sorted(point_sizes)

    def fix_btns_stylesheet(self):
        for btn in self.findChildren(QPushButton):
            btn.setStyleSheet(BTN_STYLE_SHEET)

    def fix_widgets_backgrounds(self):
        for widget_type in (QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit):
            for widget in self.findChildren(widget_type):
                widget.setStyleSheet(f'background-color: {DARK_BACKGROUND}; ')
                if widget_type == QComboBox:
                    widget.setStyleSheet(COMBOBOX_STYLE_SHEET)

    def popup(self, msg, base_msg='Missing configuration file'):
        self.print_warning_msg(html_to_plain_text(msg))
        return warning_popup(base_msg, msg)

    def file_exists(self, f_path):
        if os.path.exists(f_path):
            return True
        else:
            self.print_warning_msg(f'File "{f_path}" not found')
            return False

    def graph_by_name(self, name):
        return [g for g in self.graphs if g.objectName() == name][0]

    def remove_old_plots(self):
        for i in range(self.graphLayout.count(), -1, -1):
            graph = self.graphLayout.takeAt(i)
            if graph is not None:
                widg = graph.widget()
                widg.setParent(None)
                widg.deleteLater()
        self.graphs = []

    def setup_plots(self, dvs, graph_names=None):
        if graph_names is None:
            graph_names = [f'graph_{i}' for i in range(len(dvs))]

        self.remove_old_plots()

        n_rows, n_cols = compute_grid(len(dvs))
        n_spacers = (n_rows * n_cols) - len(dvs)
        for i in range(n_spacers):
            spacer = QWidget(parent=self)
            spacer_name = f'spacer_{i}'
            graph_names.append(spacer_name)
            dvs.append(spacer)
        dock_size = self.dataViewerDockWidgetContents.size()
        for i, dv in enumerate(dvs):
            dv.setObjectName(graph_names[i])
            row = i // n_cols
            col = i % n_cols
            if len(dvs) > 1:
                self.__resize_graph(dv, n_cols, n_rows, dock_size=dock_size)
            self.graphLayout.addWidget(dv, row, col, 1, 1)
            self.graphs.append(dv)
        self.app.processEvents()
        self.dataViewerDockWidgetContents.resize(dock_size)
        self.app.processEvents()

    def __resize_graph(self, dv, n_cols, n_rows, dock_size, margin=9, spacing=6):
        width = math.floor((dock_size.width() - (2*margin) - (n_cols-1)*spacing) / n_cols)
        height = math.floor((dock_size.height() - (2*margin) - (n_rows-1)*spacing) / n_rows)
        dv.resize(width, height)
        dv.setMinimumSize(width, height)  # required to avoid wobbly dv

    def setup_icons(self):
        self._reload_icon = self.style().standardIcon(QStyle.SP_BrowserReload)

    def patch_compound_boxes(self):
        for bx in self.findChildren(QFrame):
            bx_name = bx.objectName().lower()
            if bx_name.startswith('triplet') or bx_name.endswith('let') or \
                    bx_name.endswith('optionallineedit') or bx_name.endswith('optionalplaintextedit'):
                bx.controlsEnabled = types.MethodType(controls_enabled, bx)
                bx.getCheckBox = types.MethodType(get_check_box, bx)
                bx.enableControls = types.MethodType(enable_controls, bx)
                bx.disableControls = types.MethodType(disable_controls, bx)
                if bx_name.startswith('triplet') or bx_name.endswith('let'):  # singlet double triplet
                    bx.getValue = types.MethodType(get_value, bx)
                    bx.setValue = types.MethodType(set_value, bx)
                    bx.valueChangedConnect = types.MethodType(connect_value_changed, bx)
                elif bx_name.endswith('optionallineedit') or bx_name.endswith('optionalplaintextedit'):
                    bx.setText = types.MethodType(set_text, bx)
                    bx.text = types.MethodType(get_text, bx)
                    bx.textChangedConnect = types.MethodType(connect_text_changed, bx)
                else:
                    print(f'Skipping box "{bx_name}", type not recognised')

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

    def patch_font_size_name(self):
        font_names = {
            9: 'small',
            12: 'regular',
            14: 'big',
            22: 'huge'
        }
        for widget in self.findChildren(QWidget):
            font = widget.property('font')
            font_size_name = font_names[font.pointSize()]
            widget.setProperty('font_size_name', font_size_name)

    def monkey_patch(self):
        self.patch_compound_boxes()
        self.patch_button_boxes()
        self.patch_tool_boxes()
        self.patch_font_size_name()
        # self.fix_styles()

    @staticmethod
    def create_missing_file_msg(f_type, f_path, default_path):
        base_msg = f'No {f_type} file found at:<br>  <nobr><em>"{f_path}"</em></nobr>.'
        msg = f'{base_msg} <br><br>Do you want to load a default one from:<br>  <nobr><em>"{default_path}"</em>?</nobr>'
        return base_msg, msg

    def make_progress_dialog(self, title='Processing', n_steps=1, maximum=100, abort=None, parent=None):
        if n_steps:
            n_steps += 1  # To avoid range shrinking because starting from 1 not 0
            nested = True
            self.progress_dialog = make_nested_progress_dialog(title=title, overall_maximum=n_steps,
                                                               abort_callback=abort, parent=parent)
            self.progress_watcher.main_max_changed.connect(self.progress_dialog.mainProgressBar.setMaximum)
            self.progress_watcher.main_progress_changed.connect(self.progress_dialog.mainProgressBar.setValue)
            self.progress_watcher.main_step_name_changed.connect(self.handle_step_name_change)
        else:
            nested = False
            self.progress_dialog = make_simple_progress_dialog(title=title, abort_callback=abort,
                                                               parent=parent)

        self.progress_watcher.max_changed.connect(self.progress_dialog.subProgressBar.setMaximum)
        self.progress_watcher.progress_changed.connect(self.progress_dialog.subProgressBar.setValue)
        self.progress_watcher.sub_step_name_changed.connect(self.handle_sub_step_change)

        self.progress_watcher.finished.connect(self.signal_process_finished)

        if nested:
            self.progress_watcher.setup(main_step_name=title, main_step_length=n_steps)
        else:
            self.progress_watcher.setup(main_step_name='', main_step_length=1, sub_step_length=maximum, pattern=None)

        self.set_tabs_progress_watchers(nested=nested)

    def set_tabs_progress_watchers(self, nested=False):
        raise NotImplementedError()

    def wrap_in_thread(self, func, *args, **kwargs):
        pool = ThreadPool(processes=1)
        result = pool.apply_async(func, args, kwargs)
        while not result.ready():
            result.wait(0.25)
            self.progress_watcher.set_progress(self.progress_watcher.count_dones())
            self.app.processEvents()
        return result.get()

    def signal_process_finished(self, msg='Idle, waiting for input'):
        if not any([kw in msg.lower() for kw in ('idle', 'done', 'finish')]):
            msg += ' finished'
        self.print_status_msg(msg)
        self.log_progress(msg)
        if self.progress_dialog is not None:
            self.progress_dialog.done(1)
            self.progress_dialog = None  # del

    def handle_step_name_change(self, step_name):
        self.log_process_start(step_name)
        self.progress_dialog.mainStepNameLabel.setText(step_name)
        # self.progress_dialog.mainProgressBar.setFormat(f'step %v/%m  ({step_name})')

    def handle_sub_step_change(self, step_name):
        if self.progress_dialog is not None:
            self.progress_dialog.subProgressLabel.setText(step_name)
        self.log_progress(f'    {step_name}')

    def log_process_start(self, msg):
        self.print_status_msg(msg)
        self.log_progress(msg)
        self.save_cfg()

    def log_progress(self, msg):
        self.progress_logger.write(msg)

    def save_cfg(self):
        cfg_folder = os.path.join(self.src_folder, 'config_snapshots', datetime.now().strftime('%y%m%d_%H_%M_%S'))
        os.makedirs(cfg_folder, exist_ok=True)
        for param in self.params:
            if isinstance(param, UiParameterCollection):
                cfg = param.config
            elif isinstance(param, UiParameter):
                cfg = param._config
            else:
                continue
            cfg_f_name = os.path.basename(cfg.filename)
            with open(os.path.join(cfg_folder, cfg_f_name), 'wb') as file_obj:
                cfg.write(outfile=file_obj)

    def setup_monitoring_bars(self):
        for label, bar in zip(('CPU', 'RAM', 'GPU', 'VRAM'), (self.cpu_bar, self.ram_bar, self.gpu_bar, self.vram_bar)):
            self.statusbar.addPermanentWidget(QLabel(label))
            bar.setOrientation(Qt.Vertical)
            bar.setMaximumHeight(40)
            self.statusbar.addPermanentWidget(bar)
            # cpu_bar.setValue(20)

    def update_monitoring_bars(self):
        self.cpu_bar.setValue(psutil.cpu_percent())
        self.ram_bar.setValue(psutil.virtual_memory().percent)
        self.gpu_bar.setValue(gpu_util())
        self.vram_bar.setValue(get_percent_v_ram_use())


class ClearMapGui(ClearMapGuiBase):
    def __init__(self):
        super().__init__()
        self.config_loader = ConfigLoader('')
        self.ortho_viewer = OrthoViewer()

        self.sample_tab_mgr = SampleTab(self, tab_idx=0)
        self.alignment_tab_mgr = AlignmentTab(self, tab_idx=1)
        self.cells_tab_mgr = CellCounterTab(self, tab_idx=2)
        self.vasculature_tab_mgr = VasculatureTab(self, tab_idx=3)
        self.batch_tab_mgr = BatchTab(self, tab_idx=4)

        self.preference_editor = PreferenceUi(self)

        self.sample_tab_mgr.mini_brain_scaling, self.sample_tab_mgr.mini_brain = setup_mini_brain()

        self.setWindowIcon(QtGui.QIcon(os.path.join(ICONS_FOLDER, 'logo_cyber.png')))

        self.setupUi(self)
        self.amend_ui()

        self.actionPreferences.triggered.connect(self.preference_editor.open)

        self.app = QApplication.instance()

    def __len__(self):
        return len(self.tab_mgrs)

    def __getitem__(self, item):
        return self.tab_mgrs[item]

    @property
    def tab_mgrs(self):
        tabs = [self.sample_tab_mgr, self.alignment_tab_mgr]
        if self.cells_tab_mgr.ui is None or self.cells_tab_mgr.ui.isEnabled():
            tabs.append(self.cells_tab_mgr)
        if self.vasculature_tab_mgr.ui is None or self.vasculature_tab_mgr.ui.isEnabled():
            tabs.append(self.vasculature_tab_mgr)
        # #self.batch_tab_mgr
        return tabs

    @property
    def params(self):
        return [tab.params for tab in self.tab_mgrs]

    def patch_stdout(self):
        sys.stdout = self.logger
        sys.stderr = self.error_logger

    def amend_ui(self):
        self.logger = Printer()
        self.logger.text_updated.connect(self.textBrowser.append)
        self.error_logger = Printer(color='red', logger_type='error')
        self.error_logger.text_updated.connect(self.textBrowser.append)

        self.setup_icons()
        self.setup_tabs()
        self.preference_editor.setup(self.config_loader.get_cfg('display')[CURRENT_RES]['font_size'])

        self.monkey_patch()

        self.logoLabel.setPixmap(QtGui.QPixmap(os.path.join(ICONS_FOLDER, 'logo_cyber.png')))

        self.graphLayout.removeWidget(self.frame)

        self.print_status_msg('Idle, waiting for input')

    def setup_tabs(self):
        for tab in self.tab_mgrs:
            tab.setup()
        self.tabWidget.tabBarClicked.connect(self.handle_tab_click)
        self.tabWidget.setCurrentIndex(0)

    def reload_prefs(self):
        self.set_font_size(self.preference_editor.params.font_size)

    def set_tabs_progress_watchers(self, nested=False):
        if nested:
            for tab in self.tab_mgrs:
                tab.set_progress_watcher(self.progress_watcher)
        else:
            self.alignment_tab_mgr.preprocessor.set_progress_watcher(self.progress_watcher)

    def handle_tab_click(self, tab_index):
        all_tabs = [self.sample_tab_mgr, self.alignment_tab_mgr, self.cells_tab_mgr, self.vasculature_tab_mgr, self.batch_tab_mgr]
        if 0 < tab_index < 4 and self.alignment_tab_mgr.preprocessor.workspace is None:
            self.popup('WARNING', 'Workspace not initialised, '
                                  'cannot proceed to alignment')
            self.tabWidget.setCurrentIndex(0)
        processor_setup_functions = {
            2: self.cells_tab_mgr.setup_cell_detector,
            3: self.vasculature_tab_mgr.setup_vessel_processors
        }
        if tab_index in (2, 3):
            if all_tabs[tab_index] is None or not all_tabs[tab_index].ui.isEnabled():
                return
            if self.alignment_tab_mgr.preprocessor.was_registered:
                processor_setup_functions[tab_index]()
            else:
                self.__check_missing_alignment()
        elif tab_index == 4 and not self.batch_tab_mgr.initialised:
            cfg_name = title_to_snake(self.batch_tab_mgr.name)
            try:
                self.batch_tab_mgr.setup()
                self.batch_tab_mgr.set_params()
                results_folder = get_directory_dlg(self.preference_editor.params.start_folder,
                                                   'Select the folder where results will be written')
                was_copied, cfg_path = self.__get_cfg_path(cfg_name, ConfigLoader(results_folder))
                if was_copied:
                    self.batch_tab_mgr.params.fix_cfg_file(cfg_path)
                self.logger.set_file(os.path.join(results_folder, 'info.log'))  # WARNING: set logs to global results folder
                self.progress_watcher.log_path = self.logger.file.name
                self.error_logger.set_file(os.path.join(results_folder, 'errors.html'))
                self.progress_watcher.log_path = self.logger.file.name
                self.batch_tab_mgr.params.get_config(cfg_path)  # FIXME: try to put with other tabs init (difference with config_loader)
                self.batch_tab_mgr.initial_cfg_load()
                self.batch_tab_mgr.params.results_folder = results_folder
                self.batch_tab_mgr.params.ui_to_cfg()
                self.batch_tab_mgr.setup_workers()
            except ConfigNotFoundError:
                self.conf_load_error_msg(cfg_name)
            except FileNotFoundError:  # message already printed, just stop
                return

    def __check_missing_alignment(self):
        ok = self.__warn_missing_alignment()  # TODO: use result
        self.tabWidget.setCurrentIndex(1)  # WARNING: does not work

    def __warn_missing_alignment(self):
        return self.popup('WARNING', 'Alignment not performed, please run first') == QMessageBox.Ok

    def conf_load_error_msg(self, conf_name):
        conf_name = conf_name.replace('_', ' ').title()
        self.print_error_msg(f'Loading {conf_name} config file failed')

    def assert_src_folder_set(self):
        if not self.src_folder:
            msg = 'Missing source folder, please define first'
            self.print_error_msg(msg)
            raise FileNotFoundError(msg)

    def __get_cfg_path(self, cfg_name, config_loader=None):
        if config_loader is None:
            config_loader = self.config_loader
        cfg_path = config_loader.get_cfg_path(cfg_name, must_exist=False)
        was_copied = False
        if cfg_name in ('cell_map', 'vasculature', 'tube_map'):
            pipeline_name = title_to_snake(self.alignment_tab_mgr.params.pipeline_name)
            is_cell_map = pipeline_name == 'cell_map' and cfg_name == 'cell_map'
            is_tube_map = pipeline_name == 'tube_map' and cfg_name in ('tube_map', 'vasculature')
            is_irrelevant_tab = not(pipeline_name == 'both' or is_tube_map or is_cell_map)
            if is_irrelevant_tab:
                return False, None
        if not self.file_exists(cfg_path):
            try:
                default_cfg_file_path = config_loader.get_default_path(cfg_name)
            except FileNotFoundError as err:
                self.print_error_msg(f'Could not locate file for "{cfg_name}"')
                raise err
            base_msg, msg = self.create_missing_file_msg(cfg_name.title().replace('_', ''),
                                                         cfg_path, default_cfg_file_path)
            do_copy = self.popup(msg) == QMessageBox.Ok
            if do_copy:
                if not os.path.exists(os.path.dirname(cfg_path)):
                    os.mkdir(os.path.dirname(cfg_path))
                copyfile(default_cfg_file_path, cfg_path)
                was_copied = True
            else:
                self.error_logger.write(self.error_logger.colourise(base_msg, force=True))
                raise FileNotFoundError(html_to_ansi(base_msg))
        return was_copied, cfg_path

    def parse_cfg(self):
        self.print_status_msg('Parsing configuration')
        self.assert_src_folder_set()

        error = False
        for tab in self.tab_mgrs:
            cfg_name = title_to_snake(tab.name)
            try:
                loaded_from_defaults, cfg_path = self.__get_cfg_path(cfg_name)
                if cfg_path is None:  # skipped
                    tab.ui.setEnabled(False)
                    continue

                if tab.processing_type is None or tab.processing_type == 'batch':
                    tab.set_params()
                elif tab.processing_type == 'pre':
                    tab.set_params(self.sample_tab_mgr.params)
                elif tab.processing_type == 'post':
                    tab.set_params(self.sample_tab_mgr.params, self.alignment_tab_mgr.params)
                    tab.setup_preproc(self.alignment_tab_mgr.preprocessor)
                else:
                    raise ValueError(f'Processing type should be one of "pre", "post", "batch" or None,'
                                     f' got "{tab.processing_type}"')
                tab.params.get_config(cfg_path)
                # patch config if loaded from defaults or sample ID if it was set
                if loaded_from_defaults or (cfg_name == 'sample' and self.sample_tab_mgr.get_sample_id()):
                    tab.params.fix_cfg_file(cfg_path)  # TODO: see if this should be moved
                tab.load_params()
                tab.setup_workers()
            except ConfigNotFoundError:
                self.conf_load_error_msg(cfg_name)
                error = True
            except FileNotFoundError:  # message already printed, just stop
                return

        if not error:
            self.print_status_msg('Config loaded')
            for tab in self.tab_mgrs:
                tab.initial_cfg_load()

    def set_src_folder(self):
        self.src_folder = get_directory_dlg(self.preference_editor.params.start_folder)
        self.config_loader.src_dir = self.src_folder
        self.fix_styles()
        cfg_path = self.config_loader.get_cfg_path('sample', must_exist=False)
        if self.file_exists(cfg_path):
            cfg = self.config_loader.get_cfg_from_path(cfg_path)
            sample_id = cfg['sample_id']
            if sample_id == 'undefined':
                sample_id = ''
        else:
            sample_id = ''
        self.sample_tab_mgr.display_sample_id(sample_id)

    @property
    def src_folder(self):
        return self.sample_tab_mgr.src_folder

    @src_folder.setter
    def src_folder(self, src_folder):
        self.logger.set_file(os.path.join(src_folder, 'info.log'))
        self.progress_watcher.log_path = self.logger.file.name
        self.error_logger.set_file(os.path.join(src_folder, 'errors.html'))
        self.progress_logger.set_file(os.path.join(src_folder, 'progress.log'))
        self.sample_tab_mgr.src_folder = src_folder


def create_main_window(app):
    clearmap_main_win = ClearMapGui()
    if clearmap_main_win.preference_editor.params.start_full_screen:
        clearmap_main_win.showMaximized()  # TODO: check if redundant with show
    return clearmap_main_win


def main(app, splash):
    clearmap_main_win = create_main_window(app)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    def except_hook(exc_type, exc_value, exc_tb):
        lexer = PythonTracebackLexer()
        default_style = 'native'
        style = 'nord-darker' if 'nord-darker' in pygments.styles.get_all_styles() else default_style
        formatter = HtmlFormatter(full=True, style=style, lineos='table', wrapcode=True, noclasses=True)
        formatter.style.background_color = DarkPalette.COLOR_BACKGROUND_1
        raw_traceback = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        formatted_traceback = pygments.highlight(raw_traceback, lexer, formatter)
        clearmap_main_win.error_logger.write(formatted_traceback)
        if isinstance(exc_type(), Warning):
            clearmap_main_win.error_logger.write(f'<strong><p style="color:{WARNING_YELLOW}">'
                                                 f'THIS IS A WARNING AND CAN NORMALLY BE SAFELY IGNORED</p></strong>')

    clearmap_main_win.show()
    clearmap_main_win.fix_styles()
    clearmap_main_win.timer.start()
    splash.finish(clearmap_main_win)
    if clearmap_main_win.preference_editor.params.verbosity != 'trace':  # WARNING: will disable progress bars
        clearmap_main_win.patch_stdout()
        sys.excepthook = except_hook
    sys.exit(app.exec())


def entry_point():
    main(app, splash)


if __name__ == "__main__":
    main(app, splash)
