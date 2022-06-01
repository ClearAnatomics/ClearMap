import os
import sys

from multiprocessing.pool import ThreadPool
from shutil import copyfile
import traceback
import types

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QSpinBox, QDoubleSpinBox, QFrame, \
    QDialogButtonBox, QComboBox, QLineEdit, QStyle, QWidget, QMessageBox, QToolBox

os.environ['CLEARMAP_GUI_HOSTED'] = "1"
# ########################################### SPLASH SCREEN ###########################################################
from ClearMap.gui.dialogs import make_splash, update_pbar

# To show splash before slow imports
ICONS_FOLDER = 'ClearMap/gui/icons/'   # REFACTOR: use qrc

app = QApplication([])

from ClearMap.gui.gui_utils import get_current_res, UI_FOLDER

CURRENT_RES = get_current_res(app)

splash, progress_bar = make_splash(res=CURRENT_RES)
splash.show()
update_pbar(app, progress_bar, 10)

# ############################################  SLOW IMPORTS #########################################################

import pygments
from pygments.lexers.python import PythonTracebackLexer  # noqa
from pygments.formatters.html import HtmlFormatter

import qdarkstyle

import pyqtgraph as pg

update_pbar(app, progress_bar, 20)
from ClearMap.Utils.utilities import title_to_snake
from ClearMap.gui.gui_logging import Printer
from ClearMap.config.config_loader import ConfigLoader
from ClearMap.gui.params import ConfigNotFoundError
from ClearMap.gui.widget_monkeypatch_callbacks import get_value, set_value, controls_enabled, get_check_box, \
    enable_controls, disable_controls, set_text, get_text, connect_apply, connect_close, connect_save, connect_open, \
    connect_ok, connect_cancel, connect_value_changed, connect_text_changed
update_pbar(app, progress_bar, 40)
from ClearMap.gui.pyuic_utils import loadUiType
from ClearMap.gui.dialogs import get_directory_dlg, warning_popup, make_progress_dialog, make_nested_progress_dialog, DISPLAY_CONFIG
from ClearMap.gui.gui_utils import html_to_ansi, html_to_plain_text, compute_grid
from ClearMap.gui.style import QDARKSTYLE_BACKGROUND, DARK_BACKGROUND, PLOT_3D_BG, BTN_STYLE_SHEET

from ClearMap.gui.widgets import OrthoViewer, PbarWatcher, setup_mini_brain  # needs plot_3d
update_pbar(app, progress_bar, 60)
from ClearMap.gui.tabs import SampleTab, AlignmentTab, CellCounterTab, VasculatureTab, PreferenceUi

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

Ui_ClearMapGui, _ = loadUiType(os.path.join(UI_FOLDER, 'creator/mainwindow.ui'), patch_parent_class=False)


class ClearMapGuiBase(QMainWindow, Ui_ClearMapGui):
    def __init__(self):
        super().__init__()
        self.graph_names = {}
        self._reload_icon = self.style().standardIcon(QStyle.SP_BrowserReload)
        self.logger = None
        self.error_logger = None
        self.progress_dialog = None

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
            if btn_box.property('okText'):
                btn_box.button(QDialogButtonBox.Ok).setText(btn_box.property('okText'))
            if btn_box.property('openText'):
                btn_box.button(QDialogButtonBox.Open).setText(btn_box.property('openText'))

    def set_font_size(self, target_font_size=DISPLAY_CONFIG[CURRENT_RES]['font_size']):
        font_sizes = self.__get_font_sizes()
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
                print('Skipping widget {}'.format(widget.objectName()))
            widget.setFont(font)

    def fix_sizes(self):
        # self.set_font_size()
        self.tabWidget.setMinimumWidth(200)
        self.tabWidget.setMinimumHeight(600)

    def fix_styles(self):
        self.fix_btn_boxes_text()
        self.fix_btns_stylesheet()
        self.fix_widgets_backgrounds()
        self.fix_sizes()

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
        self.remove_old_plots()

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
        return sorted([getattr(self, attr) for attr in dir(self) if attr.startswith('graph_')])

    def resize_graphs(self):
        n_rows, n_cols = compute_grid(len(self.get_graphs()))  # WARNING: take care of placeholders
        for i in range(self.graphLayout.count(), -1, -1):  # Necessary to count backwards to get all graphs
            graph = self.graphLayout.itemAt(i)
            if graph is not None:
                widg = graph.widget()
                self.__resize_graph(widg, n_cols, n_rows)

    def __resize_graph(self, dv, n_cols, n_rows, margin=20):
        size = round((self.graphDock.width() - margin) / n_cols), round((self.graphDock.height() - margin) / n_rows)
        dv.resize(*size)
        dv.setMinimumSize(*size)  # required to avoid wobbly dv
        # dv.setMaximumSize(*size)

    def remove_old_plots(self):
        for i in range(self.graphLayout.count(), -1, -1):
            graph = self.graphLayout.takeAt(i)
            if graph is not None:
                widg = graph.widget()
                widg.setParent(None)
                widg.deleteLater()
                delattr(self, widg.objectName())
        self.graph_names = {}

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
        self.set_tabs_progress_watchers(nested=False)

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

        self.set_tabs_progress_watchers(nested=True)

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


class ClearMapGui(ClearMapGuiBase):
    def __init__(self):
        super().__init__()
        self.config_loader = ConfigLoader('')
        self.ortho_viewer = OrthoViewer()

        self.sample_tab_mgr = SampleTab(self, tab_idx=0)
        self.alignment_tab_mgr = AlignmentTab(self, tab_idx=1)
        self.cells_tab_mgr = CellCounterTab(self, tab_idx=2)
        self.vasculature_tab_mgr = VasculatureTab(self, tab_idx=3)

        self.preference_editor = PreferenceUi(self)

        self.sample_tab_mgr.mini_brain_scaling, self.sample_tab_mgr.mini_brain = setup_mini_brain()

        self.setWindowIcon(QtGui.QIcon(os.path.join(ICONS_FOLDER, 'logo_cyber.png')))

        self.setupUi(self)
        self.amend_ui()

        self.actionPreferences.triggered.connect(self.preference_editor.open)

        self.progress_watcher = PbarWatcher()
        self.app = QApplication.instance()

    def __len__(self):
        return len(self.tab_mgrs)

    def __getitem__(self, item):
        return self.tab_mgrs[item]

    @property
    def tab_mgrs(self):
        return self.sample_tab_mgr, self.alignment_tab_mgr, self.cells_tab_mgr, self.vasculature_tab_mgr

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
        if tab_index > 0 and self.alignment_tab_mgr.preprocessor.workspace is None:
            self.popup('WARNING', 'Workspace not initialised, '
                                  'cannot proceed to alignment')
            self.tabWidget.setCurrentIndex(0)
        processor_setup_functions = {
            2: self.cells_tab_mgr.setup_cell_detector,
            3: self.vasculature_tab_mgr.setup_vessel_processors
        }
        if tab_index in (2, 3):
            if self.alignment_tab_mgr.preprocessor.was_registered:
                processor_setup_functions[tab_index]()
            else:
                self.__check_missing_alignment()

    def __check_missing_alignment(self):
        ok = self.__warn_missing_alignment()  # TODO: use result
        self.tabWidget.setCurrentIndex(1)  # WARNING: does not work

    def __warn_missing_alignment(self):
        return self.popup('WARNING', 'Alignment not performed, please run first') == QMessageBox.Ok

    def conf_load_error_msg(self, conf_name):
        conf_name = conf_name.replace('_', ' ').title()
        self.print_error_msg('Loading {} config file failed'.format(conf_name))

    def assert_src_folder_set(self):
        if not self.src_folder:
            msg = 'Missing source folder, please define first'
            self.print_error_msg(msg)
            raise FileNotFoundError(msg)

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
                formatted_msg = html_to_ansi(base_msg)
                self.error_logger.write(str(formatted_msg))
                raise FileNotFoundError(formatted_msg)
        return was_copied, cfg_path

    def parse_cfg(self):
        self.print_status_msg('Parsing configuration')
        self.assert_src_folder_set()

        error = False
        for tab in self.tab_mgrs:
            cfg_name = title_to_snake(tab.name)
            try:
                was_copied, cfg_path = self.__get_cfg_path(cfg_name)
                if was_copied:
                    tab.params.fix_cfg_file(cfg_path)

                if tab.processing_type is None:
                    tab.set_params()
                elif tab.processing_type == 'pre':
                    tab.set_params(self.sample_tab_mgr.params)
                elif tab.processing_type == 'post':
                    tab.set_params(self.sample_tab_mgr.params, self.alignment_tab_mgr.params)
                    tab.setup_preproc(self.alignment_tab_mgr.preprocessor)
                else:
                    raise ValueError('Processing type should be one of "pre", "post" or None, got "{}"'
                                     .format(tab.processing_type))
                tab.params.get_config(cfg_path)
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

    @property
    def src_folder(self):
        return self.sample_tab_mgr.src_folder

    @src_folder.setter
    def src_folder(self, src_folder):
        self.logger.set_file(os.path.join(src_folder, 'info.log'))
        self.progress_watcher.log_path = self.logger.file.name
        self.error_logger.set_file(os.path.join(src_folder, 'errors.html'))
        self.sample_tab_mgr.src_folder = src_folder


def create_main_window(app):
    clearmap_main_win = ClearMapGui()
    if clearmap_main_win.preference_editor.params.start_full_screen:
        clearmap_main_win.showMaximized()  # TODO: check if redundant with show
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    return clearmap_main_win


def main(app, splash):
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
    if clearmap_main_win.preference_editor.params.verbosity != 'trace':  # WARNING: will disable progress bars
        clearmap_main_win.patch_stdout()
        sys.excepthook = except_hook
    sys.exit(app.exec())


if __name__ == "__main__":
    main(app, splash)
