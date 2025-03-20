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
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

import math
import os
import sys
import tempfile
import warnings
from datetime import datetime

from multiprocessing.pool import ThreadPool
from pathlib import Path
from shutil import copyfile
from statistics import mode

from importlib_metadata import version

# WARNING: Necessary for QCoreApplication creation
from PyQt5.QtWebEngineWidgets import QWebEngineView  # noqa: F401

from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
                             QLineEdit, QMessageBox, QToolBox, QProgressBar, QLabel,
                             QStyle, QAction)

import qdarkstyle
from qdarkstyle import DarkPalette

HARD_DEFAULT_FONT_SIZE = 11

DEBUG = False


os.environ['CLEARMAP_GUI_HOSTED'] = "1"
# ########################################### SPLASH SCREEN ###########################################################
CLEARMAP_VERSION = version('ClearMap')
from ClearMap.config.config_loader import ConfigLoader, CLEARMAP_CFG_DIR
machine_cfg = ConfigLoader.get_cfg_from_path(ConfigLoader.get_default_path('machine'))
tmp_folder = machine_cfg.get('tmp_folder', None)
if tmp_folder is not None:
    for var_name in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ[var_name] = tmp_folder
    tempfile.tempdir = None  # Force refresh of tempdir
DEBUG = machine_cfg['verbosity'] in ('trace', 'debug')

from ClearMap.Alignment.Stitching import layout_graph_utils  # noqa: F401 # WARNING: first because otherwise, breaks with pytorch

from ClearMap.gui.about import AboutInfo
from ClearMap.gui.dialogs import make_splash, update_pbar, make_simple_progress_dialog, option_dialog, input_dialog, \
    AboutDialog

# To show splash before slow imports
ICONS_FOLDER = 'ClearMap/gui/creator/icons/'   # REFACTOR: use qrc

app = QApplication([])
# Attempt to force appearance
app.setApplicationName('ClearMap')
app.setApplicationDisplayName('ClearMap')
app.setApplicationVersion('2.1')
palette = app.palette()  # WARNING: necessary because QWhatsThis does not follow stylesheets
palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(DarkPalette.COLOR_BACKGROUND_2))
palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(DarkPalette.COLOR_TEXT_2))
app.setPalette(palette)  # noqa


from ClearMap.config.update_config import update_default_config
from ClearMap.gui.widget_monkeypatch_callbacks import recursive_patch_widgets
from ClearMap.gui.gui_utils import get_current_res, UI_FOLDER, clear_layout, delete_widget

CURRENT_RES = get_current_res(app)

splash, progress_bar = make_splash(res=CURRENT_RES)
splash.show()
update_pbar(app, progress_bar, 10)

# ############################################  SLOW IMPORTS #########################################################

import pyqtgraph as pg

import torch

update_pbar(app, progress_bar, 20)
from ClearMap.Utils.utilities import title_to_snake, snake_to_title
from ClearMap.gui.gui_logging import Printer
from ClearMap.Utils.exceptions import ConfigNotFoundError, ClearMapIoException
from ClearMap.gui.params_interfaces import UiParameter, UiParameterCollection

update_pbar(app, progress_bar, 40)
from ClearMap.gui.pyuic_utils import loadUiType
from ClearMap.gui.dialogs import get_directory_dlg, warning_popup, make_nested_progress_dialog, DISPLAY_CONFIG
from ClearMap.gui.gui_utils import html_to_ansi, html_to_plain_text, compute_grid
from ClearMap.gui.style import (DARK_BACKGROUND, PLOT_3D_BG, BTN_STYLE_SHEET,
                                TOOLTIP_STYLE_SHEET, COMBOBOX_STYLE_SHEET,
                                HIGHLIGHTED_BTN_STYLE)

# Widgets import is quite slow
from ClearMap.gui.widgets import (OrthoViewer, ProgressWatcher,
                                  StructureSelector, PerfMonitor, ManipulateAssetsDialog)  # Perfmonitor needs plot_3d
update_pbar(app, progress_bar, 60)
from ClearMap.gui.tabs import (SampleInfoTab, StitchingTab, RegistrationTab, CellCounterTab,
                               VasculatureTab, GroupAnalysisTab, BatchProcessingTab, TractMapTab)
from ClearMap.gui.interfaces import BatchTab
from ClearMap.gui.preferences import PreferenceUi
from ClearMap.processors.sample_preparation import SampleManager

update_pbar(app, progress_bar, 80)

pg.setConfigOption('background', PLOT_3D_BG)


DATA_TYPE_TO_TAB_CLASS = {  # WARNING: not all data types are covered
    'nuclei': CellCounterTab,
    'cells': CellCounterTab,
    'vessels': VasculatureTab,
    'veins': VasculatureTab,
    'arteries': VasculatureTab,
    'myelin': TractMapTab,
}

# TODO
"""
Handle reset detected correctly
Fix qrc resources (ui files should not be coded by path)
Delete intermediate files
Ensure all machine config params are in the preferences UI

    
Analysis:
    
LATER:
Auto modes:
    - Run all
"""

Ui_ClearMapGui, _ = loadUiType(os.path.join(UI_FOLDER, 'creator', 'mainwindow.ui'), from_imports=True,
                               import_from='ClearMap.gui.creator', patch_parent_class=False)


class ClearMapGuiBase(QMainWindow, Ui_ClearMapGui):
    """
    The interface for the ClearMapGui class

    It deals with the basics of logic of the main window, the plotting
    progress bars and performance monitors and appearance.
    Anything more application related is coded in the child class.
    """
    def __init__(self):
        super().__init__()
        self.graphs = []
        self._reload_icon = self.style().standardIcon(QStyle.SP_BrowserReload)
        self.logger = None
        self.progress_logger = Printer(logger_type='progress')
        self.error_logger = None
        self.progress_dialog = None
        self.progress_watcher = ProgressWatcher()

        self.default_font_size = self.compute_default_font_size()

        self.cpu_bar = QProgressBar()  # noqa
        self.single_thread_bar = QProgressBar() # noqa
        self.ram_bar = QProgressBar() # noqa
        self.gpu_bar = QProgressBar() # noqa
        self.vram_bar = QProgressBar() # noqa

        if torch.cuda.is_available():
            gpu_period = 500
        else:
            gpu_period = None
        self.perf_monitor = PerfMonitor(self, 500, gpu_period)
        self.perf_monitor.cpu_vals_changed.connect(self.update_cpu_bars)
        if torch.cuda.is_available():
            self.perf_monitor.gpu_vals_changed.connect(self.update_gpu_bars)
        self.perf_monitor.start()

    def find_child_by_name(self, child_name, child_type, parent=None):
        """
        Find children in the window or any other widget

        Parameters
        ----------
        child_name : str
            The name of the widget we are looking for
        child_type : str
            The type of widget to search
        parent : QWidget
            The parent widget to start from. If none given, will use the main window.

        Returns
        -------
        QWidget
            The child widget matching the name and type
        """
        if parent is None:
            parent = self
        for child in parent.findChildren(child_type):
            if child.objectName() == child_name:
                return child

    def __print_status_msg(self, msg, color):
        self.statusbar.setStyleSheet(f'color: {color}')
        self.statusbar.showMessage(msg)

    def print_error_msg(self, msg):
        """
        Print a message in red in the statusbar

        Parameters
        ----------
        msg : str
            The message to be printed
        """
        if isinstance(msg, Exception):
            msg = str(msg)
        self.__print_status_msg(msg, 'red')

    def print_warning_msg(self, msg):
        """
        Print a message in yellow in the statusbar

        Parameters
        ----------
        msg : str
            The message to be printed
        """
        self.__print_status_msg(msg, 'yellow')

    def print_status_msg(self, msg):
        """
        Print a message in green in the statusbar

        Parameters
        ----------
        msg : str
            The message to be printed
        """
        self.__print_status_msg(msg, 'green')

    def set_font_size(self, target_font_size=DISPLAY_CONFIG[CURRENT_RES]['font_size']):
        """
        Set the font sizes of the widgets based on *target_font_size*. All fonts will
        be shifted based on that size. This should preserve the size relationships of the fonts
        in the program. It is assumed that the program uses 4 font sizes:
            - small, regular, big and huge
        This will shift font sizes so that the supplied size corresponds to the new regular size.

        Parameters
        ----------
        target_font_size : int
            The main font size
        """
        self.default_font_size = self.compute_default_font_size()  # or self.compute_default_font_size_mean()
        min_font_size = 1  # Define a minimum font size to avoid negative or zero font sizes

        scaling_factor = target_font_size / self.default_font_size

        for widget in self.findChildren(QWidget):
            font = widget.font()  #  font = widget.property("font")
            current_size = font.pointSize()
            if current_size > 0:  # Ensure the font size is valid
                # Calculate the new font size relative to the target font size
                new_size = int(current_size * scaling_factor)
                # Ensure the new font size is not less than the minimum font size
                font.setPointSize(max(new_size, min_font_size))
                widget.setFont(font)

    def set_font(self):
        for widget in self.findChildren(QWidget):
            font = widget.property("font")
            try:
                font.setFamily(self.preference_editor.params.font_family)  # REFACTOR: part of child class
            except KeyError:
                print(f'Skipping widget {widget.objectName()}')
            widget.setFont(font)

    def compute_default_font_size(self):
        """
        Gets the most represented font size in the program. This is the default font size

        Returns
        -------
        int
            The default font size
        """
        font_sizes = [widget.font().pointSize() for widget in self.findChildren(QWidget) if widget.font().pointSize() > 0]
        if font_sizes:
            return mode(font_sizes)
        return HARD_DEFAULT_FONT_SIZE  # Fallback to a default value if no valid font sizes are found

    def fix_sizes(self):
        # self.set_font_size()
        self.tabWidget.setMinimumWidth(200)
        self.tabWidget.setMinimumHeight(600)

    def fix_styles(self):
        """The meta function calling all the style functions"""
        self.setStyleSheet(BTN_STYLE_SHEET)  # Makes it look qdarkstyle
        # self.fix_btns_stylesheet()
        self.fix_widgets_backgrounds()
        self.fix_sizes()
        self.fix_tooltips_stylesheet()

        btn = self.tab_managers['sample_info'].ui.launchPatternWizzardPushButton
        btn.setStyleSheet(HIGHLIGHTED_BTN_STYLE)

    def fix_tooltips_stylesheet(self):
        for widg in self.findChildren(QWidget):
            if hasattr(widg, 'toolTip') and widg.toolTip():
                widg.setStyleSheet(TOOLTIP_STYLE_SHEET)

    def fix_btns_stylesheet(self):
        for btn in self.findChildren(QPushButton):
            btn.setStyleSheet(BTN_STYLE_SHEET)

    def fix_widgets_backgrounds(self):
        for widget_type in (QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit):
            for widget in self.findChildren(widget_type):
                widget.setStyleSheet(f'background-color: {DARK_BACKGROUND}; ')
                if widget_type == QComboBox:
                    widget.setStyleSheet(COMBOBOX_STYLE_SHEET)

    def popup(self, msg, base_msg='Missing configuration file', print_warning=True):
        """
        Display a warning popup message which the user has to dismiss by clicking OK
        Optionally, a warning is also printed to the logs

        Parameters
        ----------
        msg : str
            The more detailed text to display
        base_msg : str
            The quick description of the problem
        print_warning : bool
            Whether to also print msg to logs

        Returns
        -------
        QMessageBox
            The popup message box
        """
        if print_warning:
            self.print_warning_msg(html_to_plain_text(msg))
        return warning_popup(base_msg, msg)

    def file_exists(self, f_path):
        if os.path.exists(f_path):
            return True
        else:
            self.print_warning_msg(f'File "{f_path}" not found')
            return False

    def graph_by_name(self, name):
        """
        Search self.graphs for name

        Parameters
        ----------
        name : str
            The objectName of the graph widget we are looking for
        """
        return [g for g in self.graphs if g.objectName() == name][0]

    def clear_plots(self):
        """
        Remove all plots currently displayed in the DataViewer area of the interface
        The underlying widgets are also scheduled for garbage collection
        """
        clear_layout(self.graphLayout)
        self.graphs = []
        self.perf_monitor.start()

    def setup_plots(self, dvs, graph_names=None):
        """
        Set the plots provided as argument in the DataViewer area of the UI in a grid format.
        The items in the list have to be derived from QWidget

        Parameters
        ----------
        dvs : List[DataViewer]
            The list of DataViewer (or any other QWidget derived object) to add
            to the display grid
        graph_names: None or List[str]
            Optional The names attached to the graphs for later reference
        """
        if graph_names is None:
            graph_names = [f'graph_{i}' for i in range(len(dvs))]

        self.clear_plots()

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
        """
        Patch the icons of the program
        """
        self._reload_icon = self.style().standardIcon(QStyle.SP_BrowserReload)

    def patch_tool_boxes(self):
        """
        Sets all toolboxes to the first item on startup regardless of the last closed state o
        f the ui file in QtCreator
        """
        for tb in self.findChildren(QToolBox):
            tb.setCurrentIndex(0)

    def monkey_patch(self):
        recursive_patch_widgets(self)
        self.patch_tool_boxes()
        # self.fix_styles()

    @staticmethod
    def create_missing_file_msg(f_type, f_path, default_path):
        base_msg = f'No {f_type} file found at:<br>  <nobr><em>"{f_path}"</em></nobr>.'
        msg = f'{base_msg} <br><br>Do you want to load a default one from:<br>  <nobr><em>"{default_path}"</em>?</nobr>'
        return base_msg, msg

    def make_progress_dialog(self, title='Processing', n_steps=1, maximum=100, abort=None, parent=None):
        """
        Create a single or nested (2 nested bars) progress dialog
        The dialog is initialised with the parameters below and linked to self.progress_watcher
        to drive updates of the titles and progress bar.

        Parameters
        ----------
        title : str
            The title of the dialog
        n_steps : int
            The number of steps of the main step for nested dialogs
        maximum : int
            the maximum value of the progress bar (or the second bar for nested dialogs)
        abort : function
            The function to trigger when the abort button is clicked
        parent : QWidget
            The parent widget to the dialog
        """
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
        """
        Wraps the function in a thread to ensure that the GUI remains responsive
        In general all processing done in the GUI should go through this method

        Parameters
        ----------
        func : function
            The function to wrap
        args :
            The arguments to the function func
        kwargs :
            The keyword arguments to forward to func

        Returns
        -------
            The result of the execution of the function func
        """
        with ThreadPool(processes=1) as pool:
            result = pool.apply_async(func, args, kwargs)
            while not result.ready():
                result.wait(0.25)
                self.progress_watcher.set_progress(self.progress_watcher.count_dones())
                self.app.processEvents()
        return result.get()

    def signal_process_finished(self, msg='Idle, waiting for input'):
        """
        Handles the end of the current computation with a message to be printed
        and logged. It will also notify and close the progress dialog window
        if there is one

        Parameters
        ----------
        msg : str
            The message to display
        """
        if not any([kw in msg.lower() for kw in ('idle', 'done', 'finish')]):
            msg += ' finished'
        self.print_status_msg(msg)
        self.log_progress(msg)
        if self.progress_dialog is not None:
            self.progress_dialog.done(1)
            self.progress_dialog = None  # del

    def handle_step_name_change(self, step_name):
        """
        Handle a change of the name of the current computation step by registering
        the new step and updating the progress dialog

        Parameters
        ----------
        step_name : str
            The new step name
        """
        self.log_process_start(step_name)
        try:
            self.progress_dialog.mainStepNameLabel.setText(step_name)
        except AttributeError as err:  # FIXME: find out why might be missing
            self.error_logger.write(str(err))
        # self.progress_dialog.mainProgressBar.setFormat(f'step %v/%m  ({step_name})')

    def handle_sub_step_change(self, step_name):
        """
        Handle a change of the name of the current computation sub step
        by updating the progress dialog and logging the new substep

        Parameters
        ----------
        step_name : str
            The new step name
        """
        if self.progress_dialog is not None:
            self.progress_dialog.subProgressLabel.setText(step_name)
        self.log_progress(f'    {step_name}')

    def log_process_start(self, msg):
        """
        Log the start of a new computation and take a snapshot of the config at that point in time
        Parameters
        ----------
        msg : str
            The message to log
        """
        self.print_status_msg(msg)
        self.log_progress(msg)
        self.save_cfg()

    def log_progress(self, msg):
        self.progress_logger.write(msg)

    def save_cfg(self):
        """
        Take a snapshot of all the configuration at that instant.
        The config files will be saved to a subfolder with the datetime in the name
        """
        cfg_folder = os.path.join(self.src_folder, 'config_snapshots', datetime.now().strftime('%y%m%d_%H_%M_%S'))  # REFACTOR: use pathlib
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
        """Create the performance monitoring bars in the status bar"""
        for label, bar in zip(('CPU', None, 'RAM', 'GPU', 'VRAM'),
                              (self.cpu_bar, self.single_thread_bar, self.ram_bar, self.gpu_bar, self.vram_bar)):
            if label is not None:
                self.statusbar.addPermanentWidget(QLabel(label))
            else:
                bar.setMaximumWidth(5)
                bar.setStyleSheet('QProgressBar::chunk{background-color: yellow;}')
                bar.setTextVisible(False)
                bar.setContentsMargins(0, 0, 0, 0)
            bar.setOrientation(Qt.Vertical)
            bar.setMaximumHeight(40)
            self.statusbar.addPermanentWidget(bar)

    def update_cpu_bars(self, cpu_percent, thread_percent, ram_percent):
        """
        Update the performance monitoring bars in the status bar

        Parameters
        ----------
        cpu_percent : int
            The CPU usage percent to display
        ram_percent : int
            The RAM usage percent to display
        """
        self.cpu_bar.setValue(cpu_percent)
        self.single_thread_bar.setValue(thread_percent)
        self.ram_bar.setValue(ram_percent)

    def update_gpu_bars(self, gpu_percent, v_ram_percent):
        """
        Update the performance monitoring bars in the status bar

        Parameters
        ----------
        gpu_percent : int
            The GPU usage percent to display
        v_ram_percent : int
            The Graphics RAM usage percent to display
        """
        self.gpu_bar.setValue(gpu_percent)
        self.vram_bar.setValue(v_ram_percent)


class ClearMapGui(ClearMapGuiBase):
    """
    The Main class of the GUI. This class focuses on the real business logic of the application.
    This represents the main window which has instances of the different
    tabs derived from GenericTab (the tab_managers), which correspond to different steps of the program.
    Each tab_manager composed of a sample_manager, an optional pipeline object (e.g. Stitcher),
     a widget and a UiParameter or UiParameterCollection object.
    """
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(os.path.join(ICONS_FOLDER, 'logo_cyber.png')))

        self.config_loader = ConfigLoader('')
        self.ortho_viewer = OrthoViewer()

        # Tabs
        self.sample_manager = None  # SampleManager()
        self.tab_managers = {}
        self._init_sample_tab_mgr()

        # Menu actions
        self.preference_editor = PreferenceUi(self)
        self.actionPreferences.triggered.connect(self.preference_editor.open)
        self.structure_selector = StructureSelector(app=self)
        self.actionStructureSelector.triggered.connect(self.structure_selector.show)

        self.amend_ui()
        self.setup_monitoring_bars()
        self.app = QApplication.instance() # noqa

        if not os.path.exists(CLEARMAP_CFG_DIR):
            update_default_config()

    def __len__(self):
        return len(self.tab_managers)

    def __getitem__(self, item):
        return self.tab_managers[item]

    def reset(self):
        self.config_loader = ConfigLoader('')
        self.ortho_viewer = OrthoViewer()

        self._init_sample_tab_mgr()

        self.amend_ui()

    def _reset_tab_manager(self):
        for tab in self.tab_managers.values():
            tab.params = None
        self.tab_managers.clear()
        for tab in self.tabWidget.findChildren(QWidget):
            if tab.objectName().endswith('_tab'):
                delete_widget(tab, self.verticalLayout)

    def reset_pipeline_tabs(self):
        for name in ('stitching', 'registration'):  # FIXME: see why not postprocessing tabs too
            if tab:=self.tab_managers.get(name):
                tab.sample_manager = self.sample_manager
                tab.set_params(self.tab_managers['sample_info'].params, self.config_loader.get_cfg(name))

    def finalise_tab_params(self):   # WARNING: run only after all tabs have been added
        for tab in self.tab_managers.values():
            if not tab.params_finalised:  # FIXME: check if we really want this when we update the workspace with different channel names
                tab.finalise_set_params()

    def _init_sample_tab_mgr(self):
        """
        Clears all existing tabs and creates a new SampleInfoTab and group tabs.
        Typically called at the start of the program or when a new sample is loaded
        """
        print('Initialising sample tab manager')
        self._reset_tab_manager()
        self.sample_manager = SampleManager()
        self.add_tab(SampleInfoTab, sample_manager=self.sample_manager)
        for tab_class in (GroupAnalysisTab, BatchProcessingTab):
            self.add_tab(tab_class)
        self.finalise_tab_params()
        self.tabWidget.tabBarClicked.connect(self.handle_tab_click)
        self.tabWidget.setCurrentIndex(0)

    def __init_pipeline_tabs(self):
        """ Initialises the pipeline tabs: Stitching, Registration, cell... based on sample"""
        stitching_cfg = None
        if self.sample_manager.stitchable_channels:
            self.add_tab(StitchingTab, sample_manager=self.sample_manager, set_params=True)
            stitching_cfg = self.tab_managers['stitching'].params.config
        self.add_tab(RegistrationTab, sample_manager=self.sample_manager, set_params=True)  # WARNING: Always needed to allow setting None for atlas
        # if self.sample_manager.stitchable_channels:
        #    self.tab_managers['stitching'].layout_channel_changed.connect(
        #        self.tab_managers['registration'].handle_layout_channel_changed)
        # FIXME: do we push to config to sample_manager here ?
        self.__add_post_processing_tabs()
        self.sample_manager.load_processors_config(stitching_cfg,
                                                   self.tab_managers['registration'].params.config)
        self.finalise_tab_params()

    def __add_post_processing_tabs(self):
        sample_params = getattr(self.tab_managers['sample_info'], 'params', {})
        for channel_param in sample_params.values():
            data_content_type = channel_param.data_type
            cls = DATA_TYPE_TO_TAB_CLASS.get(data_content_type)
            if not cls:
                warnings.warn(f'No tab class found for data_type: {data_content_type}')
                continue  # Skip if no tab for that data type
            if not any([isinstance(tab, cls) for tab in self.tab_managers.values()]):
                self.add_tab(cls, sample_manager=self.sample_manager, set_params=True)


    def add_tab(self, manager_class, tab_name='', set_params=False, **kwargs):
        """
        Add a new tab to the GUI

        Parameters
        ----------
        manager_class : class
            The class of the tab manager to instantiate
        tab_name : str
            The name of the tab. If not given, will be the class name in snake case
        kwargs : dict
            The keyword arguments to pass to the manager class constructor
        """
        if not tab_name:
            tab_name = manager_class.get_tab_name()
        tab_manager_name = title_to_snake(tab_name)

        tab_idx = self._get_tab_index(tab_name, tab_class=manager_class, increment=True)
        tab = manager_class(self, tab_idx=tab_idx, **kwargs)
        self.tab_managers[tab_manager_name] = tab
        tab.setup()
        self.tabWidget.setTabText(tab.tab_idx, tab.name)
        if set_params:
            cfg_name = title_to_snake(manager_class.get_tab_name())
            loaded_from_defaults, cfg_path = self.__get_cfg_path(cfg_name)
            if cfg_path is not None:
                tab.set_params(self.tab_managers['sample_info'].params, cfg_path, loaded_from_defaults)
        return tab

    def _get_tab_index(self, tab_name, tab_class=None, increment=False):
        """
        Get the index of a tab in the tabWidget.
        If the tab does not exist and increment is True, get the index that it should have when added.

        Parameters
        ----------
        tab_name: str
            The name of the tab
        tab_class: GenericTab
            Optional, provide if increment is True. The class of the tab
        increment: bool
            Whether to increment the index if the tab does not exist

        Returns
        -------
        int
            The index of the tab in the tabWidget
        """
        tab_index = next((i for i in range(self.tabWidget.count()) if self.tabWidget.tabText(i) == tab_name), None)
        if tab_index is None and increment:
            tab_index = self.tabWidget.count()
            if not issubclass(tab_class, BatchTab):  # Insert before batch tabs
                tab_index -= len([t for t in self.tab_managers.values() if isinstance(t, BatchTab)])
        return tab_index

    @property
    def params(self):
        return [tab.params for tab in self.tab_managers.values()]

    @property
    def src_folder(self):
        return self.tab_managers['sample_info'].src_folder

    @src_folder.setter
    def src_folder(self, src_folder):
        if not src_folder:
            return
        src_folder = Path(src_folder)
        if src_folder == Path('.').absolute():
            return  # Do not write log files in the current folder
        self.logger.set_file(src_folder / 'info.log')
        self.progress_watcher.log_path = self.logger.file.name
        self.error_logger.set_file(src_folder / 'errors.html')
        self.progress_logger.set_file(src_folder / 'progress.log')
        self.tab_managers['sample_info'].src_folder = str(src_folder)
        # FIXME: self.sample_manager.src_folder = src_folder or something
        # self.reset_loggers()
        self.reset_pipeline_tabs()

    def display_about(self):  # TODO: get authors list from separate file or documentation
        info = AboutInfo(
            software_name=f'You are running ClearMap version {CLEARMAP_VERSION}',
            version=CLEARMAP_VERSION,
            authors=["Christoph Kirst", "Charly Rousseau", "Sophie Skriabine", "the ClearMap team"],
            github_url="https://github.com/ClearAnatomics/ClearMap.git",
            documentation_url="https://clearanatomics.github.io/ClearMapDocumentation/",
            website_url="https://idisco.info/",
            license_info="Released under the GNU GPLv3 License."
        )
        try:
            from ClearMap.config import commit_info # noqa
            info.commit_info = (f' from commit {commit_info.commit_hash}, {commit_info.commit_date},'
                                f' branch {commit_info.branch}')
        except ImportError:
            pass
        dialog = AboutDialog(info)
        dialog.exec_()


    def amend_ui(self):
        """Setup the loggers and all the post instantiation fixes to the UI"""
        self.reset_loggers()

        self.setup_menus()

        self.setup_icons()
        self.preference_editor.setup(self.config_loader.get_cfg('display')[CURRENT_RES]['font_size'])

        self.monkey_patch()

        self.logoLabel.setPixmap(QtGui.QPixmap(os.path.join(ICONS_FOLDER, 'logo_cyber.png')))

        self.graphLayout.removeWidget(self.frame)

        self.fix_styles()

        self.print_status_msg('Idle, waiting for input')

    def setup_menus(self):
        for action in self.menuBar().actions():
            if action.text() == "&Help":  # Skip if already exists
                return

        help_menu = self.menuBar().addMenu("&Help")
        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self.display_about)
        help_menu.addAction(self.about_action)

        workspace_menu = self.menuBar().addMenu("&Workspace")

        show_info_action = QAction("Show Info", self)
        show_info_action.triggered.connect(self.show_workspace_info)
        workspace_menu.addAction(show_info_action)

        # add_asset_action = QAction("Add Asset", self)
        # add_asset_action.triggered.connect(self.add_asset)
        # workspace_menu.addAction(add_asset_action)

        manipulate_assets_action = QAction("Manipulate Assets", self)
        manipulate_assets_action.triggered.connect(self.manipulate_assets)
        workspace_menu.addAction(manipulate_assets_action)


    def show_workspace_info(self):
        self.popup(f'Workspace: {self.src_folder}, '
                   f'{self.sample_manager.workspace.info()}')

    def add_asset(self):
        pass  # FIXME: implement

    def manipulate_assets(self):
        dlg = ManipulateAssetsDialog(self.src_folder, self.params,
                                     sample_manager=self.sample_manager,
                                     app=self)
        dlg.exec_()

    def reset_loggers(self):
        self.logger = Printer(redirects=None if DEBUG else 'stdout')
        self.logger.text_updated.connect(self.textBrowser.append)
        self.error_logger = Printer(color='red', logger_type='error', redirects=None if DEBUG else 'stderr')
        self.error_logger.text_updated.connect(self.textBrowser.append)

    def reload_prefs(self):
        self.set_font_size(self.preference_editor.params.font_size)

    def set_tabs_progress_watchers(self, nested=False):
        """
        Set the progress_watcher of all the tab managers

        Parameters
        ----------
        nested: bool
            Whether to set the progress watcher of the tab managers to a nested progress watcher
        """
        if nested:
            for tab in self.tab_managers.values():
                tab.set_progress_watcher(self.progress_watcher)
        else:
            self.sample_manager.set_progress_watcher(self.progress_watcher)

    def handle_tab_click(self, tab_index):
        """
        Slot method activated upon clicking a tab in the main tabBar

        Parameters
        ----------
        tab_index: int
            The index of the tab clicked
        """
        if not self.sample_manager.setup_complete and 0 < tab_index < self._get_tab_index('group_analysis', increment=False):
            self.popup('WARNING', 'Workspace not initialised, cannot proceed to alignment')
            self.tabWidget.setCurrentIndex(0)

        # FIXME: might be simpler to use the same string for the tab name and the manager name
        manager_tab_name = title_to_snake(self.tabWidget.tabText(tab_index))
        tab = self.tab_managers[manager_tab_name]
        if tab.processing_type == 'post':
            if not tab.ui.isEnabled():
                return
            if self.needs_registering():
                if self.popup('WARNING', 'Alignment not performed, please run first') == QMessageBox.Ok:
                    self.select_tab('registration')
                else:
                    return
            else:
                tab.finalise_workers_setup()
        elif tab.processing_type == 'batch':
            if not tab.initialised:
                cfg_name = title_to_snake(tab.name)
                try:
                    tab.setup()
                    tab.set_params(self.tab_managers['sample_info'].params, self.config_loader.get_cfg(cfg_name))
                except ConfigNotFoundError:
                    self.conf_load_error_msg(cfg_name)
                except FileNotFoundError:  # message already printed, just stop
                    return

    def needs_registering(self):  # WARNING: can't use registration_processor.was_registered here
        reg_cfg = self.sample_manager.registration_cfg
        if all([cfg['align_with'] is None for cfg in reg_cfg['channels'].values()]):
            return False  # Alignment deselected for all channels
        else:
            ref_channel = self.sample_manager.alignment_reference_channel
            align_with = reg_cfg['channels'][ref_channel]['align_with']
            moving_channel = reg_cfg['channels'][ref_channel]['moving_channel']

            asset = self.sample_manager.get('aligned', channel=ref_channel)
            fixed_channel = ref_channel if align_with == moving_channel else align_with
            return not asset.specify({'moving_channel': moving_channel, 'fixed_channel': fixed_channel}).exists

    def select_tab(self, tab_name):  # WARNING: does not work
        tab_name_to_idx = {tab_name: self.tabWidget.indexOf(tab.ui) for tab_name, tab in self.tab_managers.items()}
        self.tabWidget.setCurrentIndex(tab_name_to_idx[tab_name])

    def conf_load_error_msg(self, conf_name):
        """
        Display an error message on missing config file

        Parameters
        ----------
        conf_name : str
            The name of the config to load (without *params* and extension)
        """
        self.print_error_msg(f'Loading {snake_to_title(conf_name)} config file failed')

    def assert_src_folder_set(self):
        if not self.src_folder:
            msg = 'Missing source folder, please define first'
            self.print_error_msg(msg)
            raise FileNotFoundError(msg)

    def __get_cfg_path(self, cfg_name, config_loader=None):
        """
        Get the absolute file path of the configuration requested. If this file does not
        exist inside the currently defined experimental folder, copy it from the defaults
        in the user home folder.

        Parameters
        ----------
        cfg_name : str
            The name of the config to load (without *params* or extension)
        config_loader : str or None
            The ConfigLoader instance. Use self.config_loader if not set

        Returns
        -------
        bool, str
            was_copied, cfg_path
            The first value indicates whether the file was copied from the defaults
            and may need amending
        """
        config_loader = config_loader or self.config_loader
        cfg_path = config_loader.get_cfg_path(cfg_name, must_exist=False)
        was_copied = False
        # TODO: do as function of PostProcessingTab
        relevant_pipelines = [title_to_snake(p) for p in self.sample_manager.relevant_pipelines]
        # REFACTORING: extract synonyms
        if 'tube_map' in relevant_pipelines:
            relevant_pipelines += ['vasculature']
        if 'cell_map' in relevant_pipelines:
            relevant_pipelines += ['cell_counter']
        if (cfg_name in ('cell_map', 'cell_counter', 'vasculature', 'tube_map', 'tract_map') and
                cfg_name not in relevant_pipelines):
            return False, None
        if not self.file_exists(cfg_path):
            try:
                was_copied = self.__create_config_from_default(config_loader, cfg_name, cfg_path)
            except ClearMapIoException as err:  # Loading skipped
                return False, None
        return was_copied, cfg_path

    def __create_config_from_default(self, config_loader, cfg_name, cfg_path):
        try:
            default_cfg_file_path = config_loader.get_default_path(cfg_name)
        except FileNotFoundError as err:
            self.print_error_msg(f'Could not locate file for "{cfg_name}"')
            raise err

        base_msg, msg = self.create_missing_file_msg(cfg_name.title().replace('_', ''),
                                                     cfg_path, default_cfg_file_path)
        if self.popup(msg) == QMessageBox.Ok:  # copy default config
            if not cfg_path.parent.exists():
                os.mkdir(cfg_path.parent)
            copyfile(default_cfg_file_path, cfg_path)
            return True
        else:
            self.error_logger.write(self.error_logger.colourise(base_msg, force=True))
            raise ClearMapIoException(html_to_ansi(base_msg))

    def clone(self):
        folder = get_directory_dlg(self.preference_editor.params.start_folder,
                                   title="Choose the experiment you would like to clone")
        if not folder:
            return
        src_config_loader = ConfigLoader(folder)
        for tab in self.tab_managers.values():
            cfg_name = title_to_snake(tab.name)
            try:
                src_cfg_path = src_config_loader.get_cfg_path(cfg_name, must_exist=True)
            except FileNotFoundError:
                pass #FIXME: deal with this
            cfg_path = self.config_loader.get_cfg_path(cfg_name, must_exist=False)
            copyfile(src_cfg_path, cfg_path)

    def load_default_cfg(self):
        for tab in self.tab_managers.values():
            cfg_name = title_to_snake(tab.name)
            if cfg_name == 'sample_info':
                cfg_name = 'sample'  # WARNING: a bit hacky
            try:
                src_cfg_path = self.config_loader.get_default_path(cfg_name)
            except FileNotFoundError:
                pass #FIXME: deal with this
            cfg_path = self.config_loader.get_cfg_path(cfg_name, must_exist=False)
            if not cfg_path.exists():
                copyfile(src_cfg_path, cfg_path)

    def load_config_and_setup_ui(self):
        """
        Read (potentially from defaults), fix and load the config for each tab manager
        into the GUI
        """
        self.print_status_msg('Parsing configuration')
        self.assert_src_folder_set()

        error = False
        for tab in self.tab_managers.values():
            cfg_name = title_to_snake(tab.name)
            try:
                # Load tab config
                # FIXME: if old "alignment" or "processing": create stitching and registration cfg files and tabs
                loaded_from_defaults, cfg_path = self.__get_cfg_path(cfg_name)

                # Disable skipped tabs
                if cfg_path is None:
                    tab.disable()
                    continue

                if cfg_name == 'sample_info':
                    sample_params = None
                else:
                    sample_params = self.tab_managers['sample_info'].params

                tab.set_params(sample_params, cfg_path, loaded_from_defaults)
                if cfg_name == 'sample_info' and not tab.params.shared_sample_params.sample_id:
                    self.prompt_sample_id()

            except ConfigNotFoundError as err:
                self.conf_load_error_msg(cfg_name + str(err))
                error = True
            except FileNotFoundError as err:
                print(f'ERRROR loading config {cfg_name}; {err}')
                raise err

        if not error:
            self.print_status_msg('Config loaded')
        else:
            self.print_error_msg(f'Error loading configuration for tab: {tab.name}')

    def get_params(self, tab):
        processing_type = tab.processing_type
        tab_names = []
        if processing_type == 'pre':  # TODO: clean this up
            tab_names.append('sample_info')
        elif processing_type == 'post':
            tab_names.extend(['stitching', 'registration'])
        else:
            raise ValueError(f'Processing type should be one of "pre", "post", "batch" or None,'
                             f' got "{processing_type}"')
        params = [self.tab_managers[t].params for t in tab_names]
        return params

    def prompt_experiment_folder(self):
        """Prompt the user for the main experiment data folder and set it"""
        folder = get_directory_dlg(self.preference_editor.params.start_folder)
        if not folder:
            return
        # if folder and folder != self.src_folder:
        if folder and folder != self.tab_managers['sample_info'].src_folder:
            self.reset()
        self._set_src_folder(folder)

    def _set_src_folder(self, src_folder):
        self.src_folder = src_folder
        self.config_loader.src_dir = src_folder
        sample_cfg_path = self._load_sample_id()
        if ConfigLoader.get_cfg_from_path(sample_cfg_path)['clearmap_version'] == '2.1.0':
            from ClearMap.config.convert_config_versions import convert_v2_1_to_v3_0
            convert_v2_1_to_v3_0(src_folder)
        self.tab_managers['sample_info'].set_params(None, sample_cfg_path, False)

        self.__init_pipeline_tabs()  # TODO: check if init or this

    def prompt_sample_id(self):
        """
        Prompt the user for the sample ID and save it in the sample_params.cfg
        """
        sample_id = input_dialog('Enter the sample ID', 'No sample ID found. '
                                                        'A sample ID is required to load the config'
                                                        'Please enter the sample ID before proceeding')
        if sample_id:
            self.tab_managers['sample_info'].params.shared_sample_params.sample_id = sample_id
            self.tab_managers['sample_info'].params.write()

    def _load_sample_id(self):
        """
        Load the sample ID from the sample_params.cfg if it exists. Otherwise,
        default to empty string
        """
        sample_cfg_path = self.config_loader.get_cfg_path('sample', must_exist=False)
        if not sample_cfg_path.exists():
            match option_dialog('New experiment', 'This seems to be a new experiment. Do you want to: ',
                                ['Clone existing config', 'Load default config', 'Cancel']):
                case 0:
                    self.clone()
                case 1:
                    self.load_default_cfg()
                case 2:
                    self.src_folder = ''
                    self.config_loader.src_dir = ''
                    return
        cfg = self.config_loader.get_cfg_from_path(sample_cfg_path)
        cfg['sample_id'] = cfg['sample_id'] if cfg['sample_id'] != 'undefined' else ''
        cfg.write()
        return cfg.filename


def create_main_window(app, centered=True):
    clearmap_main_win = ClearMapGui()
    if clearmap_main_win.preference_editor.params.start_full_screen:
        clearmap_main_win.showMaximized()  # TODO: check if redundant with show
    if centered:
        clearmap_main_win.move(app.desktop().screenGeometry(0).center() / 2)
    clearmap_main_win.setWindowState(Qt.WindowActive)
    return clearmap_main_win


def main(app, splash):
    clearmap_main_win = create_main_window(app)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    clearmap_main_win.show()
    clearmap_main_win.fix_styles()
    splash.finish(clearmap_main_win)
    if clearmap_main_win.preference_editor.params.verbosity != 'trace':  # WARNING: will disable progress bars
        clearmap_main_win.error_logger.setup_except_hook()
    sys.exit(app.exec())


def entry_point():
    main(app, splash)


if __name__ == "__main__":
    main(app, splash)
