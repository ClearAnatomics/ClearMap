from __future__ import annotations

from packaging.version import Version

from ClearMap.config.convert_config_versions import convert_versions

"""
app
===

The main file for the GUI that contains the entry point to start the software

.. ::

    TODO:
    Handle reset detected correctly
    Fix qrc resources (ui files should not be coded by path)
    Delete intermediate files
    Ensure all machine config params are in the preferences UI

    Analysis:

    LATER:
    Auto modes:
        - Run all
"""

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

from ClearMap.gui.gui_utils_base import unique_connect
from ClearMap.gui.widgets import ClickableFrame

print('Starting base imports...', flush=True)
import os
import sys
import math
import shutil
import inspect
import tempfile
import warnings
from copy import deepcopy

from typing import Optional, Callable, List, Dict, Type, Any, Iterable, Tuple
import atexit
from multiprocessing.pool import ThreadPool
from pathlib import Path

from statistics import mode

# # WARNING: Necessary for QCoreApplication creation
# from PyQt5.QtWebEngineWidgets import QWebEngineView  # noqa: F401

print('Importing PyQt5...', flush=True)
from PyQt5 import QtGui
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QSpinBox, QDoubleSpinBox,
                             QComboBox, QLineEdit, QMessageBox, QToolBox, QProgressBar, QLabel,
                             QStyle, QAction, QDockWidget)

import qdarkstyle
from qdarkstyle import DarkPalette

print('Running first ClearMap imports...', flush=True)
import ClearMap.gui.dialog_helpers

os.environ['CLEARMAP_GUI_HOSTED'] = "1"
# ########################################### SPLASH SCREEN ###########################################################

from ClearMap.config.early_boot import MachineConfig, first_boot
from ClearMap.gui import gui_utils_base as base_utils

print('Running first boot checks...', flush=True)
first_boot()

_machine_cfg = MachineConfig()
DEBUG = _machine_cfg.verbosity in ('trace', 'debug')
CLEARMAP_VERSION = _machine_cfg._version

print('Building application...', flush=True)
# To show the splash screen during imports
app = base_utils.ensure_qapp()
palette = app.palette()  # WARNING: necessary because QWhatsThis does not follow stylesheets
palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(DarkPalette.COLOR_BACKGROUND_2))
palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(DarkPalette.COLOR_TEXT_2))
app.setPalette(palette)  # noqa

# Attempt to force appearance
app.setApplicationName('ClearMap')
app.setApplicationDisplayName('ClearMap')
app.setOrganizationName('ClearAnatomics')
app.setApplicationVersion(CLEARMAP_VERSION)
try:
    QGuiApplication.setDesktopFileName('clearmap.desktop')  # For Ubuntu Unity proper task bar display
except Exception:
    pass   #  some graphical envs complain but never mind

# Resolve icon from installed package REFACTOR: use qrc
here = Path(__file__).resolve()
GUI_FOLDER = here.parent
ICONS_FOLDER = GUI_FOLDER / 'creator' / 'icons'
CLEARMAP_ICON = QtGui.QIcon(str(ICONS_FOLDER / 'logo_cyber.png'))
app.setWindowIcon(CLEARMAP_ICON)

print('Creating splash screen...', flush=True)

# WARNING: first because otherwise, breaks (must be imported before processing the UI)
from ClearMap.Alignment.Stitching import layout_graph_utils  # noqa: F401

from ClearMap.gui import dialog_helpers as dlg_helpers

splash, progress_bar = ClearMap.gui.dialog_helpers.make_splash(text_zone=30)  # Use default resolution for this early version
splash.show()

def overlay_splash_message(msg: str) -> None:
    print(msg, flush=True)

    font = QtGui.QFont()
    font.setPointSize(16)
    font.setBold(True)
    font.setStyleHint(QtGui.QFont.SansSerif)
    splash.setFont(font)

    color = QtGui.QColor(255, 255, 255, 180)
    splash.showMessage(msg, Qt.AlignLeft | Qt.AlignBottom, color)
    app.processEvents()

app.processEvents()

from ClearMap.gui.pretty_imports import run_staged_imports, discover_import_tasks
app.processEvents()
SLOW_IMPORTS = discover_import_tasks(__file__)
print('Staged imports:')
if not SLOW_IMPORTS:
    print(f'No slow imports detected. This is probably a bug, '
          f'check the discover_import_tasks function and the import tasks in this file')


print('Starting heavy imports...', flush=True)
run_staged_imports(SLOW_IMPORTS, on_message=overlay_splash_message, #lambda m: splash.showMessage(m),
                   on_progress=lambda p: ClearMap.gui.dialog_helpers.update_pbar(app, progress_bar, p),
                   target_namespace=globals())

#  WARNING: the following lines serves as a marker for the `discover_import_tasks` scanner, do not alter or delete it
### SLOW IMPORTS ###
import pyqtgraph as pg
import torch

from ClearMap.Visualization.Qt.DataViewer import DataViewer

from ClearMap.Alignment.Stitching import layout_graph_utils  # noqa: F401 # WARNING: first because otherwise, breaks with pytorch

from ClearMap.Utils.utilities import title_to_snake, snake_to_title, deep_merge
from ClearMap.Utils.event_bus import EventBus, BusSubscriberMixin
from ClearMap.Utils.events import (CfgChanged, UiRequestRefreshTabs, UiTabActivated, TabActivationResult,
                                   TabsUpdated,  ChannelsSnapshot, ChannelDefaultsChanged)

from ClearMap.config.update_config import update_default_config
from ClearMap.config.config_coordinator import make_cfg_coordinator_factory
from ClearMap.config.config_handler import ConfigHandler, CLEARMAP_CFG_DIR, ALTERNATIVES_REG, \
    scan_folder_for_experiments

from ClearMap.gui.gui_utils_images import get_current_res
from ClearMap.gui.tab_registry import TabRegistry
from ClearMap.gui.widget_monkeypatch_callbacks import recursive_patch_widgets
from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager
from ClearMap.pipeline_orchestrators.experiment_controller import ExperimentController, AnalysisGroupController, \
    AppMode
from ClearMap.gui.tabs_interfaces import GenericTab, ExperimentTab, GroupTab
from ClearMap.gui.preferences import PreferenceUi
from ClearMap.gui.gui_logging import Printer
from ClearMap.gui.pyuic_utils import loadUiType
from ClearMap.gui import widgets as cmp_widgets
from ClearMap.gui import style
from ClearMap.gui import dialogs as dlgs
from ClearMap.gui.about import AboutInfo
### END SLOW IMPORTS ###

CURRENT_RES = get_current_res(app)

pg.setConfigOption('background', style.PLOT_3D_BG)

ABOUT_INFO = AboutInfo(software_name=f'You are running ClearMap version {CLEARMAP_VERSION}', version=CLEARMAP_VERSION,
              authors=["Christoph Kirst", "Charly Rousseau", "Sophie Skriabine", "the ClearMap team"],
              github_url="https://github.com/ClearAnatomics/ClearMap.git",
              documentation_url="https://clearanatomics.github.io/ClearMapDocumentation/",
              website_url="https://idisco.info/", license_info="Released under the GNU GPLv3 License.")

HARD_DEFAULT_FONT_SIZE = 11

Ui_ClearMapGui, _ = loadUiType(os.path.join(base_utils.UI_FOLDER, 'creator', 'mainwindow.ui'), from_imports=True,
                               import_from='ClearMap.gui.creator', patch_parent_class=False,
                               customWidgets={"ClickableFrame": ClickableFrame})


class ClearMapAppBase(QMainWindow, Ui_ClearMapGui):
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
        self.progress_watcher = cmp_widgets.ProgressWatcher(timer_interval_ms=250)

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
        self.perf_monitor = cmp_widgets.PerfMonitor(self, 500, gpu_period)
        self.perf_monitor.cpu_vals_changed.connect(self.update_cpu_bars)
        if torch.cuda.is_available():
            self.perf_monitor.gpu_vals_changed.connect(self.update_gpu_bars)
        self.perf_monitor.start()

    def find_child_by_name(self, child_name: str, child_type: type[QWidget],
                           parent: Optional[QWidget] = None) -> Optional[QWidget]:
        """
        Find children in the window or any other widget

        Parameters
        ----------
        child_name : str
            The name of the widget we are looking for
        child_type:
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
        else:
            return None

    def __print_status_msg(self, msg: str, color):
        self.statusbar.setStyleSheet(f'color: {color}')
        self.statusbar.showMessage(msg)

    def print_error_msg(self, msg: str | Exception):
        """
        Print a message in red in the statusbar

        Parameters
        ----------
        msg : str
            The message to be printed
        """
        if isinstance(msg, Exception):
            msg = str(msg)
        self.__print_status_msg(msg, '#ff5555')  # red from qdarkstyle palette

    def print_warning_msg(self, msg: str):
        """
        Print a message in yellow in the statusbar

        Parameters
        ----------
        msg : str
            The message to be printed
        """
        self.__print_status_msg(msg, 'yellow')

    def print_status_msg(self, msg: str):
        """
        Print a message in green in the statusbar

        Parameters
        ----------
        msg : str
            The message to be printed
        """
        self.__print_status_msg(msg, 'green')

    def set_font_size(self, target_font_size=dlg_helpers.DISPLAY_CONFIG[CURRENT_RES]['font_size']):
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
            font = widget.property('font')
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
        dock_width = round(self.width() * 4/5)
        self.resizeDocks([self.graphDock], [dock_width], Qt.Horizontal)

    def fix_styles(self):
        """The meta function calling all the style functions"""
        self.setStyleSheet(style.BTN_STYLE_SHEET)  # Makes it look qdarkstyle
        # self.fix_btns_stylesheet()
        self.fix_widgets_backgrounds()
        self.fix_sizes()
        self.fix_tooltips_stylesheet()

        try:
            tab = self.get_tab_widget('Sample info')
        except KeyError:  # FIXME: check that we are indeed in batch mode in that case
            return
        children = tab.findChildren(QWidget)
        if not children:
            return  # tab not fully loaded yet
        btn = tab.launchPatternWizardPushButton
        btn.setStyleSheet(style.HIGHLIGHTED_BTN_STYLE)

    def fix_tooltips_stylesheet(self):
        for widg in self.findChildren(QWidget):
            if hasattr(widg, 'toolTip') and widg.toolTip():
                widg.setStyleSheet(style.TOOLTIP_STYLE_SHEET)

    def fix_btns_stylesheet(self):
        for btn in self.findChildren(QPushButton):
            btn.setStyleSheet(style.BTN_STYLE_SHEET)

    def fix_widgets_backgrounds(self):
        for widget_type in (QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit):
            for widget in self.findChildren(widget_type):
                widget.setStyleSheet(f'background-color: {style.DARK_BACKGROUND}; ')
                if widget_type == QComboBox:
                    widget.setStyleSheet(style.COMBOBOX_STYLE_SHEET)

    def popup(self, msg: str, base_msg: str = 'Missing configuration file',
              print_warning: bool = True) -> QMessageBox:
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
            self.print_warning_msg(base_utils.html_to_plain_text(msg))
        return ClearMap.gui.dialog_helpers.warning_popup(base_msg, msg)

    def file_exists(self, f_path: Path) -> bool:
        if f_path.exists():
            return True
        else:
            self.print_warning_msg(f'File "{f_path}" not found')
            return False

    def graph_by_name(self, name: str) -> QWidget:
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
        base_utils.clear_layout(self.graphLayout)
        self.graphs = []
        self.perf_monitor.start()

    def setup_plots(self, dvs: list["DataViewer"], graph_names: Optional[list[str]] = None):
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

        n_rows, n_cols = base_utils.compute_grid(len(dvs))
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
    def create_missing_file_msg(f_type: str, f_path: Path | str, default_path: Path | str) -> tuple[str, str]:
        base_msg = f'No {f_type} file found at:<br>  <nobr><em>"{f_path}"</em></nobr>.'
        msg = f'{base_msg} <br><br>Do you want to load a default one from:<br>  <nobr><em>"{default_path}"</em>?</nobr>'
        return base_msg, msg

    def make_progress_dialog(self, title: str = 'Processing', n_steps: int = 1, maximum: int = 100,
                             abort: Optional[Callable] = None, parent: Optional[QWidget] = None):
        """
        Create a single or nested (2 nested bars) progress dialog.
        The dialog is initialised with the parameters below and linked to self.progress_watcher
        to drive updates of the titles and progress bar.

        Parameters
        ----------
        title : str
            The title of the dialog
        n_steps : int
            The number of steps of the main step for nested dialogs
        maximum : int
            The maximum value of the progress bar (or the second bar for nested dialogs)
        abort : function
            The function to trigger when the abort button is clicked
        parent : QWidget
            The parent widget to the dialog
        """
        if n_steps:
            n_steps += 1  # To avoid range shrinking because starting from 1 not 0
            nested = True
            self.progress_dialog = ClearMap.gui.dialog_helpers.make_nested_progress_dialog(
                title=title, overall_maximum=n_steps,
                abort_callback=abort, parent=parent)

            # Dialog-bound slots — disconnect_all because target widget changes each call
            unique_connect(self.progress_watcher.main_max_changed,
                           self.progress_dialog.mainProgressBar.setMaximum,
                           disconnect_all=True)
            unique_connect(self.progress_watcher.main_progress_changed,
                           self.progress_dialog.mainProgressBar.setValue,
                           disconnect_all=True)

            # Stable slot — just ensure no duplicate
            unique_connect(self.progress_watcher.main_step_name_changed,
                           self.handle_step_name_change)
        else:
            nested = False
            self.progress_dialog = ClearMap.gui.dialog_helpers.make_simple_progress_dialog(
                title=title, abort_callback=abort, parent=parent)

        # Dialog-bound slots — disconnect_all because target widget changes each call
        unique_connect(self.progress_watcher.max_changed,
                       self.progress_dialog.subProgressBar.setMaximum,
                       disconnect_all=True)
        unique_connect(self.progress_watcher.progress_changed,
                       self.progress_dialog.subProgressBar.setValue,
                       disconnect_all=True)

        # Stable slots — just ensure no duplicate
        unique_connect(self.progress_watcher.sub_step_name_changed,
                       self.handle_sub_step_change)
        unique_connect(self.progress_watcher.finished,
                       self.signal_process_finished)

        if nested:
            self.progress_watcher.setup(main_step_name=title, main_step_length=n_steps)
        else:
            self.progress_watcher.setup(main_step_name='', main_step_length=1, sub_step_length=maximum, pattern=None)

        self.set_tabs_progress_watchers(nested=nested)

    def set_tabs_progress_watchers(self, nested: bool = False):
        raise NotImplementedError()

    def wrap_in_thread(self, func: Callable, *args, **kwargs):
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
        self.progress_watcher.start_polling()
        with ThreadPool(processes=1) as pool:
            result = pool.apply_async(func, args, kwargs)
            while not result.ready():
                result.wait(0.25)
                self.app.processEvents()
        self.progress_watcher.stop_polling()
        self.progress_watcher.finish()
        return result.get()

    def signal_process_finished(self, msg: str = 'Idle, waiting for input'):
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
            dlg = self.progress_dialog
            self.progress_dialog = None  # ← set to None BEFORE done()
            dlg.done(1)  # so re-entrant calls are no-ops

    def handle_step_name_change(self, step_name: str):
        """
        Handle a change of the name of the current computation step by registering
        the new step and updating the progress dialog

        Parameters
        ----------
        step_name : str
            The new step name
        """
        if self.progress_dialog is not None:
            self.progress_dialog.mainStepNameLabel.setText(step_name)

    def handle_sub_step_change(self, step_name: str):
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

    def log_progress(self, msg: str):
        self.progress_logger.write(msg)

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

    def update_cpu_bars(self, cpu_percent: int, thread_percent: int, ram_percent: int):
        """
        Update the performance monitoring bars in the status bar

        Parameters
        ----------
        cpu_percent : int
            The CPU usage percent to display
        thread_percent : int
            The percent usage of the main ClearMap thread to display
        ram_percent : int
            The RAM usage percent to display
        """
        self.cpu_bar.setValue(cpu_percent)
        self.single_thread_bar.setValue(thread_percent)
        self.ram_bar.setValue(ram_percent)

    def update_gpu_bars(self, gpu_percent: int, v_ram_percent: int):
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


class ClearMapApp(ClearMapAppBase):
    """
    The Main class of the GUI. This class focuses on the real business logic of the application.
    This represents the main window which has instances of the different
    tabs derived from GenericTab , which correspond to different steps of the program.
    """
    def __init__(self, experiment_controller, bus, gui_controller):
        super().__init__()
        self.experiment_controller = experiment_controller
        self.bus = bus
        self.gui_controller = gui_controller

        self.setupUi(self)
        self.setWindowIcon(CLEARMAP_ICON)

        self.config_loader = ConfigHandler('')
        self.ortho_viewer = cmp_widgets.OrthoViewer()

        self.experimentModePushButton.clicked.connect(self._select_experiment_mode)
        self.batchModePushButton.clicked.connect(self._select_group_mode)

        self._init_sample_tab_mgr()

        # Menu actions
        self.about_action = QAction('&About', self)
        self.preference_editor = PreferenceUi(self)
        self.actionPreferences.triggered.connect(self.preference_editor.open)

        self.assetsManagerWidget: Optional[cmp_widgets.ManageAssetsWidget] = None
        self.assetsManagerDock = QDockWidget("Workspace Manager", self)
        self.assetsManagerDock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.assetsManagerDock.setFloating(False)
        self.addDockWidget(Qt.RightDockWidgetArea, self.assetsManagerDock)

        self.structure_selector = cmp_widgets.StructureSelector(app=self)
        self.actionStructureSelector.triggered.connect(self.structure_selector.show)

        self.amend_ui()
        self.setup_monitoring_bars()
        self.app = QApplication.instance() # noqa

        # FIXME: do for current version w/ config_handler
        if not os.path.exists(CLEARMAP_CFG_DIR):
            update_default_config()

        self.bus.subscribe(TabsUpdated, self._on_tabs_updated)

    def reset(self):
        self.config_loader = ConfigHandler('')
        self.ortho_viewer = cmp_widgets.OrthoViewer()

        self._init_sample_tab_mgr()

        self.amend_ui()

    def _select_experiment_mode(self):
        self.gui_controller.set_mode(AppMode.EXPERIMENT)
        self.centralStack.setCurrentIndex(1)  # tabs page

    def _read_exp_version(self, exp_dir: Path) -> Version | None:
        loader = ConfigHandler(exp_dir)
        sample_path = loader.get_cfg_path('sample', must_exist=False)
        if not sample_path or not Path(sample_path).exists():
            return None
        cfg = ConfigHandler.get_cfg_from_path(sample_path)
        v = cfg.get('clearmap_version')
        return Version(str(v)) if v else None

    def _select_group_mode(self):
        base = self.preference_editor.params.start_folder
        group_dir = ClearMap.gui.dialog_helpers.get_directory_dlg(
            base, title='Select cohort folder (contains experiment folders)')
        if not group_dir:
            return

        group_dir = Path(group_dir).resolve()

        exp_roots = scan_folder_for_experiments(group_dir)  # your new helper (roots set)

        current = Version(str(self.app.applicationVersion()))  # or Version(CLEARMAP_VERSION)

        outdated: list[tuple[Path, Version | None]] = []
        for root in sorted(exp_roots):
            v = self._read_exp_version(root)
            if v is None or v < current:
                outdated.append((root, v))

        if outdated:
            # prompt once
            msg = f'{len(outdated)} experiments appear older than {current}. Upgrade now?'
            if not ClearMap.gui.dialog_helpers.prompt_dialog('Upgrade experiments', msg):
                return

            for root, v in outdated:
                prev = str(v) if v else '2.1'  # fallback to first numbered version
                convert_versions(prev, str(current), exp_dir=root, create_app=False)

        # 1) init group controller before tabs
        self.gui_controller.group_controller.set_group_base_dir(group_dir)
        self.setup_loggers(group_dir)
        self.progress_watcher.log_path = self.logger.file.name
        # 2) switch mode
        self.gui_controller.set_mode(AppMode.GROUP)
        self.centralStack.setCurrentIndex(1)  # tabs page

    @property
    def sample_manager(self):
        return self.experiment_controller.sample_manager

    def _on_tabs_updated(self, event: TabsUpdated):
        self.set_tabs(event.tabs)

    def set_tabs(self, tabs: Iterable[GenericTab]):
        """Sets the tabs received from the GuiController"""
        self.tabWidget.clear()
        for tab in tabs:  # WARNING: not duplicate with GuiController ?
            tab.setup()  # build the QWidget  (idempotent)
            self.tabWidget.addTab(tab.ui, tab.name)  # We read but the tab is just the ref to the old if exists
        self.fix_styles()

    def get_tab_widget(self, tab_title):
        """
        Get the tab widget by its title

        Parameters
        ----------
        tab_title : str
            The title of the tab to get

        Returns
        -------
        GenericTab
            The tab matching the title
        """
        for i in range(self.tabWidget.count()):
            if self.tabWidget.tabText(i) == tab_title:
                return self.tabWidget.widget(i)
        else:
            raise KeyError(f'Tab with title "{tab_title}" not found.'
                           f'Possible titles are: {[self.tabWidget.tabText(i) for i in range(self.tabWidget.count())]}')

    def _init_sample_tab_mgr(self):
        """
        Clears all existing tabs and creates a new SampleInfoTab and group tabs.
        Typically called at the start of the program or when a new sample is loaded
        """
        print('Initialising sample tab manager')

        sample_tab = self.select_tab('Sample info')  # select first tab
        tbs = sample_tab.findChildren(QToolBox)
        if not tbs:
            return
        else:
            tbs[0].setCurrentIndex(0)

        unique_connect(self.tabWidget.tabBarClicked, self.handle_tab_click)
        unique_connect(self.tabWidget.currentChanged, self.handle_tab_click)

    @property
    def src_folder(self) -> str:
        return str(self.experiment_controller.exp_dir) if self.experiment_controller.exp_dir else ''

    @src_folder.setter
    def src_folder(self, src_folder):
        if not src_folder:
            return
        src_folder = Path(src_folder)
        if src_folder == Path('.').absolute():
            return  # Do not write log files in the current folder

        self.setup_loggers(src_folder)
        self.progress_watcher.log_path = self.logger.file.name

        self.experiment_controller.set_experiment_dir(src_folder)
        # FIXME: trigger tabs reset here or from controller to point to new sample_manager
        # self.reset_loggers()

    def setup_loggers(self, src_folder):
        self.logger.set_file(src_folder / 'info.log')
        self.error_logger.set_file(src_folder / 'errors.html')
        self.progress_logger.set_file(src_folder / 'progress.log')

    def reset_loggers(self):# FIXME: redirect should be f(log_level)
        self.logger = Printer(redirects=None if DEBUG else 'stdout')
        self.logger.text_updated.connect(self.textBrowser.append)
        self.error_logger = Printer(color='red', logger_type='error', redirects=None if DEBUG else 'stderr')
        self.error_logger.text_updated.connect(self.textBrowser.append)
        if DEBUG:
            self.patch_warnings_color()

    def patch_warnings_color(self):
        YELLOW = "\033[33m"
        RESET = "\033[0m"

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            text = warnings.formatwarning(message, category, filename, lineno, line)
            # Force to stdout so PyCharm doesn't apply stderr styling rules
            sys.stdout.write(f"{YELLOW}{text}{RESET}")
            sys.stdout.flush()

        warnings.showwarning = _showwarning

    def display_about(self):  # TODO: get authors list from separate file or documentation
        info = deepcopy(ABOUT_INFO)
        try:
            from ClearMap.config import commit_info # noqa
            info.commit_info = (f' from commit {commit_info.commit_hash}, {commit_info.commit_date},'
                                f' branch {commit_info.branch}')
        except ImportError:
            pass
        dialog = dlgs.AboutDialog(info)
        dialog.exec_()

    def _get_global_view(self):  # FIXME: should use a sort of cfg_coordinator instead to also have display
        return {sec_name: self.config_loader.get_cfg(sec_name, must_exist=True)
                for sec_name in [gp[0] for gp in ALTERNATIVES_REG._global_groups]}

    def _apply_global_patch(self, patch: dict):
        # Accepts a patch rooted at top-level; we care about 'machine'
        machine_patch = patch.get('machine', patch)

        # Read current global config
        current = self.config_loader.get_cfg('machine', must_exist=False)
        current_dict = dict(current) if current is not None else {}

        # Merge & write back to the proper GLOBAL path (user defaults dir)
        deep_merge(current_dict, machine_patch)
        target_path = self.config_loader.get_global_path('machine', must_exist=False)
        ConfigHandler.dump(target_path, current_dict)

    def amend_ui(self):
        """Setup the loggers and all the post instantiation fixes to the UI"""
        self.reset_loggers()

        self.setup_menus()
        self.setup_icons()
        self.logoLabel.setPixmap(QtGui.QPixmap(str(ICONS_FOLDER / 'logo_cyber.png')))
        self.preference_editor.setup(self.config_loader.get_cfg('display')[CURRENT_RES]['font_size'],
                                     event_bus=self.bus,
                                     view_provider=self._get_global_view,
                                     apply_patch=self._apply_global_patch)

        self.monkey_patch()
        self.graphLayout.removeWidget(self.frame)

        self.fix_styles()

        self.print_status_msg('Idle, waiting for input')

    def setup_menus(self):
        menu_names = [action.text() for action in self.menuBar().actions()]
        if "&Help" in menu_names:  # Not the first call
            return

        file_action = [action for action in self.menuBar().actions() if action.text() == 'File'][0]
        file_menu = file_action.menu()
        open_act = QAction("&Open Experiment Folder…", self)
        open_act.triggered.connect(self.prompt_experiment_folder)
        file_menu.addAction(open_act)

        workspace_menu = self.menuBar().addMenu('&Workspace')

        show_info_action = QAction('Show Info', self)
        show_info_action.triggered.connect(self.show_workspace_info)
        workspace_menu.addAction(show_info_action)

        # FIXME: enable when implemented
        # add_asset_action = QAction("Add Asset", self)
        # add_asset_action.triggered.connect(self.add_asset)
        # workspace_menu.addAction(add_asset_action)

        manipulate_assets_action = QAction('Manage Assets', self)
        manipulate_assets_action.triggered.connect(self.manage_assets)
        workspace_menu.addAction(manipulate_assets_action)

        help_menu = self.menuBar().addMenu('&Help')
        self.about_action.triggered.connect(self.display_about)
        help_menu.addAction(self.about_action)

    def show_workspace_info(self):
        if self.sample_manager.workspace is None:
            self.popup('No workspace found. Initialise sample before using this menu (workspace info).')
            return
        self.popup(f'Workspace: {self.src_folder}, {self.sample_manager.workspace.info()}')

    def add_asset(self):  pass  # FIXME: implement

    def manage_assets(self):
        if self.sample_manager.workspace is None:
            self.popup('No workspace found. Initialise sample before using this menu (manage_assets).')
            return
        sample_cfg = self.experiment_controller.get_config_view()['sample']

        self.assetsManagerWidget = cmp_widgets.ManageAssetsWidget(
            self.src_folder, sample_cfg, sample_manager=self.sample_manager, app=self)
        self.assetsManagerDock.setWidget(self.assetsManagerWidget.widget)
        self.assetsManagerDock.setVisible(True)  # In case user hid it

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
            self.experiment_controller.set_workers_progress_watcher(self.progress_watcher)
        else:
            pass
            warnings.warn('Progress watcher not set for nested progress watcher')
            # FIXME: set in the tabs (except sample info which does not support) from GuiController
            # self.sample_manager.set_progress_watcher(self.progress_watcher)

    def handle_tab_click(self, index: int):
        tab_title = self.tabWidget.tabText(index)
        self.bus.publish(UiTabActivated(key=title_to_snake(tab_title)))

    def _on_tab_activation_result(self, event: TabActivationResult):
        if not event.ok:
            self.popup('WARNING', event.message)
            # bounce back to Sample tab
            self.select_tab('Sample info')

    def select_tab(self, tab_name: str) -> QWidget | None:
        for i in range(self.tabWidget.count()):
            if self.tabWidget.tabText(i) == tab_name:
                self.tabWidget.setCurrentIndex(i)
                return self.tabWidget.widget(i)
        return None

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

    def _clone(self):
        """
        Clone an existing experiment configuration into the current one.
        The user is prompted to select the source folder to clone from.

        Returns
        -------
        bool
            True if the cloning was successful, False otherwise
        """
        folder_to_clone = ClearMap.gui.dialog_helpers.get_directory_dlg(self.preference_editor.params.start_folder,
                                                                        title='Choose the experiment you would like to clone')
        if not folder_to_clone:
            return False

        if not self.src_folder:
            self.popup('Please select or create the destination experiment folder first')
            return False

        self.experiment_controller.clone_from(folder_to_clone, Path(self.src_folder))
        self.print_status_msg(f'Cloned config from {folder_to_clone} to {self.src_folder}')
        return True

    def prompt_experiment_folder(self):
        """Prompt the user for the main experiment data folder and set it"""
        folder = ClearMap.gui.dialog_helpers.get_directory_dlg(self.preference_editor.params.start_folder)
        if not folder:
            return
        if folder and folder != self.src_folder:
            self.reset()
        self._set_src_folder(folder)

    def _set_src_folder(self, src_folder):
        self.src_folder = src_folder
        if not self.src_folder:
            return  # user cancelled

        needs_exp_open = self.load_sample()
        if needs_exp_open:
            sample_version = self.experiment_controller.read_sample_version()
            if sample_version != CLEARMAP_VERSION:  # Old version detected
                match ClearMap.gui.dialog_helpers.option_dialog(f'Old version detected ({sample_version})',
                                                                f'An old version of ClearMap was detected, do you want to: ',
                                                                [f'Upgrade to {CLEARMAP_VERSION}', 'Abort']):
                    case 0:  # Upgrade
                        self.experiment_controller.upgrade_configs(sample_version, CLEARMAP_VERSION)
                    case 1:  # Abort
                        self.src_folder = ''  # reset
                        return

            # Define sample ID after optional upgrade
            while not self.experiment_controller.get_or_init_sample_id():
                self._prompt_sample_id()

            # splash, pbar = ClearMap.gui.dialog_helpers.make_splash(
            #     message=f'Loading sample {Path(self.src_folder).name}', font_size=25)
            # splash.show()  # FIXME: no progress in splash bar during boot_open

            # TODO: progress bar updates
            self.gui_controller.begin_hydration()
            self.experiment_controller.boot_open(Path(self.src_folder))
            self.gui_controller.end_hydration()
            # ClearMap.gui.dialog_helpers.update_pbar(self.app, progress_bar, 90)
            #
            # splash.finish(self)
        self.manage_assets()

    def _prompt_sample_id(self):
        """
        Prompt the user for the sample ID and save it in the sample_params.cfg
        """
        sample_id = ClearMap.gui.dialog_helpers.input_dialog('Enter the sample ID', 'No sample ID found. '
                                                             'A sample ID is required to load the config'
                                                             'Please enter the sample ID before proceeding')
        if sample_id:
            self.experiment_controller.set_sample_id(sample_id)
        return sample_id

    def load_sample(self) -> bool:
        """
        Load the sample configuration.
        If the sample path does not exist, prompt the user to clone an existing config,
        load a default config, or cancel.
        If the sample path exists, ensure a sample ID is set.
        The return value tells the caller whether they need to call boot_open().

        Returns
        -------
        bool
           True  -> existing experiment, caller must call boot_open()
           False -> either:
                      - new experiment already bootstrapped & opened, or
                      - user canceled (src_folder is cleared)
        """
        if not self.experiment_controller.sample_path_exists():
            match ClearMap.gui.dialog_helpers.option_dialog('New experiment', 'This seems to be a new experiment. Do you want to: ',
                                                            ['Clone existing config', 'Load default config', 'Cancel']):
                case 0:  # Clone existing
                    cloned = self._clone()
                    if not cloned:
                        self.src_folder = ''
                case 1:  # Load default
                    self.experiment_controller.boot_new()
                case 2:  # Cancel
                    self.src_folder = ''
            return False

        else:
            return True
            # self.experiment_controller.boot_open(Path(self.src_folder))  # not here because potential mismatch


class GuiController(BusSubscriberMixin):
    """
    The controller of the GUI. It is responsible for building the main window,
    installing the tabs and handling events from the EventBus.
    It interacts with the ExperimentController to manage the business logic of the application.
    """
    def __init__(self, bus: EventBus, experiment, tab_registry: TabRegistry,
                 group_controller: AnalysisGroupController):
        super().__init__(bus)
        self._hydrating = False
        self.experiment_controller: ExperimentController = experiment
        self.tabs_registry: TabRegistry = tab_registry  # stays a UI concern

        self.group_controller: AnalysisGroupController = group_controller

        self.sample_manager: SampleManager = experiment.sample_manager

        self.mode: AppMode = AppMode.EXPERIMENT  # Exp as default

        self._tabs: List[GenericTab] = []  # instances
        self.window: QMainWindow | None = None

        # Hydration state flags
        self._needs_full_refresh = False
        self._tabs_initialized = False

        self.subscribe(CfgChanged, self._on_cfg_changed)
        self.subscribe(UiRequestRefreshTabs, self._on_refresh_tabs)
        self.subscribe(UiTabActivated, self._on_tab_activated)

    def start(self, app_: ClearMapApp, centered: bool = True):
        self.window = self._build_main_window()
        self._install_or_update_tabs()
        self.window._init_sample_tab_mgr()  # Not great to call from outside but... needs to happen after install_or_update_tabs
        self._post_show_setup(app_, centered)
        self.window.show()
        self.window.fix_styles()
        app_.processEvents()

    def set_mode(self, mode: AppMode):
        if mode == self.mode:
            return
        self.mode = mode
        # refresh tabs to reflect mode change
        self._install_or_update_tabs()
        self._tabs_initialized = True

    @property
    def hydrating(self) -> bool:
        """True when either the controller or the experiment is mid-hydration."""
        return self._hydrating or self.experiment_controller.hydrating

    def begin_hydration(self):
        self._hydrating = True
        self._needs_full_refresh = False

    def end_hydration(self):
        self._hydrating = False
        if self._needs_full_refresh or not self._tabs_initialized:
            self.experiment_controller.cfg_coordinator.submit(  # Force adjusters and validators to run after hydration, to ensure UI is in sync with model
                sample_manager=self.experiment_controller.sample_manager,
                do_run_adjusters=True, validate=True, commit=True)

            self._install_or_update_tabs()
            self._tabs_initialized = True
            self._refresh_tabs_from_model()
            self._needs_full_refresh = False

    def _build_main_window(self) -> QMainWindow:
        window = ClearMapApp(self.experiment_controller, self._bus, self)
        window.fix_styles()
        return window

    def _post_show_setup(self, app_, centered: bool) -> None:
        win = self.window
        if win.preference_editor.params.start_full_screen:
            win.showMaximized()
        if centered:
            screen = app_.desktop().screenGeometry(win)
            fg = win.frameGeometry()
            fg.moveCenter(screen.center())
            win.move(fg.topLeft())
        win.setWindowState(Qt.WindowActive)

    # ---------- tabs lifecycle ----------
    @property
    def tabs(self) -> Iterable[Any]:
        return list(self._tabs)

    def _tab_instances_by_class(self) -> Dict[Type[Any], Any]:
        return {t.__class__: t for t in self._tabs}

    def _get_or_create(self, cls: Type[GenericTab], tab_idx: int = -1) -> GenericTab:
        by_cls = self._tab_instances_by_class()
        t = by_cls.get(cls)
        if t is not None:
            return t

        if tab_idx == -1:
            tab_idx = len(self._tabs)

        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        kwargs = {}
        if 'sample_manager' in params:
            kwargs['sample_manager'] = self.sample_manager
        # For group tabs needing group_controller (kw-only)
        if 'group_controller' in params:
            kwargs['group_controller'] = self.group_controller

        tab = cls(self.window, tab_idx, **kwargs)
        if isinstance(tab, ExperimentTab):
            # ExperimentTab and subclasses
            tab.set_controller(self.experiment_controller)
        elif isinstance(tab, GroupTab):
            # GroupTab and subclasses
            tab.set_controller(self.group_controller)
        tab.setup()
        # TODO: check if reparametrize required when folder changes or clear takes care of it
        tab.set_params()  #apply_patch=self._apply_ui_patch, get_view=self._get_config_view)
        return tab

    def _install_or_update_tabs(self) -> None:
        """
        Decide which tabs exist (via TabRegistry + validators/materializers),
        create or reuse instances, inject callbacks, and notify UI.
        """
        self._tabs = self._build_tabs_from_registry()
        self.publish(TabsUpdated(titles=[t.name for t in self._tabs], tabs=self._tabs))

        if self.mode == AppMode.EXPERIMENT:
            names, partners = self.experiment_controller.channel_snapshot()
            self.publish(ChannelsSnapshot(names=names))
            self.publish(ChannelDefaultsChanged(partners=partners))

    def _build_tabs_from_registry(self) -> list[Any]:
        valid_tab_classes = self.tabs_registry.valid_tabs(mode=self.mode, sample_manager=self.sample_manager,
                                                          group_controller=self.group_controller)
        print(f'Valid tabs: {[cls.__name__ for cls in valid_tab_classes]}')

        new_tabs = []
        for i, cls in enumerate(valid_tab_classes):
            tab = self._get_or_create(cls, i)
            new_tabs.append(tab)
        return new_tabs

    # ---------- UI → business helpers ----------
    def _apply_ui_patch(self, patch: Dict[str, Any]) -> None:
        self.experiment_controller.apply_ui_patch(patch)

    def _get_config_view(self) -> Any:
        return self.experiment_controller.get_config_view

    # ---------- handlers ----------

    def _tabs_may_have_changed(self, changed_keys: Tuple[str, ...]) -> bool:
        """
        Infer whether the set of tabs may have changed based on which config keys changed.
        Typically if channels or their data types changed, tabs may have changed.

        Parameters
        ----------
        changed_keys: Tuple[str, ...]
            The keys that changed in the config

        Returns
        -------
        bool
            True if tabs may have changed, False otherwise
        """
        for k in changed_keys:
            if k == 'sample':  # Whole sample changed
                return True
            elif k.startswith('sample.channels.'):
                if k.count('.') == 2:  # channels list changed
                    return True
                elif k.endswith('.data_type'):  # data type of a channel changed
                    return True
        else:
            return False

    def _on_tab_activated(self, evt: UiTabActivated) -> None:
        ok, msg = self._handle_tab_activation(evt.key)
        self.publish(TabActivationResult(key=evt.key, ok=ok, message=msg))

    def _refresh_tabs_from_model(self):
        for tab in self._tabs:
            # REFACTOR: hide behind refresh() method on GenericTab?
            # FIXME: this will blow up. fix in cfg_to_ui but params must have ref to owner tab
            tab._create_channels()  # Ensure channel tabs and params are up to date
            tab.params.cfg_to_ui()

    def _handle_tab_activation(self, tab_key: str) -> Tuple[bool, str]:
        """
        Business rules when a tab is activated (called by UI). Returns (ok, msg).
        Example rules:
         - Post-processing tabs require registration/alignment first
         - Batch tabs get lazy initialization on first open
        """
        tab = self._find_tab_by_key(tab_key)
        if not tab:
            return False, f'Tab "{tab_key}" not found'

        if tab_key != 'sample' and self.sample_manager.workspace is None:
            return False, 'Workspace not initialised.'

        if tab.processing_type == 'post':  # TODO: check that required for all post tabs
            if not self.experiment_controller.worker_is_ready(tab_key):
                return False, 'Registration not completed. Please run alignment first.'

        tab.on_selected()  # let tab do lazy init if needed

        return True, ''

    def _find_tab_by_key(self, key: str) -> Optional[GenericTab]:
        """
        Find a tab instance by its key (e.g. 'sample', 'registration', etc).

        .. warning::
            The `key` parameter is expected to be in snake_case and is derived from the tab title
            (e.g. 'Sample info' → 'sample_info').

        Parameters
        ----------
        key: str
            The key of the tab to find (e.g. 'sample', 'registration', etc)
            The key is expected to be in snake_case and is derived from the tab title (e.g. 'Sample info' → 'sample_info')

        Returns
        -------
        Optional[GenericTab]
            The tab instance matching the key, or None if not found
        """
        for tab in self._tabs:
            if title_to_snake(tab.name) == key:
                return tab
        return None

    def _on_cfg_changed(self, evt: CfgChanged):
        self.sample_manager = self.experiment_controller.sample_manager  # to be sure
        if self.hydrating:  # Hydration: Cache and defer full refresh until hydration ends
            if self._tabs_may_have_changed(evt.changed_keys):
                self._needs_full_refresh = True
        else:  # Normal operation
            if self._tabs_may_have_changed(evt.changed_keys):  # Infer if channels/types changed
                self._install_or_update_tabs()
                self._tabs_initialized = True
            self._refresh_tabs_from_model()

    def _on_refresh_tabs(self, evt: UiRequestRefreshTabs):
        self._install_or_update_tabs()


def _make_bootstrap_dir() -> Path:
    """
    Create a temporary directory for this session to hold config files
    until the experiment folder is set by the user.
    We prefer a user-configured temp if available (CLEARMAP_TMP env var)
    otherwise use the system temp folder.
    The folder is named "clearmap_bootstrap/session_<pid>" to avoid clashes
    if multiple instances are running.
    The folder is removed on exit if it stayed unused (i.e. still a bootstrap dir).
    Returns
    -------
    Path
        The path to the temporary bootstrap directory
    """
    root = Path(os.environ.get("CLEARMAP_TMP", tempfile.gettempdir()))
    session = root / "clearmap_bootstrap" / f"session_{os.getpid()}"
    session.mkdir(parents=True, exist_ok=True)
    return session


def build_business_objects(bus: EventBus):
    bootstrap_dir = _make_bootstrap_dir()

    @atexit.register
    def _cleanup_bootstrap():
        try:
            if bootstrap_dir.name.startswith('session_'):
                shutil.rmtree(bootstrap_dir, ignore_errors=True)
        except Exception:
            pass

    cfg_coordinator_factory = make_cfg_coordinator_factory(bus)
    cfg_coordinator = cfg_coordinator_factory(bootstrap_dir,
                                              config_groups=(ALTERNATIVES_REG._pipeline_groups, ALTERNATIVES_REG._global_groups))

    cfg_coordinator.seed_missing_from_defaults(tabs_only=True)

    sample_manager = SampleManager(config_coordinator=cfg_coordinator, src_dir=None)
    exp_controller = ExperimentController(cfg_coordinator=cfg_coordinator,
                                          sample_manager=sample_manager, evt_bus=bus)
    analysis_group_controller = AnalysisGroupController(cfg_coordinator_factory,
                                                        event_bus=bus,
                                                        exp_controller_factory=ExperimentController)

    return exp_controller, analysis_group_controller


def main(app_, splash_):
    bus = EventBus()
    experiment, group_controller = build_business_objects(bus)
    tab_registry = TabRegistry()
    gui = GuiController(bus, experiment, tab_registry, group_controller)

    app_.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    gui.start(app_, centered=True)

    splash_.finish(gui.window)
    if gui.window.preference_editor.params.verbosity != 'trace':  # WARNING: will disable progress bars
        gui.window.error_logger.setup_except_hook()
    sys.exit(app_.exec())


def entry_point():
    main(app, splash)


if __name__ == "__main__":
    main(app, splash)
