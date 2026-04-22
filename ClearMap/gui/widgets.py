# -*- coding: utf-8 -*-
"""
widgets
=======

A set of custom widgets for the ClearMap GUI
"""
import os
import re
import tempfile
import functools
from math import floor
import getpass

import json
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import permutations
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Dict

import numpy as np
import psutil

from natsort import natsorted
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from qdarkstyle import DarkPalette

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import (QWidget, QDialogButtonBox, QListWidget, QListWidgetItem,
                             QLayout, QHBoxLayout, QVBoxLayout, QGridLayout,
                             QPushButton, QTableWidget, QTableWidgetItem, QToolBox, QRadioButton,
                             QTreeWidget, QTreeWidgetItem, QTabWidget, QFileDialog,
                             QAbstractItemView, QGroupBox, QButtonGroup, QLabel, QSlider, QFrame,
                             QCheckBox, QComboBox, QSpinBox, QApplication)

from ClearMap import Settings
from ClearMap.IO.assets_constants import DATA_CONTENT_TYPES
from ClearMap.IO.metadata import pattern_finders_from_base_dir, ChannelPatternSpec, PatternFinder
from ClearMap.Utils.utilities import gpu_params, bytes_to_human, trim_or_pad
from ClearMap.Visualization import Plot3d as plot_3d
from ClearMap.Visualization.Qt.widgets import Scatter3D
from ClearMap.config.atlas import STRUCTURE_TREE_NAMES_MAP
from ClearMap.config.config_handler import scan_folder_for_experiments

from ClearMap.gui import dialog_helpers as dlg_help
from ClearMap.gui.gui_utils_base import create_clearmap_widget, compute_grid, get_widget, delete_widget, clear_layout
from ClearMap.gui.gui_utils_images import get_pseudo_random_color, is_dark

USER_NAME = getpass.getuser()

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


# REFACTOR: belongs here ??
Pair = Tuple[str, str]


class OrthoViewer:
    """
    Orthogonal viewer for 3D images

    This is a class that allows to visualize 3D images in 3 orthogonal views.
    """
    def __init__(self, img=None, parent=None):
        """
        Initialize the viewer
        Parameters
        ----------
        img : np.ndarray
            The 3D image to visualize
        parent : QWidget
            The parent widget
        """
        self.img = img
        self.parent = parent
        self.no_scale = False
        self.params = None
        self.linear_regions = []
        self.dvs = []

    def setup(self, img, params, parent=None, no_scale=False):
        """
        Initialize the viewer after the object has been created

        Parameters
        ----------
        img : np.ndarray
            The 3D image to visualize
        params : UiParameter
            The parameters object
        parent : QWidget
            The parent widget
        """
        self.img = img
        self.params = params
        self.parent = parent
        self.no_scale = no_scale
        self.linear_regions = []

    @property
    def shape(self):
        """
        Get the shape of the image

        Returns
        -------
        tuple(int, int, int)
        """
        return self.img.shape if self.img is not None else None

    @property
    def width(self):
        """
        Get the width of the image

        Returns
        -------
        int
        """
        return self.shape[0]

    @property
    def height(self):
        """
        Get the height of the image

        Returns
        -------
        int
        """
        return self.shape[1]

    @property
    def depth(self):
        """
        Get the depth of the image

        Returns
        -------
        int
        """
        return self.shape[2]

    def update_ranges(self, ranges):
        """
        Update the ranges (min, max) for each axis of the viewer

        Parameters
        ----------
        ranges : list(tuple(float, float))
        """
        for i, rng in enumerate(ranges):
            region_item = self.linear_regions[i]
            region_item.setRegion(rng)
            self.__update_range(region_item, axis=i)

    def __update_range(self, region_item, axis=0):
        rng = region_item.getRegion()

        # Debounce
        cache = getattr(self, '_range_cache', {})
        if cache.get(axis) == rng:
            return
        cache[axis] = rng
        self._range_cache = cache

        if self.params is not None:
            if not self.no_scale:
                rng = [self.params.scale_axis(val, 'xyz'[axis]) for val in rng]
            setattr(self.params, f'crop_{"xyz"[axis]}', rng)

    def add_regions(self):  # FIXME: improve documenation
        """
        Add the regions to the viewer
        """
        # y_axis_idx = (1, 2, 0)
        for i, dv in enumerate(self.dvs):
            transparency = '4B'  # 75% transparency
            linear_region = pg.LinearRegionItem([0, self.shape[i]], brush=DarkPalette.COLOR_BACKGROUND_2 + transparency)
            linear_region.sigRegionChanged.connect(functools.partial(self.__update_range, axis=i))
            self.linear_regions.append(linear_region)
            dv.view.addItem(linear_region)

    def plot_orthogonal_views(self, img=None, parent=None):
        """
        Plot the orthogonal views of the image

        Parameters
        ----------
        img : np.ndarray
            The image to plot. If None, the image set at initialization will be used
        parent : QWidget
            The parent widget to plot into. If None, the parent set at initialization will be used

        Returns
        -------
        list(DataViewer)
        """
        if img is None:
            img = self.img.array
        if parent is None:
            parent = self.parent
            if parent is None:
                raise ValueError('Parent not set')
        xy = np.copy(img)
        yz = np.copy(img).transpose((1, 2, 0))
        zx = np.copy(img).transpose((2, 0, 1))
        dvs = plot_3d.plot([xy, yz, zx], arrange=False, lut='white', parent=parent, sync=False)
        self.dvs = dvs
        self.add_regions()
        for dv in self.dvs:
            for btn in dv.axis_buttons:
                btn.setEnabled(False)
        return dvs


class ProgressWatcher(QWidget):  # Inspired from https://stackoverflow.com/a/66266068
    """
    A QWidget that watches the progress of a process. It uses signals to update the progress bar and the text
    The main setup methods are `setup` and `prepare_for_substep`
    It is meant to be used in conjunction with a ProgressWatcherDialog to which it is connected
    through its signals
    """
    main_step_name_changed = QtCore.pyqtSignal(str)
    sub_step_name_changed = QtCore.pyqtSignal(str)

    main_progress_changed = QtCore.pyqtSignal(int)
    main_max_changed = QtCore.pyqtSignal(int)

    progress_changed = QtCore.pyqtSignal(int)
    max_changed = QtCore.pyqtSignal(int)

    finished = QtCore.pyqtSignal(str)
    aborted = QtCore.pyqtSignal(bool)  # FIXME: use

    def __init__(self, max_progress=100, main_max_progress=1, timer_interval_ms=250, parent=None):
        """
        Create a ProgressWatcher

        Parameters
        ----------
        max_progress : int
            The maximum progress value, when the progress reaches this value, the (sub-)operation is considered finished.
            default is 100
        main_max_progress : int
            The maximum progress value for the main operation. When the progress reaches this value, the main operation
            is considered finished. If all sub-operations are also finished, this is usually linked to the end of the
            whole process. Default is 1
        parent : QWidget
            The parent widget
        """
        super().__init__(parent)
        self._main_step_name = 'Processing'
        self._sub_step_name = None
        self.__main_progress = 1
        self.__main_max_progress = main_max_progress
        self.__progress = 0
        self.__max_progress = max_progress
        self.range_fraction = 1  # FIXME: unused

        self.n_dones = 0
        self.previous_log_length = 0  # The log length at the end of the previous operation
        self.log_path = None
        self.pattern = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._interval_ms = timer_interval_ms

    def __del__(self):
        self.set_main_progress(self.main_max_progress)
        self.set_progress(self.max_progress)
        if self.parentWidget() is not None:
            self.parentWidget().app.processEvents()

    def reset(self):
        """Reset all the values to their initial state"""
        self.main_step_name = 'Processing'
        self.__main_progress = 1
        self.__main_max_progress = 1
        self.__progress = 0
        self.__max_progress = 100
        self.range_fraction = 1  # FIXME: unused

        self.n_dones = 0
        self.previous_log_length = 0  # The log length at the end of the previous operation
        self.log_path = None
        self.pattern = None

    def setup(self, main_step_name, main_step_length, sub_step_length=0, pattern=None):
        """
        Post initialisation of the

        Parameters
        ----------
        main_step_name
        main_step_length
        sub_step_length
        pattern
        """
        self.main_step_name = main_step_name
        self.main_max_progress = main_step_length
        # self.sub_step_name = sub_step_name
        self.max_progress = sub_step_length
        self.pattern = pattern

        self.reset_log_length()
        self.set_main_progress(1)
        self.set_progress(0)
        # Force update
        self.main_progress_changed.emit(self.__main_progress)
        self.progress_changed.emit(self.__progress)

    def set_poll_interval(self, ms: int):
        self._interval_ms = ms
        if self._timer.isActive():
            self._timer.start(self._interval_ms)

    def start_polling(self):
        if not self.log_path:  # If not setup
            return
        self.reset_log_length()
        self._timer.start(self._interval_ms)

    def stop_polling(self):
        self._timer.stop()

    def _on_tick(self):
        if not self.pattern or not self.log_path:
            return
        dones = self.count_dones()
        self.set_progress(dones)

    def prepare_for_substep(self, step_length, pattern, step_name):
        """
        Setup the watcher for a new substep

        Parameters
        ----------
        step_name
            str
        step_length
            int The number of steps in the operation
        pattern
            str or re.Pattern or (str, re.Pattern) the text to look for in the logs to check for progress
        """
        self.max_progress = step_length
        self.pattern = pattern
        self.reset_log_length()
        self.set_progress(0)
        self.sub_step_name = step_name

    def get_progress(self):
        """
        Get the current progress

        Returns
        -------
        int
        """
        return self.__progress

    def set_progress(self, value):
        """
        Set the progress value of the current main or sub step

        Parameters
        ----------
        value: int
            The progress value
        """
        if self.__progress == value:
            return
        self.__progress = round(value)
        self.progress_changed.emit(self.__progress)

    def set_main_progress(self, value):
        """
        Set the progress value for the main step

        Parameters
        ----------
        value: int
            The progress value
        """
        if self.__main_progress == value:
            return
        self.__main_progress = round(value)
        self.reset_log_length()
        self.main_progress_changed.emit(self.__main_progress)
        if self.__main_progress != 0 and self.__main_progress == self.main_max_progress + 1:
            self.finished.emit(self.main_step_name)

    def increment_main_progress(self, increment=1):
        """
        Integer increment of the main progress

        Parameters
        ----------
        increment: int
            The increment value (default is 1)
        """
        self.set_main_progress(self.__main_progress + round(increment))

    def increment(self, increment):
        """
        Increment the progress value of the current main or sub step

        Parameters
        ----------
        increment: int or float
            The increment value. If float, it is considered as a percentage of the maximum progress value
        """
        if isinstance(increment, float):
            self.set_progress(self.__progress + int(self.max_progress * increment))
        elif isinstance(increment, int):
            self.set_progress(self.__progress + increment)

    @property
    def max_progress(self):
        return self.__max_progress

    @max_progress.setter
    def max_progress(self, value):
        self.__max_progress = round(value)
        self.max_changed.emit(self.__max_progress)

    @property
    def main_max_progress(self):
        return self.__main_max_progress

    @main_max_progress.setter
    def main_max_progress(self, value):
        self.__main_max_progress = round(value)
        self.main_max_changed.emit(self.__main_max_progress)

    @property
    def main_step_name(self):
        return self._main_step_name

    @main_step_name.setter
    def main_step_name(self, step_name):
        self._main_step_name = step_name
        self.main_step_name_changed.emit(self.main_step_name)

    @property
    def sub_step_name(self):
        return self._sub_step_name

    @sub_step_name.setter
    def sub_step_name(self, step_name):
        self._sub_step_name = step_name
        self.sub_step_name_changed.emit(self.sub_step_name)

    def __match(self, line):
        if isinstance(self.pattern, tuple):  # TODO: cache
            return self.pattern[0] in line and self.pattern[1].match(line)  # Most efficient
        elif isinstance(self.pattern, str):
            return self.pattern in line
        elif isinstance(self.pattern, re.Pattern):
            return self.pattern.match(line)

    def count_dones(self):
        """
        Parse the logs to extract the number of `done` operations (based on self.pattern)
        For each `done` operation, the progress is incremented by 1
        For efficiency, the logs are read from the last read position

        Returns
        -------
        int
            The number of `done` operations found
        """
        if self.pattern is None:
            return 0
        with open(self.log_path, 'r') as log:
            log.seek(self.previous_log_length)
            new_lines = log.readlines()
        n_dones = len([ln for ln in new_lines if self.__match(ln)])
        self.n_dones += n_dones
        self.previous_log_length += self.__get_log_bytes(new_lines)
        return self.n_dones

    def reset_log_length(self):
        """
        Resets dones to 0 and seeks to the end of the log file
        """
        with open(self.log_path, 'r') as log:
            self.previous_log_length = self.__get_log_bytes(log.readlines())
            self.n_dones = 0

    def __get_log_bytes(self, log):
        return sum([len(ln) for ln in log])

    def finish(self):
        """
        Trigger the finish signal
        """
        self.finished.emit(self.main_step_name)


# Adapted from https://stackoverflow.com/a/54917151 by https://stackoverflow.com/users/6622587/eyllanesc
class TwoListSelection(QWidget):
    itemSelectionChanged = pyqtSignal(str)
    """
    A widget that allows to select items from a list and move them to another list
    This is useful for selecting items from a list of available items and moving them to a list of selected items
    """
    def __init__(self, parent=None, input_title=None, output_title=None):
        super().__init__(parent)
        self._input_title  = input_title      # keep for setup
        self._output_title = output_title
        self.__setup_layout()
        # self.app = app

    def __setup_layout(self):
        """
        Setup the layout of the widget with the two columns for the lists and the buttons
        """
        lyt = QHBoxLayout(self)
        self.mInput = QListWidget()
        self.mOutput = QListWidget()

        # enable Ctrl-/Shift-selection
        self.mInput.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mOutput.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.mInput.itemSelectionChanged.connect(functools.partial(self.on_selection_changed, self.mInput))
        self.mOutput.itemSelectionChanged.connect(functools.partial(self.on_selection_changed, self.mOutput))

        # --------------- wrap in group-boxes if titled ------------
        if self._input_title:
            box_in  = QGroupBox(self._input_title)
            box_in.setLayout(QVBoxLayout())
            box_in.layout().addWidget(self.mInput)
            left_widget = box_in
        else:
            left_widget = self.mInput

        if self._output_title:
            box_out = QGroupBox(self._output_title)
            box_out.setLayout(QVBoxLayout())
            box_out.layout().addWidget(self.mOutput)
            right_widget = box_out
        else:
            right_widget = self.mOutput


        move_btns, up_down_btns = self.__layout_buttons()

        lyt.addWidget(left_widget)
        lyt.addLayout(move_btns)
        lyt.addWidget(right_widget)
        lyt.addLayout(up_down_btns)

        self.update_buttons_status()
        self.__connections()

    def __layout_buttons(self):
        """
        Create and lay out the control buttons of the widget
        """
        self.mButtonToSelected = QPushButton(">>")
        self.mBtnMoveToAvailable = QPushButton(">")
        self.mBtnMoveToSelected = QPushButton("<")
        self.mButtonToAvailable = QPushButton("<<")
        move_btns = QVBoxLayout()
        move_btns.addStretch()
        move_btns.addWidget(self.mButtonToSelected)
        move_btns.addWidget(self.mBtnMoveToAvailable)
        move_btns.addWidget(self.mBtnMoveToSelected)
        move_btns.addWidget(self.mButtonToAvailable)
        move_btns.addStretch()

        self.mBtnUp = QPushButton("Up")
        self.mBtnDown = QPushButton("Down")
        up_down_btns = QVBoxLayout()
        up_down_btns.addStretch()
        up_down_btns.addWidget(self.mBtnUp)
        up_down_btns.addWidget(self.mBtnDown)
        up_down_btns.addStretch()

        return move_btns, up_down_btns

    def on_selection_changed(self, list_widget):
        sel = list_widget.selectedItems()      # list[QListWidgetItem]
        if len(sel) == 1:                      # exactly one row picked
            item = sel[0]
            self.itemSelectionChanged.emit(item.text())

    @QtCore.pyqtSlot()
    def update_buttons_status(self):
        self.mBtnUp.setDisabled(not bool(self.mOutput.selectedItems()) or self.mOutput.currentRow() == 0)
        self.mBtnDown.setDisabled(not bool(self.mOutput.selectedItems()) or self.mOutput.currentRow() == (self.mOutput.count() - 1))
        self.mBtnMoveToAvailable.setDisabled(not bool(self.mInput.selectedItems()) or self.mOutput.currentRow() == 0)
        self.mBtnMoveToSelected.setDisabled(not bool(self.mOutput.selectedItems()))

    def __connections(self):
        """
        Bind the buttons to their slots
        """
        self.mInput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mOutput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mBtnMoveToAvailable.clicked.connect(self.__on_mBtnMoveToAvailable_clicked)
        self.mBtnMoveToSelected.clicked.connect(self.__on_mBtnMoveToSelected_clicked)
        self.mButtonToAvailable.clicked.connect(self.__on_mButtonToAvailable_clicked)
        self.mButtonToSelected.clicked.connect(self.__on_mButtonToSelected_clicked)
        self.mBtnUp.clicked.connect(self.__on_mBtnUp_clicked)
        self.mBtnDown.clicked.connect(self.__on_mBtnDown_clicked)

    @QtCore.pyqtSlot()
    def __on_mBtnMoveToAvailable_clicked(self):
        """move *all* selected rows from left → right"""
        for it in reversed(self.mInput.selectedItems()):          # reversed keeps order
            self.mOutput.addItem(self.mInput.takeItem(self.mInput.row(it)))

    @QtCore.pyqtSlot()
    def __on_mBtnMoveToSelected_clicked(self):
        """move *all* selected rows from right → left"""
        for it in reversed(self.mOutput.selectedItems()):
            self.mInput.addItem(self.mOutput.takeItem(self.mOutput.row(it)))

    @QtCore.pyqtSlot()
    def __on_mButtonToAvailable_clicked(self):
        while self.mOutput.count() > 0:
            self.mInput.addItem(self.mOutput.takeItem(0))

    @QtCore.pyqtSlot()
    def __on_mButtonToSelected_clicked(self):
        while self.mInput.count() > 0:
            self.mOutput.addItem(self.mInput.takeItem(0))

    @QtCore.pyqtSlot()
    def __on_mBtnUp_clicked(self):
        row = self.mOutput.currentRow()
        currentItem = self.mOutput.takeItem(row)
        self.mOutput.insertItem(row - 1, currentItem)
        self.mOutput.setCurrentRow(row - 1)

    @QtCore.pyqtSlot()
    def __on_mBtnDown_clicked(self):
        row = self.mOutput.currentRow()
        currentItem = self.mOutput.takeItem(row)
        self.mOutput.insertItem(row + 1, currentItem)
        self.mOutput.setCurrentRow(row + 1)

    # The actual user functions
    def clear(self):
        """
        Clear the lists
        """
        self.mInput.clear()
        self.mOutput.clear()

    def __add_item(self, widget: QListWidget, text, user_data=None):
        """
        Helps attach user_data.  *text* always becomes the visible label.
        """
        item = QListWidgetItem(str(text))
        if user_data is not None:
            item.setData(Qt.UserRole, user_data)
        widget.addItem(item)

    def addAvailableItems(self, items):
        """
        Add the list of available items to the left list

        Parameters
        ----------
        items: list(str)
            The list of items to add
        """
        for itm in items:
            if isinstance(itm, (tuple, list)) and len(itm) == 2:
                self.__add_item(self.mInput, itm[0], itm[1])
            else:  # plain string
                self.__add_item(self.mInput, itm)

    def setSelectedItems(self, items):
        """
        Add the list of selected items to the right list
        Parameters
        ----------
        items: list(str)
            The list of items to add
        """
        self.mOutput.clear()
        for itm in items:
            if isinstance(itm, (tuple, list)) and len(itm) == 2:
                self.__add_item(self.mOutput, itm[0], itm[1])
            else:
                self.__add_item(self.mOutput, itm)

    def __get_elements(self, list_widget, with_data=False):
        """
        Get the list of items in the list

        Parameters
        ----------
        list_widget: QListWidget
            The list widget to get the items from
        with_data: bool
            If True, return the data associated with the items

        Returns
        -------
        list(str)
            The list of items in the list
        """
        r = []
        for i in range(list_widget.count()):
            it = list_widget.item(i)
            if with_data:
                r.append((it.text(), it.data(Qt.UserRole)))
            else:
                r.append(it.text())
        return r

    def get_left_elements(self, with_data=False):
        """
        Get the list of items in the left list (available items)

        Returns
        -------
        list(str)
        """
        return self.__get_elements(self.mInput, with_data)

    def get_right_elements(self, with_data=False):
        """
        Get the list of items in the right list (selected items)

        Returns
        -------
        list(str)
        """
        return self.__get_elements(self.mOutput, with_data)


class CheckableListWidget(QWidget):
    check_state_changed = pyqtSignal(int, bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        self.layout.addWidget(self.list_widget)
        self.setLayout(self.layout)
        self.list_widget.itemChanged.connect(self.on_item_changed)

    def clear(self):
        self.list_widget.clear()

    def set_items(self, items):
        self.list_widget.clear()
        for item in items:
            list_item = QListWidgetItem(item)
            list_item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(list_item)

    def get_checked_items(self):
        checked_items = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                checked_items.append(item.text())
        return checked_items

    def check_item_at(self, index):
        if 0 <= index < self.list_widget.count():
            item = self.list_widget.item(index)
            item.setCheckState(Qt.Checked)

    def set_item_checked(self, item_name, state):
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.text() == item_name:
                item.setCheckState(Qt.Checked if state else Qt.Unchecked)
                return

    def check_items(self, items_list):
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.text() in items_list:
                item.setCheckState(Qt.Checked)

    def add_item(self, item):
        if self.get_item(item) is not None:
            return
        list_item = QListWidgetItem(item)
        list_item.setCheckState(Qt.Unchecked)
        self.list_widget.addItem(list_item)

    def get_item(self, item_name):
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.text() == item_name:
                return item
        return None

    def delete_item(self, index):
        if 0 <= index < self.list_widget.count():
            self.list_widget.takeItem(index)

    def on_item_changed(self, item):
        index = self.list_widget.row(item)
        checked = item.checkState() == Qt.Checked
        self.check_state_changed.emit(index, checked, item.text())


class DataFrameWidget(QWidget):  # TODO: optional format attribute with shape of df
    """
    A simple widget to display a pandas DataFrame
    """
    def __init__(self, df, n_digits=2, parent=None):
        """
        Initialize the widget

        Parameters
        ----------
        df: pd.DataFrame
            The DataFrame to display
        n_digits: int
            The number of digits to display for float values
        parent: QWidget
            The parent widget to attach the widget to
        """
        super().__init__(parent)
        self.df = df
        self.n_digits = n_digits
        self.table = QTableWidget(len(df.index), df.columns.size, parent=parent)
        self.table.setHorizontalHeaderLabels(self.df.columns)
        self._set_content()

    def _set_content(self):
        for i in range(len(self.df.index)):
            for j in range(self.df.columns.size):
                v = self.df.iloc[i, j]
                if self.df.dtypes[j] == np.float_:
                    self.table.setItem(i, j, QTableWidgetItem(f'{v:.{self.n_digits}}'))
                else:
                    self.table.setItem(i, j, QTableWidgetItem(f'{v}'))


class WizardWidget:
    """
    A base class for a complex dialogs designed with QT creator and exported as ui files.
    It is meant to be subclassed. The subclass should implement the `setup` and `connect_buttons` methods
    This class needs a src_folder, a ui_name, a ui_title. The ui names and titles are used to build
    a dialog from the ui file of the same name and parametrise the dialog
    """
    def __init__(self, ui_name, ui_title, src_folder="", params=None, app=None, size=None, patch_parent_class='QDialog'):
        """
        Initialize the dialog

        Parameters
        ----------
        src_folder: str
            The source folder
        ui_name: str
            The name of the ui file to use. ClearMap automatically locates and loads the file
        ui_title: str
            The title of the dialog
        size: tuple(int, int) (optional)
            The size of the dialog. If None, the dialog will be resized to its content
        params: UiParameter (optional)
            The parameters object to use to parametrise the dialog
        app: QApplication (optional)
            The QApplication instance to use. If None, ClearMap will try to use the existing instance
        """
        self.src_folder = src_folder
        self.params = params
        self.app = app or QtWidgets.QApplication.instance()

        dlg = create_clearmap_widget(ui_name, patch_parent_class=patch_parent_class, window_title=ui_title)

        if size is not None:
            if size[0] is None:
                size[0] = dlg.width()
            if size[1] is None:
                size[1] = dlg.height()
            dlg.resize(size[0], size[1])

        self.dlg = dlg

        self.setup()

        # self.dlg.setStyleSheet(qdarkstyle.load_stylesheet())
        self.__fix_btn_boxes_text()
        self.connect_buttons()

    # Abstract
    def setup(self):
        """
        Setup the dialog after creation from the ui file.
        This method is called automatically in the constructor
        but should be implemented in the subclass
        """
        raise NotImplementedError

    def __fix_btn_boxes_text(self):
        """
        Patch the name of button boxes to match the text in the ui file
        """
        for btn_box in self.dlg.findChildren(QDialogButtonBox):
            if btn_box.property('applyText'):
                btn_box.button(QDialogButtonBox.Apply).setText(btn_box.property('applyText'))

    def connect_buttons(self):
        """
        Connect the buttons to their slots.
        This method is called automatically in the constructor
        but should be implemented in the subclass
        """
        raise NotImplementedError

    @staticmethod
    def enable_widgets(widgets):
        """
        Helper method to enable a list of widgets

        Parameters
        ----------
        widgets: list(QWidget)
            The list of widgets to enable
        """
        for w in widgets:
            w.setEnabled(True)

    @staticmethod
    def hide_widgets(widgets):
        """
        Helper method to hide a list of widgets

        Parameters
        ----------
        widgets: list(QWidget)
            The list of widgets to hide
        """
        for w in widgets:
            w.setVisible(False)

    def exec(self):
        """
        Execute the dialog
        """
        if hasattr(self.dlg, "exec"):  # QDialog
            return self.dlg.exec()
        else:  # plain QWidget / QDockWidget
            return self.dlg.show()


class ManageAssetsWidget(WizardWidget):
    # WARNING: need to check between asset type_name, basename and asset for mapping
    def __init__(self, src_folder, params, sample_manager, app=None):
        self.assets = {}  # TODO: exclude is_folder assets and not asset.exists
                          #     maybe exclude asset if asset.status == True
        self.sample_manager = sample_manager
        self.resampling_params = {'x_scale': 1, 'y_scale': 1, 'z_scale': 1,
                                  'x_shape': None, 'y_shape': None, 'z_shape': None,
                                  'x_resolution': None, 'y_resolution': None, 'z_resolution': None}
        self.ortho_viewer = OrthoViewer()
        self.list_selection = TwoListSelection(input_title='Filtered assets:',
                                               output_title='Selected assets:')

        super().__init__('assets_manipulation', 'Assets management wizard',
                         src_folder, params, app, [600, 1200], 'QWidget')

        # self.dlg.setModal(False)
        # self.dlg.show()
        self.widget = self.dlg
        self.__set_assets()

    def setup(self):
        self.dlg.channelsComboBox.clear()
        chans = self.sample_manager.workspace.channels
        chans = ['-'.join(c) if isinstance(c, tuple) else c for c in chans]
        self.dlg.channelsComboBox.addItems(['Any'] + chans)
        self.dlg.channelsComboBox.setCurrentIndex(0)

        self.dlg.fileFormatCategoryComboBox.clear()
        self.dlg.fileFormatCategoryComboBox.addItems(['Any'] + list(STRUCTURE_TREE_NAMES_MAP.keys()))
        self.dlg.fileFormatCategoryComboBox.setCurrentIndex(0)

        self.dlg.listsLayout.addWidget(self.list_selection)

    def connect_buttons(self):
        # self.dlg.assetManipulationButtonBox.accepted.connect(self.__apply_changes)
        # self.dlg.assetManipulationButtonBox.rejected.connect(self.dlg.close)

        self.dlg.existingOnlyCheckBox.toggled.connect(self.__set_assets)
        self.dlg.nameFilterPlainLineEdit.editingFinished.connect(self.__set_assets)
        self.dlg.channelsComboBox.currentTextChanged.connect(self.__set_assets)
        self.dlg.fileFormatCategoryComboBox.currentTextChanged.connect(self.__set_assets)
        self.dlg.sizeMinSpinBox.valueChanged.connect(self.__set_assets)
        self.dlg.sizeMaxSpinBox.valueChanged.connect(self.__set_assets)
        # self.dlg.filterByChannelCheckBox.toggled.connect(functools.partial(self.__set_assets, channel=None))
        # self.dlg.fileFormatCategoryCheckBox.toggled.connect(functools.partial(self.__set_assets, channel=None))

        selected_model = self.list_selection.mOutput.model()
        # selected_model.rowsInserted.connect(self.__update_assets)  # Update group when selection updated
        # selected_model.rowsRemoved.connect(self.__update_assets)  # Update group when selection updated

        actions = ['compress', 'decompress', 'convert', 'plot',
                   'delete', 'resample', 'crop']
        for action_name in actions:
            btn = QPushButton(action_name.title(), self.dlg)
            btn.clicked.connect(functools.partial(self.action, action_name))
            self.dlg.actionsVerticalLayout.addWidget(btn)

        self.list_selection.itemSelectionChanged.connect(self.handle_selection_changed)

    def handle_selection_changed(self, itm_text):
        """
        Handle the selection change in the list_selection widget.
        This will update the asset info text browser with the information of the selected asset

        Parameters
        ----------
        itm_text: str
            The text of the selected item in the list_selection widget.
            This is a string representation of the asset type and channel, e.g. "(channel, asset_type)"
        """
        self.dlg.assetInfoTextBrowser.clear()
        txt = ''
        if itm_text:
            asset = self.asset_types_to_assets([itm_text])[0]
            if asset is not None:
                txt += f'Channel: {asset.channel_spec.name}\n'
                txt += f'Asset type: {asset.type_spec.name}\n'
                txt += f'File: \n'
                txt += f'    format: {asset.type_spec._file_format_category}\n'
                txt += f'    path: {asset.path}\n'
                if not asset.is_folder and asset.exists:
                    txt += f'    size: {bytes_to_human(asset.size)}\n'
                    txt += f'    extensions found: {asset.existing_extension}\n'
                if asset.compressed_path.exists():
                    txt += f'Compressed file:\n'
                    txt += f'    path: {asset.compressed_path}\n'
                    txt += f'    size: {bytes_to_human(asset.compressed_path.stat().st_size)}\n'
                if asset.is_existing_source:
                    try:
                        txt += f'Source info:\n'
                        txt += f'    shape: {asset.shape()}\n'
                        txt += f'    dtype: {asset.as_source().dtype}\n'
                    except ValueError as err:
                        if self.app is not None:
                            base_msg = f'Source info error with {asset.type_spec.name}'
                            self.app.popup(msg=str(err), base_msg=base_msg, print_warning=True)
                        else:
                            raise err
                if hasattr(asset.type_spec, 'description'):
                    txt += f'Description: {asset.type_spec.description}\n'
            self.dlg.assetInfoTextBrowser.append(txt)

    @property
    def channel(self):
        return self.dlg.channelsComboBox.currentText()

    def __set_assets(self, channel=None):
        """
        Filter assets and populate the list_selection widget with the filtered assets
        This method is called when a filter criterion is changed

        Parameters
        ----------
        channel

        Returns
        -------

        """
        self.list_selection.clear()
        self.assets = {}
        if self.channel == 'Any':
            channels = self.sample_manager.workspace.channels  # All channels of WS, not only data ones
        else:
            if not channel:
                channel = self.channel
                if channel == 'None':
                    channel = None
            channels = [channel]

        for channel in channels:
            self.assets.update({(channel, asset_type): asset for asset_type, asset in
                                self.sample_manager.workspace.asset_collections[channel].items()})

        if self.dlg.existingOnlyCheckBox.isChecked():
            self.assets = {k: asset for k, asset in self.assets.items() if asset.exists}

        if self.dlg.fileFormatCategoryComboBox.currentText() != 'Any':
            selected_format = self.dlg.fileFormatCategoryComboBox.currentText()
            self.assets = {k: asset for k, asset in self.assets.items() if
                           asset.type_spec._file_format_category == selected_format}

        if self.dlg.nameFilterPlainLineEdit.text():
            name_filter = self.dlg.nameFilterPlainLineEdit.text()
            self.assets = {k: asset for k, asset in self.assets.items() if
                           name_filter in asset.type_spec.name}

        size_min = self.dlg.sizeMinSpinBox.value()
        size_max = self.dlg.sizeMaxSpinBox.value()

        if size_max != 0:
            self.assets = {k: asset for k, asset in self.assets.items() if
                           asset.exists and
                           (size_min == 0 or asset.size >= size_min) and
                           (size_max == 0 or asset.size <= size_max)}

        self.list_selection.addAvailableItems(list(self.assets.items()))

    @property
    def selected_assets(self):
        return self.asset_types_to_assets(self.selected_asset_types)

    @property
    def selected_asset_types(self):
        return self.list_selection.get_right_elements()

    def asset_types_to_assets(self, asset_names):  #  TEST:
        asset_names = [literal_eval(asset_keys) for asset_keys in asset_names]
        return [self.sample_manager.get(asset_type, channel=channel) for channel, asset_type in asset_names]

    def action(self, action_name):
        """
        Perform the specified action on the selected assets.
        This will broadcast the action to the appropriate method of the sample manager

        Parameters
        ----------
        action_name: str
            The name of the action to perform
        """
        assets = self.selected_assets

        params = {}
        if action_name in ('decompress', 'convert', 'resample', 'crop'):
            params = self.prompt_params(action_name)

        if all([hasattr(asset, action_name) for asset in assets]):
            if action_name == 'plot':  # FIXME: add menu for overlap, side by side ...
                sources = [asset.path for asset in assets]
                if all([asset.shape() == assets[0].shape() for asset in assets]):
                    sources = [sources]  # overlay
                plot_3d.plot(sources, arrange=False, lut='grey')  # REFACTORING: in WS2 ?
            for asset in assets:
                getattr(asset, action_name)(**params)
        else:
            method = getattr(self.sample_manager, f'{action_name}_assets')
            # WARNING: resample and crop will need extra dialog to get the parameters

            if params:
                method(assets, **params)
            else:
                method(assets)

    def prompt_params(self, action_name):
        """
        Create a new dialog to prompt the user for the additional parameters of the specified action

        Parameters
        ----------
        action_name: str
            The name of the action to perform

        Returns
        -------
        dict
            The parameters to use for the action
        """
        params = {}
        if action_name == 'decompress':
            params['check'] = dlg_help.prompt_dialog('Decompression',
                                                     'Do you want to verify the integrity of the files?')
        if action_name == 'convert':
            params['processes'] = cpu_count() - 2
            for asset in self.selected_assets:
                if asset.type_spec.extensions:
                    extensions = asset.type_spec.extensions
                    break
            else:
                raise ValueError('No extension found in the selected assets')
            idx = dlg_help.option_dialog('Select the output format',
                                         'Convert the selected asset to the following format',
                                         extensions)
            params['new_extension'] = extensions[idx]
        elif action_name == 'resample':
            self.resample_dialog()
            params = self.resampling_params
        elif action_name == 'crop':
            dlg_help.prompt_dialog('Crop', 'WARNING: all files will be cropped to the same region')
            self.crop_dialog()  # Use the OrthoViewer to select the crop region
            params = self.ortho_viewer.params  # FIXME: check if correct
        return params

    def crop_dialog(self):
        dlg = create_clearmap_widget('crop_dialog.ui', patch_parent_class='QDialog')  # FIXME: create ui file
        self.ortho_viewer.setup(self.selected_assets[0].source, self.params, dlg)
        dvs = self.ortho_viewer.plot_orthogonal_views()

        n_rows, n_cols = compute_grid(len(dvs))
        n_spacers = (n_rows * n_cols) - len(dvs)
        for i in range(n_spacers):
            spacer = QWidget(parent=self)
            dvs.append(spacer)
            # graph_names.append(f'spacer_{i}')

        margin = 9
        spacing = 6
        for i, dv in enumerate(dvs):
            # dv.setObjectName(graph_names[i])
            row = i // n_cols
            col = i % n_cols
            if len(dvs) > 1:
                width = floor((dlg.width() - (2 * margin) - (n_cols - 1) * spacing) / n_cols)
                height = floor((dlg.height() - (2 * margin) - (n_rows - 1) * spacing) / n_rows)
                dv.resize(width, height)
                dv.setMinimumSize(width, height)  # required to avoid wobbly dv
            dlg.graphLayout.addWidget(dv, row, col, 1, 1)
        self.app.processEvents()

    def assert_all_images(self):
        if not all([asset.is_existing_source for asset in self.selected_assets]):
            dlg_help.warning_popup('All assets must have a source image to crop')
            return False
        else:
            return True

    def resample_dialog(self):
        dlg = create_clearmap_widget('resample_dialog.ui', patch_parent_class='QDialog')  # FIXME: create ui file
        dlg.xScaleSpinBox.valueChanged.connect(functools.partial(self.update_resample_params, 'x_scale'))
        dlg.yScaleSpinBox.valueChanged.connect(functools.partial(self.update_resample_params, 'y_scale'))
        dlg.zScaleSpinBox.valueChanged.connect(functools.partial(self.update_resample_params, 'z_scale'))

        dlg.xShapeSpinBox.valueChanged.connect(functools.partial(self.update_resample_params, 'x_shape'))
        dlg.yShapeSpinBox.valueChanged.connect(functools.partial(self.update_resample_params, 'y_shape'))
        dlg.zShapeSpinBox.valueChanged.connect(functools.partial(self.update_resample_params, 'z_shape'))

        dlg.xResSpinBox.valueChanged.connect(functools.partial(self.update_resample_params, 'x_resolution'))
        dlg.yResSpinBox.valueChanged.connect(functools.partial(self.update_resample_params, 'y_resolution'))
        dlg.zResSpinBox.valueChanged.connect(functools.partial(self.update_resample_params, 'z_resolution'))
        dlg.onAcceptButton.clicked.connect(self.resample_assets)
        dlg.exec()

    def update_resample_params(self, param_name, value):
        self.resampling_params[param_name] = value

    def resample_assets(self):
        resampling_params = {k: v for k, v in self.resampling_params.items() if v not in (1, None)}
        self.sample_manager.resample_assets(self.selected_assets, processes=cpu_count() -2, **resampling_params)


class PatternDialog(WizardWidget):
    """
    A wizard dialog to help the user define file patterns for a set of image file paths
    The dialog scans the source folder to find patterns in the file names and suggests them to the user
    there must be at least `min_file_number` files in the folder with the extension `tile_extension`
    to trigger the pattern search
    """
    def __init__(self, src_folder, params, app=None, min_file_number=10, tile_extension='.ome.tif'):
        """
        Initialize the dialog

        Parameters
        ----------
        src_folder: str
            The source folder to scan for patterns
        params: UiParameter (optional)
            The parameters object to use to parametrise the dialog
        app: QApplication (optional)
            The QApplication instance to use. If None, ClearMap will try to use the existing instance
        min_file_number: int (optional)
            The minimum number of files to trigger the pattern search. Default is 10
        tile_extension: str (optional)
            The extension of the files to consider. Default is '.ome.tif'
        """
        self.min_file_number: int = min_file_number
        self.tile_extension: str = tile_extension

        self.n_image_groups: int = 0
        self.patterns_finders: List[PatternFinder] = []
        self._pattern_results: List[ChannelPatternSpec] = []

        # Init at the end to not overwrite result of setup
        super().__init__('pattern_prompt', 'File paths wizard', src_folder, params, app, [600, None])

    def setup(self):
        """
        Setup the dialog after creation from the ui file.
        This method is called automatically in the constructor
        """
        self.n_image_groups = 0
        self.dlg.patternToolBox = QToolBox(parent=self.dlg)
        self.dlg.patternWizardLayout.insertWidget(0, self.dlg.patternToolBox)
        self.patterns_finders = self.get_patterns()
        for pattern_idx, p_finder in enumerate(self.patterns_finders):
            self.add_group()
            for axis_idx, axis_name in enumerate(p_finder.pattern.tag_names()):
                label_widget, pattern_widget, combo_widget = self.get_widgets(pattern_idx, axis_idx)
                pattern_widget.setText(p_finder.pattern.highlight_digits(axis_name))
                self.enable_widgets((label_widget, pattern_widget, combo_widget))
            for ax in range(p_finder.pattern.n_tags(), 4):  # Hide the rest
                self.hide_widgets(self.get_widgets(pattern_idx, ax))

        self._fit_to_content()

    def _fit_to_content(self):
        """
        Measure the plain-text width of all pattern labels
        and resize the dialog so nothing is clipped.
        """
        import re

        max_label_width = 0

        for page in self._get_channel_pages():
            for attr_name in ('pattern0_0', 'pattern0_1', 'pattern0_2',
                              'pattern0_3', 'result'):
                label = getattr(page, attr_name, None)
                if label is None:
                    continue

                text = label.text()
                if not text or text == '...':
                    continue

                # Strip HTML tags → measure plain text with the label's font
                plain = re.sub(r'<[^>]+>', '', text)
                w = label.fontMetrics().horizontalAdvance(plain) + 30

                # Force this label to request that width
                label.setMinimumWidth(w)
                max_label_width = max(max_label_width, w)

        if max_label_width == 0:
            return

        needed_width = max_label_width + 210

        screen = QApplication.primaryScreen()
        if screen:
            needed_width = min(needed_width, int(screen.availableGeometry().width() * 0.9))

        self.dlg.setMinimumWidth(needed_width)
        self.dlg.resize(needed_width, self.dlg.sizeHint().height())

    @staticmethod
    def _measure_rich_label(label):
        """
        Accurately measure the rendered width of a QLabel containing HTML.
        QLabel.sizeHint() is unreliable for rich text, so we use QTextDocument.
        """
        doc = QTextDocument()
        doc.setDefaultFont(label.font())
        doc.setHtml(label.text())
        doc.setDocumentMargin(0)
        return int(doc.idealWidth()) + 10  # small safety margin

    def get_widgets(self, image_group_id, axis):
        """
        Get the widgets (label, pattern and combo) for a given image group and axis

        Parameters
        ----------
        image_group_id: int
            The index of the image group
        axis: int
            The index of the axis

        Returns
        -------
        tuple(QLabel, QLabel, QComboBox)
            The label of the axis, pattern for the axis pattern and combobox containing the axis name (as a letter)
        """
        page = self.dlg.patternToolBox.widget(image_group_id)
        if page is None:
            raise IndexError(f'No widget at index {image_group_id}')
        label_widget = getattr(page, f'label0_{axis}')  # FIXME: why label0_?
        pattern_widget = getattr(page, f'pattern0_{axis}')
        combo_widget = getattr(page, f'pattern0_{axis}ComboBox')

        return label_widget, pattern_widget, combo_widget

    def add_group(self):
        """
        Add a new group of widgets to the dialog
        This is a group of widgets to define a pattern for a set of image files (typically a channel)
        """
        group_controls = create_clearmap_widget('image_group_ctrls.ui', patch_parent_class='QWidget')
        self.dlg.patternToolBox.addItem(group_controls, f'Image group {self.n_image_groups}')

        group_controls.patternButtonBox.button(QDialogButtonBox.Apply).clicked.connect(self.validate_pattern)
        group_controls.channelNameLineEdit.setText(f'channel_{self.n_image_groups}')  # FIXME: check if could read from CFG
        data_types = natsorted(list(dict.fromkeys(DATA_CONTENT_TYPES)))  # avoid duplicates while keeping order
        group_controls.dataTypeComboBox.addItems(data_types)
        group_controls.dataTypeComboBox.setCurrentText('undefined')

        self.n_image_groups += 1

    def connect_buttons(self):
        """
        Connect the buttons to their slots.
        This method is called automatically in the constructor
        """
        self.dlg.mainButtonBox.button(QDialogButtonBox.Apply).clicked.connect(self.save_results)
        self.dlg.mainButtonBox.button(QDialogButtonBox.Cancel).clicked.connect(self.dlg.close)

    def validate_pattern(self):
        """
        Validate the pattern defined by the user and update the result widget
        The result is saved in the pattern_strings attribute for the current channel name
        """
        tool_box = self.dlg.patternToolBox
        pattern_idx = tool_box.currentIndex()
        pattern = self.patterns_finders[pattern_idx].pattern

        axis_names = []
        for i in range(pattern.n_tags()):
            _, _, combo_widget = self.get_widgets(pattern_idx, i)
            axis_names.append(combo_widget.currentText())

        pattern.assign_axes_from_combo(axis_names)

        result_widget = tool_box.widget(pattern_idx).result
        result_widget.setText(pattern.relative_string(self.src_folder))

        self._fit_to_content()

    def get_patterns(self) -> List[PatternFinder]:
        """
        Scan the current source folder to get the pattern finders for the image files

        Non blocking to keep the UI responsive

        Returns
        -------
        list(PatternFinder)
            The pattern finders for the source folder
        """
        progress_bar = dlg_help.make_simple_progress_dialog(title='Scanning source folder')
        with ThreadPool(processes=1) as pool:
            result = pool.apply_async(pattern_finders_from_base_dir,
                                      [self.src_folder, self.min_file_number, self.tile_extension])
            while not result.ready():
                result.wait(0.25)
                dlg_help.update_pbar(self.app, progress_bar.mainProgressBar, 1)  # TODO: real update
                self.app.processEvents()
            pattern_finders = result.get()
        dlg_help.update_pbar(self.app, progress_bar.mainProgressBar, 100)
        return pattern_finders

    def _get_channel_pages(self) -> List[QWidget]:
        return [self.dlg.patternToolBox.widget(i) for i in range(self.dlg.patternToolBox.count())]

    def get_channel_names(self):
        return [page.channelNameLineEdit.text() for page in self._get_channel_pages()]

    def all_channels_defined(self):
        return all([page.dataTypeComboBox.currentText() != 'undefined' for page in self._get_channel_pages()])

    def get_results(self) -> List[ChannelPatternSpec]:
        return self._pattern_results

    def save_results(self):
        """
        Save the file patterns to the `sample` configuration file and close the dialog
        """
        if not self.all_channels_defined():
            dlg_help.warning_popup('Some data types are not defined, '
                                   'please select a valid data type before saving')
            return None

        specs: List[ChannelPatternSpec] = []
        for i, page in enumerate(self._get_channel_pages()):
            channel_name = page.channelNameLineEdit.text()

            if not page.result.text():
                self.dlg.patternToolBox.setCurrentIndex(i)
                self.validate_pattern()

            specs.append(ChannelPatternSpec(
                name=channel_name,
                data_type=page.dataTypeComboBox.currentText(),
                extension=self.tile_extension,
                pattern_relpath=page.result.text())
            )

        self._pattern_results = specs
        self.dlg.accept()
        return specs


class SamplePickerDialog(WizardWidget):
    """
    A dialog to help the user pick the sample folders from a source folder.
    The dialog scans the source folder to find the sample folders based on the presence
    of a `sample_params.cfg` file
    The results are displayed in two lists. The user can move the sample folders from the
    left (available items) list to the right (selected items) list.
    The samples can be split into groups to allow for different processing of the groups.
    """
    def __init__(self, src_folder, params, app=None):
        """
        Initialize the dialog

        Parameters
        ----------
        src_folder: str
            The source folder to scan for sample folders
        params: UiParameter
            The parameters object to use to parametrise the dialog
        app: QApplication (optional)
            The QApplication instance to use. If None, ClearMap will try to use the existing instance
        """
        self.group_paths = None
        self.current_group = 0
        self.list_selection = TwoListSelection()
        super().__init__('sample_picker', 'File paths wizard', src_folder=src_folder,
                         params=params, app=app, size=[None, 600])
        self.list_selection.addAvailableItems(self.parse_sample_folders())
        self.exec()

    def setup(self):
        """
        Setup the dialog after creation from the ui file.
        This method is called automatically in the constructor
        """
        self.group_paths = [[]]
        self.current_group = 1
        for i in range(self.params.n_groups - 1):
            self.__handle_add_group(add_to_params=False)
        self.list_selection = TwoListSelection()
        self.dlg.listPickerLayout.addWidget(self.list_selection)

    def connect_buttons(self):
        """
        Connect the buttons to their slots.
        This method is called automatically in the constructor
        """
        self.dlg.addGroupPushButton.clicked.connect(self.__handle_add_group)
        self.dlg.groupsComboBox.currentIndexChanged.connect(self.__handle_group_changed)
        self.dlg.buttonBox.accepted.connect(self.__apply_changes)
        self.dlg.buttonBox.rejected.connect(self.dlg.close)

        selected_model = self.list_selection.mOutput.model()
        selected_model.rowsInserted.connect(self.__update_current_group_paths)  # Update group when selection updated
        selected_model.rowsRemoved.connect(self.__update_current_group_paths)  # Update group when selection updated

    def parse_sample_folders(self):
        """
        Scan the source folder to find experiment folders based on
        the presence of a sample config file (any supported name/extension).

        Returns
        -------
        list(str)
            The list of sample folders, naturally sorted.
        """
        roots = scan_folder_for_experiments(self.src_folder)
        return natsorted(str(p) for p in roots)

    def __apply_changes(self):
        """Flush all wizard groups to params atomically."""
        groups = {}
        for group_idx, paths in enumerate(self.group_paths):
            name = f"Group_{group_idx + 1}"
            groups[name] = paths
        self.params.groups = groups          # single atomic write via ParamLink → set_value
        self.dlg.close()

    def __handle_group_changed(self):
        """
        Handle the change of the current group (display the corresponding paths in the list selection)
        """
        self.__update_current_group_paths()
        current_gp_id = self.dlg.groupsComboBox.currentIndex()
        self.current_group = max(0, current_gp_id) + 1  # WARNING: update current_group after update
        self.list_selection.setSelectedItems(self.group_paths[self.current_group - 1])

    def __update_current_group_paths(self):
        """
        Update the paths of the current group with the selected items in the list selection
        """
        self.group_paths[self.current_group - 1] = self.list_selection.get_right_elements()

    def __handle_add_group(self, add_to_params=True):
        """
        Add a new group to the dialog

        Parameters
        ----------
        add_to_params: bool
            Whether to add the group to the parameters object. Default is True
        """
        self.dlg.groupsComboBox.addItem(f'{self.dlg.groupsComboBox.count() + 1}')
        # if add_to_params:  # Only on "apply"
        #     self.params.add_group()
        self.group_paths.append([])
        self.dlg.groupsComboBox.setCurrentIndex(self.dlg.groupsComboBox.count() - 1)


class Landmark:
    def __init__(self, idx, dialog, color):
        self.index = idx
        self.dialog = dialog
        self.coords = {
            'fixed_image': (np.nan, np.nan, np.nan),
            'moving_image': (np.nan, np.nan, np.nan)
        }

        btn_name = f'marker{idx}RadioButton'
        btn = getattr(self.dialog, btn_name, None)
        if not btn:
            btn = QRadioButton(f'Marker {idx}:', self.dialog)
            btn.setObjectName(btn_name)

        color_btn_name = f'marker{idx}ColorBtn'
        color_btn = getattr(self.dialog, color_btn_name, None)
        if not color_btn:
            color_btn = QPushButton(self.dialog)
            color_btn.setObjectName(color_btn_name)
            color_btn.setStyleSheet(f'background-color: {color}')

        self.button = btn
        self.color_btn = color_btn
        self.activate()

    def __del__(self):
        for btn in (self.button, self.color_btn):
            btn.setParent(None)
            btn.deleteLater()

    def __repr__(self):
        return f'Landmark({self.coords=}, {self.color=})'

    def formatted_coords(self, img_type):
        x, y, z = self.coords[img_type]
        return f'{z} {y} {x}\n'

    @property
    def color(self):
        return self.color_btn.styleSheet().replace('background-color: ', '').strip()

    def isChecked(self):
        return self.button.isChecked()

    def is_set(self):
        """
        Coords of both fixed and moving images are set

        Returns
        -------
        bool
        """
        return all([all(coords) for coords in self.coords.values()])

    def activate(self):
        self.button.click()


class LandmarksSelectorDialog(WizardWidget):  # TODO: bind qColorDialog to color buttons
    """
    A dialog to select landmarks in 3D space for registration
    The dialog allows to select landmarks in two views (fixed and moving)
    The landmarks are displayed in two 3D viewers with matching colors
    The dialog saves the landmarks to files for the fixed and moving images
    """

    def __init__(self, fixed_image_path, moving_image_path,
                 fixed_image_landmarks_path, moving_image_landmarks_path, app=None):
        """
        Initialize the dialog

        Parameters
        ----------
        fixed_image_path: str | Path
            Path to the fixed image file.
        moving_image_path: str | Path
            Path to the moving image file.
        fixed_image_landmarks_path: str | Path
            Path to save the fixed image landmarks.
        moving_image_landmarks_path: str | Path
            Path to save the moving image landmarks.
        app: QApplication (optional)
            The QApplication instance to use. If None, ClearMap will try to use the existing instance
        """
        self.image_paths = {
            'fixed_image': Path(fixed_image_path),  # WARNING: fixed first so that in sync with data_viewers
            'moving_image': Path(moving_image_path)
        }
        self.landmarks_file_paths = {
            'fixed_image': Path(fixed_image_landmarks_path),
            'moving_image': Path(moving_image_landmarks_path)
        }
        self.data_viewers = {k: None for k in self.image_paths.keys()}
        self.markers = []

        super().__init__('landmark_selector', 'Landmark selector', app=app)
        self.dlg.setModal(False)
        self.dlg.show()

    def setup(self):
        """
        Setup the dialog after creation from the ui file.
        This method is called automatically in the constructor
        """
        self.markers = [Landmark(idx=0, dialog=self.dlg, color=None)]

    def connect_buttons(self):
        """
        Connect the buttons to their slots.
        This method is called automatically in the constructor
        """
        self.dlg.addMarkerPushButton.clicked.connect(self.add_marker)
        self.dlg.delMarkerPushButton.clicked.connect(self.remove_marker)
        self.dlg.buttonBox.accepted.connect(self.write_coords)
        self.dlg.buttonBox.rejected.connect(self.dlg.close)

    def __len__(self):
        return len(self.markers)

    def plot(self, lut=None, parent=None):
        """
        Plot the 3D landmarks onto the fixed and moving images using data viewers.

        Parameters
        ----------
        lut : str
            Lookup table for coloring the 3D plot.
        parent : QWidget
            The parent widget for the plot.
        """
        parent = parent or self.dlg.parent()
        titles = [os.path.basename(img) for img in self.image_paths.values()]
        dvs = plot_3d.plot([str(p) for p in self.image_paths.values()], title=titles, arrange=False, sync=False,
                           lut=lut, parent=parent)
        self.data_viewers['fixed_image'] = dvs[0]
        self.data_viewers['moving_image'] = dvs[1]
        self.__initialize_viewers()

    def __initialize_viewers(self):
        for img_type, dv in self.data_viewers.items():
            scatter = pg.ScatterPlotItem()
            dv.enable_mouse_clicks()
            dv.view.addItem(scatter)
            dv.scatter = scatter
            dv.scatter_coords = Scatter3D(self.get_coords(img_type),
                                          colors=np.array(self.colors),
                                          half_slice_thickness=3)
            dv.mouse_clicked.connect(functools.partial(self.set_current_coords, img_type=img_type))

    @property
    def current_marker(self):
        """
        Get the index of the currently selected  marker
        Returns
        -------
        int : the index of the currently selected marker
        """
        return [marker.isChecked() for marker in self.markers].index(True)

    def write_coords(self):
        """
        Write the coordinates of the markers to the respective landmarks files
        """
        markers = [mrkr for mrkr in self.markers if mrkr.is_set()]
        for img_type, f_path in self.landmarks_file_paths.items():
            f_path.parent.mkdir(parents=True, exist_ok=True)
            with open(f_path, 'w') as landmarks_file:
                landmarks_file.write(f'point\n{len(markers)}\n')  # FIXME: use index ??
                for marker in markers:
                    landmarks_file.write(marker.formatted_coords(img_type))
        self.dlg.close()

    def set_current_coords(self, x, y, z, img_type):
        """
        Set the coordinates for the specified image type.

        Parameters
        ----------
        img_type : str
            The type of the image ('fixed_image' or 'moving_image').
        x : float
            The x-coordinate.
        y : float
            The y-coordinate.
        z : float
            The z-coordinate.
        """
        self.markers[self.current_marker].coords[img_type] = (x, y, z)
        self._update_viewer_coords(img_type)

    def _update_viewer_coords(self, img_type):
        viewer = self.data_viewers[img_type]
        coords = self.get_coords(img_type)
        viewer.scatter_coords.set_data({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'z': coords[:, 2],
            'colour': np.array([QColor(col) for col in self.colors])
        })
        viewer.refresh()

    @property
    def colors(self):
        """
        Get the ordered list of colors of the markers

        Returns
        -------
        list(str)
            The markers colors
        """
        return [marker.color for marker in self.markers]

    @property
    def current_color(self):
        """
        Get the color of the currently selected marker

        Returns
        -------
        str
            The color of the currently selected marker
        """
        return self.markers[self.current_marker].color

    def add_marker(self):
        """
        Add a new marker to the dialog
        """
        marker = Landmark(idx=len(self), dialog=self.dlg, color=self.get_new_color())
        self.dlg.formLayout.insertRow(len(self), marker.button, marker.color_btn)
        self.markers.append(marker)
        marker.activate()

    def remove_marker(self):  # TODO: add option to remove selected marker instead of last
        """
        Remove the last marker
        """
        if self.current_marker == len(self) - 1:  # If last marker, select previous
            self.markers[-2].activate()
        marker = self.markers.pop()
        del marker

    def get_new_color(self):
        """
        Get a new color for a marker (not already used)
        Returns
        -------
        str
            The new color name
        """
        color = QColor('red')
        while color.name() in self.colors:
            color = get_pseudo_random_color('qcolor')
        return color.name()

    def clear_landmarks(self):
        """
        Clear all the markers and the landmarks file paths and reset the dialog
        """
        for marker in self.markers:
            del marker
        self.markers = []
        self.dlg.formLayout.removeRow(0, 1)
        self.add_marker()
        for f_path in self.landmarks_file_paths.values():
            f_path.unlink(missing_ok=True)

    def get_coords(self, img_type):
        """
        Get the coordinates of all the markers for the specified image type.

        Parameters
        ----------
        img_type : str
            The type of the image ('fixed_image' or 'moving_image').

        Returns
        -------
        np.ndarray
            The array of marker coordinates.
        """
        return np.array([m.coords[img_type] for m in self.markers])


class StructurePickerWidget(QTreeWidget):
    LIGHT_COLOR = 'white'
    DARK_COLOR = '#2E3436'

    def __init__(self, parent=None, json_base_name='ABA json 2022'):
        super().__init__(parent)
        self.setColumnCount(4)
        self.root = self.parse_json(json_base_name)
        self.build_tree(self.root, self)
        self.header().resizeSection(0, 300)
        self.setHeaderLabels(['Structure name', 'ID', 'Color', ''])  # TODO: see why 4 columns
        # self.itemClicked.connect(self.print_id)

    def print_id(self, itm, col):
        print([itm.text(i) for i in range(3)])

    @staticmethod
    def parse_json(base_name='ABA json 2022'):
        label_file = Path(Settings.atlas_folder) / STRUCTURE_TREE_NAMES_MAP[base_name]
        with open(label_file, 'r') as json_handle:
            aba = json.load(json_handle)
        root = aba['msg'][0]
        return root

    @staticmethod
    def build_tree(tree=None, parent=None):
        for subtree in tree['children']:
            if isinstance(subtree, dict):
                struct = QTreeWidgetItem(parent)
                struct.setText(0, subtree['name'])
                color_hex = f"#{subtree['color_hex_triplet']}"
                struct.setText(1, str(subtree['id']))
                struct.setText(2, color_hex)
                struct.setText(3, '')
                bg = QColor(color_hex)
                struct.setBackground(2, bg)
                fg = QColor(StructurePickerWidget.LIGHT_COLOR if is_dark(bg) else StructurePickerWidget.DARK_COLOR)
                struct.setForeground(2, fg)
                if 'children' in subtree.keys() and subtree['children']:
                    StructurePickerWidget.build_tree(tree=subtree, parent=struct)
            elif isinstance(subtree, list):
                StructurePickerWidget.build_tree(tree=subtree, parent=parent)
            else:
                raise ValueError(f'Unrecognised type {type(subtree)} for Tree: {subtree}')


class StructureSelector(WizardWidget):
    def __init__(self, app=None):
        super().__init__('structure_selector', 'Structure selector', app=app)
        self.structure_selected = self.picker_widget.itemClicked
        self.onAccepted = self.dlg.buttonBox.accepted.connect
        self.onRejected = self.dlg.buttonBox.rejected.connect

    def show(self):
        self.dlg.show()

    def close(self):
        self.dlg.close()

    def setup(self):
        self.picker_widget = StructurePickerWidget(self.dlg)
        self.dlg.structureLayout.addWidget(self.picker_widget)

    def connect_buttons(self):
        pass


class PerfMonitor(QWidget):
    cpu_vals_changed = QtCore.pyqtSignal(int, int, int)
    gpu_vals_changed = QtCore.pyqtSignal(int, int)

    def __init__(self, parent, fast_period, slow_period, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        if fast_period < 100 or slow_period and slow_period < 100:
            raise ValueError('Periods cannot be below 100ms')
        self.percent_cpu = 0
        self.percent_thread = 0
        """The percentage of the CPU used by the most active process of ClearMap"""
        self.percent_ram = 0
        self.percent_v_ram = 0
        self.percent_gpu = 0
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_timer = QTimer()
        self.fast_timer.setInterval(self.fast_period)
        self.fast_timer.timeout.connect(self.update_cpu_values)
        self.slow_timer = QTimer()
        if slow_period is not None:
            self.slow_timer.setInterval(self.slow_period)
            self.slow_timer.timeout.connect(self.update_gpu_values)

        self.gpu_proc_file_path = tempfile.mkstemp(suffix='_clearmap_gpu.proc')[-1]
        self.cpu_proc_file_path = tempfile.mkstemp(suffix='_clearmap_cpu.proc')[-1]
        self.file_watcher = QtCore.QFileSystemWatcher([self.gpu_proc_file_path, self.cpu_proc_file_path])
        self.file_watcher.fileChanged.connect(self.handle_proc_changed)
        self.pool = ProcessPoolExecutor(max_workers=1)

    def start(self):
        self.fast_timer.start()
        if self.slow_period is not None:
            self.slow_timer.start()

    def stop(self):
        self.fast_timer.stop()
        if self.slow_period is not None:
            self.slow_timer.stop()

    def get_cpu_percent(self):
        return round(psutil.cpu_percent())

    def get_thread_percent(self):
        try:
            clear_map_proc_cpu = [proc.cpu_percent() for proc in psutil.process_iter()
                                  if proc and 'python' in proc.name().lower() and
                                  USER_NAME in proc.username() and
                                  'clearmap' in proc.exe().lower()]
        except psutil.NoSuchProcess:
            clear_map_proc_cpu = []
        # The name filter is not sufficient but necessary because the exe is not always allowed
        return max(clear_map_proc_cpu) if clear_map_proc_cpu else 0

    def get_ram_percent(self):
        return round(psutil.virtual_memory().percent)

    def _get_cpu_vals(self):
        with ThreadPoolExecutor(max_workers=1) as pool:  # TODO: check if should use self.pool instead
            futures = [pool.submit(f) for f in (self.get_cpu_percent, self.get_thread_percent, self.get_ram_percent)]
            percents = [f.result() for f in futures]
        return percents

    def update_cpu_values(self):
        percent_cpu, percent_thread, percent_ram = self._get_cpu_vals()
        if percent_ram != self.percent_ram or percent_cpu != self.percent_cpu or percent_thread != self.percent_thread:
            self.percent_cpu = percent_cpu
            self.percent_thread = percent_thread
            self.percent_ram = percent_ram
            self.cpu_vals_changed.emit(self.percent_cpu, self.percent_thread, self.percent_ram)

    def update_gpu_values(self):
        self.pool.submit(gpu_params, self.gpu_proc_file_path)  # We don't care if exception are raised here

    def handle_proc_changed(self, file_path):
        if file_path == self.gpu_proc_file_path:
            self.handle_gpu_vals_updated()
        elif file_path == self.cpu_proc_file_path:
            self.handle_cpu_vals_updated()

    def handle_gpu_vals_updated(self):
        try:
            with open(self.gpu_proc_file_path, 'r') as proc_file:
                line = proc_file.read()
                if not line:
                    return
                elems = line.split(',')
                if len(elems) < 3:
                    return
                mem_used, mem_total, gpu_percent = [s.strip() for s in elems]
                percent_v_ram = int((float(mem_used) / float(mem_total)) * 100)
                percent_gpu = int(gpu_percent)
            if percent_gpu != self.percent_gpu or percent_v_ram != self.percent_v_ram:
                self.percent_gpu = percent_gpu
                self.percent_v_ram = percent_v_ram
                self.gpu_vals_changed.emit(self.percent_gpu, self.percent_v_ram)
        except ValueError as err:
            print(err)
            pass


class ExtendableTabWidget(QTabWidget):
    addTabClicked = pyqtSignal()
    channelChanged = pyqtSignal(str)
    channelRenamed = pyqtSignal(str, str)

    def __init__(self, parent=None, with_add_tab=True):
        super().__init__(parent)
        self.has_add_tab = with_add_tab
        if with_add_tab:
            plus_icon = QIcon(str(Path(Settings.clearmap_path) / 'gui/creator/icons/add.svg'))
            self.addTab(QWidget(), plus_icon,"")
        self.tabBarClicked.connect(self.handle_tab_bar_click)

    def handle_tab_bar_click(self, index):
        if self.has_add_tab and index == self.count() - 1:
            self.addTabClicked.emit()
        else:
            self.channelChanged.emit(self.tabText(index))

    def current_channel(self):
        return self.tabText(self.currentIndex())

    def set_current_channel_name(self, name):
        current_name = self.tabText(self.currentIndex())
        self.setTabText(self.currentIndex(), name)
        self.channelRenamed.emit(current_name, name)

    @property
    def last_real_tab_idx(self):
        return self.count() -(int(self.has_add_tab))

    def get_channels_names(self):
        return [self.tabText(i) for i in range(self.last_real_tab_idx)]

    def add_channel_widget(self, widget, name=''):
        if isinstance(name, (tuple, list)):  # For compound channels, concatenate names
            name = '-'.join(name)
        tab_name = name if name else f'Channel_{self.count() - 1}'
        self.insertTab(self.last_real_tab_idx, widget, tab_name)
        self.setCurrentWidget(widget)
        return tab_name

    def remove_channel_widget(self, name):
        widget, idx = self.get_channel_widget(name, return_idx=True)
        if widget:
            self.removeTab(idx)
            widget.deleteLater()

    def get_channel_widget(self, name=None, return_idx=False):
        if name is None:
            name = self.current_channel()
        for i in range(self.last_real_tab_idx):
            if self.tabText(i) == name:
                if return_idx:
                    return self.widget(i), i
                return self.widget(i)
        if return_idx:
            return None, -1
        return None


class FileDropListWidget(QListWidget):  # TODO: check if I need dragMoveEvent
    itemsChanged = pyqtSignal()
    def __init__(self, parent=None, plus_btn=None, minus_btn=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.plus_btn = plus_btn
        self.minus_btn = minus_btn

        if self.plus_btn:
            self.plus_btn.clicked.connect(self.add_files)
        if self.minus_btn:
            self.minus_btn.clicked.connect(self.remove_selected)

    def get_items_text(self):
        return [self.item(i).text() for i in range(self.count())]

    def addItem(self, *__args):
        super().addItem(*__args)
        self.itemsChanged.emit()

    def addItems(self, *__args):
        super().addItems(*__args)
        self.itemsChanged.emit()

    def add_files(self, file_paths=None):
        file_paths = file_paths or QFileDialog.getOpenFileNames(self, 'Select files')[0]
        if file_paths:
            self.addItems(file_paths)
            self.itemsChanged.emit()

    def remove_selected(self):
        changed = False
        for item in self.selectedItems():
            self.takeItem(self.row(item))
            changed = True
        if changed:
            self.itemsChanged.emit()

    def dragEnterEvent(self, event):
        data = event.mimeData()
        if data.hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        data = event.mimeData()
        if data.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        data = event.mimeData()
        if data.hasUrls():
            for url in data.urls():
                file_path = url.toLocalFile()
                self.addItem(file_path)
            event.acceptProposedAction()


class LandmarksWeightsPanel(QFrame):
    """
    Compact panel that renders one row per landmark-params file:
      [Label]  0 [Slider 0..100] 100%  (value label)

    Public API:
      - set_items(names: list[str], weights: list[int] | None = None)
      - get_weights() -> list[int]            # 0..100 as integers
      - set_weights(weights: list[int])
      - valueChangedConnect(cb: Callable[[], None])  # Qt-like hook for external binding

    Optional transforms can be supplied to map between slider value (0..100)
    and model value (float). Defaults are identity; keep scaling (e.g. exp) in controller.
    """
    weightsChanged = pyqtSignal(list)           # emits the full 0..100 list (ints)
    weightAtChanged = pyqtSignal(int, int)      # emits (idx, 0..100)

    def __init__(self, parent: QWidget = None,
                 value_to_model=None, model_to_value=None):
        super().__init__(parent)
        self.setObjectName("landmarksWeightsPanel")
        self._names: list[str] = []
        self._sliders: list[QSlider] = []
        self._value_labels: list[QLabel] = []  # shows current value (indexed by param file idx) (disabled if 0)
        self._value_to_model = value_to_model or (lambda v: v)
        self._model_to_value = model_to_value or (lambda v: int(round(v)))
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(6)
        self.setLayout(self._grid)

    def getValue(self):
        return self.get_weights()

    def setValue(self, weights):
        self.set_weights(weights)

    def set_items(self, names: list[str], weights: list[int] | None = None) -> None:
        """Rebuild rows for the given names; optionally seed slider positions."""
        self._clear_rows()
        self._names = list(names)
        weights = list(weights) if weights is not None else [0] * len(names)
        if len(weights) != len(names):
            weights = trim_or_pad(weights, len(names), pad_value=0)

        for row, (nm, w) in enumerate(zip(self._names, weights)):
            name_lbl = QLabel(nm, self)
            min_lbl  = QLabel("<b>0</b>", self)
            max_lbl  = QLabel("<b>100%</b>", self)
            slider   = QSlider(Qt.Horizontal, self)
            slider.setMinimum(0); slider.setMaximum(100); slider.setValue(int(w))
            val_lbl  = QLabel(f"({int(w) if w else 'disabled'})", self)

            self._sliders.append(slider)
            self._value_labels.append(val_lbl)

            self._grid.addWidget(name_lbl, row, 0)
            self._grid.addWidget(min_lbl,  row, 1)
            self._grid.addWidget(slider,   row, 2)
            self._grid.addWidget(max_lbl,  row, 3)
            self._grid.addWidget(val_lbl,  row, 4)

            slider.valueChanged.connect(self._make_row_handler(row))

        # single emit after rebuild to advertise current state
        self.weightsChanged.emit(self.get_weights())

    def get_weights(self) -> list[int]:
        """Return current raw slider values (0..100)."""
        return [int(s.value()) for s in self._sliders]

    def get_params_and_weights(self) -> dict[str, int]:
        """Return current mapping of param file name to raw slider value (0..100)."""
        return {param_name: int(s.value()) for param_name, s in zip(self._names, self._sliders)}

    def set_weights(self, weights: list[int]) -> None:
        """Set weights without rebuilding rows (length must match)."""
        n = min(len(self._sliders), len(weights))
        for i in range(n):
            self._sliders[i].blockSignals(True)
            self._sliders[i].setValue(int(weights[i]))
            self._sliders[i].blockSignals(False)
            self._update_value_label(i, int(weights[i]))
        self.weightsChanged.emit(self.get_weights())

    def valueChangedConnect(self, cb):  # Function name to match that of monkeypatched widgets
        """Qt-like connector used by our generic binder."""
        self.weightsChanged.connect(lambda *_: cb())

    def _clear_rows(self):
        """remove widgets from layout & delete"""
        while self._grid.count():  # TODO: see if we could use existing helpers
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._sliders.clear()
        self._value_labels.clear()

    def _make_row_handler(self, idx: int):
        def _on_value_changed(v: int):
            self._update_value_label(idx, v)
            self.weightAtChanged.emit(idx, int(v))
            self.weightsChanged.emit(self.get_weights())
        return _on_value_changed

    def _update_value_label(self, idx: int, v: int):
        self._value_labels[idx].setText(f"({v if v else 'disabled'})")


# Helper for the cell counter histograms. TODO: make more generic
def ensure_inline_histogram(histogram: PlotWidget | QWidget, hist_idx: int, layout: QLayout):
    widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
    n_plotted_histograms = len([w for w in widgets if isinstance(w, PlotWidget)])
    if n_plotted_histograms < 2:  # Histograms not yet added
        label, _ = get_widget(layout, widget_type=QLabel, index=hist_idx)
        controls, _ = get_widget(layout, key='Doublet', index=hist_idx)

        graph_width = label.width() + controls.width()
        graph_height = 50
        histogram.resize(graph_width, graph_height)
        histogram.setMaximumSize(graph_width, graph_height)

        row = 2 * n_plotted_histograms

        layout.addWidget(histogram, row,    0, 1, 3)
        layout.addWidget(label,     row+1,  0, 1, 1)
        layout.addWidget(controls,  row+1,  1, 1, 2)

        container = layout.parent().parent().parent().parent()
        container.setMinimumHeight(container.parent().height() - container.height() + layout.parent().height())

    return histogram

class GraphFilterList(QWidget):
    """A vertical list of graph-filter rows with AND/OR combiners between them.

    Compatibility guarantees for GraphFilterParams:
      - The container layout is a QVBoxLayout named 'filterParamsVerticalLayout'
      - Each filter row widget is named 'filter_{idx}'
      - Between row i and i+1 we insert a QFrame that contains two QRadioButtons named:
            'filter_{i}_and_btn'  (checked by default)
            'filter_{i}_or_btn'
    """
    filtersChanged = pyqtSignal()

    def __init__(self, layout: QVBoxLayout, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = layout
        self._row_count = 0

    def add_filter_row(self, *, title: str | None = None) -> QWidget:
        """
        Add a new filter row.
        Returns the filter QWidget to pass to GraphFilterParams.
        """
        if self._row_count > 0:  # Insert combine ctrls (AND/OR) **after** each filter -> needs 1
            comb_idx = self._row_count - 1
            combine_widget = self._create_combine_widget(comb_idx)
            self._layout.addWidget(combine_widget)

        # The actual filter row UI from your .ui file
        filter_widget = create_clearmap_widget('graph_filter_params', 'QWidget')
        filter_widget.setObjectName(f'filter_{self._row_count}')
        title = title or f'Filter {self._row_count + 1}'
        filter_widget.groupBox.setTitle(title)

        self._layout.addWidget(filter_widget)
        self._row_count += 1
        self.filtersChanged.emit()
        return filter_widget

    def _create_combine_widget(self, comb_idx: int) -> QFrame:
        combine_widget = QFrame(self)
        combine_widget.setLayout(QHBoxLayout())

        and_btn = QRadioButton('AND', combine_widget)
        and_btn.setObjectName(f'filter_{comb_idx}_and_btn')  # WARNING: required by GraphFilterParams
        and_btn.setChecked(True)

        or_btn = QRadioButton('OR', combine_widget)
        or_btn.setObjectName(f'filter_{comb_idx}_or_btn')  # WARNING: required by GraphFilterParams

        group = QButtonGroup(combine_widget)  # exclusive group
        group.addButton(and_btn)
        group.addButton(or_btn)

        combine_widget.layout().addWidget(and_btn)
        combine_widget.layout().addWidget(or_btn)
        return combine_widget

    def row_count(self) -> int:
        return self._row_count


# REFACTOR: check if this belongs to gui_utils_base.py
@dataclass
class ComparisonsModel:
    group_names: List[str] = field(default_factory=list)
    selected: List[Pair] = field(default_factory=list)
    sep: str = " vs "

    def all_pairs(self) -> List[Pair]:
        return list(permutations(self.group_names, 2))

    def serialize_label(self, pair: Pair) -> str:
        return self.sep.join(pair)

    def parse_label(self, label: str) -> Pair:
        a, b = label.split(self.sep)
        return a, b


class ComparisonsWidgetAdapter:
    """
    Owns and (re)builds the comparisons area. Exposes only high-level ops
    so GroupAnalysisParams doesn't touch low-level widgetry.
    """
    def __init__(self, layout, groups_sep: str = " vs "):
        self._layout = layout
        self._sep = groups_sep
        self._checkboxes: List[QCheckBox] = []
        self._plot_btns: List[QPushButton] = []
        self._channel_combo: Optional[QComboBox] = None

    def rebuild(self, model: ComparisonsModel, *, on_plot_group: Callable[[str], None],
                channels: List[str], on_channel_changed: Callable[[str], None],
                preselected_comparisons: Optional[List[Pair]] = None) -> None:
        clear_layout(self._layout)
        wanted = set(preselected_comparisons or [])

        # checkboxes
        self._checkboxes = []
        for i, pair in enumerate(model.all_pairs()):
            chk = QCheckBox(model.serialize_label(pair))
            chk.setChecked(pair in wanted)
            self._layout.addWidget(chk)
            self._checkboxes.append(chk)

        self._layout.addStretch()

        # plot buttons per group
        self._plot_btns = []
        for gp in model.group_names:
            btn = QPushButton(f"Plot {gp} group density maps")
            btn.clicked.connect(lambda _=False, g=gp: on_plot_group(g))  # WARNING: new
            self._layout.addWidget(btn)
            self._plot_btns.append(btn)

        self._layout.addStretch()

        # Channel to plot combobox
        if channels:
            plot_channel_combobox = QComboBox()
            plot_channel_combobox.addItems(channels)
            plot_channel_combobox.currentTextChanged.connect(on_channel_changed)
            self._layout.addWidget(plot_channel_combobox)
            self._channel_combo = plot_channel_combobox

        self._layout.addStretch()

    def selected_pairs(self, model: ComparisonsModel) -> List[Pair]:
        out: List[Pair] = []
        for chk in self._checkboxes:
            if chk.isChecked():
                out.append(model.parse_label(chk.text()))
        return out


class GroupPage:
    """Wrapper around one sample_group_controls.ui page."""
    def __init__(self, *, ui_name: str = 'sample_group_controls.ui', start_folder_getter=lambda: ""):
        self._widget: QWidget = create_clearmap_widget(ui_name, patch_parent_class='QWidget')
        self._start = start_folder_getter
        self._config_connected = False  # avoid dupes when wiring up new pages in GroupsWidgetAdapter

    @property
    def widget(self) -> QWidget:
        return self._widget

    @property
    def name(self) -> str:
        return self._widget.gpNameLineEdit.text().strip()

    @name.setter
    def name(self, v: str) -> None:
        self._widget.gpNameLineEdit.setText(v)

    @property
    def paths(self) -> List[str]:
        lst_widget = self._widget.gpListWidget
        return [lst_widget.item(i).text() for i in range(lst_widget.count())]

    @paths.setter
    def paths(self, items: List[str]) -> None:
        lst_widget = self._widget.gpListWidget
        lst_widget.clear()
        lst_widget.addItems(items or [])

    def connect(self, on_changed: Callable[[], None]) -> None:
        """Wire config-writing callbacks. Idempotent."""
        if self._config_connected:
            return
        # self.connect_group_name_changed(on_changed)
        self._widget.gpNameLineEdit.editingFinished.connect(on_changed)
        self._widget.gpAddSrcFolderBtn.clicked.connect(lambda:
            self._add_folder(on_changed))
        self._widget.gpRemoveSrcFolderBtn.clicked.connect(lambda:
            self._remove_selected(on_changed))
        self._config_connected = True

    # --- actions ---
    def _add_folder(self, on_changed: Callable[[], None]) -> None:
        base = self._start()
        folder = dlg_help.get_directory_dlg(base, 'Select sample folder')
        if folder:
            self._widget.gpListWidget.addItem(str(folder))
            on_changed()

    def _remove_selected(self, on_changed: Callable[[], None]) -> None:
        lst_widget = self._widget.gpListWidget
        row = lst_widget.currentRow()
        if row >= 0:
            lst_widget.takeItem(row)
            on_changed()

    def connect_group_name_changed(self, on_name_changed: Callable[[str], None]) -> None:
        try:
            self._widget.gpNameLineEdit.textChanged.disconnect(on_name_changed)  # avoid dupes
        except TypeError:
            pass
        self._widget.gpNameLineEdit.textChanged.connect(on_name_changed)


class GroupsWidgetAdapter(QWidget):
    """
    ParamLink-compatible adapter:
      - owns a list[GroupPage] kept **in the same order** as the toolbox pages.
      - add/remove uses the provided buttons and QToolBox currentIndex.
      - set_value/get_value talk only dict[str, list[str]].
    """
    def __init__(self, *, toolbox: QToolBox, container_layout: QLayout,
                 add_btn: QPushButton, remove_btn: QPushButton,
                 start_folder_getter=lambda: "", groups_ui_file='sample_group_controls.ui'):
        super().__init__(toolbox.parent())
        self._toolbox = toolbox
        self._layout = container_layout
        self._add_btn = add_btn
        self._rm_btn = remove_btn
        self._start = start_folder_getter
        self._ui_name = groups_ui_file
        self._pages: List[GroupPage] = []
        self._on_changed: Optional[Callable[[], None]] = None

    # ---- ParamLink surface -------------------------------------------------
    def set_value(self, groups: Dict[str, List[str]]) -> None:
        self._clear()
        for name, paths in groups.items():
            idx = self._append_page(name, paths)
            # keep tab label in sync with gpNameLineEdit
            # TODO: check if redundant
            # self._pages[idx].connect_group_name_changed(
            #     lambda txt, i=idx: self._toolbox.setItemText(i, self.__gp_name(txt, i))
            # )

    def get_value(self) -> Dict[str, List[str]]:
        return {page.name: page.paths for page in self._pages}

    def connect(self, on_changed: Callable[[], None]) -> None:
        self._on_changed = on_changed
        # add/remove group
        self._add_btn.clicked.connect(lambda: (self._append_page(), on_changed()))
        self._rm_btn.clicked.connect(lambda: (self._remove_current_page(), on_changed()))
        # wire pages
        for i, p in enumerate(self._pages):
            p.connect(on_changed)
            self._sync_label(i)

   # ---- public API for wizards / params -----------------------------------
    def add_group(self, name: Optional[str] = None, paths: Optional[List[str]] = None) -> int:
        """Add a new (possibly empty) group and return its index."""
        idx = self._append_page(name=name, paths=paths)
        if self._on_changed:
            self._on_changed()          # propagate change to config
        return idx

    # BatchParameters API
    def group_count(self) -> int:
        return len(self._pages)

    def set_paths(self, idx: int, paths: list[str]) -> None:
        self._pages[idx].paths = paths

    def get_paths(self, idx: int) -> list[str]:
        return self._pages[idx].paths

    def get_all_paths(self) -> list[str]:
        out = []
        for p in self._pages:
            out.extend(p.paths)
        return out

    @property
    def group_names(self) -> list[str]:
        return [self.__gp_name(p.name, i) for i, p in enumerate(self._pages)]

    @group_names.setter
    def group_names(self, names: list[str]) -> None:
        # resize pages to match new names length
        lens = (len(names), len(self._pages))
        n_pages = max(*lens) - min(*lens)
        if len(names) < len(self._pages):
            for _ in range(n_pages):
                self._remove_page_at(len(self._pages) - 1)
        elif len(names) > len(self._pages):
            for _ in range(n_pages):
                self._append_page()

        # apply names and refresh toolbox labels
        self._sync_labels(names)

    def _sync_labels(self, names: list[str]):
        for i, (page_name, page) in enumerate(zip(names, self._pages)):
            page.name = page_name
            self._toolbox.setItemText(i, self.__gp_name(page_name, i))

    def _sync_label(self, idx: int) -> None:
        page_name = self.__gp_name(self._pages[idx].name, idx)
        self._toolbox.setItemText(idx, page_name)

    # Private utils
    def _clear(self) -> None:
        while self._pages:
            self._remove_page_at(len(self._pages) - 1)

    def _append_page(self, name: Optional[str] = None, paths: Optional[List[str]] = None,
                     on_changed: Optional[Callable[[], None]] = None) -> int:
        on_changed = on_changed or self._on_changed
        page = GroupPage(ui_name=self._ui_name, start_folder_getter=self._start)
        if name:
            page.name = name
        if paths:
            page.paths = paths
        idx = self._toolbox.count()
        self._toolbox.addItem(page.widget, self.__gp_name(name, idx))
        self._pages.append(page)
        # self._layout.addWidget(self._toolbox)
        page.connect_group_name_changed(
            lambda txt, w=page.widget:
            self._toolbox.setItemText(self._toolbox.indexOf(w), self.__gp_name(txt, self._toolbox.indexOf(w)))
        )
        if on_changed:
            page.connect(on_changed)
        return idx

    def remove_current_page(self) -> Tuple[int, str]:
        idx = self._toolbox.currentIndex()
        name = self.__gp_name(self._pages[idx].name, idx)
        _ = self._remove_current_page()
        return idx, name

    def _remove_current_page(self) -> int:
        idx = self._toolbox.currentIndex()
        if idx < 0:
            return -1
        delete_widget(tool_box=self._toolbox, toolbox_page_index=idx)
        self._pages.pop(idx)
        # re-sync label change hooks if you rely on index in the lambda above
        for i, p in enumerate(self._pages):
            p.connect_group_name_changed(lambda txt, idx_=i: self._toolbox.setItemText(idx_, self.__gp_name(txt, idx_)))
        return idx

    def _remove_page_at(self, idx: int) -> None:
        if 0 <= idx < len(self._pages):
            delete_widget(tool_box=self._toolbox, toolbox_page_index=idx)
            self._pages.pop(idx)

    def __gp_name(self, name, idx):
        return name or f"Group {idx + 1}"


class NProcessesWidget(QWidget):
    """
    Simple 'n_processes' widget: label + spinbox.

    Exposes value()/setValue() and valueChanged signal so it can be used
    transparently by ParamLink.
    """
    valueChanged = pyqtSignal(int)

    def __init__(self, parent=None, label: str = "n_processes"):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label, self)
        self._spin = QSpinBox(self)
        self._spin.setMinimum(1)
        # Slightly conservative default; user can override in Designer or from code
        self._spin.setMaximum(max(1, cpu_count()))

        layout.addWidget(self._label)
        layout.addWidget(self._spin)

        self._spin.valueChanged[int].connect(self._on_spin_changed)

    # API compatible with QSpinBox / ParamLink
    def value(self) -> int:
        return self._spin.value()

    def setValue(self, v: int):
        self._spin.setValue(int(v))

    def _on_spin_changed(self, v: int):
        # print("NProcessesWidget changed:", v)
        self.valueChanged.emit(v)

    def setRange(self, minimum: int, maximum: int):
        self._spin.setRange(minimum, maximum)

    def setMinimum(self, minimum: int):
        self._spin.setMinimum(minimum)

    def setMaximum(self, maximum: int):
        self._spin.setMaximum(maximum)


class BlockProcessingWidget(QGroupBox):
    """
    Reusable UI for block_processing parameters:
      - size_min
      - size_max
      - overlap
      - n_processes

    The internal controls are spinboxes. We expose properties with
    value()/setValue() style methods to play nice with ParamLink.
    """

    def __init__(self, parent=None, title: str = 'Block processing',
                 with_overlap: bool = True, default_min: int = 1_000_000, default_max: int = 10_000_000):
        super().__init__(title, parent)

        self._layout = QVBoxLayout(self)
        # n_processes
        self._nproc_widget = NProcessesWidget(self, label='n_processes')
        self._layout.addWidget(self._nproc_widget)

        grid = QGridLayout()
        self._layout.addLayout(grid)

        # size_min
        self._size_min_spin = QSpinBox(self)
        self._size_min_spin.setRange(1, 2**31 - 1)
        self._size_min_spin.setValue(default_min)
        grid.addWidget(QLabel('size_min'), 0, 0)
        grid.addWidget(self._size_min_spin, 0, 1)

        # size_max
        self._size_max_spin = QSpinBox(self)
        self._size_max_spin.setRange(1, 2**31 - 1)
        self._size_max_spin.setValue(default_max)
        grid.addWidget(QLabel('size_max'), 1, 0)
        grid.addWidget(self._size_max_spin, 1, 1)

        # overlap (optional)
        self._overlap_spin = None
        if with_overlap:
            self._overlap_spin = QSpinBox(self)
            self._overlap_spin.setRange(0, 2**31 - 1)
            self._overlap_spin.setValue(0)
            grid.addWidget(QLabel('overlap'), 2, 0)
            grid.addWidget(self._overlap_spin, 2, 1)

    @property
    def n_processes(self) -> int:
        return self._nproc_widget.value()

    @n_processes.setter
    def n_processes(self, v: int):
        self._nproc_widget.setValue(int(v))

    @property
    def size_min(self) -> int:
        return self._size_min_spin.value()

    @size_min.setter
    def size_min(self, v: int):
        self._size_min_spin.setValue(int(v))

    @property
    def size_max(self) -> int:
        return self._size_max_spin.value()

    @size_max.setter
    def size_max(self, v: int):
        self._size_max_spin.setValue(int(v))

    @property
    def overlap(self) -> int | None:
        if self._overlap_spin is None:
            return None
        return self._overlap_spin.value()

    @overlap.setter
    def overlap(self, v: int | None):
        if self._overlap_spin is not None and v is not None:
            self._overlap_spin.setValue(int(v))


class ClickableFrame(QFrame):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)
