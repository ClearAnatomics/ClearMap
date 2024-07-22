# -*- coding: utf-8 -*-
"""
widgets
=======

A set of custom widgets for the ClearMap GUI
"""
import functools
import json
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.pool import ThreadPool


import numpy as np
import psutil
import pyqtgraph as pg
from qdarkstyle import DarkPalette

from skimage import transform as sk_transform  # WARNING: Slowish import, should be replaced

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QDialogButtonBox, QListWidget, QHBoxLayout, QPushButton, QVBoxLayout, QTableWidget, \
    QTableWidgetItem, QToolBox, QRadioButton, QTreeWidget, QTreeWidgetItem

from ClearMap.Alignment.Annotation import annotation
from ClearMap.IO import TIF
from ClearMap.IO.metadata import pattern_finders_from_base_dir
from ClearMap.Settings import atlas_folder
from ClearMap.Utils.utilities import gpu_params
from ClearMap.Visualization import Plot3d as plot_3d
from ClearMap.config.config_loader import ConfigLoader
from ClearMap.gui.dialogs import make_splash, update_pbar
from ClearMap.gui.gui_utils import create_clearmap_widget, get_pseudo_random_color, is_dark

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


def setup_mini_brain(mini_brain_scaling=(5, 5, 5)):  # TODO: scaling in prefs
    """
    Create a downsampled version of the Allen Brain Atlas for the mini brain widget

    Parameters
    ----------
    mini_brain_scaling : tuple(int, int, int)
        The scaling factors for the mini brain. Default is (5, 5, 5)

    Returns
    -------
    tuple(scale, downsampled_array)
    """
    atlas_path = os.path.join(atlas_folder, 'ABA_25um_annotation.tif')
    arr = TIF.Source(atlas_path).array
    return mini_brain_scaling, sk_transform.downscale_local_mean(arr, mini_brain_scaling)


class OrthoViewer(object):
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
        self.params = None
        self.linear_regions = []
        self.dvs = []

    def setup(self, img, params, parent=None):
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

        Returns
        -------

        """
        self.img = img
        self.params = params
        self.parent = parent
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
        if self.params is not None:
            rng = [self.params.scale_axis(val, 'xyz'[axis]) for val in rng]
            setattr(self.params, f'crop_{"xyz"[axis]}_min', rng[0])
            setattr(self.params, f'crop_{"xyz"[axis]}_max', rng[1])

    def add_regions(self):  # FIXME: improve documenation
        """
        Add the regions to the viewer
        """
        # y_axis_idx = (1, 2, 0)
        for i, dv in enumerate(self.dvs):
            transparency = '4B'
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

    def __init__(self, max_progress=100, main_max_progress=1, parent=None):
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

    def __del__(self):
        self.set_main_progress(self.main_max_progress)
        self.set_progress(self.max_progress)
        if self.parentWidget() is not None:
            self.parentWidget().app.processEvents()

    def reset(self):
        """
        Reset all the values to their initial state

        Returns
        -------

        """
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

        Returns
        -------

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
    """
    A widget that allows to select items from a list and move them to another list
    This is useful for selecting items from a list of available items and moving them to a list of selected items
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__setup_layout()
        # self.app = app

    def __setup_layout(self):
        """
        Setup the layout of the widget with the two columns for the lists and the buttons
        """
        lay = QHBoxLayout(self)
        self.mInput = QListWidget()
        self.mOuput = QListWidget()

        move_btns, up_down_btns = self.__layout_buttons()

        lay.addWidget(self.mInput)
        lay.addLayout(move_btns)
        lay.addWidget(self.mOuput)
        lay.addLayout(up_down_btns)

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

    @QtCore.pyqtSlot()
    def update_buttons_status(self):
        self.mBtnUp.setDisabled(not bool(self.mOuput.selectedItems()) or self.mOuput.currentRow() == 0)
        self.mBtnDown.setDisabled(not bool(self.mOuput.selectedItems()) or self.mOuput.currentRow() == (self.mOuput.count() -1))
        self.mBtnMoveToAvailable.setDisabled(not bool(self.mInput.selectedItems()) or self.mOuput.currentRow() == 0)
        self.mBtnMoveToSelected.setDisabled(not bool(self.mOuput.selectedItems()))

    def __connections(self):
        """
        Bind the buttons to their slots
        """
        self.mInput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mOuput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mBtnMoveToAvailable.clicked.connect(self.__on_mBtnMoveToAvailable_clicked)
        self.mBtnMoveToSelected.clicked.connect(self.__on_mBtnMoveToSelected_clicked)
        self.mButtonToAvailable.clicked.connect(self.__on_mButtonToAvailable_clicked)
        self.mButtonToSelected.clicked.connect(self.__on_mButtonToSelected_clicked)
        self.mBtnUp.clicked.connect(self.__on_mBtnUp_clicked)
        self.mBtnDown.clicked.connect(self.__on_mBtnDown_clicked)

    @QtCore.pyqtSlot()
    def __on_mBtnMoveToAvailable_clicked(self):
        self.mOuput.addItem(self.mInput.takeItem(self.mInput.currentRow()))

    @QtCore.pyqtSlot()
    def __on_mBtnMoveToSelected_clicked(self):
        self.mInput.addItem(self.mOuput.takeItem(self.mOuput.currentRow()))

    @QtCore.pyqtSlot()
    def __on_mButtonToAvailable_clicked(self):
        while self.mOuput.count() > 0:
            self.mInput.addItem(self.mOuput.takeItem(0))

    @QtCore.pyqtSlot()
    def __on_mButtonToSelected_clicked(self):
        while self.mInput.count() > 0:
            self.mOuput.addItem(self.mInput.takeItem(0))

    @QtCore.pyqtSlot()
    def __on_mBtnUp_clicked(self):
        row = self.mOuput.currentRow()
        currentItem = self.mOuput.takeItem(row)
        self.mOuput.insertItem(row - 1, currentItem)
        self.mOuput.setCurrentRow(row - 1)

    @QtCore.pyqtSlot()
    def __on_mBtnDown_clicked(self):
        row = self.mOuput.currentRow()
        currentItem = self.mOuput.takeItem(row)
        self.mOuput.insertItem(row + 1, currentItem)
        self.mOuput.setCurrentRow(row + 1)

    # The actual user functions
    def addAvailableItems(self, items):
        """
        Add the list of available items to the left list

        Parameters
        ----------
        items: list(str)
            The list of items to add
        """
        self.mInput.addItems(items)

    def setSelectedItems(self, items):
        """
        Add the list of selected items to the right list
        Parameters
        ----------
        items: list(str)
            The list of items to add
        """
        self.mOuput.clear()
        self.mOuput.addItems(items)

    def get_left_elements(self):
        """
        Get the list of items in the left list (available items)

        Returns
        -------
        list(str)
        """
        r = []
        for i in range(self.mInput.count()):
            it = self.mInput.item(i)
            r.append(it.text())
        return r

    def get_right_elements(self):
        """
        Get the list of items in the right list (selected items)

        Returns
        -------
        list(str)
        """
        r = []
        for i in range(self.mOuput.count()):
            it = self.mOuput.item(i)
            r.append(it.text())
        return r


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


class WizardDialog:
    """
    A base class for a complex dialogs designed with QT creator and exported as ui files.
    It is meant to be subclassed. The subclass should implement the `setup` and `connect_buttons` methods
    This class needs a src_folder, a ui_name, a ui_title. The ui names and titles are used to build
    a dialog from the ui file of the same name and parametrise the dialog
    """
    def __init__(self, ui_name, ui_title, src_folder="", params=None, app=None, size=None):
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

        dlg = create_clearmap_widget(f'{ui_name}.ui', patch_parent_class='QDialog', window_title=ui_title)

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
        self.dlg.exec()


class PatternDialog(WizardDialog):
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
        self.tile_extension = tile_extension
        self.min_file_number = min_file_number
        super().__init__('pattern_prompt', 'File paths wizard', src_folder, params, app, [600, None])
        self.n_image_groups = 0
        self.pattern_strings = {}
        self.patterns_finders = None

    def setup(self):
        """
        Setup the dialog after creation from the ui file.
        This method is called automatically in the constructor
        """
        self.n_image_groups = 0
        self.dlg.patternToolBox = QToolBox(parent=self.dlg)
        self.dlg.patternWizzardLayout.insertWidget(0, self.dlg.patternToolBox)
        self.pattern_strings = {}
        self.patterns_finders = self.get_patterns()
        for pattern_idx, p_finder in enumerate(self.patterns_finders):
            self.add_group()
            for axis, digits_idx in enumerate(p_finder.pattern.digit_clusters):
                label_widget, pattern_widget, combo_widget = self.get_widgets(pattern_idx, axis)
                pattern_widget.setText(p_finder.pattern.highlight_digits(axis))
                self.enable_widgets((label_widget, pattern_widget, combo_widget))
            for ax in range(axis + 1, 4):  # Hide the rest
                self.hide_widgets(self.get_widgets(pattern_idx, ax))

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
        label_widget = getattr(page, f'label0_{axis}')
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
        pattern_idx = self.dlg.patternToolBox.currentIndex()
        pattern = self.patterns_finders[pattern_idx].pattern
        for subpattern_idx, digit_cluster in enumerate(pattern.digit_clusters):
            _, _, combo_widget = self.get_widgets(pattern_idx, subpattern_idx)
            axis_name = combo_widget.currentText()
            n_axis_chars = len(pattern.digit_clusters[subpattern_idx])

            if axis_name == 'C':
                raise NotImplementedError('Channel splitting is not implemented yet')
            else:
                pattern_element = '<{axis},{length}>'.format(axis=axis_name, length=n_axis_chars)
                pattern.pattern_elements[subpattern_idx] = pattern_element

        result_widget = self.dlg.patternToolBox.widget(pattern_idx).result
        pattern_string = pattern.get_formatted_pattern()
        pattern_string = os.path.join(self.patterns_finders[pattern_idx].folder, pattern_string)
        pattern_string = os.path.relpath(pattern_string, start=self.src_folder)

        result_widget.setText(pattern_string)

        channel_name = self.dlg.patternToolBox.widget(pattern_idx).channelComboBox.currentText()
        self.pattern_strings[channel_name] = pattern_string

    def get_patterns(self):
        """
        Scan the current source folder to get the pattern finders for the image files

        Returns
        -------
        list(PatternFinder)
            The pattern finders for the source folder
        """
        splash, progress_bar = make_splash(bar_max=100)
        splash.show()
        with ThreadPool(processes=1) as pool:
            result = pool.apply_async(pattern_finders_from_base_dir,
                                      [self.src_folder, None, self.min_file_number, self.tile_extension])
            while not result.ready():
                result.wait(0.25)
                update_pbar(self.app, progress_bar, 1)  # TODO: real update
                self.app.processEvents()
            pattern_finders = result.get()
        update_pbar(self.app, progress_bar, 100)
        splash.finish(self.dlg)
        return pattern_finders

    def save_results(self):
        """
        Save the file patterns to the `sample` configuration file and close the dialog
        """
        config_loader = ConfigLoader(self.src_folder)
        sample_cfg = config_loader.get_cfg('sample')
        for channel_name, pattern_string in self.pattern_strings.items():
            sample_cfg['src_paths'][channel_name] = pattern_string
        sample_cfg.write()
        self.params.cfg_to_ui()
        self.dlg.close()


class SamplePickerDialog(WizardDialog):
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
        self.dlg.addGroupPushButton.clicked.connect(functools.partial(self.__handle_add_group, add_to_params=True))
        self.dlg.groupsComboBox.currentIndexChanged.connect(self.__handle_group_changed)
        self.dlg.buttonBox.accepted.connect(self.__apply_changes)
        self.dlg.buttonBox.rejected.connect(self.dlg.close)

        selected_model = self.list_selection.mOuput.model()
        selected_model.rowsInserted.connect(self.__update_current_group_paths)  # Update group when selection updated
        selected_model.rowsRemoved.connect(self.__update_current_group_paths)  # Update group when selection updated

    def parse_sample_folders(self):
        """
        Scan the source folder to find the sample folders based on
        the presence of a `sample_params.cfg` file (skip `config_snapshots` folders)

        Returns
        -------
        list(str)
            The list of sample folders
        """
        sample_folders = []
        for root, dirs, files in os.walk(self.src_folder):
            for fldr in dirs:
                if fldr == 'config_snapshots' or root.endswith('config_snapshots'):
                    continue
                fldr = os.path.join(root, fldr)
                if 'sample_params.cfg' in os.listdir(fldr):
                    sample_folders.append(fldr)
        return sample_folders

    def __apply_changes(self):
        """
        Save the selected groups of samples to the configuration file and close the dialog
        """
        for group, paths in enumerate(self.group_paths):
            if group > self.params.n_groups:
                self.params.add_group()
            if paths:
                self.params.set_paths(group+1, paths)
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
        if add_to_params:
            self.params.add_group()
        self.group_paths.append([])


class LandmarksSelectorDialog(WizardDialog):  # TODO: bind qColorDialog to color buttons
    """
    A dialog to select landmarks in 3D space for registration
    The dialog allows to select landmarks in two views (fixed and moving)
    The landmarks are displayed in two 3D viewers with matching colors
    """

    def __init__(self, app=None):
        """
        Initialize the dialog

        Parameters
        ----------
        app: QApplication (optional)
            The QApplication instance to use. If None, ClearMap will try to use the existing instance
        """
        self.markers = []
        self.coords = []
        super().__init__('landmark_selector', 'Landmark selector', app=app)
        # self.dlg.setModal(False)
        self.data_viewers = {}
        self.dlg.show()

    def setup(self):
        """
        Setup the dialog after creation from the ui file.
        This method is called automatically in the constructor
        """
        btn = self.dlg.marker0RadioButton
        btn.setChecked(True)
        color_btn = self.dlg.marker0ColorBtn
        self.markers = [(btn, color_btn)]
        self.coords = [[(np.nan, np.nan, np.nan),
                        (np.nan, np.nan, np.nan)]]

    def __len__(self):
        return len(self.markers)

    @property
    def current_marker(self):
        """
        Get the index of the currently selected  marker
        Returns
        -------
        int : the index of the currently selected marker
        """
        return [marker[0].isChecked() for marker in self.markers].index(True)

    # def get_marker_btn(self, idx):
    #     return getattr(self.dlg, f'marker{idx}RadioButton', None)
    #
    # def get_marker_color_label(self, idx):
    #     return getattr(self.dlg, f'marker{idx}ColorLabel', None)

    def connect_buttons(self):
        """
        Connect the buttons to their slots.
        This method is called automatically in the constructor
        """
        self.dlg.addMarkerPushButton.clicked.connect(self.add_marker)
        self.dlg.delMarkerPushButton.clicked.connect(self.remove_marker)
        # self.dlg.buttonBox.accepted.connect(self.accept)
        self.dlg.buttonBox.rejected.connect(self.dlg.close)

    @property
    def fixed_coords(self):
        return np.array([c[0] for c in self.coords])

    @property
    def moving_coords(self):
        return np.array([c[1] for c in self.coords])

    def set_fixed_coords(self, x, y, z):
        self.coords[self.current_marker][0] = (x, y, z)
        self._set_viewer_coords(0, self.fixed_coords)

    def set_moving_coords(self, x, y, z):
        self.coords[self.current_marker][1] = (x, y, z)
        self._set_viewer_coords(1, self.moving_coords)

    def _set_viewer_coords(self, viewer_idx, coords):
        viewer = self.data_viewers[viewer_idx]
        viewer.scatter_coords.set_data({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'z': coords[:, 2],
            'colour': np.array([QColor(col) for col in self.colors])
        })
        viewer.refresh()

    @property
    def __style_sheets(self):
        return [color_btn.styleSheet() for _, color_btn in self.markers]

    @property
    def colors(self):
        """
        Get the ordered list of colors of the markers

        Returns
        -------
        list(str)
            The markers colors
        """
        return [sheet.replace('background-color: ', '').strip() for sheet in self.__style_sheets]

    @property
    def current_color(self):
        """
        Get the color of the currently selected marker

        Returns
        -------
        str
            The color of the currently selected marker
        """
        return self.colors[self.current_marker]

    def add_marker(self):
        new_idx = len(self)
        btn = QRadioButton(f'Marker {new_idx}:', self.dlg)
        btn.setObjectName(f'marker{new_idx}RadioButton')
        color_btn = QPushButton(self.dlg)
        color_btn.setObjectName(f'marker{new_idx}ColorBtn')
        color_btn.setStyleSheet(f'background-color: {self.get_new_color()}')
        self.dlg.formLayout.insertRow(len(self), btn, color_btn)
        self.markers.append((btn, color_btn))
        self.coords.append([None, None])
        btn.click()

    def remove_marker(self):
        if self.current_marker == len(self) - 1:
            self.markers[-2][0].setChecked(True)
        btn, color_btn = self.markers[len(self) - 1]
        for widg in (btn, color_btn):
            widg.setParent(None)
            widg.deleteLater()
        self.markers.pop()
        self.coords.pop()

    def get_new_color(self):
        color = QColor(*[c*255 for c in get_pseudo_random_color()])
        while color.name() in self.colors:
            color = QColor(*[c*255 for c in get_pseudo_random_color()])
        return color.name()


class StructurePickerWidget(QTreeWidget):
    LIGHT_COLOR = 'white'
    DARK_COLOR = '#2E3436'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.root = self.parse_json()
        self.build_tree(self.root, self)
        self.header().resizeSection(0, 300)
        self.setHeaderLabels(['Structure name', 'ID', 'Color', ''])  # TODO: see why 4 columns
        # self.itemClicked.connect(self.print_id)

    def print_id(self, itm, col):
        print([itm.text(i) for i in range(3)])

    @staticmethod
    def parse_json():
        with open(annotation.label_file, 'r') as json_handle:  # FIXME: make parameter in case it hasn't been initialised
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


class StructureSelector(WizardDialog):
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
                                  if proc and 'python' in proc.name().lower() and 'clearmap' in proc.exe().lower()]
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
        self.pool.submit(gpu_params, self.gpu_proc_file_path)

    def handle_proc_changed(self, file_path):
        if file_path == self.gpu_proc_file_path:
            self.handle_gpu_vals_updated()
        elif file_path == self.cpu_proc_file_path:
            self.handle_cpu_vals_updated()

    def handle_gpu_vals_updated(self):
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
