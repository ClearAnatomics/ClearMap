# -*- coding: utf-8 -*-
"""
widgets
=======

A set of custom widgets
"""
import json
import os
import re
from multiprocessing.pool import ThreadPool


import numpy as np
import pyqtgraph as pg

from skimage import transform as sk_transform  # Slowish

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QWidget, QDialogButtonBox, QListWidget, QHBoxLayout, QPushButton, QVBoxLayout, QTableWidget, \
    QTableWidgetItem, QToolBox, QRadioButton, QTreeWidget, QTreeWidgetItem

from ClearMap.Alignment.Annotation import annotation
from ClearMap.IO import TIF
from ClearMap.IO.metadata import pattern_finders_from_base_dir
from ClearMap.Settings import atlas_folder
from ClearMap.Visualization.Qt import Plot3d as q_plot_3d
from ClearMap.config.config_loader import ConfigLoader
from ClearMap.gui.dialogs import make_splash, get_directory_dlg, update_pbar
from ClearMap.gui.gui_utils import pseudo_random_rgb_array, create_clearmap_widget, get_pseudo_random_color, is_dark

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


def setup_mini_brain(mini_brain_scaling=(5, 5, 5)):  # TODO: scaling in prefs
    atlas_path = os.path.join(atlas_folder, 'ABA_25um_annotation.tif')
    arr = TIF.Source(atlas_path).array
    return mini_brain_scaling, sk_transform.downscale_local_mean(arr, mini_brain_scaling)


class RectItem(pg.GraphicsObject):  # Derived from https://stackoverflow.com/a/60012800
    def __init__(self, rect, parent=None):
        super().__init__(parent)
        self._rect = rect
        self.picture = QtGui.QPicture()
        self._generate_picture()
        self.name = 'rect'

    def __str__(self):
        return 'Rect {}, coordinates: {}'.format(self.name, self.rect.getCoords())

    @property
    def rect(self):
        return self._rect

    # TODO: setWidth, setLeft ... that call self.rect.setWidth + self._generate_picture

    def _generate_picture(self):
        painter = QtGui.QPainter(self.picture)
        painter.setPen(pg.mkPen("#FFFF007d"))
        painter.setBrush(pg.mkBrush("#2e34367d"))
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())


class RedCross(pg.GraphicsObject):
    def __init__(self, coords=None, size=[10, 10], parent=None):
        super().__init__(parent)
        self.color = "#FF00007d"
        if coords is None:
            coords = [0, 0]
        self.size = size
        self._rect = QRectF(coords[0]-self.size[0]/2, coords[1]-self.size[1]/2, self.size[0], self.size[1])  # TODO@ set as fraction of image size
        self.coords = coords
        self.picture = QtGui.QPicture()

    def set_coords(self, coords):
        x, y = coords
        self.coords = coords
        self._rect.setCoords(x-self.size[0]/2, y-self.size[1]/2,
                             x+self.size[0]/2, y+self.size[1]/2)
        self._generate_picture()

    def _generate_picture(self):
        painter = QtGui.QPainter(self.picture)
        pen = pg.mkPen(self.color)
        pen.setWidth(4)
        painter.setPen(pen)
        painter.setBrush(pg.mkBrush(self.color))
        painter.drawLine(self._rect.topLeft(), self._rect.bottomRight())
        painter.drawLine(self._rect.bottomLeft(), self._rect.topRight())
        painter.drawPicture(0, 0, self.picture)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):  # TODO: check if we need this method
        return QRectF(self.picture.boundingRect())


class OrthoViewer(object):
    def __init__(self, img=None, parent=None):
        self.img = img
        self.parent = parent
        self.params = None
        self.rectangles = []
        self.dvs = []

    def setup(self, img, params, parent=None):
        self.img = img
        self.params = params
        self.parent = parent
        self.rectangles = []

    @property
    def shape(self):
        return self.img.shape if self.img is not None else None

    @property
    def width(self):
        return self.shape[0]

    @property
    def height(self):
        return self.shape[1]

    @property
    def depth(self):
        return self.shape[2]

    def update_x_min(self, val):
        if self.params is not None:
            val = self.params.scale_x(val)
        self._update_rect('x', val, 'min')

    def update_x_max(self, val):
        if self.params is not None:
            val = self.params.scale_x(val)
        self._update_rect('x', val, 'max')

    def update_y_min(self, val):
        if self.params is not None:
            val = self.params.scale_y(val)
        self._update_rect('y', val, 'min')

    def update_y_max(self, val):
        if self.params is not None:
            val = self.params.scale_y(val)
        self._update_rect('y', val, 'max')

    def update_z_min(self, val):
        if self.params is not None:
            val = self.params.scale_z(val)
        self._update_rect('z', val, 'min')

    def update_z_max(self, val):
        if self.params is not None:
            val = self.params.scale_z(val)
        self._update_rect('z', val, 'max')

    def update_ranges(self):
        self.update_x_min(self.params.crop_x_min)
        self.update_x_max(self.params.crop_x_max)
        self.update_y_min(self.params.crop_y_min)
        self.update_y_max(self.params.crop_y_max)
        self.update_z_min(self.params.crop_z_min)
        self.update_z_max(self.params.crop_z_max)

    def get_rect(self, axis, min_or_max):
        axes = ('x', 'y', 'z')
        if axis in axes:
            axis = axes.index(axis)
        idx = axis * 2 + (min_or_max == 'max')
        return self.rectangles[idx]

    def _update_rect(self, axis, val, min_or_max='min'):
        if not self.rectangles:
            return
        rect_itm = self.get_rect(axis, min_or_max)
        if min_or_max == 'min':
            rect_itm.rect.setWidth(val)
        else:
            rect_itm.rect.setLeft(val)
        try:
            graph = self.parent.graph_by_name(axis)
        except KeyError:
            print('Wrong graphs displayed, skipping')
            return
        rect_itm._generate_picture()
        graph.view.update()

    def add_cropping_bars(self):
        self.rectangles = []
        y_axis_idx = (1, 2, 0)
        for i, dv in enumerate(self.dvs):
            height = self.shape[y_axis_idx[i]]
            min_rect = RectItem(QRectF(0, 0, 0, height))
            self.rectangles.append(min_rect)
            dv.view.addItem(min_rect)
            max_rect = RectItem(QRectF(self.shape[i], 0, 0, height))
            self.rectangles.append(max_rect)
            dv.view.addItem(max_rect)

    def plot_orthogonal_views(self, img=None, parent=None):
        if img is None:
            img = self.img.array
        if parent is None:
            parent = self.parent
            if parent is None:
                raise ValueError('Parent not set')
        xy = np.copy(img)
        yz = np.copy(img).transpose((1, 2, 0))
        zx = np.copy(img).transpose((2, 0, 1))
        dvs = q_plot_3d.plot([xy, yz, zx], arange=False, lut='white', parent=parent, sync=False)
        self.dvs = dvs
        # FIXME: disable axes buttons
        return dvs


class PbarWatcher(QWidget):  # Inspired from https://stackoverflow.com/a/66266068
    progress_name_changed = QtCore.pyqtSignal(str)
    progress_changed = QtCore.pyqtSignal(int)
    max_changed = QtCore.pyqtSignal(int)

    main_max_changed = QtCore.pyqtSignal(int)
    main_progress_changed = QtCore.pyqtSignal(int)

    def __init__(self, max_progress=100, main_max_progress=1, parent=None):
        super().__init__(parent)
        self.__progress = 0
        self.__main_progress = 1
        self.__max_progress = max_progress
        self.__main_max_progress = main_max_progress
        self.range_fraction = 1

        self.log_path = None
        self.previous_log_length = 0  # The log length at the end of the previous operation
        self.n_dones = 0
        self.pattern = None

    def get_progress(self):
        return self.__progress

    def set_progress(self, value):
        if self.__progress == value:
            return
        self.__progress = round(value)
        self.progress_changed.emit(self.__progress)

    def set_main_progress(self, value):
        if self.__main_progress == value:
            return
        self.__main_progress = round(value)
        self.main_progress_changed.emit(self.__main_progress)

    def increment_main_progress(self, increment=1):
        self.set_main_progress(self.__main_progress + round(increment))

    def increment(self, increment):
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

    def __match(self, line):
        if isinstance(self.pattern, tuple):  # TODO: cache
            return self.pattern[0] in line and self.pattern[1].match(line)  # Most efficient
        elif isinstance(self.pattern, str):
            return self.pattern in line
        elif isinstance(self.pattern, re.Pattern):
            return self.pattern.match(line)

    def count_dones(self):
        with open(self.log_path, 'r') as log:
            log.seek(self.previous_log_length)
            new_lines = log.readlines()
        n_dones = len([ln for ln in new_lines if self.__match(ln)])
        self.n_dones += n_dones
        self.previous_log_length += self.__get_log_bytes(new_lines)
        return self.n_dones

    def reset_log_length(self):  # To bind
        with open(self.log_path, 'r') as log:
            self.previous_log_length = self.__get_log_bytes(log.readlines())
            self.n_dones = 0

    def __get_log_bytes(self, log):
        return sum([len(ln) for ln in log])

    def prepare_for_substep(self, step_length, pattern, step_name):
        """

        Parameters
        ----------
        step_name
            str
        step_length
            int The number of steps in the operation
        pattern
            str or re.Pattern or (str, re.Pattern) the text to look for in the logs to check for progress

        Returns
        -------

        """
        self.max_progress = step_length
        self.pattern = pattern
        self.reset_log_length()
        self.set_progress(0)
        self.progress_name_changed.emit(step_name)


class Scatter3D:
    def __init__(self, coordinates, smarties=False, colors=None, hemispheres=None, half_slice_thickness=None):
        self.default_symbol = '+'
        self.alternate_symbol = 'p'
        self.half_slice_thickness = half_slice_thickness
        self.coordinates = coordinates
        self.axis = 2
        if smarties and colors is None:
            n_samples = self.coordinates.shape[0]
            self.colours = (pseudo_random_rgb_array(n_samples) * 255).astype(np.int)
            self.colours = np.array([QColor(*col) for col in self.colours])
        else:
            self.colours = colors
        self.hemispheres = hemispheres
        if self.hemispheres is not None:
            self.symbols = np.chararray(self.hemispheres.shape)
            self.symbols[self.hemispheres == 0] = self.default_symbol
            self.symbols[self.hemispheres == 255] = self.alternate_symbol
            self.symbols = self.symbols.decode()

    def get_all_data(self, main_slice_idx, half_slice_thickness=3):  # FIXME: rename
        """
        Get surrounding markers

        Parameters
        ----------
        main_slice_idx int

        half_slice_thickness int

        Returns
        -------

        """
        pos = np.empty((0, 2))
        if self.colours is not None:
            if self.colours.ndim == 1:
                colours = np.empty(0)
            else:
                colours = np.empty((0, self.colours.shape[1]))
        sizes = np.empty(0)
        if self.half_slice_thickness is not None:
            half_slice_thickness = self.half_slice_thickness
        for i in range(main_slice_idx - half_slice_thickness, main_slice_idx + half_slice_thickness):
            if i < 0:  # or i > self.coordinates[:, 2].max()
                continue
            else:
                current_slice = i
            pos = np.vstack((pos, self.get_pos(current_slice)))
            if self.colours is not None:
                current_z_colors = self.get_colours(current_slice)
                colours = np.hstack((colours, current_z_colors))
            sizes = np.hstack((sizes, self.get_symbol_sizes(main_slice_idx, current_slice, half_size=half_slice_thickness)))
        data = {'pos': pos,
                'size': sizes}
        if self.colours is not None:
            data['pen'] = [pg.mkPen(c) for c in colours]
        return data

    def get_symbol_sizes(self, main_slice_idx, slice_idx, half_size=3):
        marker_size = round(10 * ((half_size - abs(main_slice_idx - slice_idx)) / half_size))
        n_markers = len(self.get_pos(slice_idx))
        return np.full(n_markers, marker_size)

    def get_colours(self, current_slice):
        indices = self.current_slice_indices(current_slice)
        if indices is not None:
            return self.colours[indices]
        else:
            return np.array([])

    def current_slice_indices(self, current_slice):
        if len(self.coordinates):
            return self.coordinates[:, self.axis] == current_slice

    def get_pos(self, current_slice):
        indices = self.current_slice_indices(current_slice)
        if indices is not None:
            slice_coords = self.coordinates[indices]
            return np.vstack([slice_coords[:, i] for i in range(3) if i != self.axis]).T  # coordinates in the two other axes
            # return self.coordinates[indices][:, :2]
        else:
            return np.empty((0, 2))

    def get_symbols(self, current_slice):
        if self.hemispheres is not None:
            indices = self.current_slice_indices(current_slice)
            if indices is not None:
                return self.symbols[indices]
            else:
                return np.array([])
        else:
            return self.default_symbol


# Adapted from https://stackoverflow.com/a/54917151 by https://stackoverflow.com/users/6622587/eyllanesc
class TwoListSelection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_layout()
        # self.app = app

    def setup_layout(self):
        lay = QHBoxLayout(self)
        self.mInput = QListWidget()
        self.mOuput = QListWidget()

        move_btns, up_down_btns = self.layout_buttons()

        lay.addWidget(self.mInput)
        lay.addLayout(move_btns)
        lay.addWidget(self.mOuput)
        lay.addLayout(up_down_btns)

        self.update_buttons_status()
        self.connections()

    def layout_buttons(self):
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

    def connections(self):
        self.mInput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mOuput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mBtnMoveToAvailable.clicked.connect(self.on_mBtnMoveToAvailable_clicked)
        self.mBtnMoveToSelected.clicked.connect(self.on_mBtnMoveToSelected_clicked)
        self.mButtonToAvailable.clicked.connect(self.on_mButtonToAvailable_clicked)
        self.mButtonToSelected.clicked.connect(self.on_mButtonToSelected_clicked)
        self.mBtnUp.clicked.connect(self.on_mBtnUp_clicked)
        self.mBtnDown.clicked.connect(self.on_mBtnDown_clicked)

    @QtCore.pyqtSlot()
    def on_mBtnMoveToAvailable_clicked(self):
        self.mOuput.addItem(self.mInput.takeItem(self.mInput.currentRow()))

    @QtCore.pyqtSlot()
    def on_mBtnMoveToSelected_clicked(self):
        self.mInput.addItem(self.mOuput.takeItem(self.mOuput.currentRow()))

    @QtCore.pyqtSlot()
    def on_mButtonToAvailable_clicked(self):
        while self.mOuput.count() > 0:
            self.mInput.addItem(self.mOuput.takeItem(0))

    @QtCore.pyqtSlot()
    def on_mButtonToSelected_clicked(self):
        while self.mInput.count() > 0:
            self.mOuput.addItem(self.mInput.takeItem(0))

    @QtCore.pyqtSlot()
    def on_mBtnUp_clicked(self):
        row = self.mOuput.currentRow()
        currentItem = self.mOuput.takeItem(row)
        self.mOuput.insertItem(row - 1, currentItem)
        self.mOuput.setCurrentRow(row - 1)

    @QtCore.pyqtSlot()
    def on_mBtnDown_clicked(self):
        row = self.mOuput.currentRow()
        currentItem = self.mOuput.takeItem(row)
        self.mOuput.insertItem(row + 1, currentItem)
        self.mOuput.setCurrentRow(row + 1)

    def addAvailableItems(self, items):
        self.mInput.addItems(items)

    def setSelectedItems(self, items):
        self.mOuput.clear()
        self.mOuput.addItems(items)

    def get_left_elements(self):
        r = []
        for i in range(self.mInput.count()):
            it = self.mInput.item(i)
            r.append(it.text())
        return r

    def get_right_elements(self):
        r = []
        for i in range(self.mOuput.count()):
            it = self.mOuput.item(i)
            r.append(it.text())
        return r


class DataFrameWidget(QWidget):
    def __init__(self, df, n_digits=2, parent=None):
        super().__init__(parent)
        self.df = df
        self.n_digits = n_digits
        self.table = QTableWidget(len(df.index), df.columns.size, parent=parent)
        self.table.setHorizontalHeaderLabels(self.df.columns)
        self.set_content()

    def set_content(self):
        for i in range(len(self.df.index)):
            for j in range(self.df.columns.size):
                v = self.df.iloc[i, j]
                if self.df.dtypes[j] == np.float_:
                    self.table.setItem(i, j, QTableWidgetItem(f'{v:.{self.n_digits}}'))
                else:
                    self.table.setItem(i, j, QTableWidgetItem(f'{v}'))


class WizardDialog:
    def __init__(self, src_folder, ui_name, ui_title, size, params=None, app=None):
        self.src_folder = src_folder
        self.params = params
        if app is None:
            app = QtWidgets.QApplication.instance()
        self.app = app

        dlg = create_clearmap_widget(f'{ui_name}.ui', patch_parent_class='QDialog')
        dlg.setWindowTitle(ui_title)
        dlg.setupUi()

        if size is not None:
            if size[0] is None:
                size[0] = dlg.width()
            if size[1] is None:
                size[1] = dlg.height()
            dlg.resize(size[0], size[1])

        self.dlg = dlg

        self.setup()

        # self.dlg.setStyleSheet(qdarkstyle.load_stylesheet())
        self.fix_btn_boxes_text()
        self.connect_buttons()

    def setup(self):
        raise NotImplementedError

    def fix_btn_boxes_text(self):
        for btn_box in self.dlg.findChildren(QDialogButtonBox):
            if btn_box.property('applyText'):
                btn_box.button(QDialogButtonBox.Apply).setText(btn_box.property('applyText'))

    def connect_buttons(self):
        raise NotImplementedError

    @staticmethod
    def enable_widgets(widgets):
        for w in widgets:
            w.setEnabled(True)

    @staticmethod
    def hide_widgets(widgets):
        for w in widgets:
            w.setVisible(False)

    def exec(self):
        self.dlg.exec()


class PatternDialog(WizardDialog):
    def __init__(self, src_folder, params=None, app=None, min_file_number=10, tile_extension='.ome.tif'):
        self.tile_extension = tile_extension
        self.min_file_number = min_file_number
        super().__init__(src_folder, 'pattern_prompt', 'File paths wizard', [600, None], params, app)

    def setup(self):
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
            for axis in range(axis + 1, 4):  # Hide the rest
                self.hide_widgets(self.get_widgets(pattern_idx, axis))

    def get_widgets(self, image_group_id, axis):
        page = self.dlg.patternToolBox.widget(image_group_id)
        if page is None:
            raise IndexError(f'No widget at index {image_group_id}')
        label_widget = getattr(page, f'label0_{axis}')
        pattern_widget = getattr(page, f'pattern0_{axis}')
        combo_widget = getattr(page, f'pattern0_{axis}ComboBox')

        return label_widget, pattern_widget, combo_widget

    def add_group(self):
        group_controls = create_clearmap_widget('image_group_ctrls.ui',
                                                patch_parent_class='QWidget')
        group_controls.setupUi()
        self.dlg.patternToolBox.addItem(group_controls, f'Image group {self.n_image_groups}')

        group_controls.patternButtonBox.button(QDialogButtonBox.Apply).clicked.connect(self.validate_pattern)

        self.n_image_groups += 1

    def connect_buttons(self):
        self.dlg.mainButtonBox.button(QDialogButtonBox.Apply).clicked.connect(self.save_results)
        self.dlg.mainButtonBox.button(QDialogButtonBox.Cancel).clicked.connect(self.dlg.close)

    def validate_pattern(self):
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
        splash, progress_bar = make_splash(bar_max=100)
        splash.show()
        pool = ThreadPool(processes=1)
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
        config_loader = ConfigLoader(self.src_folder)
        sample_cfg = config_loader.get_cfg('sample')
        for channel_name, pattern_string in self.pattern_strings.items():
            sample_cfg['src_paths'][channel_name] = pattern_string
        sample_cfg.write()
        self.params.cfg_to_ui()
        self.dlg.close()


class SamplePickerDialog(WizardDialog):
    def __init__(self, src_folder, params=None, app=None):
        super().__init__(src_folder, 'sample_picker', 'File paths wizard', [None, 600], params, app)
        self.exec()

    def setup(self):
        self.group_paths = [[]]
        self.current_group = 1
        for i in range(self.params.n_groups - 1):
            self.handle_add_group()
        self.list_selection = TwoListSelection()
        self.dlg.listPickerLayout.addWidget(self.list_selection)

    def connect_buttons(self):
        self.dlg.mainFolderPushButton.clicked.connect(self.handle_main_folder_clicked)
        self.dlg.addGroupPushButton.clicked.connect(self.handle_add_group)
        self.dlg.groupsComboBox.currentIndexChanged.connect(self.handle_group_changed)
        self.dlg.buttonBox.accepted.connect(self.apply_changes)
        self.dlg.buttonBox.rejected.connect(self.dlg.close)

        selected_model = self.list_selection.mOuput.model()
        selected_model.rowsInserted.connect(self.update_current_group_paths)  # Update group when selection updated
        selected_model.rowsRemoved.connect(self.update_current_group_paths)  # Update group when selection updated

    def apply_changes(self):
        for group, paths in enumerate(self.group_paths):
            if group > self.params.n_groups:
                self.params.add_group()
            if paths:
                self.params.set_paths(group+1, paths)
        self.dlg.close()

    def handle_group_changed(self):
        self.update_current_group_paths()
        current_gp_id = self.dlg.groupsComboBox.currentIndex()
        self.current_group = max(0, current_gp_id) + 1  # WARNING: update current_group after update
        self.list_selection.setSelectedItems(self.group_paths[self.current_group - 1])

    def update_current_group_paths(self):
        self.group_paths[self.current_group - 1] = self.list_selection.get_right_elements()

    def handle_add_group(self):
        self.dlg.groupsComboBox.addItem(f'{self.dlg.groupsComboBox.count() + 1}')
        self.group_paths.append([])

    def handle_main_folder_clicked(self):
        self.src_folder = get_directory_dlg('~/')
        if self.src_folder:
            # self.dlg.groupsComboBox.clear()
            # self.dlg.groupsComboBox.addItems(self.parse_sample_folders())
            self.list_selection.addAvailableItems(self.parse_sample_folders())

    def parse_sample_folders(self):
        sample_folders = []
        for root, dirs, files in os.walk(self.src_folder):
            for fldr in dirs:
                fldr = os.path.join(root, fldr)
                if 'sample_params.cfg' in os.listdir(fldr):
                    sample_folders.append(fldr)
        return sample_folders


class LandmarksSelectorDialog(WizardDialog):  # TODO: bind qColorDialog to color buttons

    def __init__(self, src_folder, params=None, app=None):
        super().__init__(src_folder, 'landmark_selector', 'Landmark selector', None, params, app)
        # self.dlg.setModal(False)
        self.dlg.show()

    def setup(self):
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
        return [marker[0].isChecked() for marker in self.markers].index(True)

    # def get_marker_btn(self, idx):
    #     return getattr(self.dlg, f'marker{idx}RadioButton', None)
    #
    # def get_marker_color_label(self, idx):
    #     return getattr(self.dlg, f'marker{idx}ColorLabel', None)

    def connect_buttons(self):
        self.dlg.addMarkerPushButton.clicked.connect(self.add_marker)
        self.dlg.delMarkerPushButton.clicked.connect(self.remove_marker)
        # self.dlg.buttonBox.accepted.connect(self.accept)
        self.dlg.buttonBox.rejected.connect(self.dlg.close)

    # def accept(self):
    #     self.print_values()
    #     self.dlg.close()
    #
    # def print_values(self):
    #     print(self.current_marker)
    #     print(self.colors)
    #     print(self.coords)

    def fixed_coords(self):
        return np.array([c[0] for c in self.coords])
        # return np.array([c[0] for c in self.coords if c[0] is not None])

    def moving_coords(self):
        return np.array([c[1] for c in self.coords])
        # return np.array([c[1] for c in self.coords if c[1] is not None])

    def set_fixed_coords(self, x, y, z):
        self.coords[self.current_marker][0] = (x, y, z)
        self.data_viewers[0].scatter_coords.coordinates = self.fixed_coords()
        self.data_viewers[0].scatter_coords.colours = np.array([QColor(col) for col in self.colors])
        self.data_viewers[0].refresh()

    def set_moving_coords(self, x, y, z):
        self.coords[self.current_marker][1] = (x, y, z)
        self.data_viewers[1].scatter_coords.coordinates = self.moving_coords()
        self.data_viewers[1].scatter_coords.colours = np.array([QColor(col) for col in self.colors])
        self.data_viewers[1].refresh()

    @property
    def colors(self):
        return [sheet.replace('background-color: ', '').strip() for sheet in self.style_sheets]

    @property
    def current_color(self):
        return self.colors[self.current_marker]

    @property
    def style_sheets(self):
        return [color_btn.styleSheet() for _, color_btn in self.markers]

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
        # self.tree = QTreeWidget()
        self.root = self.parse_json()
        self.build_tree(self.root, self)
        self.itemClicked.connect(self.print_id)

    def print_id(self, itm, col):
        print([itm.text(i) for i in range(3)])

    @staticmethod
    def parse_json():
        with open(annotation.label_file, 'r') as json_handle:
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
