import os
import re
from multiprocessing.pool import ThreadPool


import numpy as np
import pyqtgraph as pg

from skimage import transform as sk_transform  # Slowish

import qdarkstyle
from PyQt5.QtGui import QColor


from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QWidget, QDialogButtonBox, QListWidget, QHBoxLayout, QPushButton, QVBoxLayout, QTableWidget, \
    QTableWidgetItem, QToolBox

from ClearMap.IO import TIF
from ClearMap.IO.metadata import pattern_finders_from_base_dir
from ClearMap.Settings import resources_path
from ClearMap.Visualization import Plot3d as plot_3d
from ClearMap.config.config_loader import ConfigLoader
from ClearMap.gui.dialogs import make_splash, get_directory_dlg, update_pbar
from ClearMap.gui.gui_utils import pseudo_random_rgb_array, create_clearmap_widget


def setup_mini_brain(mini_brain_scaling=(5, 5, 5)):  # TODO: scaling in prefs
    atlas_path = os.path.join(resources_path, 'Atlas', 'ABA_25um_annotation.tif')  # TODO: function of chosen atlas ?
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
        if img is not None:
            self.shape = img.shape
        else:
            self.shape = None
        self.rectangles = []

    def setup(self, img, params, parent=None):
        self.img = img
        self.params = params
        self.parent = parent
        self.shape = img.shape
        self.rectangles = []

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
            graph = getattr(self.parent, self.parent.graph_names[axis])  # REFACTOR: not the cleanest
        except KeyError:
            print('Wrong graphs displayed, skipping')
            return
        rect_itm._generate_picture()
        graph.view.update()

    def add_cropping_bars(self):
        self.rectangles = []
        matched_axes = (1, 0, 1)  # TODO: compute
        for i, dv in enumerate(self.dvs):
            min_rect = RectItem(QRectF(0, 0, 0, self.shape[matched_axes[i]]))
            self.rectangles.append(min_rect)
            dv.view.addItem(min_rect)
            max_rect = RectItem(QRectF(self.shape[i], 0, 0, self.shape[matched_axes[i]]))
            self.rectangles.append(max_rect)
            dv.view.addItem(max_rect)

    def plot_orthogonal_views(self, img=None, parent=None):
        if img is None:
            img = self.img.array
        if parent is None:
            parent = self.parent
            if parent is None:
                raise ValueError('Parent not set')
        x = np.copy(img)
        y = np.copy(img).swapaxes(0, 1)
        z = np.copy(img).swapaxes(0, 2)
        dvs = plot_3d.plot([x, y, z], arange=False, lut='white', parent=parent, sync=False)
        self.dvs = dvs
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

    def get_all_data(self, main_slice_idx, half_slice_thickness=3):
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
            if i < 0:
                continue
            else:
                current_slice = i
            pos = np.vstack((pos, self.get_pos(current_slice)))
            if self.colours is not None:
                current_z_colors = self.get_colours(current_slice)
                colours = np.hstack((colours, current_z_colors))
            sizes = np.hstack((sizes, self.get_symbol_sizes(main_slice_idx, current_slice)))
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
        return self.colours[self.current_slice_indices(current_slice)]

    def current_slice_indices(self, current_slice):
        return self.coordinates[:, self.axis] == current_slice

    def get_pos(self, current_slice):
        return self.coordinates[self.current_slice_indices(current_slice)][:, :2]

    def get_symbols(self, current_slice):
        if self.hemispheres is not None:
            return self.symbols[self.current_slice_indices(current_slice)]
        else:
            return self.default_symbol


class PatternDialog:
    def __init__(self, params, src_folder, app=None):
        self.params = params
        self.src_folder = src_folder
        if app is None:
            app = QtWidgets.QApplication.instance()
        self.app = app

        self.n_image_groups = 0

        dlg = create_clearmap_widget('pattern_prompt.ui', patch_parent_class='QDialog')
        dlg.setWindowTitle('File paths wizard')
        dlg.setupUi()
        dlg.resize(600, dlg.height())
        self.dlg = dlg

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

        self.fix_btn_boxes_text()
        self.connect_buttons()

    def get_widgets(self, image_group_id, axis):
        page = self.dlg.patternToolBox.widget(image_group_id)
        if page is None:
            raise IndexError(f'No widget at index {image_group_id}')
        label_widget = getattr(page, f'label0_{axis}')
        pattern_widget = getattr(page, f'pattern0_{axis}')
        combo_widget = getattr(page, f'pattern0_{axis}ComboBox')

        return label_widget, pattern_widget, combo_widget

    @staticmethod
    def enable_widgets(widgets):
        for w in widgets:
            w.setEnabled(True)

    @staticmethod
    def hide_widgets(widgets):
        for w in widgets:
            w.setVisible(False)

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

        result_widget.setText(pattern_string)

        channel_name = self.dlg.patternToolBox.widget(pattern_idx).channelComboBox.currentText()
        self.pattern_strings[channel_name] = pattern_string

    def fix_btn_boxes_text(self):
        for btn_box in self.dlg.findChildren(QDialogButtonBox):
            if btn_box.property('applyText'):
                btn_box.button(QDialogButtonBox.Apply).setText(btn_box.property('applyText'))

    def get_patterns(self):
        splash, progress_bar = make_splash(bar_max=100)
        splash.show()
        pool = ThreadPool(processes=1)
        result = pool.apply_async(pattern_finders_from_base_dir, [self.src_folder])
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

    def exec(self):
        self.dlg.exec()


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


class SamplePickerDialog:
    def __init__(self, src_folder, params=None, app=None):
        self.group_paths = [[]]
        self.src_folder = src_folder
        self.params = params
        self.app = app

        dlg = create_clearmap_widget('sample_picker.ui', patch_parent_class='QDialog')
        dlg.setWindowTitle('File paths wizard')
        dlg.setupUi()
        self.dlg = dlg
        self.dlg.setMinimumWidth(800)
        self.dlg.setMinimumHeight(600)

        self.fix_btn_boxes_text()

        self.current_group = 1
        for i in range(self.params.n_groups - 1):
            self.handle_add_group()

        self.list_selection = TwoListSelection()
        self.dlg.listPickerLayout.addWidget(self.list_selection)
        # self.dlg.setStyleSheet(qdarkstyle.load_stylesheet())
        self.connect_buttons()
        self.dlg.exec()

    def fix_btn_boxes_text(self):
        for btn_box in self.dlg.findChildren(QDialogButtonBox):
            if btn_box.property('applyText'):
                btn_box.button(QDialogButtonBox.Apply).setText(btn_box.property('applyText'))

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
