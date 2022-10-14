# -*- coding: utf-8 -*-
"""
params
======

All the classes that define parameters or group thereof for the tabs of the graphical interface
"""

import os
import string
from itertools import combinations

import numpy as np

from ClearMap.Utils.utilities import get_item_recursive
from ClearMap.config.atlas import ATLAS_NAMES_MAP
from ClearMap.gui.gui_utils import create_clearmap_widget

from ClearMap.gui.dialogs import get_directory_dlg
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import QInputDialog, QToolBox, QCheckBox

from ClearMap.config.config_loader import ConfigLoader


__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


class ConfigNotFoundError(Exception):
    pass


class ParamsOrientationError(ValueError):
    pass


class UiParameter(QObject):
    def __init__(self, tab, src_folder=None):
        super().__init__()
        self.tab = tab
        self.src_folder = src_folder
        self._config = None
        self._default_config = None
        self.params_dict = None
        self.attrs_to_invert = []
        self.connect()

    def connect(self):
        """Connect GUI slots here"""
        pass

    def fix_cfg_file(self, f_path):
        """Fix the file if it was copied from defaults, tailor to current sample"""
        pass

    @property
    def path(self):
        return self._config.filename

    def get_config(self, cfg_path):
        self._config = ConfigLoader.get_cfg_from_path(cfg_path)
        if not self._config:
            raise ConfigNotFoundError
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        self._default_config = ConfigLoader.get_cfg_from_path(ConfigLoader.get_default_path(cfg_name))

    def write_config(self):
        self._config.write()

    def reload(self):
        self._config.reload()

    def _translate_state(self, state):
        if state is True:
            state = Qt.Checked
        elif state is False:
            state = Qt.Unchecked
        else:
            raise NotImplementedError('Unknown state {}'.format(state))
        return state

    def ui_to_cfg(self):
        self._ui_to_cfg()
        self.write_config()

    def _ui_to_cfg(self):
        pass

    def cfg_to_ui(self):
        if self.params_dict is None:
            raise NotImplementedError
        else:
            any_amended = False
            for attr, keys_list in self.params_dict.items():
                current_amended = False
                try:
                    val = get_item_recursive(self.config, keys_list)
                except KeyError:  # TODO: add msg
                    val = get_item_recursive(self._default_config, keys_list)
                    any_amended = True
                    current_amended = True
                if attr in self.attrs_to_invert:
                    val = -val
                # Update the UI
                setattr(self, attr, val)
                if current_amended:
                    # Update the config
                    get_item_recursive(self.config, keys_list[:-1])[keys_list[-1]] = val
            if any_amended:
                self.ui_to_cfg()  # Add the newly parsed field

    def is_checked(self, check_box):
        return check_box.checkState() == Qt.Checked

    def set_check_state(self, check_box, state):
        state = self._translate_state(state)
        check_box.setCheckState(state)

    def sanitize_nones(self, val):
        return val if val is not None else -1

    def sanitize_neg_one(self, val):
        return val if val != -1 else None


class UiParameterCollection:
    """
    For multi-section UiParameters that share the same config file. This ensures the file remains consistent.
    """
    def __init__(self, tab, src_folder=None):
        self.tab = tab
        self.src_folder = src_folder
        self.config = None

    def fix_cfg_file(self, f_path):
        """Fix the file if it was copied from defaults, tailor to current sample"""
        pass

    @property
    def params(self):
        raise NotImplementedError('Please subclass UiParameterCollection and implement params property')

    def get_config(self, cfg_path):
        self.config = ConfigLoader.get_cfg_from_path(cfg_path)
        if not self.config:
            raise ConfigNotFoundError
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        default_path = ConfigLoader.get_default_path(cfg_name)
        self._default_config = ConfigLoader.get_cfg_from_path(default_path)
        for param in self.params:
            param._config = self.config
            param._default_config = self._default_config

    def write_config(self):
        self.config.write()

    def reload(self):
        self.config.reload()

    def ui_to_cfg(self):
        self.write_config()

    def cfg_to_ui(self):
        for param in self.params:
            param.cfg_to_ui()


class AlignmentParams(UiParameterCollection):
    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.stitching_general = GeneralStitchingParams(tab, src_folder)
        self.stitching_rigid = RigidStitchingParams(tab, src_folder)
        self.stitching_wobbly = WobblyStitchingParams(tab, src_folder)
        self.registration = RegistrationParams(tab, src_folder)

    def fix_cfg_file(self, f_path):
        cfg = ConfigLoader.get_cfg_from_path(f_path)
        pipeline_name, ok = QInputDialog.getItem(self.tab, 'Please select pipeline type',
                                                 'Pipeline name:', ['CellMap', 'TubeMap', 'Both'], 0, False)
        if not ok:
            raise ValueError('Missing sample ID')
        cfg['pipeline_name'] = pipeline_name
        cfg.write()

    @property
    def params(self):
        return self.stitching_general, self.stitching_rigid, self.stitching_wobbly, self.registration

    @property
    def all_stitching_params(self):
        return self.stitching_general, self.stitching_rigid, self.stitching_wobbly

    @property
    def pipeline_name(self):
        return self.config['pipeline_name']

    @pipeline_name.setter
    def pipeline_name(self, name):
        self.config['pipeline_name'] = name

    # @property
    # def pipeline_is_cell_map(self):
    #     return self.pipeline_name.lower().replace('_', '') == 'cellmap'
    #
    # @property
    # def pipeline_is_tube_map(self):
    #     return self.pipeline_name.lower().replace('_', '') == 'tubemap'
    #
    # @property
    # def pipeline_is_both(self):
    #     return self.pipeline_name.lower().replace('_', '') == 'both'


class SampleParameters(UiParameter):

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'sample_id': ['sample_id'],
            'use_id_as_prefix': ['use_id_as_prefix'],
            'tile_extension': ['src_paths', 'tile_extension'],
            'raw_path': ['src_paths', 'raw'],
            'autofluo_path': ['src_paths', 'autofluorescence'],
            'arteries_path': ['src_paths', 'arteries'],
            'raw_resolution': ['resolutions', 'raw'],
            'autofluorescence_resolution': ['resolutions', 'arteries'],
            'arteries_resolution': ['resolutions', 'autofluorescence'],
            'slice_x': ['slice_x'],
            'slice_y': ['slice_y'],
            'slice_z': ['slice_z'],
            'orientation': ['orientation']  # WARNING: Finish by orientation in case invalid,
        }
        if self.sample_id:
            self.handle_sample_id_changed(self.sample_id)

    @property
    def config(self):
        return self._config

    def connect(self):
        self.tab.sampleIdTxt.editingFinished.connect(self.handle_sample_id_changed)
        self.tab.useIdAsPrefixCheckBox.stateChanged.connect(self.handle_use_id_as_prefix_changed)

        self.tab.tileExtensionLineEdit.textChanged.connect(self.handle_tile_extension_changed)
        self.tab.rawPath.textChanged.connect(self.handle_raw_path_changed)
        self.tab.arteriesPathOptionalPlainTextEdit.textChangedConnect(self.handle_arteries_path_changed)
        self.tab.autofluoPathOptionalPlainTextEdit.textChangedConnect(self.handle_autofluo_path_changed)

        self.tab.orient_x.currentTextChanged.connect(self.handle_orientation_changed)
        self.tab.orient_y.currentTextChanged.connect(self.handle_orientation_changed)
        self.tab.orient_z.currentTextChanged.connect(self.handle_orientation_changed)

        self.tab.sliceXDoublet.valueChangedConnect(self.handle_slice_x_changed)
        self.tab.sliceYDoublet.valueChangedConnect(self.handle_slice_y_changed)
        self.tab.sliceZDoublet.valueChangedConnect(self.handle_slice_z_changed)

        self.tab.rawResolutionTriplet.valueChangedConnect(self.handle_raw_resolution_changed)
        self.tab.autofluorescenceResolutionTriplet.valueChangedConnect(self.handle_autofluo_resolution_changed)
        self.tab.arteriesResolutionTriplet.valueChangedConnect(self.handle_arteries_resolution_changed)

    def _ui_to_cfg(self):
        self._config['base_directory'] = self.src_folder

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()

    def fix_cfg_file(self, f_path):
        cfg = ConfigLoader.get_cfg_from_path(f_path)
        cfg['base_directory'] = os.path.dirname(f_path)
        if not self.sample_id:
            sample_id, ok = QInputDialog.getText(self.tab, 'Warning: missing ID',
                                                 '<b>Missing sample ID</b><br>Please input below')
            self.sample_id = sample_id
            if not ok:
                raise ValueError('Missing sample ID')
        cfg['sample_id'] = self.sample_id
        cfg.write()

    # Sample params
    @property
    def sample_id(self):
        return self.tab.sampleIdTxt.text()

    @sample_id.setter
    def sample_id(self, _id):
        self.tab.sampleIdTxt.setText(_id)

    def handle_sample_id_changed(self, _id):
        if self.config is not None:
            self.config['sample_id'] = self.sample_id
            self.ui_to_cfg()   # FIXME: check

    @property
    def use_id_as_prefix(self):
        return self.tab.useIdAsPrefixCheckBox.isChecked()

    @use_id_as_prefix.setter
    def use_id_as_prefix(self, value):
        self.set_check_state(self.tab.useIdAsPrefixCheckBox, value)

    def handle_use_id_as_prefix_changed(self, _):
        self._config['use_id_as_prefix'] = self.use_id_as_prefix

    @property
    def tile_extension(self):
        return self.tab.tileExtensionLineEdit.text()

    @tile_extension.setter
    def tile_extension(self, value):
        self.tab.tileExtensionLineEdit.setText(value)

    def handle_tile_extension_changed(self):
        self._config['src_paths']['tile_extension'] = self.tile_extension

    @property
    def raw_path(self):
        f_path = self.tab.rawPath.toPlainText()
        f_path = f_path if f_path else None
        return f_path

    @raw_path.setter
    def raw_path(self, f_path):
        f_path = f_path if f_path is not None else ''
        self.tab.rawPath.setPlainText(f_path)

    def handle_raw_path_changed(self):
        self.config['src_paths']['raw'] = self.raw_path

    @property
    def autofluo_path(self):
        return self.tab.autofluoPathOptionalPlainTextEdit.text()

    @autofluo_path.setter
    def autofluo_path(self, f_path):
        self.tab.autofluoPathOptionalPlainTextEdit.setText(f_path)

    def handle_autofluo_path_changed(self):
        self.config['src_paths']['autofluorescence'] = self.autofluo_path

    @property
    def arteries_path(self):
        return self.tab.arteriesPathOptionalPlainTextEdit.text()

    @arteries_path.setter
    def arteries_path(self, f_path):
        self.tab.arteriesPathOptionalPlainTextEdit.setText(f_path)

    def handle_arteries_path_changed(self):
        self.config['src_paths']['arteries'] = self.arteries_path

    @property
    def raw_resolution(self):
        return self.tab.rawResolutionTriplet.getValue()

    @raw_resolution.setter
    def raw_resolution(self, res):
        self.tab.rawResolutionTriplet.setValue(res)

    def handle_raw_resolution_changed(self, res):
        self.config['resolutions']['raw'] = self.raw_resolution

    @property
    def autofluorescence_resolution(self):
        return self.tab.autofluorescenceResolutionTriplet.getValue()

    @autofluorescence_resolution.setter
    def autofluorescence_resolution(self, res):
        self.tab.autofluorescenceResolutionTriplet.setValue(res)

    def handle_autofluo_resolution_changed(self, res):
        self.config['resolutions']['autofluorescence'] = self.autofluorescence_resolution

    @property
    def arteries_resolution(self):
        return self.tab.arteriesResolutionTriplet.getValue()

    @arteries_resolution.setter
    def arteries_resolution(self, res):
        self.tab.arteriesResolutionTriplet.setValue(res)

    def handle_arteries_resolution_changed(self, res):
        self.config['resolutions']['arteries'] = self.arteries_resolution

    @property
    def slice_x(self):
        return self.tab.sliceXDoublet.getValue()

    @slice_x.setter
    def slice_x(self, slc):
        self.tab.sliceXDoublet.setValue(slc)

    def handle_slice_x_changed(self, val):
        self.config['slice_x'] = self.slice_x

    @property
    def slice_y(self):
        return self.tab.sliceYDoublet.getValue()

    @slice_y.setter
    def slice_y(self, slc):
        self.tab.sliceYDoublet.setValue(slc)

    def handle_slice_y_changed(self, val):
        self.config['slice_y'] = self.slice_y

    @property
    def slice_z(self):
        return self.tab.sliceZDoublet.getValue()

    @slice_z.setter
    def slice_z(self, slc):
        self.tab.sliceZDoublet.setValue(slc)

    def handle_slice_z_changed(self, val):
        self.config['slice_z'] = self.slice_z

    @property
    def orientation(self):
        x = int(self.tab.orient_x.currentText())
        y = int(self.tab.orient_y.currentText())
        z = int(self.tab.orient_z.currentText())
        n_axes = len(set([abs(e) for e in (x, y, z)]))
        if n_axes != 3:
            raise ParamsOrientationError('Number of different axis is only {} instead of 3. '
                                         'Please amend duplicate axes'.format(n_axes))
        return x, y, z

    @orientation.setter
    def orientation(self, orientation):
        n_axes = len(set([abs(e) for e in orientation]))
        if n_axes != 3:
            raise ParamsOrientationError('Number of different axis is only {} instead of 3. '
                                         'Please amend duplicate axes'.format(n_axes))
        self.tab.orient_x.setCurrentText('{}'.format(orientation[0]))
        self.tab.orient_y.setCurrentText('{}'.format(orientation[1]))
        self.tab.orient_z.setCurrentText('{}'.format(orientation[2]))

    def handle_orientation_changed(self, val):  # WARNING: does not seem to move up the stack because of pyqtsignals
        try:
            orientation = self.orientation
        except ParamsOrientationError as err:
            print('Invalid orientation, keeping current')
            return
        self._config['orientation'] = orientation


class RigidStitchingParams(UiParameter):
    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'x_overlap': ['overlap_x'],
            'y_overlap': ['overlap_y'],
            'projection_thickness': ['project_thickness'],
            'max_shifts_x': ['max_shifts_x'],
            'max_shifts_y': ['max_shifts_y'],
            'max_shifts_z': ['max_shifts_z'],
            'background_level': ['background_level'],
            'background_pixels': ['background_pixels']
        }

    def connect(self):
        self.tab.skipRigidCheckbox.stateChanged.connect(self.handle_skip_state_changed)
        self.tab.xOverlapSinglet.valueChangedConnect(self.handle_x_overlap_value_changed)
        self.tab.yOverlapSinglet.valueChangedConnect(self.handle_y_overlap_value_changed)
        self.tab.projectionThicknessDoublet.valueChangedConnect(self.handle_projection_thickness_changed)
        self.tab.rigidMaxShiftsXDoublet.valueChangedConnect(self.handle_max_shifts_x_changed)
        self.tab.rigidMaxShiftsYDoublet.valueChangedConnect(self.handle_max_shifts_y_changed)
        self.tab.rigidMaxShiftsZDoublet.valueChangedConnect(self.handle_max_shifts_z_changed)
        self.tab.rigidBackgroundLevel.valueChanged.connect(self.handle_background_level_changed)
        self.tab.rigidBackgroundPixels.valueChanged.connect(self.handle_background_pixels_changed)

    @property
    def config(self):
        return self._config['stitching']['rigid']

    @property
    def skip(self):
        return self.is_checked(self.tab.skipRigidCheckbox)

    @skip.setter
    def skip(self, state):
        self.set_check_state(self.tab.skipRigidCheckbox, state)

    def handle_skip_state_changed(self):
        self.config['skip'] = self.skip

    @property
    def x_overlap(self):
        return self.tab.xOverlapSinglet.getValue()

    @x_overlap.setter
    def x_overlap(self, overlap):
        self.tab.xOverlapSinglet.setValue(overlap)

    def handle_x_overlap_value_changed(self, overlap):
        self.config['overlap_x'] = self.x_overlap

    @property
    def y_overlap(self):
        return self.tab.yOverlapSinglet.getValue()

    @y_overlap.setter
    def y_overlap(self, overlap):
        self.tab.yOverlapSinglet.setValue(overlap)

    def handle_y_overlap_value_changed(self, overlap):
        self.config['overlap_y'] = self.y_overlap

    @property
    def projection_thickness(self):
        val = self.tab.projectionThicknessDoublet.getValue()
        if val is not None:
            val.append(None)
        return val

    @projection_thickness.setter
    def projection_thickness(self, thickness):
        if thickness is not None:
            thickness = thickness[:2]
        self.tab.projectionThicknessDoublet.setValue(thickness)

    def handle_projection_thickness_changed(self, thickness):
        self.config['project_thickness'] = self.projection_thickness  # To get formatting

    @property
    def max_shifts_x(self):
        return self.tab.rigidMaxShiftsXDoublet.getValue()

    @max_shifts_x.setter
    def max_shifts_x(self, max_shift):
        self.tab.rigidMaxShiftsXDoublet.setValue(max_shift)

    def handle_max_shifts_x_changed(self, val):
        self.config['max_shifts_x'] = self.max_shifts_x

    @property
    def max_shifts_y(self):
        return self.tab.rigidMaxShiftsYDoublet.getValue()

    @max_shifts_y.setter
    def max_shifts_y(self, max_shift):
        self.tab.rigidMaxShiftsYDoublet.setValue(max_shift)

    def handle_max_shifts_y_changed(self, val):
        self.config['max_shifts_y'] = self.max_shifts_y

    @property
    def max_shifts_z(self):
        return self.tab.rigidMaxShiftsZDoublet.getValue()

    @max_shifts_z.setter
    def max_shifts_z(self, max_shift):
        self.tab.rigidMaxShiftsZDoublet.setValue(max_shift)

    def handle_max_shifts_z_changed(self, val):
        self.config['max_shifts_z'] = self.max_shifts_z

    @property
    def background_level(self):
        return self.tab.rigidBackgroundLevel.value()

    @background_level.setter
    def background_level(self, lvl):
        self.tab.rigidBackgroundLevel.setValue(lvl)

    def handle_background_level_changed(self, lvl):
        self.config['background_level'] = self.background_level

    @property
    def background_pixels(self):
        return self.tab.rigidBackgroundPixels.value()

    @background_pixels.setter
    def background_pixels(self, n_pix):
        self.tab.rigidBackgroundPixels.setValue(n_pix)

    def handle_background_pixels_changed(self, n_pix):
        self.config['background_pixels'] = self.background_pixels


class WobblyStitchingParams(UiParameter):

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'skip': ['skip'],
            'max_shifts_x': ['max_shifts_x'],
            'max_shifts_y': ['max_shifts_y'],
            'max_shifts_z': ['max_shifts_z'],
            'valid_range': ['valid_range'],
            'slice_range': ['slice_range'],
            'slice_pixel_size': ['slice_pixel_size']
        }

    def connect(self):
        self.tab.skipWobblyCheckBox.stateChanged.connect(self.handle_skip_changed)
        self.tab.wobblyMaxShiftsXDoublet.valueChangedConnect(self.handle_max_shifts_x_changed)
        self.tab.wobblyMaxShiftsYDoublet.valueChangedConnect(self.handle_max_shifts_y_changed)
        self.tab.wobblyMaxShiftsZDoublet.valueChangedConnect(self.handle_max_shifts_z_changed)
        self.tab.wobblyValidRangeDoublet.valueChangedConnect(self.handle_valid_range_changed)
        self.tab.wobblySliceRangeDoublet.valueChangedConnect(self.handle_slice_range_changed)
        self.tab.wobblySlicePixelSizeSinglet.valueChangedConnect(self.handle_slice_pixel_size_changed)

    @property
    def config(self):
        return self._config['stitching']['wobbly']
        
    @property
    def skip(self):
        return self.is_checked(self.tab.skipWobblyCheckBox)

    @skip.setter
    def skip(self, state):
        self.set_check_state(self.tab.skipWobblyCheckBox, state)

    def handle_skip_changed(self):
        self.config['skip'] = self.skip

    @property
    def max_shifts_x(self):
        return self.tab.wobblyMaxShiftsXDoublet.getValue()

    @max_shifts_x.setter
    def max_shifts_x(self, max_shift):
        self.tab.wobblyMaxShiftsXDoublet.setValue(max_shift)

    def handle_max_shifts_x_changed(self, max_shift):
        self.config['max_shifts_x'] = self.max_shifts_x

    @property
    def max_shifts_y(self):
        return self.tab.wobblyMaxShiftsYDoublet.getValue()

    @max_shifts_y.setter
    def max_shifts_y(self, max_shift):
        self.tab.wobblyMaxShiftsYDoublet.setValue(max_shift)

    def handle_max_shifts_y_changed(self, max_shift):
        self.config['max_shifts_y'] = self.max_shifts_y

    @property
    def max_shifts_z(self):
        return self.tab.wobblyMaxShiftsZDoublet.getValue()

    @max_shifts_z.setter
    def max_shifts_z(self, max_shift):
        self.tab.wobblyMaxShiftsZDoublet.setValue(max_shift)

    def handle_max_shifts_z_changed(self, max_shift):
        self.config['max_shifts_z'] = self.max_shifts_z

    @property
    def valid_range(self):
        rng = self.tab.wobblyValidRangeDoublet.getValue()
        # rng = minus_1_to_none(rng)
        return rng

    @valid_range.setter
    def valid_range(self, rng):
        # rng = none_to_minus_1(rng)
        self.tab.wobblyValidRangeDoublet.setValue(rng)

    def handle_valid_range_changed(self, rng):
        self.config['valid_range'] = self.valid_range

    @property
    def slice_range(self):
        rng = self.tab.wobblySliceRangeDoublet.getValue()
        # rng = minus_1_to_none(rng)
        return rng

    @slice_range.setter
    def slice_range(self, rng):
        # rng = none_to_minus_1(rng)
        self.tab.wobblySliceRangeDoublet.setValue(rng)

    def handle_slice_range_changed(self, rnd):
        self.config['slice_range'] = self.slice_range
        
    @property
    def slice_pixel_size(self):
        return self.tab.wobblySlicePixelSizeSinglet.getValue()

    @slice_pixel_size.setter
    def slice_pixel_size(self, size):
        self.tab.wobblySlicePixelSizeSinglet.setValue(size)

    def handle_slice_pixel_size_changed(self, size):
        self.config['slice_pixel_size'] = self.slice_pixel_size


class GeneralStitchingParams(UiParameter):
    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.attrs_to_invert = ['convert_output']
        self.params_dict = {
            'use_npy': ['conversion', 'use_npy'],
            'run_raw': ['stitching', 'run', 'raw'],
            'run_arteries': ['stitching', 'run', 'arteries'],
            'preview_raw': ['stitching', 'preview', 'raw'],
            'preview_arteries': ['stitching', 'preview', 'arteries'],
            'convert_output': ['stitching', 'output_conversion', 'skip'],
            'convert_raw': ['stitching', 'output_conversion', 'raw'],
            'convert_arteries': ['stitching', 'output_conversion', 'arteries'],
            'conversion_fmt': ['stitching', 'output_conversion', 'format'],
        }

    def connect(self):
        self.tab.stitchingUseNpyCheckBox.stateChanged.connect(self.handle_use_npy_changed)
        self.tab.stitchingRunRawCheckBox.stateChanged.connect(self.handle_run_raw_changed)
        self.tab.stitchingRunArteriesCheckBox.stateChanged.connect(self.handle_run_arteries_changed)
        self.tab.stitchingPreviewRawCheckBox.stateChanged.connect(self.handle_preview_raw_changed)
        self.tab.stitchingPreviewArteriesCheckBox.stateChanged.connect(self.handle_preview_arteries_changed)
        self.tab.skipOutputConversioncheckBox.stateChanged.connect(self.handle_convert_output_changed)
        self.tab.stitchingConvertRawCheckBox.stateChanged.connect(self.handle_convert_raw_changed)
        self.tab.stitchingConvertArteriesCheckBox.stateChanged.connect(self.handle_convert_arteries_changed)
        self.tab.outputConversionFormat.currentTextChanged.connect(self.handle_conversion_fmt_changed)

    @property
    def config(self):
        return self._config

    @property
    def use_npy(self):
        return self.tab.stitchingUseNpyCheckBox.isChecked()

    @use_npy.setter
    def use_npy(self, state):
        self.tab.stitchingUseNpyCheckBox.setChecked(state)

    def handle_use_npy_changed(self, state):
        self._config['conversion']['use_npy'] = self.use_npy

    @property
    def run_raw(self):
        return self.is_checked(self.tab.stitchingRunRawCheckBox)

    @run_raw.setter
    def run_raw(self, state):
        self.set_check_state(self.tab.stitchingRunRawCheckBox, state)

    def handle_run_raw_changed(self, state):
        self.config['stitching']['run']['raw'] = self.run_raw

    @property
    def run_arteries(self):
        return self.is_checked(self.tab.stitchingRunArteriesCheckBox)

    @run_arteries.setter
    def run_arteries(self, state):
        self.set_check_state(self.tab.stitchingRunArteriesCheckBox, state)

    def handle_run_arteries_changed(self, state):
        self.config['stitching']['run']['arteries'] = self.run_arteries

    @property
    def preview_raw(self):
        return self.is_checked(self.tab.stitchingPreviewRawCheckBox)

    @preview_raw.setter
    def preview_raw(self, state):
        self.set_check_state(self.tab.stitchingPreviewRawCheckBox, state)

    def handle_preview_raw_changed(self, state):
        self.config['stitching']['preview']['raw'] = self.preview_raw

    @property
    def preview_arteries(self):
        return self.is_checked(self.tab.stitchingPreviewArteriesCheckBox)

    @preview_arteries.setter
    def preview_arteries(self, state):
        self.set_check_state(self.tab.stitchingPreviewArteriesCheckBox, state)

    def handle_preview_arteries_changed(self, state):
        self.config['stitching']['preview']['arteries'] = self.preview_arteries

    @property
    def convert_output(self):
        return self.tab.skipOutputConversioncheckBox.checkState() == Qt.Unchecked  # unchecked to invert

    @convert_output.setter
    def convert_output(self, skip):
        self.set_check_state(self.tab.skipOutputConversioncheckBox, not skip)

    def handle_convert_output_changed(self, state):
        self.config['stitching']['output_conversion']['skip'] = not self.convert_output

    @property
    def convert_raw(self):
        return self.is_checked(self.tab.stitchingConvertRawCheckBox)

    @convert_raw.setter
    def convert_raw(self, state):
        self.set_check_state(self.tab.stitchingConvertRawCheckBox, state)

    def handle_convert_raw_changed(self, state):
        self.config['stitching']['output_conversion']['raw'] = self.convert_raw

    @property
    def convert_arteries(self):
        return self.is_checked(self.tab.stitchingConvertArteriesCheckBox)

    @convert_arteries.setter
    def convert_arteries(self, state):
        self.set_check_state(self.tab.stitchingConvertArteriesCheckBox, state)

    def handle_convert_arteries_changed(self, state):
        self.config['stitching']['output_conversion']['arteries'] = self.convert_arteries

    @property
    def conversion_fmt(self):
        return self.tab.outputConversionFormat.currentText()

    @conversion_fmt.setter
    def conversion_fmt(self, fmt):
        self.tab.outputConversionFormat.setCurrentText(fmt)

    def handle_conversion_fmt_changed(self, fmt):
        self.config['stitching']['output_conversion']['format'] = self.conversion_fmt


class RegistrationParams(UiParameter):
    atlas_id_changed = pyqtSignal(str)
    atlas_structure_tree_id_changed = pyqtSignal(str)

    def __init__(self, tab, src_folder=None):
        self.atlas_info = ATLAS_NAMES_MAP
        super().__init__(tab, src_folder)
        self.params_dict = {
            'skip_resampling': ['resampling', 'skip'],
            'atlas_resolution': ['resampling', 'raw_sink_resolution'],
            'atlas_id': ['atlas', 'id'],
            'atlas_folder': ['atlas', 'align_files_folder'],
            'channel_affine_file_path': ['atlas', 'align_channels_affine_file'],
            'ref_affine_file_path': ['atlas', 'align_reference_affine_file'],
            'ref_bspline_file_path': ['atlas', 'align_reference_bspline_file']
        }

    def connect(self):
        self.tab.skipRegistrationResamplingCheckBox.stateChanged.connect(self.handle_skip_resampling_changed)
        self.tab.atlasResolutionTriplet.valueChangedConnect(self.handle_atlas_resolution_changed)
        self.tab.atlasIdComboBox.currentTextChanged.connect(self.handle_atlas_id_changed)
        self.tab.structureTreeIdComboBox.currentTextChanged.connect(self.handle_structure_tree_id_changed)
        # self.tab.atlasFolderPath.textChanged.connect(self.handle_atlas_folder_changed)  # WARNING: ensure that set correctly by picking
        self.tab.channelAffineFilePath.textChanged.connect(self.handle_channel_affine_file_path_changed)
        self.tab.refAffineFilePath.textChanged.connect(self.handle_ref_affine_file_path_changed)
        self.tab.refBsplineFilePath.textChanged.connect(self.handle_ref_bspline_file_path_changed)

    @property
    def atlas_base_name(self):
        return self.atlas_info[self.atlas_id]['base_name']

    @property
    def config(self):
        return self._config['registration']

    @property
    def skip_resampling(self):  # WARNING: skip resampling not registration altogether
        return self.is_checked(self.tab.skipRegistrationResamplingCheckBox)

    @skip_resampling.setter
    def skip_resampling(self, state):
        self.set_check_state(self.tab.skipRegistrationResamplingCheckBox, state)

    def handle_skip_resampling_changed(self, state):
        self.config['resampling']['skip'] = self.skip_resampling
        
    @property
    def atlas_resolution(self):
        return self.tab.atlasResolutionTriplet.getValue()

    @atlas_resolution.setter
    def atlas_resolution(self, res):
        self.tab.atlasResolutionTriplet.setValue(res)

    def handle_atlas_resolution_changed(self, state):
        self.config['resampling']['raw_sink_resolution'] = self.atlas_resolution
        self.config['resampling']['autofluo_sink_resolution'] = self.atlas_resolution

    @property
    def atlas_id(self):
        return self.tab.atlasIdComboBox.currentText()

    @atlas_id.setter
    def atlas_id(self, value):
        self.tab.atlasIdComboBox.setCurrentText(value)

    def handle_atlas_id_changed(self):
        self.config['atlas']['id'] = self.atlas_id
        self.atlas_resolution = self.atlas_info[self.atlas_id]['resolution']
        self.ui_to_cfg()
        self.atlas_id_changed.emit(self.atlas_base_name)

    @property
    def structure_tree_id(self):
        return self.tab.structureTreeIdComboBox.currentText()

    @structure_tree_id.setter
    def structure_tree_id(self, value):
        self.tab.structureTreeIdComboBox.setCurrentText(value)

    def handle_structure_tree_id_changed(self):
        self.config['atlas']['structure_tree_id'] = self.structure_tree_id
        self.ui_to_cfg()   # TODO: check if required
        self.atlas_structure_tree_id_changed.emit(self.structure_tree_id)

    @property
    def atlas_folder(self):
        return self.tab.atlasFolderPath.text()

    @atlas_folder.setter
    def atlas_folder(self, folder):
        self.tab.atlasFolderPath.setText(folder)

    def handle_atlas_folder_changed(self, folder):
        self.config['atlas']['align_files_folder'] = self.atlas_folder

    @property
    def channel_affine_file_path(self):
        return self.tab.channelAffineFilePath.text()

    @channel_affine_file_path.setter
    def channel_affine_file_path(self, f_path):
        self.tab.channelAffineFilePath.setText(f_path)

    def handle_channel_affine_file_path_changed(self, f_path):
        self.config['atlas']['align_channels_affine_file'] = self.channel_affine_file_path

    @property
    def ref_affine_file_path(self):
        return self.tab.refAffineFilePath.text()

    @ref_affine_file_path.setter
    def ref_affine_file_path(self, f_path):
        self.tab.refAffineFilePath.setText(f_path)

    def handle_ref_affine_file_path_changed(self, f_path):
        self.config['atlas']['align_reference_affine_file'] = self.ref_affine_file_path

    @property
    def ref_bspline_file_path(self):
        return self.tab.refBsplineFilePath.text()

    @ref_bspline_file_path.setter
    def ref_bspline_file_path(self, f_path):
        self.tab.refBsplineFilePath.setText(f_path)

    def handle_ref_bspline_file_path_changed(self, f_path):
        self.config['atlas']['align_reference_bspline_file'] = self.ref_bspline_file_path


class CellMapParams(UiParameter):
    def __init__(self, tab, sample_params=None, preprocessing_params=None, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'background_correction_diameter': ['detection', 'background_correction', 'diameter'],
            'detection_threshold': ['detection', 'shape_detection', 'threshold'],
            'cell_filter_size': ['cell_filtration', 'thresholds', 'size'],
            'voxelization_radii': ['voxelization', 'radii'],
            'plot_when_finished': ['run', 'plot_when_finished'],
            'crop_x_min': ['detection', 'test_set_slicing', 'dim_0', 0],
            'crop_x_max': ['detection', 'test_set_slicing', 'dim_0', 1],
            'crop_y_min': ['detection', 'test_set_slicing', 'dim_1', 0],
            'crop_y_max': ['detection', 'test_set_slicing', 'dim_1', 1],
            'crop_z_min': ['detection', 'test_set_slicing', 'dim_2', 0],
            'crop_z_max': ['detection', 'test_set_slicing', 'dim_2', 1],
        }
        self.sample_params = sample_params
        self.preprocessing_params = preprocessing_params

    def connect(self):
        self.tab.runCellMapPlotCheckBox.stateChanged.connect(self.handle_plot_when_finished)
        self.tab.backgroundCorrectionDiameter.valueChanged.connect(self.handle_background_correction_diameter_changed)
        self.tab.detectionThreshold.valueChanged.connect(self.handle_detection_threshold_changed)
        self.tab.cellFilterThresholdSizeDoublet.valueChangedConnect(self.handle_filter_size_changed)
        self.tab.voxelizationRadiusTriplet.valueChangedConnect(self.handle_voxelization_radii_changed)
        self.tab.cellDetectionPlotCheckBox.stateChanged.connect(self.handle_plot_detected_cells_changed)
        self.tab.detectionSubsetXRangeMin.valueChanged.connect(self.handle_x_val_min_change)
        self.tab.detectionSubsetXRangeMax.valueChanged.connect(self.handle_x_val_max_change)
        self.tab.detectionSubsetYRangeMin.valueChanged.connect(self.handle_y_val_min_change)
        self.tab.detectionSubsetYRangeMax.valueChanged.connect(self.handle_y_val_max_change)
        self.tab.detectionSubsetZRangeMin.valueChanged.connect(self.handle_z_val_min_change)
        self.tab.detectionSubsetZRangeMax.valueChanged.connect(self.handle_z_val_max_change)

    @property
    def config(self):
        return self._config

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()

    @property
    def ratios(self):
        raw_res = np.array(self.sample_params.raw_resolution)
        atlas_res = np.array(self.preprocessing_params.registration.atlas_resolution)
        ratios = raw_res / atlas_res  # to original
        return ratios

    # def _scale_crop_values(self, ratios):
    #     crop_values = []
    #     ui_crops = self.crop_x_min, self.crop_x_max, self.crop_y_min, self.crop_y_max, self.crop_z_min, self.crop_z_max
    #     for ratio, val in zip(np.repeat(ratios, 2), ui_crops):
    #         crop_values.append(round(ratio * val))
    #     return crop_values

    @property
    def plot_when_finished(self):
        return self.is_checked(self.tab.runCellMapPlotCheckBox)

    @plot_when_finished.setter
    def plot_when_finished(self, state):
        self.set_check_state(self.tab.runCellMapPlotCheckBox, state)

    def handle_plot_when_finished(self, state):
        self.config['run']['plot_when_finished'] = self.plot_when_finished

    @property
    def background_correction_diameter(self):
        return [self.tab.backgroundCorrectionDiameter.value()] * 2

    @background_correction_diameter.setter
    def background_correction_diameter(self, shape):
        if isinstance(shape, (list, tuple)):
            shape = shape[0]
        self.tab.backgroundCorrectionDiameter.setValue(shape)

    def handle_background_correction_diameter_changed(self, val):
        self.config['detection']['background_correction']['diameter'] = self.background_correction_diameter

    @property
    def detection_threshold(self):
        return self.tab.detectionThreshold.value()

    @detection_threshold.setter
    def detection_threshold(self, thrsh):
        self.tab.detectionThreshold.setValue(thrsh)

    def handle_detection_threshold_changed(self, thrsh):
        self.config['detection']['shape_detection']['threshold'] = self.detection_threshold
        
    @property
    def cell_filter_size(self):
        return self.tab.cellFilterThresholdSizeDoublet.getValue()

    @cell_filter_size.setter
    def cell_filter_size(self, size):
        self.tab.cellFilterThresholdSizeDoublet.setValue(size)

    def handle_filter_size_changed(self, size):
        self.config['cell_filtration']['thresholds']['size'] = self.cell_filter_size

    @property
    def voxelization_radii(self):
        return self.tab.voxelizationRadiusTriplet.getValue()

    @voxelization_radii.setter
    def voxelization_radii(self, radii):
        self.tab.voxelizationRadiusTriplet.setValue(radii)

    def handle_voxelization_radii_changed(self, radii):
        self.config['voxelization']['radii'] = self.voxelization_radii

    @property
    def crop_x_min(self):
        return self.tab.detectionSubsetXRangeMin.value()

    @crop_x_min.setter
    def crop_x_min(self, val):
        self.tab.detectionSubsetXRangeMin.setValue(val)

    @property
    def crop_x_max(self):  # TODO: if 99.9 % source put to 100% (None)
        return self.tab.detectionSubsetXRangeMax.value()

    @crop_x_max.setter
    def crop_x_max(self, val):
        self.tab.detectionSubsetXRangeMax.setValue(val)

    def handle_x_val_min_change(self):
        self.config['detection']['test_set_slicing']['dim_0'][0] = self.crop_x_min

    def handle_x_val_max_change(self):
        self.config['detection']['test_set_slicing']['dim_0'][1] = self.crop_x_max

    def scale_x(self, val):
        return round(val * self.ratios[0])

    @property
    def crop_y_min(self):
        return self.tab.detectionSubsetYRangeMin.value()

    @crop_y_min.setter
    def crop_y_min(self, val):
        self.tab.detectionSubsetYRangeMin.setValue(val)

    @property
    def crop_y_max(self):  # TODO: if 99.9 % source put to 100% (None)
        return self.tab.detectionSubsetYRangeMax.value()

    @crop_y_max.setter
    def crop_y_max(self, val):
        self.tab.detectionSubsetYRangeMax.setValue(val)

    def handle_y_val_min_change(self):
        self.config['detection']['test_set_slicing']['dim_1'][0] = self.crop_y_min

    def handle_y_val_max_change(self):
        self.config['detection']['test_set_slicing']['dim_1'][1] = self.crop_y_max

    def scale_y(self, val):
        return round(val * self.ratios[1])

    @property
    def crop_z_min(self):
        return self.tab.detectionSubsetZRangeMin.value()

    @crop_z_min.setter
    def crop_z_min(self, val):
        self.tab.detectionSubsetZRangeMin.setValue(val)

    @property
    def crop_z_max(self):  # TODO: if 99.9 % source put to 100% (None)
        return self.tab.detectionSubsetZRangeMax.value()

    @crop_z_max.setter
    def crop_z_max(self, val):
        self.tab.detectionSubsetZRangeMax.setValue(val)

    def handle_z_val_min_change(self):
        self.config['detection']['test_set_slicing']['dim_2'][0] = self.crop_z_min

    def handle_z_val_max_change(self):
        self.config['detection']['test_set_slicing']['dim_2'][1] = self.crop_z_max

    def scale_z(self, val):
        return round(val * self.ratios[2])

    @property
    def slicing(self):
        return (slice(self.crop_x_min, self.crop_x_max),
                slice(self.crop_y_min, self.crop_y_max),
                slice(self.crop_z_min, self.crop_z_max))

    @property
    def plot_detected_cells(self):
        return self.is_checked(self.tab.cellDetectionPlotCheckBox)

    @plot_detected_cells.setter
    def plot_detected_cells(self, state):
        self.set_check_state(self.tab.cellDetectionPlotCheckBox, state)

    def handle_plot_detected_cells_changed(self, state):
        self.config['detection']['plot_cells'] = self.plot_detected_cells


class VesselParams(UiParameterCollection):
    def __init__(self, tab, sample_params, preprocessing_params, src_folder=None):
        super().__init__(tab, src_folder)
        # self.sample_params = sample_params  # TODO: check if required
        # self.preprocessing_params = preprocessing_params  # TODO: check if required
        self.binarization_params = VesselBinarizationParams(tab, src_folder)
        self.graph_params = VesselGraphParams(tab,  sample_params, preprocessing_params, src_folder)
        self.visualization_params = VesselVisualizationParams(tab, src_folder)

    @property
    def params(self):
        return self.binarization_params, self.graph_params


class VesselBinarizationParams(UiParameter):

    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.attrs_to_invert = ['run_arteries_binarization']
        self.params_dict = {
            'raw_binarization_clip_range': ['binarization', 'raw', 'clip_range'],
            'raw_binarization_threshold': ['binarization', 'raw', 'threshold'],
            'post_process_raw': ['binarization', 'raw', 'post_process'],
            'run_arteries_binarization': ['binarization', 'arteries', 'skip'],
            'arteries_binarization_clip_range': ['binarization', 'arteries', 'clip_range'],
            'arteries_binarization_threshold': ['binarization', 'arteries', 'threshold'],
            'post_process_arteries': ['binarization', 'arteries', 'post_process'],
            'fill_main_channel': ['vessel_filling', 'main'],
            'fill_secondary_channel': ['vessel_filling', 'secondary']
        }

    def connect(self):
        self.tab.rawBinarizationClipRangeDoublet.valueChangedConnect(self.handle_raw_binarization_clip_range_changed)
        self.tab.rawBinarizationThresholdSpinBox.valueChanged.connect(self.handle_raw_binarization_treshold_changed)
        self.tab.rawBinarizationPostprocessingCheckBox.stateChanged.connect(self.handle_post_process_raw_changed)

        self.tab.runArteriesBinarizationCheckBox.stateChanged.connect(self.handle_run_arteries_binarization_changed)
        self.tab.arteriesBinarizationClipRangeDoublet.valueChangedConnect(
            self.handle_arteries_binarization_clip_range_changed)
        self.tab.arteriesBinarizationThresholdSpinBox.valueChanged.connect(
            self.handle_arteries_binarization_threshold_changed)
        self.tab.arteriesBinarizationPostprocessingCheckBox.stateChanged.connect(
            self.handle_post_process_arteries_changed)

        self.tab.vesselFillingMainChannelCheckBox.stateChanged.connect(self.handle_fill_main_channel_changed)
        self.tab.vesselFillingSecondaryChannelCheckBox.stateChanged.connect(self.handle_fill_secondary_channel_changed)

    @property
    def config(self):
        return self._config['binarization']

    @property
    def raw_binarization_clip_range(self):
        return self.tab.rawBinarizationClipRangeDoublet.getValue()

    @raw_binarization_clip_range.setter
    def raw_binarization_clip_range(self, value):
        self.tab.rawBinarizationClipRangeDoublet.setValue(value)

    def handle_raw_binarization_clip_range_changed(self, value):
        self.config['binarization']['raw']['clip_range'] = self.raw_binarization_clip_range

    @property
    def raw_binarization_threshold(self):
        return self.sanitize_neg_one(self.tab.rawBinarizationThresholdSpinBox.value())

    @raw_binarization_threshold.setter
    def raw_binarization_threshold(self, value):
        self.tab.rawBinarizationThresholdSpinBox.setValue(self.sanitize_nones(value))

    def handle_raw_binarization_treshold_changed(self, value):
        self.config['binarization']['raw']['threshold'] = self.raw_binarization_threshold

    @property
    def post_process_raw(self):
        return self.is_checked(self.tab.rawBinarizationPostprocessingCheckBox)

    @post_process_raw.setter
    def post_process_raw(self, state):
        self.set_check_state(self.tab.rawBinarizationPostprocessingCheckBox, state)

    def handle_post_process_raw_changed(self, state):
        self.config['binarization']['raw']['post_process'] = self.post_process_raw

    @property
    def run_arteries_binarization(self):
        return self.is_checked(self.tab.runArteriesBinarizationCheckBox)

    @run_arteries_binarization.setter
    def run_arteries_binarization(self, state):
        self.set_check_state(self.tab.runArteriesBinarizationCheckBox, state)

    def handle_run_arteries_binarization_changed(self, state):
        self.config['binarization']['arteries']['skip'] = not self.run_arteries_binarization

    @property
    def arteries_binarization_clip_range(self):
        return self.tab.arteriesBinarizationClipRangeDoublet.getValue()

    @arteries_binarization_clip_range.setter
    def arteries_binarization_clip_range(self, value):
        self.tab.arteriesBinarizationClipRangeDoublet.setValue(value)

    def handle_arteries_binarization_clip_range_changed(self, value):
        self.config['binarization']['arteries']['clip_range'] = self.arteries_binarization_clip_range

    @property
    def arteries_binarization_threshold(self):
        return self.sanitize_neg_one(self.tab.arteriesBinarizationThresholdSpinBox.value())

    @arteries_binarization_threshold.setter
    def arteries_binarization_threshold(self, value):
        self.tab.arteriesBinarizationThresholdSpinBox.setValue(self.sanitize_nones(value))

    def handle_arteries_binarization_threshold_changed(self, value):
        self.config['binarization']['arteries']['threshold'] = self.arteries_binarization_threshold

    @property
    def post_process_arteries(self):
        return self.is_checked(self.tab.arteriesBinarizationPostprocessingCheckBox)

    @post_process_arteries.setter
    def post_process_arteries(self, state):
        self.set_check_state(self.tab.arteriesBinarizationPostprocessingCheckBox, state)

    def handle_post_process_arteries_changed(self, state):
        self.config['binarization']['arteries']['post_process'] = self.post_process_arteries

    @property
    def fill_main_channel(self):
        return self.is_checked(self.tab.vesselFillingMainChannelCheckBox)

    @fill_main_channel.setter
    def fill_main_channel(self, state):
        self.set_check_state(self.tab.vesselFillingMainChannelCheckBox, state)

    def handle_fill_main_channel_changed(self, state):
        self.config['vessel_filling']['main'] = self.fill_main_channel

    @property
    def fill_secondary_channel(self):
        return self.is_checked(self.tab.vesselFillingSecondaryChannelCheckBox)

    @fill_secondary_channel.setter
    def fill_secondary_channel(self, state):
        self.set_check_state(self.tab.vesselFillingSecondaryChannelCheckBox, state)

    def handle_fill_secondary_channel_changed(self, state):
        self.config['vessel_filling']['secondary'] = self.fill_secondary_channel


class VesselGraphParams(UiParameter):
    crop_ranges_changed = pyqtSignal()
    def __init__(self, tab, sample_params=None, preprocessing_params=None, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'crop_x_min': ['graph_construction', 'slicing', 'dim_0', 0],
            'crop_x_max': ['graph_construction', 'slicing', 'dim_0', 1],
            'crop_y_min': ['graph_construction', 'slicing', 'dim_1', 0],
            'crop_y_max': ['graph_construction', 'slicing', 'dim_1', 1],
            'crop_z_min': ['graph_construction', 'slicing', 'dim_2', 0],
            'crop_z_max': ['graph_construction', 'slicing', 'dim_2', 1],
            'vein_intensity_range_on_arteries_channel': ['vessel_type_postprocessing' 'pre_filtering', 'vein_intensity_range_on_arteries_ch'],
            'restrictive_min_vein_radius': ['vessel_type_postprocessing', 'pre_filtering', 'restrictive_vein_radius'],
            'permissive_min_vein_radius': ['vessel_type_postprocessing', 'pre_filtering', 'permissive_vein_radius'],
            'final_min_vein_radius': ['vessel_type_postprocessing', 'pre_filtering', 'final_vein_radius'],
            'arteries_min_radius': ['vessel_type_postprocessing', 'pre_filtering', 'arteries_min_radius'],
            'max_arteries_tracing_iterations': ['vessel_type_postprocessing', 'tracing', 'max_arteries_iterations'],
            'max_veins_tracing_iterations': ['vessel_type_postprocessing', 'tracing', 'max_veins_iterations'],
            'min_artery_size': ['vessel_type_postprocessing', 'capillaries_removal', 'min_artery_size'],
            'min_vein_size': ['vessel_type_postprocessing', 'capillaries_removal', 'min_vein_size'],
        }
        self.sample_params = sample_params
        self.preprocessing_params = preprocessing_params

    def connect(self):
        self.tab.veinIntensityRangeOnArteriesChannelDoublet.valueChangedConnect(
            self.handle_vein_intensity_range_on_arteries_channel_changed)
        self.tab.restrictiveMinVeinRadiusSpinBox.valueChanged.connect(
            self.handle_restrictive_min_vein_radius_changed)
        self.tab.permissiveMinVeinRadiusSpinBox.valueChanged.connect(self.handle_permissive_min_vein_radius_changed)
        self.tab.finalMinVeinRadiusSpinBox.valueChanged.connect(self.handle_final_min_vein_radius_changed)

        self.tab.arteriesMinRadiusSpinBox.valueChanged.connect(self.handle_arteries_min_radius_changed)
        self.tab.maxArteriesTracingIterationsSpinBox.valueChanged.connect(
            self.handle_max_arteries_tracing_iterations_changed)
        self.tab.maxVeinsTracingIterationsSpinBox.valueChanged.connect(self.handle_max_veins_tracing_iterations_changed)
        self.tab.minArterySizeSpinBox.valueChanged.connect(self.handle_min_artery_size_changed)
        self.tab.minVeinSizeSpinBox.valueChanged.connect(self.handle_min_vein_size_changed)

        self.tab.graphConstructionSlicerXRangeMin.valueChanged.connect(self.handle_x_val_min_change)  # REFACTOR: this feels messy having the repeats
        self.tab.graphConstructionSlicerXRangeMax.valueChanged.connect(self.handle_x_val_max_change)
        self.tab.graphConstructionSlicerYRangeMin.valueChanged.connect(self.handle_y_val_min_change)
        self.tab.graphConstructionSlicerYRangeMax.valueChanged.connect(self.handle_y_val_max_change)
        self.tab.graphConstructionSlicerZRangeMin.valueChanged.connect(self.handle_z_val_min_change)
        self.tab.graphConstructionSlicerZRangeMax.valueChanged.connect(self.handle_z_val_max_change)

        self.tab.vesselProcessingSlicerXRangeMin.valueChanged.connect(self.handle_x_val_min_change)
        self.tab.vesselProcessingSlicerXRangeMax.valueChanged.connect(self.handle_x_val_max_change)
        self.tab.vesselProcessingSlicerYRangeMin.valueChanged.connect(self.handle_y_val_min_change)
        self.tab.vesselProcessingSlicerYRangeMax.valueChanged.connect(self.handle_y_val_max_change)
        self.tab.vesselProcessingSlicerZRangeMin.valueChanged.connect(self.handle_z_val_min_change)
        self.tab.vesselProcessingSlicerZRangeMax.valueChanged.connect(self.handle_z_val_max_change)

    @property
    def config(self):
        return self._config

    @property
    def ratios(self):
        raw_res = np.array(self.sample_params.raw_resolution)
        atlas_res = np.array(self.preprocessing_params.registration.atlas_resolution)
        ratios = raw_res / atlas_res  # to original
        return ratios

    @property
    def crop_x_min(self):
        return self.tab.graphConstructionSlicerXRangeMin.value()

    @crop_x_min.setter
    def crop_x_min(self, val):
        self.tab.graphConstructionSlicerXRangeMin.setValue(val)
        self.tab.vesselProcessingSlicerXRangeMin.setValue(val)

    @property
    def crop_x_max(self):  # TODO: if 99.9 % source put to 100% (None)
        return self.tab.graphConstructionSlicerXRangeMax.value()

    @crop_x_max.setter
    def crop_x_max(self, val):
        self.tab.graphConstructionSlicerXRangeMax.setValue(val)
        self.tab.vesselProcessingSlicerXRangeMax.setValue(val)

    def handle_x_val_min_change(self):
        self._config['graph_construction']['slicing']['dim_0'][0] = self.crop_x_min
        self.crop_ranges_changed.emit()

    def handle_x_val_max_change(self):
        self._config['graph_construction']['slicing']['dim_0'][1] = self.crop_x_max
        self.crop_ranges_changed.emit()

    def scale_x(self, val):
        return round(val * self.ratios[0])

    @property
    def crop_y_min(self):
        return self.tab.graphConstructionSlicerYRangeMin.value()

    @crop_y_min.setter
    def crop_y_min(self, val):
        self.tab.graphConstructionSlicerYRangeMin.setValue(val)
        self.tab.vesselProcessingSlicerYRangeMin.setValue(val)

    @property
    def crop_y_max(self):  # TODO: if 99.9 % source put to 100% (None)
        return self.tab.graphConstructionSlicerYRangeMax.value()

    @crop_y_max.setter
    def crop_y_max(self, val):
        self.tab.graphConstructionSlicerYRangeMax.setValue(val)
        self.tab.vesselProcessingSlicerYRangeMax.setValue(val)

    def handle_y_val_min_change(self):
        self._config['graph_construction']['slicing']['dim_1'][0] = self.crop_y_min
        self.crop_ranges_changed.emit()

    def handle_y_val_max_change(self):
        self._config['graph_construction']['slicing']['dim_1'][1] = self.crop_y_max
        self.crop_ranges_changed.emit()

    def scale_y(self, val):
        return round(val * self.ratios[1])

    @property
    def crop_z_min(self):
        return self.tab.graphConstructionSlicerZRangeMin.value()

    @crop_z_min.setter
    def crop_z_min(self, val):
        self.tab.graphConstructionSlicerZRangeMin.setValue(val)
        self.tab.vesselProcessingSlicerZRangeMin.setValue(val)

    @property
    def crop_z_max(self):  # TODO: if 99.9 % source put to 100% (None)
        return self.tab.graphConstructionSlicerZRangeMax.value()

    @crop_z_max.setter
    def crop_z_max(self, val):
        self.tab.graphConstructionSlicerZRangeMax.setValue(val)
        self.tab.vesselProcessingSlicerZRangeMax.setValue(val)

    def handle_z_val_min_change(self):
        self._config['graph_construction']['slicing']['dim_2'][0] = self.crop_z_min
        self.crop_ranges_changed.emit()

    def handle_z_val_max_change(self):
        self._config['graph_construction']['slicing']['dim_2'][1] = self.crop_z_max
        self.crop_ranges_changed.emit()

    def scale_z(self, val):
        return round(val * self.ratios[2])

    @property
    def slicing(self):
        return (slice(self.crop_x_min, self.crop_x_max),
                slice(self.crop_y_min, self.crop_y_max),
                slice(self.crop_z_min, self.crop_z_max))
    
    @property
    def vein_intensity_range_on_arteries_channel(self):
        return self.tab.veinIntensityRangeOnArteriesChannelDoublet.getValue()
    
    @vein_intensity_range_on_arteries_channel.setter
    def vein_intensity_range_on_arteries_channel(self, value):
        self.tab.veinIntensityRangeOnArteriesChannelDoublet.setValue(value)

    def handle_vein_intensity_range_on_arteries_channel_changed(self, _):
        self.config['vessel_type_postprocessing']['pre_filtering']['vein_intensity_range_on_arteries_ch'] = \
            self.vein_intensity_range_on_arteries_channel
        
    @property
    def restrictive_min_vein_radius(self):
        return self.tab.restrictiveMinVeinRadiusSpinBox.value()
    
    @restrictive_min_vein_radius.setter
    def restrictive_min_vein_radius(self, value):
        self.tab.restrictiveMinVeinRadiusSpinBox.setValue(value)

    def handle_restrictive_min_vein_radius_changed(self, _):
        self.config['vessel_type_postprocessing']['pre_filtering']['restrictive_vein_radius'] = \
            self.restrictive_min_vein_radius

    @property
    def permissive_min_vein_radius(self):
        return self.tab.permissiveMinVeinRadiusSpinBox.value()

    @permissive_min_vein_radius.setter
    def permissive_min_vein_radius(self, value):
        self.tab.permissiveMinVeinRadiusSpinBox.setValue(value)

    def handle_permissive_min_vein_radius_changed(self, _):
        self.config['vessel_type_postprocessing']['pre_filtering']['permissive_vein_radius'] = \
            self.permissive_min_vein_radius
        
    @property
    def final_min_vein_radius(self):
        return self.tab.finalMinVeinRadiusSpinBox.value()

    @final_min_vein_radius.setter
    def final_min_vein_radius(self, value):
        self.tab.finalMinVeinRadiusSpinBox.setValue(value)

    def handle_final_min_vein_radius_changed(self, _):
        self.config['vessel_type_postprocessing']['pre_filtering']['final_vein_radius'] = self.final_min_vein_radius

    @property
    def arteries_min_radius(self):
        return self.tab.arteriesMinRadiusSpinBox.value()
    
    @arteries_min_radius.setter
    def arteries_min_radius(self, value):
        self.tab.arteriesMinRadiusSpinBox.setValue(value)

    def handle_arteries_min_radius_changed(self, _):
        self.config['vessel_type_postprocessing']['pre_filtering']['arteries_min_radius'] = self.arteries_min_radius

    @property
    def max_arteries_tracing_iterations(self):
        return self.tab.maxArteriesTracingIterationsSpinBox.value()

    @max_arteries_tracing_iterations.setter
    def max_arteries_tracing_iterations(self, value):
        self.tab.maxArteriesTracingIterationsSpinBox.setValue(value)

    def handle_max_arteries_tracing_iterations_changed(self, _):
        self.config['vessel_type_postprocessing']['tracing']['max_arteries_iterations'] = \
            self.max_arteries_tracing_iterations

    @property
    def max_veins_tracing_iterations(self):
        return self.tab.maxVeinsTracingIterationsSpinBox.value()

    @max_veins_tracing_iterations.setter
    def max_veins_tracing_iterations(self, value):
        self.tab.maxVeinsTracingIterationsSpinBox.setValue(value)

    def handle_max_veins_tracing_iterations_changed(self, _):
        self.config['vessel_type_postprocessing']['tracing']['max_veins_iterations'] = self.max_veins_tracing_iterations
        
    @property
    def min_artery_size(self):
        return self.tab.minArterySizeSpinBox.value()
    
    @min_artery_size.setter
    def min_artery_size(self, value):
        self.tab.minArterySizeSpinBox.setValue(value)

    def handle_min_artery_size_changed(self, _):
        self.config['vessel_type_postprocessing']['capillaries_removal']['min_artery_size'] = self.min_artery_size

    @property
    def min_vein_size(self):
        return self.tab.minVeinSizeSpinBox.value()

    @min_vein_size.setter
    def min_vein_size(self, value):
        self.tab.minVeinSizeSpinBox.setValue(value)

    def handle_min_vein_size_changed(self, _):
        self.config['vessel_type_postprocessing']['capillaries_removal']['min_vein_size'] = self.min_vein_size


class VesselVisualizationParams(UiParameter):
    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {'voxelization_size': ['voxelization', 'size']}

    def connect(self):
        self.tab.vasculatureVoxelizationRadiusTriplet.valueChangedConnect(self.handle_voxelization_size_changed)

    @property
    def config(self):
        return self._config['visualization']

    @property
    def voxelization_size(self):
        return self.tab.vasculatureVoxelizationRadiusTriplet.getValue()

    @voxelization_size.setter
    def voxelization_size(self, value):
        self.tab.vasculatureVoxelizationRadiusTriplet.setValue(value)

    def handle_voxelization_size_changed(self, _):
        self.config['voxelization']['size'] = self.voxelization_size


class PreferencesParams(UiParameter):
    def __init__(self, tab, src_folder=None):
        super().__init__(tab, src_folder)
        self.params_dict = {
            'verbosity': ['verbosity'],
            'n_processes_file_conv': ['n_processes_file_conv'],
            'n_processes_stitching': ['n_processes_stitching'],
            'n_processes_cell_detection': ['n_processes_cell_detection'],
            'chunk_size_min': ['detection_chunk_size_min'],
            'chunk_size_max': ['detection_chunk_size_max'],
            'chunk_size_overlap': ['detection_chunk_overlap'],
            'start_folder': ['start_folder'],
            'start_full_screen': ['start_full_screen'],
            'lut': ['default_lut'],
            'font_size': ['font_size'],
            'pattern_finder_min_n_files': ['pattern_finder_min_n_files'],
            'three_d_plot_bg': ['three_d_plot_bg']
        }

    @property
    def config(self):
        return self._config

    def _ui_to_cfg(self):  # TODO: check if live update (i.e. connected handlers) or only on save
        cfg = self._config
        cfg['verbosity'] = self.verbosity
        cfg['n_processes_file_conv'] = self.n_processes_file_conv
        cfg['n_processes_stitching'] = self.n_processes_stitching
        cfg['n_processes_cell_detection'] = self.n_processes_cell_detection
        cfg['detection_chunk_size_min'] = self.chunk_size_min
        cfg['detection_chunk_size_max'] = self.chunk_size_max
        cfg['detection_chunk_overlap'] = self.chunk_size_overlap
        cfg['start_folder'] = self.start_folder
        cfg['start_full_screen'] = self.start_full_screen
        cfg['default_lut'] = self.lut
        cfg['font_size'] = self.font_size
        cfg['pattern_finder_min_n_files'] = self.pattern_finder_min_n_files
        cfg['three_d_plot_bg'] = self.three_d_plot_bg

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()

    @property
    def three_d_plot_bg(self):
        return self.tab.threeDPlotsBackgroundComboBox.currentText().lower()

    @three_d_plot_bg.setter
    def three_d_plot_bg(self, value):
        self.tab.threeDPlotsBackgroundComboBox.setCurrentText(value)

    @property
    def start_folder(self):
        return self.tab.startFolderLineEdit.text()

    @start_folder.setter
    def start_folder(self, dir_path):
        self.tab.startFolderLineEdit.setText(dir_path)

    @property
    def start_full_screen(self):
        return self.is_checked(self.tab.startFullScreenCheckBox)

    @start_full_screen.setter
    def start_full_screen(self, state):
        self.set_check_state(self.tab.startFullScreenCheckBox, state)

    @property
    def verbosity(self):
        return self.tab.verbosityComboBox.currentText().lower()

    @verbosity.setter
    def verbosity(self, lvl):
        self.tab.verbosityComboBox.setCurrentText(lvl.capitalize())

    @property
    def n_processes_file_conv(self):
        return self.sanitize_neg_one(self.tab.nProcessesFileConversionSpinBox.value())

    @n_processes_file_conv.setter
    def n_processes_file_conv(self, n_procs):
        self.tab.nProcessesFileConversionSpinBox.setValue(self.sanitize_nones(n_procs))

    @property
    def n_processes_stitching(self):
        return self.sanitize_neg_one(self.tab.nProcessesStitchingSpinBox.value())

    @n_processes_stitching.setter
    def n_processes_stitching(self, value):
        self.tab.nProcessesStitchingSpinBox.setValue(self.sanitize_nones(value))

    @property
    def n_processes_cell_detection(self):
        return self.sanitize_neg_one(self.tab.nProcessesCellDetectionSpinBox.value())

    @n_processes_cell_detection.setter
    def n_processes_cell_detection(self, n_procs):
        self.tab.nProcessesCellDetectionSpinBox.setValue(self.sanitize_nones(n_procs))

    @property
    def chunk_size_min(self):
        return self.tab.chunkSizeMinSpinBox.value()

    @chunk_size_min.setter
    def chunk_size_min(self, size):
        self.tab.chunkSizeMinSpinBox.setValue(size)

    @property
    def chunk_size_max(self):
        return self.tab.chunkSizeMaxSpinBox.value()

    @chunk_size_max.setter
    def chunk_size_max(self, size):
        self.tab.chunkSizeMaxSpinBox.setValue(size)

    @property
    def chunk_size_overlap(self):
        return self.tab.chunkSizeOverlapSpinBox.value()

    @chunk_size_overlap.setter
    def chunk_size_overlap(self, size):
        self.tab.chunkSizeOverlapSpinBox.setValue(size)

    @property
    def lut(self):
        return self.tab.lutComboBox.currentText().lower()

    @lut.setter
    def lut(self, lut_name):
        self.tab.lutComboBox.setCurrentText(lut_name)

    @property
    def font_size(self):
        return self.tab.fontSizeSpinBox.value()

    @font_size.setter
    def font_size(self, value):
        self.tab.fontSizeSpinBox.setValue(value)

    @property
    def font_family(self):
        return self.tab.fontComboBox.currentFont().family()

    # @font_family.setter
    # def font_family(self, font):
    #     self.tab.fontComboBox.setCurrentFont(font)

    @property
    def pattern_finder_min_n_files(self):
        return self.tab.patternFinderMiNFilesSpinBox.value()

    @pattern_finder_min_n_files.setter
    def pattern_finder_min_n_files(self, n):
        self.tab.patternFinderMiNFilesSpinBox.setValue(n)


class BatchParams(UiParameter):

    def __init__(self, tab, src_folder=None, preferences=None):
        super().__init__(tab, src_folder)
        self.group_concatenator = ' vs '
        self.preferences = preferences
        self.tab.sampleFoldersToolBox = QToolBox(parent=self.tab)
        self.tab.sampleFoldersPageLayout.addWidget(self.tab.sampleFoldersToolBox, 3, 0)

        self.comparison_checkboxes = []

    @property
    def config(self):
        return self._config

    def _ui_to_cfg(self):
        self.config['paths']['results_folder'] = self.results_folder
        self.config['groups'] = self.groups
        self.config['comparisons'] = {letter: pair for letter, pair in zip(string.ascii_lowercase,
                                                                           self.selected_comparisons)}

    def cfg_to_ui(self):
        self.reload()
        self.results_folder = self.config['paths']['results_folder']
        for i in range(len(self.config['groups'].keys())):
            self.add_group()
        self.group_names = self.config['groups'].keys()  # FIXME: check that ordered
        for i, paths in enumerate(self.config['groups'].values()):
            self.set_paths(i+1, paths)
        self.update_comparisons()
        for chk_bx in self.comparison_checkboxes:
            if chk_bx.text().split(self.group_concatenator) in self.config['comparisons'].values():
                self.set_check_state(chk_bx, True)

    def connect(self):
        self.tab.addGroupPushButton.clicked.connect(self.add_group)
        self.tab.removeGroupPushButton.clicked.connect(self.remove_group)
        self.tab.resultsFolderLineEdit.textChanged.connect(self.handle_results_folder_changed)

    def connect_groups(self):
        for btn in self.gp_add_folder_buttons:
            self.__connect_btn(btn, self.handle_add_src_folder_clicked)
        for btn in self.gp_remove_folder_buttons:
            self.__connect_btn(btn, self.handle_remove_src_folder_clicked)
        for ctrl in self.gp_group_name_ctrls:
            self.__connect_line_edit(ctrl, self.update_comparisons)

    def __connect_btn(self, btn, callback):
        try:
            btn.clicked.connect(callback, type=Qt.UniqueConnection)
        except TypeError as err:
            if err.args[0] == 'connection is not unique':
                btn.clicked.disconnect()
                btn.clicked.connect(callback, type=Qt.UniqueConnection)
            else:
                raise err

    def __connect_line_edit(self, ctrl, callback):
        try:
            ctrl.editingFinished.connect(callback, type=Qt.UniqueConnection)
        except TypeError as err:
            if err.args[0] == 'connection is not unique':
                ctrl.editingFinished.disconnect()
                ctrl.editingFinished.connect(callback, type=Qt.UniqueConnection)
            else:
                raise err

    @property
    def comparisons(self):
        """

        Returns
        -------
            The list of all possible pairs of groups
        """
        return list(combinations(self.group_names, 2))

    @property
    def selected_comparisons(self):
        return [box.text().split(self.group_concatenator) for box in self.comparison_checkboxes if box.isChecked()]

    def update_comparisons(self):
        self.comparison_checkboxes = []
        for i in range(self.tab.comparisonsVerticalLayout.count(), -1, -1):  # Clear
            item = self.tab.comparisonsVerticalLayout.takeAt(i)
            if item is not None:
                widg = item.widget()
                widg.setParent(None)
                widg.deleteLater()
        for pair in self.comparisons:
            chk = QCheckBox(self.group_concatenator.join(pair))
            chk.setChecked(False)
            self.tab.comparisonsVerticalLayout.addWidget(chk)
            self.comparison_checkboxes.append(chk)

    def add_group(self):  # REFACTOR: better in tab object
        new_gp_id = self.n_groups + 1
        group_controls = create_clearmap_widget('sample_group_controls.ui', patch_parent_class='QWidget')
        group_controls.setupUi()
        self.tab.sampleFoldersToolBox.addItem(group_controls, f'Group {new_gp_id}')

        self.connect_groups()

    def remove_group(self):
        last_idx = self.tab.sampleFoldersToolBox.count() - 1
        widg = self.tab.sampleFoldersToolBox.widget(last_idx)
        self.tab.sampleFoldersToolBox.removeItem(last_idx)
        widg.setParent(None)
        widg.deleteLater()

    @property
    def n_groups(self):
        return self.tab.sampleFoldersToolBox.count()

    @property
    def group_names(self):
        return [lbl.text() for lbl in self.gp_group_name_ctrls]

    @group_names.setter
    def group_names(self, names):
        for w, name in zip(self.gp_group_name_ctrls, names):
            w.setText(name)

    @property
    def gp_group_name_ctrls(self):
        return self.get_gp_ctrls('NameLineEdit')

    @property
    def gp_add_folder_buttons(self):
        return self.get_gp_ctrls('AddSrcFolderBtn')
    
    @property
    def gp_remove_folder_buttons(self):
        return self.get_gp_ctrls('RemoveSrcFolderBtn')

    @property
    def gp_list_widget(self):
        return self.get_gp_ctrls('ListWidget')

    def get_gp_ctrls(self, ctrl_name):
        return [getattr(self.tab.sampleFoldersToolBox.widget(i), f'gp{ctrl_name}') for i in range(self.n_groups)]

    def set_paths(self, gp, paths):
        list_widget = self.gp_list_widget[gp-1]
        list_widget.clear()
        list_widget.addItems(paths)

    def get_paths(self, gp):  # TODO: should exist from group name
        list_widget = self.gp_list_widget[gp-1]
        return [list_widget.item(i).text() for i in range(list_widget.count())]

    def get_all_paths(self):
        return [self.get_paths(gp+1) for gp in range(self.n_groups)]

    @property
    def groups(self):
        return {gp: paths for gp, paths in zip(self.group_names, self.get_all_paths())}

    def handle_add_src_folder_clicked(self):
        gp = self.tab.sampleFoldersToolBox.currentIndex()
        folder_path = get_directory_dlg(self.preferences.start_folder, 'Select sample folder')
        if folder_path:
            self.gp_list_widget[gp].addItem(folder_path)

    def handle_remove_src_folder_clicked(self):
        print('call')
        gp = self.tab.sampleFoldersToolBox.currentIndex()
        sample_idx = self.gp_list_widget[gp].currentRow()
        _ = self.gp_list_widget[gp].takeItem(sample_idx)
        print(_.text())

    @property
    def results_folder(self):
        return self.tab.resultsFolderLineEdit.text()

    @results_folder.setter
    def results_folder(self, value):
        self.tab.resultsFolderLineEdit.setText(value)

    def handle_results_folder_changed(self):
        self.config['paths']['results_folder'] = self.results_folder

    @property
    def align(self):
        return self.tab.batchAlignCheckBox.isChecked()

    @align.setter
    def align(self, value):
        self.tab.batchAlignCheckBox.setChecked(value)

    # def handle_align_changed(self):
    #     self.

    @property
    def count_cells(self):
        return self.tab.batchCountCellsCheckBox.isChecked()

    @count_cells.setter
    def count_cells(self, value):
        self.tab.batchCountCellsCheckBox.isChecked()

    # def handle_count_cells_changed(self):

    @property
    def run_vaculature(self):
        return self.tab.batchVasculatureCheckBox.isChecked()

    @run_vaculature.setter
    def run_vaculature(self, value):
        self.tab.batchVasculatureCheckBox.setChecked(value)
