# -*- coding: utf-8 -*-
"""
params
======

All the classes that define parameters or group thereof for the tabs of the graphical interface
"""
import functools
import string
import warnings
from copy import deepcopy
from itertools import permutations
from pathlib import Path
from typing import List

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QInputDialog, QToolBox, QCheckBox, QPushButton, QLabel, QSlider, QHBoxLayout, QComboBox

from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE
from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.Utils.utilities import validate_orientation, snake_to_title, DEFAULT_ORIENTATION, get_item_recursive
from ClearMap.config.atlas import ATLAS_NAMES_MAP

from ClearMap.gui.gui_utils import create_clearmap_widget, clear_layout
from ClearMap.gui.dialogs import get_directory_dlg
from ClearMap.gui.params_interfaces import (ParamLink, UiParameter, UiParameterCollection,
                                            ChannelsUiParameterCollection, ChannelUiParameter)

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

from ClearMap.processors.sample_preparation import SampleManager


class SampleChannelParameters(ChannelUiParameter):
    nameChanged = pyqtSignal(str, str)
    orientationChanged = pyqtSignal(str, tuple)
    cropChanged = pyqtSignal(str, list, list, list)

    geometry_settings_from: str
    data_type: str
    extension: str
    path: str
    resolution: List[float]
    comments: str
    slice_x: List[int]
    slice_y: List[int]
    slice_z: List[int]

    def __init__(self, tab, channel_name):
        super().__init__(tab, channel_name, 'nameLineEdit')
        self.params_dict = {
            'geometry_settings_from': ParamLink(None, self.tab.sampleChannelGeometryChannelComboBox),
            'data_type': ParamLink(['data_type'], self.tab.dataTypeComboBox),
            'extension': ParamLink(['extension'], self.tab.extensionComboBox),
            'path': ParamLink(['path'], self.tab.pathPlainTextEdit),
            'resolution': ParamLink(['resolution'], self.tab.resolutionTriplet),
            'comments': ParamLink(['comments'], self.tab.commentsPlainTextEdit),
            'slice_x': ParamLink(['slicing', 'x'], self.tab.sliceXDoublet),
            'slice_y': ParamLink(['slicing', 'y'], self.tab.sliceYDoublet),
            'slice_z': ParamLink(['slicing', 'z'], self.tab.sliceZDoublet),
            'orientation': ['orientation']  #  Last in case of validation issues
        }
        # property to be dynamic
        # self.cfg_subtree = ['channels', channel_name]
        self.connect()

    def set_geometry_settings_from_options(self, items):
        self.tab.sampleChannelGeometryChannelComboBox.clear()
        self.tab.sampleChannelGeometryChannelComboBox.addItems(items)

    @property
    def cfg_subtree(self):
        return ['channels', self.name]

    def handle_name_changed(self):
        tab_widget = self.tab.parent().parent()  # Not clear what is n+1
        tab_widget.setTabText(self.page_index, self.name)
        cached_name = self._cached_name
        self._config['channels'][self.name] = self._config['channels'].pop(self._cached_name)
        self._cached_name = self.name
        self.nameChanged.emit(cached_name, self.name)

    def connect(self):
        self.nameWidget.editingFinished.connect(self.handle_name_changed)
        self.tab.orientXSpinBox.valueChanged.connect(self.handle_orientation_changed)  # REFACTOR: push to paramslinki instead
        self.tab.orientYSpinBox.valueChanged.connect(self.handle_orientation_changed)
        self.tab.orientZSpinBox.valueChanged.connect(self.handle_orientation_changed)
        self.connect_simple_widgets()

    @property
    def orientation(self):
        x = self.tab.orientXSpinBox.value()
        y = self.tab.orientYSpinBox.value()
        z = self.tab.orientZSpinBox.value()
        orientation = (x, y, z)
        return self.validate_orientation(orientation)  # FIXME: add validator in paramslink instead

    @orientation.setter
    def orientation(self, orientation):  # FIXME: only when all 3 are set
        orientation = self.validate_orientation(orientation)
        self.tab.orientXSpinBox.setValue(orientation[0])
        self.tab.orientYSpinBox.setValue(orientation[1])
        self.tab.orientZSpinBox.setValue(orientation[2])

    def validate_orientation(self, orientation):
        return validate_orientation(orientation, self.name, raise_error=False)

    def handle_orientation_changed(self, _):  # WARNING: does not seem to move up the stack because of pyqtsignals
        # WARNING: bypasses the setter and hence the validation
        if self.orientation == DEFAULT_ORIENTATION or 0 not in self.orientation:  # Default or fully defined, proceed
            self.config['orientation'] = self.orientation
        if 0 not in self.orientation:  # i.e. fully defined
            self.orientationChanged.emit(self.name, self.orientation)

    def handle_slice_x_changed(self):
        self.config['slicing']['x'] = self.slice_x
        self.cropChanged.emit(self.name, self.slice_x, self.slice_y, self.slice_z)

    def handle_slice_y_changed(self):
        self.config['slicing']['y'] = self.slice_y
        self.cropChanged.emit(self.name, self.slice_x, self.slice_y, self.slice_z)

    def handle_slice_z_changed(self):
        self.config['slicing']['z'] = self.slice_z
        self.cropChanged.emit(self.name, self.slice_x, self.slice_y, self.slice_z)


class SampleParameters(UiParameterCollection):  # FIXME: why is this not a ChannelsUiParameterCollection
    """
    Class that links the sample params file to the UI
    """
    plotMiniBrain = pyqtSignal(int)    # Bind by number because name may change
    plotAtlas = pyqtSignal(int)    # Bind by number because name may change
    channelNameChanged = pyqtSignal(str, str)
    channelsChanged = pyqtSignal(list, list)
    orientationChanged = pyqtSignal(str, tuple)
    cropChanged = pyqtSignal(str, list, list, list)

    def __init__(self, tab, src_folder=None):
        self.shared_sample_params = SharedSampleParams(tab, src_folder=src_folder)
        self.channel_params = {}
        super().__init__(tab)

    def __getitem__(self, channel):
        return self.channel_params[channel]

    def __setitem__(self, key, value):
        self.channel_params[key] = value

    def get(self, channel, default_value=None):
        return self.channel_params.get(channel, default_value)

    def keys(self):
        return self.channel_params.keys()

    def values(self):
        return self.channel_params.values()

    def items(self):
        return self.channel_params.items()

    def __iter__(self):
        return iter(self.channel_params)

    @property
    def channels(self):
        return list(self.channel_params.keys())

    @property
    def params(self):
        return [self.shared_sample_params] + list(self.channel_params.values())

    # def fix_cfg_file(self, f_path):

    def cfg_to_ui(self):
        self.shared_sample_params.cfg_to_ui()
        for channel in self.config['channels'].keys():
            self.add_channel(channel)
            self.channel_params[channel].cfg_to_ui()

    def add_channel(self, channel_name):
        channels_before = []
        if channel_name not in self.channel_params:
            if channel_name not in self.config['channels']:  # i.e. if we add after loading
                channels_before = self.channels  # FIXME: what do we do if chan in self.config but not in self.channels
                self.config['channels'][channel_name] = deepcopy(self.default_channel_config())
            channel_params = SampleChannelParameters(self.tab, channel_name)
            channel_params.nameChanged.connect(self.handle_channel_name_changed)
            channel_params.orientationChanged.connect(self.handle_orientation_changed)
            channel_params.cropChanged.connect(self.handle_slice_changed)

            channel_params.tab.plotMiniBrainPushButton.clicked.connect(
                functools.partial(self.plotMiniBrain.emit, channel_params.page_index))
            channel_params.tab.sampleViewAtlasPushButton.clicked.connect(
                functools.partial(self.plotAtlas.emit, channel_params.page_index))
            channel_params._config = self.config
            self.channel_params[channel_name] = channel_params
            channels_after = self.channels

        for chan, params in self.channel_params.items():
            new_items = list(set(self.channels) - {chan})
            params.set_geometry_settings_from_options(new_items)
            params.tab.sampleChannelGeometryChannelCopyPushButton.clicked.connect(
                functools.partial(self.propagate_params, chan))

        if channels_before:  # TODO: check if empty list should not be passed
            self.channelsChanged.emit(channels_before, channels_after)

    def handle_channel_name_changed(self, old_name, new_name):
        self.channelNameChanged.emit(old_name, new_name)

    def handle_orientation_changed(self, channel, orientation):
        self.orientationChanged.emit(channel, orientation)

    def handle_slice_changed(self, channel, slice_x, slice_y, slice_z):
        self.cropChanged.emit(channel, slice_x, slice_y, slice_z)

    def get_channel_name(self, channel_idx):
        return self.tab.channelsParamsTabWidget.tabText(channel_idx)

    def propagate_params(self, channel):
        target_params = self[channel]
        ref_params = self[target_params.geometry_settings_from]
        for key in ('slice_x', 'slice_y', 'slice_z', 'resolution', 'orientation'):
            setattr(target_params, key, getattr(ref_params, key))

    def default_channel_config(self):
        return {
            'data_type': 'undefined',
            'extension': '.ome.tif',
            'path': '',
            'resolution': [0, 0, 0],
            'orientation': (1, 2, 3),
            'comments': '',
            'slicing': {'x': None, 'y': None, 'z': None}
        }


class SharedSampleParams(UiParameter):
    """
    Links the shared sample parameters (i.e. independent of channel like
    folder, sample_id...) to the UI

    Attributes
    ----------
    sample_id : str
        The ID of the sample. This must be unique
    use_id_as_prefix : bool
        Whether to use the sample ID as a prefix for the output files
    default_tile_extension : str
        The extension of the tile files
    """
    sample_id: str
    use_id_as_prefix: bool
    default_tile_extension: str

    def __init__(self, tab, src_folder=None):
        super().__init__(tab)
        self.params_dict = {
            'sample_id': ParamLink(['sample_id'], self.tab.sampleIdTxt, connect=self.handle_sample_id_changed),
            'use_id_as_prefix': ParamLink(['use_id_as_prefix'], self.tab.useIdAsPrefixCheckBox),
            'default_tile_extension': ParamLink(['default_tile_extension'], self.tab.defaultTileExtensionLineEdit),
        }
        self.src_folder = src_folder
        self.connect()
        if self.sample_id:
            self.handle_sample_id_changed(self.sample_id)

    @property
    def channels(self):
        return list(self.config['channels'].keys())

    def connect(self):
        # self.tab.sampleIdTxt.editingFinished.connect(self.handle_sample_id_changed)
        self.connect_simple_widgets()

    def _ui_to_cfg(self):
        self._config['base_directory'] = self.src_folder

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()

    def fix_cfg_file(self, f_path):  # REFACTOR: seems wrong to pass f_path just for that usage
        f_path = Path(f_path)
        self.config['base_directory'] = f_path.parent  # WARNING: needs to be self.config
                                                                 #  to be sure that we are up to date
                                                                 #  (otherwise write but potentially no reload)
        if not self.sample_id:
            sample_id, ok = QInputDialog.getText(self.tab, 'Warning: missing ID',
                                                 '<b>Missing sample ID</b><br>Please input below')
            self.sample_id = sample_id
            if not ok:
                raise ValueError('Missing sample ID')
        self.config['sample_id'] = self.sample_id
        self.config['use_id_as_prefix'] = self.use_id_as_prefix
        self.config.write()

    # Sample params
    # @property
    # def sample_id(self):
    #     return self.tab.sampleIdTxt.text()
    #
    # @sample_id.setter
    # def sample_id(self, id_):
    #     self.tab.sampleIdTxt.setText(id_)

    def handle_sample_id_changed(self, id_=None):
        if self.config is not None:
            self.config['sample_id'] = self.sample_id
            self.ui_to_cfg()   # FIXME: check


class StitchingParams(ChannelsUiParameterCollection):
    """
    Class that groups all the parameters related to the stitching of the sample
    (i.e. rigid and wobbly stitching)
    """
    def __init__(self, tab):
        super().__init__(tab)
        self.__extra_channel = {}

    def add_channel(self, channel_name, data_type=None):
        if channel_name in self.channels:
            return
        else:
            if 'channel_x' in self.config['channels']:
                self.fix_default_config(channel_name)
            if channel_name not in self.config['channels']:
                self.config['channels'][channel_name] = deepcopy(self.__extra_channel)
                self.config['channels'][channel_name]['layout_channel'] = self.channels[0]
                self.config.write()
            self[channel_name] = ChannelStitchingParams(self.tab, channel_name, config=self.config)

    def fix_default_config(self, channel_name):
        default_section = deepcopy(self.config['channels']['channel_x'])
        self.__extra_channel = dict(self.config['channels']['channel_y'])
        self.config['channels'] = {}
        self.config['channels'][channel_name] = default_section
        self.config['channels'][channel_name]['layout_channel'] = channel_name
        self.config.write()

    def handle_layout_channel_changed(self, channel, layout_channel):
        self.layoutChannelChanged.emit(channel, layout_channel)

    def compute_layout(self, channel):
        return self[channel].compute_layout()

    def fix_cfg_file(self, f_path):
        pass

    def get_channels_to_run(self):
        return [channel for channel in self.keys() if self[channel].run]  # FIXME: do not bind run

    def set_channels_to_run(self, channels):
        for channel in self.keys():
            status = self[channel].run
            self[channel].run = status or (channel in channels)

    @property
    def params(self):
        return list(self.values())


class ChannelStitchingParams(UiParameterCollection):
    layoutChannelChanged = pyqtSignal(str, str)
    def __init__(self, tab, channel, config):
        super().__init__(tab)
        self.name = channel
        self.ready = False
        self.shared = GeneralChannelStitchingParams(tab, channel)
        self.stitching_rigid = None
        self.stitching_wobbly = None
        self.read_configs(cfg=config)  # Required for cfg_to_ui and compute_layout
        self.shared.tab.rigidParamsGroupBox.setVisible(self.compute_layout())
        self.shared.tab.wobblyParamsGroupBox.setVisible(self.compute_layout())
        self.shared.tab.rigidParamsGroupBox.setEnabled(not self.shared.use_existing_layout)
        self.shared.tab.wobblyParamsGroupBox.setEnabled(not self.shared.use_existing_layout)

        if self.compute_layout():
            self.stitching_rigid = RigidChannelStitchingParams(tab, channel)
            if 'rigid' not in self.config['channels'][channel].keys():
                raise ClearMapValueError('Missing rigid stitching config although set for computing')
            self.stitching_wobbly = WobblyChannelStitchingParams(tab, channel)
            if 'wobbly' not in self.config['channels'][channel].keys():
                raise ClearMapValueError('Missing wobbly stitching config although set for computing')
            self.read_configs(cfg=config)

        self.shared.layoutChannelChanged.connect(self.handle_layout_channel_changed)  # FIXME: why not in self.connect
        self.shared.useExistingLayoutChanged.connect(self.handle_use_existing_layout_changed)

        self.cfg_to_ui()

    def write_config(self):
        cfg = deepcopy(self.config)
        if not self.compute_layout():
            cfg['channels'][self.name].pop('rigid', None)
            cfg['channels'][self.name].pop('wobbly', None)
        cfg.write()

    def handle_layout_channel_changed(self, channel, layout_channel):
        if not self.ready:
            return
        compute_layout = self.compute_layout()
        created = False
        if self.stitching_rigid and self.stitching_wobbly:
            self.stitching_rigid.set_visible(compute_layout)
            self.stitching_wobbly.set_visible(compute_layout)
        else:
            if compute_layout:
                self.stitching_rigid = RigidChannelStitchingParams(self.tab, self.name)
                self.stitching_wobbly = WobblyChannelStitchingParams(self.tab, self.name)
                self.stitching_rigid.set_visible(compute_layout)
                self.stitching_wobbly.set_visible(compute_layout)
                created = True
        if created:
            self.read_configs(cfg=self.config)
        config = self.config['channels'][self.name]
        if compute_layout and 'rigid' not in config.keys():
            config['rigid'] = self._default_config['channels']['channel_x']['rigid']
            config['wobbly'] = self._default_config['channels']['channel_x']['wobbly']
            self.write_config()
        if not compute_layout and 'rigid' in config.keys():
            self.config['channels'][self.name].pop('rigid')
            self.config['channels'][self.name].pop('wobbly')
            self.write_config()

        self.shared.tab.useExistingLayoutCheckBox.setVisible(compute_layout)

    def handle_use_existing_layout_changed(self, channel, use_existing_layout):
        if not self.ready:
            return
        compute_layout = self.compute_layout()
        if compute_layout:
            self.stitching_rigid.set_enabled(not use_existing_layout)
            self.stitching_wobbly.set_enabled(not use_existing_layout)

    def compute_layout(self):
        """
        Checks if the layout should be computed for this channel.
        This is the case if the layout channel is the current channel.
        If the config has not been pushed to the ui yet, it will use the config directly.

        Returns
        -------
        bool
            Whether the stitching layout should be computed for this channel
        """
        layout_channel = self.shared.layout_channel or self.shared.config['layout_channel']
        return layout_channel == self.name

    @property
    def params(self):
        return self.shared, self.stitching_rigid, self.stitching_wobbly  # TODO: check if None is a problem


class GeneralChannelStitchingParams(ChannelUiParameter):
    layoutChannelChanged = pyqtSignal(str, str)
    useExistingLayoutChanged = pyqtSignal(str, bool)

    use_npy: bool
    run: bool
    layout_channel: str
    use_existing_layout: bool

    def __init__(self, tab, channel_name):
        super().__init__(tab, channel_name)
        self.params_dict = {
            'use_npy': ParamLink(['use_npy'], self.tab.useNpyCheckBox),
            # 'run': ParamLink(['run'], self.tab.runCheckBox),
            'run': ['run'],
            'layout_channel': ParamLink(['layout_channel'], self.tab.layoutChannelComboBox),
            'use_existing_layout': ParamLink(['use_existing_layout'], self.tab.useExistingLayoutCheckBox, default=False),
        }
        self.connect()

    @property
    def run(self):
        return self.config['run']

    @run.setter
    def run(self, value):
        self.config['run'] = value

    @property
    def cfg_subtree(self):
        return ['channels', self.name]

    def handle_name_changed(self, old_name, new_name):
        if old_name != self._cached_name:
            warnings.warn(f'Channel name changed from {old_name} to {new_name} but was not expected')
        # private config because absolute path
        # TODO: check if dict() is required
        self._config['channels'][self.name] = self._config['channels'].pop(self._cached_name)
        self._cached_name = self.name

    def connect(self):
        self.nameWidget.channelRenamed.connect(self.handle_name_changed)
        self.connect_simple_widgets()
        self.tab.layoutChannelComboBox.currentTextChanged.connect(self.handle_layout_channel_changed)
        self.tab.useExistingLayoutCheckBox.stateChanged.connect(self.handle_use_existing_layout_changed)

    def handle_layout_channel_changed(self, layout_channel):
        self.config['layout_channel'] = layout_channel
        self.layoutChannelChanged.emit(self.name, layout_channel)

    def handle_use_existing_layout_changed(self, state):
        state = bool(state)
        self.config['use_existing_layout'] = state
        self.useExistingLayoutChanged.emit(self.name, state)


class RigidChannelStitchingParams(ChannelUiParameter):
    skip: bool
    x_overlap: int
    y_overlap: int
    projection_thickness: List[int]
    max_shifts_x: List[int]
    max_shifts_y: List[int]
    max_shifts_z: List[int]
    background_level: int
    background_pixels: int

    def __init__(self, tab, channel_name):
        super().__init__(tab, channel_name)
        self.params_dict = {
            'skip': ParamLink(['skip'], self.tab.rigidParamsGroupBox),
            'x_overlap': ParamLink(['overlap_x'], self.tab.xOverlapSinglet),
            'y_overlap': ParamLink(['overlap_y'], self.tab.yOverlapSinglet),
            'projection_thickness': ['projection_thickness'],
            'max_shifts_x': ParamLink(['max_shifts_x'], self.tab.rigidMaxShiftsXDoublet),
            'max_shifts_y': ParamLink(['max_shifts_y'], self.tab.rigidMaxShiftsYDoublet),
            'max_shifts_z': ParamLink(['max_shifts_z'], self.tab.rigidMaxShiftsZDoublet),
            'background_level': ParamLink(['background_level'], self.tab.rigidBackgroundLevel),
            'background_pixels': ParamLink(['background_pixels'], self.tab.rigidBackgroundPixels)
        }
        self.attrs_to_invert = ['skip']
        self.connect()

    def connect(self):
        self.tab.projectionThicknessDoublet.valueChangedConnect(self.handle_projection_thickness_changed)
        self.connect_simple_widgets()

    @property
    def cfg_subtree(self):
        return ['channels', self.name, 'rigid']

    def handle_name_changed(self):
        pass  #  handled by shared params

    def set_visible(self, state):
        self.tab.rigidParamsGroupBox.setVisible(state)

    def set_enabled(self, state):
        self.tab.rigidParamsGroupBox.setEnabled(state)

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

    def handle_projection_thickness_changed(self):
        self.config['projection_thickness'] = self.projection_thickness


class WobblyChannelStitchingParams(ChannelUiParameter):
    skip: bool
    max_shifts_x: List[int]
    max_shifts_y: List[int]
    max_shifts_z: List[int]
    stack_valid_range: list
    stack_pixel_size: int | None
    slice_valid_range: List[int]
    slice_pixel_size: int  | None

    def __init__(self, tab, channel_name):
        super().__init__(tab, channel_name)
        self.params_dict = {
            'skip': ParamLink(['skip'], self.tab.wobblyParamsGroupBox),
            'max_shifts_x': ParamLink(['max_shifts_x'], self.tab.wobblyMaxShiftsXDoublet),
            'max_shifts_y': ParamLink(['max_shifts_y'], self.tab.wobblyMaxShiftsYDoublet),
            'max_shifts_z': ParamLink(['max_shifts_z'], self.tab.wobblyMaxShiftsZDoublet),
            'stack_valid_range': ParamLink(['stack_valid_range'], self.tab.wobblyStackValidRangeDoublet),
            'stack_pixel_size': ParamLink(['stack_pixel_size'], self.tab.wobblyStackPixelSizeSinglet),
            'slice_valid_range': ParamLink(['slice_valid_range'], self.tab.wobblySliceRangeDoublet),
            'slice_pixel_size': ParamLink(['slice_pixel_size'], self.tab.wobblySlicePixelSizeSinglet)
        }
        self.attrs_to_invert = ['skip']
        self.connect()

    def connect(self):
        self.connect_simple_widgets()

    @property
    def cfg_subtree(self):
        return ['channels', self.name, 'wobbly']

    def handle_name_changed(self):
        pass  #  handled by shared params

    def set_visible(self, state):
        self.tab.wobblyParamsGroupBox.setVisible(state)

    def set_enabled(self, state):
        self.tab.wobblyParamsGroupBox.setEnabled(state)


class ChannelRegistrationParams(ChannelUiParameter):  # FIXME: add signal for align_with_changed
    align_with_changed = pyqtSignal(str, str)

    def __init__(self, tab, channel_name):
        super().__init__(tab, channel_name)
        self.params_dict = {
            'resample': ParamLink(['resample'], self.tab.resampleCheckBox),
            'resampled_resolution': ParamLink(['resampled_resolution'], self.tab.resampleResolutionTriplet),
            'align_with': ParamLink(['align_with'], self.tab.alignWithComboBox),
            'moving_channel': ParamLink(['moving_channel'], self.tab.movingChannelComboBox),
            'params_files': ParamLink(['params_files'], self.tab.paramsFilesListWidget),
            # 'landmarks_weights': ['landmarks_weights'],
        }
        self.connect()

    @property
    def cfg_subtree(self):
        return ['channels', self.name]

    def handle_name_changed(self, old_name, new_name):
        if old_name != self._cached_name:
            warnings.warn(f'Channel name changed from {old_name} to {new_name} but was not expected')
        # private config because absolute path
        # TODO: check if dict() is required
        self._config['channels'][self.name] = self._config['channels'].pop(self._cached_name)
        self._cached_name = self.name

    @property
    def n_registration_files(self):
        return self.tab.paramsFilesListWidget.count()

    @property
    def landmarks_weights(self):
        return {getattr(self.tab, f'param{x}Label').text(): getattr(self.tab, f'param{x}HorizontalSlider').value()
                for x in range(self.n_registration_files)}

    @landmarks_weights.setter
    def landmarks_weights(self, value):
        for param_idx in range(self.n_registration_files):
            param_name = getattr(self.tab, f'param{param_idx}Label').text()
            if param_name in value:
                getattr(self.tab, f'param{param_idx}HorizontalSlider').setValue(value[param_name])

    @property
    def use_landmarks_for(self):
        return [k for k, v in self.landmarks_weights.items() if v > 0]

    def connect(self):
        self.nameWidget.channelRenamed.connect(self.handle_name_changed)
        self.connect_simple_widgets()
        self.tab.alignWithComboBox.currentTextChanged.connect(self.handle_align_with_changed)
        if hasattr(self.tab.paramsFilesListWidget, 'itemsChanged'):
            self.tab.paramsFilesListWidget.itemsChanged.connect(self.handle_params_files_changed)

    def handle_align_with_changed(self, align_with):
        if align_with == 'None':
            align_with = None
        self.config['align_with'] = align_with  # TODO: check why necessary
        # self.ui_to_cfg()  # TODO: check
        self.align_with_changed.emit(self.name, align_with)

    def handle_params_files_changed(self):  # TODO: hide by default (unless advanced mode)

        def update_label_value(idx, value):
            lbl = self._value_labels[idx]
            lbl.setText(f"({value if value else 'disabled'})")

        clear_layout(self.tab.landmarksWeightsLayout)

        new_params_files = [snake_to_title(p.split('.')[0]) for p in self.tab.paramsFilesListWidget.get_items_text()]
        if len(new_params_files) > len(self.config['landmarks_weights']):
            self.config['landmarks_weights'] += [0] * (len(new_params_files) - len(self.config['landmarks_weights']))
        self._value_labels = []
        for idx, param in enumerate(new_params_files):
            label = QLabel(param)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0); slider.setMaximum(100)
            slider.setValue(self.config['landmarks_weights'][idx])

            self._value_labels.append(QLabel(f"({slider.value()})"))

            # Create a vertical layout for the slider and value label
            top_layout = QHBoxLayout()
            top_layout.addWidget(label)
            top_layout.addWidget(QLabel("<b>0</b>"))
            top_layout.addWidget(slider)
            top_layout.addWidget(QLabel("<b>100%</b>"))
            top_layout.addWidget(self._value_labels[idx])

            self.tab.landmarksWeightsLayout.addLayout(top_layout, idx, 0)

            slider.valueChanged.connect(functools.partial(update_label_value, idx))
            slider.valueChanged.connect(functools.partial(self.handle_landmarks_weight_changed, idx))

    def handle_landmarks_weight_changed(self, idx, value):
        value = self.scale_landmarks(value)
        if idx < len(self.config['landmarks_weights']):
            self.config['landmarks_weights'][idx] = value
        elif idx == len(self.config['landmarks_weights']):
            self.config['landmarks_weights'].append(value)
        else:
            raise ValueError(f'Index {idx} out of bounds for channel {self.name} '
                             f'landmarks weights {self.config["landmarks_weights"]}')
        # self.ui_to_cfg()

    def scale_landmarks(self, value):
        """
        Scale the weight of the landmarks using an exponential function so that
        the ratio between the weights follows a geometric progression with the slider value

        Parameters
        ----------
        value: float
            The value of the slider (between 0 and 100)

        Returns
        -------
        float
            The scaled weight (between 0 and max_ratio)
        """
        min_ratio = 1/20000
        max_ratio = 200
        if value == 0:
            return 0
        else:
            return min_ratio * (max_ratio / min_ratio) ** (value / 100)


class SharedRegistrationParams(UiParameter):
    def __init__(self, tab):
        super().__init__(tab)
        self.params_dict = {
            'plot_channel': ParamLink(None, self.tab.plotChannelComboBox),
            'plot_composite': ParamLink(None, self.tab.plotCompositeCheckBox),
        }
        self.connect()


class RegistrationParams(ChannelsUiParameterCollection):  # FIXME: does not seem to follow tab click
    launchLandmarksDialog = pyqtSignal(int)  # Bind by number because name may change

    def __init__(self, tab):
        super().__init__(tab)
        self.atlas_params = AtlasParams(tab)
        self.shared_params = SharedRegistrationParams(tab)

    def add_channel(self, channel_name, data_type=None):
        if channel_name in self.keys():
            return
        else:
            if channel_name not in self.config['channels']:
                warnings.warn(f'Channel {channel_name} not in config, adding default')
                if data_type == 'autofluorescence':
                    absolute_default = self._default_config['channels']['autofluorescence'] # because default config
                else:
                    absolute_default = self._default_config['channels']['channel_x']
                default_cfg = self._default_config['channels'].get(channel_name, absolute_default)
                self.config['channels'][channel_name] = deepcopy(default_cfg)
                if data_type != 'autofluorescence':
                    for other_channel, cfg in self.config['channels'].items():
                        if cfg['align_with'] == 'atlas':  # Ref channel
                            self.config['channels'][channel_name]['align_with'] = other_channel
                            self.config['channels'][channel_name]['moving_channel'] = other_channel
                            break
            channel_params = ChannelRegistrationParams(self.tab, channel_name)
            self[channel_name] = channel_params
            channel_params.tab.selectLandmarksPushButton.clicked.connect(
                functools.partial(self.launchLandmarksDialog.emit, channel_params.page_index))

    @property
    def params(self):
        return [self.atlas_params, self.shared_params] + list(self.values())

    def get_channel_name(self, channel_idx):
        return self.tab.channelsParamsTabWidget.tabText(channel_idx)


class AtlasParams(UiParameter):
    atlas_id_changed = pyqtSignal(str)
    atlas_structure_tree_id_changed = pyqtSignal(str)

    atlas_id: str
    structure_tree_id: str
    atlas_folder: str

    def __init__(self, tab):
        super().__init__(tab)
        self.params_dict = {
            'atlas_id': ['id'],
            'structure_tree_id': ['structure_tree_id'],
            'atlas_folder': ParamLink(['align_files_folder'], self.tab.atlasFolderPath, connect=False),
            'atlas_resolution': ParamLink(None, self.tab.atlasResolutionTriplet),  # TODO: check if we bind to cfg here
        }
        self.atlas_info = ATLAS_NAMES_MAP
        self.cfg_subtree = ['atlas']
        self.connect()
        # WARNING: after connect
        self.tab.atlasResolutionTriplet.setValue([self.atlas_info[self.atlas_id]['resolution']] * 3)

    def connect(self):
        self.tab.atlasResolutionTriplet.valueChangedConnect(self.handle_atlas_resolution_changed)  # TODO: replace with label
        self.tab.atlasIdComboBox.currentTextChanged.connect(self.handle_atlas_id_changed)
        self.tab.structureTreeIdComboBox.currentTextChanged.connect(self.handle_structure_tree_id_changed)
        self.connect_simple_widgets()

    @property
    def atlas_base_name(self):
        return self.atlas_info[self.atlas_id]['base_name']

    def handle_atlas_resolution_changed(self, state):
        # WARNING: use parent config
        if self._config is not None:  #  only if config is loaded
            for channel_cfg in self._config['channels'].values():
                channel_cfg['resampled_resolution'] = self.atlas_resolution

    @property
    def atlas_id(self):
        return self.tab.atlasIdComboBox.currentText()

    @atlas_id.setter
    def atlas_id(self, value):
        self.tab.atlasIdComboBox.setCurrentText(value)

    def handle_atlas_id_changed(self):
        self.config['id'] = self.atlas_id
        resolution = [self.atlas_info[self.atlas_id]['resolution']] * 3
        self.tab.atlasResolutionTriplet.setValue(resolution)
        self.ui_to_cfg()
        self.atlas_id_changed.emit(self.atlas_base_name)

    @property
    def structure_tree_id(self):
        return self.tab.structureTreeIdComboBox.currentText()

    @structure_tree_id.setter
    def structure_tree_id(self, value):
        self.tab.structureTreeIdComboBox.setCurrentText(value)

    def handle_structure_tree_id_changed(self):
        self.config['structure_tree_id'] = self.structure_tree_id
        self.ui_to_cfg()   # TODO: check if required
        self.atlas_structure_tree_id_changed.emit(self.structure_tree_id)


class CellMapParams(ChannelsUiParameterCollection):
    def __init__(self, tab, sample_params, _, registration_params):
        super().__init__(tab)
        self.pipeline_name = 'CellMap'
        self.sample_params = sample_params
        self.registration_params = registration_params

    @property
    def channels_to_detect(self):
        return [c for c, v in self.sample_params.config['channels'].items() if
                v['data_type'] and CONTENT_TYPE_TO_PIPELINE[v['data_type']] == self.pipeline_name]

    def default_channel_config(self):
        return self._default_config['channels']['example']

    @property
    def channel_params(self):
        return self._channels

    def add_channel(self, channel_name, data_type=None):  # FIXME: not called
        if channel_name not in self.keys():
            if channel_name not in self.config['channels']:
                self.config['channels'][channel_name] = dict(deepcopy(self.default_channel_config()))
            dtype = np.uint16  # FIXME: read from stitched file
            channel_params = ChannelCellMapParams(self.tab, channel_name, self, dtype=dtype)
            channel_params._config = self.config
            self[channel_name] = channel_params

    @property
    def params(self):
        return self.values()

    # def handle_advanced_state_changed(self, state):
    #     for channel in self.values():
    #         channel.tab.handle_advanced_state_changed(state)


class ChannelCellMapParams(ChannelUiParameter):
    background_correction_diameter: List[int]
    maxima_shape: int
    detection_threshold: int
    cell_filter_size: List[int]
    cell_filter_intensity: List[int]
    voxelization_radii: List[int]
    detect_cells: bool
    filter_cells: bool
    voxelize: bool
    save_shape: bool
    colocalization_compatible: bool
    plot_when_finished: bool
    plot_detected_cells: bool
    crop_x_min: int
    crop_x_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_y_min: int
    crop_y_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_z_min: int
    crop_z_max: int  # TODO: if 99.9 % source put to 100% (None)
    n_detected_cells: int
    n_filtered_cells: int

    def __init__(self, tab, channel, main_params, dtype=np.uint16):
        super().__init__(tab, channel)
        self.params_dict = {
            'background_correction_diameter': ['detection', 'background_correction', 'diameter'],
            'maxima_shape': ParamLink(['detection', 'maxima_detection', 'shape'], self.tab.maximaShape),
            'detection_threshold': ParamLink(['detection', 'shape_detection', 'threshold'], self.tab.detectionThreshold),
            'cell_filter_size': ParamLink(['cell_filtration', 'thresholds', 'size'], self.tab.cellFilterThresholdSizeDoublet),
            'cell_filter_intensity': ParamLink(['cell_filtration', 'thresholds', 'intensity'],
                                               self.tab.cellFilterThresholdIntensityDoublet,
                                               connect=False),
            'voxelization_radii': ParamLink(['voxelization', 'radii'], self.tab.voxelizationRadiusTriplet),
            'crop_x_min': ParamLink(['detection', 'test_set_slicing', 'dim_0', 0], self.tab.detectionSubsetXRangeMin),
            'crop_x_max': ParamLink(['detection', 'test_set_slicing', 'dim_0', 1], self.tab.detectionSubsetXRangeMax),
            'crop_y_min': ParamLink(['detection', 'test_set_slicing', 'dim_1', 0], self.tab.detectionSubsetYRangeMin),
            'crop_y_max': ParamLink(['detection', 'test_set_slicing', 'dim_1', 1], self.tab.detectionSubsetYRangeMax),
            'crop_z_min': ParamLink(['detection', 'test_set_slicing', 'dim_2', 0], self.tab.detectionSubsetZRangeMin),
            'crop_z_max': ParamLink(['detection', 'test_set_slicing', 'dim_2', 1], self.tab.detectionSubsetZRangeMax),
            'plot_when_finished': ParamLink(['run', 'plot_when_finished'], self.tab.runCellMapPlotCheckBox),
            'detect_cells': ParamLink(None, self.tab.runCellMapDetectCellsCheckBox),
            'filter_cells': ParamLink(None, self.tab.runCellMapFilterCellsCheckBox),
            'voxelize': ParamLink(None, self.tab.runCellMapVoxelizeCheckBox),
            'save_shape': ParamLink(None, self.tab.runCellMapSaveShapeCheckBox),
            'colocalization_compatible': ParamLink(['detection', 'colocalization_compatible'],
                                                   self.tab.runCellMapColocalizationCompatibleCheckBox,
                                                   default=False, missing_ok=True),
            'n_detected_cells': ParamLink(None, self.tab.nDetectedCellsLabel),
            'n_filtered_cells': ParamLink(None, self.tab.nDetectedCellsAfterFilterLabel),
        }
        self.main_params = main_params
        self.dtype = dtype
        self.advanced_controls = [self.tab.detectionShapeGroupBox]
        self.connect()

    def handle_advanced_state_changed(self, state):
        for ctrl in self.advanced_controls:
            ctrl.setVisible(state)
        self.tab.detectionShapeGroupBox.setVisible(state)  # FIXME: seems redundant

    @property
    def cfg_subtree(self):
        return ['channels', self.name]

    def handle_name_changed(self):
        # private config because absolute path
        # TODO: check if dict() is required
        self._config['channels'][self.name] = self._config['channels'].pop(self._cached_name)
        self._cached_name = self.name

    def handle_colocalization_compatible_changed(self, state):
        self.config['detection']['colocalization_compatible'] = self.colocalization_compatible
        if self.colocalization_compatible:
            self.save_shape = True
        self.tab.runCellMapSaveShapeCheckBox.setEnabled(not self.colocalization_compatible)

    def connect(self):
        self.tab.backgroundCorrectionDiameter.valueChanged.connect(self.handle_background_correction_diameter_changed)
        self.tab.cellFilterThresholdIntensityDoublet.valueChangedConnect(self.handle_filter_intensity_changed)
        self.tab.runCellMapColocalizationCompatibleCheckBox.stateChanged.connect(self.handle_colocalization_compatible_changed)
        self.connect_simple_widgets()  # |TODO: automatise in parent class

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()

    @property
    def ratios(self):
        raw_res = np.array(self.main_params.sample_params[self.name].resolution)
        resampled_res = np.array(self.main_params.registration_params[self.name].resampled_resolution)
        ratios = resampled_res / raw_res  # to original
        return ratios

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
    def cell_filter_intensity(self):
        intensities = self.tab.cellFilterThresholdIntensityDoublet.getValue()
        if intensities is None:
            return
        else:
            intensities = list(intensities)
            if intensities[-1] == -1:
                intensities[-1] = np.iinfo(self.dtype).max
            return intensities

    @cell_filter_intensity.setter
    def cell_filter_intensity(self, intensity):
        self.tab.cellFilterThresholdIntensityDoublet.setValue(intensity)

    def handle_filter_intensity_changed(self, _):
        self.config['cell_filtration']['thresholds']['intensity'] = self.cell_filter_intensity

    def scale_axis(self, val, axis='x'):
        axis_ratio = self.ratios['xyz'.index(axis)]
        return round(val * axis_ratio)

    def reverse_scale_axis(self, val, axis='x'):
        axis_ratio = self.ratios['xyz'.index(axis)]
        return round(val / axis_ratio)

    @property
    def slice_tuples(self):
        return ((self.crop_x_min, self.crop_x_max),
                (self.crop_y_min, self.crop_y_max),
                (self.crop_z_min, self.crop_z_max))

    @property
    def slicing(self):
        return tuple([slice(ax[0], ax[1]) for ax in self.slice_tuples])


class TractMapParams(ChannelsUiParameterCollection):
    def __init__(self, tab, sample_params, _, registration_params):
        super().__init__(tab)
        self.pipeline_name = 'TractMap'
        self.sample_params = sample_params
        self.registration_params = registration_params

    @property
    def channels_to_process(self):
        return [c for c, v in self.sample_params.config['channels'].items() if
                CONTENT_TYPE_TO_PIPELINE[v['data_type']] == self.pipeline_name]

    def default_channel_config(self):
        return self._default_config['channels']['example']

    @property
    def channel_params(self):
        return self._channels

    def add_channel(self, channel_name, data_type=None):
        if channel_name not in self.keys():
            if channel_name not in self.config['channels']:
                self.config['channels'][channel_name] = dict(deepcopy(self.default_channel_config()))
            channel_params = ChannelTractMapParams(self.tab, channel_name, self)
            channel_params._config = self.config
            self[channel_name] = channel_params

    @property
    def params(self):
        return self.values()


class ChannelTractMapParams(ChannelUiParameter):
    clipping_decimation_ratio: int
    clipping_percents: List[float]
    clip_range: List[int]
    crop_x_min: int
    crop_x_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_y_min: int
    crop_y_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_z_min: int
    crop_z_max: int  # TODO: if 99.9 % source put to 100% (None)
    display_decimation_ratio: int  # For the "cells.feather" file
    voxelization_radii: List[int]
    binarize: bool
    extract_coordinates: bool
    transform_coordinates: bool
    label_coordinates: bool
    voxelize: bool
    export_df: bool
    # [[[parallel_params]]]
    # min_point_list_size = 1000000
    # max_point_list_size = 10000000
    # n_processes_binarization = 15
    # n_processes_resampling = 15
    # n_processes_where = 23
    # n_processes_transform = 23
    # n_processes_label = 23

    def __init__(self, tab, channel, main_params):
        super().__init__(tab, channel)
        self.params_dict = {
            'clipping_decimation_ratio': ParamLink(['binarization', 'decimation_ratio'], self.tab.clippingDecimationRatioSpinBox),
            'clipping_percents': ParamLink(['binarization', 'percentage_range'],
                                           self.tab.clippingPixelsPercentDoublet,
                                           default=[70, 99.999]),
            'clip_range': ParamLink(['binarization', 'clip_range'], self.tab.clipRangeDoublet),
            'crop_x_min': ParamLink(['test_set_slicing', 'dim_0', 0], self.tab.detectionSubsetXRangeMin),
            'crop_x_max': ParamLink(['test_set_slicing', 'dim_0', 1], self.tab.detectionSubsetXRangeMax),
            'crop_y_min': ParamLink(['test_set_slicing', 'dim_1', 0], self.tab.detectionSubsetYRangeMin),
            'crop_y_max': ParamLink(['test_set_slicing', 'dim_1', 1], self.tab.detectionSubsetYRangeMax),
            'crop_z_min': ParamLink(['test_set_slicing', 'dim_2', 0], self.tab.detectionSubsetZRangeMin),
            'crop_z_max': ParamLink(['test_set_slicing', 'dim_2', 1], self.tab.detectionSubsetZRangeMax),
            'display_decimation_ratio': ParamLink(['display', 'decimation_ratio'], self.tab.displayDecimationRatioSpinBox),
            'voxelization_radii': ParamLink(['voxelization', 'radii'], self.tab.voxelizationRadiusTriplet),
            'binarize': ParamLink(['steps', 'binarize'], self.tab.binarizeCheckBox),
            'extract_coordinates': ParamLink(['steps', 'extract_coordinates'], self.tab.extractCoordinatesCheckBox),
            'transform_coordinates': ParamLink(['steps', 'transform_coordinates'], self.tab.transformCoordinatesCheckBox),
            'label_coordinates': ParamLink(['steps', 'label_coordinates'], self.tab.labelCoordinatesCheckBox),
            'voxelize': ParamLink(['steps', 'voxelize'], self.tab.voxelizeCheckBox),
            'export_df': ParamLink(['steps', 'export_df'], self.tab.exportDfCheckBox),
        }
        self.main_params = main_params
        self.advanced_controls = []  # FIXME: put as default in parent class
        self.connect()

    @property
    def cfg_subtree(self):
        return ['channels', self.name]

    def handle_advanced_state_changed(self, state):
        for ctrl in self.advanced_controls:
            ctrl.setVisible(state)

    def handle_name_changed(self):
        # private config because absolute path
        # TODO: check if dict() is required
        self._config['channels'][self.name] = self._config['channels'].pop(self._cached_name)
        self._cached_name = self.name

    def connect(self):
        self.connect_simple_widgets()

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()

    @property
    def ratios(self):
        raw_res = np.array(self.main_params.sample_params[self.name].resolution)
        resampled_res = np.array(self.main_params.registration_params[self.name].resampled_resolution)
        ratios = resampled_res / raw_res  # to original
        return ratios

    def scale_axis(self, val, axis='x'):
        axis_ratio = self.ratios['xyz'.index(axis)]
        return round(val * axis_ratio)

    def reverse_scale_axis(self, val, axis='x'):
        axis_ratio = self.ratios['xyz'.index(axis)]
        return round(val / axis_ratio)

    @property
    def slice_tuples(self):
        return ((self.crop_x_min, self.crop_x_max),
                (self.crop_y_min, self.crop_y_max),
                (self.crop_z_min, self.crop_z_max))

    @property
    def slicing(self):
        return tuple([slice(ax[0], ax[1]) for ax in self.slice_tuples])


class ColocalizationParams(ChannelsUiParameterCollection):
    def __init__(self, tab, sample_params, _):#, registration_params):
        super().__init__(tab)
        self.pipeline_name = 'Colocalization'
        self.sample_params = sample_params
        # self.registration_params = registration_params

    @property
    def channels_to_process(self):
        return [c for c, v in self.sample_params.config['channels'].items() if
                CONTENT_TYPE_TO_PIPELINE[v['data_type']] == self.pipeline_name]  # FIXME:

    def default_channel_config(self):
        return self._default_config['channels']['example']

    @property
    def channel_params(self):
        return self._channels

    def add_channel(self, channel_name, data_type=None):
        if isinstance(channel_name, (tuple, list)):
            channel_name = '-'.join(channel_name)
        if channel_name not in self.keys():
            if channel_name not in self.config['channels']:
                self.config['channels'][channel_name] = dict(deepcopy(self.default_channel_config()))
            channel_params = ChannelColocalizationParams(self.tab, channel_name, self)
            channel_params._config = self.config
            self[channel_name] = channel_params

    @property
    def params(self):
        return self.values()


class ChannelColocalizationParams(ChannelUiParameter):
    particle_diameter: int
    n_processes: int
    max_particle_distance: float
    relative_overlap_threshold: float
    voxel_number_overlap_threshold: int

    def __init__(self, tab, channel, main_params):
        super().__init__(tab, channel)
        self.params_dict = {
            'particle_diameter': ParamLink(['comparison', 'particle_diameter'],
                                           self.tab.colocalizationDiameterSpinBox),
            'n_processes': ParamLink(['performance', 'n_processes'], self.tab.colocalizationNProcessesSpinBox),
            'max_particle_distance': ParamLink(['analysis', 'max_particle_distance'],
                                               self.tab.colocalizationMaxDistanceSpinBox),
            'relative_overlap_threshold': ParamLink(['analysis', 'relative_overlap_threshold'],
                                                    self.tab.colocalizationRelativeThresholdDoubleSpinBox),
            'voxel_number_overlap_threshold': ParamLink(['analysis', 'voxel_number_overlap_threshold'],
                                                        self.tab.colocalizationAbsoluteOverlapThresholdSpinBox),
        }
        self.main_params = main_params
        self.advanced_controls = []  # FIXME: put as default in parent class
        self.connect()

    @property
    def cfg_subtree(self):
        return ['channels', self.name]

    def handle_advanced_state_changed(self, state):
        for ctrl in self.advanced_controls:
            ctrl.setVisible(state)

    def handle_name_changed(self):
        # private config because absolute path
        # TODO: check if dict() is required
        self._config['channels'][self.name] = self._config['channels'].pop(self._cached_name)
        self._cached_name = self.name

    def connect(self):
        self.connect_simple_widgets()

    def cfg_to_ui(self):
        self.reload()
        super().cfg_to_ui()


class SharedVesselBinarizationParams(UiParameter):
    fill_combined: bool
    plot_step_1: str
    plot_step_2: str
    plot_channel_1: str
    plot_channel_2: str

    def __init__(self, tab):
        super().__init__(tab)
        self.params_dict = {
            'fill_combined': ParamLink(['combined', 'binary_fill'], self.tab.binarizationConbineBinaryFillingCheckBox),
            'plot_step_1': ParamLink(None, self.tab.binarizationPlotStep1ComboBox),
            'plot_step_2': ParamLink(None, self.tab.binarizationPlotStep2ComboBox),
            'plot_channel_1': ParamLink(None, self.tab.binarizationPlotChannel1ComboBox),
            'plot_channel_2': ParamLink(None, self.tab.binarizationPlotChannel2ComboBox),
        }

    def connect(self):
        self.connect_simple_widgets()


class VesselParams(ChannelsUiParameterCollection):

    def __init__(self, tab, sample_params, stitching_params, registration_params):
        super().__init__(tab)
        # self.sample_params = sample_params  # TODO: check if required
        # self.preprocessing_params = preprocessing_params  # TODO: check if required
        self.shared_binarization_params = SharedVesselBinarizationParams(tab)
        self.graph_params = VesselGraphParams(tab)
        self.visualization_params = VesselVisualizationParams(tab, sample_params, stitching_params, registration_params)

    @property
    def params(self):
        return list(self.values()) + [self.graph_params, self.visualization_params]

    def get_selected_steps_and_channels(self):
        steps = (self.plot_step_1, self.plot_step_2)
        channels = (self.plot_channel_1, self.plot_channel_2)
        channels = [c for s, c in zip(steps, channels) if s is not None]
        steps = [s for s in steps if s is not None]
        return steps, channels

    def add_channel(self, channel_name, data_type=None):
        if channel_name in self.channels:
            return
        else:
            if self.config['is_default']:
                self.fix_default_config()
            if channel_name not in self.config['binarization']:
                self.patch_config_section(channel_name, data_type)
            self[channel_name] = VesselBinarizationParams(self.tab, channel_name)

            if data_type == 'arteries':
                self.graph_params.use_arteries = True

    def fix_default_config(self):
        self._default_vessels_section = deepcopy(dict(self.config['binarization']['vessels']))
        self._default_arteries_section = dict(self.config['binarization']['arteries'])
        self._default_combined_section = dict(self.config['binarization']['combined'])
        self.config['binarization'] = {'combined': self._default_combined_section}
        self.config['is_default'] = False
        self.config.write()

    def patch_config_section(self, channel_name, data_type=None):
        if data_type in ('vessels', None):
            self.config['binarization']['vessels'] = self._default_vessels_section
        else:
            self.config['binarization'][channel_name] = self._default_arteries_section
        self.config.write()

    def fix_cfg_file(self, f_path):
        pass


class VesselBinarizationParams(ChannelUiParameter):
    run_binarization: bool
    binarization_clip_range: List[int]
    binarization_threshold: int
    run_smoothing: bool
    run_binary_filling: bool
    run_deep_filling: bool

    def __init__(self, tab, channel_name):
        super().__init__(tab, channel_name)
        self.params_dict = {
            # FIXME: add tabs to UI with matching control names
            'run_binarization': ParamLink(['binarize', 'run'], self.tab.runBinarizationCheckBox),
            'binarization_clip_range': ParamLink(['binarize', 'clip_range'], self.tab.binarizationClipRangeDoublet),
            'binarization_threshold': ['binarize', 'threshold'],  # WARNING: handled below
            'run_smoothing': ParamLink(['smooth', 'run'], self.tab.binarizationSmoothingCheckBox),
            'run_binary_filling': ParamLink(['binary_fill', 'run'], self.tab.binarizationBinaryFillingCheckBox),
            'run_deep_filling': ParamLink(['deep_fill', 'run'], self.tab.binarizationDeepFillingCheckBox),
        }
        # self.tab.binarizationControlsGroupBox.setTitle(channel_name)
        self.connect()

    # WARNING: we need to redefine this only because of the binarization key, should we use channels instead
    # REFACTORING:
    @property
    def default_config(self):
        if self.cfg_subtree:
            if self.name in self.cfg_subtree:
                default_channel = self._default_config['binarization'].keys()[0]
                default_sub_tree = self.cfg_subtree.copy()
                default_sub_tree[default_sub_tree.index(self.name)] = default_channel
                return get_item_recursive(self._default_config, default_sub_tree)
            else:
                try:
                    return get_item_recursive(self._default_config, self.cfg_subtree)
                except KeyError as err:
                    if self.name in str(err):
                        raise KeyError(f'Could not find channel {self.name} in default config file. '
                                       f'config sub tree: {self.cfg_subtree}')
        else:
            return self._default_config

    def handle_name_changed(self, old_name, new_name):
        if old_name != self._cached_name:
            warnings.warn(f'Channel name changed from {old_name} to {new_name} but was not expected')
        # private config because absolute path
        # TODO: check if dict() is required
        self._config['binarization'][self.name] = self._config['binarization'].pop(self._cached_name)
        self._cached_name = self.name
        # self.tab.binarizationControlsGroupBox.setTitle(new_name)

    @property
    def cfg_subtree(self):
        return ['binarization', self.name]

    def connect(self):
        self.nameWidget.channelRenamed.connect(self.handle_name_changed)
        self.tab.binarizationThresholdSpinBox.valueChanged.connect(self.handle_binarization_threshold_changed)
        self.connect_simple_widgets()

    @property
    def n_steps(self):
        n_steps = self.run_binarization
        n_steps += self.run_smoothing or self.run_binary_filling
        n_steps += self.run_deep_filling
        return

    @property
    def binarization_threshold(self):
        return self.sanitize_neg_one(self.tab.binarizationThresholdSpinBox.value())

    @binarization_threshold.setter
    def binarization_threshold(self, value):
        self.tab.binarizationThresholdSpinBox.setValue(self.sanitize_nones(value))

    def handle_binarization_threshold_changed(self):
        self.config['binarize']['threshold'] = self.binarization_threshold


class VesselGraphParams(UiParameter):
    skeletonize: bool
    build: bool
    clean: bool
    reduce: bool
    transform: bool
    annotate: bool
    use_arteries: bool
    vein_intensity_range_on_arteries_channel: List[int]
    restrictive_min_vein_radius: float
    permissive_min_vein_radius: float
    final_min_vein_radius: float
    arteries_min_radius: float
    max_arteries_tracing_iterations: int
    max_veins_tracing_iterations: int
    min_artery_size: int
    min_vein_size: int

    def __init__(self, tab):
        super().__init__(tab)
        self.params_dict = {
            'skeletonize': ParamLink(['graph_construction', 'skeletonize'], self.tab.buildGraphSkeletonizeCheckBox),
            'build': ParamLink(['graph_construction', 'build'], self.tab.buildGraphBuildCheckBox),
            'clean': ParamLink(['graph_construction', 'clean'], self.tab.buildGraphCleanCheckBox),
            'reduce': ParamLink(['graph_construction', 'reduce'], self.tab.buildGraphReduceCheckBox),
            'transform': ParamLink(['graph_construction', 'transform'], self.tab.buildGraphTransformCheckBox),
            'annotate':  ParamLink(['graph_construction', 'annotate'], self.tab.buildGraphRegisterCheckBox),
            'use_arteries': ParamLink(
                ['graph_construction', 'use_arteries'],
                self.tab.buildGraphUseArteriesCheckBox),
            'vein_intensity_range_on_arteries_channel': ParamLink(
                ['vessel_type_postprocessing', 'pre_filtering', 'vein_intensity_range_on_arteries_ch'],
                self.tab.veinIntensityRangeOnArteriesChannelDoublet),
            'restrictive_min_vein_radius': ParamLink(
                ['vessel_type_postprocessing', 'pre_filtering', 'restrictive_vein_radius'],
                self.tab.restrictiveMinVeinRadiusDoubleSpinBox),
            'permissive_min_vein_radius': ParamLink(
                ['vessel_type_postprocessing', 'pre_filtering', 'permissive_vein_radius'],
                self.tab.permissiveMinVeinRadiusDoubleSpinBox),
            'final_min_vein_radius': ParamLink(
                ['vessel_type_postprocessing', 'pre_filtering', 'final_vein_radius'],
                self.tab.finalMinVeinRadiusDoubleSpinBox),
            'arteries_min_radius': ParamLink(
                ['vessel_type_postprocessing', 'pre_filtering', 'arteries_min_radius'],
                self.tab.arteriesMinRadiusDoubleSpinBox),
            'max_arteries_tracing_iterations': ParamLink(
                ['vessel_type_postprocessing', 'tracing', 'max_arteries_iterations'],
                self.tab.maxArteriesTracingIterationsSpinBox),
            'max_veins_tracing_iterations': ParamLink(
                ['vessel_type_postprocessing', 'tracing', 'max_veins_iterations'],
                self.tab.maxVeinsTracingIterationsSpinBox),
            'min_artery_size': ParamLink(
                ['vessel_type_postprocessing', 'capillaries_removal', 'min_artery_size'],
                self.tab.minArterySizeSpinBox),  # WARNING: not the same unit as below
            'min_vein_size': ParamLink(['vessel_type_postprocessing', 'capillaries_removal', 'min_vein_size'],
                                       self.tab.minVeinSizeDoubleSpinBox)
        }
        self.connect()

    def connect(self):
        self.connect_simple_widgets()


class VesselVisualizationParams(UiParameter):
    crop_x_min: int
    crop_x_max: int
    crop_y_min: int
    crop_y_max: int
    crop_z_min: int
    crop_z_max: int
    graph_step: str
    plot_type: str
    voxelization_size: List[int]
    vertex_degrees: str
    weight_by_radius: bool

    def __init__(self, tab, sample_params=None, stitching_params=None, registration_params=None):
        super().__init__(tab)
        self.params_dict = {  # TODO: if 99.9 % source put to 100% (None)
            'crop_x_min': ParamLink(['slicing', 'dim_0', 0], self.tab.graphConstructionSlicerXRangeMin),
            'crop_x_max': ParamLink(['slicing', 'dim_0', 1], self.tab.graphConstructionSlicerXRangeMax),
            'crop_y_min': ParamLink(['slicing', 'dim_1', 0], self.tab.graphConstructionSlicerYRangeMin),
            'crop_y_max': ParamLink(['slicing', 'dim_1', 1], self.tab.graphConstructionSlicerYRangeMax),
            'crop_z_min': ParamLink(['slicing', 'dim_2', 0], self.tab.graphConstructionSlicerZRangeMin),
            'crop_z_max': ParamLink(['slicing', 'dim_2', 1], self.tab.graphConstructionSlicerZRangeMax),
            'graph_step': ParamLink(None, self.tab.graphSlicerStepComboBox, connect=False),
            'plot_type': ParamLink(None, self.tab.graphPlotTypeComboBox, connect=False),
            'voxelization_size': ParamLink(['voxelization', 'size'], self.tab.vasculatureVoxelizationRadiusTriplet),
            'vertex_degrees': ParamLink(None, self.tab.vasculatureVoxelizationFilterDegreesSinglet, connect=False),
            'weight_by_radius': ParamLink(None, self.tab.voxelizationWeightByRadiusCheckBox, connect=False)
            # 'filter_name': ParamLink(None, self.tab.voxelizationFitlerNameComboBox, connect=False),
            # 'weight_name': ParamLink(None, self.tab.voxelizationWeightNameComboBox, connect=False),
        }
        self.structure_id = None
        self.cfg_subtree = ['visualization']
        self.sample_params = sample_params
        self.stitching_params = stitching_params
        self.registration_params = registration_params
        self.connect()

    def connect(self):
        self.connect_simple_widgets()

    def set_structure_id(self, structure_widget):
        self.structure_id = int(structure_widget.text(1))

    @property
    def ratios(self):
        # First TubeMap channel since they should share resolution
        channel = [k for k, v in self.sample_params.items() if CONTENT_TYPE_TO_PIPELINE[v.data_type] == 'TubeMap'][0]
        raw_res = np.array(self.sample_params[channel].resolution)
        resampled_res = np.array(self.registration_params[channel].resampled_resolution)
        ratios = resampled_res / raw_res  # to original
        return ratios

    def scale_axis(self, val, axis='x'):
        return round(val * self.ratios['xyz'.index(axis)])

    def reverse_scale_axis(self, val, axis='x'):
        axis_ratio = self.ratios['xyz'.index(axis)]
        return round(val / axis_ratio)

    @property
    def slice_tuples(self):
        return ((self.crop_x_min, self.crop_x_max),
                (self.crop_y_min, self.crop_y_max),
                (self.crop_z_min, self.crop_z_max))

    @property
    def slicing(self):
        return tuple([slice(ax[0], ax[1]) for ax in self.slice_tuples])


class PreferencesParams(UiParameter):
    verbosity: str
    n_processes_file_conv: int
    n_processes_resampling: int
    n_processes_stitching: int
    n_processes_cell_detection: int
    n_processes_binarization: int
    chunk_size_min: int
    chunk_size_max: int
    chunk_size_overlap: int
    start_folder: str
    start_full_screen: bool
    lut: str
    font_size: int
    pattern_finder_min_n_files: int
    three_d_plot_bg: str

    def __init__(self, tab):
        super().__init__(tab)
        self.params_dict = {
            'verbosity': ['verbosity'],
            'n_processes_file_conv': ['n_processes_file_conv'],
            'n_processes_resampling': ['n_processes_resampling'],
            'n_processes_stitching': ['n_processes_stitching'],
            'n_processes_cell_detection': ['n_processes_cell_detection'],
            'n_processes_binarization': ['n_processes_binarization'],
            'chunk_size_min': ParamLink(['detection_chunk_size_min'], self.tab.chunkSizeMinSpinBox, connect=False),
            'chunk_size_max': ParamLink(['detection_chunk_size_max'], self.tab.chunkSizeMaxSpinBox, connect=False),
            'chunk_size_overlap': ParamLink(['detection_chunk_overlap'], self.tab.chunkSizeOverlapSpinBox, connect=False),
            'start_folder': ParamLink(['start_folder'], self.tab.startFolderLineEdit, connect=False),
            'start_full_screen': ParamLink(['start_full_screen'], self.tab.startFullScreenCheckBox, connect=False),
            'lut': ['default_lut'],
            'font_size': ParamLink(['font_size'], self.tab.fontSizeSpinBox, connect=False),
            'pattern_finder_min_n_files': ParamLink(['pattern_finder_min_n_files'],
                                                    self.tab.patternFinderMinFilesSpinBox, connect=False),
            'three_d_plot_bg': ['three_d_plot_bg']
        }
        self.connect()

    def _ui_to_cfg(self):  # TODO: check if live update (i.e. connected handlers) or only on save
        cfg = self._config
        cfg['verbosity'] = self.verbosity
        cfg['n_processes_file_conv'] = self.n_processes_file_conv
        cfg['n_processes_resampling'] = self.n_processes_resampling
        cfg['n_processes_stitching'] = self.n_processes_stitching
        cfg['n_processes_cell_detection'] = self.n_processes_cell_detection
        cfg['n_processes_binarization'] = self.n_processes_binarization
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
    def n_processes_resampling(self):
        return self.sanitize_neg_one(self.tab.nProcessesResamplingSpinBox.value())

    @n_processes_resampling.setter
    def n_processes_resampling(self, value):
        self.tab.nProcessesResamplingSpinBox.setValue(self.sanitize_nones(value))

    @property
    def n_processes_cell_detection(self):
        return self.sanitize_neg_one(self.tab.nProcessesCellDetectionSpinBox.value())

    @n_processes_cell_detection.setter
    def n_processes_cell_detection(self, n_procs):
        self.tab.nProcessesCellDetectionSpinBox.setValue(self.sanitize_nones(n_procs))

    @property
    def n_processes_binarization(self):
        return self.sanitize_neg_one(self.tab.nProcessesBinarizationSpinBox.value())

    @n_processes_binarization.setter
    def n_processes_binarization(self, value):
        self.tab.nProcessesBinarizationSpinBox.setValue(self.sanitize_nones(value))

    @property
    def lut(self):
        return self.tab.lutComboBox.currentText().lower()

    @lut.setter
    def lut(self, lut_name):
        self.tab.lutComboBox.setCurrentText(lut_name)

    @property
    def font_family(self):
        return self.tab.fontComboBox.currentFont().family()

    # @font_family.setter
    # def font_family(self, font):
    #     self.tab.fontComboBox.setCurrentFont(font)


class BatchParameters(UiParameter):
    def __init__(self, tab, preferences=None):
        super().__init__(tab)
        self.group_concatenator = ' vs '
        self.preferences = preferences
        self.tab.sampleFoldersToolBox = QToolBox(parent=self.tab)
        self.tab.sampleFoldersPageLayout.addWidget(self.tab.sampleFoldersToolBox, 3, 0)

        self.connect()

    def _ui_to_cfg(self):
        self.config['paths']['results_folder'] = self.results_folder
        self.config['groups'] = self.groups

    def cfg_to_ui(self):
        self.reload()
        self.results_folder = self.config['paths']['results_folder']
        self.group_names = self.config['groups'].keys()
        for i, gp_name in enumerate(self.group_names):
            if i >= self.n_groups:
                self.add_group()
            self.set_group_name(i, gp_name)
            self.set_paths(i, self.config['groups'][gp_name])

    def __connect_btn(self, btn, callback):
        try:
            btn.clicked.connect(callback, type=Qt.UniqueConnection)
        except TypeError as err:
            if err.args[0] == 'connection is not unique':
                btn.clicked.disconnect()
                btn.clicked.connect(callback, type=Qt.UniqueConnection)
            else:
                raise err

    def _connect_line_edit(self, ctrl, callback):
        try:
            ctrl.editingFinished.connect(callback, type=Qt.UniqueConnection)
        except TypeError as err:
            if err.args[0] == 'connection is not unique':
                ctrl.editingFinished.disconnect()
                ctrl.editingFinished.connect(callback, type=Qt.UniqueConnection)
            else:
                raise err

    def connect(self):
        self.tab.addGroupPushButton.clicked.connect(self.add_group)
        self.tab.removeGroupPushButton.clicked.connect(self.remove_group)
        self.tab.resultsFolderLineEdit.textChanged.connect(self.handle_results_folder_changed)
        # self.connect_simple_widgets()

    def connect_groups(self):
        for btn in self.gp_add_folder_buttons:
            self.__connect_btn(btn, self.handle_add_src_folder_clicked)
        for btn in self.gp_remove_folder_buttons:
            self.__connect_btn(btn, self.handle_remove_src_folder_clicked)

    def add_group(self):  # REFACTOR: better in tab object
        new_gp_id = self.n_groups + 1
        group_controls = create_clearmap_widget('sample_group_controls.ui', patch_parent_class='QWidget')
        self.tab.sampleFoldersToolBox.addItem(group_controls, f'Group {new_gp_id}')

        self.connect_groups()

    def remove_group(self):
        # last_idx = self.n_groups - 1  # remove current group instead
        current_idx = self.tab.sampleFoldersToolBox.currentIndex()
        group_name = self.group_names[current_idx]
        widg = self.tab.sampleFoldersToolBox.widget(current_idx)
        self.tab.sampleFoldersToolBox.removeItem(current_idx)
        widg.setParent(None)
        widg.deleteLater()
        self.config['groups'].pop(group_name)
        for k, v in self.config['comparisons'].items():
            if group_name in v:
                self.config['comparisons'].pop(k)
        # TODO: check if we write config

    @property
    def n_groups(self):
        return self.tab.sampleFoldersToolBox.count()

    @property
    def group_names(self):
        return [lbl.text() for lbl in self.gp_group_name_ctrls]

    @group_names.setter
    def group_names(self, names):
        if len(names) > self.n_groups:
            for i, name in enumerate(names):
                if i >= self.n_groups:
                    self.add_group()
        for w, name in zip(self.gp_group_name_ctrls, names):
            w.setText(name)

    def set_group_name(self, idx, name):
        self.gp_group_name_ctrls[idx].setText(name)

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

    def set_paths(self, gp_idx, paths):
        """
        Set the sample folder paths for group nb `gp_idx`

        .. warning::
            contrary to the interface, gp_idx is the 0 based index, not the displayed number

        Parameters
        ----------
        gp_idx: int
            The 0 based index of the group
        paths: List[str]
            The list of sample folder paths to set for this group
        """
        if gp_idx >= self.n_groups:
            self.add_group()
        list_widget = self.gp_list_widget[gp_idx]
        list_widget.clear()
        list_widget.addItems(paths)

    def get_paths(self, gp):  # TODO: should exist from group name
        list_widget = self.gp_list_widget[gp - 1]
        return [list_widget.item(i).text() for i in range(list_widget.count())]

    def get_all_paths(self):
        return [self.get_paths(gp + 1) for gp in range(self.n_groups)]

    @property
    def groups(self):
        return {gp: paths for gp, paths in zip(self.group_names, self.get_all_paths())}

    def handle_add_src_folder_clicked(self):
        gp = self.tab.sampleFoldersToolBox.currentIndex()
        folder_path = get_directory_dlg(self.preferences.start_folder, 'Select sample folder')
        if folder_path:
            self.gp_list_widget[gp].addItem(folder_path)

    def handle_remove_src_folder_clicked(self):
        gp = self.tab.sampleFoldersToolBox.currentIndex()
        sample_idx = self.gp_list_widget[gp].currentRow()
        _ = self.gp_list_widget[gp].takeItem(sample_idx)

    @property
    def results_folder(self):
        return self.tab.resultsFolderLineEdit.text()

    @results_folder.setter
    def results_folder(self, value):
        self.tab.resultsFolderLineEdit.setText(value)
        self.config['paths']['results_folder'] = value

    def handle_results_folder_changed(self):
        self.config['paths']['results_folder'] = self.results_folder


class GroupAnalysisParams(BatchParameters):
    """
    Essentially batch parameters with comparisons
    """
    # plot_channel: str
    compute_sd_and_effect_size: bool

    def __init__(self, tab, preferences=None):
        super().__init__(tab, preferences)
        self.params_dict = {
            # 'plot_channel': ParamLink(None, self.tab.plotChannelComboBox),
            'compute_sd_and_effect_size': ParamLink(None, self.tab.computeSdAndEffectSizeCheckBox),
        }
        self.plot_density_maps_buttons = []
        self.comparison_checkboxes = []
        self.plot_channel = ''

    def _ui_to_cfg(self):
        super()._ui_to_cfg()
        self.config['comparisons'] = {letter: pair for letter, pair in zip(string.ascii_lowercase,
                                                                           self.selected_comparisons)}

    def cfg_to_ui(self):
        super().cfg_to_ui()
        self.update_comparisons()
        for chk_bx in self.comparison_checkboxes:
            if chk_bx.text().split(self.group_concatenator) in self.config['comparisons'].values():
                self.set_check_state(chk_bx, True)

    def connect_groups(self):
        super().connect_groups()
        for ctrl in self.gp_group_name_ctrls:
            self._connect_line_edit(ctrl, self.update_comparisons)

    @property
    def comparisons(self):
        """

        Returns
        -------
            The list of all possible pairs of groups
        """
        return list(permutations(self.group_names, 2))

    @property
    def selected_comparisons(self):
        return [box.text().split(self.group_concatenator) for box in self.comparison_checkboxes if box.isChecked()]

    def update_comparisons(self):
        clear_layout(self.tab.comparisonsVerticalLayout)

        # checkboxes
        self.comparison_checkboxes = []
        for i, pair in enumerate(self.comparisons):
            chk = QCheckBox(self.group_concatenator.join(pair))
            chk.setChecked(i == 0)
            self.tab.comparisonsVerticalLayout.addWidget(chk)
            self.comparison_checkboxes.append(chk)

        self.tab.comparisonsVerticalLayout.addStretch()

        # plot buttons
        self.plot_density_maps_buttons = []
        for gp in self.group_names:
            btn = QPushButton(f'Plot {gp} group density maps')
            self.tab.comparisonsVerticalLayout.addWidget(btn)
            self.plot_density_maps_buttons.append(btn)

        self.tab.comparisonsVerticalLayout.addStretch()

        plot_channel_combobox = QComboBox()
        sample_manager = SampleManager()
        sample_folders_paths = self.get_all_paths()
        if sample_folders_paths:
            sample_manager.setup(src_dir=sample_folders_paths[0][0])  # gp 0, sample 0
            plot_channel_combobox.addItems(sample_manager.channels_to_detect)
            self.tab.comparisonsVerticalLayout.addWidget(plot_channel_combobox)
            self.plot_channel = sample_manager.channels_to_detect[0]
            plot_channel_combobox.currentTextChanged.connect(self.handle_plot_channel_changed)
        self.plot_channel_combobox = plot_channel_combobox

    def handle_plot_channel_changed(self):
        self.plot_channel = self.plot_channel_combobox.currentText()


class BatchProcessingParams(BatchParameters):
    """
    Essentially BatchParameters with processing steps
    """

    def __init__(self, tab, preferences=None):
        super().__init__(tab, preferences)

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
