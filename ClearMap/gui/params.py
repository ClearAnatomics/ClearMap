"""
params
======

All the classes that define parameters or group thereof for the tabs of the graphical interface
"""
import functools
import string
import warnings
from typing import List, Optional, Callable

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QToolBox, QCheckBox, QLabel, QHBoxLayout, QVBoxLayout,
                             QSpinBox, QLineEdit, QDoubleSpinBox, QRadioButton, QFrame, QWidget)

from ClearMap.config.atlas import ATLAS_NAMES_MAP
from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.Utils.utilities import (validate_orientation, snake_to_title, set_item_recursive,
                                      DEFAULT_ORIENTATION,  trim_or_pad)
from ClearMap.Utils.event_bus import Publishes, EventBus
from ClearMap.Utils.events import (UiChannelRenamed, UiCropChanged, UiOrientationChanged, UiRequestPlotAtlas,
                                   UiConvertToClearMapFormat, UiRequestPlotMiniBrain, UiChannelsChanged,
                                   UiLayoutChannelChanged, UiUseExistingLayoutChanged, UiAlignWithChanged,
                                   UiRequestLandmarksDialog, UiAtlasIdChanged, UiAtlasStructureTreeIdChanged,
                                   UiVesselGraphFiltersChanged, UiBatchResultsFolderChanged, UiBatchGroupsChanged)

from .gui_utils_base import replace_widget
from .params_mixins import OrthoviewerSlicingMixin
from .widget_monkeypatch_callbacks import recursive_patch_compound_boxes
from .params_interfaces import (ParamLink, UiParameter, ChannelUiParameter, UiParameterCollection,
                                ChannelsUiParameterCollection, VectorLink, invert, param_setter, param_handler,
                                list_widget_setter, list_widget_getter)
from .widgets import LandmarksWeightsPanel, ComparisonsModel, ComparisonsWidgetAdapter, Pair, GroupsWidgetAdapter, \
    FileDropListWidget

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


class NProcessesParams(UiParameter):
    """
    Minimal helper for performance.<step>.n_processes
    """
    n_processes: int

    def __init__(self, tab, *, cfg_path: list[str],
                 event_bus: EventBus, get_view=None, apply_patch=None):
        self.cfg_path = list(cfg_path)
        super().__init__(tab, event_bus=event_bus,
                         get_view=get_view, apply_patch=apply_patch)

    def build_params_dict(self):
        return {'n_processes': ParamLink(self.cfg_path, self.tab.nProcessesSpinBox),}

    @property
    def cfg_subtree(self):
        # parent: e.g. ['tract_map', 'performance', 'binarization']
        return self.cfg_path[:-1]


class BlockProcessingParams(UiParameter):
    """
    Generic block-processing parameters (n_processes + size_min / size_max / overlap).

    You can attach this to any 'block_processing' subtree by passing cfg_prefix.
    tab is expected to be the BlockProcessingParamsWidget for that step.
    """

    n_processes: int
    size_min: int | None
    size_max: int | None
    overlap: int | None

    def __init__(self, tab, *, cfg_prefix: list[str],
                 event_bus: EventBus, get_view=None, apply_patch=None):
        self.cfg_prefix = list(cfg_prefix)
        super().__init__(tab, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)


    def build_params_dict(self):
        def path(*tail):
            return self.cfg_prefix + list(tail)

        return {
            'n_processes': ParamLink(path('n_processes'), self.tab.nProcessesSpinBox),
            'size_min': ParamLink(path('size_min'), self.tab.sizeMinSpinBox,
                                  disabled_value=None, ui_sentinel=-1, enforce_sentinel_min=True),
            'size_max': ParamLink(path('size_max'), self.tab.sizeMaxSpinBox,
                                  disabled_value=None, ui_sentinel=-1, enforce_sentinel_min=True),
            'overlap': ParamLink(path('overlap'), self.tab.overlapSpinBox,
                                 disabled_value=None, ui_sentinel=-1, enforce_sentinel_min=True),
        }

    @property
    def cfg_subtree(self):
        return self.cfg_prefix[:-1]


class SampleChannelParameters(ChannelUiParameter):
    publishes = Publishes(UiOrientationChanged, UiCropChanged)

    geometry_settings_from: str
    data_type: str
    extension: str
    path: str
    resolution: List[float]
    wavelength: Optional[int]
    comments: str
    slice_x: List[int]
    slice_y: List[int]
    slice_z: List[int]

    def __init__(self, tab, channel_name, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, channel_name, event_bus=event_bus, name_widget_name='nameLineEdit',
                         get_view=get_view, apply_patch=apply_patch)

    def build_params_dict(self):
        return {
            'geometry_settings_from': ParamLink(None, self.tab.sampleChannelGeometryChannelComboBox),
            'data_type': ParamLink(['data_type'], self.tab.dataTypeComboBox),
            'extension': ParamLink(['extension'], self.tab.extensionComboBox),
            'path': ParamLink(['path'], self.tab.pathPlainTextEdit),
            'resolution': VectorLink(['resolution'], self.tab.resolutionTriplet,
                                     disable_globally=False, disabled_value=None, ui_sentinel=-1,
                                     enforce_sentinel_min=True),
            'wavelength': ParamLink(['wavelength'], self.tab.wavelengthSpinBox,
                                    disabled_value=None, ui_sentinel=-1, enforce_sentinel_min=True),
            'comments': ParamLink(['comments'], self.tab.commentsPlainTextEdit),
            'slice_x': ParamLink(['slicing', 'x'], self.tab.sliceXDoublet,
                                 notify_apply=self._publish_crop_changed),
            'slice_y': ParamLink(['slicing', 'y'], self.tab.sliceYDoublet,
                                 notify_apply=self._publish_crop_changed),
            'slice_z': ParamLink(['slicing', 'z'], self.tab.sliceZDoublet,
                                 notify_apply=self._publish_crop_changed),
            'orientation': ['orientation']  #  Last in case of validation issues
        }

    def _publish_crop_changed(self, _=None):
        self.publish(UiCropChanged(channel_name=self.name, slice_x=self.slice_x,
                                   slice_y=self.slice_y, slice_z=self.slice_z))

    def set_geometry_settings_from_options(self, items):
        self.tab.sampleChannelGeometryChannelComboBox.clear()
        self.tab.sampleChannelGeometryChannelComboBox.addItems(items)

    @property
    def cfg_subtree(self):
        return ['sample', 'channels', self.name]    # REFACTOR: section name from config_handler

    @param_handler
    def handle_name_changed(self, old_name: str, new_name: str):
        self._apply_patch({'$rename': {'channels': {old_name: new_name}}})  # config emits event

    def connect(self):
        def _on_name_editing_finished():
            # tab_widget = self.tab.parent().parent()  # Not clear what is n+1
            channels_tab_w = self.tab.channelsParamsTabWidget
            old_name = channels_tab_w.tabText(self.page_index)  # still the old name at this time
            new_name = self.nameWidget.text().strip()
            if new_name and new_name != old_name:
                self.handle_name_changed(old_name, new_name)            # send patch
                channels_tab_w.setTabText(self.page_index, new_name)    # amend UI
                self.publish(UiChannelRenamed(old=old_name, new=new_name))

        self.nameWidget.editingFinished.connect(_on_name_editing_finished)
        self.tab.orientXSpinBox.valueChanged.connect(self.handle_orientation_changed)  # REFACTOR: push to paramslink instead
        self.tab.orientYSpinBox.valueChanged.connect(self.handle_orientation_changed)
        self.tab.orientZSpinBox.valueChanged.connect(self.handle_orientation_changed)

    @property
    def orientation(self):
        x = self.tab.orientXSpinBox.value()
        y = self.tab.orientYSpinBox.value()
        z = self.tab.orientZSpinBox.value()
        orientation = (x, y, z)
        return self.validate_orientation(orientation)  # REFACTOR: add validator in paramslink instead

    @orientation.setter
    @param_setter
    def orientation(self, orientation):  # FIXME: only when all 3 are set
        orientation = self.validate_orientation(orientation)
        self.tab.orientXSpinBox.setValue(orientation[0])
        self.tab.orientYSpinBox.setValue(orientation[1])
        self.tab.orientZSpinBox.setValue(orientation[2])

    def validate_orientation(self, orientation):
        return validate_orientation(orientation, self.name, raise_error=False)

    @param_handler
    def handle_orientation_changed(self, _):
        ori = list(self.orientation)
        if ori == DEFAULT_ORIENTATION or 0 not in ori:  # Default or fully defined, proceed
            self._update_value(['orientation'], ori)
        if 0 not in ori:  # i.e. fully defined
            self.publish(UiOrientationChanged(channel_name=self.name, orientation=ori))


class SampleParameters(ChannelsUiParameterCollection):
    """
    Class that links the sample params file to the UI
    """
    publishes = Publishes(UiConvertToClearMapFormat, UiRequestPlotMiniBrain, UiRequestPlotAtlas,
                          UiChannelRenamed, UiChannelsChanged, UiOrientationChanged, UiCropChanged)

    cfg_subtree = ['sample']

    def __init__(self, tab, *, event_bus: EventBus, get_view=None, apply_patch=None):
        self.shared_sample_params = SharedSampleParams(tab, event_bus=event_bus,
                                                       get_view=get_view, apply_patch=apply_patch)
        self._last_channels: Optional[List[str]] = None
        super().__init__(tab, pipeline_name='Sample', event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self._emit_timer = QTimer(self.tab)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.timeout.connect(self._flush_channels_changed)
        self._pending = None

    @property
    def config_channels(self):
        return list(self.view.get('channels', {}).keys())

    @property
    def params(self):
        return [self.shared_sample_params] + list(self.channel_params.values())

    def cfg_to_ui(self):
        # hydrate shared first
        self.shared_sample_params.cfg_to_ui()
        # create UI elements for channels in config
        for channel_name in self.config_channels:
            if channel_name not in self.channels:
                self.tab.add_channel_tab(channel_name)
                # self.ensure_channel_param(channel_name)
            self.channel_params[channel_name].cfg_to_ui()
        self.reconcile_from_config()

    def request_add_channel(self, channel_name: str):
        # guard against double-clicks based on current view
        if channel_name in self.config_channels:
            return
        self._update_value(['channels', channel_name], {})

    def ensure_channel_param(self, channel_name: str):
        if channel_name in self.channels:
            return
        channel_params = SampleChannelParameters(self.tab, channel_name, event_bus=self._bus,
                                                 get_view=self._get_view,
                                                 apply_patch=self._apply_patch)
        self._bind_channel_signals(channel_params)  # connect only on creation
        self.channel_params[channel_name] = channel_params

    def add_channel(self, channel_name: str):
        self.ensure_channel_param(channel_name)

    def _bind_channel_signals(self, channel_params: "SampleChannelParameters"):
        def get_current_index():
            tab_widget = self.tab.channelsParamsTabWidget
            idx = tab_widget.indexOf(channel_params.tab)
            return idx if idx != -1 else None

        def publish_with_current_index(event_class):
            idx = get_current_index()
            if idx is not None:
                self.publish(event_class(channel_index=idx))

        def publish_with_current_name(event_class):
            idx = get_current_index()
            if idx is not None:
                name = self.get_channel_name(idx)
                self.publish(event_class(channel_name=name))

        def run_with_current_name(func):
            idx = get_current_index()
            if idx is not None:
                func(self.get_channel_name(idx))

        channel_params.tab.convertToClearMapPushButton.clicked.connect(
            functools.partial(publish_with_current_name, UiConvertToClearMapFormat),
            type=Qt.UniqueConnection)  # avoid double binding (PyQt >= 5.14)
        channel_params.tab.plotMiniBrainPushButton.clicked.connect(
            functools.partial(publish_with_current_index, UiRequestPlotMiniBrain),
            type=Qt.UniqueConnection)
        channel_params.tab.sampleViewAtlasPushButton.clicked.connect(
            functools.partial(publish_with_current_index, UiRequestPlotAtlas),
            type=Qt.UniqueConnection)
        channel_params.tab.sampleChannelGeometryChannelCopyPushButton.clicked.connect(
            functools.partial(run_with_current_name, self.propagate_params),
            type=Qt.UniqueConnection)

    def reconcile_from_config(self):
        """Makes channels in UI match view (i.e. config)."""
        # create params for new channels
        for channel_name in set(self.config_channels) - set(self.channels):
            self.ensure_channel_param(channel_name)

        # remove params for deleted channels
        for channel_name in set(self.channels) - set(self.config_channels):
            self.channel_params[channel_name].teardown()
            del self.channel_params[channel_name]

        # Update list of channels geometry can be copied from
        for chan, params in self.channel_params.items():
            other_channels = list(set(self.config_channels) - {chan})
            params.set_geometry_settings_from_options(other_channels)

        # Signal that channels changed
        self._publish_channels_changed(self.config_channels)

    def _publish_channels_changed(self, new_list):
        """
        Publish UiChannelsChanged signal if the list of channels changed since last call.
        This is cached so that multiple rapid changes to the channels only emit one signal.

        Parameters
        ----------
        new_list: List[str]
            The new list of channels to compare to the last emitted one.
        """
        before = self._last_channels
        after = list(new_list)
        if self._last_channels is None:
            # establish baseline silently on first call
            self._last_channels = after
            return
        if self._last_channels != after:
            self._pending = (before, after)
            self._emit_timer.start(0)

    def _flush_channels_changed(self):
        if not self._pending:
            return
        before, after = self._pending
        self.publish(UiChannelsChanged(before=before, after=after))
        self._last_channels = after
        self._pending = None

    def get_channel_name(self, channel_idx):
        return self.tab.channelsParamsTabWidget.tabText(channel_idx)

    def propagate_params(self, channel):
        target_params = self[channel]
        ref_params = self[target_params.geometry_settings_from]
        for key in ('slice_x', 'slice_y', 'slice_z', 'resolution', 'orientation'):
            setattr(target_params, key, getattr(ref_params, key))


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
    src_folder: str
    sample_id: str
    use_id_as_prefix: bool
    default_tile_extension: str

    cfg_subtree = ['sample']

    def build_params_dict(self):
        return {
            'sample_id': ParamLink(['sample_id'], self.tab.sampleIdTxt),
            'use_id_as_prefix': ParamLink(['use_id_as_prefix'], self.tab.useIdAsPrefixCheckBox),
            'default_tile_extension': ParamLink(['default_tile_extension'], self.tab.defaultTileExtensionLineEdit),
            'src_folder': ParamLink(keys=None, widget=self.tab.srcFolderTxt, default=''),
        }

    @property
    def channels(self):
        return list(self.view['channels'].keys())


class StitchingParams(ChannelsUiParameterCollection):
    """
    Class that groups all the parameters related to the stitching of the sample
    (i.e. rigid and wobbly stitching)
    """
    publishes = Publishes(UiLayoutChannelChanged)

    def __init__(self, tab, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, pipeline_name='Stitching', event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        # self.cfg_to_ui()
        # self.reconcile_children_from_view()
        self.subscribe(UiChannelsChanged, self.reconcile_children_from_view)

    def add_channel(self, channel_name, data_type=None):
        """
        Add a channel to the stitching parameters if not already present

        .. warning::

            This only adds the channel to the config it **does not** create the UI elements.
            This will be handled by
        Parameters
        ----------
        channel_name: str
            The name of the channel to add
        data_type: str, optional
        """
        if channel_name in self.channels:
            return
        else:
            if channel_name not in self.view['channels'].keys():
                self._update_value(['channels', channel_name], {})  # Adjuster fills w/ defaults
            self.reconcile_children_from_view()  # ensure param object created

    def reconcile_children_from_view(self, *_):
        desired_channels = tuple(self.view['channels'])  # FIXME: config not hydrated. still has templates
        # add if missing
        self.create_missing_channels_from_view(desired_channels)
        self.prune_obsolete_channels(desired_channels)

    def create_missing_channels_from_view(self, desired_channels: tuple):
        for ch in desired_channels:
            self.materialize_channel(ch)

    def materialize_channel(self, ch):
        if ch not in self.channel_params:
            tw = self.tab.channelsParamsTabWidget
            page = tw.get_channel_widget(ch)
            if page is not None:  # Skip (defer) otherwise
                self.channel_params[ch] = ChannelStitchingParams(self.tab, ch, event_bus=self._bus,
                                                                 get_view=self._get_view,
                                                                 apply_patch=self._apply_patch)

    def prune_obsolete_channels(self, desired_channels: tuple):
        for ch in list(self.channel_params.keys()):
            if ch not in desired_channels:
                self.channel_params.pop(ch)

    def handle_layout_channel_changed(self, channel, layout_channel):
        self.publish(UiLayoutChannelChanged(channel_name=channel, layout_channel=layout_channel))

    def compute_layout(self, channel):
        return self[channel].compute_layout()

    def get_channels_to_run(self):
        # FIXME: do not bind run
        return [channel for channel in self.channels if self[channel].shared.run]

    def set_channels_to_run(self, channels):
        for channel in self.channels:
            status = self[channel].shared.run
            self[channel].shared.run = status or (channel in channels)

    @property
    def params(self):
        return list(self.values())


class ChannelStitchingParams(UiParameterCollection):
    def __init__(self, tab, channel, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, pipeline_name='Stitching', event_bus=event_bus,
                         get_view=get_view, apply_patch=apply_patch)
        self.name = channel
        self.ready = False

        # The sub-params
        self.shared = GeneralChannelStitchingParams(tab, channel, event_bus=event_bus,
                                                    get_view=get_view, apply_patch=apply_patch)
        self.stitching_rigid = None
        self.stitching_wobbly = None

        self.connect()

        self.cfg_to_ui()  # Initial setup

        initial_compute_val = self.compute_layout()
        self._set_sections_visibility(initial_compute_val)
        self._set_sections_enabled(not self.shared.use_existing_layout)

        if initial_compute_val:  # WARNING: only created if compute layout
            self._materialize_section_params()

        self.ready = True

    def _materialize_section_params(self):
        if self.stitching_rigid is None:
            self.stitching_rigid = RigidChannelStitchingParams(self.tab, self.name, event_bus=self._bus,
                                                                get_view=self._get_view, apply_patch=self._apply_patch)
        if self.stitching_wobbly is None:
            self.stitching_wobbly = WobblyChannelStitchingParams(self.tab, self.name, event_bus=self._bus,
                                                                get_view=self._get_view, apply_patch=self._apply_patch)

    def _set_sections_visibility(self, visible: bool):
        shared_tab = self.shared.tab
        shared_tab.rigidParamsGroupBox.setVisible(visible)
        shared_tab.wobblyParamsGroupBox.setVisible(visible)

    def _set_sections_enabled(self, enabled: bool):
        shared_tab = self.shared.tab
        shared_tab.rigidParamsGroupBox.setEnabled(enabled)
        shared_tab.wobblyParamsGroupBox.setEnabled(enabled)

    def connect(self):
        self.subscribe(UiLayoutChannelChanged, self.handle_layout_channel_changed)
        self.subscribe(UiUseExistingLayoutChanged, self.handle_use_existing_layout_changed)

    @param_handler  # FIXME: check
    def handle_layout_channel_changed(self, layout_changed_event: UiLayoutChannelChanged):
        if not self.ready:
            return
        if layout_changed_event.channel_name != self.name:
            return
        compute_layout = self.compute_layout()

        if compute_layout and (self.stitching_rigid is None or self.stitching_wobbly is None):
            self._materialize_section_params()
        self._set_sections_visibility(compute_layout)
        self.shared.tab.useExistingLayoutCheckBox.setVisible(compute_layout)

    @param_handler  # FIXME: check
    def handle_use_existing_layout_changed(self, use_existing_layout_event: UiUseExistingLayoutChanged):
        if use_existing_layout_event.channel_name != self.name:
            return
        if not self.ready:
            return
        if self.compute_layout():
            self._set_sections_enabled(not use_existing_layout_event.use_existing)

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
        layout_channel = (self.shared.layout_channel or
                          (self.shared.view.get('layout_channel') if self.shared.view else None))
        return layout_channel == self.name

    @property
    def params(self):
        return self.shared, self.stitching_rigid, self.stitching_wobbly  # TODO: check if None is a problem


class GeneralChannelStitchingParams(ChannelUiParameter):
    publishes = Publishes(UiLayoutChannelChanged, UiUseExistingLayoutChanged)

    use_npy: bool
    run: bool
    layout_channel: str
    use_existing_layout: bool

    def build_params_dict(self):
        return {
            'use_npy': ParamLink(['use_npy'], self.tab.useNpyCheckBox),
            # 'run': ParamLink(['run'], self.tab.runCheckBox),
            'run': ['run'],
            'layout_channel': ParamLink(
                ['layout_channel'], self.tab.layoutChannelComboBox,
                notify_apply=lambda: self.publish(
                    UiLayoutChannelChanged(channel_name=self.name,
                                           layout_channel=self.layout_channel)),
                # extra_connect=self.refresh_layout_channel_items,
                cast_to_ui=lambda v: v or '',   # show empty when unset
                cast_from_ui = lambda s: None if (s in ('', 'undefined')) else s  ),  #TEST:
            'use_existing_layout': ParamLink(
                ['use_existing_layout'], self.tab.useExistingLayoutCheckBox, default=False,
                missing_ok=True,  # WARNING: only if self.layout_channel != self.name
                present_if=self._existing_layout_relevant,
                notify_apply=lambda: self.publish(
                    UiUseExistingLayoutChanged(channel_name=self.name, use_existing=self.use_existing_layout)))
        }

    def _existing_layout_relevant(self, v: dict) -> bool:
        return v.get('stitching', {}).get('layout_channel') == self.name

    @property
    def run(self):
        return self.view['run']

    @run.setter
    @param_setter
    def run(self, value):
        if self._painting:
            return
        self._update_value(['run'], value)

    @property
    def cfg_subtree(self):
        return ['stitching', 'channels', self.name]    # REFACTOR: section name from config_handler

    def refresh_layout_channel_items(self):
        """Populate and keep in sync the layoutChannelComboBox items."""
        layout_combobox = self.tab.layoutChannelComboBox

        channel_names = list(self._get_view()['stitching']['channels'].keys())

        current = layout_combobox.currentText()
        layout_combobox.blockSignals(True)  # No update during refresh
        layout_combobox.clear()
        layout_combobox.addItems(['undefined'] + channel_names)
        # reselect current value if still valid (else explicit 'undefined')
        layout_combobox.setCurrentText(current if current in channel_names else 'undefined')
        layout_combobox.blockSignals(False)


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

    def build_params_dict(self):
        return {
            'skip': ParamLink(['skip'], self.tab.rigidParamsGroupBox,
                              cast_to_ui=invert, cast_from_ui=invert),
            'x_overlap': VectorLink(['overlap_x'], self.tab.xOverlapSinglet,
                                     disable_globally=True, disabled_value='auto',
                                     show_sentinel_when_off=True),
            'y_overlap': VectorLink(['overlap_y'], self.tab.yOverlapSinglet,
                                     disable_globally=True, disabled_value='auto',
                                     show_sentinel_when_off=True),
            'projection_thickness': VectorLink(['projection_thickness'], self.tab.projectionThicknessDoublet,
                                                disable_globally=False, disabled_value=None,
                                                cast_from_ui=lambda v: (v + [None]) if isinstance(v, list) else [v, None]),
                                                # Projection thickness has 2 axis in UI but 3 in config
            'max_shifts_x': ParamLink(['max_shifts_x'], self.tab.rigidMaxShiftsXDoublet),
            'max_shifts_y': ParamLink(['max_shifts_y'], self.tab.rigidMaxShiftsYDoublet),
            'max_shifts_z': ParamLink(['max_shifts_z'], self.tab.rigidMaxShiftsZDoublet),
            'background_level': ParamLink(['background_level'], self.tab.rigidBackgroundLevel),
            'background_pixels': ParamLink(['background_pixels'], self.tab.rigidBackgroundPixels)
        }

    @property
    def cfg_subtree(self):
        return ['stitching', 'channels', self.name, 'rigid']   # REFACTOR: section name from config_handler

    def set_visible(self, state):
        self.tab.rigidParamsGroupBox.setVisible(state)

    def set_enabled(self, state):
        self.tab.rigidParamsGroupBox.setEnabled(state)


class WobblyChannelStitchingParams(ChannelUiParameter):
    skip: bool
    max_shifts_x: List[int]
    max_shifts_y: List[int]
    max_shifts_z: List[int]
    stack_valid_range: list
    stack_pixel_size: int | None
    slice_valid_range: List[int]
    slice_pixel_size: int  | None

    def build_params_dict(self):
        return {
            'skip': ParamLink(['skip'], self.tab.wobblyParamsGroupBox,
                              cast_to_ui=invert, cast_from_ui=invert),
            'max_shifts_x': ParamLink(['max_shifts_x'], self.tab.wobblyMaxShiftsXDoublet),
            'max_shifts_y': ParamLink(['max_shifts_y'], self.tab.wobblyMaxShiftsYDoublet),
            'max_shifts_z': ParamLink(['max_shifts_z'], self.tab.wobblyMaxShiftsZDoublet),
            'stack_valid_range': VectorLink(['stack_valid_range'], self.tab.wobblyStackValidRangeDoublet),
            'stack_pixel_size': VectorLink(['stack_pixel_size'], self.tab.wobblyStackPixelSizeSinglet),
            'slice_valid_range': VectorLink(['slice_valid_range'], self.tab.wobblySliceRangeDoublet),
            'slice_pixel_size': VectorLink(['slice_pixel_size'], self.tab.wobblySlicePixelSizeSinglet)
        }

    @property
    def cfg_subtree(self):
        return ['stitching', 'channels', self.name, 'wobbly']    # REFACTOR: section name from config_handler

    def set_visible(self, state):
        self.tab.wobblyParamsGroupBox.setVisible(state)

    def set_enabled(self, state):
        self.tab.wobblyParamsGroupBox.setEnabled(state)


class ChannelRegistrationParams(ChannelUiParameter):
    publishes = Publishes(UiAlignWithChanged)

    def __init__(self, tab, channel_name, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, channel_name, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.tab.landmarksWeightsPanel = LandmarksWeightsPanel(self.tab)
        self.tab.landmarksWeightsLayout.addWidget(self.tab.landmarksWeightsPanel)

        self.handle_params_files_changed()  # Initial setup

    def build_params_dict(self):
        return {
            'resample': ParamLink(['resample'], self.tab.resampleCheckBox),
            'resampled_resolution': ParamLink(['resampled_resolution'], self.tab.resampleResolutionTriplet),
            'align_with': ParamLink(['align_with'], self.tab.alignWithComboBox,
                                    cast_to_ui=lambda v: 'None' if v is None else v,
                                    cast_from_ui=lambda v: None if v == 'None' else v,
                                    notify_apply=lambda: self.publish(
                                        UiAlignWithChanged(channel_name=self.name, align_with=self.align_with))),
            'moving_channel': ParamLink(['moving_channel'], self.tab.movingChannelComboBox),
            # WARNING: broken by class replacement
            'params_files': ParamLink(['params_files'], self.tab.paramsFilesListWidget,
                                      object_name='paramsFilesListWidget', scope_root=self.tab),
            # 'landmarks_weights': ['landmarks_weights'],
        }

    @property
    def cfg_subtree(self):
        return ['registration', 'channels', self.name]   # REFACTOR: section name from config_handler

    @property
    def n_registration_files(self) -> int:
        return self.tab.paramsFilesListWidget.count()

    @property
    def use_landmarks_for(self):
        params_to_weights = self.tab.landmarksWeightsPanel.get_params_and_weights()
        return [k for k, v in params_to_weights.items() if v > 0]

    def connect(self):
        if hasattr(self.tab.paramsFilesListWidget, 'itemsChanged'):
            self.tab.paramsFilesListWidget.itemsChanged.connect(self.handle_params_files_changed)
        if hasattr(self.tab, 'landmarksWeightsPanel'):
            self.tab.landmarksWeightsPanel.weightAtChanged.connect(
                self.handle_landmarks_weight_changed)

    @property
    def params_files(self):
        if hasattr(self.tab.paramsFilesListWidget, 'get_items_text'):
            return self.tab.paramsFilesListWidget.get_items_text()
        else:
            return list_widget_getter(self.tab.paramsFilesListWidget)

    @params_files.setter
    def params_files(self, value):
        list_widget_setter(self.tab.paramsFilesListWidget, value)

    @param_handler  # FIXME: check
    def handle_params_files_changed(self):  # TODO: hide by default (unless advanced mode)
        if not isinstance(self.tab.paramsFilesListWidget, FileDropListWidget):
            warnings.warn("ChannelRegistrationParams.handle_params_files_changed"
                          " called but paramsFilesListWidget is not a FileDropListWidget yet"
                          " ensure proper initialization order",
                RuntimeWarning)
            return
        def file_name_to_title(file_name: str) -> str:
            if '.' in file_name:
                base = file_name.split('.')[0]
            else:
                base = file_name
            return snake_to_title(base)
        params_files_with_ext = self.tab.paramsFilesListWidget.get_items_text()
        new_params_files = [file_name_to_title(p) for p in params_files_with_ext]
        # Now match length of weights (prune or pad with 0)
        new_weights = trim_or_pad(self.view['landmarks_weights'], len(new_params_files), pad_value=0)

        self.tab.landmarksWeightsPanel.set_items(new_params_files, new_weights)

        patch = {}
        set_item_recursive(patch, self.cfg_subtree + ['params_files'], params_files_with_ext)
        scaled_weights = [self.scale_landmarks(w) for w in new_weights]
        set_item_recursive(patch, self.cfg_subtree + ['landmarks_weights'], scaled_weights)
        self._apply_patch(patch)

    @param_handler  # FIXME: check
    def handle_landmarks_weight_changed(self, idx: int, value: int):
        scaled = self.scale_landmarks(value)
        new_weights = self.view['landmarks_weights'][:]
        if idx == len(new_weights):
            new_weights.append(scaled)
        elif idx < len(new_weights):
            new_weights[idx] = scaled
        else:
            raise ValueError(f'Index {idx} out of bounds for channel {self.name} '
                             f'landmarks weights {self.view["landmarks_weights"]}')
        self._update_value(['landmarks_weights'], new_weights)

    @staticmethod
    def scale_landmarks(value: int):
        """
        Scale the weight of the landmarks using an exponential function so that
        the ratio between the weights follows a geometric progression with the slider value

        Parameters
        ----------
        value: int
            The value of the slider (between 0 and 100)

        Returns
        -------
        float
            The scaled weight (between 0 and max_ratio)
        """
        if value == 0:
            return 0
        else:
            min_ratio = 1 / 20000
            max_ratio = 200
            return min_ratio * (max_ratio / min_ratio) ** (value / 100)


class SharedRegistrationParams(UiParameter):
    def build_params_dict(self):
        return {
            'plot_channel': ParamLink(None, self.tab.plotChannelComboBox),
            'plot_composite': ParamLink(None, self.tab.plotCompositeCheckBox),
        }


class RegistrationParams(ChannelsUiParameterCollection):  # TEST: does not seem to follow tab click
    publishes = Publishes(UiRequestLandmarksDialog)   # Bind by number because name may change

    def __init__(self, tab, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, pipeline_name='Registration', event_bus=event_bus,
                         get_view=get_view, apply_patch=apply_patch)
        self.atlas_params = AtlasParams(tab, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.shared_params = SharedRegistrationParams(tab, event_bus=event_bus,
                                                     get_view=get_view, apply_patch=apply_patch)

    def add_channel(self, channel_name, data_type=None):
        if channel_name in self.channels:
            return
        else:
            self._update_value(['channels', channel_name], {})  # To be materialized w/ defaults
            channel_params = ChannelRegistrationParams(self.tab, channel_name, event_bus=self._bus,
                                                        get_view=self._get_view, apply_patch=self._apply_patch)
            self[channel_name] = channel_params
            channel_params.tab.selectLandmarksPushButton.clicked.connect(
                lambda ch_idx=channel_params.page_index: self.publish(
                    UiRequestLandmarksDialog(page_index=ch_idx)))

    @property
    def params(self):
        return [self.atlas_params, self.shared_params] + list(self.values())

    def get_channel_name(self, channel_idx):
        return self.tab.channelsParamsTabWidget.tabText(channel_idx)


class AtlasParams(UiParameter):
    publishes = Publishes(UiAtlasIdChanged, UiAtlasStructureTreeIdChanged)

    atlas_id: str
    structure_tree_id: str
    atlas_folder: str

    def __init__(self, tab, *, event_bus: EventBus, get_view=None, apply_patch=None):
        self.atlas_info = ATLAS_NAMES_MAP
        self.cfg_subtree = ['registration', 'atlas']  # REFACTOR: section name from config_handler
        super().__init__(tab, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.update_atlas_resolution()  # WARNING: after connect

    def build_params_dict(self):
        return {
            'atlas_id': ParamLink(['id'], self.tab.atlasIdComboBox,
                                  extra_connect=lambda w, cb: (
                                      w.currentTextChanged.connect(lambda _t: self.update_atlas_resolution()),
                                      lambda: None,  # provide a real disconnector if you have one
                                  )[1], notify_apply=self.notify_atlas_id_changed, ),
            'structure_tree_id': ParamLink(['structure_tree_id'], self.tab.structureTreeIdComboBox,
                                           notify_apply=lambda: self.publish(UiAtlasStructureTreeIdChanged(
                                               tree_id=self.structure_tree_id))),
            'atlas_folder': ParamLink(['align_files_folder'], self.tab.atlasFolderPath, connect=False),
            'atlas_resolution': ParamLink(None, self.tab.atlasResolutionTriplet),  # TODO: check if we bind to cfg here
        }

    def connect(self):
        self.tab.atlasResolutionTriplet.valueChangedConnect(self.handle_atlas_resolution_changed)  # TODO: replace with label

    def update_atlas_resolution(self):
        self.tab.atlasResolutionTriplet.setValue([self.atlas_info[self.atlas_id]['resolution']] * 3)

    @property
    def atlas_base_name(self):
        return self.atlas_info[self.atlas_id]['base_name']

    @param_handler  # FIXME: check
    def handle_atlas_resolution_changed(self, state):
        # WARNING: uses parent config
        view = self._get_view()
        if view is None:
            return
        view = view['registration']  # REFACTOR: use parent param.cfg_subtree
        patch = {}
        for channel in view['channels'].keys():
            set_item_recursive(patch, ['registration', 'channels', channel, 'resampled_resolution'],
                               self.atlas_resolution)
        self._apply_patch(patch)

    def notify_atlas_id_changed(self):  # it's not directly the handler
        self.publish(UiAtlasIdChanged(atlas_base_name=self.atlas_base_name))


class CellMapParams(ChannelsUiParameterCollection):
    def __init__(self, tab, sample_params, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, pipeline_name='CellMap', event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.sample_params = sample_params
        self._perf_params: dict[str, ChannelCellMapPerformanceParams] = {}


    @property
    def params(self):
        return list(self.values()) + list(self._perf_params.values())

    def add_channel(self, channel_name, data_type=None):
        if channel_name not in self.keys():
            self._update_value(['channels', channel_name], {})  # To be materialized w/ defaults
            dtype = np.uint16  # FIXME: read from stitched file
            self[channel_name] = ChannelCellMapParams(self.tab, channel_name, main_params=self,
                                                      event_bus=self._bus, dtype=dtype,
                                                      get_view=self._get_view, apply_patch=self._apply_patch)

    def add_perf_channel(self, channel_name: str):#, page_widget: QWidget):
        """
        Called from the tab once the channel UI page exists.
        """
        if channel_name in self._perf_params:
            return
        self._perf_params[channel_name] = ChannelCellMapPerformanceParams(
            self.tab, channel_name, event_bus=self._bus,
            get_view=self._get_view, apply_patch=self._apply_patch)

    def pop(self, channel_name: str):
        # tear down both algo + perf params
        if channel_name in self:
            self[channel_name].teardown()
            super().pop(channel_name)
        if channel_name in self._perf_params:
            self._perf_params[channel_name].teardown()
            del self._perf_params[channel_name]


class ChannelCellMapParams(ChannelUiParameter, OrthoviewerSlicingMixin):
    background_correction_diameter: List[int]
    maxima_shape: int
    h_max: int | None
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
    crop_x_min: int; crop_x_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_y_min: int; crop_y_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_z_min: int; crop_z_max: int  # TODO: if 99.9 % source put to 100% (None)
    n_detected_cells: int
    n_filtered_cells: int

    def __init__(self, tab, channel, *, main_params, event_bus: EventBus, dtype=np.uint16,
                 get_view=None, apply_patch=None):
        super().__init__(tab, channel, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.main_params = main_params
        self.dtype = dtype
        self.advanced_controls = [self.tab.detectionShapeGroupBox]

    def build_params_dict(self):
        return {
            'background_correction_diameter':
                ParamLink(['detection', 'background_correction', 'diameter'], self.tab.backgroundCorrectionDiameter,
                          cast_to_ui = lambda v: (v[0] if isinstance(v, (list, tuple)) and len(v) else v),
                          cast_from_ui = lambda v: [int(v), int(v)]),
            'maxima_shape': ParamLink(['detection', 'maxima_detection', 'shape'], self.tab.maximaShape),
            'h_max': ParamLink(['detection', 'maxima_detection', 'h_max'], self.tab.hMaxSinglet,
                               default=None, missing_ok=True),
            'detection_threshold': ParamLink(['detection', 'shape_detection', 'threshold'], self.tab.detectionThreshold),
            'cell_filter_size': ParamLink(['cell_filtration', 'thresholds', 'size'], self.tab.cellFilterThresholdSizeDoublet),
            'cell_filter_intensity': VectorLink(['cell_filtration', 'thresholds', 'intensity'],
                                                 self.tab.cellFilterThresholdIntensityDoublet,
                                                 disabled_value=None,  # -1 sentinel maps to None by default
                                                 cast_from_ui=self.cast_max_from_ui),
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

    @param_handler  # FIXME: check
    def handle_advanced_state_changed(self, state):
        super().handle_advanced_state_changed(state)
        self.tab.detectionShapeGroupBox.setVisible(state)  # FIXME: seems redundant

    @property
    def cfg_subtree(self):
        return ['cell_map', 'channels', self.name]  # REFACTOR: section name from config_handler

    @param_handler  # FIXME: check
    def handle_colocalization_compatible_changed(self, state):
        self._update_value(['detection', 'colocalization_compatible'],
                           self.colocalization_compatible)
        if self.colocalization_compatible:
            self.save_shape = True
        self.tab.runCellMapSaveShapeCheckBox.setEnabled(not self.colocalization_compatible)

    def connect(self):
        self.tab.runCellMapColocalizationCompatibleCheckBox.stateChanged.connect(self.handle_colocalization_compatible_changed)

    def cast_max_from_ui(self, cfg_vals):
        """After sentinel mapping back to tokens (None/'auto'), push upper to dtype.max."""
        out = list(cfg_vals)
        if out[-1] in (None, 'auto', -1):  # If sentinel, store as dtype max
            out[-1] = int(np.iinfo(self.dtype).max)
        return out


class ChannelCellMapPerformanceParams(ChannelUiParameter):
    """
    Links per-channel detection.block_processing performance to the UI widget.
    """

    n_processes: int
    size_min: int
    size_max: int
    overlap: int | None

    def build_params_dict(self):
        bp = self.tab.detectionBlockProcessingWidget  # we’ll attach this attr in _setup_channel
        return {
            'n_processes': ParamLink(['detection', 'block_processing', 'n_processes'], bp._nproc_widget),
            'size_min': ParamLink(['detection', 'block_processing', 'size_min'], bp._size_min_spin),
            'size_max': ParamLink(['detection', 'block_processing', 'size_max'], bp._size_max_spin),
            'overlap': ParamLink(['detection', 'block_processing', 'overlap'],
                                 bp._overlap_spin, disabled_value=None, ui_sentinel=-1, enforce_sentinel_min=True),
        }

    @property
    def cfg_subtree(self):
        # Root for this channel’s perf; ParamLinks are already fully qualified, so this is mostly for helpers
        return ['cell_map', 'performance', self.name]


class TractMapParams(ChannelsUiParameterCollection):
    def __init__(self, tab, sample_params, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, pipeline_name='TractMap', event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.sample_params = sample_params
        self.performance_params = TractMapPerformanceParams(tab, event_bus=event_bus,
                                                            get_view=get_view, apply_patch=apply_patch)

    @property
    def params(self):
        return list(self.values()) + [self.performance_params]

    def add_channel(self, channel_name, data_type=None):
        if channel_name not in self.channels:
            self._update_value(['channels', channel_name], {})
            self[channel_name] = ChannelTractMapParams(self.tab, channel_name, main_params=self, event_bus=self._bus,
                                                       get_view=self._get_view, apply_patch=self._apply_patch)


class ChannelTractMapParams(ChannelUiParameter, OrthoviewerSlicingMixin):
    clipping_decimation_ratio: int
    clipping_percents: List[float]
    clip_range: List[int]
    crop_x_min: int; crop_x_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_y_min: int; crop_y_max: int  # TODO: if 99.9 % source put to 100% (None)
    crop_z_min: int; crop_z_max: int  # TODO: if 99.9 % source put to 100% (None)
    display_decimation_ratio: int  # For the "cells.feather" file
    voxelization_radii: List[int]
    binarize: bool
    extract_coordinates: bool
    transform_coordinates: bool
    label_coordinates: bool
    voxelize: bool
    export_df: bool

    def __init__(self, tab, channel, *, main_params, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, channel, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.main_params = main_params

    def build_params_dict(self):
        return {
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

    @property
    def cfg_subtree(self):
        return ['tract_map', 'channels', self.name]  # REFACTOR: section name from config_handler


class TractMapPerformanceParams(UiParameter):
    """
    tract_map.performance.<section> → performanceGroupBox widgets.
    Shared (not per-channel) in v3.1.
    """
    cfg_subtree = ['tract_map', 'performance']

    def build_params_dict(self):
        t = self.tab  # shorthand
        return {
            # binarization.performance
            'binarization_n_processes': ParamLink(['performance', 'binarization', 'n_processes'],
                                                  t.binarizationPerf),

            # where.performance
            'where_n_processes': ParamLink(['performance', 'where', 'n_processes'],
                                           t.wherePerf),

            # transform.block_processing
            'transform_size_min': ParamLink(['performance', 'transform', 'block_processing', 'size_min'],
                                            t.transformBlock._size_min_spin),
            'transform_size_max': ParamLink(['performance', 'transform', 'block_processing', 'size_max'],
                                            t.transformBlock._size_max_spin),
            'transform_overlap': ParamLink(['performance', 'transform', 'block_processing', 'overlap'],
                                           t.transformBlock._overlap_spin),
            'transform_n_processes': ParamLink(['performance', 'transform', 'block_processing', 'n_processes'],
                                               t.transformBlock._nproc_widget),

            # label.block_processing
            'label_size_min': ParamLink(['performance', 'label', 'block_processing', 'size_min'],
                                        t.labelBlock._size_min_spin),
            'label_size_max': ParamLink(['performance', 'label', 'block_processing', 'size_max'],
                                        t.labelBlock._size_max_spin),
            'label_overlap': ParamLink(['performance', 'label', 'block_processing', 'overlap'],
                                       t.labelBlock._overlap_spin),
            'label_n_processes': ParamLink(['performance', 'label', 'block_processing', 'n_processes'],
                t.labelBlock._nproc_widget),
        }


class ColocalizationParams(ChannelsUiParameterCollection):
    def __init__(self, tab, sample_params, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, pipeline_name='Colocalization', event_bus=event_bus,
                         get_view=get_view, apply_patch=apply_patch)
        self.sample_params = sample_params

    @property
    def params(self):
        return self.values()

    def add_channel(self, channel_name, data_type=None):
        if isinstance(channel_name, (tuple, list)):
            channel_name = '-'.join(channel_name)
        if channel_name in self.keys():
            return
        else:
            self._update_value(['channels', channel_name], {})  # Will be materialized w/ defaults
            self[channel_name] = ChannelColocalizationParams(self.tab, channel_name, main_params=self,
                                                             event_bus=self._bus, get_view=self._get_view,
                                                             apply_patch=self._apply_patch)


class ChannelColocalizationParams(ChannelUiParameter):
    particle_diameter: int
    n_processes: int
    max_particle_distance: float
    relative_overlap_threshold: float
    voxel_number_overlap_threshold: int

    def __init__(self, tab, channel, *, main_params, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, channel, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.main_params = main_params

    def build_params_dict(self):
        return {
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

    @property
    def cfg_subtree(self):
        return ['colocalization', 'channels', self.name]   # REFACTOR: section name from config_handler


class SharedVesselBinarizationParams(UiParameter):
    fill_combined: bool
    plot_step_1: str
    plot_step_2: str
    plot_channel_1: str
    plot_channel_2: str

    def build_params_dict(self):
        return {
            'fill_combined': ParamLink(['combined', 'binary_fill'], self.tab.binarizationConbineBinaryFillingCheckBox),
            'plot_step_1': ParamLink(None, self.tab.binarizationPlotStep1ComboBox),
            'plot_step_2': ParamLink(None, self.tab.binarizationPlotStep2ComboBox),
            'plot_channel_1': ParamLink(None, self.tab.binarizationPlotChannel1ComboBox),
            'plot_channel_2': ParamLink(None, self.tab.binarizationPlotChannel2ComboBox),
        }


class VesselParams(ChannelsUiParameterCollection):

    def __init__(self, tab, sample_params, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, pipeline_name='TubeMap', event_bus=event_bus,
                         get_view=get_view, apply_patch=apply_patch)
        # self.sample_params = sample_params  # TODO: check if required
        # self.preprocessing_params = preprocessing_params  # TODO: check if required
        self.shared_binarization_params = SharedVesselBinarizationParams(tab, event_bus=event_bus,
                                                                       get_view=get_view,
                                                                       apply_patch=apply_patch)
        self.graph_params = VesselGraphParams(tab, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.visualization_params = VesselVisualizationParams(tab, sample_params=sample_params, event_bus=event_bus,
                                                                get_view=get_view, apply_patch=apply_patch)

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
            self._update_value(['binarization', channel_name], {})  # Will be materialized w/ defaults)
            self[channel_name] = VesselBinarizationParams(self.tab, channel_name, event_bus=self._bus,
                                                            get_view=self._get_view, apply_patch=self._apply_patch)

            if data_type == 'arteries':
                self.graph_params.use_arteries = True


class VesselBinarizationParams(ChannelUiParameter):
    run_binarization: bool
    binarization_clip_range: List[int]
    binarization_threshold: int
    run_smoothing: bool
    run_binary_filling: bool
    run_deep_filling: bool

    def build_params_dict(self):
        return {
            # FIXME: add tabs to UI with matching control names
            'run_binarization': ParamLink(['binarize', 'run'], self.tab.runBinarizationCheckBox),
            'binarization_clip_range': ParamLink(['binarize', 'clip_range'], self.tab.binarizationClipRangeDoublet),
            'binarization_threshold': ParamLink(['binarize','threshold'],
                                                self.tab.binarizationThresholdSpinBox,
                                                disabled_value=None,
                                                ui_sentinel=-1,
                                                enforce_sentinel_min=True,
                                                # cast_to_ui=self.sanitize_nones,  # REFACTOR: use VectorLink instead ?
                                                # cast_from_ui=self.sanitize_neg_one
                                                ),
            'run_smoothing': ParamLink(['smooth', 'run'], self.tab.binarizationSmoothingCheckBox),
            'run_binary_filling': ParamLink(['binary_fill', 'run'], self.tab.binarizationBinaryFillingCheckBox),
            'run_deep_filling': ParamLink(['deep_fill', 'run'], self.tab.binarizationDeepFillingCheckBox),
        }
        # self.tab.binarizationControlsGroupBox.setTitle(channel_name)

    @property
    def cfg_subtree(self):
        return ['vasculature', 'binarization', self.name]   # REFACTOR: section name from config_handler

    @property
    def n_steps(self):
        n_steps = self.run_binarization
        n_steps += self.run_smoothing or self.run_binary_filling
        n_steps += self.run_deep_filling
        return n_steps

class VesselGraphParams(UiParameter):
    publishes = Publishes(UiVesselGraphFiltersChanged)

    cfg_subtree = ['vasculature']

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

    def __init__(self, tab, *, event_bus: EventBus, get_view=None, apply_patch=None):
        super().__init__(tab, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.filter_params = []

    def build_params_dict(self):
        return {
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

    def add_graph_filter_params(self, widget, graph):
        self.filter_params.append(GraphFilterParams(main_params=self, widget=widget,
                                                    graph=graph, event_bus=self._bus,
                                                    get_view=self._get_view, apply_patch=self._apply_patch))
        self.filtersChanged.emit()

    @property
    def n_filters(self):
        return len(self.filter_params)

    def compute_filter_suffix(self):
        suffix = '_'.join([f'{f.property_name}_{f.get_property_value()}' for f in self.filter_params])
        # TODO: consider:
        #   parts = []
        #   for i, f in enumerate(fs):
        #       parts.append(f"{f.property_name}_{f.get_property_value()}")
        #       op = f.combine_operator_name
        #       if op and i < len(fs)-1:
        #           parts.append(op)
        #   suffix = '_'.join(parts)
        return suffix


class GraphFilterParams(UiParameter):  # FIXME: do we really pass the graph as argument or just the prop names/types ?
    def __init__(self, *, main_params, widget, graph, event_bus: EventBus, get_view=None, apply_patch=None):
        self.main_params = main_params
        super().__init__(widget, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)  # self.tab will be based on filter_name
        self.index = int(widget.objectName().split('_')[-1])
        self.layout = self.tab.parent().findChild(QVBoxLayout, 'filterParamsVerticalLayout')

        self.graph = graph
        self.update_properties()
        self.connect()

    def connect(self):
        self.tab.vertexFilterRadioButton.toggled.connect(self.update_properties)
        self.tab.graphFilterPropertyNameComboBox.currentTextChanged.connect(self.handle_property_name_changed)

    def update_properties(self):
        if self.filter_type == 'vertex':
            properties = self.graph._base.vertex_properties
        else:
            properties = self.graph._base.edge_properties
        properties_names = list(properties.keys())  # Guarantee order
        dtypes = [properties[prop_name].python_value_type() for prop_name in properties_names]
        dtype_names = []
        for dtype in dtypes:
            if isinstance(dtype, tuple):
                dtype_names.append([t.__name__ for t in dtype][-1])  # if List[type] -> type
            else:
                dtype_names.append(dtype.__name__)

        self.tab.graphFilterPropertyNameComboBox.clear()
        for prop_name, dtype in zip(properties_names + ['degrees'], dtype_names + ['int']):
            self.tab.graphFilterPropertyNameComboBox.addItem(prop_name, userData=dtype)

        self.handle_property_name_changed()  # Set default value for the first property

        self.main_params.filtersChanged.emit()  # Notify that properties changed

    @property
    def cfg_subtree(self):
        return ['vasculature']  # REFACTOR: section name from config_handler

    @property
    def filter_type(self):
        return 'vertex' if self.tab.vertexFilterRadioButton.isChecked() else 'edge'

    @property
    def property_name(self):
        return self.tab.graphFilterPropertyNameComboBox.currentText()

    @property
    def current_dtype(self):
        return self.tab.graphFilterPropertyNameComboBox.currentData()

    def get_default_property_value(self):
        property_dtype = self.current_dtype
        if property_dtype == 'bool':
            return False
        elif property_dtype in ('int', 'float'):
            return 0
        elif property_dtype == 'str':
            return ''
        else:
            raise ClearMapValueError(f'Unknown property type: {property_dtype}')

    def get_property_value(self):
        property_dtype = self.current_dtype
        value_widget = self.tab.graphFilterPropertyValueWidget
        if property_dtype == 'bool':
            return value_widget.isChecked()
        elif property_dtype in ('int', 'float'):
            return value_widget.getValue()
            # return self.tab.graphFilterPropertyValueDoublet.getValue()
        elif property_dtype == 'str':
            return value_widget.text()
        else:
            raise ClearMapValueError(f'Unknown property type: {property_dtype}')

    @property
    def combine_operator_name(self):
        """
        Read the action from the following combine button
        Returns
        -------

        """
        p2 = self.layout.parent().parent()
        and_button = p2.findChild(QRadioButton, f'filter_{self.index}_and_btn')
        # or_button = p2.findChild(QRadioButton, f'filter_{self.index}_or_btn')
        if not and_button:  # Last filter
            return None
        return 'and' if and_button.isChecked() else 'or'

    def suffix(self):
        suffix = f'{self.property_name}_{self.get_property_value()}'
        combine_action = self.combine_operator_name
        if combine_action is not None:
            suffix += f'_{combine_action}'
        return suffix

    @param_handler  # FIXME: check
    def handle_property_name_changed(self):
        property_dtype = self.current_dtype
        value = self.get_default_property_value()
        controls_layout = self.tab.filterControlsGridLayout
        if property_dtype == 'bool':
            widget = QCheckBox(self.tab)
            widget.setChecked(value)
        elif property_dtype in ('int', 'float'):
            if property_dtype == 'int':
                widget_class = QSpinBox
            else:
                widget_class = QDoubleSpinBox
            widget = self.__create_range_ctrl(widget_class, signed=False)  # FIXME: decide on signed

            widget.setValue(value)
        elif property_dtype == 'str':
            widget = QLineEdit(self.tab)
            widget.setText(value)
        else:
            raise ClearMapValueError(f'Unknown property type: {property_dtype}')
        self.tab.graphFilterPropertyValueWidget = replace_widget(
            self.tab.graphFilterPropertyValueWidget, widget, layout=controls_layout)

        self.main_params.filtersChanged.emit()

    def __create_range_ctrl(self, widget_class, signed=True):
        """
        Create a doublet control for a range of values, e.g. for min and max values.

        .. warning::
            QSpinBox takes only signed 32 bit integers (hence a range of -2**31 to 2**31-1)

        Parameters
        ----------
        widget_class
        signed

        Returns
        -------

        """
        range_abs = 2 ** 31
        if signed:
            range_min, range_max = -range_abs, range_abs - 1
        else:
            range_min, range_max = 0, range_abs - 1
        widget = QFrame(self.tab)
        widget.setObjectName('graphFilterPropertyValueDoublet')
        frame_layout = QHBoxLayout(widget)
        min_ctrl = widget_class(self.tab)
        min_ctrl.setMaximumWidth(60)  # FIXME: resize to content instead
        min_ctrl.setObjectName(f'graphFilterPropertyValueSpinBox_1')  # For doublet ctrl behaviour (needs sorting)
        min_ctrl.setRange(range_min, range_max)
        frame_layout.addWidget(min_ctrl)
        lbl = QLabel(self.tab)
        lbl.setText('to')
        frame_layout.addWidget(lbl)
        max_ctrl = widget_class(self.tab)
        max_ctrl.setMaximumWidth(60)
        max_ctrl.setObjectName(f'graphFilterPropertyValueSpinBox_2')  # For doublet ctrl behaviour (needs sorting)
        max_ctrl.setRange(range_min, range_max)
        frame_layout.addWidget(max_ctrl)
        recursive_patch_compound_boxes(self.tab)
        return widget


class VesselVisualizationParams(UiParameter, OrthoviewerSlicingMixin):
    crop_x_min: int; crop_x_max: int
    crop_y_min: int; crop_y_max: int
    crop_z_min: int; crop_z_max: int
    graph_step: str
    plot_type: str
    voxelization_size: List[int]
    vertex_degrees: str
    weight_by_radius: bool

    pipeline = 'TubeMap'

    def __init__(self, tab, *, sample_params=None, event_bus: EventBus, get_view=None, apply_patch=None):
        self.cfg_subtree = ['vasculature', 'visualization']  # REFACTOR: section name from config_handler
        super().__init__(tab, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.structure_id = None
        self.sample_params = sample_params

    def build_params_dict(self):
        return {  # TODO: if 99.9 % source put to 100% (None)
            'crop_x_min': ParamLink(['slicing', 'dim_0', 0], self.tab.graphConstructionSlicerXRangeMin),
            'crop_x_max': ParamLink(['slicing', 'dim_0', 1], self.tab.graphConstructionSlicerXRangeMax),
            'crop_y_min': ParamLink(['slicing', 'dim_1', 0], self.tab.graphConstructionSlicerYRangeMin),
            'crop_y_max': ParamLink(['slicing', 'dim_1', 1], self.tab.graphConstructionSlicerYRangeMax),
            'crop_z_min': ParamLink(['slicing', 'dim_2', 0], self.tab.graphConstructionSlicerZRangeMin),
            'crop_z_max': ParamLink(['slicing', 'dim_2', 1], self.tab.graphConstructionSlicerZRangeMax),
            'graph_step': ParamLink(None, self.tab.graphSlicerStepComboBox, connect=False),
            'plot_type': ParamLink(None, self.tab.graphPlotTypeComboBox, connect=False),
            'voxelization_size': ParamLink(['voxelization', 'size'], self.tab.vasculatureVoxelizationRadiusTriplet),
            'weight_by_radius': ParamLink(None, self.tab.voxelizationWeightByRadiusCheckBox, connect=False)
        }

    def set_structure_id(self, structure_widget):
        self.structure_id = int(structure_widget.text(1))


class PreferencesParams(UiParameter):
    verbosity: str  # == loglevel
    start_folder: str
    start_full_screen: bool
    lut: str
    font_size: int
    pattern_finder_min_n_files: int
    three_d_plot_bg: str

    cfg_subtree = ['machine']

    def build_params_dict(self):
        return {
            "verbosity": ParamLink(["verbosity"], self.tab.verbosityComboBox,
                                   cast_to_ui=self.str_to_capitalize, cast_from_ui=self.str_to_lower, default="info"),
            "lut": ParamLink(["default_lut"], self.tab.lutComboBox,
                             cast_to_ui=self.str_to_capitalize, cast_from_ui=self.str_to_lower, default="viridis"),
            "font_size": ParamLink(["font_size"], self.tab.fontSizeSpinBox,),
            "pattern_finder_min_n_files": ParamLink(["pattern_finder_min_n_files"],
                                                    self.tab.patternFinderMinFilesSpinBox, default=2),
            "three_d_plot_bg": ParamLink(["three_d_plot_bg"], self.tab.threeDPlotsBackgroundComboBox,
                                         cast_to_ui=self.str_to_capitalize, cast_from_ui=self.str_to_lower,
                                         default="#000000"),
            "start_folder": ParamLink(["start_folder"], self.tab.startFolderLineEdit,
                                      cast_to_ui=self.sanitize_path_read, cast_from_ui=self.sanitize_path_write),
            "start_full_screen": ParamLink(["start_full_screen"], self.tab.startFullScreenCheckBox)
        }


class BatchParameters(UiParameter):
    publishes = Publishes(UiBatchResultsFolderChanged, UiBatchGroupsChanged)

    results_folder: str
    groups: dict[str, list[str]]

    cfg_subtree = None  # Must be set in subclass

    def __init__(self, tab, *, event_bus: EventBus, preferences=None, get_view=None, apply_patch=None):
        super().__init__(tab, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.group_concatenator = ' vs '
        self.preferences = preferences

        # TODO: check if I need that ?
        if not hasattr(self.tab, "sampleFoldersToolBox") or self.tab.sampleFoldersToolBox is None:
            self.tab.sampleFoldersToolBox = QToolBox(parent=self.tab)
            self.tab.sampleFoldersPageLayout.addWidget(self.tab.sampleFoldersToolBox, 3, 0)

        self.groups_adapter = GroupsWidgetAdapter(
            toolbox=self.tab.sampleFoldersToolBox,
            container_layout=self.tab.sampleFoldersPageLayout,
            add_btn=self.tab.addGroupPushButton,
            remove_btn=self.tab.removeGroupPushButton,
            start_folder_getter=lambda: (self.preferences.start_folder if self.preferences else "")
        )
        self.params_dict = {  # FIXME: make work with UiParameter auto mechanism
            'results_folder': ParamLink(['paths', 'results_folder'], self.tab.resultsFolderLineEdit,
                                        notify_apply=lambda: self.publish(UiBatchResultsFolderChanged(self.results_folder))),
            'groups': ParamLink(['groups'], self.groups_adapter,
                                notify_apply=lambda: self.publish(UiBatchGroupsChanged(self.groups)))
        }
        self.connect_simple_widgets()

    def cfg_to_ui(self):
        self.groups = self.view['groups']
        results_folder = self.view['paths'].get('results_folder', '')
        if results_folder:
            self.results_folder = results_folder

    def remove_current_group(self):  # FIXME: not bound
        idx, removed_name = self.groups_adapter.remove_current_page()
        if idx < 0: return
        self.groups = self.groups_adapter.get_value()
        self.publish(UiBatchGroupsChanged(self.groups))

    @property
    def group_names(self) -> list[str]:
        return self.groups_adapter.group_names

    @group_names.setter
    @param_setter
    def group_names(self, names: list[str]) -> None:
        self.groups_adapter.group_names = names
        self.publish(UiBatchGroupsChanged(self.groups))

    @property
    def n_groups(self) -> int:
        return self.groups_adapter.group_count()

    def set_paths(self, gp_idx: int, paths: list[str]) -> None:
        self.groups_adapter.set_paths(gp_idx, paths)
        # fire one bus event; ParamLink will pick it up on next apply
        self.publish(UiBatchGroupsChanged(self.groups))

    def get_paths(self, gp_idx: int) -> list[str]:
        return self.groups_adapter.get_paths(gp_idx)

    def get_all_paths(self) -> list[str]:
        return self.groups_adapter.get_all_paths()


class GroupAnalysisParams(BatchParameters):
    """
    Essentially batch parameters with comparisons
    """
    # plot_channel: str
    compute_sd_and_effect_size: bool
    density_suffix: str
    pipeline: str
    cfg_subtree = ['group_analysis']  # FIXME: not treated as pipeline because not collection

    def __init__(self, tab, *, event_bus: EventBus, preferences=None, get_view=None, apply_patch=None):
        super().__init__(tab, preferences=preferences, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.extend_params_dict({
            # 'plot_channel': ParamLink(None, self.tab.plotChannelComboBox),
            'compute_sd_and_effect_size': ParamLink(None, self.tab.computeSdAndEffectSizeCheckBox),
            'density_suffix': ParamLink(None, self.tab.densitySuffixTextFilterLineEdit),
            'pipeline': ParamLink(['pipeline'], self.tab.batchPipelineNameComboBox)
        })

        self._cmp_model = ComparisonsModel(sep=self.group_concatenator)
        self._cmp_ui = ComparisonsWidgetAdapter(self.tab.comparisonsVerticalLayout,
                                                groups_sep=self.group_concatenator)

        # FIXME: do I want this ?
        self._channels_provider: Optional[Callable[[], list[str]]] = None
        self._on_plot_group: Optional[Callable[[str], None]] = None

        self.plot_channel = ''

        # FIXME: do I remove this ?
        self.subscribe(UiBatchGroupsChanged, self._rebuild_comparisons)

    def _ui_to_cfg(self):
        self._update_value(['comparisons'], self._comparisons_dict())

    def cfg_to_ui(self):
        super().cfg_to_ui()
        if 'comparisons' not in self.view:  #  not set
            return
        persisted = [tuple(v) for v in self.view['comparisons'].values()]
        self._rebuild_comparisons_core(preselected=persisted)

    def set_pipelines(self, pipelines: list[str]):
        self.tab.batchPipelineNameComboBox.clear()
        self.tab.batchPipelineNameComboBox.addItems(pipelines)

    def _comparisons_dict(self) -> dict[str, list[str]]:
        comp_names = string.ascii_lowercase  # Just a,b,c,...,z
        comparisons = self.selected_comparisons
        comps = {name: list(pair) for name, pair in zip(comp_names, comparisons)}
        return comps

    # injection points from the tab
    def set_channels_provider(self, provider: Callable[[], list[str]]):
        self._channels_provider = provider

    def set_on_plot_group(self, handler: Callable[[str], None]):
        self._on_plot_group = handler

    def _rebuild_comparisons(self, event: Optional[UiBatchGroupsChanged] = None):
        self._rebuild_comparisons_core(preselected=self._cmp_model.selected)

    def _rebuild_comparisons_core(self, *, preselected: Optional[list[Pair]] = None):
        self._cmp_model.group_names = list(self.group_names)
        self._cmp_model.selected = preselected or []
        channels = self._channels_provider() if callable(self._channels_provider) else []

        def _on_channel_changed(ch: str):
            self.plot_channel = ch

        self._cmp_ui.rebuild(self._cmp_model,
                             on_plot_group=(self._on_plot_group or (lambda _g: None)),
                             channels=channels, on_channel_changed=_on_channel_changed,
                             preselected_comparisons=self._cmp_model.selected,)
        if channels and not self.plot_channel:
            self.plot_channel = channels[0]

    @property
    def comparisons(self) -> list[Pair]:
        return self._cmp_model.all_pairs()

    @property
    def selected_comparisons(self) -> list[Pair]:
        return self._cmp_ui.selected_pairs(self._cmp_model)


class BatchProcessingParams(BatchParameters):
    """
    Essentially BatchParameters with processing steps
    """
    align: bool
    count_cells: bool
    run_vasculature: bool

    cfg_subtree = ['batch_processing']  # FIXME: not treated as pipeline because not collection

    def __init__(self, tab, *, event_bus: EventBus, preferences=None, get_view=None, apply_patch=None):
        super().__init__(tab, preferences=preferences, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)
        self.extend_params_dict({
            'align': ParamLink(None, self.tab.batchAlignCheckBox),
            'count_cells': ParamLink(None, self.tab.batchCountCellsCheckBox),
            'run_vasculature': ParamLink(None, self.tab.batchVasculatureCheckBox)
        })
