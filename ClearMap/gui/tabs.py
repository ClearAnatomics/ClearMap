# -*- coding: utf-8 -*-
"""
tabs
====

The different tabs that correspond to different functionalities of the GUI.

All the classes in this module are subclasses (direct or indirect) of the `GenericTab` class which
provides the basic structure and methods that are common to all tabs.

Presentation
------------
Each **tab** is responsible for a specific part of the processing pipeline and has its own UI elements.
It is composed of:

- A `ui` object which is the result of the construction of a `.ui` file that defines the layout and the widgets.
- A `sample_manager` object that handles the sample data.
- A `sample_params` object that links the UI to the sample configuration file.
- A `params` object that links the UI to the configuration file.
(For the `SampleInfoTab`, this is the `SampleParameters` object.)
- A `name` that is used to identify the tab in the GUI.
- A `processing_type` that identifies the type of tab. It can be one of (None, 'pre', 'post', 'batch').
- A `progress_watcher` object that tracks the progress of the computation.

Optionally, for `Pipeline` tabs (`PreProcessingTab` and `PostProcessingTab`),
a `processor` object is used to handle the processing steps and computation.
For `PostProcessing` tabs, the pre_processor objects (stitcher and aligner) can also be passed.

Tab setup
---------

Inherited setup methods
***********************

- `init_ui`: Sets up the UI elements. This is already implemented in the `GenericTab` class.

Setup methods to be implemented in the concrete tab classes
***********************************************************

- `setup`: Sets up the UI elements and the signal/slot connections (it should call `init_ui` at the beginning).
- `set_params`: Sets the `params` object which links the UI and the configuration file.
- `set_progress_watcher`: Sets up the watcher object that will handle the progress in the computation for this tab.

`PipelineTab` additional setup methods
**************************************

- `setup_workers`: Sets up the worker (Processor) which handles the computations associated with this tab.
Called in `set_params` and also when the sample config is applied.

`PostProcessingTab` additional setup methods
********************************************

- `finalise_workers_setup`: Sets up the worker (Processor).
Called when the tab is selected to ensure the workers are fully set up.

Tabs with channels
******************

Some tabs can have sub UIs for the different channels that are processed.
In this case, the `add_channel_tab` method is called to add a new channel tab to the UI.
This method is implemented in the `GenericTab` class. To control its behavior,
the following methods can be implemented in the concrete tab classes:

- `_set_channel_config` (optional): Sets the configuration for the channel.
- `_setup_channel` (optional): Additional setup for the ui (before binding).
- `_bind_channel` (Must be implemented): Binds the signal/slots of the UI elements
for `channel` which are not automatically set through the params object attribute.

- `add_channel_tab` is called in:

  - `set_params`
  - when (+) button is clicked (SampleInfoTab)

Setup order
***********

Typical calling sequence is:

- `tab.setup` (calls `tab.init_ui`)
- `tab.set_params`
    - sets sample_params if not SampleInfoTab
    - calls `tab.set_pre_processors` for PostProcessingTab instances
    - calls `tab.__set_params`
    - calls `tab._read_configs`
    - calls `tab._fix_config` if loaded from defaults
    - calls `tab.setup_workers`
    - calls `tab._create_channels`
    - calls `tab._load_config_to_gui`  (called after fix_config except for BatchTab)
    - calls `tab._bind_params_signals`

- `tab.finalise_workers_setup`  # Only for PostProcessingTab

Processing steps
----------------

The rest of the methods mainly handle the signal/slot connections for the buttons (as the
other controls are handled by the `params` object) and the processing steps.

====================
"""

import functools
import itertools
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QApplication, QLabel, QButtonGroup
import pyqtgraph as pg
from natsort import natsorted
from pyqtgraph import PlotWidget

import mpld3

from ClearMap.IO.assets_constants import DATA_CONTENT_TYPES, EXTENSIONS
from ClearMap.processors.colocalization import ColocalizationProcessor
from ClearMap.processors.tract_map import TractMapProcessor

app = QApplication.instance()
if app is not None and app.applicationName() == 'ClearMap':
    from PyQt5.QtWebEngineWidgets import QWebEngineView
from qdarkstyle import DarkPalette

from ClearMap.config.atlas import ATLAS_NAMES_MAP, STRUCTURE_TREE_NAMES_MAP

import ClearMap.IO.IO as clm_io
from ClearMap.Analysis.Statistics.group_statistics import make_summary, density_files_are_comparable, compare_groups, \
    check_ids_are_unique
from ClearMap.Utils.exceptions import ClearMapVRamException, GroupStatsError, MissingRequirementException

from ClearMap.gui.dialogs import option_dialog
from ClearMap.gui.interfaces import GenericTab, PostProcessingTab, PreProcessingTab, BatchTab, PipelineTab
from ClearMap.gui.widgets import (PatternDialog, DataFrameWidget, LandmarksSelectorDialog,
                                  CheckableListWidget, FileDropListWidget, ExtendableTabWidget)
from ClearMap.gui.gui_utils import format_long_nb, np_to_qpixmap, replace_widget, unique_connect, get_widget
from ClearMap.gui.params import (VesselParams, SampleParameters, StitchingParams,
                                 CellMapParams, GroupAnalysisParams, BatchProcessingParams, RegistrationParams,
                                 TractMapParams, ColocalizationParams)
from ClearMap.Visualization.Matplotlib.PlotUtils import plot_sample_stats_histogram, plot_volcano
from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
from ClearMap.Visualization.Qt import Plot3d as plot_3d

from ClearMap.processors.sample_preparation import (SampleManager, StitchingProcessor,
                                                    RegistrationProcessor, init_sample_manager_and_processors)
from ClearMap.processors.cell_map import CellDetector
from ClearMap.processors.tube_map import BinaryVesselProcessor, VesselGraphProcessor
from ClearMap.processors.batch_process import BatchProcessor

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


class SampleInfoTab(GenericTab):
    """
    The tab manager to define the parameters of the sample
    This refers to values that are intrinsic to the sample and the acquisition
    like resolution, orientation ...
    """
    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'sample_tab', tab_idx)
        self.sample_manager = sample_manager

        self.channels_ui_name = 'channel_params'
        self.with_add_btn = True
        self.names_map = []
        self.detached = False  # WARNING: To avoid calling update when channels are setup by
                               #   the wizzard

    def _set_params(self):
        self.params = SampleParameters(self.ui, self.main_window.src_folder)
        # REFACTOR: trigger signal to update the UI

    def _setup_workers(self):  # WARNING: a bit far fetched but necessary to have a consistent setup
        self.sample_manager.setup(src_dir=self.main_window.src_folder, watcher=self.main_window.progress_watcher)
        self.sample_manager.config = self.params.config  # WARNING: hacky way to force shared reference

    def _bind_params_signals(self):
        self.params.convertToClearMapFormat.connect(self.convert_to_clearmap_format)
        self.params.plotMiniBrain.connect(self.plot_mini_brain)
        self.params.plotAtlas.connect(self.display_atlas)
        self.params.channelNameChanged.connect(self.update_workspace)
        self.params.channelsChanged.connect(self.update_pipelines)
        self.params.orientationChanged.connect(self.update_atlas)
        self.params.cropChanged.connect(self.update_atlas)

    def _get_channels(self):
        return self.sample_manager.channels  # or list(self.params.config['channels'].keys())

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        self.ui.channelsParamsTabWidget.addTabClicked.connect(self.add_channel_tab)

        self.ui.srcFolderBtn.clicked.connect(self.main_window.prompt_experiment_folder)

        self.ui.launchPatternWizzardPushButton.clicked.connect(self.launch_pattern_wizard)
        self.ui.updateWorkspacePushButton.clicked.connect(self.update_workspace)

        self.ui.removeCurrentChannelToolButton.clicked.connect(self.remove_current_channel)

    def _bind_channel(self, page_widget, channel):
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute
        """
        if len(self.names_map) < self.ui.channelsParamsTabWidget.last_real_tab_idx:
            self.names_map.append([channel, channel])
        else:
            self.names_map[self.ui.channelsParamsTabWidget.last_real_tab_idx - 1] = [channel, channel]
        content_types = natsorted(list(set(DATA_CONTENT_TYPES)))
        page_widget.dataTypeComboBox.addItems(content_types)
        page_widget.dataTypeComboBox.setCurrentText('undefined')
        page_widget.extensionComboBox.addItems(EXTENSIONS['image'])
        self.params[channel].nameChanged.connect(
            functools.partial(self.invalidate_pipelines, channel=channel, name_changed=True))
        page_widget.dataTypeComboBox.currentTextChanged.connect(
            functools.partial(self.invalidate_pipelines, channel=channel, dtype_changed=True))

    def invalidate_pipelines(self, channel, name_changed=False, dtype_changed=False):
        """
        This should be called whenever the channels change.
        Either
        Parameters
        ----------
        channel
        name_changed
        dtype_changed

        Returns
        -------

        """
        if channel not in self.sample_manager.channels:
            pass # Remove from all tabs
        else:
            if dtype_changed:
                self.update_pipelines()
                # for each pipeline, check if that channel makes sense. If yes, keep or add otherwise, remove

    def save_cfg(self):  # REFACTOR: use this instead of direct calls to ui_to_cfg
        """Save the config to file"""
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Sample config saved')

    def rename_channels(self, old_name='', new_name=''):
        if old_name:
            names_map = [[old_name, new_name]]
        else:
            for i, name in enumerate(self.ui.channelsParamsTabWidget.get_channels_names()):
                try:
                    self.names_map[i][1] = name  # Update to new name
                except IndexError:
                    self.names_map.append((name, name))  # FIXME: what is the old_name in that case ??
            names_map = self.names_map
        self.sample_manager.rename_channels(names_map)  # REFACTOR: delegate to self.params instead
        # FIXME: rename mini_brains in aligner
        # FIXME: rename annotators in aligner
        self.params.ui_to_cfg()  # The channels now match so we can update the other params
        self.sample_manager.update_workspace()

        if old_name:
            self.names_map[self.ui.channelsParamsTabWidget.currentIndex] = [[new_name, new_name]]
        else:
            self.names_map = []

    def remove_current_channel(self):
        """Remove the current channel from the sample"""
        self.remove_channel(self.ui.channelsParamsTabWidget.current_channel())

    def remove_channel(self, channel):
        self.params.pop(channel)

    def update_workspace(self, old_name='', new_name=''):  # Necessary intermediate because at the beginning sample_manager is not set
        self.rename_channels(old_name, new_name)  # Includes call to sample_manager.update_workspace
        self._setup_workers()  # Load the stitching and registration processors
        # WARNING: not ideal to reload all.
        #  Should have option to reload only the workers if already setup (and not new exp)
        # Update other configs.
        self.main_window.update_tabs()
        self.main_window.finalise_tab_params()
        self.save_cfg()

    def update_pipelines(self, former_channels=None, new_channels=None):  # WARNING: unused args
        if self.detached:
            return
        self.params.ui_to_cfg()
        self.main_window.update_tabs(former_channels, new_channels)

    def update_atlas(self, channel, *args):
        self.params.ui_to_cfg()
        aligner = self.main_window.tab_managers['registration'].aligner
        try:
            aligner.update_atlas_asset(channel=channel)
        except KeyError as err:
            warnings.warn(f'Could not update atlas for channel {channel} because it is not in the workspace. '
                          f'Setting all atlases to None; {err}')
            aligner.setup_atlases()

    @property
    def src_folder(self):
        return self.ui.srcFolderTxt.text()

    @src_folder.setter
    def src_folder(self, folder):
        self.ui.srcFolderTxt.setText(folder)

    def display_sample_id(self, sample_id):
        """
        Display the sample ID to the corresponding UI widget
        Parameters
        ----------
        sample_id : str
            The unique ID for that sample
        """
        self.ui.sampleIdTxt.setText(sample_id)

    def display_use_id_as_prefix(self, use_id):
        """
        Displays whether to use the ID as prefix in the corresponding
        widget of the UI

        Parameters
        ----------
        use_id : bool
            Whether to se the sample ID as prefix in the file names
        """
        self.ui.useIdAsPrefixCheckBox.setChecked(use_id)

    def get_sample_id(self):
        """Get the sample ID from the GUI widget"""
        return self.ui.sampleIdTxt.text()

    def go_to_orientation(self):
        """Jump to the sample orientation (space info) tab"""
        self.ui.toolBox.setCurrentIndex(2)
        self.main_window.tabWidget.setCurrentIndex(0)

    def launch_pattern_wizard(self):
        """
        Start the pattern selection wizard. This wizard helps create the
        pattern strings for the individual tiles, with specific characters
        representing the digits for the different axes.
        """
        self.detached = True
        former_channels = self.params.channels
        dlg = PatternDialog(self.src_folder, self.params, tab=self,
                            min_file_number=self.main_window.preference_editor.params.pattern_finder_min_n_files,
                            tile_extension=self.params.shared_sample_params.default_tile_extension)
        dlg.exec()
        self.detached = False
        self.update_pipelines(former_channels, self.params.channels)

    def plot_mini_brain(self, channel):
        """
        Plot the brain icon which represents the acquisition sample orientation graphically
        to help users pick the right orientation.
        """
        if isinstance(channel, int):
            channel = self.params.get_channel_name(channel)
        # REFACTOR: a bit hacky to refer to other tab
        aligner = self.main_window.tab_managers['registration'].aligner
        if aligner.setup_complete:
            mask, proj = aligner.project_mini_brain(channel)
            self.get_channel_ui(channel).miniBrainLabel.setPixmap(np_to_qpixmap(proj, mask))
        else:
            self.update_workspace()
            if aligner.setup_complete:
                mask, proj = aligner.project_mini_brain(channel)
                self.get_channel_ui(channel).miniBrainLabel.setPixmap(np_to_qpixmap(proj, mask))
            else:
                warnings.warn('RegistrationProcessor not setup, cannot plot mini brain. '
                              'Please call registration_tab.finalise_set_params() first')

    def display_atlas(self, channel):
        """Plot the atlas as a grayscale image in the viewer"""
        if isinstance(channel, int):
            channel = self.params.get_channel_name(channel)
        # REFACTOR: a bit hacky to refer to other tab
        aligner = self.main_window.tab_managers['registration'].aligner
        if aligner.setup_complete:
            self.wrap_plot(aligner.plot_atlas, channel)
        else:
            self.update_workspace()
            if aligner.setup_complete:
                self.wrap_plot(aligner.plot_atlas, channel)
            else:
                warnings.warn('RegistrationProcessor not setup, cannot plot atlas. '
                              'Please call registration_tab.finalise_set_params() first')

    def convert_to_clearmap_format(self, channel):
        self.params.ui_to_cfg()
        self.sample_manager.config.reload()
        # channel = self.ui.channelsParamsTabWidget.current_channel()
        stitching_processor = self.main_window.tab_managers['stitching'].stitcher
        if self.sample_manager.is_tiled(channel):
            stitching_processor.convert_tiles_channel(channel)
        else:
            stitching_processor.copy_or_stack(channel)


class StitchingTab(PreProcessingTab):
    """
    The tab responsible for all the alignments, including the stitching and
    aligning to the atlas.
    """
    # layout_channel_changed = pyqtSignal(str, str)

    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'stitching_tab', tab_idx)
        self.channels_ui_name = 'stitching_params'
        self.sample_manager = sample_manager
        self.stitcher = StitchingProcessor(self.sample_manager)
        self.advanced_controls_names = [
            'channel.useNpyCheckBox',
        ]
        self.set_relevant_data_types(DATA_TYPE_TO_TAB_CLASS)

    def filter_relevant_channels(self, channels, must_exist=True):
        if channels is None:
            return
        if must_exist:
            new_channels = [c for c in channels if self.sample_manager.is_tiled(c)]
        else:
            new_channels = [c for c in channels if
                            c not in self.sample_params or self.sample_manager.is_tiled(c)]
        return new_channels

    def _read_configs(self, cfg_path):
        if self.sample_manager.stitching_cfg:
            self.params.read_configs(cfg=self.sample_manager.stitching_cfg)
        else:
            self.params.read_configs(cfg_path=cfg_path)

    def _load_config_to_gui(self):
        super()._load_config_to_gui()
        self.update_plotable_channels()

    def _set_channels_names(self):  # FIXME: move to stitcher
        stitchable_channels = self._get_channels()
        if not stitchable_channels:
            return
            # rename defaults
        if 'channel_x' in self.stitcher.config['channels'].keys():  # i.e. is default, otherwise already set
            first_channel = stitchable_channels[0]
            self.stitcher.config['channels'][first_channel] = self.stitcher.config['channels'].pop('channel_x')
            self.stitcher.config['channels'][first_channel]['layout_channel'] = first_channel
            for i in range(1, len(stitchable_channels)):
                self.stitcher.config['channels'][stitchable_channels[i]] = deepcopy(self.stitcher.config['channels']['channel_y'])
                self.stitcher.config['channels'][stitchable_channels[i]]['layout_channel'] = first_channel
            self.stitcher.config['channels'].pop('channel_y')
        else:  # rename existing
            for old_name, new_name in zip(self.stitcher.config['channels'].keys(), stitchable_channels):
                if old_name == new_name:
                    continue
                self.stitcher.config['channels'][new_name] = self.stitcher.config['channels'].pop(old_name)
                if self.stitcher.config['channels'][new_name]['layout_channel'] == old_name:
                    self.stitcher.config['channels'][new_name]['layout_channel'] = new_name
        self.stitcher.config.write()

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        if not isinstance(self.ui.runChannelsCheckableListWidget, CheckableListWidget):
            self.ui.runChannelsCheckableListWidget = replace_widget(self.ui.runChannelsCheckableListWidget,
                                                                    CheckableListWidget(self.ui),
                                                                    self.ui.runAndDisplayGridLayout)
            self.ui.runChannelsCheckableListWidget.check_state_changed.connect(self.set_run_channel)

        if not isinstance(self.ui.plotChannelsCheckableListWidget, CheckableListWidget):
            self.ui.plotChannelsCheckableListWidget = replace_widget(self.ui.plotChannelsCheckableListWidget,
                                                                     CheckableListWidget(self.ui),
                                                                     self.ui.runAndDisplayGridLayout)

        self.ui.runStitchingPushButton.clicked.connect(self.run_stitching)
        self.ui.displayStitchingPushButton.clicked.connect(self.plot_stitching_results)
        self.ui.displayStitchingClearPlots.clicked.connect(self.main_window.clear_plots)

    def _bind_params_signals(self):  # WARNING: not really params signals but hack necessary to update the UI
        self.ui.runChannelsCheckableListWidget.set_items(self._get_channels())
        for chan in self._get_channels():
            self.ui.runChannelsCheckableListWidget.set_item_checked(chan, self.params[chan].shared.run)
            # self.ui.plotChannelsCheckableListWidget.set_item_checked(chan, self.params[chan].shared.plot)

    def _set_params(self):
        self.params = StitchingParams(self.ui)  # , self.ui.channelsParamsTabWidget is deduced from the UI

    def _get_channels(self):
        return self.sample_manager.get_stitchable_channels()

    def _set_channel_config(self, channel):
        # Force same instance  # TODO: inherited from parent
        self.params[channel].read_configs(cfg=self.stitcher.config)  # WARNING: should this be self.sample_manager.stitching_config ?

    def _setup_channel(self, page_widget, channel):
        if stitchable_channels := self.sample_manager.get_stitchable_channels():
            self.params[channel].ready = self.params_finalised
            page_widget.layoutChannelComboBox.addItems(['undefined'] + stitchable_channels)
        is_first_channel = self.ui.channelsParamsTabWidget.last_real_tab_idx == 1
        # layout_channel = channel if is_first_channel else self.sample_manager.get_stitchable_channels()[0]
        page_widget.layoutChannelComboBox.setCurrentText('undefined')
        layout_chan_from_cfg = self.params[channel].config['channels'][channel]['layout_channel']  # FIXME: why is the config not pointing to channel directly
        if layout_chan_from_cfg not in ('undefined', 'channel_x', 'channel_y'):
            self.params[channel].layout_channel = channel
        self.params[channel].ready = True
        # self.params[channel].ui_to_cfg()  # FIXME: check if we need to put this back

    def _bind_channel(self, page_widget, channel):
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute
        """
        buttons_functions = [
            ('previewStitchingPushButton', self.preview_stitching_dumb, {'color': True}),
            ('stitchingPreviewLevelsPushButton', self.preview_stitching_dumb, {'color': False}),
            ('stitchingPreviewRigidPushButton', self.preview_stitching_smart, {'asset_sub_type': 'aligned_axis'})
        ]
        for btn_name, func, kwargs in buttons_functions:
            self._bind_btn(btn_name, func, channel, page_widget, **kwargs)
        self.ui.runChannelsCheckableListWidget.set_item_checked(channel, self.params[channel].shared.run)
        # self.params[channel].layout_channel_changed.connect(self.handle_layout_channel_changed)

    def _setup_workers(self):
        """
        Setup the worker (Processor) which handles the computations associated with this tab
        """
        self.sample_params.ui_to_cfg()
        # FIXME: must ensure this is called before adding channels
        self.stitcher.setup(self.sample_manager, convert_tiles=False)

        # if self.sample_manager.has_tiles() and not self.sample_manager.has_npy():
        #     if  prompt_dialog('Tile conversion', 'Convert individual tiles to npy for efficiency'):
        #         self.convert_tiles()

    # def handle_layout_channel_changed(self, channel, layout_channel):
    #    self.layout_channel_changed.emit(channel, layout_channel)

    def convert_tiles(self):
        if not self.sample_manager.has_tiles():
            return
        self.wrap_step('Converting tiles', self.stitcher.convert_tiles, step_kw_args={'_force': True}, n_steps=0,
                       abort_func=self.stitcher.stop_process, save_cfg=False, nested=False)

    def set_run_channel(self, _, state, channel):
        """
        Set the channels to run the stitching on

        Parameters
        ----------
        state : bool
            Whether the channel is checked
        channel : str
            The name of the channel
        """
        self.params[channel].shared.run = state

    def set_progress_watcher(self, watcher):  # REFACTOR: could be for worker in self.workers
        """
        Setup the watcher object that will handle the progress in the computation for this tab

        Parameters
        ----------
        watcher : ProgressWatcher
            The object that tracks the progress of the computation
        """
        self.stitcher.set_progress_watcher(watcher)

    def prompt_conversion(self, channel):
        """
        Prompt the user to convert the tiles to npy for efficiency

        Parameters
        ----------
        channel : str
            The channel to convert
        """
        if not self.sample_manager.has_npy(channel):
            choices = ['Yes', 'No', 'Cancel']
            choice = option_dialog('Convert tiles', 'This operation is much slower with tiff files. ' \
                                                    'Convert to npy for efficiency ?',
                                   options=choices, )
            if choice == choices.index('Yes'):
                self.convert_tiles()
            elif choice == choices.index('Cancel'):
                return 'cancel'

    def preview_stitching_dumb(self, channel, color):
        """
        Preview the stitching based only on a *dumb* overlay of the tiles
        i.e. only using the fixed guess overlap

        Parameters
        ----------
        channel : str
            The channel to preview
        color : bool
            Whether to stitch in chessboard or continuous grayscale
        """
        choice = self.prompt_conversion(channel)
        if choice == 'cancel':
            return
        self.params.ui_to_cfg()
        stitched = self.stitcher.stitch_overlay(channel, color)
        if color:
            overlay = [pg.image(stitched)]
        else:  # TODO: make DataViewer work with 2D color
            overlay = plot_3d.plot(stitched, lut='flame', min_max=(100, 5000))
        self.main_window.setup_plots(overlay)

    def preview_stitching_smart(self, channel, asset_sub_type='aligned_axis'):
        """
        Preview the stitching based on the actual stitching variable, rigid by default.

        Parameters
        ----------
        channel : str
            The channel to preview
        asset_sub_type : str
            One of ('aligned_axis', 'aligned', 'placed')
        """
        choice = self.prompt_conversion(channel)
        if choice == 'cancel':
            return
        self.params.ui_to_cfg()
        n_steps = self.stitcher.n_rigid_steps_to_run
        self.wrap_step('Stitching', self.stitcher.stitch_channel_rigid,
                       step_args=[channel], step_kw_args={'_force': True},
                       n_steps=n_steps, abort_func=self.stitcher.stop_process)
        overlay = [pg.image(self.stitcher.plot_layout(channel=channel, asset_sub_type=asset_sub_type))]
        self.main_window.setup_plots(overlay)

    def run_stitching(self):
        """Run the actual stitching steps based on the values in the config file (set from the UI)."""
        self.params.ui_to_cfg()
        self.stitcher.config.reload()
        for channel in self.sample_manager.channels:  # FIXME: check if should do and if done
            if not self.sample_manager.is_tiled(channel):  # BYPASS stitching, just copy or stack
                self.wrap_step('Stitching', self.stitcher.copy_or_stack, step_args=[channel], )

        n_steps = self.stitcher.n_rigid_steps_to_run + self.stitcher.n_wobbly_steps_to_run
        for stitching_tree in self.stitcher.get_stitching_order().values():
            for channel in stitching_tree:
                cfg = self.params[channel]
                if not cfg.shared.run:
                    continue
                if self.params[channel].shared.use_npy and not self.sample_manager.has_npy(channel):
                    self.convert_tiles()
                kwargs = {'n_steps': n_steps, 'abort_func': self.stitcher.stop_process, 'close_when_done': False}
                try:
                    if channel == cfg.shared.layout_channel and not cfg.shared.use_existing_layout:  # Used as reference
                        self.wrap_step('Stitching', self.stitcher.stitch_channel_rigid,
                                       step_args=[channel], step_kw_args={'_force': True}, **kwargs)
                        self.wrap_step(task_name='', func=self.stitcher.stitch_channel_wobbly, step_args=[channel],
                                       step_kw_args={'_force': cfg.stitching_rigid.skip}, **kwargs)
                    else:  # Uses other channel as reference
                        self.wrap_step('', self.stitcher._stitch_layout_wobbly,
                                       step_args=[channel],  **kwargs)
                except MissingRequirementException as err:
                    error_msg = str(err).replace("\n", "<br>")
                    self.main_window.print_status_msg(f'Skipping stitching for {channel} because of missing requirements: {error_msg}')

        self.update_plotable_channels()
        self.progress_watcher.finish()

    def update_plotable_channels(self):
        self.ui.plotChannelsCheckableListWidget.clear()
        for chan in self.sample_manager.stitchable_channels:
            # if not self.params[chan].skip:
            if self.sample_manager.get('stitched', channel=chan).exists:
                self.ui.plotChannelsCheckableListWidget.add_item(chan)
                # self.ui.plotChannelsCheckableListWidget.set_item_checked(chan, self.params[chan].plot)

    def plot_stitching_results(self, _):
        """Plot the stitched image in 3D in the viewer"""
        mode = self.ui.stitchingPlotModeComboBox.currentText()
        channels = self.ui.plotChannelsCheckableListWidget.get_checked_items() or []
        for channel in channels:
            self.params[channel].shared.ui_to_cfg()
        if not channels:
            return self.main_window.print_status_msg('No channels selected to plot')
        self.wrap_plot(self.stitcher.plot_stitching_results, channels=channels,
                       mode=mode, parent=self.main_window.centralWidget())


class RegistrationTab(PreProcessingTab):
    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'registration_tab', tab_idx)
        self.sample_manager = sample_manager
        self.aligner = RegistrationProcessor(self.sample_manager)

        self.channels_ui_name = 'registration_params'

        self.advanced_controls_names = [
            'advancedAtlasSettingsGroupBox',
            'channel.registrationRunResamplingPushButton',
            'channel.parameterFilesLabel',
            'channel.paramsFilesListWidget',
            'channel.addParamFilePushButton',
            'channel.removeParamFilePushButton',
            'channel.selectLandmarksPushButton',
            'channel.selectLandmarksPushButtonInfoToolButton',
            'channel.landmarksWeightsGroupBox',
        ]
        # self.set_relevant_data_types(DATA_TYPE_TO_TAB_CLASS)
        self.relevant_data_types = 'all'

    def _read_configs(self, cfg_path):
        if self.sample_manager.registration_cfg:
            self.params.read_configs(cfg=self.sample_manager.registration_cfg)
        else:
            self.params.read_configs(cfg_path=cfg_path)

    def _set_channels_names(self):
        if 'channel_x' in self.aligner.config['channels']:  # i.e. is default, otherwise already set
            autofluo_default_config = deepcopy(self.aligner.config['channels'].get('autofluorescence'))
            data_channel_default_config = deepcopy(self.aligner.config['channels']['channel_x'])

            self.aligner.config['channels'] = {}
            reference_channel = self.sample_manager.alignment_reference_channel
            data_channel_default_config['align_with'] = reference_channel
            data_channel_default_config['moving_channel'] = reference_channel
            for channel in self.sample_manager.channels:
                if self.sample_manager.config['channels'][channel]['data_type'] == 'autofluorescence':
                        self.aligner.config['channels'][channel] = autofluo_default_config
                else:
                    self.aligner.config['channels'][channel] = data_channel_default_config
                self.handle_align_with_changed(channel, self.aligner.config['channels'][channel]['align_with'])
            self.aligner.config.write()  # FIXME: ensure that aligner defined. Why not self.params.ui_to_cfg()

        self.aligner.add_pipeline()

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        self.ui.registerPushButton.clicked.connect(self.run_registration)
        self.ui.plotRegistrationResultsPushButton.clicked.connect(self.plot_registration_results)

        # Populate atlas and structure tree combo boxes from config
        for k in ATLAS_NAMES_MAP.keys():
            if self.ui.atlasIdComboBox.findText(k) == -1:
                self.ui.atlasIdComboBox.addItem(k)

        for k in STRUCTURE_TREE_NAMES_MAP.keys():
            if self.ui.structureTreeIdComboBox.findText(k) == -1:
                self.ui.structureTreeIdComboBox.addItem(k)

    def _bind_params_signals(self):
        self.params.atlas_params.atlas_id_changed.connect(self.aligner.setup_atlases)
        self.params.atlas_params.atlas_structure_tree_id_changed.connect(self.aligner.setup_atlases)
        self.params.launchLandmarksDialog.connect(self.launch_landmarks_dialog)
        self._update_plotable_channels()
        for channel in self.aligner.config['channels']:
            self.__update_channel_combo_boxes(channel)

    def _set_params(self):
        self.params = RegistrationParams(self.ui)

    def _get_channels(self):
        return self.sample_manager.channels  # All channels so we can decide whether to register in UI

    def _set_channel_config(self, channel):
        self.params[channel]._config = self.aligner.config
        self.params[channel].cfg_to_ui()  # Force it while the tab is active

    def __update_channel_combo_boxes(self, channel, page_widget=None):
        if self.params[channel].align_with:
            partner_channel = self.params[channel].align_with
        else:
            partner_channel = self.aligner.config['channels'][channel]['align_with']
        if page_widget is None:
            page_widget = self.ui.channelsParamsTabWidget.get_channel_widget(channel)
        other_channels = list(set(self.aligner.channels_to_register()) - {channel})
        page_widget.alignWithComboBox.clear()
        page_widget.alignWithComboBox.addItems([None, 'atlas'] + other_channels)
        page_widget.movingChannelComboBox.clear()
        page_widget.movingChannelComboBox.addItems([None, 'atlas', 'intrinsically aligned'] + other_channels + [channel])
        channel_dtype = self.sample_manager.config['channels'][channel]['data_type']
        print(id(self.sample_manager.config))
        print(f'Configuring alignment partners for {channel=}, {channel_dtype=}, {partner_channel=}')
        if not partner_channel or partner_channel == channel:
            if channel_dtype == 'autofluorescence':
                partner_channel = 'atlas'
            elif channel_dtype:
                partner_channel = self.sample_manager.alignment_reference_channel
            else:
                warnings.warn(f'Skipping partner selection for {channel} because none was found')
                return
        page_widget.alignWithComboBox.setCurrentText(partner_channel)
        page_widget.movingChannelComboBox.setCurrentText(partner_channel)

    def _setup_channel(self, page_widget, channel):
        self.__update_channel_combo_boxes(channel, page_widget)
        alignment_files = [page_widget.paramsFilesListWidget.item(i).text() for i in
                  range(page_widget.paramsFilesListWidget.count())]  # no shortcut for standard QListWidget
        page_widget.paramsFilesListWidget = replace_widget(page_widget.paramsFilesListWidget,
                                                           FileDropListWidget(page_widget,
                                                                              page_widget.addParamFilePushButton,
                                                                              page_widget.removeParamFilePushButton),
                                                           page_widget.registrationChannelGridLayout)
        page_widget.paramsFilesListWidget.addItems(alignment_files)  # Transfer existing files to new widget

    def _bind_channel(self, page_widget, channel):
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute
        """
        # TODO: set value of comboboxes to good defaults
        page_widget.paramsFilesListWidget.itemsChanged.connect(self.params[channel].handle_params_files_changed)
        self.params[channel].handle_params_files_changed()  # Force update
        self.params[channel].align_with_changed.connect(self.handle_align_with_changed)
        page_widget.registrationRunResamplingPushButton.clicked.connect(
            functools.partial(self.resample_channel, channel)
        )

    def _setup_workers(self):
        self.sample_params.ui_to_cfg()
        self.aligner.setup()
        self.params.read_configs(cfg=self.aligner.config)
        if self.sample_manager.setup_complete:
            self.wrap_step('Setting up atlas', self.setup_atlas, n_steps=1, save_cfg=False, nested=False)  # TODO: abort_func=self.aligner.stop_process

    # def handle_layout_channel_changed(self, channel, layout_channel):
    #    """To select automatically "intrinsically aligned" if 2 channels have same stitching layout"""
    #     self.params[channel].align_with = layout_channel
    #     self.params[channel].moving_channel = 'intrinsically aligned'
    #     self.params[channel].cfg_to_ui()  # Update the UI to reflect the changes

    def set_progress_watcher(self, watcher):
        """
        Setup the watcher object that will handle the progress in the computation for this tab

        Parameters
        ----------
        watcher : ProgressWatcher
            The object that tracks the progress of the computation
        """
        self.aligner.set_progress_watcher(watcher)

    def setup_atlas(self):  # TODO: check if we delay this to update_workspace
        """Setup the atlas that corresponds to the orientation and cropping of the sample"""
        self.sample_params.ui_to_cfg()  # To make sure we have the slicing up to date
        self.params.atlas_params.ui_to_cfg()
        self.aligner.setup_atlases()

    def clear_landmarks(self, channel):
        self.aligner.clear_landmarks(channel)
        # TODO: use landmark_selector

    def launch_landmarks_dialog(self, channel):
        if isinstance(channel, int):
            channel = self.params.get_channel_name(channel)
        # We have to keep reference to make it persistent but should be per channel
        self.landmark_selector = LandmarksSelectorDialog(
            fixed_image_path=self.aligner.get_fixed_image(channel).path,
            moving_image_path=self.aligner.get_moving_image(channel).path,
            fixed_image_landmarks_path=self.aligner.get_elx_asset('fixed_landmarks',
                                                                  channel=channel).path,
            moving_image_landmarks_path=self.aligner.get_elx_asset('moving_landmarks',
                                                                   channel=channel).path)
        self.landmark_selector.plot(lut=self.main_window.preference_editor.params.lut,
                                    parent=self.main_window.centralWidget())
        self.main_window.setup_plots(self.landmark_selector.data_viewers.values())

    def write_registration_landmark_coords(self, channel):
        """
        Write the corresponding landmarks to file for use in landmark optimised registration
        """
        self.landmark_selector.write_coords()
        self.landmark_selector.dlg.close()
        self.landmark_selector = None

    def handle_align_with_changed(self, channel, align_with):
        if align_with == channel:
            raise ValueError(f'Cannot align {channel=} with itself')  # FIXME: popup instead of crashing whole app (QT anyway)
        elif align_with is None:
            # Remove registration pipeline from channel
            pass
        else:
            if self.sample_manager.setup_complete:
                self.sample_manager.workspace.add_pipeline('registration', channel_id=channel)
                self.aligner.parametrize_assets()
            else:
                warnings.warn('Workspace not setup, cannot add registration pipeline')

    def resample_channel(self, channel):
        self.main_window.make_progress_dialog('Registering', n_steps=2, abort=self.aligner.stop_process,
                                              parent=self.main_window)
        self.setup_atlas()
        self.main_window.progress_watcher.increment_main_progress()
        self.sample_manager.delete_resampled_files(channel)
        self.wrap_step(f'Resampling {channel} for registration', self.aligner.resample_channel,
                       step_kw_args={'channel': channel, 'increment_main': False})
        self.main_window.progress_watcher.finish()
        self.main_window.print_status_msg(f'Channel {channel} resampled for registration')

    def run_registration(self):
        """
        Run the actual registration between the sample and the reference atlas.
        """
        self.params.ui_to_cfg()
        # TODO: compute n_steps (part of processor; n_channels * n_steps_per_channel)
        self.main_window.make_progress_dialog('Registering', n_steps=4, abort=self.aligner.stop_process,
                                              parent=self.main_window)
        self.setup_atlas()
        for i, channel in enumerate(self.params.keys()):
            if self.params[channel].resample:
                asset = self.aligner.get('stitched', channel=channel)
                if not asset.exists:
                    asset = self.aligner.get('raw', channel=channel)
                    if asset.is_tiled and not asset.all_tiles_exist:
                        self.main_window.progress_watcher.finish()
                        self.main_window.print_status_msg(f'Registration skipped because of missing tiles'
                                                          f'for channel {channel}')
                        return
                try:
                    self.wrap_step(f'Resampling {channel} for registration', self.aligner.resample_channel,
                                   step_kw_args={'channel': channel, 'increment_main':(i != 0)})
                except FileExistsError:  # REFACTOR: factorise with the above
                    option_idx = option_dialog('Files exist',
                                               f'Resampled files exist for {channel}, do you want to: ',
                                               ['Delete and retry', 'Skip resampling and continue'])
                    if option_idx == 0:
                        self.sample_manager.delete_resampled_files(channel)
                        self.wrap_step(f'Resampling {channel} for registration', self.aligner.resample_channel,
                                       step_kw_args={'channel': channel, 'increment_main': (i != 0)})
                    else:
                        continue
        self.main_window.wrap_in_thread(self.aligner.align)
        self._update_plotable_channels()
        self.main_window.print_status_msg('Registered')

    def _update_plotable_channels(self):
        self.ui.plotChannelComboBox.clear()  # TODO: extract to method
        if not self.sample_manager.setup_complete:
            return
        for channel in self.params.keys():
            asset = self.aligner.get_elx_asset('aligned', channel=channel)
            if asset is not None and asset.exists:
                self.ui.plotChannelComboBox.addItem(channel)

    def __prepare_registration_results_graph(self, channel):
        img_paths = [
            self.aligner.get_fixed_image(channel).path,
            self.aligner.get_aligned_image(channel)
        ]
        if not all([p.exists() for p in img_paths]):
            raise ValueError(f'Missing requirements {img_paths}')
        titles = [img.parent.stem if 'aligned_to' in str(img) else img.stem for img in img_paths]
        # TODO: replace result<N,1> by channel name
        return img_paths, titles

    def plot_registration_results(self):
        """
        Plot the result of the registration between 2 channels. Either side by side or as a composite

        If the composite checkbox is checked, the two images are overlayed.
        Otherwise, they are displayed side by side
        """
        channel = self.params.shared_params.plot_channel
        composite = self.params.shared_params.plot_composite
        self.main_window.clear_plots()
        image_sources, titles = self.__prepare_registration_results_graph(channel)
        if composite:
            image_sources = [image_sources, ]
        dvs = plot_3d.plot(image_sources, title=titles, arrange=False, sync=True,
                           lut=self.main_window.preference_editor.params.lut,
                           parent=self.main_window.centralWidget())
        self.main_window.setup_plots(dvs, graph_names=titles)
        if not composite:
            link_dataviewers_cursors(dvs)


class CellCounterTab(PostProcessingTab):
    """
    The tab responsible for the cell detection and cell coordinates alignment
    """
    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'cell_map_tab', tab_idx)

        self.sample_manager = sample_manager
        self.aligner = None  # Will be configured by self.set_pre_processors

        self.cell_detectors = {}

        self.channels_ui_name = 'cell_map_params'

        self.cell_intensity_histogram = None
        self.cell_size_histogram = None

        self.advanced_controls_names = ['channel.detectionShapeGroupBox']
        self.set_relevant_data_types(DATA_TYPE_TO_TAB_CLASS)

    def _set_channels_names(self):
        """Patch the config with the actual channels if not already done"""
        # Update if channels are not a subset of valid channels
        if not set(self.params.config['channels'].keys()) <= set(self.sample_manager.channels):
            default_channel_config = deepcopy(self.params.config['channels']['example'])
            self.params.config['channels'] = {}
            for channel in self.params.channels_to_detect:
                self.params.config['channels'][channel] = default_channel_config
            self.params.config.write()
            self._setup_workers()  # WARNING: a bit circular but if we need to amend channels
                                   #    then we also reset the workers

    def _bind(self):
        pass

    def _bind_params_signals(self):  # Execute at the end of finalise_set_params
        pass
        # if len(self.params.channels_to_detect) > 1:
        #     if not self.main_window.has_tab(ColocalizationTab):
        #         self.main_window.add_tab(ColocalizationTab, sample_manager=self.sample_manager, set_params=True)
        # self.ui.advancedCheckBox.stateChanged.connect(self.params.handle_advanced_state_changed)

    def _set_params(self):
        # REFACTORING: accessing main_window is not the most elegant way
        self.params = CellMapParams(self.ui, self.sample_params, None,
                                    self.main_window.tab_managers['registration'].params)

    def _get_channels(self):
        return self.params.channels_to_detect

    def _setup_workers(self):
        """
        Setup the cell detection worker, which handle the computations associated with this tab
        """
        if self.sample_manager.workspace is not None:
            if self.config_initialized():
                self.params.ui_to_cfg()  # WARNING: check why
                self.cell_detectors = {channel: CellDetector(self.sample_manager, channel)
                                       for channel in self.params.channels_to_detect}
            else:
                self.main_window.print_warning_msg('Channels not yet set in config. Cannot define workers.')
        else:
            self.main_window.print_warning_msg('Workspace not initialised')

    def config_initialized(self):
        return 'example' not in self.params.config['channels']

    def finalise_workers_setup(self):  #  When tab clicked  # FIXME: trigger with signal from self.sample_manager
        """Post configuration of the CellDetector object"""
        is_first_channel = True
        for channel, detector in self.cell_detectors.items():
            if detector.registration_processor is None and self.aligner is not None:  # TODO: might be redundant with setup_workers
                if is_first_channel:
                    self.params.ui_to_cfg()
                    is_first_channel = False
                detector.setup(self.sample_manager, channel, self.aligner)
            self.update_cell_number(channel)
        if self.sample_manager.is_colocalization_compatible:
            for channel in self.cell_detectors.keys():
                self.make_colocalization_compatible(channel)
            if not self.main_window.has_tab(ColocalizationTab):
                self.main_window.add_tab(ColocalizationTab, sample_manager=self.sample_manager, set_params=True)

    def _set_channel_config(self, channel):
        """Add a new channel tab to the UI"""
        pass
        # self.params.add_channel(channel)  # FIXME:
        # self.params[channel].config = self.detectors[channel].config

    def _bind_channel(self, page_widget, channel):
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute
        """
        page_widget.toolBox.currentChanged.connect(self.handle_tool_tab_changed)
        page_widget.runCellMapColocalizationCompatibleCheckBox.setVisible(False)
        buttons_functions = [
            ('detectionPreviewTuningOpenPushButton', self.plot_debug_cropping_interface),  # TODO: add load icon
            ('detectionPreviewTuningCropPushButton', self.create_cell_detection_tuning_sample),
            ('detectionPreviewPushButton', self.run_tuning_cell_detection),
            ('previewCellFiltersPushButton', self.preview_cell_filter),
            ('runCellMapPushButton', self.run_channel),

            ('cellMapPlotVoxelizationPushButton', self.plot_cell_map_results),
            ('cellMap3dScatterOnRefPushButton', functools.partial(self.plot_labeled_cells_scatter, raw=False)),
            ('cellMap3dScatterOnStitchedPushButton', functools.partial(self.plot_labeled_cells_scatter, raw=True)),
        ]
        for btn_name, func in buttons_functions:
            self._bind_btn(btn_name, func, channel, page_widget)

    def make_colocalization_compatible(self, channel):
        page_widget = self.ui.channelsParamsTabWidget.get_channel_widget(channel)
        page_widget.runCellMapColocalizationCompatibleCheckBox.setVisible(True)
        self.params[channel].colocalization_compatible = True

    def setup_cell_param_histogram(self, cells, plot_item, key='size', x_log=False):
        """
        Plots the histogram of the cell parameter defined by key. This is used to display the
        distribution of cell sizes or intensities

        Parameters
        ----------
        cells : pd.DataFrame
            The Cells dataframe containing one row per detected cell
        plot_item : QWidget or None
            The Plot element to plot into. If None, creates a new one
        key : str
            The key (cell attribute) in the dataframe to plot.
             One of 'size' or 'source'
        x_log : bool
            X axis is logarithmic
        """
        values = cells[key].values
        hist, bin_edges = np.histogram(values, bins=20)
        if plot_item is None:
            widget = pg.plot(hist, bin_edges[:-1], pen=pg.mkPen(DarkPalette.COLOR_ACCENT_2))
        else:
            widget = plot_item
            widget.plot(hist, bin_edges[:-1], pen=pg.mkPen(DarkPalette.COLOR_ACCENT_2), clear=True)
        widget.setBackground(DarkPalette.COLOR_BACKGROUND_2)
        widget.setLogMode(x=x_log)
        return widget

    def voxelize(self, channel):
        """Creates the cell density plot """
        if self.sample_manager.get('cells', channel=channel, postfix='filtered').exists:
            detector = self.cell_detectors[channel]
            self.wrap_step('Voxelization', detector.voxelize,
                           abort_func=detector.stop_process, nested=False)#, main_thread=True)
        else:
            self.main_window.popup('Could not run voxelization, missing filtered cells table. '
                                   'Please ensure that cell filtering has been run.', base_msg='Missing file')

    def set_progress_watcher(self, watcher):
        """
        Setup the watcher object that will handle the progress in the computation for this tab

        Parameters
        ----------
        watcher : ProgressWatcher
            The object that tracks the progress
        """
        for detector in self.cell_detectors.values():
            if detector is not None and detector.sample_manager is not None:  # If initialised
                detector.set_progress_watcher(watcher)

    def plot_debug_cropping_interface(self, channel):
        """
        Plot the orthoslicer to select a subset of the sample to perform cell detections
        tests on
        """
        self.plot_slicer('detectionSubset', self.ui.channelsParamsTabWidget.get_channel_widget(channel),
                         self.params[channel], channel)

    def handle_tool_tab_changed(self, tab_idx):
        """
        Triggered when a new sub tab (tooltab) of the cell detection tab is selected.
        It will either plot the cell parameter distributions or update the cell count display.

        Parameters
        ----------
        tab_idx
        """
        channel = self.ui.channelsParamsTabWidget.current_channel()
        if tab_idx == 1:
            self.__try_plot_histograms(channel)
        elif tab_idx == 3:
            self.update_cell_number(channel)

    def __try_plot_histograms(self, channel):
        for sample_type in ('normal', 'debug'):
            old_status = self.sample_manager.workspace.debug
            try:
                self.sample_manager.workspace.debug = sample_type == 'debug'
                self.__plot_histograms(channel)
                break  # Exit as soon as any cells df is found
            except FileNotFoundError:
                end = 'skipping' if sample_type == 'debug' else 'trying debug'
                print(f'Could not find {sample_type} cells dataframe file, {end}')
            finally:
                self.sample_manager.workspace.debug = old_status
        else:
            self.main_window.popup('No cells file found, cannot display histograms yet')

    def __plot_histograms(self, channel):
        df_path = self.sample_manager.get_path('cells', channel=channel, postfix='raw')
        cells_df = pd.DataFrame(np.load(df_path))
        self.cell_size_histogram = self.__plot_histogram(channel, cells_df, 'size', self.cell_size_histogram)
        self.cell_intensity_histogram = self.__plot_histogram(channel, cells_df, 'source', self.cell_intensity_histogram)

    def __plot_histogram(self, channel, cells_df, key, histogram):
        histogram = self.setup_cell_param_histogram(cells_df, histogram, key)
        layout = self.get_channel_ui(channel).cellDetectionThresholdsLayout

        widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
        n_plots = len([w for w in widgets if isinstance(w, PlotWidget)])

        count = 0 if key == 'size' else 1

        if n_plots < 2:  # Histograms not yet added
            label, idx = get_widget(layout, widget_type=QLabel, index=count)
            controls, idx = get_widget(layout, key='Doublet', index=count)
            graph_width = label.width() + controls.width()
            graph_height = 50
            histogram.resize(graph_width, graph_height)
            histogram.setMaximumSize(graph_width, graph_height)
            row = 2 * n_plots
            layout.addWidget(histogram, row, 0, 1, 3)
            layout.addWidget(label, row+1, 0, 1, 1)
            layout.addWidget(controls, row+1, 1, 1, 2)

            container = layout.parent().parent().parent().parent()
            container.setMinimumHeight(container.parent().height() - container.height() + layout.parent().height())
        # else:
        #     layout.  # update widget
        return histogram

    def create_cell_detection_tuning_sample(self, channel):
        """Create an array from a subset of the sample to perform tests on """
        detector = self.cell_detectors[channel]
        self.wrap_step('Creating tuning sample', detector.create_test_dataset,
                       step_kw_args={'slicing': self.params[channel].slicing}, nested=False)

    def run_tuning_cell_detection(self, channel):
        """
        Run the cell detection on a subset of the sample which was previously selected
        """
        detector = self.cell_detectors[channel]
        self.wrap_step('Cell detection preview', detector.run_cell_detection,
                       step_kw_args={'tuning': True})
        if detector.stopped:
            return
        try:
            detector.workspace.debug = True
            self.plot_detection_results(channel)
        finally:
            detector.workspace.debug = False

    def detect_cells(self, channel):  # TODO: merge w/ above w/ tuning option
        """Run the cell detection on the whole sample"""
        detector = self.cell_detectors[channel]
        self.wrap_step('Detecting cells', detector.run_cell_detection,
                       step_kw_args={'tuning': False,
                                     'save_shape': self.params[channel].save_shape or self.params[channel].colocalization_compatible,
                                     'save_as_binary_mask': self.params[channel].colocalization_compatible},  # FIXME: seems to crash
                       abort_func=detector.stop_process)
        if detector.stopped:
            return
        self.update_cell_number(channel)

    def update_cell_number(self, channel):
        """
        Update the cell count number displayed based on the size of the raw and filtered cell detection files
        """
        detector = self.cell_detectors[channel]
        params = self.params[channel]
        params.n_detected_cells = format_long_nb(detector.get_n_detected_cells())
        params.n_filtered_cells = format_long_nb(detector.get_n_filtered_cells())

    # def reset_detected(self):
    #     self.cell_detector.detected = False

    def plot_detection_results(self, channel):
        """Display the different steps of the cell detection in a grid to evaluate the filters"""
        dvs = self.wrap_plot(self.cell_detectors[channel].preview_cell_detection,
                             parent=self.main_window.centralWidget(), arrange=False, sync=True)
        if len(dvs) == 1:
            self.main_window.print_warning_msg('Preview not run, '
                                               'will only display stitched image for memory usage reasons')
        else:
            link_dataviewers_cursors(dvs)

    def plot_cell_filter_results(self, channel):
        """
        Plot the cells as colored dots on top of the raw image fraction used for tests
        """
        self.wrap_plot(self.cell_detectors[channel].plot_filtered_cells, smarties=True)

    def plot_labeled_cells_scatter(self, channel, raw=False):
        """
        Plot the cells as colored symbols on top of either the raw stitched (not aligned) image
        or the resampled (aligned) image
        """
        self.wrap_plot(self.cell_detectors[channel].plot_cells_3d_scatter_w_atlas_colors, raw=raw)

    def __filter_cells(self, channel, is_last_step=True):
        if self.sample_manager.get('cells', postfix='raw').exists:
            detector = self.cell_detectors[channel]
            self.wrap_step('Filtering cells', detector.filter_cells, n_steps=2 + (1 - is_last_step),
                           abort_func=detector.stop_process, close_when_done=False)
            self.wrap_step('Voxelizing', detector.voxelize, step_args=['filtered'], save_cfg=False,
                           close_when_done=is_last_step)  # , main_thread=True)
        self.update_cell_number(channel)
        self.plot_cell_filter_results(channel)

    def preview_cell_filter(self, channel):  # TEST: circular calls
        detector = self.cell_detectors[channel]
        try:
            detector.workspace.debug = True
            self.__filter_cells(channel)
        finally:
            detector.workspace.debug = False

    def filter_cells(self, channel):
        self.__filter_cells(channel, is_last_step=False)
        detector = self.cell_detectors[channel]
        self.wrap_step('Aligning', detector.atlas_align, abort_func=detector.stop_process, save_cfg=False)
        detector.export_collapsed_stats()

    def run_cell_map(self):
        """Run the whole pipeline at once"""
        for channel in self.params.channels_to_detect:
            self.run_channel(channel)

    def run_channel(self, channel):
        """Run the whole pipeline at once for a single channel"""
        self.params.ui_to_cfg()
        self.update_cell_number(channel)
        params = self.params[channel]
        if params.detect_cells:
            self.detect_cells(channel)
            if not self.sample_manager.get('cells', channel=channel, asset_sub_type='raw').exists:
                print('Cell detection aborted or failed. Exiting')
                return
            self.update_cell_number(channel)
        if params.filter_cells:
            self.cell_detectors[channel].post_process_cells()  # FIXME: save cfg and use progress
            self.update_cell_number(channel)
        if params.voxelize:
            self.voxelize(channel)
        if params.plot_when_finished:
            self.plot_cell_map_results(channel)
        # WARNING: some plots in .post_process_cells() without UI params

    def plot_cell_map_results(self, channel):
        """Plot the voxelization (density map) result"""
        self.wrap_plot(self.cell_detectors[channel].plot_voxelized_counts, arrange=False)


class VasculatureTab(PostProcessingTab):
    """
    The tab responsible for the vasculature tracts detection, graph extraction and analysis
    """
    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'vasculature_tab', tab_idx)

        self.sample_manager = sample_manager
        self.aligner = None  # Will be configured by self.set_pre_processors

        self.binary_vessel_processor = None
        self.vessel_graph_processor = None

        self.channels_ui_name = 'vasculature_params'

        self.set_relevant_data_types(DATA_TYPE_TO_TAB_CLASS)

    # FIXME: create channels here ??
    def _set_channels_names(self):
        if self.params.config['is_default']:
            self.params.fix_default_config()
        for channel in self.sample_manager.get_channels_by_pipeline('TubeMap'):  # TODO: see if shouldn't be handled by params instead
            channel_type = self.sample_manager.get_channel_type(channel)
            if channel not in self.params.config['binarization'].keys():
                self.params.patch_config_section(channel, channel_type)
        self.params.config.write()

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        # ######################################## BINARIZATION ##############################
        # WARNING: some buttons need channels so setup when processor is defined
        self.ui.binarizationCombinePushButton.clicked.connect(self.combine)

        self.ui.binarizationPlotSideBySidePushButton.clicked.connect(
            functools.partial(self.plot_binarization_results, plot_side_by_side=True))
        self.ui.binarizationPlotOverlayPushButton.clicked.connect(
            functools.partial(self.plot_binarization_results, plot_side_by_side=False))

        # ######################################## GRAPH ##############################
        self.ui.buildGraphSelectAllCheckBox.stateChanged.connect(self.__select_all_graph_steps)
        self.ui.buildGraphPushButton.clicked.connect(self.build_graph)
        self.ui.unloadGraphsPushButton.clicked.connect(self.unload_temporary_graphs)

        self.ui.postProcessVesselTypesPushButton.clicked.connect(self.post_process_graph)

        # ######################################## DISPLAY ##############################
        # slicer
        self.ui.graphSlicerButtonBox.connectOpen(self.plot_graph_type_processing_chunk_slicer)

        self.ui.plotGraphPickRegionPushButton.clicked.connect(self.pick_region)
        self.ui.plotGraphChunkPushButton.clicked.connect(self.display_graph_chunk_from_cfg)
        self.ui.plotGraphClearPlotPushButton.clicked.connect(self.main_window.clear_plots)

        self.ui.voxelizeGraphPushButton.clicked.connect(self.voxelize)
        self.ui.plotGraphVoxelizationPushButton.clicked.connect(self.plot_voxelization)
        self.ui.runAllVasculaturePushButton.clicked.connect(self.run_all)

        self.ui.saveStatsPushButton.clicked.connect(self.save_stats)

    def _set_params(self):
        # REFACTORING: accessing main_window is not the most elegant way
        self.params = VesselParams(self.ui, self.sample_params, None,
                                   self.main_window.tab_managers['registration'].params)

    def _setup_workers(self):
        """
        Setup the BinaryVesselProcessor and VesselGraphProcessor workers,
        which handle the computations associated with this tab
        """
        self._set_channels_names()
        if self.sample_manager.workspace is not None:
            self.params.ui_to_cfg()
            self.binary_vessel_processor = BinaryVesselProcessor(self.sample_manager)
            self.vessel_graph_processor = VesselGraphProcessor(self.sample_manager, self.aligner)
        else:
            self.main_window.print_warning_msg('Workspace not initialised')

    def finalise_workers_setup(self):  # When tab clicked  # FIXME: trigger with signal from self.sample_manager
        """
        Post configuration of the BinaryVesselProcessor and VesselGraphProcessor objects.
        Typically called when the tab is clicked
        """
        if self.sample_manager.setup_complete:
            self.params.ui_to_cfg()
            self.binary_vessel_processor.setup(self.sample_manager)
            self.vessel_graph_processor.setup(self.sample_manager, self.aligner)

    # def _bind_params_signals(self):
    #     pass

    def _get_channels(self):
        return self.sample_manager.get_channels_by_pipeline('TubeMap')

    def _set_channel_config(self, channel):
        self.params[channel]._config = self.params.config
        # FIXME: it seems config for that channel was not set, emtpy dict
        self.params[channel].cfg_to_ui()  # Force it while the tab is active

    def _bind_channel(self, page_widget, channel):
        buttons_functions = [('binarizePushButton', self.binarize_channel)]
        for btn_name, func in buttons_functions:
            self._bind_btn(btn_name, func, channel, page_widget)

    def unload_temporary_graphs(self):
        """Unload the temporary vasculature graph objects to free up RAM"""
        self.vessel_graph_processor.unload_temporary_graphs()

    def set_progress_watcher(self, watcher):  # REFACTOR: for worker in self.workers.values():
        """
        Setup the watcher object that will handle the progress in the computation for this tab

        Parameters
        ----------
        watcher: ProgressWatcher
            The object that tracks the progress
        """
        if self.binary_vessel_processor is not None and self.binary_vessel_processor.sample_manager is not None:
            self.binary_vessel_processor.set_progress_watcher(watcher)
        if self.vessel_graph_processor is not None and self.vessel_graph_processor.sample_manager is not None:
            self.vessel_graph_processor.set_progress_watcher(watcher)

    # ####################### BINARY  #######################

    def binarize_channel(self, channel, stop_on_error=False):
        """
        Perform all the selected binarization steps on the given channel

        Parameters
        ----------
        channel: str
            The name of the channel to binarize.
        stop_on_error: bool
            Whether to stop the process if an error occurs.
        """
        # TODO: n_steps = self.params.binarization_params.n_steps
        self.binary_vessel_processor.assert_input_shapes_match()
        if not self.binary_vessel_processor.inputs_match:
            self.main_window.print_status_msg('Cannot binarize because of shape mismatch between channels')
            return
        try:
            self.wrap_step('Vessel binarization', self.binary_vessel_processor.binarize_channel,
                           step_args=[channel], abort_func=self.binary_vessel_processor.stop_process)
            self.wrap_step('Vessel binarization', self.binary_vessel_processor.smooth_channel,
                           step_args=[channel], abort_func=self.binary_vessel_processor.stop_process)
            self.wrap_step('Vessel binarization', self.binary_vessel_processor.fill_channel,
                           step_args=[channel], abort_func=self.binary_vessel_processor.stop_process,
                           main_thread=True)  # WARNING: The parallel cython loops inside cannot run from child thread
            self.wrap_step('Vessel binarization', self.binary_vessel_processor.deep_fill_channel,
                           step_args=[channel], abort_func=self.binary_vessel_processor.stop_process)
        except ClearMapVRamException as err:
            if stop_on_error:
                raise err

    def combine(self):
        """Combine the binarized (thresholded) version of the different channels."""
        self.wrap_step('Combining channels', self.binary_vessel_processor.combine_binary,
                       abort_func=self.binary_vessel_processor.stop_process)

    def plot_binarization_results(self, plot_side_by_side=True):
        """
        Plot the thresholded images resulting from the binarization at the steps specified
        by the comboboxes in the UI.

        Parameters
        ----------
        plot_side_by_side: bool
            Whether to plot the images side by side (True) or overlay them (False).
        """
        steps, channels = self.params.binarization_params.get_selected_steps_and_channels()
        self.wrap_plot(self.binary_vessel_processor.plot_results,
                       steps, channels=channels, side_by_side=plot_side_by_side,
                       arrange=False, parent=self.main_window)

    # ###########################  GRAPH  #############################

    def __select_all_graph_steps(self, state):
        for chk_bx in (self.ui.buildGraphSkeletonizeCheckBox, self.ui.buildGraphBuildCheckBox,
                       self.ui.buildGraphCleanCheckBox, self.ui.buildGraphReduceCheckBox,
                       self.ui.buildGraphTransformCheckBox, self.ui.buildGraphRegisterCheckBox):
            chk_bx.setCheckState(state)  # TODO: check that not tristate

    def run_all(self):
        """Run the whole vasculature pipeline at once"""
        try:
            # FIXME: ask binary_vessel_processor about channels
            self.binarize_channel(self.binary_vessel_processor.all_vessels_channel, stop_on_error=True)
            self.binarize_channel(self.binary_vessel_processor.arteries_channel, stop_on_error=True)
        except ClearMapVRamException:
            return
        self.combine()
        self.build_graph()
        self.post_process_graph()
        self.voxelize()

    def build_graph(self):
        """Run the pipeline to build the vasculature graph"""
        # TODO: n_steps = 4
        tile = 'Building vessel graph'
        self.wrap_step(tile, self.vessel_graph_processor.skeletonize_and_build_graph,
                       abort_func=self.vessel_graph_processor.stop_process, main_thread=True)
        self.wrap_step(tile, self.vessel_graph_processor.clean_graph,
                       abort_func=self.vessel_graph_processor.stop_process)
        self.wrap_step(tile, self.vessel_graph_processor.reduce_graph,
                       abort_func=self.vessel_graph_processor.stop_process)
        self.wrap_step(tile, self.vessel_graph_processor.register,
                       abort_func=self.vessel_graph_processor.stop_process)

    def plot_graph_type_processing_chunk_slicer(self):  # Refactor: rename
        """
        Plot the ortho-slicer to pick a sub part of the graph to display because
        depending on the display options, the whole graph may not fit in memory
        """
        self.plot_slicer('graphConstructionSlicer', self.ui, self.params.visualization_params,
                         channel=self.vessel_graph_processor.parent_channels)
        # TODO: check iif best option is to
        #   average the parent channels

    def display_graph_chunk(self, graph_step):
        """
        Display a chunk of the graph selected with the slicer

        Parameters
        ----------
        graph_step : str
            The name of the step to display (from 'raw', 'cleaned', 'reduced', 'annotated')
        """
        self.params.visualization_params.ui_to_cfg()  # Fix for lack of binding between 2 sets of range interfaces
        self.wrap_plot(self.vessel_graph_processor.visualize_graph_annotations,
                       self.params.visualization_params.slicing,
                       plot_type='mesh', graph_step=graph_step, show=False)
        self.main_window.perf_monitor.stop()

    def display_graph_chunk_from_cfg(self):  # REFACTOR: split ?
        self.display_graph_chunk(self.params.visualization_params.graph_step)

    def plot_graph_structure(self):
        """
        Plot a subregion of the vasculature graph corresponding to a structure
        using the atlas registration results
        """
        structure_id = self.params.visualization_params.structure_id
        if structure_id is not None:
            annotator = self.aligner.annotators['atlas']  # TODO: check but atlas should be OK to just do lookup
            color = annotator.find(structure_id, key='id')['rgb']
            self._plot_graph_structure(structure_id, color)
        else:
            print('No structure ID')
        self.main_window.structure_selector.close()

    def post_process_graph(self):
        """Post process the graph by filtering, tracing and removing capillaries """
        self.wrap_step('Post processing vasculature graph', self.vessel_graph_processor.post_process,
                       abort_func=self.vessel_graph_processor.stop_process)  # TODO: n_steps = 8

    def pick_region(self):
        """Open a dialog to select a brain region and plot it """
        picker = self.main_window.structure_selector
        picker.structure_selected.connect(self.params.visualization_params.set_structure_id)
        picker.onAccepted(self.plot_graph_structure)
        picker.onRejected(picker.close)
        picker.show()

    def _plot_graph_structure(self, structure_id, structure_color):
        dvs = self.wrap_plot(self.vessel_graph_processor.plot_graph_structure,
                       structure_id, self.params.visualization_params.plot_type)
        if dvs:
            self.main_window.perf_monitor.stop()

    def voxelize(self):
        """Run the voxelization (density map) on the vasculature graph """
        voxelization_params = {
            'weight_by_radius': self.params.visualization_params.weight_by_radius,
            'vertex_degrees': self.params.visualization_params.vertex_degrees
        }
        self.wrap_step('Running voxelization', self.vessel_graph_processor.voxelize,
                       step_kw_args=voxelization_params)#, main_thread=True)

    def plot_voxelization(self):
        """Plot the density map """
        self.wrap_plot(self.vessel_graph_processor.plot_voxelization, self.main_window.centralWidget())

    def save_stats(self):
        """Save the stats of the graph to a feather file"""
        self.wrap_step('Saving stats', self.vessel_graph_processor.write_vertex_table)


class TractMapTab(PostProcessingTab):
    """
    The tab responsible for the tract map processing and visualization.
    """
    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'tract_map_tab', tab_idx)
        self.sample_manager = sample_manager
        self.aligner = None

        self.tract_mappers = {}

        self.channels_ui_name = 'tract_map_params'

        # self.advanced_controls_names = ['channel.tractMapAdvancedGroupBox']

        self.set_relevant_data_types(DATA_TYPE_TO_TAB_CLASS)

    def _set_channels_names(self):
        """Patch the config with the actual channels if not already done"""
        if not set(self.params.config['channels'].keys()) <= set(self.sample_manager.channels):
            default_channel_config = deepcopy(self.params.config['channels']['example'])
            self.params.config['channels'] = {}
            for channel in self.params.channels_to_process:
                self.params.config['channels'][channel] = default_channel_config
            self.params.config.write()
            self._setup_workers()

    def _bind(self):
        pass

    def _bind_params_signals(self):
        pass

    def _set_params(self):
        self.params = TractMapParams(self.ui, self.sample_params, None,
                                     self.main_window.tab_managers['registration'].params)

    def _get_channels(self):
        return self.params.channels_to_process

    def _setup_workers(self):
        if self.sample_manager.workspace is not None:
            if self.config_initialised():
                self.params.ui_to_cfg()
                self.tract_mappers = {channel: TractMapProcessor(self.sample_manager, channel)
                                  for channel in self.params.channels_to_process}
            else:
                self.main_window.print_warning_msg('Channels not yet set in config. Cannot define workers.')
        else:
            self.main_window.print_warning_msg('Workspace not initialised')

    def config_initialised(self):
        return 'example' not in self.params.config['channels']

    def finalise_workers_setup(self):
        is_first_channel = True
        for channel, processor in self.tract_mappers.items():
            if processor.registration_processor is None and self.aligner is not None:
                if is_first_channel:
                    self.params.ui_to_cfg()
                    is_first_channel = False
                processor.setup(self.sample_manager, channel, self.aligner)

    def _set_channel_config(self, channel):
        pass

    def _bind_channel(self, page_widget, channel):
        buttons_functions = [
            ('tractMapComputeClippRangePushButton', self.compute_clipping_range),
            ('tractMapComputePixelsPercentRangePushButton', self.intensities_to_percentiles),
            ('tractMapPlotBinarizationThresholdsPushButton', self.plot_binarization_thresholds),
            ('tractMapPreviewTuningOpenPushButton', self.plot_debug_cropping_interface),
            ('tractMapPreviewTuningCropPushButton', self.create_tract_map_tuning_sample),
            ('tractMapPreviewPushButton', self.run_tuning_tract_map),
            ('runTractMapPushButton', self.run_tract_map),
            ('tractMapPlotBinaryPushButton', self.plot_binary),
            ('tractMapPlotVoxelizationPushButton', self.plot_tract_map_results),
            ('tractMap3dScatterOnRefPushButton', functools.partial(self.plot_labeled_tracts_scatter, raw=False)),
            ('tractMap3dScatterOnStitchedPushButton', functools.partial(self.plot_labeled_tracts_scatter, raw=True)),
            # TODO: self.voxelize
        ]
        for btn_name, func in buttons_functions:
            self._bind_btn(btn_name, func, channel, page_widget)

    def set_progress_watcher(self, watcher):
        for processor in self.tract_mappers.values():
            if processor is not None and processor.sample_manager is not None:
                processor.set_progress_watcher(watcher)

    def run_tuning_tract_map(self, channel):
        # self.run_tract_map(channel, tuning=True)
        self.ui.channelsParamsTabWidget.get_channel_widget(channel).toolBox.setCurrentIndex(3)

    def run_tract_map(self, channel):
        tuning = self.ui.channelsParamsTabWidget.get_channel_widget(channel).tractMapStepsUseDebugCheckBox.isChecked()
        processor = self.tract_mappers[channel]
        if tuning:
            status_backup = processor.workspace.debug
        processor.workspace.debug = tuning
        # self.wrap_step('Running tract map', processor.run_pipeline,
        #                step_kw_args={'tuning': tuning},
        #                abort_func=processor.stop_process)
        self.run_channel(channel, tuning=tuning)
        if processor.stopped:
            return
        if tuning:
            processor.workspace.debug = status_backup


    def binarize_channel(self, channel):
        processor = self.tract_mappers[channel]
        self.wrap_step('Binarization', processor.binarize,
                       step_args=self.params[channel].clip_range,
                       abort_func=processor.stop_process)

    def extract_coordinates(self, channel, tuning):
        processor = self.tract_mappers[channel]
        self.wrap_step('Extracting coordinates', processor.mask_to_coordinates,
                       step_kw_args={'as_memmap': True},
                       abort_func=processor.stop_process)
        if tuning:
            processor.shift_coordinates()

    def transform_coordinates(self, channel):
        processor = self.tract_mappers[channel]
        self.wrap_step('Transforming coordinates', processor.parallel_transform,
                       abort_func=processor.stop_process)

    def label_coordinates(self, channel):
        processor = self.tract_mappers[channel]
        self.wrap_step('Labeling coordinates', processor.label,
                       abort_func=processor.stop_process)

    def export_df(self, channel):
        processor = self.tract_mappers[channel]
        self.wrap_step('Exporting coordinates', processor.export_df,
                       step_kw_args={'asset_sub_type': None},
                       abort_func=processor.stop_process)

    def run_channel(self, channel, tuning):  # FIXME: missing calls to wrap step
        self.params.ui_to_cfg()
        params = self.params[channel]
        processor = self.tract_mappers[channel]
        processor.reload_config()
        if params.binarize:
            self.binarize_channel(channel)
        if params.extract_coordinates:
            self.extract_coordinates(channel, tuning=tuning)  # mask_to_coordinates
        if params.transform_coordinates:
            self.transform_coordinates(channel)
        if params.label_coordinates:
            self.label_coordinates(channel)
        if params.voxelize:
            self.voxelize(channel)
        if params.export_df:
            processor.export_df()

    def voxelize(self, channel):
        if self.tract_mappers[channel].get('binary', asset_sub_type='coordinates_transformed', channel=channel).exists:
            self.wrap_step('Voxelization', self.tract_mappers[channel].voxelize,
                           abort_func=self.tract_mappers[channel].stop_process, nested=False)
        else:
            self.main_window.popup('Could not run voxelization, missing transformed coordinates. ',
                                   base_msg='Missing file')

    def plot_debug_cropping_interface(self, channel):
        """
        Plot the orthoslicer to select a subset of the sample to perform tracts detections
        tests on
        """
        self.plot_slicer('detectionSubset', self.ui.channelsParamsTabWidget.get_channel_widget(channel),
                         self.params[channel], channel)

    def create_tract_map_tuning_sample(self, channel):
        """Create an array from a subset of the sample to perform tests on """
        processor = self.tract_mappers[channel]
        self.wrap_step('Creating tuning sample', processor.create_test_dataset,
                       step_kw_args={'slicing': self.params[channel].slicing}, nested=False)
        self.sample_manager.workspace.debug = False  # FIXME

    def compute_clipping_range(self, channel):
        processor = self.tract_mappers[channel]
        # TODO: use wrap_step but must include return
        pixel_percents = self.params[channel].clipping_percents
        self.params.ui_to_cfg()
        low_intensity, high_intensity = processor.compute_clip_range(pixel_percents)
        self.params[channel].clip_range = [low_intensity, high_intensity]

    def intensities_to_percentiles(self, channel):
        """
        Convert the intensities to percentiles
        """
        processor = self.tract_mappers[channel]
        low_intensity, high_intensity = self.params[channel].clip_range
        self.params.ui_to_cfg()
        low_percent, high_percent = processor.intensities_to_percentiles(low_intensity, high_intensity)
        self.params[channel].clipping_percents = [low_percent, high_percent]

    def plot_binary(self, channel):
        page = self.ui.channelsParamsTabWidget.currentWidget()
        debug = page.tractMapDebugCheckBox.isChecked()
        self.wrap_plot(self.tract_mappers[channel].plot_binary, debug=debug)

    def plot_binarization_thresholds(self, channel):
        page = self.ui.channelsParamsTabWidget.currentWidget()
        low_level_spin_box = page.binarizationThresholdsLowSpinBox_1
        high_level_spin_box = page.binarizationThresholdsHighSpinBox_2
        self.wrap_plot(self.tract_mappers[channel].plot_binarization_levels,
                       low_level_spin_box, high_level_spin_box)

    def plot_tract_map_results(self, channel):
        self.wrap_plot(self.tract_mappers[channel].plot_voxelized_counts)

    def plot_labeled_tracts_scatter(self, channel, raw=False):
        self.main_window.clear_plots()
        tract_mapper = self.tract_mappers[channel]
        page = self.ui.channelsParamsTabWidget.get_channel_widget(channel)
        coords_source_is_debug = page.tractMapDebugCheckBox.isChecked()
        coords_target_is_debug = page.tractMapTargetDebugCheckBox.isChecked()
        self.wrap_plot(tract_mapper.plot_tracts_3d_scatter_w_atlas_colors, raw=raw,
                       coordinates_from_debug=coords_source_is_debug,
                       plot_onto_debug=coords_target_is_debug)


class ColocalizationTab(PostProcessingTab):
    def __init__(self, main_window, tab_idx, sample_manager, cell_tab: CellCounterTab | None = None):
        super().__init__(main_window, 'colocalization_tab', tab_idx)
        self.sample_manager = sample_manager
        self.colocalization_processors = {}
        if cell_tab is None:
            self.main_window.print_warning_msg('Cell tab not passed to colocalization tab')
        self.cell_tab = cell_tab
        self.channels_ui_name = 'colocalization_params'
        # self._setup_colocalization_pairs()

    def _set_params(self):
        self.params = ColocalizationParams(self.ui, self.sample_params, None)

    def _get_channels(self):
        # Create combinations (e.g., [('Ch0','Ch1'), ('Ch0','Ch2'), ('Ch1','Ch2')])
        return itertools.combinations(self.sample_manager.channels_to_detect, 2)

    def _set_channels_names(self):
        if self.config_initialized():
            return
        default_channel_config = deepcopy(self.params.config['channels'].get('example'))
        if default_channel_config:
            self.params.config['channels'] = {}
            for pair in self._get_channels():
                pair_str = ('-'.join(pair)).lower()
                if pair_str not in self.params.config['channels']:
                    self.params.config['channels'][pair_str] = default_channel_config
            self.params.config.write()
            self._setup_workers()  # WARNING: a bit circular but if we need to amend channels
                                   #    then we also reset the workers

    def config_initialized(self):
        return 'example' not in self.params.config['channels']

    def _bind(self):
        pass  # OK

    def _bind_params_signals(self):
        pass  # OK

    def _setup_workers(self):
        if self.sample_manager.workspace is not None:
            if self.config_initialized():  # FIXME: we need to ensure the ui has already been loaded
                # self.params.ui_to_cfg()
                for pair in self._get_channels():
                    if pair not in self.colocalization_processors:
                        self.colocalization_processors[pair] = ColocalizationProcessor(
                            sample_manager=self.sample_manager,
                            channels=pair)
                    elif not self.colocalization_processors[pair].setup_finalised:
                        self.colocalization_processors[pair].finalise_setup()
            else:
                self.main_window.print_warning_msg(f'Channels not yet set in config. '
                                                   f'Cannot define workers for {self.__class__}.')
        else:
            self.main_window.print_warning_msg('Workspace not initialised')

    def finalise_workers_setup(self):  # REFACTORING: see if could be on_tab_clicked
        pass # FIXME: assert that all channels detected
        # is_first_channel = True
        # for pair, processor in self.colocalization_processors.items():
        #     if processor.registration_processor is None and self.aligner is not None:  # TODO: might be redundant with setup_workers
        #         if is_first_channel:
        #             self.params.ui_to_cfg()
        #             is_first_channel = False
        #         processor.setup(self.sample_manager, pair)

    def _set_channel_config(self, channel):
        """Add a new channel tab to the UI"""
        pass  # OK

    def _create_channels(self):  # WARNING: override necessary because we have pairs of channels
        if not hasattr(self.ui, 'channelsParamsTabWidget'):
            return
        if not isinstance(self.ui.channelsParamsTabWidget, ExtendableTabWidget):
            warnings.warn(f'Channel tab widget not finalised for  {self.name}, skipping channel creation')
            return
        for pair in self._get_channels():
            channels_names_str = ('-'.join(pair)).lower()
            if channels_names_str not in self.ui.channelsParamsTabWidget.get_channels_names():
                self.add_channel_tab(channels_names_str)

    def _bind_channel(self, page_widget, channel):
        channel_a, channel_b = channel.split('-')
        buttons_functions = [
            ('colocalizationRunPushButton', functools.partial(self.run_colocalization_for_pair,
                                                              channel_a=channel_a,
                                                              channel_b=channel_b)),  # TODO: add load icon
            ('colocalizationPlotPushButton', functools.partial(self.plot,
                                                               channel_a=channel_a,
                                                               channel_b=channel_b)),
            ('colocalizationPlotSaveFilteredTablePushButton', functools.partial(self.save_filtered_table,
                                                                                channel_a=channel_a,
                                                                                channel_b=channel_b)),
            ('colocalizationVoxelizeFilteredTablePushButton', functools.partial(self.voxelize_filtered_table,
                                                                                channel_a=channel_a,
                                                                                channel_b=channel_b)),
        ]
        for btn_name, func in buttons_functions:
            getattr(page_widget, btn_name).clicked.connect(func)

        group = QButtonGroup(page_widget)
        group.addButton(page_widget.colocalizationChannelAFirstRadioButton)
        group.addButton(page_widget.colocalizationChannelBFirstRadioButton)
        group.setExclusive(True)

    def sort_channels(self, channel_a, channel_b):
        """
        Return the channels ordered based on the state of the First channel radio buttons

        Parameters
        ----------
        channel_a: str
            The name of the first channel
        channel_b: str
            The name of the second channel

        Returns
        -------
        List[str]
            The ordered list of channels where the first channel is the one selected by the user
        """
        page_widget = self.ui.channelsParamsTabWidget.get_channel_widget(f'{channel_a}-{channel_b}')
        if page_widget.colocalizationChannelAFirstRadioButton.isChecked():
            return channel_a, channel_b
        else:
            return channel_b, channel_a

    def run_colocalization_for_pair(self, channel_a, channel_b):
        self._setup_workers()  # FIXME: see why not already setup
        self.params.ui_to_cfg()
        processor = self.colocalization_processors.get((channel_a, channel_b))
        if processor:
            processor.compute_colocalization(*self.sort_channels(channel_a, channel_b))


    def plot(self, channel_a, channel_b):
        self.params.ui_to_cfg()
        processor = self.colocalization_processors.get((channel_a, channel_b))
        if processor:
            sorted_chan_a, sorted_chan_b = self.sort_channels(channel_a, channel_b)
            self.wrap_plot(processor.plot_nearest_neighbors, channel_a=sorted_chan_a, channel_b=sorted_chan_b)

    def save_filtered_table(self, channel_a, channel_b):
        self.params.ui_to_cfg()
        processor = self.colocalization_processors.get((channel_a, channel_b))
        if processor:
            processor.save_filtered_table(*self.sort_channels(channel_a, channel_b))

    def voxelize_filtered_table(self, channel_a, channel_b):
        self.params.ui_to_cfg()
        processor = self.colocalization_processors.get((channel_a, channel_b))
        if processor:
            processor.voxelize_filtered_table(*self.sort_channels(channel_a, channel_b))


################################################################################################


class GroupAnalysisProcessor:
    def __init__(self, progress_watcher, results_folder=None):
        self.results_folder = Path(results_folder) if results_folder is not None else None
        self.progress_watcher = progress_watcher

    def plot_p_vals(self, selected_comparisons, groups, channel, parent=None):
        p_vals_imgs = []
        results_folder = Path(self.results_folder)
        for pair in selected_comparisons:  # TODO: Move to processor object to be wrapped
            gp1_name, gp2_name = pair
            # Reread because of cm_io orientation
            p_val_path = results_folder / f'{channel}_p_val_colors_{gp1_name}_{gp2_name}.tif'

            p_vals_imgs.append(clm_io.read(p_val_path))
        folder = results_folder / groups[selected_comparisons[0][0]][0]
        processors = init_sample_manager_and_processors(folder)
        registration_processor = processors['registration_processor']
        if len(p_vals_imgs) == 1:
            gp1_name, gp2_name = selected_comparisons[0]
            gp1_avg = clm_io.read(results_folder / f'{channel}_avg_density_{gp1_name}.tif')
            gp1_sd_path = results_folder / f'{channel}_sd_density_{gp1_name}.tif'
            gp2_avg = clm_io.read(results_folder / f'{channel}_avg_density_{gp2_name}.tif')
            gp2_sd_path = results_folder / f'{channel}_sd_density_{gp2_name}.tif'
            colored_atlas = registration_processor.annotators[channel].create_color_annotation()
            gp1_imgs = gp1_avg
            if gp1_sd_path.exists():
                gp1_sd = clm_io.read(gp1_sd_path)
                gp1_imgs = [gp1_avg, gp1_sd]
            gp2_imgs = gp2_avg
            if gp2_sd_path.exists():
                gp2_sd = clm_io.read(gp2_sd_path)
                gp2_imgs = [gp2_avg, gp2_sd]
            stats_imgs = p_vals_imgs[0]
            stats_title = 'P values'
            stats_luts = None
            effect_size_path = results_folder / f'{channel}_effect_size_{gp1_name}_{gp2_name}.tif'
            if effect_size_path.exists():
                stats_imgs = [stats_imgs, clm_io.read(effect_size_path)]
                stats_title += ' and effect size'
                stats_luts = [None, 'flame']
            images = [gp1_imgs, gp2_imgs, stats_imgs, colored_atlas]
            titles = [gp1_name, gp2_name, stats_title, 'colored_atlas']
            luts = ['flame', 'flame', stats_luts, None]
            min_maxes = [None, None, None, (0, 255)]
        else:
            images = p_vals_imgs
            titles = [f'{gp1_name} vs {gp2_name} p values' for gp1_name, gp2_name in selected_comparisons]
            luts = None
            min_maxes = None
        dvs = plot_3d.plot(images, title=titles, arrange=False, sync=True,
                           lut=luts, min_max=min_maxes,
                           parent=parent)
        names_map = registration_processor.annotators[channel].get_names_map()  # FIXME: AttributeError("'RegistrationProcessor' object has no attribute 'annotator'")
        for dv in dvs:
            dv.atlas = registration_processor.annotators[channel].atlas
            dv.structure_names = names_map
        link_dataviewers_cursors(dvs)
        return dvs

    def compute_p_vals(self, selected_comparisons, groups, wrapping_func, channels, advanced=False):
        for pair in selected_comparisons:
            gp1_name, gp2_name = pair
            gp1, gp2 = [groups[gp_name] for gp_name in pair]
            for channel in channels:
                _ = density_files_are_comparable(self.results_folder, gp1, gp2, channel)
            check_ids_are_unique(gp1, gp2)
            # compare_groups is automatically for each channel (loads the first sample to find the channels)
            wrapping_func(compare_groups, self.results_folder, gp1_name, gp2_name, gp1, gp2, advanced=advanced)
            self.progress_watcher.increment_main_progress()

    def run_plots(self, plot_function, selected_comparisons, plot_kw_args):
        dvs = []
        for pair in selected_comparisons:
            gp1_name, gp2_name = pair
            if 'group_names' in plot_kw_args.keys() and plot_kw_args['group_names'] is None:
                plot_kw_args['group_names'] = pair
            stats_df_path = self.results_folder / f'statistics_{gp1_name}_{gp2_name}.csv'
            fig = plot_function(pd.read_csv(stats_df_path), **plot_kw_args)
            browser = QWebEngineView()
            browser.setHtml(mpld3.fig_to_html(fig))  # WARNING: won't work if external objects. Then
            #   saving the html to URl (file///) and then
            #   .load(Url) would be required
            dvs.append(browser)
        return dvs

    def plot_density_maps(self, group_folders, channel, parent=None):
        density_map_paths = []
        titles = []
        for folder in group_folders:
            sample_manager = SampleManager()
            sample_manager.setup(src_dir=folder)
            map_path = sample_manager.get('density', channel=channel, asset_sub_type='counts').path
            density_map_paths.append(map_path)  # TODO: make work for tubemap too
            titles.append(sample_manager.config['sample_id'])
        luts = ['flame'] * len(density_map_paths)
        dvs = plot_3d.plot(density_map_paths, title=titles, arrange=False, sync=True, lut=luts, parent=parent)
        link_dataviewers_cursors(dvs)
        return dvs


class GroupAnalysisTab(BatchTab):
    def __init__(self, main_window, tab_idx):
        super().__init__(main_window, tab_idx)
        self.processor = GroupAnalysisProcessor(self.main_window.progress_watcher)

        self.advanced_controls_names = ['computeSdAndEffectSizeCheckBox']

    def _set_channels_names(self):
        pass  # TODO: check if required

    def _set_params(self):  # FIXME: not called before clicking "Results Folder"
        self.params = GroupAnalysisParams(self.ui, preferences=self.main_window.preference_editor.params)

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        super()._bind()
        self.ui.runPValsPushButton.clicked.connect(self.run_p_vals)
        self.ui.plotPValsPushButton.clicked.connect(self.plot_p_vals)
        self.ui.batchStatsPushButton.clicked.connect(self.make_group_stats_tables)
        self.ui.batchToolBox.currentChanged.connect(self.handle_tool_changed)

    def _setup_workers(self):
        self.processor.results_folder = self.params.results_folder
        self.processor.progress_watcher = self.main_window.progress_watcher

    # def _setup_workers(self):
    #     self.processor = BatchProcessor(self.params.config)

    def handle_tool_changed(self, idx):
        if idx == 1:  # Density maps
            for i, gp in enumerate(self.params.group_names):
                self.params.plot_density_maps_buttons[i].clicked.connect(
                    functools.partial(self.plot_density_maps, gp))

    def get_analysable_channels(self):
        """
        List the channels that have density maps available for analysis

        .. warning:: This method assumes that all the samples have the same channels

        Returns
        -------
        list of str
            The list of channels that have density maps available
        """
        # Use the very first sample as template for the others
        sample_manager = SampleManager()
        sample_manager.setup(src_dir=self.params.get_all_paths()[0][0])

        analysable_channels = []
        for channel in sample_manager.channels:
            asset = sample_manager.get('density', channel=channel, asset_sub_type='counts', default=None)
            if asset is not None and asset.exists:
                analysable_channels.append(channel)
        return analysable_channels

    def plot_density_maps(self, group_name):
        self.params.ui_to_cfg()
        self.main_window.print_status_msg('Plotting density maps')

        self.main_window.clear_plots()  # TODO: use wrap_plot
        dvs = self.processor.plot_density_maps(self.params.groups[group_name],
                                               channel=self.params.plot_channel,
                                               parent=self.main_window.centralWidget())
        self.main_window.setup_plots(dvs)

    def run_p_vals(self):
        self.params.ui_to_cfg()

        self.main_window.print_status_msg('Computing p_val maps')
        # TODO: set abort callback
        self.main_window.make_progress_dialog('P value maps', n_steps=len(self.params.selected_comparisons))
        try:
            self.processor.compute_p_vals(self.params.selected_comparisons, self.params.groups,
                                          self.main_window.wrap_in_thread,
                                          channels=self.get_analysable_channels(),
                                          advanced=self.params.compute_sd_and_effect_size)
        except GroupStatsError as err:
            self.main_window.popup(str(err), base_msg='Cannot proceed with analysis')
        self.main_window.signal_process_finished()

    def make_group_stats_tables(self):
        self.main_window.print_status_msg('Computing stats table')
        self.main_window.clear_plots()
        # TODO: set abort callback
        self.main_window.make_progress_dialog('Group stats', n_steps=len(self.params.selected_comparisons))
        dvs = []
        for gp1_name, gp2_name in self.params.selected_comparisons:
            dfs = self.main_window.wrap_in_thread(make_summary, self.params.results_folder,
                                                 gp1_name, gp2_name,
                                                 self.params.groups[gp1_name], self.params.groups[gp2_name],
                                                 output_path=None, save=True)
            self.main_window.progress_watcher.increment_main_progress()
            dvs.append(DataFrameWidget(dfs[self.params.plot_channel]).table)
        self.main_window.setup_plots(dvs)
        self.main_window.signal_process_finished()

    def plot_p_vals(self):
        self.main_window.clear_plots()
        self.main_window.print_status_msg('Plotting p_val maps')
        dvs = self.processor.plot_p_vals(self.params.selected_comparisons, self.params.groups,
                                         channel=self.params.plot_channel,
                                         parent=self.main_window.centralWidget())
        self.main_window.setup_plots(dvs)  # TODO: use wrap_plot

    def run_plots(self, plot_function, plot_kw_args):
        self.main_window.clear_plots()
        dvs = self.processor.run_plots(plot_function, self.params.selected_comparisons, plot_kw_args)
        self.main_window.setup_plots(dvs)  # TODO: use wrap_plot

    def plot_volcanoes(self):
        self.run_plots(plot_volcano, {'group_names': None, 'p_cutoff': 0.05, 'show': False, 'save_path': ''})

    def plot_histograms(self, fold_threshold=2):
        folder = self.params.results_folder / self.params.groups[self.params.group_names[0]][0] # FIXME: check if absolute
        processors = init_sample_manager_and_processors(folder)
        registration_processor = processors['registration_processor']
        annotator = registration_processor.annotators[self.params.plot_channel]
        aba_df = annotator.df
        # aba_json_df_path = annotation.default_label_file  # FIXME: aba_json needs fold levels
        self.run_plots(plot_sample_stats_histogram, {'aba_df': pd.read_csv(aba_df),
                                                     'sort_by_order': True, 'value_cutoff': 0,
                                                     'fold_threshold': fold_threshold, 'fold_regions': True,
                                                     'show': False})


class BatchProcessingTab(BatchTab):
    def __init__(self, main_window, tab_idx):
        super().__init__(main_window, tab_idx)
        self.processor = BatchProcessor(self.main_window.progress_watcher)

    def _set_channels_names(self):
        pass  # TODO: check if required

    def _set_params(self):
        self.params = BatchProcessingParams(self.ui, preferences=self.main_window.preference_editor.params)

    def _setup_workers(self):
        self.processor.params = self.params

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        super()._bind()
        self.ui.batchRunPushButton.clicked.connect(self.run_batch_process)

    def run_batch_process(self):
        self.params.ui_to_cfg()
        self.main_window.make_progress_dialog('Analysing samples', n_steps=0, maximum=0)  # TODO: see abort callback
        self.main_window.wrap_in_thread(self.processor.process_folders)


DATA_TYPE_TO_TAB_CLASS = {  # WARNING: not all data types are covered
    'nuclei': CellCounterTab,
    'cells': CellCounterTab,
    'vessels': VasculatureTab,
    'veins': VasculatureTab,
    'arteries': VasculatureTab,
    'myelin': TractMapTab,
    'autofluorescence': RegistrationTab,
    'no-pipeline': None,
    'undefined': None,
    None: None,
}
