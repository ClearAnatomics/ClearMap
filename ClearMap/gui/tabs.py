"""
tabs
====

GUI tabs for ClearMap.

All the classes in this module are subclasses (direct or indirect) of the `ExperimentTab` and
 'GroupTab` classes which derive from `GenericTab` and provide the basic structure and methods that are common to all tabs.

Presentation
------------
Each **tab** manages a specific part of the processing pipeline and has its own UI elements.
It is composed of:
- `ui`: QWidget constructed from a `.ui` file that defines the layout and the widgets.
- `sample_manager`: handles the sample metadata and workspace.
- `sample_params`: experiment-level object that links the UI to the sample configuration file.
- `params`: tab specific parameter link object (UI - config).
(For the `SampleInfoTab`, this is the `SampleParameters` object.)
- `name`: used to identify the tab in the GUI.
- `processing_type`: identifies the type of tab, one of (None, 'pre', 'post', 'batch').

Abstract tabs Hierarchy
-----------------------
- **GenericTab**: base class for all tabs, handles the UI and channel pages.
- **ExperimentTab**: base for tabs tied to a single experiment (SampleInfo, Stitching, Registration, CellMap, TractMap, Colocalization).
- **PipelineTab**: base for Pre/Post-processing tabs.
- **PreProcessingTab / PostProcessingTab**: Specialisation of PipelineTab that form the basis for concrete tabs.
- **GroupTab**: base for group/multi-experiment tabs.
- **BatchTab**: Abstract tab for batch/group workflows (e.g., batch processing, group analysis).

Optionally, for `Pipeline` tabs (`PreProcessingTab` and `PostProcessingTab`),
a `worker` object is used to handle the processing steps and computation.

Tab setup flow
--------------

Setup order
***********

Typical calling sequence is:

- `tab.setup`
- `tab.set_params`
    - sets sample_params if not SampleInfoTab
    - calls `tab._set_params`
    - calls `tab._setup_workers` (for pipeline tabs)
    - calls `tab._create_channels`  (if the tab has channel pages)
    - calls `tab._load_config_to_gui` (via params)
    - calls `tab._bind_params_signals`  (for tab specific signals)

Tabs with channels
******************

Some tabs can have channel pages.
Channel pages are created by `add_channel_tab()` (invoked from `set_params()` and, for tabs that allow it, the (+) button).
To control the behavior of this method, the following methods can be implemented in the concrete tab classes:

- `_setup_channel(page_widget, channel)` (optional): Additional setup for the ui (before binding).
- `_bind_channel` (**required**: wire channel page specific actions which are not automatically
set through the params object attribute).

`PipelineTab` additional setup methods
**************************************

- `_setup_workers`: Sets up the worker (Processor) which handles the computations associated with this tab.
Called in `set_params` and also when the sample config is applied.

Processing steps
----------------

Pre/Post processing tabs expose worker orchestration methods.
UI widgets are wired via ParamLinks (and bus events), while buttons typically call small
wrappers that run steps in threads and update progress dialogs.
====================
"""
from __future__ import annotations

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


import functools
import itertools
import re
import warnings

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from PyQt5.QtCore import QSignalBlocker

from ClearMap.Utils.tag_expression import Expression

from PyQt5.QtWidgets import QButtonGroup, QWidget, QDialog
import pyqtgraph as pg
from natsort import natsorted

from ClearMap.Analysis.graphs.graph_filters import GraphFilter
from ClearMap.IO.assets_constants import DATA_CONTENT_TYPES, EXTENSIONS

# app = QApplication.instance()
# if app is not None and app.applicationName() == 'ClearMap':
#     from PyQt5.QtWebEngineWidgets import QWebEngineView
from qdarkstyle import DarkPalette

from ClearMap.config.atlas import ATLAS_NAMES_MAP, STRUCTURE_TREE_NAMES_MAP

from ClearMap.pipeline_orchestrators.utils import init_sample_manager_and_processors
from ClearMap.pipeline_orchestrators.batch_process import BatchProcessor

from ClearMap.Visualization.Matplotlib.PlotUtils import plot_sample_stats_histogram, plot_volcano
from ClearMap.Visualization.Qt.utils import link_dataviewers_cursors
from ClearMap.Visualization.Qt import Plot3d as plot_3d

from ClearMap.Utils.exceptions import ClearMapVRamException, GroupStatsError, MissingRequirementException, \
    ClearMapWorkspaceError
from ClearMap.Utils.events import ChannelsChanged, UiConvertToClearMapFormat, UiRequestPlotMiniBrain, \
    UiRequestPlotAtlas, UiOrientationChanged, UiCropChanged, ChannelDefaultsChanged, \
    UiRequestLandmarksDialog, UiAlignWithChanged, UiVesselGraphFiltersChanged, RegistrationStatusChanged, \
    UiBatchResultsFolderChanged, UiBatchGroupsChanged, UiChannelsChanged, WorkspaceChanged

from .dialog_helpers import option_dialog, make_splash, prompt_dialog
from .dialogs import ResourceTypeToFolderDialog
from .tabs_interfaces import PostProcessingTab, PreProcessingTab, BatchTab, ExperimentTab, GenericTab
from .widgets import (PatternDialog, DataFrameWidget, LandmarksSelectorDialog,
                      CheckableListWidget, FileDropListWidget, ExtendableTabWidget, ensure_inline_histogram,
                      GraphFilterList, NProcessesWidget, BlockProcessingWidget)
from .gui_utils_base import (format_long_nb, replace_widget, add_missing_combobox_items,
                             populate_combobox, delete_widget)
from .gui_utils_images import np_to_qpixmap
from .params import (VesselParams, SampleParameters, StitchingParams, CellMapParams, GroupAnalysisParams,
                     BatchProcessingParams, RegistrationParams, TractMapParams, ColocalizationParams)
from ClearMap.IO.metadata import parse_ome_info

if TYPE_CHECKING:
    from ClearMap.pipeline_orchestrators.experiment_controller import AnalysisGroupController
    from ClearMap.IO.metadata import ChannelPatternSpec


def ui_task_progress(title_fn, steps_fn):
    """
    Wraps a tab method that performs a task and updates the main progress.
    - title_fn(self) -> str    e.g. lambda s: "Group stats"
    - steps_fn(self) -> int    e.g. lambda s: len(s.params.selected_comparisons)
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *_, **kwargs):
            title = title_fn(self)
            self.main_window.print_status_msg(title)
            steps = steps_fn(self)
            self.main_window.make_progress_dialog(title, n_steps=steps)
            try:
                return fn(self, **kwargs)
            finally:
                self.main_window.signal_process_finished()
        return wrapper
    return deco


class SampleInfoTab(ExperimentTab):
    """
    The tab manager to define the parameters of the sample
    This refers to values that are intrinsic to the sample and the acquisition
    like resolution, orientation ...
    """
    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'sample_tab', tab_idx)
        self.sample_manager = sample_manager

        self.channels_ui_name = 'channel_params'
        self.with_add_btn = True  # Add (+) button to add channels
        self.detached = False  # WARNING: To avoid calling update when channels are setup by
                               #   the wizard

    def _set_params(self):
        exp_ctrl = self.main_window.experiment_controller
        self.params = SampleParameters(self.ui, event_bus=self._bus,
                                       get_view=exp_ctrl.get_config_view, apply_patch=exp_ctrl.apply_ui_patch)

    def _bind_params_signals(self):
        self.subscribe(UiConvertToClearMapFormat, self.convert_to_clearmap_format)
        self.subscribe(UiRequestPlotMiniBrain, self.plot_mini_brain)
        self.subscribe(UiRequestPlotAtlas, self.display_atlas)

        self.subscribe(UiOrientationChanged, self.update_atlas)
        self.subscribe(UiCropChanged, self.update_atlas)

        self.subscribe(ChannelsChanged, self._on_bus_channels_changed)
        self.subscribe(UiChannelsChanged, self._on_bus_channels_changed)

        self.subscribe(WorkspaceChanged, self._on_workspace_changed)

    def _get_channels(self):
        return self.sample_manager.channels

    # Pipeline tabs use ChannelsChanged because cfg based but here display
    def _on_bus_channels_changed(self, event: UiChannelsChanged | ChannelsChanged):
        self.reconcile_channel_pages(event.after)

    def _on_workspace_changed(self, event: WorkspaceChanged):
        if getattr(self, "params"):
            self.params.shared_sample_params.src_folder = event.exp_dir

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        ui = self.ui
        ui.channelsParamsTabWidget.addTabClicked.connect(self.add_channel_tab)

        ui.srcFolderBtn.clicked.connect(self.main_window.prompt_experiment_folder)

        ui.launchPatternWizardPushButton.clicked.connect(self.launch_pattern_wizard)
        ui.updateWorkspacePushButton.clicked.connect(self.sample_manager.update_workspace)

        ui.removeCurrentChannelToolButton.clicked.connect(self.remove_current_channel)
        ui.editWorkspaceFoldersPushButton.clicked.connect(self.edit_workspace_folders)

    def _bind_channel(self, page_widget, channel):
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute
        """
        self.params.set_painting(True)
        t = self.params[channel].data_type  # TEST: remove after
        content_types = natsorted(list(set(DATA_CONTENT_TYPES)))

        data_type_box = page_widget.dataTypeComboBox
        if data_type_box.count() != 0:
            raise RuntimeError(f'Channel page already bound, found the following items:'
                               f' {list(data_type_box.itemText(i) for i in range(data_type_box.count()))}')
        data_type_box.addItems(content_types)
        data_type_box.setCurrentText('undefined')  # FIXME: from cfg
        ext_box = page_widget.extensionComboBox
        if ext_box.count() == 0:  # REFACTOR: is the opposite even possible ?
            ext_box.addItems(EXTENSIONS['image'])
        self.params.set_painting(False)

    def remove_current_channel(self):
        """Remove the current channel from the sample"""
        self.remove_channel(self.ui.channelsParamsTabWidget.current_channel())

    def remove_channel(self, channel):
        self.params.pop(channel)

    def update_atlas(self, event):
        channel = event.channel_name
        aligner = self.exp_controller.get_worker('registration')
        try:
            aligner.update_atlas_asset(channel=channel)
        except KeyError as err:
            warnings.warn(f'Could not update atlas for channel {channel} because it is not in the workspace. '
                          f'Setting all atlases to None; {err}')
            aligner.setup_atlases()

    @property
    def src_folder(self):
        return self.main_window.src_folder

    # @src_folder.setter
    # def src_folder(self, folder):
    #     self.exp_controller.exp_dir = folder

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
        if not str(self.src_folder):
            self.main_window.popup('Please select a source folder first.')
            return
        dlg = PatternDialog(self.src_folder, self.params,
                            min_file_number=self.main_window.preference_editor.params.pattern_finder_min_n_files,
                            tile_extension=self.params.shared_sample_params.default_tile_extension)
        if  dlg.exec():
            specs = dlg.get_results()
            self._apply_pattern_specs(specs)
        self.detached = False

    def edit_workspace_folders(self):
        """
        Open a dialog to edit workspace.resource_type_to_folder and
        propagate changes via SampleManager.set_resource_type_to_folder().
        """
        if self.sample_manager is None:
            self.main_window.popup('Sample manager not available.')
            return

        if self.sample_manager.workspace is None:
            self.main_window.popup('Workspace not initialised yet.')
            return

        current = dict(self.sample_manager.workspace.resource_type_to_folder)

        dlg = ResourceTypeToFolderDialog(current, parent=self.main_window)
        if dlg.exec_() != QDialog.Accepted:
            return  # user cancelled

        new_mapping, migrate = dlg.result()

        try:
            plan = self.sample_manager.set_resource_type_to_folder(new_mapping, migrate=migrate, dry_run=False)
        except ClearMapWorkspaceError as e:
            self.main_window.popup(str(e))
            return

        status = 'updated' if plan else 'unchanged'
        self.main_window.print_status_msg(f'Workspace folder layout {status}.')

        print(self.sample_manager.workspace.info())

    def _apply_pattern_specs(self, specs: list["ChannelPatternSpec"]):
        """
        Apply the new/update channels from the Wizard to the params

        Parameters
        ----------
        specs: list of ChannelPatternSpec
            The list of channel pattern specifications to apply
        """
        gui = self.main_window.gui_controller
        gui.begin_hydration()  # WARNING: required to avoid collisions with config and UI rebuild
        try:
            desired_channels = [s.name for s in specs]
            existing_channels = list(self.params.channels)

            # Remove obsolete channels
            obsolete_channels = set(existing_channels) - set(desired_channels)
            for ch in obsolete_channels:
                self.remove_channel(ch)

            # SORT BY CHANNEL INDEX if possible to match order in .ome xml
            def _maybe_channel_index(spec: "ChannelPatternSpec") -> int | None:
                # try to extract from pattern like C00 / _C01 / channel-02 etc.
                m = re.search(r"[Cc](\d{2})", spec.pattern_relpath)
                return int(m.group(1)) if m else None

            specs_sorted = sorted(specs, key=lambda s: (_maybe_channel_index(s) is None, _maybe_channel_index(s) or 1000))

            # Add / update channels
            channels_patch = {}
            for i, pattern_spec in enumerate(specs_sorted):
                if isinstance(pattern_spec.extension, (list, tuple)):  # REFACTOR: find more elegant handling here
                    warnings.warn('Multiple extensions found, picking the first one.')
                    ext = pattern_spec.extension[0]
                else:
                    ext = pattern_spec.extension  # WARNING: Will do successive modifications. Check if we should batch update
                entry = {
                    'path': pattern_spec.pattern_relpath,
                    'data_type': pattern_spec.data_type,
                    'extension': ext,
                }
                # If we have a pattern, we can stitch:
                exp = Expression(pattern_spec.pattern_relpath)
                axes = exp.tag_names()  # e.g. ['Z', 'Y', 'X']
                first_tile = exp.string(values={axis: 0 for axis in axes})  # Ideally, pick min(axis) for each
                ome_info = parse_ome_info(Path(self.src_folder) / first_tile)
                if ome_info.get('resolution') is not None:
                    res = ome_info['resolution']
                    if (isinstance(res, (list, tuple))
                        and len(res) == 3
                        and all(v in (1, 2, 3) for v in res)):
                        entry['resolution'] = list(res)
                if ome_info.get('channels_excitation') is not None:
                    entry['wavelength'] = ome_info['channels_excitation'][i]
                channels_patch[pattern_spec.name] = entry

            # Single submit_patch → adjusters see complete data_types → correct reference resolution
            self.main_window.experiment_controller.apply_ui_patch({'sample': {'channels': channels_patch}})

            # Now create UI tabs from the fully reconciled config
            for pattern_spec in specs_sorted:
                if pattern_spec.name not in self.ui.channelsParamsTabWidget.get_channels_names():
                    self.add_channel_tab(pattern_spec.name)
                if pattern_spec.name in self.params:
                    self.params[pattern_spec.name].cfg_to_ui()

            self.publish(UiChannelsChanged(before=existing_channels, after=desired_channels))
        finally:
            gui.end_hydration()

    def plot_mini_brain(self, event: UiRequestPlotMiniBrain):
        """
        Plot the brain icon which represents the acquisition sample orientation graphically
        to help users pick the right orientation.
        """
        channel = self.params.get_channel_name(event.channel_index)
        aligner = self.exp_controller.get_worker('registration')
        if aligner.setup_complete:
            mask, proj = aligner.project_mini_brain(channel)
            self.get_channel_ui(channel).miniBrainLabel.setPixmap(np_to_qpixmap(proj, mask))
        else:
            self.sample_manager.update_workspace()
            if aligner.setup_complete:
                mask, proj = aligner.project_mini_brain(channel)
                self.get_channel_ui(channel).miniBrainLabel.setPixmap(np_to_qpixmap(proj, mask))
            else:
                warnings.warn('RegistrationProcessor not setup, cannot plot mini brain. '
                              'Please call registration_tab.finalise_set_params() first')

    def display_atlas(self, event: UiRequestPlotAtlas):
        """Plot the atlas as a grayscale image in the viewer"""
        channel = self.params.get_channel_name(event.channel_index)
        aligner = self.exp_controller.get_worker('registration')
        if aligner.setup_complete:
            self.wrap_plot(aligner.plot_atlas, channel)
        else:
            self.sample_manager.update_workspace()
            if aligner.setup_complete:
                self.wrap_plot(aligner.plot_atlas, channel)
            else:
                warnings.warn('RegistrationProcessor not setup, cannot plot atlas. '
                              'Please call registration_tab.finalise_set_params() first')

    def convert_to_clearmap_format(self, event: UiConvertToClearMapFormat):
        stitching_processor = self.exp_controller.get_worker('stitching')
        channel = event.channel_name
        if self.sample_manager.is_tiled(channel):
            stitching_processor.convert_tiles_channel(channel)
        else:
            stitching_processor.copy_or_stack(channel)


class StitchingTab(PreProcessingTab):
    """
    The tab responsible for all the alignments, including the stitching and
    aligning to the atlas.
    """
    channels_ui_name = 'stitching_params'
    pipeline_name = 'stitching'

    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'stitching_tab', tab_idx)

        self.sample_manager = sample_manager
        self.advanced_controls_names = [
            'channel.useNpyCheckBox',
        ]

    def on_selected(self):
        self.update_plotable_channels()
        if self._selected_once:
            return
        return  # TODO: reenable after testing
        chans = self._get_channels()
        sample_view = self.main_window.experiment_controller.get_config_view()['sample']['channels']
        for chan in chans:
            if sample_view[chan]['extension'] == '.ome.tif':
                if prompt_dialog('Create layout from OME metadata',
                                 f'Channel {chan} uses .ome.tif files. '
                                 f'Creating the layout from OME metadata is faster. '
                                 f'Do you want to create the layout now ?'):
                    self.worker.create_layout_from_ome(channel=chan)

    def _load_config_to_gui(self):
        desired = self._get_channels()
        self.reconcile_channel_pages(desired)
        self.params.reconcile_children_from_view()
        super()._load_config_to_gui()  # == self.params.cfg_to_ui()
        self.update_plotable_channels()

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

        self.subscribe(ChannelsChanged, self._on_bus_channels_changed)

    def _on_bus_channels_changed(self, event: ChannelsChanged):
        """update the run list (preserve checks if possible)"""
        stitchable = set(self._get_channels())
        desired = [c for c in event.after if c in stitchable]
        self.reconcile_channel_pages(desired)  # WARNING: do not rely on event.after here
                                               #   because stitching tab may fewer channels than sample tab

    def _after_channels_reconciled(self, desired_channels: list[str]) -> None:
        self._refresh_ui()
        self.update_plotable_channels()

    def _bind_params_signals(self):  # WARNING: not really params signals but hack necessary to update the UI
        self._refresh_ui()  # Force refresh on initial setup

        # self.subscribe(UiChannelsChanged, self.params.reconcile_children_from_view)
        # self.subscribe(UiChannelsChanged, self._refresh_ui)
        #   - page changes
        self.ui.channelsParamsTabWidget.currentChanged.connect(lambda _idx: self._refresh_ui())

    def _refresh_ui(self, event=None):
        """Keep layout combobox in sync w/ channels"""
        chans = self._get_channels()

        run_chans_widget = self.ui.runChannelsCheckableListWidget
        run_chans_widget.blockSignals(True)
        run_chans_widget.set_items(chans)
        for ch in chans:
            if ch in self.params.channel_params:
                run_chans_widget.set_item_checked(
                    ch, bool(self.params[ch].shared.run))
        run_chans_widget.blockSignals(False)

        # refresh the layout combobox for the active page
        active_channel = self.ui.channelsParamsTabWidget.current_channel()
        if active_channel and active_channel in self.params.channel_params:
            self.params[active_channel].shared.refresh_layout_channel_items()

    def _set_params(self):
        self.params = StitchingParams(self.ui, event_bus=self._bus,
                                      get_view=self.main_window.experiment_controller.get_config_view,
                                      apply_patch=self.main_window.experiment_controller.apply_ui_patch)
                                      # self.ui.channelsParamsTabWidget is deduced from the UI

    def _get_channels(self):
        return self.sample_manager.get_stitchable_channels()

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

    def convert_tiles(self):
        if not self.sample_manager.has_tiles():
            return
        self.wrap_step('Converting tiles', self.worker.convert_tiles, step_kw_args={'_force': True}, n_steps=0,
                       abort_func=self.worker.stop_process, save_cfg=False, nested=False)

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
        stitched = self.worker.stitch_overlay(channel, color)
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
        n_steps = self.worker.n_rigid_steps_to_run
        self.wrap_step('Stitching', self.worker.align_channel_rigid,
                       step_args=[channel], step_kw_args={'_force': True},
                       n_steps=n_steps, abort_func=self.worker.stop_process)
        overlay = [pg.image(self.worker.plot_layout(channel=channel, asset_sub_type=asset_sub_type))]
        self.main_window.setup_plots(overlay)

    def run_stitching(self):
        """Run the actual stitching steps based on the values in the config file (set from the UI)."""
        for channel in self.sample_manager.channels:  # FIXME: check if should do and if done
            if not self.sample_manager.is_tiled(channel):  # BYPASS stitching, just copy or stack
                self.wrap_step('Stitching', self.worker.copy_or_stack, step_args=[channel], )

        n_steps = self.worker.n_rigid_steps_to_run + self.worker.n_wobbly_steps_to_run
        for stitching_tree in self.worker.get_stitching_order().values():
            for channel in stitching_tree:
                cfg = self.params[channel]
                if not cfg.shared.run:
                    continue
                # REFACTOR: self.worker.should_convert_tiles(channel)
                if self.params[channel].shared.use_npy and not self.sample_manager.has_npy(channel):
                    self.convert_tiles()
                kwargs = {'n_steps': n_steps, 'abort_func': self.worker.stop_process, 'close_when_done': False}
                try:
                    if channel == cfg.shared.layout_channel and not cfg.shared.use_existing_layout:  # Used as reference
                        self.wrap_step('Stitching', self.worker.align_channel_rigid,
                                       step_args=[channel], step_kw_args={'_force': True}, **kwargs)
                        self.wrap_step(task_name='', func=self.worker.stitch_channel_wobbly, step_args=[channel],
                                       step_kw_args={'_force': cfg.stitching_rigid.skip}, **kwargs)
                    else:  # Uses other channel as reference
                        self.wrap_step('', self.worker._stitch_layout_wobbly,  # REFACTOR: private
                                       step_args=[channel],  **kwargs)
                except MissingRequirementException as err:
                    error_msg = str(err).replace("\n", "<br>")
                    self.main_window.print_status_msg(f'Skipping stitching for {channel} because of missing requirements: {error_msg}')

        self.update_plotable_channels()
        self.main_window.progress_watcher.finish()

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
        if not channels:
            return self.main_window.print_status_msg('No channels selected to plot')
        self.wrap_plot(self.worker.plot_stitching_results, channels=channels,
                       mode=mode, parent=self.main_window.centralWidget())


class RegistrationTab(PreProcessingTab):
    pipeline_name = 'registration'  # WARNING: does that belong here (multiple change locations) ?
    channels_ui_name = 'registration_params'

    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'registration_tab', tab_idx)
        self.sample_manager = sample_manager

        self.landmark_selector: Optional[LandmarksSelectorDialog] = None

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

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        self.ui.registerPushButton.clicked.connect(self.run_registration)
        self.ui.plotRegistrationResultsPushButton.clicked.connect(self.plot_registration_results)

        self.__populate_atlas_comboboxes()

        self.subscribe(ChannelsChanged, self._on_bus_channels_changed)
        self.subscribe(ChannelDefaultsChanged, self._on_bus_partner_defaults)
        self.subscribe(UiAlignWithChanged, self.handle_align_with_changed)

    def __populate_atlas_comboboxes(self):
        # Populate atlas and structure tree combo boxes from config
        add_missing_combobox_items(self.ui.atlasIdComboBox, ATLAS_NAMES_MAP.keys())
        add_missing_combobox_items(self.ui.structureTreeIdComboBox, STRUCTURE_TREE_NAMES_MAP.keys())

    def _on_bus_channels_changed(self, event: ChannelsChanged):
        self.reconcile_channel_pages(event.after)

    def _after_channels_reconciled(self, desired_channels: list[str]) -> None:
        for ch in desired_channels:
            self.__update_channel_combo_boxes(ch)
        self._update_plotable_channels()

    def _on_channel_removed(self, channel_name: str) -> None:
        if channel_name in self.params:
            self.params.pop(channel_name)
        self._update_plotable_channels()

    def _on_bus_partner_defaults(self, event: ChannelDefaultsChanged):
        channel = self.ui.channelsParamsTabWidget.current_channel()
        if not channel or channel not in event.partners:
            return
        self.__update_channel_combo_boxes(channel)

    def __update_channel_combo_boxes(self, channel, page_widget=None):
        # Update possible channels in combo boxes
        if page_widget is None:
            page_widget = self.ui.channelsParamsTabWidget.get_channel_widget(channel)
        other_channels = list(set(self.worker.channels_to_register()) - {channel})
        populate_combobox(page_widget.alignWithComboBox, [None, 'atlas'] + other_channels)
        populate_combobox(page_widget.movingChannelComboBox,
                          [None, 'atlas', 'intrinsically_aligned'] + other_channels + [channel])

        partner_channel = self.worker.get_align_with(channel)
        moving_channel = self.worker.get_moving_channel(channel)

        # Guards to avoid triggering adjusters too often  # FIXME: check these are not overly harsh: TEST
        with QSignalBlocker(page_widget.alignWithComboBox):
            page_widget.alignWithComboBox.setCurrentText(partner_channel)
        with QSignalBlocker(page_widget.movingChannelComboBox):
            page_widget.movingChannelComboBox.setCurrentText(moving_channel)

    def _bind_params_signals(self):
        self.subscribe(UiRequestLandmarksDialog, self.launch_landmarks_dialog)
        self.subscribe(RegistrationStatusChanged, self._update_plotable_channels)
        for channel in self.worker.channels:
            self.__update_channel_combo_boxes(channel)
        self._update_plotable_channels()

    def _set_params(self):
        self.params = RegistrationParams(self.ui, event_bus=self._bus,
                                         get_view=self.main_window.experiment_controller.get_config_view,
                                         apply_patch=self.main_window.experiment_controller.apply_ui_patch)

    def _get_channels(self):
        return self.sample_manager.channels  # All channels so we can decide whether to register in UI

    def _setup_channel(self, page_widget, channel):
        self.__update_channel_combo_boxes(channel, page_widget)
        # alignment_files = [page_widget.paramsFilesListWidget.item(i).text() for i in
        #                    range(page_widget.paramsFilesListWidget.count())]  # no shortcut for standard QListWidget
        page_widget.paramsFilesListWidget = replace_widget(page_widget.paramsFilesListWidget,
                                                           FileDropListWidget(page_widget,
                                                                              page_widget.addParamFilePushButton,
                                                                              page_widget.removeParamFilePushButton),
                                                           page_widget.registrationChannelGridLayout)
        # page_widget.paramsFilesListWidget.addItems(alignment_files)  # Transfer existing files to new widget

    def _bind_channel(self, page_widget, channel):
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute
        """
        # TODO: set value of comboboxes to good defaults
        page_widget.paramsFilesListWidget.itemsChanged.connect(self.params[channel].handle_params_files_changed)
        # self.params[channel].handle_params_files_changed()  # Force update
        page_widget.registrationRunResamplingPushButton.clicked.connect(
            functools.partial(self.resample_channel, channel))

    def setup_atlas(self):  # TODO: check if we delay this to update_workspace
        """Setup the atlas that corresponds to the orientation and cropping of the sample"""
        self.worker.setup_atlases()

    def clear_landmarks(self, channel):
        self.worker.clear_landmarks(channel)
        # TODO: use landmark_selector

    def launch_landmarks_dialog(self, channel):
        if isinstance(channel, int):
            channel = self.params.get_channel_name(channel)
        # We have to keep reference to make it persistent but should be per channel
        self.landmark_selector = LandmarksSelectorDialog(
            fixed_image_path=self.worker.get_fixed_image(channel).path,
            moving_image_path=self.worker.get_moving_image(channel).path,
            fixed_image_landmarks_path=self.worker.get_elx_asset('fixed_landmarks',
                                                                  channel=channel).path,
            moving_image_landmarks_path=self.worker.get_elx_asset('moving_landmarks',
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

    def handle_align_with_changed(self, event: UiAlignWithChanged):
        channel = event.channel_name
        align_with = event.align_with
        if align_with == channel:
            raise ValueError(f'Cannot align {channel=} with itself')  # FIXME: popup instead of crashing whole app (QT anyway)
        if align_with is None:
            # Remove registration pipeline from channel
            return

        sample_mgr = self.sample_manager
        workspace = sample_mgr.workspace
        if not sample_mgr.setup_complete:
            warnings.warn('Workspace not setup, cannot add registration pipeline')
            return

        if channel in workspace:
            workspace.ensure_pipeline('registration', channel_id=channel,
                                      sample_id=sample_mgr.prefix, create_channel=False)
        else:  # Try from sample config
            try:
                content_type = sample_mgr.data_type(channel)
            except KeyError:
                warnings.warn(
                    f'Channel "{channel}" not found in sample config; cannot create registration pipeline')
                return

            if not content_type or content_type == 'undefined':
                warnings.warn(f'Channel "{channel}" has undefined data_type; '
                              f'cannot create registration pipeline before data_type is set.')
                return

            # Create logical channel + registration assets in workspace
            workspace.ensure_pipeline('registration', channel_id=channel, sample_id=sample_mgr.prefix,
                                      create_channel=True, channel_content_type=content_type)
        self.worker.parametrize_assets()

    def resample_channel(self, channel):
        self.main_window.make_progress_dialog('Registering', n_steps=2, abort=self.worker.stop_process,
                                              parent=self.main_window)
        self.setup_atlas()
        self.main_window.progress_watcher.increment_main_progress()
        self.sample_manager.delete_resampled_files(channel)
        self.wrap_step(f'Resampling {channel} for registration', self.worker.resample_channel,
                       step_kw_args={'channel': channel, 'increment_main': False})
        self.main_window.progress_watcher.finish()
        self.main_window.print_status_msg(f'Channel {channel} resampled for registration')

    def run_registration(self):
        """
        Run the actual registration between the sample and the reference atlas.
        """
        # TODO: compute n_steps (part of processor; n_channels * n_steps_per_channel)
        self.main_window.make_progress_dialog('Registering', n_steps=4, abort=self.worker.stop_process,
                                              parent=self.main_window)
        self.setup_atlas()
        for i, channel in enumerate(self.params.keys()):
            if self.params[channel].resample:
                asset = self.worker.get('stitched', channel=channel)
                if not asset.exists:
                    asset = self.worker.get('raw', channel=channel)
                    if asset.is_tiled and not asset.is_complete:
                        self.main_window.progress_watcher.finish()
                        self.main_window.print_status_msg(f'Registration skipped because of missing tiles'
                                                          f'for channel {channel}')
                        return
                try:
                    self.wrap_step(f'Resampling {channel} for registration', self.worker.resample_channel,
                                   step_kw_args={'channel': channel, 'increment_main':(i != 0)})
                except FileExistsError:  # REFACTOR: factorise with the above
                    option_idx = option_dialog('Files exist',
                                               f'Resampled files exist for {channel}, do you want to: ',
                                               ['Delete and retry', 'Skip resampling and continue'])
                    if option_idx == 0:
                        self.sample_manager.delete_resampled_files(channel)
                        self.wrap_step(f'Resampling {channel} for registration', self.worker.resample_channel,
                                       step_kw_args={'channel': channel, 'increment_main': (i != 0)})
                    else:
                        continue
        self.main_window.wrap_in_thread(self.worker.align)
        self.main_window.print_status_msg('Registered')

    def _update_plotable_channels(self, event=None):
        if not self.sample_manager.setup_complete:
            return

        registered_channels = [ch for ch in self.params.keys() if self.worker.channel_was_registered(ch)]
        populate_combobox(self.ui.plotChannelComboBox, registered_channels)

    def plot_registration_results(self):
        """
        Plot the result of the registration between 2 channels. Either side by side or as a composite

        If the composite checkbox is checked, the two images are overlayed.
        Otherwise, they are displayed side by side
        """
        channel = self.params.shared_params.plot_channel
        composite = self.params.shared_params.plot_composite
        self.main_window.clear_plots()
        dvs, titles = self.worker.plot_registration_results(
            channel=channel, composite=composite, parent=self.main_window.centralWidget())
        self.main_window.setup_plots(dvs, graph_names=titles)
        if not composite:
            link_dataviewers_cursors(dvs)


class CellCounterTab(PostProcessingTab):
    """
    The tab responsible for the cell detection and cell coordinates alignment
    """
    channels_ui_name = 'cell_map_params'
    pipeline_name = 'cell_map'
    workers_are_global = False

    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'cell_map_tab', tab_idx)

        self.sample_manager = sample_manager

        self.cell_intensity_histogram = None
        self.cell_size_histogram = None

        self.advanced_controls_names = [
            'channel.detectionShapeGroupBox',
            'channel.hMaxSinglet',
            'channel.cellMapPerformanceGroupBox',
        ]

    def on_selected(self):
        if not self.sample_manager.workspace:
            self.main_window.print_warning_msg("Workspace not initialised")
            return
        for ch in self._get_channels():
            self.update_cell_number(ch)

    def _bind(self):
        pass

    def _bind_params_signals(self):  # Execute at the end of finalise_set_params
        pass

    def _set_params(self):
        self.params = CellMapParams(self.ui, self.sample_params, event_bus=self._bus,
                                    get_view=self.main_window.experiment_controller.get_config_view,
                                    apply_patch=self.main_window.experiment_controller.apply_ui_patch)

    def _get_channels(self):
        return self.params.relevant_channels

    def _setup_channel(self, page_widget: QWidget, channel: str):
        """
        Called once per channel page, before binding.
        Here we replace the placeholder with our BlockProcessingWidget.
        """
        # Replace detectionPerfPlaceholder with a real BlockProcessingWidget
        bp_widget = BlockProcessingWidget(parent=page_widget)
        page_widget.detectionBlockProcessingWidget = replace_widget(
            page_widget.detectionPerfPlaceholder, bp_widget,
            layout=page_widget.cellDetectionPerfVerticalLayout
        )

    def _on_channel_added(self, channel: str):
        """
        This is called at the end of add_channel_tab.
        At this point:
          - channel UI page exists
          - main CellMapParams knows the channel
        """
        page_widget = self.get_channel_ui(channel)
        if page_widget is None:
            return
        # Make sure the BlockProcessingWidget exists for this page
        if not hasattr(page_widget, 'detectionBlockProcessingWidget'):
            self._setup_channel(page_widget, channel)

        self.params.add_perf_channel(channel)

    def _bind_channel(self, page_widget, channel):
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute
        """
        page_widget.toolBox.currentChanged.connect(self.handle_tool_tab_changed)
        page_widget.runCellMapColocalizationCompatibleCheckBox.setVisible(False)
        buttons_functions = [
            ('detectionPreviewTuningOpenPushButton', self.plot_debug_cropping_interface),  # TODO: add load icon
            ('detectionPreviewTuningCropPushButton', self.create_tuning_sample),
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
            worker = self.get_worker(channel)
            self.wrap_step('Voxelization', worker.voxelize, abort_func=worker.stop_process, nested=False)
        else:
            self.main_window.popup('Could not run voxelization, missing filtered cells table. '
                                   'Please ensure that cell filtering has been run.', base_msg='Missing file')

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
        hist_idx = 0 if key == 'size' else 1
        ensure_inline_histogram(histogram, hist_idx, layout)
        return histogram

    def run_tuning_cell_detection(self, channel):
        """
        Run the cell detection on a subset of the sample which was previously selected
        """
        detector = self.get_worker(channel)
        self.wrap_step('Cell detection preview', detector.run_cell_detection,
                       step_kw_args={'tuning': True},
                       abort_func=detector.stop_process)
        if detector.stopped:
            return
        with self.debug_mode(channel, True):
            self.plot_detection_results(channel)

    def detect_cells(self, channel):  # TODO: merge w/ above w/ tuning option
        """Run the cell detection on the whole sample"""
        detector = self.get_worker(channel)
        coloc_compatible = self.params[channel].colocalization_compatible
        self.wrap_step('Detecting cells', detector.run_cell_detection,
                       step_kw_args={'tuning': False,
                                     'save_shape': self.params[channel].save_shape or coloc_compatible,
                                     'save_as_binary_mask': coloc_compatible},  # FIXME: seems to crash
                       abort_func=detector.stop_process)
        if detector.stopped:
            return
        if not self.sample_manager.get('cells', channel=channel, asset_sub_type='raw').exists:
            print('Cell detection aborted or failed. Exiting')
            return
        self.update_cell_number(channel)

    def post_process_cells(self, channel):  # WARNING: some plots in .post_process_cells() without UI params
        worker = self.get_worker(channel)
        self.wrap_step('Post processing cells', worker.post_process_cells, abort_func=worker.stop_process)
        self.update_cell_number(channel)

    def update_cell_number(self, channel):
        """
        Update the cell count number displayed based on the size of the raw and filtered cell detection files
        """
        worker = self.get_worker(channel)
        if worker is None:
            return
        params = self.params[channel]
        params.n_detected_cells = format_long_nb(worker.get_n_detected_cells())
        params.n_filtered_cells = format_long_nb(worker.get_n_filtered_cells())

    # def reset_detected(self):
    #     self.cell_detector.detected = False

    def plot_detection_results(self, channel):
        """Display the different steps of the cell detection in a grid to evaluate the filters"""
        dvs = self.wrap_plot(self.get_worker(channel).preview_cell_detection,
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
        self.wrap_plot(self.get_worker(channel).plot_filtered_cells, smarties=True)

    def plot_labeled_cells_scatter(self, channel, raw=False):
        """
        Plot the cells as colored symbols on top of either the raw stitched (not aligned) image
        or the resampled (aligned) image
        """
        self.wrap_plot(self.get_worker(channel).plot_cells_3d_scatter_w_atlas_colors, raw=raw)

    def __filter_cells(self, channel, is_last_step=True):
        if self.sample_manager.get('cells', channel=channel, asset_sub_type='raw').exists:
            detector = self.get_worker(channel)
            self.wrap_step('Filtering cells', detector.filter_cells, n_steps=2 + (1 - is_last_step),
                           abort_func=detector.stop_process, close_when_done=False)
            self.wrap_step('Voxelizing', detector.voxelize, step_args=['filtered'], save_cfg=False,
                           close_when_done=is_last_step)  # , main_thread=True)
        self.update_cell_number(channel)
        self.plot_cell_filter_results(channel)

    def preview_cell_filter(self, channel):  # TEST: circular calls
        with self.debug_mode(channel, True):
            self.__filter_cells(channel)

    def filter_cells(self, channel):
        self.__filter_cells(channel, is_last_step=False)
        detector = self.get_worker(channel)
        self.wrap_step('Aligning', detector.atlas_align, abort_func=detector.stop_process, save_cfg=False)
        detector.export_collapsed_stats()

    def run_cell_map(self):
        """Run the whole pipeline at once"""
        for channel in self.params.relevant_channels:
            self.run_channel(channel)

    def run_channel(self, channel):
        """Run the whole pipeline at once for a single channel"""
        self.update_cell_number(channel)
        params = self.params[channel]
        if params.detect_cells:
            self.detect_cells(channel)
        if params.filter_cells:
            self.post_process_cells(channel)
        if params.voxelize:
            self.voxelize(channel)
        if params.plot_when_finished:
            self.plot_cell_map_results(channel)

    def plot_cell_map_results(self, channel):
        """Plot the voxelization (density map) result"""
        self.wrap_plot(self.get_worker(channel).plot_voxelized_counts, arrange=False)


class TractMapTab(PostProcessingTab):
    """
    The tab responsible for the tract map processing and visualization.
    """
    pipeline_name = 'tract_map'
    channels_ui_name = 'tract_map_params'
    workers_are_global = False

    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'tract_map_tab', tab_idx)
        self.sample_manager = sample_manager

        self.advanced_controls_names = [# 'channel.tractMapAdvancedGroupBox'
            'channel.performanceGroupBox',
        ]

    def _bind(self):
        self._build_performance_ui()

    def _bind_params_signals(self):
        pass

    def _build_performance_ui(self):
        gb = self.ui.performanceGroupBox
        layout = gb.verticalLayout()

        # --- binarization: just n_processes ---
        # TODO: check self.binarizationPerf or self.ui.binarizationPerf
        self.binarizationPerf = NProcessesWidget(gb, label="Binarization n_processes")
        layout.addWidget(self.binarizationPerf)

        # --- where: just n_processes ---
        self.wherePerf = NProcessesWidget(gb, label="Where n_processes")
        layout.addWidget(self.wherePerf)

        # --- transform: full block_processing ---
        self.transformBlock = BlockProcessingWidget(gb, title="Transform block processing")
        layout.addWidget(self.transformBlock)

        # --- label: full block_processing ---
        self.labelBlock = BlockProcessingWidget(gb, title="Label block processing")
        layout.addWidget(self.labelBlock)

    def _set_params(self):
        self.params = TractMapParams(self.ui, self.sample_params, event_bus=self._bus,
                                     get_view=self.main_window.experiment_controller.get_config_view,
                                     apply_patch=self.main_window.experiment_controller.apply_ui_patch)

    def _get_channels(self):
        return self.params.relevant_channels

    def _bind_channel(self, page_widget, channel):
        buttons_functions = [
            ('tractMapComputeClippRangePushButton', self.compute_clipping_range),
            ('tractMapComputePixelsPercentRangePushButton', self.intensities_to_percentiles),
            ('tractMapPlotBinarizationThresholdsPushButton', self.plot_binarization_thresholds),
            ('tractMapPreviewTuningOpenPushButton', self.plot_debug_cropping_interface),
            ('tractMapPreviewTuningCropPushButton', self.create_tuning_sample),
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

    def run_tuning_tract_map(self, channel):
        # self.run_tract_map(channel, tuning=True)
        self.ui.channelsParamsTabWidget.get_channel_widget(channel).toolBox.setCurrentIndex(3)

    def run_tract_map(self, channel):
        tuning = self.ui.channelsParamsTabWidget.get_channel_widget(channel).tractMapStepsUseDebugCheckBox.isChecked()
        processor = self.get_worker(channel)

        with self.debug_mode(channel, debug_status=tuning):
            self.run_channel(channel, tuning=tuning)
            if processor.stopped:
                return

    def binarize_channel(self, channel):
        processor = self.get_worker(channel)
        self.wrap_step('Binarization', processor.binarize,
                       step_args=self.params[channel].clip_range,
                       abort_func=processor.stop_process)

    def extract_coordinates(self, channel, tuning):
        processor = self.get_worker(channel)
        self.wrap_step('Extracting coordinates', processor.mask_to_coordinates,
                       step_kw_args={'as_memmap': True},
                       abort_func=processor.stop_process)
        if tuning:
            processor.shift_coordinates()

    def transform_coordinates(self, channel):
        processor = self.get_worker(channel)
        self.wrap_step('Transforming coordinates', processor.parallel_transform,
                       abort_func=processor.stop_process)

    def label_coordinates(self, channel):
        processor = self.get_worker(channel)
        self.wrap_step('Labeling coordinates', processor.label,
                       abort_func=processor.stop_process)

    def export_df(self, channel):
        processor = self.get_worker(channel)
        self.wrap_step('Exporting coordinates', processor.export_df,
                       step_kw_args={'asset_sub_type': None},
                       abort_func=processor.stop_process)

    def run_channel(self, channel, tuning):
        params = self.params[channel]
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
            self.export_df(channel)

    def voxelize(self, channel):
        worker = self.get_worker(channel)
        if worker.get('binary', asset_sub_type='coordinates_transformed').exists:
            self.wrap_step('Voxelization', worker.voxelize, abort_func=worker.stop_process, nested=False)
        else:
            self.main_window.popup('Could not run voxelization, missing transformed coordinates. ',
                                   base_msg='Missing file')

    def plot_debug_cropping_interface(self, channel):
        """
        Plot the ortho-slicer to select a subset of the sample to perform tracts detections
        tests on
        """
        self.plot_slicer('detectionSubset', self.ui.channelsParamsTabWidget.get_channel_widget(channel),
                         self.params[channel], channel)

    def create_tuning_sample(self, channel):
        """Create an array from a subset of the sample to perform tests on """
        super().create_tuning_sample(channel)
        self.sample_manager.workspace.debug = False  # FIXME

    def compute_clipping_range(self, channel):
        processor = self.get_worker(channel)
        # TODO: use wrap_step but must include return
        pixel_percents = self.params[channel].clipping_percents
        low_intensity, high_intensity = processor.compute_clip_range(pixel_percents)
        self.params[channel].clip_range = [low_intensity, high_intensity]

    def intensities_to_percentiles(self, channel):
        """
        Convert the intensities to percentiles
        """
        processor = self.get_worker(channel)
        low_intensity, high_intensity = self.params[channel].clip_range
        low_percent, high_percent = processor.intensities_to_percentiles(low_intensity, high_intensity)
        self.params[channel].clipping_percents = [low_percent, high_percent]

    def plot_binary(self, channel):
        page = self.ui.channelsParamsTabWidget.currentWidget()
        debug = page.tractMapDebugCheckBox.isChecked()
        self.wrap_plot(self.get_worker(channel).plot_binary, debug=debug)

    def plot_binarization_thresholds(self, channel):
        page = self.ui.channelsParamsTabWidget.currentWidget()
        low_level_spin_box = page.binarizationThresholdsLowSpinBox_1
        high_level_spin_box = page.binarizationThresholdsHighSpinBox_2
        self.wrap_plot(self.get_worker(channel).plot_binarization_levels,
                       low_level_spin_box, high_level_spin_box)

    @GenericTab.ui_plot('Tract map voxelization')
    def plot_tract_map_results(self, channel):
        return self.get_worker(channel).plot_voxelized_counts()

    def plot_labeled_tracts_scatter(self, channel, raw=False):
        self.main_window.clear_plots()
        tract_mapper = self.get_worker(channel)
        page = self.ui.channelsParamsTabWidget.get_channel_widget(channel)
        coords_source_is_debug = page.tractMapDebugCheckBox.isChecked()
        coords_target_is_debug = page.tractMapTargetDebugCheckBox.isChecked()
        self.wrap_plot(tract_mapper.plot_tracts_3d_scatter_w_atlas_colors, raw=raw,
                       coordinates_from_debug=coords_source_is_debug,
                       plot_onto_debug=coords_target_is_debug)


class VasculatureTab(PostProcessingTab):
    """
    The tab responsible for the vasculature tracts detection, graph extraction and analysis
    """
    pipeline_name = "vasculature"
    channels_ui_name = 'vasculature_params'
    workers_are_global = True
    _workers_sub_steps = ('binary', 'graph')

    def __init__(self, main_window, tab_idx, sample_manager=None):
        super().__init__(main_window, 'vasculature_tab', tab_idx)

        self.sample_manager = sample_manager

        self.advanced_controls_names = [
            'channel.binarizationPerformanceGroupBox'
        ]

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        # WARNING: cannot go in __init__ because needs to come after setup of tab ui
        self.filters_list_widget = GraphFilterList(layout=self.ui.filterParamsVerticalLayout, parent=self.ui)
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

        self.ui.addFilterPushButton.clicked.connect(self.add_graph_filter)
        self.subscribe(UiVesselGraphFiltersChanged, self.update_file_suffix)

        self.ui.voxelizeGraphPushButton.clicked.connect(self.voxelize)
        self.ui.plotGraphVoxelizationPushButton.clicked.connect(self.plot_voxelization)
        self.ui.runAllVasculaturePushButton.clicked.connect(self.run_all)

        self.ui.saveStatsPushButton.clicked.connect(self.save_stats)

    def _set_params(self):
        self.params = VesselParams(self.ui, self.sample_params, event_bus=self._bus,
                                   get_view=self.main_window.experiment_controller.get_config_view,
                                   apply_patch=self.main_window.experiment_controller.apply_ui_patch)

    def _get_channels(self):
        return self.sample_manager.get_channels_by_pipeline('TubeMap', as_list=True)

    def _bind_channel(self, page_widget, channel):
        for btn_name, func in [('binarizePushButton', self.binarize_channel)]:
            self._bind_btn(btn_name, func, channel, page_widget)

    def _setup_channel(self, page_widget: QWidget, channel: str):
        """
        Per-channel setup (called after channel page UI exists, before binding).
        We build the performance widgets for binarization steps here, so perf params can bind to them.
        """
        # Try to locate a sensible container; prefer a dedicated groupbox if the .ui has one.
        gp_bx = getattr(page_widget, 'binarizationPerformanceGroupBox', None)
        if gp_bx is None:
            # TODO print error in app
            return

        # Idempotent setup
        if hasattr(page_widget, 'binarizationBlockProcessingWidget'):
            return

        v_layout = gp_bx.layout()

        def register_bp_widget(parent, title, layout):
            widget = BlockProcessingWidget(parent, title=title)
            layout.addWidget(widget)
            return widget

        # --- binarize: full block_processing ---
        page_widget.binarizationBlockProcessingWidget = register_bp_widget(
            gp_bx, title='Binarize block processing', layout=v_layout)

        # --- smooth: full block_processing ---
        page_widget.smoothingBlockProcessingWidget = register_bp_widget(
            gp_bx, title='Smoothing block processing', layout=v_layout)

        # --- binary_fill: ONLY n_processes ---
        page_widget.binaryFillingNProcessesSpinBox = NProcessesWidget(gp_bx, label='Binary filling n_processes')
        v_layout.addWidget(page_widget.binaryFillingNProcessesSpinBox)

        # --- deep_fill: full block_processing ---
        page_widget.deepFillingBlockProcessingWidget = register_bp_widget(
            gp_bx, title='Deep filling block processing', layout=v_layout)

        if hasattr(page_widget, 'placeholderWidget'):
            delete_widget(page_widget.placeholderWidget)

    def _on_channel_added(self, channel: str):
        """
        Hook invoked by add_channel_tab() once the page exists and has been setup/bound.
        We create the perf params now.
        """
        page_widget = self.get_channel_ui(channel)
        if page_widget is None:
            return
        if not hasattr(page_widget, 'binarizationBlockProcessingWidget'):
            self._setup_channel(page_widget, channel)
        self.params.add_perf_channel(channel)

    def add_graph_filter(self):
        filter_widget = self.filters_list_widget.add_filter_row()

        # FIXME: splash not shown
        splash, pbar = make_splash(message=f'Loading graph ', font_size=25)
        splash.show()
        # update_pbar(self.app, progress_bar, 20)
        self.main_window.processEvents()
        worker = self.get_worker(substep='graph')
        # update_pbar(self.app, progress_bar, 100)
        splash.finish(self.main_window)
        self.params.graph_params.add_graph_filter_params(filter_widget, worker.graph_annotated)

    def update_file_suffix(self, event: UiVesselGraphFiltersChanged):
        """
        Update the file suffix for the filtered graph
        """
        graph_params = self.params.graph_params
        if graph_params.n_filters == 0:
            self.ui.fileSuffixLineEdit.clear()
            return

        self.ui.fileSuffixLineEdit.setText(graph_params.compute_file_suffix())

    def unload_temporary_graphs(self):
        """Unload the temporary vasculature graph objects to free up RAM"""
        self.get_worker(substep='graph').unload_temporary_graphs()

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
        worker: "BinaryVesselProcessor" = self.get_worker(substep='binary')
        worker.assert_input_shapes_match()
        if not worker.inputs_match:
            shapes = worker.inputs_shapes
            if shapes == (None, None):
                self.main_window.print_error_msg(f'Cannot binarize because input channels not found at'
                                                 f'{[a.path for a in worker.assets_to_binarize()]}')
            else:
                self.main_window.print_error_msg(f'Cannot binarize because of shape mismatch between channels'
                                                 f'Got {shapes[0]}vx and {shapes[1]}vx')
            return
        try:
            kwargs = {'step_args': [channel], 'abort_func': worker.stop_process}
            self.wrap_step('Vessel binarization', worker.binarize_channel, **kwargs)
            self.wrap_step('Vessel binarization', worker.smooth_channel, **kwargs)
            self.wrap_step('Vessel binarization', worker.deep_fill_channel, **kwargs)
            # WARNING: The parallel cython loops inside cannot run from child thread
            self.wrap_step('Vessel binarization', worker.fill_channel, main_thread=True, **kwargs)
        except ClearMapVRamException as err:
            if stop_on_error:
                raise err

    def combine(self):
        """Combine the binarized (thresholded) version of the different channels."""
        worker = self.get_worker(substep='binary')
        self.wrap_step('Combining channels', worker.combine_binary, abort_func=worker.stop_process)

    def plot_binarization_results(self, plot_side_by_side=True):
        """
        Plot the thresholded images resulting from the binarization at the steps specified
        by the comboboxes in the UI.

        Parameters
        ----------
        plot_side_by_side: bool
            Whether to plot the images side by side (True) or overlay them (False).
        """
        steps, channels = self.params.get_selected_steps_and_channels()
        worker = self.get_worker(substep='binary')
        self.wrap_plot(worker.plot_results, steps, channels=channels,
                       side_by_side=plot_side_by_side, arrange=False, parent=self.main_window)

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
            worker = self.get_worker(substep='binary')
            self.binarize_channel(worker.all_vessels_channel, stop_on_error=True)
            self.binarize_channel(worker.arteries_channel, stop_on_error=True)
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
        worker = self.get_worker(substep='graph')
        self.wrap_step(tile, worker.skeletonize_and_build_graph, abort_func=worker.stop_process, main_thread=True)
        self.wrap_step(tile, worker.clean_graph, abort_func=worker.stop_process)
        self.wrap_step(tile, worker.reduce_graph, abort_func=worker.stop_process)
        try:
            self.wrap_step(tile, worker.register, abort_func=worker.stop_process)
        except MissingRequirementException:
            pass  # Already raise by wrap_step

    def plot_graph_type_processing_chunk_slicer(self):  # Refactor: rename
        """
        Plot the ortho-slicer to pick a sub part of the graph to display because
        depending on the display options, the whole graph may not fit in memory
        """
        self.plot_slicer('graphConstructionSlicer', self.ui, self.params.visualization_params,
                         channel=self.get_worker(substep='graph').parent_channels)
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
        self.wrap_plot(self.get_worker(substep='graph').visualize_graph_annotations,
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
            aligner = self.exp_controller.get_worker('registration')
            annotator = aligner.annotators['atlas']  # TODO: check but atlas should be OK to just do lookup
            color = annotator.find(structure_id, key='id')['rgb']
            self._plot_graph_structure(structure_id, color)
        else:
            print('No structure ID')
        self.main_window.structure_selector.close()

    def post_process_graph(self):
        """Post process the graph by filtering, tracing and removing capillaries """
        worker = self.get_worker(substep='graph')
        self.wrap_step('Post processing vasculature graph', worker.post_process,
                       abort_func=worker.stop_process)  # TODO: n_steps = 8

    def pick_region(self):
        """Open a dialog to select a brain region and plot it """
        picker = self.main_window.structure_selector
        picker.structure_selected.connect(self.params.visualization_params.set_structure_id)
        picker.onAccepted(self.plot_graph_structure)
        picker.onRejected(picker.close)
        picker.show()

    def _plot_graph_structure(self, structure_id, structure_color):
        dvs = self.wrap_plot(self.get_worker(substep='graph').plot_graph_structure,
                             structure_id, self.params.visualization_params.plot_type)
        if dvs:
            self.main_window.perf_monitor.stop()

    def voxelize(self):
        """Run the voxelization (density map) on the vasculature graph """
        voxelization_params = {
            'weight_by_radius': self.params.visualization_params.weight_by_radius,
        }
        worker = self.get_worker(substep='graph')
        if self.params.graph_params.filter_params:
            voxelization_params['filters'] = [
                GraphFilter(worker.graph_annotated, filter_type=g_filter.filter_type,
                            property_name=g_filter.property_name, property_value=g_filter.get_property_value())
                for g_filter in self.params.graph_params.filter_params]
            voxelization_params['operators'] = [g_filter.combine_operator_name
                                                for g_filter in self.params.graph_params.filter_params if
                                                g_filter.combine_operator_name is not None]  # skip first one
        self.wrap_step('Running voxelization', worker.voxelize,
                       step_kw_args=voxelization_params)#, main_thread=True)

    @GenericTab.ui_plot('Plotting vasculature graph voxelization')
    def plot_voxelization(self):
        """Plot the density map """
        return self.get_worker(substep='graph').plot_voxelization(self.main_window.centralWidget())

    def save_stats(self):
        """Save the stats of the graph to a feather file"""
        self.wrap_step('Saving stats', self.get_worker(substep='graph').write_vertex_table)


class ColocalizationTab(PostProcessingTab):
    pipeline_name = 'colocalization'
    channels_ui_name = 'colocalization_params'
    workers_are_global = False

    def __init__(self, main_window, tab_idx, sample_manager):
        super().__init__(main_window, 'colocalization_tab', tab_idx)
        self.sample_manager = sample_manager
        # FIXME: on tab click, assert that all channels detected

    def _set_params(self):
        self.params = ColocalizationParams(self.ui, self.sample_params, event_bus=self._bus,
                                           get_view=self.main_window.experiment_controller.get_config_view,
                                           apply_patch=self.main_window.experiment_controller.apply_ui_patch)

    def _get_channels(self):
        """
        Create combinations (e.g., [('Ch0','Ch1'), ('Ch0','Ch2'), ('Ch1','Ch2')])
        (but not permutations) of the channels to detect
        """
        return itertools.combinations(self.sample_manager.channels_to_detect, 2)

    def _bind(self):
        pass  # OK

    def _bind_params_signals(self):
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
        chan_args = {'channel_a': channel_a, 'channel_b': channel_b}
        buttons_functions = [
            ('colocalizationRunPushButton', functools.partial(self.run_colocalization_for_pair, **chan_args)),  # TODO: add load icon
            ('colocalizationPlotPushButton', functools.partial(self.plot, **chan_args)),
            ('colocalizationPlotSaveFilteredTablePushButton', functools.partial(self.save_filtered_table, **chan_args)),
            ('colocalizationVoxelizeFilteredTablePushButton', functools.partial(self.voxelize_filtered_table, **chan_args)),
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
        processor = self.get_worker((channel_a, channel_b))
        if processor:
            processor.compute_colocalization(*self.sort_channels(channel_a, channel_b))

    def plot(self, channel_a, channel_b):
        processor = self.get_worker((channel_a, channel_b))
        if processor:
            sorted_chan_a, sorted_chan_b = self.sort_channels(channel_a, channel_b)
            self.wrap_plot(processor.plot_nearest_neighbors, channel_a=sorted_chan_a, channel_b=sorted_chan_b)

    def save_filtered_table(self, channel_a, channel_b):
        processor = self.get_worker((channel_a, channel_b))
        if processor:
            processor.save_filtered_table(*self.sort_channels(channel_a, channel_b))

    def voxelize_filtered_table(self, channel_a, channel_b):
        processor = self.get_worker((channel_a, channel_b))
        if processor:
            processor.voxelize_filtered_table(*self.sort_channels(channel_a, channel_b))


###################################### GROUPS  #################################

class GroupAnalysisTab(BatchTab):
    def __init__(self, main_window, tab_idx, *, group_controller: AnalysisGroupController):
        super().__init__(main_window, tab_idx)
        self.group_controller = group_controller

        self.advanced_controls_names = [
            'computeSdAndEffectSizeCheckBox',
            'densitySuffixTextFilterLabel',
            'densitySuffixTextFilterLineEdit'
        ]

    def _set_params(self):
        self.params = GroupAnalysisParams(self.ui, preferences=self.main_window.preference_editor.params,
                                          event_bus=self._bus,
                                          get_view=self.group_controller.get_config_view,
                                          apply_patch=self.group_controller.apply_patch)
        self.params.set_pipelines(['CellMap', 'TractMap', 'TubeMap', 'Colocalization'])

        def _channels_provider(params):
            sample_folders_paths = params.get_all_paths()
            if sample_folders_paths:
                example_exp_dir = sample_folders_paths[0]  # random sorting sample 0
                sample_manager = self.group_controller.get_sample_manager(example_exp_dir)
                channels = sample_manager.get_channels_by_pipeline(
                    params.pipeline, as_list=True)
                return channels
            return []

        self.params.set_channels_provider(functools.partial(_channels_provider, self.params))
        self.params.set_on_plot_group(self.plot_density_maps)

        # self.subscribe(UiBatchResultsFolderChanged, self.group_controller.set_group_base_dir)
        # FIXME: UiBatchResultsFolderChanged should completely restart group mode
        self.subscribe(UiBatchGroupsChanged, self.handle_groups_changed)

    def handle_groups_changed(self, event: UiBatchGroupsChanged):
        self.group_controller.set_groups(self.params.groups)  # REFACTOR: use events data

    @property
    def processor(self):
        return self.group_controller.get_density_orchestrator()  # FIXME:

    def _setup_workers(self):
        if self.params.results_folder is not None:
            # self.group_controller.set_group_base_dir(self.params.results_folder)
            self.group_controller.set_groups(self.params.groups)
        self.group_controller.set_progress_watcher(self.main_window.progress_watcher)
        self.group_controller.set_thread_wrapper(self.main_window.wrap_in_thread)

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        super()._bind()
        self.ui.runPValsPushButton.clicked.connect(self.run_p_vals)
        self.ui.plotPValsPushButton.clicked.connect(self.plot_p_vals)
        self.ui.batchStatsPushButton.clicked.connect(self.make_group_stats_tables)

    def get_analysable_channels(self):
        """
        List the channels that have density maps available for analysis

        .. warning:: This method assumes that all the samples have the same channels

        Returns
        -------
        list of str
            The list of channels that have density maps available
        """
        density_orchestrator = self.processor
        return density_orchestrator.find_analysable_channels(density_suffix=self.params.density_suffix)

    @GenericTab.ui_plot('Plotting density maps')
    def plot_density_maps(self, group_name):
        return self.processor.plot_density_maps(
            self.params.groups[group_name], channel=self.params.plot_channel,
            density_suffix=self.params.density_suffix, parent=self.main_window.centralWidget())

    def run_p_vals(self):
        self.main_window.print_status_msg('Computing p_val maps')
        # TODO: set abort callback
        self.main_window.make_progress_dialog('P value maps', n_steps=len(self.params.selected_comparisons))
        try:
            self.processor.compute_p_values(self.params.selected_comparisons,
                                            channels=self.get_analysable_channels(),
                                            advanced=self.params.compute_sd_and_effect_size,
                                            density_files_suffix=self.params.density_suffix)
        except GroupStatsError as err:
            self.main_window.popup(str(err), base_msg='Cannot proceed with analysis')
        self.main_window.signal_process_finished()

    @ui_task_progress(lambda s: 'Group stats', lambda s: len(s.params.selected_comparisons))
    def make_group_stats_tables(self):
        self.main_window.clear_plots()
        tables_by_pair = self.processor.compute_stats_tables(self.params.selected_comparisons, save=True)
        dvs = [DataFrameWidget(tables[self.params.plot_channel]).table for tables in tables_by_pair.values()]
        self.main_window.setup_plots(dvs)  # TODO: use wrap_plot

    @GenericTab.ui_plot('Plotting p_val maps')
    def plot_p_vals(self, *_, **__):
        return self.processor.plot_p_value_maps(comparisons=self.params.selected_comparisons,
                                                channel=self.params.plot_channel, suffix=self.params.density_suffix,
                                                parent=self.main_window.centralWidget())

    def run_df_plots(self, plot_function, plot_kw_args):
        self.main_window.clear_plots()
        dvs = self.processor.run_plots(plot_function, self.params.selected_comparisons, plot_kw_args)
        self.main_window.setup_plots(dvs)
        return dvs

    def plot_volcanoes(self):  # TODO: check plot wraps
        self.run_df_plots(plot_volcano, {'group_names': None, 'p_cutoff': 0.05, 'show': False, 'save_path': ''})

    def plot_histograms(self, fold_threshold=2):  # TODO: check plot wraps
        folder = Path(self.params.results_folder) / self.params.groups[self.params.group_names[0]][0] # FIXME: check if absolute
        processors = init_sample_manager_and_processors(folder)
        registration_processor = processors['registration_processor']
        annotator = registration_processor.annotators[self.params.plot_channel]
        aba_df = annotator.df
        # aba_json_df_path = annotation.default_label_file  # FIXME: aba_json needs fold levels
        self.run_df_plots(plot_sample_stats_histogram,
                       {'aba_df': aba_df, 'sort_by_order': True, 'value_cutoff': 0,
                        'fold_threshold': fold_threshold, 'fold_regions': True, 'show': False})


class BatchProcessingTab(BatchTab):
    def __init__(self, main_window, tab_idx, *, group_controller=None):
        super().__init__(main_window, tab_idx)
        # FIXME: use GroupOrchestratorBase derived class
        self.group_controller = group_controller
        self.processor = BatchProcessor(self.main_window.progress_watcher)

    def _set_params(self):
        self.params = BatchProcessingParams(self.ui, preferences=self.main_window.preference_editor.params,
                                            event_bus=self._bus,
                                            get_view=self.group_controller.get_config_view,
                                            apply_patch=self.group_controller.apply_patch)

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
