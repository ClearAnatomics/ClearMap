"""
This module contains the interfaces to the different tabs and dialogs in the ClearMap GUI.
"""
from __future__ import annotations

import os
import functools
import warnings
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import final, List, Optional, Callable, Any, Dict, Tuple, TYPE_CHECKING

import numpy as np
from PyQt5.QtWidgets import QWhatsThis, QWidget

from ClearMap.Utils.event_bus import BusSubscriberMixin
from ClearMap.Utils.exceptions import MissingRequirementException, PlotGraphError
from ClearMap.Utils.utilities import title_to_snake
from ClearMap.config.config_handler import ConfigHandler

from .dialog_helpers import get_directory_dlg
from .gui_utils_base import create_clearmap_widget, replace_widget
from .widgets import ExtendableTabWidget, SamplePickerDialog
if TYPE_CHECKING:
    from PyQt5.QtWidgets import QMainWindow, QToolButton
    from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager
    from ClearMap.pipeline_orchestrators.generic_orchestrators import PipelineOrchestrator
    from .app import ClearMapApp
    from .params_interfaces import UiParameter, UiParameterCollection
    from .params_mixins import OrthoviewerSlicingMixin
    from .params import SampleParameters
    from ClearMap.pipeline_orchestrators.experiment_controller import ExperimentController, AnalysisGroupController



PathLike = str | Path


def channel_is_compound(channel) -> bool:
    return (isinstance(channel, str) and '-' in channel) or \
        (isinstance(channel, (tuple, list)))


class GenericUi:
    """
    The first layer of interface. This is not implemented directly but is the base class
    of GenericTab and GenericDialog, themselves interfaces
    """
    # FIXME: seems to be called several times
    def __init__(self, main_window: QMainWindow, ui_file_name: str, name: str, widget_class_name: str):
        """

        Parameters
        ----------
        main_window: ClearMapGui
        name: str
        ui_file_name: str
        widget_class_name: str
        """
        self.main_window: QMainWindow = main_window
        self.name: str = name
        self.ui_file_name: str = ui_file_name
        self.widget_class_name: str = widget_class_name
        self.ui: Optional[QWidget] = None
        self.params: Optional[UiParameter | UiParameterCollection] = None

    def _load_dot_ui(self):
        return create_clearmap_widget(self.ui_file_name, patch_parent_class=self.widget_class_name)

    def _init_ui(self):
        self.ui = self._load_dot_ui()

    # @abstractmethod
    def setup(self):
        """Setup more advanced features of the UI, notably the callbacks"""
        raise NotImplementedError()

    # @abstractmethod
    def set_params(self, *args):
        """Set the params object which links the UI and the configuration file"""
        raise NotImplementedError()

    def _load_config_to_gui(self):
        """Set every control on the UI to the value in the params object"""
        self.params.cfg_to_ui()


class GenericDialog(GenericUi):
    """
    Interface to any dialog associated with parameters
    """
    def __init__(self, main_window, name, file_name):
        super().__init__(main_window, file_name, name, 'QDialog')

    def _init_ui(self):
        super()._init_ui()
        self.ui.setWindowTitle(self.name.title())

    # @abstractmethod
    def set_params(self, *args):
        raise NotImplementedError()

# TODO: SubTab needs a parent widget, tab and main_window


@dataclass
class TabRequirements:
    needs_workspace: bool = False   # requires an experiment folder + workspace
    needs_channels: bool = False    # requires sample channels known/materialized


class GenericTab(GenericUi, BusSubscriberMixin):
    """
    The interface to all tab managers.
    A tab manager includes a tab widget, the associated parameters and
    an optional processor object which handles the computations.
    """
    requirements = TabRequirements(needs_workspace=False, needs_channels=False)  # Think SampleInfoTab

    processing_type: Optional[str] = None  # Not a pipeline tab by default
    channels_ui_name: str = ''  # The name of the ui file to create the channel tabs
    with_add_btn: bool = False  # Whether to add the add channel (+) button to the channels tab

    def __init__(self, main_window: ClearMapApp, ui_file_name: str, tab_idx: int, name: str = ''):
        """

        Parameters
        ----------
        main_window: ClearMapApp
        ui_file_name: str
        tab_idx: int
        name: str
        """
        name = name or self.get_tab_name()
        super().__init__(main_window, ui_file_name, name, 'QTabWidget')
        # REFACTORING: avoid direct access to a private attribute
        BusSubscriberMixin.__init__(self, bus=main_window.experiment_controller._bus)
        self.tab_idx = tab_idx

        self.params = None
        self.sample_manager = None

        self.inited = False
        self.setup_complete = False
        self.params_set = False

        self.advanced_controls_names = []

        self.minimum_width = 200  # REFACTOR:

        self._selected_once = False  # Whether the tab has been selected once

    @classmethod
    def requirements_fulfilled(cls, sample_manager) -> bool:
        """Whether this tab may be shown for the current sample_manager state."""
        reqs = cls.requirements
        workspace_ready = sample_manager is not None and sample_manager.workspace is not None
        ch_ready = workspace_ready and sample_manager.channels  #  or sample_manager.stitchable_channels
        return (workspace_ready or not reqs.needs_workspace) and (ch_ready or not reqs.needs_channels)

    def on_selected(self):
        """Called when this tab is selected; override in subclasses if needed."""
        pass

    @final
    def notify_selected(self):
        """
        Notify that the tab has been selected (i.e. clicked on).
        It will also store that the tab has been selected at least once
        """
        try:
            self.on_selected()
            self._selected_once = True
        except Exception as e:
            # optional logging
            print(f"[{self.name}] on_selected error: {e}")

    @classmethod
    def get_tab_name(cls):
        snake = title_to_snake(cls.__name__.replace('Tab', ''))
        words = snake.split('_')
        tab_name = words[0].title() + ' ' + ' '.join(words[1:])
        return tab_name.strip()

    def _init_ui(self):
        """
        Create and arrange the UI elements.
        Does minimum binding of signals.

        .. note::
            It is normally called by the setup method but can be called by client code
            explicitly if needed. However, it is protected to avoid calling it multiple times.
        """
        if self.inited:
            return
        super()._init_ui()
        if hasattr(self.ui, 'setMinimumWidth'):
            self.ui.setMinimumWidth(self.minimum_width)
        tab_widget = getattr(self.main_window.tabWidget, 'tabWidget', None)
        if tab_widget and all([hasattr(tab_widget, attr) for attr in ('tabText', 'removeTab', 'insertTab')]):
            if tab_widget.tabText(self.tab_idx) == self.name:
                tab_widget.removeTab(self.tab_idx)  # remove if same tab
            tab_widget.insertTab(self.tab_idx, self.ui, self.name.title())
        if hasattr(self.ui, 'advancedCheckBox'):
            self.set_advanced_controls_visibility(False)
            self.ui.advancedCheckBox.stateChanged.connect(self.handle_advanced_checked)
        self.inited = True

    # @final
    def setup(self):
        """Setup more advanced features of the UI, notably the callbacks"""
        self._init_ui()  #  Call protected by "self.inited". Called in case the tab is not yet initialised
        if self.setup_complete:
            return

        self._swap_channels_tab_widget(self.with_add_btn)
        self._bind()
        self._connect_children_whats_this()

        self.setup_complete = True

    # @abstractmethod
    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        raise NotImplementedError(f"Method _bind not implemented in {self.__class__.__name__}")

    # @abstractmethod
    def _set_params(self):
        """Set the params object which links the UI and the configuration file"""
        pass

    # @abstractmethod
    def _bind_params_signals(self):
        """Bind the signals of the params object"""
        pass

    # @abstractmethod
    def _get_channels(self):
        return []  # Default to no channels when implementing a tab without channels

    def _setup_workers(self):
        """Set up the optional workers (which handle the computations) associated with this tab"""
        pass

    def finalise_workers_setup(self):
        """Finalise the setup of the workers. Typically called when the tab is selected"""
        pass

    # @final   # FIXME: never set
    def set_params(self, sample_params: Optional[SampleParameters] = None):  # REFACTOR: rename to initialise or similar
        """Set the params object which links the UI and the configuration file"""
        self._set_params()
        self.params_set = True
        if self.params._get_view is not None:
            self.finalise_set_params()

    @final
    def finalise_set_params(self):
        if not self.params_set:
            warnings.warn(f'Params not set for {self.__class__}. Call set_params before finalise_set_params')
            return
        self._setup_workers()
        self._create_channels()  # Creates missing channels. FIXME: What about existing but renamed ones?
        self._load_config_to_gui()
        self._bind_params_signals()
        self.handle_advanced_checked()  # in case channel.QtControl in advanced_controls_names

    def _create_channels(self):
        if not hasattr(self.ui, 'channelsParamsTabWidget'):
            return
        if not isinstance(self.ui.channelsParamsTabWidget, ExtendableTabWidget):
            warnings.warn(f'Channel tab widget not finalised for  {self.name}, skipping channel creation')
            return
        for channel in self._get_channels():
            if channel not in self.ui.channelsParamsTabWidget.get_channels_names():
                self.add_channel_tab(channel)

    def get_channel_ui(self, channel: str):
        """ Get the UI widget for a specific channel """
        return self.ui.channelsParamsTabWidget.get_channel_widget(channel)

    def _swap_channels_tab_widget(self, with_add_btn: bool = False):
        """
        Substitute the placeholder channel tab widget by the dynamic one

        Parameters
        ----------
        with_add_btn: bool
            Whether to add the add channel button
        """
        if (not hasattr(self.ui, 'channelsParamsTabWidget') or
                not hasattr(self.ui, 'channelsParamsTabWidgetLayout')):
            print(f'No channel tab widget found for {self.name}, skipping swap')
            return
        if not isinstance(self.ui.channelsParamsTabWidget, ExtendableTabWidget):
            layout = self.ui.channelsParamsTabWidgetLayout
            self.ui.channelsParamsTabWidget = replace_widget(self.ui.channelsParamsTabWidget,
                                                             ExtendableTabWidget(self.ui, with_add_tab=with_add_btn),
                                                             layout)

    # @final
    def add_channel_tab(self, channel: str = ''):
        """
        Add a tab for a specific channel.
        This should then call the connect_channel method to set up the tab bindings.

        .. hint::
            This method is meant to be used for tab with channels. To use it
            ensure that your tab has a channelsParamsTabWidget attribute and set
            channels_ui_name to the name of the UI file for the channel tabs.
        """
        if not self.channels_ui_name:
            warnings.warn(f'No channel UI name set for {self.name}. '
                          f'This method is meant to be used for tab with channels. To use it, '
                          f'ensure that your tab has a channelsParamsTabWidget attribute and set '
                          f'channels_ui_name to the name of the UI file for the channel tabs.')
            return
        channel, page_widget, created = self._get_or_init_channel_ui(channel)
        # FIXME: bug in StitchingTab when add_channel calls reconcile_children_from_view ->
        #   this triggers the creation of new channel params that don't have UI yet
        if channel not in self.params.keys():
            self._create_channel_params(channel)
            self._setup_channel(page_widget, channel)
            self._bind_channel(page_widget, channel)
            self._connect_children_whats_this(page_widget)
            self._on_channel_added(channel)
        else:
            if created:
                self.params[channel].teardown()
                self._setup_channel(page_widget, channel)
                self._bind_channel(page_widget, channel)
                self._connect_children_whats_this(page_widget)
                self._on_channel_added(channel)

    def _create_channel_params(self, channel):
        if isinstance(self, PipelineTab) and not channel_is_compound(channel):
            # WARNING: ConfigObj Section does not support get() method
            d_type = self.sample_manager.data_type(channel)
            self.params.add_channel(channel, d_type)
        else:
            self.params.add_channel(channel)

    def _on_channel_added(self, channel: str):
        """Child class should implement this to trigger
        any action needed when a channel is added"""
        pass

    def remove_channel_tab(self, channel: str):
        """Wrapper so removal always triggers the hook."""
        try:
            if hasattr(self, 'params') and hasattr(self.params, 'pop'):
                self.params.pop(channel)
            else:  # Fallback only to ensure we do through the proper params teardown
                if hasattr(self.ui, 'channelsParamsTabWidget'):
                    self.ui.channelsParamsTabWidget.remove_channel_widget(channel)
        finally:
            self._on_channel_removed(channel)

    def _on_channel_removed(self, channel_name: str) -> None:
        """Child classes should implement this to trigger cleanup actions after a channel is removed"""
        pass

    def _get_or_init_channel_ui(self, channel: str = '') -> Tuple[str, QWidget]:
        """
        Initialise the UI for a specific channel.
        This only creates the UI widget and adds it to the tab widget.
        Further setup should be done in the `__setup_channel` and __bind_channel methods

        Parameters
        ----------
        channel: str
            The name of the channel

        Returns
        -------
        tuple
            The channel name and the page widget
        """
        tab_widget = self.ui.channelsParamsTabWidget
        page = tab_widget.get_channel_widget(channel)
        if page:
            return channel, page, False
        else:
            page_widget = create_clearmap_widget(self.channels_ui_name, patch_parent_class='QWidget')
            channel = tab_widget.add_channel_widget(page_widget, name=channel)
            return channel, page_widget, True

    def _bind_channel(self, page_widget: QWidget, channel: str):
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute

        .. important::
            All button bindings for channels should be done here
        """
        pass

    def _setup_channel(self, page_widget: QWidget, channel: str):
        """
        Perform additional setup for the channel (before binding)
        For example set default values or populate lists

        .. note::
            Implement in subclass if needed
        """
        pass

    def handle_advanced_checked(self):
        """Activate the *advanced* mode which will display more controls"""
        if not hasattr(self.ui, 'advancedCheckBox'):  # e.g. BatchProcessingTab ATM
            return
        self.set_advanced_controls_visibility(self.ui.advancedCheckBox.isChecked())

    def set_advanced_controls_visibility(self, visible: bool):
        """
        Set the visibility of the advanced controls

        Parameters
        ----------
        visible : bool
            Whether to show the advanced controls
        """
        if not self.params:  # Tab not yet initialised
            return

        for ctrl_name in self.advanced_controls_names:
            if ctrl_name.startswith('channel.'):
                _, ctrl_name = ctrl_name.split('.')
                if not hasattr(self.ui, 'channelsParamsTabWidget'):  # Channels not set yet
                    warnings.warn(f'Could not find channel tab widget for {self.name}')
                    return
                for channel in self._get_channels():
                    ctrl = getattr(self.ui.channelsParamsTabWidget.get_channel_widget(channel), ctrl_name)
                    ctrl.setVisible(visible)
            else:
                ctrl = getattr(self.ui, ctrl_name, None)
                if ctrl:
                    ctrl.setVisible(visible)
                else:
                    warnings.warn(f'Could not find control {ctrl_name} in {self.name}')

    def disable(self):
        """Disable this tab (UI element)"""
        self.ui.setEnabled(False)

    def step_exists(self, step_name: str, file_list: List[PathLike] | PathLike) -> bool:
        """
        Check that prerequisite step step_name has been run and produced
        the outputs in file_list

        Parameters
        ----------
        step_name: str
            The name of the step
        file_list: list[str] | list[Path] | str | Path
            The list of files that should have been produced by the step

        Returns
        -------
        bool
            Whether the step has been run
        """
        if isinstance(file_list, (str, Path)):
            file_list = [file_list]
        for f_path in file_list:
            if not os.path.exists(f_path):
                self.main_window.print_error_msg(f'Missing {step_name} file {f_path}. '
                                                 f'Please ensure {step_name} is run first.')
                return False
        return True

    def connect_whats_this_btn(self, info_btn: QToolButton, whats_this_ctrl: QWidget):
        """
        Utility function to bind the info button to the display of
        the detailed *whatsThis* message associated with the control

        Parameters
        ----------
        info_btn: QToolButton
            The button to display the info
        whats_this_ctrl: QWidget
            The control to display the info for (it should have a *whatsThis* message)
        """
        def show_whats_this(widget):
            QWhatsThis.showText(widget.pos(), widget.whatsThis(), widget)
        info_btn.clicked.connect(lambda: show_whats_this(whats_this_ctrl))

    def _connect_children_whats_this(self, parent: Optional[QWidget] = None):  # TODO: make tests safe
        """
        Connect all the what's this buttons of the widgets `parent` to the corresponding labels

        Parameters
        ----------
        parent : QWidget | None
            The widget to connect
        """
        parent = parent or self.ui
        children = {child.objectName(): child for child in parent.findChildren(QWidget)}
        for ctrl_name, ctrl in children.items():
            if ctrl_name.endswith('InfoToolButton'):
                base_name = ctrl_name.replace('InfoToolButton', '')
                widgets = [widget for name, widget in children.items() if name.startswith(base_name) and name != ctrl_name]
                widgets = [w for w in widgets if w.whatsThis()]
                if len(widgets) == 1:
                    self.connect_whats_this_btn(ctrl, widgets[0])
                else:
                    raise ValueError(f'Could not find unique widget for "{base_name}" in "{parent.objectName()}",'
                                     f' got {[(w.objectName(), w) for w in widgets]}')

    def wrap_step(self, task_name: str, func: Callable, step_args: Optional[List[Any]]=None,
                  step_kw_args: Optional[Dict[str, Any]] = None, n_steps: int = 1, abort_func: Optional[Callable] = None,
                  save_cfg: bool =True, nested:bool = True, close_when_done: bool =True, main_thread: bool = False):
        """
        This function aims to start a new thread for the function being wrapped to ensure that
        the UI remains responsive (unless main_thread is set to True).
        It will also start a progress dialog

        Parameters
        ----------
        task_name : str
            The name of the task to be displayed
        func : function
            The function to run
        step_args : list
            The positional arguments to func
        step_kw_args : dict
            The keyword arguments to func
        n_steps : int
            The number of top level steps in the computation. This will be disabled if nested is False.
        abort_func : function
            The function to trigger to abort the execution of the computation (bound to the abort button)
        save_cfg : bool
            Whether to save the configuration to disk before running the computation.
            This is usually the right choice to ensure that the config reloaded by func is up to date.
        nested : bool
            Whether the computation has 2 levels of progress
        close_when_done : bool
            Close the progress dialog when func has finished executing
        main_thread : bool
            Whether to run in the main thread. Default is False to ensure that the UI thread remains
            responsive, a new thread will be spawned.
        """
        step_args = step_args or []
        step_kw_args = step_kw_args or {}

        if task_name:
            n_steps = n_steps if nested else 0
            self.main_window.make_progress_dialog(task_name, n_steps=n_steps, abort=abort_func)

        try:
            if main_thread:
                func(*step_args, **step_kw_args)
            else:
                self.main_window.wrap_in_thread(func, *step_args, **step_kw_args)
        except MissingRequirementException as err:
            self.main_window.print_error_msg(err)
            self.main_window.popup(str(err), base_msg=f'Could not run operation {func.__name__}', print_warning=False)
            raise err
        finally:
            if self.sample_manager is not None and self.sample_manager.workspace is not None:  # WARNING: hacky
                self.sample_manager.workspace.executor = None  # FIXME: do not pass workspace but semaphore instead
            if close_when_done:
                self.main_window.progress_watcher.finish()
            else:
                # FIXME: message different if exception
                msg = f'{self.main_window.progress_watcher.main_step_name} finished'
                self.main_window.print_status_msg(msg)
                self.main_window.log_progress(f'    : {msg}')

    def wrap_plot(self, plot_method: Callable, *args, **kwargs):
        """
        Wrapper to plot a graph and display it in the main window.
        It also handles MissingRequirementException and PlotGraphError

        Parameters
        ----------
        plot_method: function
            The function (or method) to plot the graph
        args: list
            The positional arguments to plot_function
        kwargs: dict
            The keyword arguments to plot_function

        Returns
        -------
        list[DataViewer]
            The data viewers returned by plot_function
        """
        self.main_window.clear_plots()
        try:
            dvs = plot_method(self, *args, **kwargs)  # We need to pass `self` because method
        except MissingRequirementException as err:
            self.main_window.print_error_msg(f'Missing {plot_method.__name__} files {str(err)}. '
                                             f'Please ensure previous steps are run first.')
            return []
        except PlotGraphError as err:
            self.main_window.popup(str(err), base_msg='PlotGraphError')
            return []
        if not dvs:
            return []
        if isinstance(dvs[0], list):
            dvs, titles = dvs
            self.main_window.setup_plots(dvs, titles)
        else:
            self.main_window.setup_plots(dvs)
        from ClearMap.Visualization.Qt.DataViewer import DataViewer
        return [widget for widget in dvs if isinstance(widget, DataViewer)]

    @staticmethod
    def ui_plot(status_msg: str = ''):
        """
        Decorator for tab methods that produce plot widgets.

        Handles:
        - status message display
        - clearing previous plots
        - MissingRequirementException / PlotGraphError
        - (dvs, titles) tuple return pattern
        - setup_plots() call
        - filtering return to DataViewer instances only

        Usage:
            @GenericTab.ui_plot("Plotting p-values…")
            def plot_p_vals(self, ...):
                # pure plotting logic
                return [widget1, widget2]
        """

        def deco(fn):
            @functools.wraps(fn)
            def wrapper(self, *args, **kwargs):
                if status_msg:
                    self.main_window.print_status_msg(status_msg)
                return self.wrap_plot(fn, *args, **kwargs)
            return wrapper
        return deco


class ExperimentTab(GenericTab):
    def __init__(self, main_window: QMainWindow, ui_file_name: str, tab_idx: int, name: str = ''):
        super().__init__(main_window, ui_file_name, tab_idx, name)
        self.sample_manager: Optional[SampleManager] = None
        self.sample_params: Optional[SampleParameters] = None
        self.exp_controller: Optional[ExperimentController] = None

    def set_controller(self, controller: ExperimentController):
        self.exp_controller = controller

    def setup_sample_manager(self, sample_manager: SampleManager):
        self.sample_manager = sample_manager

    def set_params(self, sample_params: Optional[SampleParameters] = None):  # FIXME: never set
        if sample_params:
            self.sample_params = sample_params  # REFACTORING: consider using sample_manager
        super().set_params()

    def reconcile_channel_pages(self, desired_channels: List[str]):
        """
        Update the configuration for the channels and the associated page widgets

        Parameters
        ----------
        desired_channels: list[str]
            The list of channels that should be present
        """
        if not self.params or not hasattr(self.ui, 'channelsParamsTabWidget'):  # If not setup or tab without channels
            return

        tab_widget = self.ui.channelsParamsTabWidget
        # ordered list
        desired_channels = list(dict.fromkeys(desired_channels))  # dedupe, keep order

        # remove obsolete pages
        obsolete_channels = [c for c in tab_widget.get_channels_names() if c not in desired_channels]
        for ch in obsolete_channels:
            self.params.pop(ch)  # self.remove_channel_tab(ch) should be implicit in pop
            self._on_channel_removed(ch)

        # add missing pages (in desired order)
        for ch in desired_channels:
            if ch not in tab_widget.get_channels_names():
                self.add_channel_tab(ch)  # will call _setup_channel/_bind_channel
                # setup workers ?
                # params.add_channel(ch) ??

        self._after_channels_reconciled(desired_channels)
        self._setup_workers()

    def _after_channels_reconciled(self, desired_channels: list[str]) -> None:
        """
        Triggered at the end of channel reconciliation between UI and cfg
        Child classes should implement this method to e.g. populate lists or ComboBoxes
        that depend on the list of channels
        """
        pass

    def _setup_workers(self):
        """
        Setup the optional workers (which handle the computations) associated with this tab

        .. warning::
            This method must be implemented in the subclasses of PipelineTab

        .. note::
            is required for
                create_channels  (to get list of channels)
                bind_params_signals
        """
        pass


class GroupTab(GenericTab):
    processing_type = 'group'
    def __init__(self, main_window: QMainWindow, ui_file_name: str, tab_idx: int, name: str = ''):
        super().__init__(main_window, ui_file_name, tab_idx, name)
        self.config_handler: Optional[ConfigHandler] = None
        self.group_controller: Optional[AnalysisGroupController] = None

    def set_controller(self, controller: AnalysisGroupController):
        self.group_controller = controller



class PipelineTab(ExperimentTab):
    requirements = TabRequirements(needs_workspace=True, needs_channels=True)

    processing_type = ''
    pipeline_name: str = ''  # Must be set in subclass
    workers_are_global: bool = False  # Whether the workers are shared between channels
    _workers_sub_steps: Optional[Tuple[str]] = None  # The sub-steps to create workers for (if any)

    def __init__(self, main_window: QMainWindow, ui_file_name: str, tab_idx: int, name: str = ''):
        super().__init__(main_window, ui_file_name, tab_idx, name)
        self.sample_params = None
        self.sample_manager = None  # REFACTORING: check if redundant

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.processing_type:
            raise NotImplementedError(
                f"Class '{cls.__name__}' must override 'processing_type' with a string value."
            )

    def _iter_substeps(self) -> tuple[str | None, ...]:
        return self._workers_sub_steps or (None,)

    def _workers_channel_keys(self) -> list:
        """
        Worker keys handed to ExperimentController.reconcile_workers:
          - global workers: [None]
          - per-channel/per-pair: list from _get_channels()
        """
        return [None] if self.workers_are_global else list(self._get_channels())

    def setup_sample_manager(self, sample_manager: SampleManager):
        """
        Associate the sample_manager to the current tab

        Parameters
        ----------
        sample_manager : SampleManager
            The object that handles the sample data
        """
        self.sample_manager = sample_manager

    def _bind_btn(self, btn_name: str, func: Callable, channel: Optional[str] = None,
                  page_widget: Optional[QWidget] = None, **kwargs):
        if channel:
            getattr(page_widget, btn_name).clicked.connect(functools.partial(func, channel, **kwargs))
        else:
            getattr(self.ui, btn_name).clicked.connect(func)

    # @abstractmethod


class PreProcessingTab(PipelineTab):
    processing_type = 'pre'
    workers_are_global = True

    def __init__(self, main_window: QMainWindow, ui_file_name: str, tab_idx: int, name: str = ''):
        super().__init__(main_window, ui_file_name, tab_idx, name)

    @property
    def worker(self):
        return self.exp_controller.get_worker(self.pipeline_name)

    # @abstractmethod
    def _setup_workers(self):
        if self.sample_manager.setup_complete:
            worker = self.worker
            if worker is not None:  # e.g. not SampleInfoTab
                self.wrap_step('Setting up worker', worker.setup_if_needed, n_steps=1,
                               save_cfg=False, nested=False)


class PostProcessingTab(PipelineTab):
    """
    Interface to all the tab managers in charge of post-processing the data (e.e. typically detecting relevant info in the data).
    One particularity of a post-processing tab manager is that it includes the corresponding pre-processor.
    A tab manager includes a tab widget, the associated parameters
    and potentially a processor object which handles the computations.
    """
    processing_type = 'post'

    def __init__(self, main_window: QMainWindow, ui_file_name: str, tab_idx: int, name: str = ''):
        super().__init__(main_window, ui_file_name, tab_idx, name)

    def _setup_workers(self):
        if not self.sample_manager.setup_complete:
            self.main_window.print_warning_msg("SampleManager not initialised")
            return
        desired_channels = self._workers_channel_keys()
        for substep in self._iter_substeps():
            workers = self.exp_controller.reconcile_workers(
                self.pipeline_name,
                desired_channels=desired_channels,
                substep=substep,
                keep_global=self.workers_are_global
            )
            for w in workers.values():
                w.setup_if_needed()

    def get_worker(self, channel: Optional[str | Tuple[str, str]] = None,
                   substep: Optional[str] = None) -> PipelineOrchestrator:
        if substep and substep not in self._iter_substeps():
            raise ValueError(f'Sub-step {substep} not in {self._workers_sub_steps}')
        elif not substep and self._workers_sub_steps:
            raise ValueError(f'Sub-step must be specified, available: {self._workers_sub_steps}')
        if self.workers_are_global:
            channel = None
        return self.exp_controller.get_worker(self.pipeline_name, channel=channel, substep=substep)

    @contextmanager
    def debug_mode(self, channel: str, debug_status: str | bool):
        worker = self.get_worker(channel)
        status_backup = worker.workspace.debug
        try:
            worker.workspace.debug = debug_status
            yield
        finally:
            worker.workspace.debug = status_backup

    def create_tuning_sample(self, channel):
        """Create an array from a subset of the sample to perform tests on """
        worker = self.get_worker(channel)
        if not hasattr(worker, 'create_test_dataset'):
            return
        self.wrap_step('Creating tuning sample', worker.create_test_dataset,
                       step_kw_args={'slicing': self.params[channel].slicing}, nested=False)

    def plot_slicer(self, slicer_prefix: str, tab: QWidget, params: OrthoviewerSlicingMixin, channel: str):
        """
        Display the ortho-slicer to pick a subset of 3D data.
        This is typically used to create a small  dataset to evaluate parameters
        for long-running operations before analysing the whole sample

        Parameters
        ----------
        slicer_prefix
        tab
        params
        """
        self.main_window.clear_plots()
        no_scale = False
        if isinstance(channel, (list, tuple)):
            sources = [self.sample_manager.get('resampled', channel=ch).as_source() for ch in channel]
            if not np.all([src.shape == sources[0].shape for src in sources]):
                raise ValueError('Channels have different shapes')
            plot_image = np.mean([self.sample_manager.get('resampled', channel=ch).as_source() for ch in channel], axis=0)
            channel = channel[0]
        else:
            # asset = self.sample_manager.get('stitched', channel=channel)
            # if asset.exists:
            #     no_scale = True
            # else:  # missing stitched
            # FIXME: we cannot currently use stitched because we need to transpose it and it is too heavy for that
            asset = self.sample_manager.get('resampled', channel=channel)
            plot_image = asset.as_source()
        self.main_window.ortho_viewer.setup(plot_image, params, parent=self.main_window, no_scale=no_scale)
        dvs = self.main_window.ortho_viewer.plot_orthogonal_views()
        if not no_scale:
            ranges = [[params.reverse_scale_axis(v, ax) for v in vals] for ax, vals in zip('xyz', params.slice_tuples)]
            self.main_window.ortho_viewer.update_ranges(ranges)
        self.main_window.setup_plots(dvs, ['x', 'y', 'z'])

        # WARNING: needs to be done after setup
        for axis, ax_max in zip('XYZ', self.sample_manager.stitched_shape(channel)):  # WARNING: assumes stitched shape
            getattr(tab, f'{slicer_prefix}{axis}RangeMax').setMaximum(round(ax_max))


class BatchTab(GroupTab):
    def __init__(self,  main_window: QMainWindow, tab_idx: int):
        super().__init__(main_window, title_to_snake(self.__class__.__name__), tab_idx)
        self.processing_type = 'batch'
        self.config_loader = None

    @property
    def initialised(self):
        return self.params is not None

    # @abstractmethod
    def _setup_workers(self):
        pass

    def _bind(self):
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute

        .. warning::
            The child classes should call this method in their own bind method

        """
        self.ui.resultsFolderPushButton.clicked.connect(self.setup_results_folder)
        self.ui.folderPickerHelperPushButton.clicked.connect(self.create_wizard)
        self.ui.batchToolBox.setCurrentIndex(0)

    def setup_results_folder(self):
        results_folder = get_directory_dlg(self.main_window.preference_editor.params.start_folder,
                                           'Select the folder where results will be written')
        if not results_folder:
            return
        else:
            results_folder = Path(results_folder)

        self.main_window.logger.set_file(results_folder / 'info.log')  # WARNING: set logs to global results folder
        self.main_window.error_logger.set_file(results_folder / 'errors.html')
        self.main_window.progress_watcher.log_path = self.main_window.logger.file.name

        self.set_params()
        self.params.results_folder = str(results_folder)
        self._load_config_to_gui()
        self._setup_workers()

    def create_wizard(self):
        return SamplePickerDialog(self.params.results_folder, self.params)  # FIXME: check if results_folder or make both equal with self.params.src_folder
