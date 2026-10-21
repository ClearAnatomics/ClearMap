"""
This module contains the interfaces to the different tabs and dialogs in the ClearMap GUI.
"""
from __future__ import annotations

import os
import functools
import warnings
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager, ExitStack
from typing import final, Callable, Any, TYPE_CHECKING, TypeVar, Generic, ParamSpec, Tuple, Union

import numpy as np
from PyQt5.QtWidgets import QWhatsThis, QWidget, QApplication

from ClearMap.Utils.event_bus import BusSubscriberMixin, GUI_THREAD
from ClearMap.Utils.utilities import title_to_snake
from ClearMap.config.channel import Channel
from ClearMap.config.compound_keys import is_compound_channel
from ClearMap.config.config_handler import ConfigHandler, ALTERNATIVES_REG

from .dialog_helpers import get_directory_dlg, make_simple_progress_dialog
from .exception_handler import handle_exception
from .gui_utils_base import create_clearmap_widget, replace_widget
from .widgets import ExtendableTabWidget, SamplePickerDialog, channel_page_id_of, assign_channel_page_id_if_known

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QMainWindow, QToolButton
    from ClearMap.pipeline_orchestrators.sample_info_management import SampleManager
    from ClearMap.pipeline_orchestrators.generic_orchestrators import PipelineOrchestrator
    from .app import ClearMapApp
    from .params_interfaces import UiParameter, UiParameterCollection
    from .params_mixins import OrthoviewerSlicingMixin
    from .params import SampleParameters
    from ClearMap.pipeline_orchestrators.experiment_controller import ExperimentController, AnalysisGroupController
    from ClearMap.Visualization.Qt.DataViewer import DataViewer


PathLike = str | Path

# Binds the worker type to a concrete PipelineOrchestrator subclass in each concrete tab.
# Free (unbound) in abstract intermediates (PipelineTab, Pre/PostProcessingTab),
# bound at the concrete leaf (e.g. CellCounterTab[CellDetector]).
TWorker = TypeVar('TWorker', bound='PipelineOrchestrator')
P = ParamSpec('P')

# What a channel page is a page of: a channel, or the channels of a compound (e.g. a colocalization pair)
PageKey = Union[Channel, Tuple[Channel, ...]]


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
        self.ui: QWidget | None = None
        self.params: UiParameter | UiParameterCollection | None = None

    def _load_dot_ui(self) -> QWidget:
        return create_clearmap_widget(self.ui_file_name, patch_parent_class=self.widget_class_name)

    def _init_ui(self) -> None:
        self.ui = self._load_dot_ui()

    def setup(self) -> None:
        """Setup more advanced features of the UI, notably the callbacks"""
        raise NotImplementedError()

    def set_params(self, *args) -> None:
        """Set the params object which links the UI and the configuration file"""
        raise NotImplementedError()

    def _load_config_to_gui(self) -> None:
        """Set every control on the UI to the value in the params object"""
        self.params.cfg_to_ui()


class GenericDialog(GenericUi):
    """
    Interface to any dialog associated with parameters
    """
    def __init__(self, main_window, name: str, file_name: str):
        super().__init__(main_window, file_name, name, 'QDialog')

    def _init_ui(self) -> None:
        super()._init_ui()
        self.ui.setWindowTitle(self.name.title())

    def set_params(self, *args) -> None:
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
    subscriber_affinity = GUI_THREAD  # The handlers touch widgets
    requirements = TabRequirements(needs_workspace=False, needs_channels=False)  # Think SampleInfoTab

    processing_type: str | None = None  # Not a pipeline tab by default
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
        self.tab_idx: int = tab_idx

        self.params = None
        # Plain None here; ExperimentTab replaces this with a raising @property
        # backed by _sample_manager so the type narrows to SampleManager after assignment.
        # GroupTab / BatchTab keep it as a plain None attribute (no property needed).
        self.sample_manager = None

        self.inited: bool = False
        self.setup_complete: bool = False
        self.params_set: bool = False

        self.advanced_controls_names: list[str] = []
        self.minimum_width: int = 200  # REFACTOR:
        self._selected_once: bool = False  # Whether the tab has been selected once

    @classmethod
    def requirements_fulfilled(cls, sample_manager) -> bool:
        """Whether this tab may be shown for the current sample_manager state."""
        reqs = cls.requirements
        workspace_ready = sample_manager is not None and sample_manager.workspace is not None
        ch_ready = workspace_ready and sample_manager.channels  #  or sample_manager.stitchable_channels
        return (workspace_ready or not reqs.needs_workspace) and (ch_ready or not reqs.needs_channels)

    def on_selected(self) -> None:
        """Called when this tab is selected; override in subclasses if needed."""
        pass

    @final
    def notify_selected(self) -> None:
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
    def get_tab_name(cls) -> str:
        snake = title_to_snake(cls.__name__.replace('Tab', ''))
        words = snake.split('_')
        tab_name = words[0].title() + ' ' + ' '.join(words[1:])
        return tab_name.strip()

    def _init_ui(self) -> None:
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

    def setup(self) -> None:
        """Setup more advanced features of the UI, notably the callbacks"""
        self._init_ui()  #  Call protected by "self.inited". Called in case the tab is not yet initialised
        if self.setup_complete:
            return

        self._swap_channels_tab_widget(self.with_add_btn)
        self._bind()
        self._connect_children_whats_this()

        self.setup_complete = True

    def _bind(self) -> None:
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute
        """
        raise NotImplementedError(f"Method _bind not implemented in {self.__class__.__name__}")

    def _set_params(self) -> None:
        """Set the params object which links the UI and the configuration file"""
        pass

    def _bind_params_signals(self) -> None:
        """Bind the signals of the params object"""
        pass

    def _get_channels(self) -> list[str]:
        return []  # Default to no channels when implementing a tab without channels

    def _setup_workers(self) -> None:
        """Set up the optional workers (which handle the computations) associated with this tab"""
        pass

    def finalise_workers_setup(self) -> None:
        """Finalise the setup of the workers. Typically called when the tab is selected"""
        pass

    def set_params(self, sample_params: SampleParameters | None = None) -> None:  # REFACTOR: rename to initialise or similar
        """Set the params object which links the UI and the configuration file"""
        self._set_params()
        self.params_set = True
        if self.params._get_view is not None:
            self.finalise_set_params()

    @final
    def finalise_set_params(self) -> None:
        if not self.params_set:
            warnings.warn(f'Params not set for {self.__class__}. Call set_params before finalise_set_params')
            return
        self._setup_workers()
        self.reconcile_channel_pages()
        self._load_config_to_gui()
        self._bind_params_signals()
        self.handle_advanced_checked()  # in case channel.QtControl in advanced_controls_names

    def reconcile_channel_pages(self) -> None:
        """Make the channel pages of the tab follow the config. Nothing to do for tabs without channel pages"""
        pass

    def get_channel_ui(self, channel: str) -> QWidget | None:
        """Get the UI widget for a specific channel"""
        return self.ui.channelsParamsTabWidget.get_channel_widget(channel)

    def _swap_channels_tab_widget(self, with_add_btn: bool = False) -> None:
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

    def add_channel_tab(self, channel: str = '') -> None:
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
        # TODO: check if still present:
        #  bug in StitchingTab when add_channel calls reconcile_children_from_view ->
        #  this triggers the creation of new channel params that don't have UI yet
        if channel not in self.params.keys():
            self._create_channel_params(channel)
            self._set_up_channel_page(page_widget, channel)
        else:
            if created:
                self.params[channel].teardown()
                self._set_up_channel_page(page_widget, channel)

    def _set_up_channel_page(self, page_widget: QWidget, channel: str) -> None:
        """Set up, bind and announce a new page. Setting the widgets up is not an edit: it writes nothing to the config"""
        with self.params[channel]._suppress_handlers():
            self._setup_channel(page_widget, channel)
        self._bind_channel(page_widget, channel)
        self._connect_children_whats_this(page_widget)
        self._on_channel_added(channel)

    def _create_channel_params(self, channel: str) -> None:
        if isinstance(self, PipelineTab) and not is_compound_channel(channel):
            # isinstance narrows self to PipelineTab → ExperimentTab,
            # so self.sample_manager here is SampleManager (raising property) ✓
            d_type = self.sample_manager.data_type(channel)
            self.params.add_channel(channel, d_type)
        else:
            self.params.add_channel(channel)

    def _on_channel_added(self, channel: str) -> None:
        """Child class should implement this to trigger any action needed when a channel is added"""
        pass

    def _on_channel_removed(self, channel_name: str) -> None:
        """Child classes should implement this to trigger cleanup actions after a channel is removed"""
        pass

    def _get_or_init_channel_ui(self, channel: str = '') -> tuple[str, QWidget, bool]:
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
        tuple[str, QWidget, bool]
            The channel name, the page widget and whether the page was just created.
        """
        tab_widget = self.ui.channelsParamsTabWidget
        page = tab_widget.get_channel_widget(channel)
        creating = not page
        if creating:
            page = create_clearmap_widget(self.channels_ui_name, patch_parent_class='QWidget')
            channel = tab_widget.add_channel_widget(page, name=channel)
        self._assign_channel_page_id(page, channel)
        return channel, page, creating

    def _channel_registry(self):
        """The registry of the channels of the sample, None for a tab without a sample"""
        return None

    def _assign_channel_page_id(self, page, channel: str) -> None:
        """
        Give the page the id of its channel (see ChannelPageId), if the registry knows the channel. A page made
        before the commit of its channel (a new channel tab) gets it at the next reconcile.
        """
        registry = self._channel_registry()
        if registry is not None:
            assign_channel_page_id_if_known(page, channel, registry)

    def _bind_channel(self, page_widget: QWidget, channel: str) -> None:
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute

        .. important::
            All button bindings for channels should be done here
        """
        pass

    def _setup_channel(self, page_widget: QWidget, channel: str) -> None:
        """
        Perform additional setup for the channel (before binding)
        For example set default values or populate lists

        .. note::
            Implement in subclass if needed
        """
        pass

    def handle_advanced_checked(self) -> None:
        """Activate the *advanced* mode which will display more controls"""
        if not hasattr(self.ui, 'advancedCheckBox'):  # e.g. BatchProcessingTab ATM
            return
        self.set_advanced_controls_visibility(self.ui.advancedCheckBox.isChecked())

    def set_advanced_controls_visibility(self, visible: bool) -> None:
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

    def disable(self) -> None:
        """Disable this tab (UI element)"""
        self.ui.setEnabled(False)

    def step_exists(self, step_name: str, file_list: list[PathLike] | PathLike) -> bool:
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
        the detailed *whatsThis* message of `whats_this_control`

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

    def _connect_children_whats_this(self, parent: QWidget | None = None) -> None:  # TODO: make tests safe
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

    def wrap_step(self, task_name: str, func: Callable, step_args: list[Any] | None = None,
                  step_kw_args: dict[str, Any] | None = None, n_steps: int = 1,
                  abort_func: Callable | None = None,
                  bypass_exceptions: Tuple[Exception] | None = None,
                  save_cfg: bool = True, nested: bool = True, close_when_done: bool = True,
                  main_thread: bool = False) -> None:
        """
        Start a new thread (unless `main_thread` = `True`) for `func` and show a progress dialog.
        Ensures that the UI remains responsive.

        Parameters
        ----------
        task_name : str
            The name of the task to be displayed
        func : Callable
            The function to run
        step_args : list[Any] | None
            The positional arguments to `func`
        step_kw_args : dict[str, Any] | None
            The keyword arguments to `func`
        n_steps : int
            The number of top level steps in the computation. This will be disabled if nested is False.
        abort_func : Callable | None
            The function to trigger to abort the execution of the computation (bound to the abort button)
        bypass_exceptions: Tuple[Exception] | None
            A tuple of exception types to bypass the GUI exception handling for.
            Defaults to empty.
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
        except Exception as err:
            if bypass_exceptions and isinstance(err, tuple(bypass_exceptions)):
                raise err  # Leave to caller
            if not getattr(err, '_gui_handled', False):
                func_name = getattr(func, '__name__', str(func))
                action = handle_exception(err, parent=self.main_window, context=f'{self.name} → {func_name}')
                err._gui_handled = True
                if action == 'reset_workspace':
                    self.main_window.trigger_workspace_reset()
                elif action == 'close':
                    QApplication.instance().quit()
            raise err  # Re-raise so caller loops (like run_stitching) can interrupt their sequence
        finally:
            # Access the private backing field directly instead of going through the property:
            #   - avoids triggering ExperimentTab's raising property when _sample_manager is None
            #   - returns None gracefully for GroupTab / BatchTab (no _sample_manager attribute at all)
            sm = getattr(self, '_sample_manager', None)  # WARNING: intentional private access
            if sm is not None and sm.workspace is not None:  # WARNING: hacky
                sm.workspace.executor = None  # FIXME: do not pass workspace but semaphore instead
            if close_when_done:
                self.main_window.progress_watcher.finish()
            else:
                # FIXME: message different if exception
                msg = f'{self.main_window.progress_watcher.main_step_name} finished'
                self.main_window.print_status_msg(msg)
                self.main_window.log_progress(f'    : {msg}')

    @contextmanager
    def busy_dialog(self, title: str, message: str = ''):
        """
        Show a "please wait" dialog (busy indicator, no abort) around a short computation which
        has to run in the main thread, typically building the widgets of a plot
        (see the ``busy_title`` argument of :meth:`wrap_plot`).

        Unlike :meth:`wrap_step`, it does not touch the progress watcher.

        .. code-block:: python

            with self.busy_dialog('Building the view'):
                do_something_short_in_the_main_thread()
        """
        dlg = make_simple_progress_dialog(title=title, sub_process_name=message, parent=self.main_window)
        dlg.subProgressBar.setMaximum(0)  # Busy (indeterminate) indicator
        dlg.buttonBox.setVisible(False)  # Nothing to abort
        QApplication.processEvents()  # Paint the dialog before blocking the main thread
        try:
            yield dlg
        finally:
            dlg.done(1)

    def wrap_plot(self, plot_method:  Callable[P, Any], *args: P.args, busy_title: str | None = None,
                  **kwargs: P.kwargs) -> 'list[DataViewer]':
        """
        Wrapper to plot a graph and display it in the main window.
        It also handles ClearMap exceptions gracefully

        Parameters
        ----------
        plot_method: function
            The function (or method) to plot the graph
        args: list
            The positional arguments to plot_function
        busy_title: str | None
            If given, a "please wait" dialog with this title is displayed while the plot is built.
            Use it for plots which take more than a moment to build (large marker sets, big volumes...)
            since it is not worth flashing a dialog for the others.
            Keyword only; it is not forwarded to plot_method.
        kwargs: dict
            The keyword arguments to plot_function

        Returns
        -------
        list[DataViewer]
            The data viewers returned by plot_function
        """
        self.main_window.clear_plots()
        with ExitStack() as busy:
            if busy_title:
                busy.enter_context(self.busy_dialog(busy_title))
            try:
                dvs = plot_method(*args, **kwargs)
            except Exception as err:
                busy.close()  # Before the error message box, not under it
                if not getattr(err, '_gui_handled', False):
                    func_name = getattr(plot_method, '__name__', str(plot_method))
                    handle_exception(err, parent=self.main_window, context=f'{self.name} → {func_name}')
                    err._gui_handled = True
                return []

            if not dvs:
                return []
            if isinstance(dvs[0], list):
                dvs, titles = dvs
                self.main_window.setup_plots(dvs, titles)
            else:
                self.main_window.setup_plots(dvs)
            from ClearMap.Visualization.Qt.DataViewer import DataViewer  # runtime import for isinstance
            return [widget for widget in dvs if isinstance(widget, DataViewer)]

    @staticmethod
    def ui_plot(status_msg: str = '', busy_title: str = '') -> Callable:
        """
        Decorator for tab methods that produce plot widgets.

        Handles:

        - Status message display.
        - Clearing previous plots.
        - ``MissingRequirementException`` / ``PlotGraphError``.
        - ``(dvs, titles)`` tuple return pattern.
        - ``setup_plots()`` call.
        - Filtering return to ``DataViewer`` instances only.

        Parameters
        ----------
        status_msg : str, optional
            Status message to display during plotting (default: ``''``).
        busy_title : str, optional
            If given, display a "please wait" dialog with this title while plotting (default: none).

        Returns
        -------
        Callable
            Decorated function that wraps plot logic with error handling
            and setup.

        Examples
        --------

        .. code-block:: python

            @GenericTab.ui_plot("Plotting p-values…")
            def plot_p_vals(self, ...):
                # pure plotting logic
                return [widget1, widget2]
        """
        def deco(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(self, *args, **kwargs):
                if status_msg:
                    self.main_window.print_status_msg(status_msg)
                bound_method = functools.partial(fn, self)
                return self.wrap_plot(bound_method, *args, busy_title=busy_title or None, **kwargs)
            return wrapper
        return deco


class ExperimentTab(GenericTab):
    _reconciling: bool = False  # Whether reconcile_channel_pages is running
    _reconcile_again: bool = False  # Whether it was asked again meanwhile

    def __init__(self, main_window: QMainWindow, ui_file_name: str, tab_idx: int, name: str = ''):
        super().__init__(main_window, ui_file_name, tab_idx, name)
        self._sample_manager: SampleManager | None = None
        self._exp_controller: ExperimentController | None = None
        self.sample_params: SampleParameters | None = None
        self._page_names: dict[PageKey, str] = {}  # The name each channel page was last made for

    # ------------------------------------------------------------------ #
    #  Raising properties — eliminates "| None has no attribute X" noise  #
    # ------------------------------------------------------------------ #

    @property
    def sample_manager(self) -> SampleManager:
        """
        The sample manager for this tab.

        Raises
        ------
        RuntimeError
            If accessed before :meth:`setup_sample_manager` has been called.
        """
        if self._sample_manager is None:
            raise RuntimeError(
                f'{type(self).__name__}.sample_manager accessed before assignment; '
                f'call setup_sample_manager() first'
            )
        return self._sample_manager

    @sample_manager.setter
    def sample_manager(self, value: SampleManager | None) -> None:
        self._sample_manager = value

    @property
    def exp_controller(self) -> ExperimentController:
        """
        The experiment controller for this tab.

        Raises
        ------
        RuntimeError
            If accessed before :meth:`set_controller` has been called.
        """
        if self._exp_controller is None:
            raise RuntimeError(
                f'{type(self).__name__}.exp_controller accessed before assignment; '
                f'call set_controller() first'
            )
        return self._exp_controller

    @exp_controller.setter
    def exp_controller(self, value: ExperimentController | None) -> None:
        self._exp_controller = value

    # ------------------------------------------------------------------ #

    def set_controller(self, controller: ExperimentController) -> None:
        self._exp_controller = controller

    def setup_sample_manager(self, sample_manager: SampleManager) -> None:
        self._sample_manager = sample_manager

    def set_params(self, sample_params: SampleParameters | None = None) -> None:  # FIXME: never set
        if sample_params:
            self.sample_params = sample_params  # REFACTORING: consider using sample_manager
        super().set_params()

    def _channel_registry(self):
        return self._sample_manager.channel_registry if self._sample_manager is not None else None

    def _desired_channel_pages(self) -> list[tuple[PageKey, str]]:
        """
        The pages the tab should show, each as (what it is a page of, its current name).
        The first is the identity, which a rename does not change, so a page can be told from a new one.
        """
        registry = self.sample_manager.channel_registry
        return [(registry.resolve(name), name) for name in self._get_channels()]

    def reconcile_channel_pages(self) -> None:
        """
        Make the channel pages follow the config, whatever the changes since the last time (it reads the
        sample manager, never the payload of an event, so it is the same from any trigger): the pages of the
        channels which are gone are dropped, those of the new channels are added and the page of a channel
        which was renamed is made again under its new name.

        .. important::
            The view follows the config: nothing here writes the config.

        .. note::
            A renamed page is made again, rather than renamed, because the buttons of a page were bound to
            the name the channel had then. They become channels (and the pages kept) when the tabs are migrated.
        """
        if self._reconciling:  # Asked from inside a reconcile (e.g. by an event of a commit): once more when done
            self._reconcile_again = True
            return
        self._reconciling = self._reconcile_again = True
        try:
            while self._reconcile_again:
                self._reconcile_again = False
                self._reconcile_channel_pages_once()
        finally:
            self._reconciling = False

    def _reconcile_channel_pages_once(self) -> None:
        if self._sample_manager is None or not self.params or not hasattr(self.ui, 'channelsParamsTabWidget'):
            return  # If not setup or tab without channels
        tab_widget = self.ui.channelsParamsTabWidget
        if not isinstance(tab_widget, ExtendableTabWidget):
            warnings.warn(f'Channel tab widget not finalised for {self.name}, skipping channel pages')
            return

        desired = {name: page_of for page_of, name in self._desired_channel_pages()}  # dedupe, keeps the order
        registry = self._channel_registry()
        changed = False

        # 0. Ids. The registry is the truth, the tab only shows it.
        for tab_name in tab_widget.get_channels_names():
            page = tab_widget.get_channel_widget(tab_name)
            page_id = channel_page_id_of(page)
            if page_id is None:  # Made before its channel was committed
                self._assign_channel_page_id(page, tab_name)
            elif not page_id.is_known_by(registry):  # Its channel is gone (removed, or the registry was replaced)
                self._drop_channel_page(tab_name)
                changed = True

        # 1. Renames: a page whose identity is known under another name is made again under the new one.
        for name, page_of in desired.items():
            previous_name = self._page_names.get(page_of)
            if previous_name is not None and previous_name != name:
                self._rename_channel_page(previous_name, name)
                changed = True

        # 2. Drops: a page whose channel is no longer in the config is dropped.
        for tab_name in tab_widget.get_channels_names():
            if tab_name not in desired:
                self._drop_channel_page(tab_name)
                changed = True

        # 3. Adds: a page for a new channel is added.
        for name in desired:
            if name not in tab_widget.get_channels_names():
                self.add_channel_tab(name)  # will call _setup_channel/_bind_channel
                changed = True

        # 4. Update the page names mapping to reflect the current state. (Record what is shown now, for the next call.)
        shown = tab_widget.get_channels_names()
        self._page_names = {page_of: name for name, page_of in desired.items() if name in shown}
        if changed:
            self._after_channels_reconciled(list(desired))
            self._setup_workers()
            self.handle_advanced_checked()  # The new pages follow the Advanced checkbox too

    def _drop_channel_page(self, name: str) -> None:
        self.params.drop_channel(name)
        self._on_channel_removed(name)

    def _rename_channel_page(self, old_name: str, new_name: str) -> None:
        """Make the page of ``old_name`` again, in the same place, under ``new_name``"""
        tab_widget = self.ui.channelsParamsTabWidget
        page, index = tab_widget.get_channel_widget(old_name, return_idx=True)
        if page is None:  # Already renamed (the user typed the new name in this page): nothing to make again
            return
        current_name = tab_widget.current_channel()
        self._drop_channel_page(old_name)
        if new_name not in tab_widget.get_channels_names():
            self.add_channel_tab(new_name)
        tab_widget.move_channel_widget(new_name, index)
        tab_widget.setCurrentWidget(tab_widget.get_channel_widget(new_name if current_name == old_name
                                                                  else current_name))

    def _after_channels_reconciled(self, desired_channels: list[str]) -> None:
        """
        Triggered at the end of channel reconciliation between UI and cfg
        Child classes should implement this method to e.g. populate lists or ComboBoxes
        that depend on the list of channels
        """
        pass

    def _setup_workers(self) -> None:
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
        self.config_handler: ConfigHandler | None = None
        self.group_controller: AnalysisGroupController | None = None

    def set_controller(self, controller: AnalysisGroupController) -> None:
        self.group_controller = controller


class PipelineTab(ExperimentTab, Generic[TWorker]):
    """
    Introduces the TWorker TypeVar into the hierarchy.
    Concrete subclasses bind it::

        class CellCounterTab(PostProcessingTab[CellDetector]): ...
    """
    requirements = TabRequirements(needs_workspace=True, needs_channels=True)

    processing_type = ''
    pipeline_name: str = ''  # Must be set in subclass
    workers_are_global: bool = False  # Whether the workers are shared between channels
    _workers_sub_steps: tuple[str, ...] | None = None  # The sub-steps to create workers for (if any)

    def __init__(self, main_window: QMainWindow, ui_file_name: str, tab_idx: int, name: str = ''):
        super().__init__(main_window, ui_file_name, tab_idx, name)
        self.sample_params = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.processing_type:
            raise NotImplementedError(f'Class "{cls.__name__}" must override "processing_type" with a string value.')

    @classmethod
    def requirements_fulfilled(cls, sample_manager) -> bool:
        if not super().requirements_fulfilled(sample_manager):
            return False
        # Config section must exist before the tab can be created
        if cls.pipeline_name:
            section = ALTERNATIVES_REG.pipeline_to_section_name(cls.pipeline_name)
            if section and not sample_manager.cfg_coordinator.get_config_view(section):
                return False
        return True

    def _iter_substeps(self) -> tuple[str | None, ...]:
        return self._workers_sub_steps or (None,)

    def _workers_channel_keys(self) -> list:
        """
        Worker keys handed to ExperimentController.reconcile_workers:
          - global workers: [None]
          - per-channel/per-pair: list from _get_channels()
        """
        return [None] if self.workers_are_global else list(self._get_channels())

    def setup_sample_manager(self, sample_manager: SampleManager) -> None:
        """
        Associate the sample_manager to the current tab

        Parameters
        ----------
        sample_manager : SampleManager
            The object that handles the sample data
        """
        self._sample_manager = sample_manager

    def _bind_btn(self, btn_name: str, func: Callable, channel: str | None = None,
                  page_widget: QWidget | None = None, **kwargs) -> None:
        if channel:
            getattr(page_widget, btn_name).clicked.connect(functools.partial(func, channel, **kwargs))
        else:
            getattr(self.ui, btn_name).clicked.connect(func)


class PreProcessingTab(PipelineTab[TWorker]):
    processing_type = 'pre'
    workers_are_global = True

    def __init__(self, main_window: QMainWindow, ui_file_name: str, tab_idx: int, name: str = ''):
        super().__init__(main_window, ui_file_name, tab_idx, name)

    @property
    def worker(self) -> TWorker:
        """
        The single global worker for this pipeline step.

        Raises
        ------
        RuntimeError
            If the worker has not yet been created (call _setup_workers first).
        """
        w = self.exp_controller.get_worker(self.pipeline_name)
        if w is None:
            raise RuntimeError(
                f'{type(self).__name__}.worker: no worker found for pipeline '
                f'"{self.pipeline_name}". Ensure _setup_workers() has been called.'
            )
        return w

    def _setup_workers(self) -> None:
        # Use private field to avoid triggering the raising property for the None check
        if self._sample_manager is None or not self._sample_manager.workspace_ready:
            return
        worker = self.worker
        if worker is not None:  # e.g. not SampleInfoTab
            self.wrap_step('Setting up worker', worker.setup_if_needed, n_steps=1,
                           save_cfg=False, nested=False)


class PostProcessingTab(PipelineTab[TWorker]):
    """
    Interface to all the tab managers in charge of post-processing the data.
    One particularity of a post-processing tab manager is that it includes the corresponding pre-processor.
    A tab manager includes a tab widget, the associated parameters
    and potentially a processor object which handles the computations.
    """
    processing_type = 'post'

    def __init__(self, main_window: QMainWindow, ui_file_name: str, tab_idx: int, name: str = ''):
        super().__init__(main_window, ui_file_name, tab_idx, name)

    def _setup_workers(self) -> None:
        # Use private field to avoid triggering the raising property for the None check
        if self._sample_manager is None or not self._sample_manager.workspace_ready:
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

    def get_worker(self, channel: str | tuple[str, str] | None = None,
                   substep: str | None = None) -> TWorker:
        """
        Retrieve the worker for the given channel / substep combination.

        Raises
        ------
        ValueError
            If substep is inconsistent with _workers_sub_steps.
        RuntimeError
            If no worker is found (setup not complete).
        """
        if substep and substep not in self._iter_substeps():
            raise ValueError(f'Sub-step {substep} not in {self._workers_sub_steps}')
        elif not substep and self._workers_sub_steps:
            raise ValueError(f'Sub-step must be specified, available: {self._workers_sub_steps}')
        if self.workers_are_global:
            channel = None
        worker = self.exp_controller.get_worker(self.pipeline_name, channel=channel, substep=substep)
        if worker is None:
            raise RuntimeError(
                f'{type(self).__name__}.get_worker: no worker for pipeline="{self.pipeline_name}" '
                f'{channel=!r} {substep=!r}. Ensure _setup_workers() has been called.'
            )
        return worker

    @contextmanager
    def debug_mode(self, channel: str, debug_status: str | bool):
        worker = self.get_worker(channel)
        status_backup = worker.workspace.debug
        try:
            worker.workspace.debug = debug_status
            yield
        finally:
            worker.workspace.debug = status_backup

    def create_tuning_sample(self, channel: str) -> None:
        """Create an array from a subset of the sample to perform tests on"""
        worker = self.get_worker(channel)
        if not hasattr(worker, 'create_test_dataset'):
            return
        self.wrap_step('Creating tuning sample', worker.create_test_dataset,
                       step_kw_args={'slicing': self.params[channel].slicing}, nested=False)

    def plot_slicer(self, slicer_prefix: str, tab: QWidget,
                    params: OrthoviewerSlicingMixin, channel: str | list[str]) -> None:
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
            sources = [self.sample_manager.get('resampled', channel=ch).open_ro() for ch in channel]
            if not np.all([src.shape == sources[0].shape for src in sources]):
                raise ValueError('Channels have different shapes')
            plot_image = np.mean([self.sample_manager.get('resampled', channel=ch).read() for ch in channel], axis=0)
            channel = channel[0]
        else:
            # asset = self.sample_manager.get('stitched', channel=channel)
            # if asset.exists:
            #     no_scale = True
            # else:  # missing stitched
            # FIXME: we cannot currently use stitched because we need to transpose it and it is too heavy for that
            asset = self.sample_manager.get('resampled', channel=channel)
            plot_image = asset.open_ro()
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
    def __init__(self, main_window: QMainWindow, tab_idx: int):
        super().__init__(main_window, title_to_snake(self.__class__.__name__), tab_idx)
        self.processing_type = 'batch'
        self.config_loader = None

    @property
    def initialised(self) -> bool:
        return self.params is not None

    def _setup_workers(self) -> None:
        pass

    def _bind(self) -> None:
        """
        Bind the signal/slots of the UI elements which are not
        automatically set through the params object attribute.

        .. warning::
            Child classes should call this method in their own _bind.
        """
        self.ui.resultsFolderPushButton.clicked.connect(self.setup_results_folder)
        self.ui.folderPickerHelperPushButton.clicked.connect(self.create_wizard)
        self.ui.batchToolBox.setCurrentIndex(0)

    def setup_results_folder(self) -> None:
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

    def create_wizard(self) -> SamplePickerDialog:
        return SamplePickerDialog(self.params.results_folder, self.params)  # FIXME: check if results_folder or make both equal with self.params.src_folder

    def _infer_default_results_folder(self) -> str | None:
        """Derive a default results folder from the experiment source
        folder, falling back to preferences.start_folder."""
        src = self.main_window.gui_controller.group_controller.group_base_dir
        return src
