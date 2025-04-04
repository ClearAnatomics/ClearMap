"""
This module contains the interfaces to the different tabs and dialogs in the ClearMap GUI.
"""
import functools
import os
import pathlib
import warnings

from pathlib import Path
from shutil import copyfile

from abc import abstractmethod
from typing import final

import numpy as np
from PyQt5.QtWidgets import QWhatsThis, QToolButton, QWidget

from ClearMap.Utils.exceptions import MissingRequirementException, PlotGraphError
from ClearMap.Utils.utilities import title_to_snake
from ClearMap.config.config_loader import ConfigLoader
from ClearMap.gui.dialogs import get_directory_dlg
from ClearMap.gui.gui_utils import create_clearmap_widget, replace_widget
from ClearMap.gui.widget_monkeypatch_callbacks import recursive_patch_widgets
from ClearMap.gui.widgets import ExtendableTabWidget, SamplePickerDialog


class GenericUi:
    """
    The first layer of interface. This is not implemented directly but is the base class
    of GenericTab and GenericDialog, themselves interfaces
    """
    # FIXME: seems to be called several times
    def __init__(self, main_window, ui_file_name, name, widget_class_name):
        """

        Parameters
        ----------
        main_window: ClearMapGui
        name: str
        ui_file_name: str
        widget_class_name: str
        """
        self.main_window = main_window
        self.name = name
        self.ui_file_name = ui_file_name
        self.widget_class_name = widget_class_name
        self.ui = None
        self.params = None
        self.progress_watcher = self.main_window.progress_watcher

    def _init_ui(self):
        self.ui = create_clearmap_widget(f'{self.ui_file_name}.ui', patch_parent_class=self.widget_class_name)
        recursive_patch_widgets(self.ui)

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

    def set_progress_watcher(self, watcher):
        pass


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
class GenericTab(GenericUi):
    """
    The interface to all tab managers.
    A tab manager includes a tab widget, the associated parameters and
    an optional processor object which handles the computations.
    """
    processing_type = None  # Not a pipeline tab by default

    def __init__(self, main_window, ui_file_name, tab_idx, name=''):
        """

        Parameters
        ----------
        main_window: ClearMapGui
        ui_file_name: str
        tab_idx: int
        name: str
        """
        name = name or self.get_tab_name()
        super().__init__(main_window, ui_file_name, name, 'QTabWidget')

        self.inited = False
        self.setup_complete = False
        self.params_set = False
        self.params_finalised = False

        self.sample_manager = None
        self.sample_params = None

        # Channels
        self.channels_ui_name = ''  # The name of the ui file to create the channel tabs
        self.with_add_btn = False  # Whether to add the add channel (+) button to the channels tab

        self.tab_idx = tab_idx

        self.minimum_width = 200  # REFACTOR:

        self.advanced_controls_names = []

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
        self.ui.setMinimumWidth(self.minimum_width)
        if self.main_window.tabWidget.tabText(self.tab_idx) == self.name:
            self.main_window.tabWidget.removeTab(self.tab_idx)  # remove if same tab
        self.main_window.tabWidget.insertTab(self.tab_idx, self.ui, self.name.title())
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

    # @final
    def set_params(self, sample_params=None, cfg_path='', loaded_from_defaults=False):  # REFACTOR: rename to initialise or similar
        """Set the params object which links the UI and the configuration file"""
        if sample_params:
            self.sample_params = sample_params  # REFACTORING: consider using sample_manager
        if isinstance(self, PostProcessingTab):
            self.set_pre_processors()
        self._set_params()
        self._read_configs(cfg_path)
        if loaded_from_defaults:
            self._fix_config()

        self.params_set = True

        if not loaded_from_defaults:  #FIXME: check if needs to be called later for sample too
            self.finalise_set_params()  # FIXME: check if better to pass loaded_from_defaults or inspect channel names

    @final
    def finalise_set_params(self):
        if not self.params_set:
            warnings.warn(f'Params not set for {self.__class__}. Call set_params before finalise_set_params')
            return
        self._setup_workers()
        self._set_channels_names()
        self._create_channels()  # Creates missing channels. FIXME: What about existing but renamed ones?
        self._load_config_to_gui()
        self._bind_params_signals()
        self.handle_advanced_checked()  # in case channel.qtControl in advanced_controls_names
        self.params_finalised = True

    @abstractmethod
    def _set_channels_names(self):
        """Set the names of the channels based on the sample manager and data types"""
        pass

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

    def _create_channels(self):
        if not hasattr(self.ui, 'channelsParamsTabWidget'):
            return
        if not isinstance(self.ui.channelsParamsTabWidget, ExtendableTabWidget):
            warnings.warn(f'Channel tab widget not finalised for  {self.name}, skipping channel creation')
            return
        for channel in self._get_channels():
            if channel not in self.ui.channelsParamsTabWidget.get_channels_names():
                self.add_channel_tab(channel)

    def _setup_workers(self):
        """Setup the optional workers (which handle the computations) associated with this tab"""
        pass

    def finalise_workers_setup(self):
        """Finalise the setup of the workers. Typically called when the tab is selected"""
        pass

    def get_channel_ui(self, channel):
        """ Get the UI widget for a specific channel """
        return self.ui.channelsParamsTabWidget.get_channel_widget(channel)

    def _swap_channels_tab_widget(self, with_add_btn=False):
        """
        Substitute the placeholder channel tab widget by the dynamic one

        Parameters
        ----------
        with_add_btn: bool
            Whether to add the add channel button
        """
        if not hasattr(self.ui, 'channelsParamsTabWidget'):
            print(f'No channel tab widget found for {self.name}, skipping swap')
            return
        if not isinstance(self.ui.channelsParamsTabWidget, ExtendableTabWidget):
            layout = self.ui.channelsParamsTabWidgetLayout
            self.ui.channelsParamsTabWidget = replace_widget(self.ui.channelsParamsTabWidget,
                                                             ExtendableTabWidget(self.ui, with_add_tab=with_add_btn),
                                                             layout)

    # @final
    def add_channel_tab(self, channel=''):
        """
        Add a tab for a specific channel.
        This should then call the connect_channel method to setup the tab bindings.

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
        channel, page_widget = self._init_channel_ui(channel)
        if channel not in self.params.keys():
            if isinstance(self, PipelineTab):
                chan_params = self.sample_params.get(channel, {})
                d_type = getattr(chan_params, 'data_type', None)
                self.params.add_channel(channel, d_type)
            else:
                self.params.add_channel(channel)
        self._set_channel_config(channel)
        self._setup_channel(page_widget, channel)
        self._bind_channel(page_widget, channel)
        self._connect_children_whats_this(page_widget)

    def _init_channel_ui(self, channel: str = '') -> (str, QWidget):
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
        page_widget = create_clearmap_widget(self.channels_ui_name, patch_parent_class='QWidget')
        channel = self.ui.channelsParamsTabWidget.add_channel_widget(page_widget, name=channel)
        return channel, page_widget

    def _set_channel_config(self, channel):
        """
        Set the configuration for the channel

        .. note::
            Implement this method in the subclass if you want to force the same instance of config
            between the channel_params and the processor

        Parameters
        ----------
        channel: str
            The name of the channel
        """
        pass

    def _bind_channel(self, page_widget, channel):
        """
        Bind the signal/slots of the UI elements for `channel` which are not
        automatically set through the params object attribute

        .. important::
            All button bindings for channels should be done here
        """
        pass

    def _setup_channel(self, page_widget, channel):
        """
        Perform additional setup for the channel (before binding)
        For example set default values or populate lists

        .. note::
            Implement in subclass if needed
        """
        pass

    def _read_configs(self, cfg_path):  # REFACTOR: parse_configs
        """
        Read the configuration file associated with the params from the filesystem

        Parameters
        ----------
        cfg_path: str
            The path to the configuration file
        """
        self.params.read_configs(cfg_path)

    def _fix_config(self):  # TODO: check if could make part of self.params may not be possible since not set
        """Amend the config for the tabs that required live patching the config"""
        self.params.fix_cfg_file(self.params.config_path)

    def handle_advanced_checked(self):
        """Activate the *advanced* mode which will display more controls"""
        self.set_advanced_controls_visibility(self.ui.advancedCheckBox.isChecked())

    def set_advanced_controls_visibility(self, visible):
        """
        Set the visibility of the advanced controls

        Parameters
        ----------
        visible : bool
            Whether to show the advanced controls
        """
        for ctrl_name in self.advanced_controls_names:
            if ctrl_name.startswith('channel.'):
                if not self.params:  # Tab not yet initialised
                    return
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

    def step_exists(self, step_name, file_list):
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
        if isinstance(file_list, (str, pathlib.Path)):
            file_list = [file_list]
        for f_path in file_list:
            if not os.path.exists(f_path):
                self.main_window.print_error_msg(f'Missing {step_name} file {f_path}. '
                                                 f'Please ensure {step_name} is run first.')
                return False
        return True

    def connect_whats_this_btn(self, info_btn, whats_this_ctrl):
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

    def _connect_children_whats_this(self, parent=None):
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

    def wrap_step(self, task_name, func, step_args=None, step_kw_args=None, n_steps=1, abort_func=None, save_cfg=True,
                  nested=True, close_when_done=True, main_thread=False):  # FIXME: saving config should be default
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

        if save_cfg:
            self.params.ui_to_cfg()
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
                self.progress_watcher.finish()
            else:
                # FIXME: message different if exception
                msg = f'{self.progress_watcher.main_step_name} finished'
                self.main_window.print_status_msg(msg)
                self.main_window.log_progress(f'    : {msg}')

    def wrap_plot(self, plot_function, *args, **kwargs):
        """
        Wrapper to plot a graph and display it in the main window.
        It also handles MissingRequirementException and PlotGraphError

        Parameters
        ----------
        plot_function: function
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
            dvs = plot_function(*args, **kwargs)
        except MissingRequirementException as err:
            self.main_window.print_error_msg(f'Missing {plot_function.__name__} files {str(err)}. '
                                             f'Please ensure previous steps are run first.')
            return []
        except PlotGraphError as err:
            self.main_window.popup(str(err), base_msg='PlotGraphError')
            return []
        if isinstance(dvs[0], list):
            dvs, titles = dvs
            self.main_window.setup_plots(dvs, titles)
        else:
            self.main_window.setup_plots(dvs)
        return dvs


class PipelineTab(GenericTab):
    processing_type = ''
    def __init__(self, main_window, ui_file_name, tab_idx, name=''):
        super().__init__(main_window, ui_file_name, tab_idx, name)
        self.sample_params = None
        self.sample_manager = None  # REFACTORING: check if redundant
        self.relevant_data_types = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.processing_type:
            raise NotImplementedError(
                f"Class '{cls.__name__}' must override 'processing_type' with a string value."
            )

    def filter_relevant_channels(self, channels, must_exist=True):
        if channels is None:
            return
        if must_exist:
            new_channels = [c for c in channels if self.relevant_data_types == 'all' or
                            self.sample_params[c].data_type in self.relevant_data_types]
        else:
            new_channels = [c for c in channels if
                            self.relevant_data_types == 'all' or
                            c not in self.sample_params or
                            self.sample_params[c].data_type in self.relevant_data_types]
        return new_channels

    def set_relevant_data_types(self, data_types_dict):
        """
        Set the relevant data types for this tab

        Parameters
        ----------
        data_types_dict: dict
            The dictionary of data types
        """
        self.relevant_data_types = list(set([k for k, v in data_types_dict.items() if v == self.__class__]))

    def update_channels(self, former_channels=None, new_channels=None):
        """
        Update the configuration for the channels and the associated widgets

        Parameters
        ----------
        former_channels: List[str] | None
            The current list of channels
        new_channels: List[str] | None
            The updated list of channels
        """
        print(f'Updating channels for {self.name} from {former_channels} to {new_channels}')
        if not self.params:
            return
        if not former_channels and not new_channels:  # FIXME: we need to calculate former channels
            new_channels = self._get_channels()
            if not former_channels:
                former_channels = self.ui.channelsParamsTabWidget.get_channels_names()

        former_channels_set = set(former_channels or [])
        new_channels_set = set(new_channels or [])
        if former_channels_set == new_channels_set:
            return
        if removed_channels := former_channels_set - new_channels_set:
            for ch in removed_channels:
                self.params.pop(ch)
                # FIXME: try block to avoid error when channel not found
                self.ui.channelsParamsTabWidget.remove_channel_widget(ch)
        if added_channels := new_channels_set - former_channels_set:
            for ch in added_channels:
                try:
                    self._setup_workers()  # FIXME: called in too many places
                except KeyError:
                    pass
                self.add_channel_tab(ch)
                self._setup_workers()
                self.params.add_channel(ch)
        # if new_channels is a superset of former_channels, we can just add the new channels



    def setup_sample_manager(self, sample_manager):
        """
        Associate the sample_manager to the current tab

        Parameters
        ----------
        sample_manager : SampleManager
            The object that handles the sample data
        """
        self.sample_manager = sample_manager

    def _bind_btn(self, btn_name, func, channel=None, page_widget=None, **kwargs):
        if channel:
            getattr(page_widget, btn_name).clicked.connect(functools.partial(func, channel, **kwargs))
        else:
            getattr(self.ui, btn_name).clicked.connect(func)

    # @abstractmethod
    def _setup_workers(self):
        """
        Setup the optional workers (which handle the computations) associated with this tab

        .. warning::
            This method must be implemented in the subclasses of PipelineTab

        .. note::
            calls
                sample_params.ui_to_cfg in PreProcessingTab
                params.ui_to_cfg in PostProcessingTab  # TODO: check why
            is required for
                create_channels  (to get list of channels)
                bind_params_signals
        """
        pass


class PreProcessingTab(PipelineTab):
    processing_type = 'pre'
    def __init__(self, main_window, ui_file_name, tab_idx, name=''):
        super().__init__(main_window, ui_file_name, tab_idx, name)

    # @abstractmethod
    def _setup_workers(self):
        pass


class PostProcessingTab(PipelineTab):
    """
    Interface to all the tab managers in charge of post processing the data (e.e. typically detecting relevant info in the data).
    One particularity of a post processing tab manager is that it includes the corresponding pre processor.
    A tab manager includes a tab widget, the associated parameters
    and potentially a processor object which handles the computations.
    """
    processing_type = 'post'
    def __init__(self, main_window, ui_file_name, tab_idx, name=''):
        super().__init__(main_window, ui_file_name, tab_idx, name)
        self.stitcher = None
        self.aligner = None

    # @abstractmethod
    def _setup_workers(self):
        pass

    def set_pre_processors(self, stitcher=None, aligner=None):
        """
        Associate the pre-processors to the current tab

        Parameters
        ----------
        stitcher : Stitcher
            The object that handles the stitching of the sample data
        aligner : Aligner
            The object that handles the alignment of the sample data
        """
        if stitcher:
            self.stitcher = stitcher
        if aligner:
            self.aligner = aligner
        else:
            self.aligner = self.main_window.tab_managers['registration'].aligner

    def plot_slicer(self, slicer_prefix, tab, params, channel):
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


class BatchTab(GenericTab):
    def __init__(self,  main_window, tab_idx):
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
        results_folder = Path(get_directory_dlg(self.main_window.preference_editor.params.start_folder,
                                           'Select the folder where results will be written'))
        self.config_loader = ConfigLoader(results_folder)
        cfg_path = self.config_loader.get_cfg_path('batch', must_exist=False)
        if not cfg_path.exists():
            try:
                default_cfg_file_path = self.config_loader.get_default_path('batch')
                copyfile(default_cfg_file_path, cfg_path)
                self.params.fix_cfg_file(cfg_path)
            except FileNotFoundError as err:
                self.main_window.print_error_msg(f'Could not locate file for "batch"')
                raise err

        self.set_params(cfg_path=cfg_path)

        self.main_window.logger.set_file(results_folder / 'info.log')  # WARNING: set logs to global results folder
        self.main_window.error_logger.set_file(results_folder / 'errors.html')
        self.main_window.progress_watcher.log_path = self.main_window.logger.file.name

        self.params.read_configs(cfg_path)
        self.params.results_folder = results_folder  # FIXME: patch config

        self._load_config_to_gui()
        self._setup_workers()

    def create_wizard(self):
        self.params.ui_to_cfg()
        return SamplePickerDialog(self.params.results_folder, self.params)  # FIXME: check if results_folder or make both equal with self.params.src_folder
