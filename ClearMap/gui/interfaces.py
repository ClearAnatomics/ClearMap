from PyQt5.QtWidgets import QWhatsThis

from ClearMap.Utils.exceptions import MissingRequirementException
from ClearMap.gui.gui_utils import create_clearmap_widget


class GenericUi:
    """
    The first layer of interface. This is not implemented directly but is the base class
    of GenericTab and GenericDialog, themselves interfaces
    """
    def __init__(self, main_window, name, ui_file_name, widget_class_name):
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

    def init_ui(self):
        self.ui = create_clearmap_widget(f'{self.ui_file_name}.ui', patch_parent_class=self.widget_class_name)
        self.patch_button_boxes()

    def set_params(self, *args):
        """
        Set the params object which links the UI and the configuration file
        Parameters
        ----------
        args

        Returns
        -------

        """
        raise NotImplementedError()

    def load_config_to_gui(self):
        """
        Set every control on the UI to the value in the params
        Returns
        -------

        """
        self.params.cfg_to_ui()

    def set_progress_watcher(self, watcher):
        pass

    def patch_button_boxes(self):
        """
        Patch the button boxes in the ui so that the text corresponds to that defined
        in QtCreqtor

        Returns
        -------

        """
        self.main_window.patch_button_boxes(self.ui)


class GenericDialog(GenericUi):
    """
    Interface to any dialog associated with parameters
    """
    def __init__(self, main_window, name, file_name):
        super().__init__(main_window, name, file_name, 'QDialog')

    def init_ui(self):
        super().init_ui()
        self.ui.setWindowTitle(self.name.title())

    def set_params(self, *args):
        raise NotImplementedError()


class GenericTab(GenericUi):
    """
    The interface to all tab managers.
    A tab manager includes a tab widget,
     the associated parameters and potentially a processor object
     which handles the computations.
    """
    def __init__(self, main_window, name, tab_idx, ui_file_name):
        """

        Parameters
        ----------
        main_window: ClearMapGui
        name: str
        tab_idx: int
        ui_file_name: str
        """

        super().__init__(main_window, name, ui_file_name, 'QTabWidget')

        self.processing_type = None
        self.tab_idx = tab_idx

        self.minimum_width = 200  # REFACTOR:

    def init_ui(self):
        super().init_ui()
        self.ui.setMinimumWidth(self.minimum_width)
        self.main_window.tabWidget.removeTab(self.tab_idx)
        self.main_window.tabWidget.insertTab(self.tab_idx, self.ui, self.name.title())

    def set_params(self, *args):
        """
        Set the params object which links the UI and the configuration file
        Parameters
        ----------
        args

        Returns
        -------

        """
        raise NotImplementedError()

    def read_configs(self, cfg_path):  # FIXME: REFACTOR: parse_configs
        """
        Read the configuration file associated with the params from the filesystem

        Parameters
        ----------
        cfg_path

        Returns
        -------

        """
        self.params.read_configs(cfg_path)

    def fix_config(self):  # TODO: check if could make part of self.params may not be possible since not set
        """
        Amend the config for the tabs that required live patching the config
        Returns
        -------

        """
        self.params.fix_cfg_file(self.params.config_path)

    def disable(self):
        """
        Disable this tab (UI element)
        Returns
        -------

        """
        self.ui.setEnabled(False)

    def step_exists(self, step_name, file_list):
        """
        Check that prerequisite step step_name has been run and produced
        the outputs in file_list

        Parameters
        ----------
        step_name
        file_list

        Returns
        -------

        """
        if isinstance(file_list, str):
            file_list = [file_list]
        for f_path in file_list:
            if not os.path.exists(f_path):
                self.main_window.print_error_msg(f'Missing {step_name} file {f_path}. '
                                                 f'Please ensure {step_name} is run first.')
                return False
        return True

    def setup_workers(self):
        """
        Setup the optional workers (which handle the computations) associated with this tab
        Returns
        -------

        """
        pass

    def display_whats_this(self, widget):
        """
        Utility function to display the detailed *whatsThis* message
        associated with the control

        Parameters
        ----------
        widget

        Returns
        -------

        """
        QWhatsThis.showText(widget.pos(), widget.whatsThis(), widget)

    def connect_whats_this(self, info_btn, whats_this_ctrl):
        """
        Utility function to bind the info button to the display of
        the detailed *whatsThis* message associated with the control

        Parameters
        ----------
        widget

        Returns
        -------

        """
        info_btn.clicked.connect(lambda: self.display_whats_this(whats_this_ctrl))

    def wrap_step(self, task_name, func, step_args=None, step_kw_args=None, n_steps=1, abort_func=None, save_cfg=True,
                  nested=True, close_when_done=True, main_thread=False):  # FIXME: saving config should be default
        """
        This function wraps the computations of the tab. It should start a new thread to ensure that
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

        Returns
        -------

        """
        if step_args is None:
            step_args = []
        if step_kw_args is None:
            step_kw_args = {}

        if save_cfg:
            self.params.ui_to_cfg()
        if task_name:
            if not nested:
                n_steps = 0
            self.main_window.make_progress_dialog(task_name, n_steps=n_steps, abort=abort_func)

        try:
            if main_thread:
                func(*step_args, **step_kw_args)
            else:
                self.main_window.wrap_in_thread(func, *step_args, **step_kw_args)
        except MissingRequirementException as ex:
            self.main_window.print_error_msg(ex)
            self.main_window.popup(str(ex), base_msg=f'Could not run operation {func.__name__}', print_warning=False)
        finally:
            if self.preprocessor is not None and self.preprocessor.workspace is not None:  # WARNING: hacky
                self.preprocessor.workspace.executor = None
            if close_when_done:
                self.progress_watcher.finish()
            else:
                msg = f'{self.progress_watcher.main_step_name} finished'
                self.main_window.print_status_msg(msg)
                self.main_window.log_progress(f'    : {msg}')


class PostProcessingTab(GenericTab):
    """
    Interface to all the tab managers in charge of post processing the data (e.e. typically detecting relevant info in the data).
    One particularity of a post processing tab manager is that it includes the corresponding pre processor.
    A tab manager includes a tab widget,
     the associated parameters and potentially a processor object
     which handles the computations.
    """
    def __init__(self, main_window, name, tab_idx, ui_file_name):
        super().__init__(main_window, name, tab_idx, ui_file_name)

        self.preprocessor = None
        self.processing_type = 'post'

    def set_params(self, sample_params, alignment_params):
        """
        Set the params object which links the UI and the configuration file
        Parameters
        ----------
        args

        Returns
        -------

        """
        raise NotImplementedError()

    def setup_preproc(self, pre_processor):
        """
        Set the PreProcessor object associated with the sample

        Parameters
        ----------
        pre_processor : PreProcessor

        Returns
        -------

        """
        self.preprocessor = pre_processor

    def plot_slicer(self, slicer_prefix, tab, params):
        """
        Display the orthoslicer to pick a subset of 3D data.
        This is typically used to create a small  dataset to test parameters
        for long running operations before analysing the whole sample

        Parameters
        ----------
        slicer_prefix
        tab
        params

        Returns
        -------

        """
        self.main_window.clear_plots()
        # if self.preprocessor.was_registered:
        #     img = mhd_read(self.preprocessor.annotation_file_path)  # FIXME: does not work (probably compressed format)
        # else:
        img = self.preprocessor.workspace.source('resampled')
        self.main_window.ortho_viewer.setup(img, params, parent=self.main_window)
        dvs = self.main_window.ortho_viewer.plot_orthogonal_views()
        ranges = [[params.reverse_scale_axis(v, ax) for v in vals] for ax, vals in zip('xyz', params.slice_tuples)]
        self.main_window.ortho_viewer.update_ranges(ranges)
        self.main_window.setup_plots(dvs, ['x', 'y', 'z'])

        # WARNING: needs to be done after setup
        for axis, ax_max in zip('XYZ', self.preprocessor.raw_stitched_shape):  # FIXME: not always raw stitched
            getattr(tab, f'{slicer_prefix}{axis}RangeMax').setMaximum(ax_max)
