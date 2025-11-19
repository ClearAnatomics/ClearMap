"""
This module handles the preferences of the graphical interface.
The values are persisted in a file
located in the home folder of the user. The preferences are used to store values related to
ClearMap performance as well as values that affect the appearance of the software.
"""
from ClearMap.gui.params import PreferencesParams
from ClearMap.gui.tabs_interfaces import GenericDialog


class PreferenceUi(GenericDialog):
    """
    This class handles the global preferences of the graphical interface
    It links a graphical dialog and the preference file located in the
    home folder of the user
    """
    def __init__(self, main_window):
        super().__init__(main_window, 'Preferences', 'preferences_editor')

    def setup(self, font_size, **kwargs):
        self._init_ui()
        self.ui.setMinimumHeight(700)  # FIXME: adapt to screen resolution

        self.setup_preferences(**kwargs)

        # self.ui.buttonBox.connectApply(self.params.ui_to_cfg)
        self.ui.buttonBox.connectOk(self.apply_prefs_and_close)
        self.ui.buttonBox.connectCancel(self.ui.close)

        self.params.font_size = font_size

        self.ui.fontComboBox.currentFontChanged.connect(self.main_window.set_font)

    def set_params(self, *args, **kwargs):
        """
        Associate the params object to the dialog
        Parameters
        ----------
        args

        Returns
        -------

        """
        self.params = PreferencesParams(self.ui, **kwargs)

    def setup_preferences(self, **kwargs):
        """
        Setup the dialog with the values from the preference fil in the home folder
        Returns
        -------

        """
        view_provider = kwargs.pop('view_provider', None)
        apply_patch = kwargs.pop('apply_patch', None)

        self.set_params(**kwargs)

        machine_cfg_path = self.main_window.config_loader.get_global_path('machine')
        if self.main_window.file_exists(machine_cfg_path):
            if view_provider is not None:
                self.params.bind_view_provider(view_provider)
            if apply_patch is not None:
                self.params.bind_apply_patch(apply_patch)
            self.params.cfg_to_ui()
        else:
            msg = 'Missing machine config file. Please ensure a machine_params.cfg file ' \
                  'is available at {}. This should be done at installation'.format(machine_cfg_path)
            self.main_window.print_error_msg(msg)
            raise FileNotFoundError(msg)

    def open(self):
        return self.ui.exec()

    def apply_prefs_and_close(self):
        # self.params.ui_to_cfg()
        self.ui.close()
        self.main_window.reload_prefs()  # REFACTOR: use event bus?
