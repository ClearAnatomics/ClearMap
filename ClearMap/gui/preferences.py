"""
This module handles the preferences of the graphical interface.
The values are persisted in a file
located in the home folder of the user. The preferences are used to store values related to
ClearMap performance as well as values that affect the appearance of the software.
"""
from ClearMap.gui.interfaces import GenericDialog
from ClearMap.gui.params import PreferencesParams


class PreferenceUi(GenericDialog):
    """
    This class handles the global preferences of the graphical interface
    It links a graphical dialog and the preference file located in the
    home folder of the user
    """
    def __init__(self, main_window):
        super().__init__(main_window, 'Preferences', 'preferences_editor')

    def setup(self, font_size):
        self.init_ui()
        self.ui.setMinimumHeight(700)  # FIXME: adapt to screen resolution

        self.setup_preferences()

        self.ui.buttonBox.connectApply(self.params.ui_to_cfg)
        self.ui.buttonBox.connectOk(self.apply_prefs_and_close)
        self.ui.buttonBox.connectCancel(self.ui.close)

        self.params.font_size = font_size

        self.ui.fontComboBox.currentFontChanged.connect(self.main_window.set_font)

    def set_params(self, *args):
        """
        Associate the params object to the dialog
        Parameters
        ----------
        args

        Returns
        -------

        """
        self.params = PreferencesParams(self.ui)

    def setup_preferences(self):
        """
        Setup the dialog with the values from the preference fil in the home folder
        Returns
        -------

        """
        self.set_params()
        machine_cfg_path = self.main_window.config_loader.get_default_path('machine')
        if self.main_window.file_exists(machine_cfg_path):
            self.params.read_configs(machine_cfg_path)
            self.params.cfg_to_ui()
        else:
            msg = 'Missing machine config file. Please ensure a machine_params.cfg file ' \
                  'is available at {}. This should be done at installation'.format(machine_cfg_path)
            self.main_window.print_error_msg(msg)
            raise FileNotFoundError(msg)

    def open(self):
        return self.ui.exec()

    def apply_prefs_and_close(self):
        self.params.ui_to_cfg()
        self.ui.close()
        self.main_window.reload_prefs()
