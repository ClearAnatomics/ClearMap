import functools
import os

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QCheckBox, QLabel, QLineEdit, QSpinBox, QFrame, QComboBox

from ClearMap.Utils.utilities import set_item_recursive, get_item_recursive
from ClearMap.config.config_loader import ConfigLoader
from ClearMap.Utils.exceptions import ConfigNotFoundError


class ParamLink:
    def __init__(self, keys, widget, attr_name=None, default=None, connect=True):
        self.keys = keys
        self.widget = widget
        self.attr_name = attr_name
        self.default = default
        self.connect = connect  # FIXME: can take function


class UiParameter(QObject):
    def __init__(self, tab, src_folder=None, params_dict=None):
        super().__init__()
        params_dict = params_dict if params_dict is not None else {}
        self.params_dict = params_dict
        self.tab = tab
        self.src_folder = src_folder
        self._config = None
        self._default_config = None
        self.cfg_subtree = None
        self.attrs_to_invert = []
        if self.params_dict:
            self.connect()

    def is_simple_attr(self, key):
        if key == 'params_dict' or not self.params_dict:
            return False
        is_graphical_param = key in self.params_dict.keys()
        if not is_graphical_param:
            return False
        attr = self.params_dict[key]
        is_simple = isinstance(attr, ParamLink)
        return is_simple

    def __getattr__(self, item):  # FiXME: use binder.default
        if self.is_simple_attr(item):
            binder = self.params_dict[item]
            widget = binder.widget
            if isinstance(widget, QCheckBox):
                return widget.isChecked()
            elif isinstance(widget, (QLabel, QLineEdit)):
                return widget.text()
            elif isinstance(widget, QSpinBox):
                return widget.value()
            elif isinstance(widget, QFrame):
                name = widget.objectName()
                if name.endswith('let'):  # singlets, doublets and triplets
                    return widget.getValue()
                elif name.endswith('TextEdit'):
                    return widget.text()
                else:
                    raise ValueError(f'Unrecognised frame with name "{name}"')
            elif isinstance(widget, QComboBox):  # FIXME: 'None' or '' = None
                return widget.currentText()
            else:
                raise NotImplementedError(f'Unhandled object of type {type(widget)}')
        else:
            raise AttributeError(f'Unknown attribute {item}')

    def __setattr__(self, key, value):
        if self.is_simple_attr(key):
            binder = self.params_dict[key]
            widget = binder.widget
            if isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, (QLabel, QLineEdit)):
                widget.setText(value)
            elif isinstance(widget, QSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QFrame):  # frames = singlets, doublets and triplets
                name = widget.objectName()
                if name.endswith('let'):  # singlets, doublets and triplets
                    widget.setValue(value)
                elif name.endswith('TextEdit'):
                    widget.setText(value)
                else:
                    raise ValueError(f'Unrecognised frame with name "{name}"')
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(value)    # FIXME: None = 'None' or ''
            else:
                raise NotImplementedError(f'Unhandled object of type {type(widget)}')
        else:
            QObject.__setattr__(self, key, value)
            # print(hasattr(self, key))

    def connect_simple_widgets(self):
        for k in self.params_dict.keys():
            if self.is_simple_attr(k) and self.params_dict[k].connect and not hasattr(self, f'handle_{k}_changed'):
                self.__connect_widget(k)

    def __connect_widget(self, key):
        widget = self.params_dict[key].widget
        callback = functools.partial(self.handle_widget_changed, attr_name=key)
        if isinstance(widget, QCheckBox):
            widget.stateChanged.connect(callback)  # FIXME: change depending on type
        elif isinstance(widget, QSpinBox):
            widget.valueChanged.connect(callback)
        elif isinstance(widget, (QLabel, QLineEdit)):  # WARNING: QLabel before QFrame because QLabel inherits QFrame
            widget.textChanged.connect(callback)
        elif isinstance(widget, QFrame):
            name = widget.objectName()
            if name.endswith('let'):  # singlets, doublets and triplets
                widget.valueChangedConnect(callback)
            elif name.endswith('PlainTextEdit'):  # WARNING: plainTextEdit.textChanged is argument less
                widget.textChangedConnect(functools.partial(self.handle_widget_changed, value=None, attr_name=key))
            elif name.endswith('TextEdit'):
                widget.textChangedConnect(callback)
            else:
                raise ValueError(f'Unrecognised frame with name "{name}"')
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(callback)
        else:
            raise ValueError(f'Unhandled object of type {type(widget)}')

    def handle_widget_changed(self, value, attr_name):
        keys = self.params_dict[attr_name].keys
        property_value = getattr(self, attr_name)
        set_item_recursive(self.config, keys, property_value)

    def connect(self):
        """Connect GUI slots here"""
        pass

    def fix_cfg_file(self, f_path):
        """Fix the file if it was copied from defaults, tailor to current sample"""
        pass

    @property
    def path(self):
        return self._config.filename

    def read_configs(self, cfg_path):
        self._config = ConfigLoader.get_cfg_from_path(cfg_path)
        if not self._config:
            raise ConfigNotFoundError
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        self._default_config = ConfigLoader.get_cfg_from_path(ConfigLoader.get_default_path(cfg_name))

    @property
    def config_path(self):
        return self._config.filename

    @property
    def config(self):
        if self.cfg_subtree:
            return get_item_recursive(self._config, self.cfg_subtree)
        else:
            return self._config

    @property
    def default_config(self):
        if self.cfg_subtree:
            return get_item_recursive(self._default_config, self.cfg_subtree)
        else:
            return self._default_config

    def write_config(self):
        self._config.write()

    def reload(self):
        self._config.reload()

    def _translate_state(self, state):
        if state is True:
            state = Qt.Checked
        elif state is False:
            state = Qt.Unchecked
        else:
            raise NotImplementedError(f'Unknown state {state}')
        return state

    def ui_to_cfg(self):
        self._ui_to_cfg()
        self.write_config()

    def _ui_to_cfg(self):
        pass

    def cfg_to_ui(self):  # FIXME: add reload here but make sure that compatible with all uses (especially UiParamCollection)
        if not self.params_dict:
            raise NotImplementedError('params_dict not set')
        else:
            any_amended = False
            for attr, keys_list in self.params_dict.items():
                if isinstance(keys_list, ParamLink):
                    keys_list = keys_list.keys
                    if keys_list is None:  # For params without cfg
                        continue
                current_amended = False
                try:
                    val = get_item_recursive(self.config, keys_list)
                except KeyError:  # TODO: add msg
                    val = get_item_recursive(self.default_config, keys_list)
                    any_amended = True
                    current_amended = True
                if attr in self.attrs_to_invert:
                    val = not val
                if current_amended:
                    # Update the config
                    set_item_recursive(self.config, keys_list, val)
                # Update the UI
                setattr(self, attr, val)  # comes after the cfg otherwise, key will be missing in the callback
            if any_amended:
                self.ui_to_cfg()  # Add the newly parsed field

    def is_checked(self, check_box):
        return check_box.checkState() == Qt.Checked

    def set_check_state(self, check_box, state):
        state = self._translate_state(state)
        check_box.setCheckState(state)

    def sanitize_nones(self, val):
        return val if val is not None else -1

    def sanitize_neg_one(self, val):
        return val if val != -1 else None


class UiParameterCollection:
    """
    For multi-section UiParameters that share the same config file. This ensures the file remains consistent.
    """
    def __init__(self, tab, src_folder=None):
        self.tab = tab
        self.src_folder = src_folder
        self.config = None

    def fix_cfg_file(self, f_path):
        """Fix the file if it was copied from defaults, tailor to current sample"""
        pass

    @property
    def params(self):
        raise NotImplementedError('Please subclass UiParameterCollection and implement params property')

    def read_configs(self, cfg_path):
        self.config = ConfigLoader.get_cfg_from_path(cfg_path)
        if not self.config:
            raise ConfigNotFoundError
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        default_path = ConfigLoader.get_default_path(cfg_name)
        self._default_config = ConfigLoader.get_cfg_from_path(default_path)
        for param in self.params:
            param._config = self.config
            param._default_config = self._default_config

    @property
    def config_path(self):
        return self.config.filename

    def write_config(self):
        self.config.write()

    def reload(self):
        self.config.reload()

    def ui_to_cfg(self):
        self.write_config()

    def cfg_to_ui(self):
        for param in self.params:
            param.cfg_to_ui()
