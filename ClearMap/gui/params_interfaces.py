import ast
import functools
import warnings
from abc import ABC, abstractmethod, ABCMeta
from pathlib import Path
from typing import List, Any

from importlib_metadata import version
from packaging.version import Version

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QLabel, QLineEdit, QSpinBox, QFrame, QComboBox, QPlainTextEdit, QTextEdit, \
    QGroupBox, QWidget, QListWidget, QDoubleSpinBox

from ClearMap.config import convert_config_versions
from ClearMap.config.config_loader import ConfigLoader
from ClearMap.Utils.utilities import set_item_recursive, get_item_recursive
from ClearMap.Utils.exceptions import ConfigNotFoundError, ClearMapValueError
from ClearMap.gui.widgets import ExtendableTabWidget, FileDropListWidget

CLEARMAP_VERSION = Version(version('ClearMap'))

class ParamLink:
    """
    A class to link a widget to a config file. This is the object passed in
    the ``params_dict`` attribute of the ``UiParameter`` class.
    It is used in the __getattr__ and __setattr__ and __connect_widget
    methods to link the widget to the config file.

    Attributes
    ----------
    keys: list(str) (optional)
        The list of keys to access the value in the config file
        If None, the attribute is not present in the config file and
        will not be connected
    widget: QWidget
        The GUI widget to be connected to the config file
    attr_name: str
        The name of the attribute in the class
    default: Any
        The default value to be used if the value is not found in the config file
    connect: function | bool
        A function to call to do the binding between the widget and the corresponding section of
        the config file or a boolean to indicate whether the widget should be connected to the config file.
        If set to `True`, the widget will be connected automatically by simply updating the value in
        the config file. For more complex widgets, a function can be passed to handle the connection.
        Set to `False` to not connect or connect manually.
    """
    def __init__(self, keys: List[str] | None,
                 widget: QWidget,
                 attr_name: str | None=None,
                 default: Any | None=None,
                 connect: bool=True,
                 missing_ok: bool=False):
        if keys is None:
            connect = False
        self.keys = keys
        self.widget = widget
        self.attr_name = attr_name
        self.default = default
        self.missing_ok = missing_ok or bool(default)
        self.connect = connect

    def has_connect_function(self):
        return callable(self.connect)


def frame_getter(widget):
    """
    For compound objects wrapped in a frame like
    singlets, doublets and triplets, with or without a checkbox
    or for text edits with an enable checkbox
    """
    name = widget.objectName()
    if name.endswith('let'):  # singlets, doublets and triplets
        return widget.getValue()
    elif name.endswith('TextEdit'):
        return widget.text()
    else:
        raise ValueError(f'Unrecognised frame with name "{name}"')


def frame_setter(widget, value):
    """
    For compound objects wrapped in a frame like
    singlets, doublets and triplets, with or without a checkbox
    or for text edits with an enable checkbox
    """
    name = widget.objectName()
    if name.endswith('let'):  # singlets, doublets and triplets
        widget.setValue(value)
    elif name.endswith('TextEdit'):
        widget.setText(value)
    else:
        raise ValueError(f'Unrecognised frame with name "{name}"')


def combobox_getter(widget):
    txt = widget.currentText()
    if txt in ('None', ''):
        return None
    return txt


def combobox_setter(widget, value):
    if value is None:
        value = 'None'  # TODO: decide 'None' or ''
    found = widget.findText(value)
    if found == -1 and widget.count() > 0:
        warnings.warn(f'Value "{value}" not found in combobox, '
                      f'possible values are "{", ".join([widget.itemText(i) for i in range(widget.count())])}"')
    widget.setCurrentText(value)


def line_edit_getter(widget):
    txt = widget.text()
    if txt in ('None', ''):
        return None
    count = sum([txt.count(symbol) for symbol in '[]{}'])
    if count:  # List or dict
        if count % 2 == 0:  # (not perfect but should work for most cases)
            txt = ast.literal_eval(txt)
        else:
            warnings.warn(f'Unbalanced brackets in {txt}')
    return txt


def list_widget_getter(widget):
    widgets = [widget.item(i) for i in range(widget.count())]
    return [itm.text() for itm in widgets]


def list_widget_setter(widget, itm_list):
    try:
        widget.clear()
        widget.addItems(itm_list)
    except RuntimeError:  # Widget has been deleted
        print(f'Widget {widget} has been deleted')
        pass


widget_getters = {
    QCheckBox: lambda w: w.isChecked(),
    QLabel: lambda w: w.text(),
    QLineEdit: line_edit_getter,
    QSpinBox: lambda w: w.value(),
    QDoubleSpinBox: lambda w: w.value(),
    QPlainTextEdit: lambda w: w.toPlainText(),
    QTextEdit: lambda w: w.toHtml(),  # TODO: check if this always what we want
    QComboBox: combobox_getter,
    QGroupBox: lambda w: w.isChecked(),
    FileDropListWidget: list_widget_getter,
    QListWidget: list_widget_getter,  # includes FileDropListWidget
    QFrame: frame_getter,  # QFrame always last because least specific
}

widget_setters = {
    QCheckBox: lambda w, v: w.setChecked(v),
    QLabel: lambda w, v: w.setText(v),
    QLineEdit: lambda w, v: w.setText(str(v)),
    QSpinBox: lambda w, v: w.setValue(v),
    QDoubleSpinBox: lambda w, v: w.setValue(v),
    QPlainTextEdit: lambda w, v: w.setPlainText(v),
    QTextEdit: lambda w, v: w.setText(v),
    QComboBox: combobox_setter,
    QGroupBox: lambda w, v: w.setChecked(v),
    FileDropListWidget: list_widget_setter,
    QListWidget: list_widget_setter,  # includes FileDropListWidget
    QFrame: frame_setter,  # QFrame always last because least specific
}

widget_connectors = {
    QCheckBox: lambda w, cb: w.stateChanged.connect(cb),# FIXME: change depending on type
    QLabel: lambda w, cb: w.textChanged.connect(cb),
    QLineEdit: lambda w, cb: w.textChanged.connect(cb),
    QSpinBox: lambda w, cb: w.valueChanged.connect(cb),
    QDoubleSpinBox: lambda w, cb: w.valueChanged.connect(cb),
    QPlainTextEdit: lambda w, cb: w.textChanged.connect(cb),
    QComboBox: lambda w, cb: w.currentTextChanged.connect(cb),
    QGroupBox: lambda w, cb: w.toggled.connect(cb),
    FileDropListWidget: lambda w, cb: w.itemsChanged.connect(cb),
    QListWidget: lambda w, cb: w.itemChanged.connect(cb),  # FIXME: not the right signal (should be for any change)
    QFrame: None,  # updated in __connect_widget  # QFrame always last because least specific
}


class UiParameter(QObject):
    """
    This is a class to link the GUI widgets to the config file.
    This is done automatically from parsing the ``params_dict`` attribute.
    The ``params_dict`` attribute is a dictionary of the form::

        {'attr_name': ParamLink(keys, widget, attr_name=, default=, connect=)}
        or
        {'attr_name': keys,}

    where:

    - ``attr_name`` is the name of the attribute in the class
    - ``keys`` is a list of keys (chain) to access the value in the config file.

    If ``None``, the attribute is not connected widget is the GUI widget
    ``connect`` is a boolean to indicate whether the widget should be connected
    to the config file. Set to ``False`` to not connect or connect manually.
    If the value is not a ``ParamLink``, it is assumed that the keys point to the value and
    the widget connection is done through accessors and mutators.

    Attributes
    ----------
    tab: QWidget
        The tab where the widget is located
    params_dict: dict
        The dictionary of attributes and their corresponding ParamLink
    attrs_to_invert: list
        The list of attributes that should be inverted. This is useful for checkboxes.
    cfg_subtree: list
        The list of keys to access the config file. This allows a relative path to be used from the
        whole config file.
    """
    def __init__(self, tab, params_dict=None):
        super().__init__()
        params_dict = params_dict if params_dict is not None else {}
        self.params_dict = params_dict
        self.tab = tab
        self._config = None
        self._default_config = None
        self._cfg_subtree = []
        self.attrs_to_invert = []
        if self.params_dict:
            self.connect()

    @property
    def cfg_subtree(self):
        return self._cfg_subtree

    @cfg_subtree.setter
    def cfg_subtree(self, value):
        self._cfg_subtree = value

    def is_simple_attr(self, key):
        if key == 'params_dict' or not hasattr(self, 'params_dict') or self.params_dict is None:
            return False
        if not key in self.params_dict.keys():  # Not set through GUI
            return False
        attr = self.params_dict[key]
        is_simple = isinstance(attr, ParamLink)
        return is_simple

    def __getattr__(self, item):  # FiXME: use binder.default
        if self.is_simple_attr(item):
            binder = self.params_dict[item]
            widget = binder.widget
            for k in widget_getters.keys():
                if isinstance(widget, k): # REFACTOR: use widget_getters.get(type(widget))
                    return widget_getters[k](widget)
            else:
                raise NotImplementedError(f'Unhandled object of type {type(widget)}')
        else:
            raise AttributeError(f'{self.__class__.__name__}, unknown attribute "{item}"')

    def __setattr__(self, key, value):
        if self.is_simple_attr(key):
            binder = self.params_dict[key]
            widget = binder.widget
            for w_type, setter in widget_setters.items():
                if isinstance(widget, w_type):  # REFACTOR: use widget_setters.get(type(widget))
                    setter(widget, value)
                    return
            else:
                raise NotImplementedError(f'Unhandled object of type {type(widget)}')
        else:
            QObject.__setattr__(self, key, value)

    def get(self, item, default_value=None):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            return default_value

    @property
    def config(self):
        if self.cfg_subtree:
            if self._config is None:
                raise ValueError(f'Config not set for {self.__class__.__name__}')
            return get_item_recursive(self._config, self.cfg_subtree)
        else:
            return self._config

    @property
    def default_config(self):
        if self.cfg_subtree:
            if self.name in self.cfg_subtree:
                default_channel = self._default_config['channels'].keys()[0]
                default_sub_tree = self.cfg_subtree.copy()
                default_sub_tree[default_sub_tree.index(self.name)] = default_channel
                return get_item_recursive(self._default_config, default_sub_tree)
            else:
                try:
                    return get_item_recursive(self._default_config, self.cfg_subtree)
                except KeyError as err:
                    if self.name in str(err):
                        raise KeyError(f'Could not find channel {self.name} in default config file. '
                                       f'config sub tree: {self.cfg_subtree}')
        else:
            return self._default_config

    def connect_simple_widgets(self):
        for k in self.params_dict.keys():
            link = self.params_dict[k]
            if isinstance(link, ParamLink) and link.has_connect_function():
                self.__connect_widget(k)
            elif self.is_simple_attr(k) and link.connect and not hasattr(self, f'handle_{k}_changed'):
                self.__connect_widget(k)

    def __connect_widget(self, key):
        widget = self.params_dict[key].widget
        callback = self.params_dict[key].connect
        if not callable(callback):
            callback = functools.partial(self.handle_widget_changed, attr_name=key)
        def connect_frame(widget, callback):
            name = widget.objectName()
            if name.endswith('let'):  # singlets, doublets and triplets
                try:
                    widget.valueChangedConnect(callback)
                except AttributeError as err:
                    print(f'No valueChangedConnect for {name}')
                    raise err
            elif name.endswith('TextEdit'):  # WARNING: plainTextEdit.textChanged is argument less
                # widget.textChangedConnect(callback)
                widget.textChangedConnect(functools.partial(self.handle_widget_changed, value=None, attr_name=key))
            # elif name.endswith('TextEdit'):
            #     widget.textChangedConnect(callback)
            else:
                raise ValueError(f'Unrecognised frame with name "{name}"')
        widget_connectors[QFrame] = connect_frame
        for w_type, connector in widget_connectors.items():
            if isinstance(widget, w_type):
                connector(widget, callback)
                return
        else:
            raise ValueError(f'Unhandled object of type {type(widget)}')

    def handle_widget_changed(self, value=None, attr_name=None):
        keys = self.params_dict[attr_name].keys
        property_value = getattr(self, attr_name)
        try:
            set_item_recursive(self.config, keys, property_value)
        except ValueError as err:
            print(f'Error setting {keys} to {property_value} for {attr_name}')
            raise err

    def connect(self):
        """Connect GUI slots here"""
        pass

    def fix_cfg_file(self, f_path):
        """Fix the file if it was copied from defaults, tailor to current sample"""
        pass

    def read_configs(self, cfg_path):
        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            cfg_path = ConfigLoader.get_cfg_path(cfg_path, allow_previous_versions=True)

        self._config = ConfigLoader.get_cfg_from_path(cfg_path)
        if not self._config:
            raise ConfigNotFoundError

        if Version(self._config.get('clearmap_version', '2.1.0')) < CLEARMAP_VERSION:
            cfg_path = convert_config_versions.convert(cfg_path, backup=True, overwrite=True)

        cfg_name = ConfigLoader.strip_version_suffix(cfg_path.stem)
        try:
            self._default_config = ConfigLoader.get_cfg_from_path(ConfigLoader.get_default_path(cfg_name))
        except FileNotFoundError:
            from ClearMap.config import config_loader
            warnings.warn(f'No file found for {cfg_name} in clearmap configuration directory,'
                          f' using package default instead (from "{Path(config_loader.__file__).parent.absolute()}").'
                          f'Please consider copying the default file to the clearmap configuration directory.')
            self._default_config = ConfigLoader.get_cfg_from_path(ConfigLoader.get_default_path(cfg_name, from_package=True))

    @property
    def config_path(self):
        return self._config.filename

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
                try:
                    val, current_amended = self._get_config_value(keys_list)
                except (KeyError, ConfigNotFoundError) as err:
                    if self.params_dict[attr].missing_ok:
                        val = self.params_dict[attr].default
                        current_amended = True
                    else:
                        raise err
                if attr in self.attrs_to_invert:
                    val = not val
                if current_amended: # Update the config
                    any_amended = True
                    set_item_recursive(self.config, keys_list, val)
                # Update the UI
                setattr(self, attr, val)  # comes after the cfg otherwise, key will be missing in the callback
            if any_amended:
                self.ui_to_cfg()  # Add the newly parsed field

    def get_config_base_name(self, n_iter=5):
        try:
            return Path(self.config.filename).stem
        except AttributeError:
            cfg = self.config
            for i in range(n_iter):
                if hasattr(cfg, 'filename'):
                    return Path(cfg.filename).stem
                else:
                    cfg = cfg.parent
            else:
                raise ValueError(f'Could not find config filename after {n_iter} iterations. '
                                 f'Please set it manually or check the config file.')

    def _get_config_value(self, keys_list):
        """
        Try to get the value (recursively) from the config file.
        If it is not found, get it from the default config.

        Parameters
        ----------
        keys_list: List[str]
            The list of keys to access the value in the config file

        Returns
        -------
        Any, bool
            The value found, and a boolean indicating whether the value was found in the default config
        """
        try:
            val = get_item_recursive(self.config, keys_list)
            return val, False
        except KeyError:
            if self._default_config is None:
                try:
                    cfg_name = ConfigLoader.strip_version_suffix(self.get_config_base_name())
                    self._default_config = ConfigLoader.get_cfg_from_path(ConfigLoader.get_default_path(cfg_name))
                except FileNotFoundError:
                    try:
                        self._default_config = ConfigLoader.get_cfg_from_path(
                            ConfigLoader.get_default_path(cfg_name, from_package=True))
                    except FileNotFoundError:
                        raise ConfigNotFoundError(f'Default config not set for {self.__class__.__name__}. '
                                                  f'Regular config path is {self._config.filename}')
            val = get_item_recursive(self.default_config, keys_list)
            return val, True

    def is_checked(self, check_box):
        return check_box.checkState() == Qt.Checked

    def set_check_state(self, check_box, state):
        state = self._translate_state(state)
        check_box.setCheckState(state)

    def sanitize_nones(self, val):
        """
        In python, the maximum item (last item) in a list can be accessed with -1.
        In ClearMap, the maximum is often represented by None.
        This function is used to do the conversion.

        Parameters
        ----------
        val: int or None
            The value to be converted

        Returns
        -------
        int
            The converted value
        """
        return val if val is not None else -1

    def sanitize_neg_one(self, val):
        """
        In python, the maximum item (last item) in a list can be accessed with -1.
        In ClearMap, the maximum is often represented by None.
        This function is used to do the conversion.

        Parameters
        ----------
        val: int
            The value to be converted

        Returns
        -------
        int or None
            The converted value
        """
        return val if val != -1 else None


class UiParameterMeta(type(UiParameter), ABCMeta):
    pass


class ChannelUiParameter(UiParameter, ABC, metaclass=UiParameterMeta):
    def __init__(self, tab, channel_name, name_widget_name=None):  # FIXME: not really a tab but a widget
        tab_widget = tab.channelsParamsTabWidget
        page, self.page_index = tab_widget.get_channel_widget(channel_name, return_idx=True)
        if page is None:
            raise ClearMapValueError(f'Channel {channel_name} not found in tab for {self.__class__.__name__}. '
                                     f'Please add it first.')
        if name_widget_name is not None:
            self.nameWidget = getattr(page, name_widget_name)
            if page is None:
                raise ValueError(f'Channel {channel_name} not found in tab')
            self.nameWidget.setText(channel_name)  # WARNING: needs to come first since used by all the rest
            # self.name = channel_name  # WARNING: needs to come first since used by all the rest
        else:
            self.nameWidget = tab_widget
        super().__init__(page)  # TEST: check if works
        self._cached_name = channel_name

    @property
    @abstractmethod
    def cfg_subtree(self):
        pass

    @cfg_subtree.setter
    def cfg_subtree(self, value):
        warnings.warn(f'cfg_subtree should not be set for ChannelParameters. '
                      f'Trying to set it to {value}')

    @property
    def name(self):
        if isinstance(self.nameWidget, QLineEdit):
            return self.nameWidget.text()
        elif isinstance(self.nameWidget, ExtendableTabWidget):
            return self.nameWidget.tabText(self.page_index)

    @name.setter
    def name(self, value):
        if value == self.name:
            return
        if isinstance(self.nameWidget, QLineEdit):
            self.nameWidget.setText(value)
        elif isinstance(self.nameWidget, ExtendableTabWidget):
            self.nameWidget.setTabText(self.page_index, value)

    @abstractmethod
    def handle_name_changed(self):
        pass


class UiParameterCollection(QObject):
    """
    For multi-section UiParameters that share the same config file.
    This ensures the file remains consistent.
    Although the attributes of contained `UiParameters` are accessible,
    it is recommended to first dereference them to avoid clashes.
    """
    channelsChanged = pyqtSignal(list, list)  # FIXME: should be in ChannelsUiParameterCollection
    def __init__(self, tab, parent=None, *args, **kwargs):
        if parent is None:
            parent = tab
        super().__init__(parent, *args, **kwargs)
        self.tab = tab
        self.config = None
        self._default_config = None

    def pop(self, key):
        if not self.channels:
            warnings.warn(f'Could not pop "{key}" from empty channels')
        channels_before = self.channels
        if key not in self.channels:
            warnings.warn(f'Could not remove channel "{key}" not found in channels')
            return
        self.config['channels'].pop(key)
        popped_channel = self.channel_params.pop(key)
        tab = self.tab.channelsParamsTabWidget
        tab.remove_channel_widget(key)
        channels_after = self.channels
        self.write_config()
        self.channelsChanged.emit(channels_before, channels_after)
        return popped_channel

    def fix_cfg_file(self, f_path):
        """Fix the file if it was copied from defaults, tailor to current sample"""
        pass

    @property
    def version(self):
        return Version(self.config['clearmap_version'])

    @property
    def params(self) -> list[UiParameter]:
        raise NotImplementedError('Please subclass UiParameterCollection and implement params property')

    def read_configs(self, cfg_path='', cfg=None):
        if not cfg_path and not cfg:
            raise ValueError('Either cfg_path or cfg must be provided')
        if cfg:
            self.config = cfg
            cfg_path = Path(cfg.filename)
        else:
            cfg_path = Path(cfg_path)
            self.config = ConfigLoader.get_cfg_from_path(cfg_path)
            if not self.config:
                raise ConfigNotFoundError(f'Config file not found: {cfg_path}')

            if self.version < CLEARMAP_VERSION:
                convert_config_versions.convert(cfg_path, backup=True, overwrite=True)

            self.config.reload()

        cfg_name = ConfigLoader.strip_version_suffix(cfg_path.stem)
        try:
            default_path = Path(ConfigLoader.get_default_path(cfg_name))
        except FileNotFoundError:
            from ClearMap.config import config_loader
            warnings.warn(f'No file found for {cfg_name} in clearmap configuration directory,'
                          f' using package default instead (from "{Path(config_loader.__file__).parent.absolute()}").'
                          f'Please consider copying the default file to the clearmap configuration directory.')
            default_path = Path(ConfigLoader.get_default_path(cfg_name, from_package=True))
        if not default_path.exists():
            raise ConfigNotFoundError(f'Default config file not found: {default_path}')
        self._default_config = ConfigLoader.get_cfg_from_path(default_path)
        for param in self.params:  # FIXME: ensure that we do that when adding channels + add here if any found
            if isinstance(param, UiParameterCollection):
                param.read_configs(cfg=self.config)
            elif isinstance(param, UiParameter):
                param._config = self.config
                param._default_config = self._default_config
            elif param is None:
                print(f'Skipping None param in {self.__class__.__name__}')
            else:
                raise NotImplementedError(f'Unknown param type {type(param)}')

    @property
    def config_path(self):
        return self.config.filename

    def write(self):
        self.write_config()

    def write_config(self):
        self.config.write()

    def reload(self):
        self.config.reload()

    def ui_to_cfg(self):
        self.write_config()

    def cfg_to_ui(self):
        for param in self.params:
            if param:
                param.cfg_to_ui()
            else:
                print(f'Skipping None param in {self.__class__.__name__}')


class ChannelsUiParameterCollection(UiParameterCollection):
    def __init__(self, tab):
        super().__init__(tab)
        self._channels = {}

    @property
    def channels(self):
        return list(self._channels.keys())

    @property
    def channel_params(self):  # FIXME: just to match SampleParameters
        return self._channels

    def __getitem__(self, item):
        return self._channels[item]

    def __setitem__(self, key, value):
        if not isinstance(value, (ChannelUiParameter, UiParameterCollection)):  # WARNING: collection of collections
            raise ClearMapValueError(f'Value must be a ChannelUiParameter. Got "{type(value)}" instead.')
        self._channels[key] = value

    def __contains__(self, item):
        return item in self._channels

    def __iter__(self):
        return iter(self._channels)

    def keys(self):
        return self._channels.keys()

    def values(self):
        return self._channels.values()

    def items(self):
        return self._channels.items()
