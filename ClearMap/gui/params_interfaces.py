import ast
import functools
import traceback
import warnings
from abc import ABC, abstractmethod
from typing import List, Any, Callable, Mapping, Optional, Dict, Type, Tuple, Sequence, Union

from importlib_metadata import version
from packaging.version import Version

from PyQt5 import sip
from PyQt5.QtWidgets import (QCheckBox, QLabel, QLineEdit, QSpinBox, QFrame,
                             QComboBox, QPlainTextEdit, QTextEdit, QGroupBox,
                             QWidget, QListWidget, QDoubleSpinBox)

from ClearMap.IO.assets_constants import CONTENT_TYPE_TO_PIPELINE
from ClearMap.Utils.event_bus import BusSubscriberMixin, EventBus, Publishes
from ClearMap.Utils.events import ChannelsChanged
from ClearMap.Utils.utilities import set_item_recursive, get_item_recursive, DELETE
from ClearMap.Utils.exceptions import ConfigNotFoundError, ClearMapValueError
from ClearMap.config.config_handler import ALTERNATIVES_REG
from ClearMap.gui.gui_utils_base import disconnect_widget_signal
from ClearMap.gui.widgets import ExtendableTabWidget, FileDropListWidget, LandmarksWeightsPanel, GroupsWidgetAdapter, \
    NProcessesWidget

CLEARMAP_VERSION = Version(version('ClearMap'))
DEBUG_PAINT_GUARD = True   # FIXME: base on machine params log_level

_UNSET = object()  # sentinel singleton


def identity_op(x):
    return x

def invert(x):
    return not bool(x)


def _is_alive(widget):
    try:
        return (widget is not None) and not sip.isdeleted(widget)
    except Exception:
        return widget is not None  # FIXME: what about ParamLinks with None widgets


# REFACTOR: merge with VectorLink version
def _ensure_sentinel_min_for_scalar(widget: QWidget, sentinel, label: Optional[str]):
    if isinstance(widget, QDoubleSpinBox) or isinstance(widget, QSpinBox):
        if widget.minimum() > sentinel:
            widget.setMinimum(sentinel)
        if sentinel == -1 and label is not None:
            try:
                widget.setSpecialValueText(label)
            except ValueError:
                warnings.warn(f"Sentinel label could not be applied to {widget.objectName()}")


class ParamLink:
    """
    A class to link a widget to a config file. This is the object passed in
    the ``params_dict`` attribute of the ``UiParameter`` class.
    It is used in the __getattr__ and __setattr__ and __connect_widget
    methods to link the widget to the config file.

    .. warning::
        extra_connect needs to return the disconnector function if required.

    Attributes
    ----------
    keys: list(str) (optional)
        The list of keys to access the value in the config file
        If None, the attribute is not present in the config file and
        will not be connected
    _widget: QWidget
        The GUI widget to be connected to the config file
    object_name: str

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
                 object_name: Optional[str] = None,
                 scope_root: Optional[QWidget] = None,
                 default: Optional[Any] = None,
                 connect: bool = True,
                 cast_to_ui: Optional[Callable[[Any], Any]] = identity_op,
                 cast_from_ui: Optional[Callable[[Any], Any]] = identity_op,
                 notify_apply: Optional[Callable[[], None]] = None,
                 extra_connect: Optional[Callable[[QWidget, Callable[[], None]], Callable | None]] = None,
                 missing_ok: bool = False,
                 present_if: Optional[Callable[[dict], bool]] = None,
                 disabled_value: Any =None, ui_sentinel: Any =None, enforce_sentinel_min=False,):
        if keys is None:
            connect = False
        self.keys: List[str] = keys
        self._widget: QWidget = widget
        self.object_name: Optional[str] = object_name  # Name of the widget to find inside scope_root
        self.scope_root: Optional[QWidget] = scope_root  # Where to find the widget by name
        self.default: Optional[Any] = default
        self.connect: bool = connect

        self.cast_to_ui: Optional[Callable[[Any], Any]] = cast_to_ui
        self.cast_from_ui: Optional[Callable[[Any], Any]] = cast_from_ui

        self.notify_apply = notify_apply
        self.extra_connect = extra_connect

        self.missing_ok: bool = missing_ok or bool(default)
        self.present_if: Optional[Callable[[dict], bool]] = present_if

        self._disconnectors: List[Callable[[], None]] = []

        self.disabled_value = disabled_value
        self.ui_sentinel = ui_sentinel
        self.enforce_sentinel_min = enforce_sentinel_min

        # If the UI sentinel is numeric and == -1, ensure spinboxes can display it nicely
        if self.enforce_sentinel_min and self.ui_sentinel == -1:
            _ensure_sentinel_min_for_scalar(widget, -1, label=str(self.disabled_value))


        if self.ui_sentinel is not None:
            # Wrap casts with sentinel-aware adapters (exactly like VectorLink)
            def _final_cast_to_ui(cfg_val: Any):
                # cfg disabled_value -> ui_sentinel
                if self.ui_sentinel is not None and _equals_disabled_value(cfg_val, self.disabled_value):
                    return self.ui_sentinel
                return cast_to_ui(cfg_val)

            def _final_cast_from_ui(ui_val: Any):
                # ui_sentinel -> cfg disabled_value
                if self.ui_sentinel is not None and ui_val == self.ui_sentinel:
                    return self.disabled_value
                return cast_from_ui(ui_val)

            self.cast_to_ui = _final_cast_to_ui
            self.cast_from_ui = _final_cast_from_ui

    def is_relevant(self, view):
        return True if self.present_if is None  else bool(self.present_if(view))

    @property
    def widget(self):
        w = self._widget
        if _is_alive(w):
            return w

        if self.scope_root is None or not self.object_name:
            return None

        # stale → re-find inside scope_root
        resolved = self.scope_root.findChild(QWidget, self.object_name)
        self._widget = resolved  # cache for next time
        return resolved

    @widget.setter
    def widget(self, value: QWidget):  # to set explicitly (especially in child clkass)
        self._widget = value

    def has_connect_function(self):
        return callable(self.connect)

    def add_disconnector(self, fn: Callable[[], None]):
        if callable(fn):
            self._disconnectors.append(fn)

    def disconnect_all(self):
        # run in reverse registration order, just in case dependencies exist
        for fn in reversed(self._disconnectors):
            try:
                fn()
            except Exception:
                pass
        self._disconnectors.clear()


def param_setter(func):
    """
    For UiParameter setters that touch widgets.
    Brackets the setter with self._painting = True/False.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        old = getattr(self, "_painting", False)
        self._painting = True
        try:
            return func(self, *args, **kwargs)
        finally:
            self._painting = old
    return wrapper


def param_handler(func):
    """
    For handlers/slots that write back to config.
    If we're painting, warn (in debug) and skip.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_painting", False):
            if DEBUG_PAINT_GUARD:
                warnings.warn(
                    f"{self.__class__.__name__}.{func.__name__} fired while painting. "
                    f"This will be ignored, but it's probably a bug.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return
        return func(self, *args, **kwargs)
    return wrapper

# ################ For compound widgets wrapped in a QFrame #####################
def frame_getter(widget: QFrame):
    """
    For compound objects wrapped in a frame like
    singlets, doublets and triplets, with or without a checkbox
    or for text edits with an enable checkbox
    """
    if hasattr(widget, 'getValue'):   # singlets, doublets and triplets
        return widget.getValue()
    elif hasattr(widget, 'text') and callable(widget.text): # TextEdit with checkbox
        return widget.text()
    else:
        raise ValueError(f'Unrecognised frame with name "{widget.objectName()}"')


def frame_setter(widget: QFrame, value):
    """
    For compound objects wrapped in a frame like
    singlets, doublets and triplets, with or without a checkbox
    or for text edits with an enable checkbox
    """
    if hasattr(widget, 'setValue'):  # singlets, doublets and triplets
        widget.setValue(value)
    elif hasattr(widget, 'setText') and callable(widget.setText):  # TextEdit with checkbox
        widget.setText(value)
    else:
        raise ValueError(f'Unrecognised frame with name "{widget.objectName()}"')


def connect_frame(widget, callback):
    name = widget.objectName()
    if hasattr(widget, 'valueChangedConnect'):  # singlets, doublets and triplets
        widget.valueChangedConnect(lambda *_: callback())
    elif hasattr(widget, 'textChangedConnect'):  # WARNING: plainTextEdit.textChanged is argument less
        widget.textChangedConnect(lambda *_: callback())
    else:
        raise ValueError(f'Unrecognised frame with "{name=}"')


def _ensure_list(v: Any, n: int) -> List[Any]:
    if v is None:
        return [None] * n
    if isinstance(v, (list, tuple)):
        L = list(v)
        if len(L) > n:
            return L[:n]  # silently truncate if too long
            # raise ValueError(f"Expected at most {n} elements, got {len(L)}")
        else:
            return L + [None] * (n - len(L))  # pad if too short
    return [v]


def _ensure_sentinel_min(widget: QFrame, sentinel: Union[int, float], label: Optional[str] = "auto"):
    # ensure each child spinbox can display the sentinel; decorate with specialValueText when sentinel == -1
    for sb in list(widget.findChildren(QSpinBox)) + list(widget.findChildren(QDoubleSpinBox)):
        if sb.minimum() > sentinel:
            sb.setMinimum(sentinel)
        if sentinel == -1 and label is not None:
            try:
                sb.setSpecialValueText(label)
            except ValueError:
                warnings.warn(f'Sentinel display label could not be enforced for {widget.objectName()}')
                pass


def _equals_disabled_value(val, token):
    return (val is None or val == 'auto') if (token is None) else (val == token)


# def _disabled_to_sentinel(x: Any, token: Any, sentinel: Union[int, float]) -> Any:
#     # treat both the exact token and None as “disabled element”
#     return sentinel if (x is None or _equals_disabled_value(x, token)) else x
#
#
# def _sentinel_to_disabled(x: Any, token: Any, sentinel: Union[int, float]) -> Any:
#     return token if x == sentinel else x


class VectorLink(ParamLink):
    def __init__(self, keys: Sequence[str], widget: QFrame, *,
                disable_globally: bool | str = "auto",
                disabled_value: Optional[Union[str, None]] = None,
                cast_to_ui: Optional[Callable[[Any], Any]] = None,
                cast_from_ui: Optional[Callable[[Any], Any]] = None,
                collapse_singular: bool = True,
                enforce_sentinel_min: bool = True,
                n: Optional[int] = None,
                dtype: Callable[[Any], Any] | str = "auto",
                ui_sentinel: Union[int, float] = -1,
                notify_apply: Callable[[], None] | None = None,
                extra_connect: Callable[[QFrame, Callable[[], None]], None] | None = None,
                show_sentinel_when_off: bool = False):
        """
        ParamLink specialisation for numeric vectors with optional global disable.
        This is a **meta**widget that maps a vector of ints or bools to several spinboxes
        wrapped in a QFrame (Singlet, Doublet, Triplet) with optional checkbox.
        If a checkbox is present, it can disable the whole control (global toggle).

        Semantics:
          - If disable_globally (global toggle present/True):
              UI: None  <->  config: disabled_value  (None or "auto")
              UI: [values...]  <->  config: typed list/scalar
            (no per-element sentinel mapping is performed if globally disabled)
          - Else:
              UI always returns list/number; any element that equals 'sentinel' is cast to None in config,
              and vice versa (None/'auto' -> ui_sentinel) for display.

        Notes:
          - Arity is inferred from objectName suffix, unless 'n' is given.
          - dtype is inferred from child spinboxes unless specified.
          - To show "auto" text at -1, 'enforce_sentinel_min=True' will set minimum=-1 and specialValueText="auto".

        Parameters
        ----------
        keys: Sequence[str]
            The keys for the config
        widget: QFrame
            The QFrame widget containing the singlet/doublet/triplet
        disable_globally: bool | str
            Whether the widget has a global disable checkbox.
            If "auto" (default), it is inferred from the presence of a checkbox in the QFrame.
        disabled_value: Optional[Union[str, None]]
            The value to store in the config when the widget is disabled. (None or "auto", default: None)
        cast_to_ui: Optional[Callable[[Any], Any]]
            Optional function to cast the config value to the UI value.
        cast_from_ui: Optional[Callable[[Any], Any]]
            Optional function to cast the UI value to the config value.
        collapse_singular: bool
            If True (default) and n=1, the config value is a scalar instead of a single-element list.
        enforce_sentinel_min: bool
            If True (default) and disable_globally is False, ensure each child spinbox can display the sentinel
            (minimum <= ui_sentinel, and specialValueText=disabled_value when value==sentinel).
        n: Optional[int]
            The number of elements (1, 2 or 3). If None (default), it is inferred from the objectName suffix.
        dtype: Callable[[Any], Any] | str
            The type to cast the values to (int or float). If "auto", it is inferred from child spinboxes.
        ui_sentinel: Union[int, float]
            The sentinel value to use for None or 'auto'.
        """
        self.keys = list(keys)
        self.widget = widget
        if disabled_value not in (None, "auto"):
            raise ValueError("disabled_value must be None or 'auto'")
        self.disable_globally = disable_globally
        self.disabled_value = disabled_value

        self.collapse_singular = collapse_singular
        self._n = n
        self._dtype = dtype
        self.ui_sentinel = ui_sentinel

        if enforce_sentinel_min and ui_sentinel == -1:
            _ensure_sentinel_min(widget, -1, label=str(self.disabled_value))

        if show_sentinel_when_off:
            widget.setProperty('showSentinelWhenOff', True)

        def final_cast_to_ui(cfg_val: Any):
            ui_val = self.cast_sentinel_to_ui(cfg_val)
            return cast_to_ui(ui_val) if cast_to_ui is not None else ui_val

        def final_cast_from_ui(ui_val: Any):
            cfg_val = self.cast_sentinel_from_ui(ui_val)
            return cast_from_ui(cfg_val) if cast_from_ui is not None else cfg_val

        # def connector(w, cb):
        #     if hasattr(w, 'valueChangedConnect'):
        #         w.valueChangedConnect(lambda *_: cb())
        #     else:
        #         # fallback: use WidgetOps registry
        #         try:
        #             w.valueChanged.connect(cb)
        #         except Exception:
        #             pass

        super().__init__(keys=self.keys, widget=widget,
                         cast_to_ui=final_cast_to_ui, cast_from_ui=final_cast_from_ui,
                         connect=True, notify_apply=notify_apply, extra_connect=extra_connect,
                         disabled_value=disabled_value, ui_sentinel=ui_sentinel)

    def _has_global_toggle(self) -> bool:
        get_cb = getattr(self.widget, "getCheckBox", None)
        return callable(get_cb) and (get_cb() is not None)

    def _detect_numeric_dtype(self) -> Callable[[Any], Any]:
        """
        Detect the numeric type (int or float) from the child spinboxes.
        any DoubleSpinBox => float, else int
        """
        return float if self.widget.findChildren(QDoubleSpinBox) else int

    @functools.cached_property
    def has_global_toggle(self):
        return self._has_global_toggle() if self.disable_globally == "auto" else bool(self.disable_globally)

    def _infer_arity(self) -> int:
        name = self.widget.objectName() or ""
        if name.endswith("Singlet"):
            return 1
        elif name.endswith("Doublet"):
            return 2
        elif name.endswith("Triplet"):
            return 3
        else:  # fallback: count spinboxes
            n_spin_boxes = (len(self.widget.findChildren(QDoubleSpinBox)) +
                            len(self.widget.findChildren(QSpinBox)))
            return max(1, n_spin_boxes)

    @functools.cached_property
    def n_numbers(self):
        return self._infer_arity() if self._n is None else self._n

    @functools.cached_property
    def dtype(self):
        return self._detect_numeric_dtype() if self._dtype == "auto" else self._dtype

    def ensure_list(self, val):
        return _ensure_list(val, self.n_numbers)

    def equals_disabled_value(self, val):
        return _equals_disabled_value(val, self.disabled_value)

    def cast_sentinel_to_ui(self, cfg_val: Any):
        # whole-control disabled
        if self.has_global_toggle and self.equals_disabled_value(cfg_val):
            return None  # UI is unaware of disabled value (always None)

        vals = self.ensure_list(cfg_val)
        out = []
        for x in vals:
            if self.equals_disabled_value(x):
                out.append(self.ui_sentinel)
            else:
                out.append(self.dtype(x))

        return out[0] if (self.n_numbers == 1 and self.collapse_singular) else out

    def cast_sentinel_from_ui(self, ui_val: Any):
        # whole-control disabled
        if self.has_global_toggle and ui_val is None:
            return self.disabled_value

        vals = self.ensure_list(ui_val)
        out = []
        for x in vals:
            if x == self.ui_sentinel or self.equals_disabled_value(x):
                out.append(self.disabled_value)
            else:
                out.append(self.dtype(x))
        return out[0] if (self.n_numbers == 1 and self.collapse_singular) else out


def optional_text_link(keys: Sequence[str], widget: QFrame,
                       disabled_value: Optional[Union[str, None]] = None) -> ParamLink:
    """
    Create a ParamLink for an optional text field.
    This is a QFrame widgets that contain a text editable (QLineEdit or QPlainTextEdit)
    and an enable/disable checkbox.

    Parameters
    ----------
    keys: List[str]
        The keys for the config
    widget: QFrame
        The QFrame widget containing the text edit and checkbox
    disabled_value: Optional[Union[str, None]]
        The value to be used when the widget is disabled.
        If None, the config value will be None when disabled.
        If "auto", the config value will be "auto" when disabled.
        Default is None.

    Returns
    -------
    ParamLink
        The ParamLink object to be used in the params_dict
    """
    def cast_to_ui(cfg_val: Any):
        if (disabled_value is None and cfg_val is None) or (disabled_value == "auto" and cfg_val == "auto"):
            return None
        return "" if cfg_val is None else str(cfg_val)

    def cast_from_ui(ui_val: Any):
        if ui_val is None:
            return disabled_value
        return None if ui_val == "" else str(ui_val)

    return ParamLink(keys=list(keys), widget=widget, cast_to_ui=cast_to_ui, cast_from_ui=cast_from_ui,
                     connect=lambda w, cb: w.textChangedConnect(cb))



def combobox_getter(widget: QComboBox):
    txt = widget.currentText()
    if txt in ('None', ''):
        return None
    return txt


def combobox_setter(widget: QComboBox, value):
    if value is None:
        value = 'None'  # TODO: decide 'None' or ''
    found = widget.findText(value)
    if found == -1 and widget.count() > 0:
        warnings.warn(f'Value "{value}" not found in combobox, possible values: ' 
                      f'[{", ".join([widget.itemText(i) for i in range(widget.count())])}]')
    widget.setCurrentText(value)


def line_edit_getter(widget: QLineEdit):
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


def list_widget_getter(widget: QListWidget):
    widgets = [widget.item(i) for i in range(widget.count())]
    return [itm.text() for itm in widgets]


def list_widget_setter(widget: QListWidget, itm_list: List[str]):
    try:
        widget.clear()
        widget.addItems(itm_list)
    except RuntimeError:  # Widget has been deleted
        print(f'Widget {widget} has been deleted')
        pass


Getter     = Callable[[QWidget], Any]
Setter     = Callable[[QWidget, Any], None]
Connector  = Callable[[QWidget, Callable[[], None]], None]
OpsTuple   = Tuple[Optional[Getter], Optional[Setter], Optional[Connector]]


class WidgetOps:
    """Registry + cache for widget getters/setters/connectors."""
    def __init__(self) -> None:
        self._getters: Dict[Type[QWidget], Getter] = {}
        self._setters: Dict[Type[QWidget], Setter] = {}
        self._connectors: Dict[Type[QWidget], Connector] = {}
        self._cache: Dict[Type[QWidget], OpsTuple] = {}
        self._install_defaults()

    def register(self, cls: Type[QWidget], *, getter: Optional[Getter] = None,
                 setter: Optional[Setter] = None, connector: Optional[Connector] = None) -> None:
        """
        Register operations for a given widget class.
        If an operation is None, it will not be registered.
        This will clear the cache which will be rebuilt on demand.

        Parameters
        ----------
        cls: Type[QWidget]
            The widget class to register
        getter: Getter
            The getter function for the widget class
        setter: Setter
            The setter function for the widget class
        connector: Connector
            The connector function for the widget class
        """
        if getter:    self._getters[cls]   = getter
        if setter:    self._setters[cls]   = setter
        if connector: self._connectors[cls]= connector
        self._cache.clear()

    def _resolve_ops(self, cls: Type[QWidget]) -> OpsTuple:
        """
        Resolve the operations for a given widget class, using MRO to find
        the most specific registered operations.
        +
        Cache the results for faster lookup next time.

        Parameters
        ----------
        cls: Type[QWidget]
            The widget class to resolve the operations for

        Returns
        -------
        OpsTuple
            A tuple of (getter, setter, connector) functions or None if not found
        """
        if cls in self._cache:
            return self._cache[cls]
        g = s = k = None
        # Walk MRO: most specific class wins
        for c in cls.mro():
            if g is None and c in self._getters:    g = self._getters[c]
            if s is None and c in self._setters:    s = self._setters[c]
            if k is None and c in self._connectors: k = self._connectors[c]
            if g and s and k: break
        ops = (g, s, k)
        self._cache[cls] = ops
        return ops

    def resolve(self, widget: QWidget) -> OpsTuple:
        """
        Resolve the operations for a given widget instance.
        Uses the class of the widget to lookup the operations.
        This will also use MRO to find the most specific registered operations.
        Caches the results for faster lookup next time.
        
        Parameters
        ----------
        widget: QWidget
            The widget instance to resolve the operations for

        Returns
        -------
        OpsTuple
            A tuple of (getter, setter, connector) functions or None if not found
        """
        return self._resolve_ops(widget.__class__)

    def get(self, widget: QWidget) -> Any:
        getter_fn, _, _ = self.resolve(widget)
        if getter_fn is None:
            raise NotImplementedError(f"No getter for {type(widget).__name__}")
        return getter_fn(widget)

    def set(self, widget: QWidget, value: Any, *, silent: bool = False) -> None:
        _, setter_fn, _ = self.resolve(widget)
        if setter_fn is None:
            raise NotImplementedError(f"No setter for {type(widget).__name__}")
        if silent and hasattr(widget, "blockSignals"):
            old = widget.blockSignals(True)  # type: ignore[attr-defined]
            try:
                setter_fn(widget, value)
            except:
                print(f"Error setting {value=} on widget {widget.objectName()}:")
                raise
            finally:
                widget.blockSignals(old)     # type: ignore[attr-defined]
        else:
            setter_fn(widget, value)

    def connect(self, widget: QWidget, cb: Callable[[], None]) -> Optional[Callable[[], None]]:
        _, _, connector = self.resolve(widget)
        if connector is None:  # For e.g. QLabel
            warnings.warn(f'No connector for {type(widget).__name__}, skipping')
            return None
        else:
            return connector(widget, cb)

    def _install_defaults(self) -> None:
        def check_box_connector(w: QCheckBox, cb: Callable[[], None]):
            w.stateChanged.connect(cb)
            return lambda: disconnect_widget_signal(w.stateChanged, slot=cb)

        self.register(QCheckBox,
            getter=lambda w: w.isChecked(),
            setter=lambda w, v: w.setChecked(bool(v)),
            connector=check_box_connector,  # TODO: Check if not setChecked
        )
        self.register(QLabel,
            getter=lambda w: w.text(),
            setter=lambda w, v: w.setText(v),
            connector=None)  # FIXME: QLabel does not have textChanged signal

        def text_changed_connector(w: QLineEdit, cb: Callable[[], None]):
            w.textChanged.connect(cb)
            return lambda: disconnect_widget_signal(w.textChanged, slot=cb)

        def editing_finished_connector(w: QLineEdit, cb: Callable[[], None]):
            w.editingFinished.connect(cb)
            return lambda: disconnect_widget_signal(w.editingFinished, slot=cb)

        self.register(QLineEdit,
            getter=line_edit_getter,
            setter=lambda w, v: w.setText(str(v)),
            connector=editing_finished_connector,
        )

        def value_changed_connector(w: QSpinBox, cb: Callable[[], None]):
            w.setKeyboardTracking(False)
            w.valueChanged.connect(cb)
            return lambda: disconnect_widget_signal(w.valueChanged, slot=cb)

        self.register(QSpinBox,
            getter=lambda w: w.value(),
            setter=lambda w, v: w.setValue(int(v)),
            connector=value_changed_connector,
        )
        self.register(QDoubleSpinBox,
            getter=lambda w: w.value(),
            setter=lambda w, v: w.setValue(float(v)),
            connector=value_changed_connector,
        )
        self.register(QPlainTextEdit,
            getter=lambda w: w.toPlainText(),
            setter=lambda w, v: w.setPlainText("" if v is None else str(v)),
            connector=text_changed_connector,
        )
        self.register(QTextEdit,  # FIXME: check if HTML for both getter and setter is what we want
            getter=lambda w: w.toHtml(),  # TODO: check if this always what we want
            setter=lambda w, v: w.setPlainText("" if v is None else str(v)),
            connector=editing_finished_connector,
        )

        def current_text_changed_connector(w: QComboBox, cb: Callable[[], None]):
            w.currentTextChanged.connect(cb)
            return lambda: disconnect_widget_signal(w.currentTextChanged, slot=cb)

        self.register(QComboBox,
            getter=combobox_getter,
            setter=combobox_setter,
            connector=current_text_changed_connector,
        )

        def toggled_connector(w: QGroupBox, cb: Callable[[], None]):
            w.toggled.connect(cb)
            return lambda: disconnect_widget_signal(w.toggled, slot=cb)

        self.register(QGroupBox,
            getter=lambda w: w.isChecked(),
            setter=lambda w, v: w.setChecked(bool(v)),
            connector=toggled_connector,
        )

        def item_changed_connector(w: QListWidget, cb: Callable[[], None]):
            w.itemChanged.connect(cb)
            return lambda: disconnect_widget_signal(w.itemChanged, slot=cb)

        self.register(QListWidget,  # FIXME: not the right signal (should be for any change). + also rows inserted/removed
            getter=list_widget_getter,
            setter=list_widget_setter,
            connector=item_changed_connector,
        )

        def items_changed_connector(w: FileDropListWidget, cb: Callable[[], None]):
            w.itemsChanged.connect(cb)
            return lambda: disconnect_widget_signal(w.itemsChanged, slot=cb)

        self.register(FileDropListWidget,
            getter=list_widget_getter,
            setter=list_widget_setter,
            connector=items_changed_connector,
        )
        self.register(QFrame,
            getter=frame_getter,
            setter=frame_setter,
            connector=connect_frame
        )


WIDGET_OPS = WidgetOps()
WIDGET_OPS.register(  # Not strictly necessary, but more explicit
    LandmarksWeightsPanel,
    getter=lambda w: w.get_weights(),
    setter=lambda w, v: w.set_weights(v or []),
    connector=lambda w, cb: w.valueChangedConnect(cb),
)
WIDGET_OPS.register(
    GroupsWidgetAdapter,
    getter=lambda w: w.get_value(),
    setter=lambda w, v: w.set_value(v or {}),
    connector=lambda w, cb: w.connect(cb),
)


def n_proc_getter(widget: NProcessesWidget):
    val = widget.value()
    return val


def n_proc_setter(widget: NProcessesWidget, value):
    widget.setValue(int(value))


def n_proc_connector(widget: NProcessesWidget, cb):
    def _slot(_val: int):
        cb()
    widget.valueChanged.connect(_slot)
    return lambda: disconnect_widget_signal(widget.valueChanged, slot=_slot)


WIDGET_OPS.register(NProcessesWidget, getter=n_proc_getter,
                    setter=n_proc_setter, connector=n_proc_connector)

class UiParameter(BusSubscriberMixin):
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
    params_dict: Dict[str, ParamLink | List[str]
        The dictionary of attributes and their corresponding ParamLink
    _cfg_subtree: List[str]
        The list of keys to access the config file. This allows a relative path to be used from the
        whole config file.
    """
    def __init__(self, tab: QWidget, *, event_bus: EventBus,
                 get_view: Optional[Callable[[], Mapping[str, Any]]] = None,
                 apply_patch: Optional[Callable[[dict], None]] = None):
        super().__init__(event_bus)
        if tab is None:
            raise ValueError('Tab widget cannot be None')
        self.tab: QWidget = tab

        self.advanced_controls: List[QWidget] = []

        self._get_view: Callable[[], Mapping[str, Any]] | None = get_view
        self._apply_patch: Callable[[dict], None] | None = apply_patch
        self._cfg_subtree = getattr(self, "_cfg_subtree", [])  # FIXME: pass as arg to ctor?
        self._painting = False  # True when updating the UI from the config file
        self.widget_ops = WIDGET_OPS

        self._connected_param_keys: set[str] = set()
        self.params_dict = self.build_params_dict()
        self._validate_params_dict()
        self.connect()  # User overridable hook
        self.connect_simple_widgets()
        self.post_connect()

    def build_params_dict(self) -> dict:
        raise NotImplementedError

    def connect(self):
        pass

    def post_connect(self):
        pass

    def _validate_params_dict(self):
        for k, v in self.params_dict.items():
            if isinstance(v, ParamLink) and v.widget is None:
                raise RuntimeError(f"{self.__class__.__name__}: ParamLink {k} has no widget")

    def teardown(self):
        for link in self.params_dict.values():
            if isinstance(link, ParamLink):
                link.disconnect_all()
                # Drop widget refs
                try:
                    link.widget = None
                except Exception:
                    pass

    def _update_value(self, keys, value):
        if self._apply_patch is None:
            raise ValueError('Patch sink not set. Please call bind_apply_patch first.')
        if self._painting and DEBUG_PAINT_GUARD:
            warnings.warn(
                f"{self.__class__.__name__}._update_value called while painting "
                f"for keys={keys!r}. This is probably a cfg→UI→cfg loop.\n"
                f"Stack:\n{''.join(traceback.format_stack(limit=10))}",
                RuntimeWarning,
            )
        patch = {}
        set_item_recursive(patch, self.cfg_subtree + keys, value)
        self._apply_patch(patch)

    @property
    def cfg_subtree(self) -> List[str]:  # WARNING: needs to be dynamic since channels can change
        return self._cfg_subtree

    @cfg_subtree.setter
    def cfg_subtree(self, value: List[str]):
        self._cfg_subtree = value

    def is_simple_attr(self, key):
        if key == 'params_dict' or not hasattr(self, 'params_dict') or self.params_dict is None:
            return False
        if not key in self.params_dict.keys():  # Not set through GUI
            return False
        attr = self.params_dict[key]
        is_simple = isinstance(attr, ParamLink)
        return is_simple

    def bind_apply_patch(self, fn: Callable[[dict], None]) -> None:
        """Provide a sink to apply a patch (already scoped to the section)."""
        self._apply_patch = fn

    def bind_view_provider(self, fn: Callable[[], Mapping[str, Any]]) -> None:
        """Provide a readonly view of the current section (fresh each call)."""
        self._get_view = fn

    @property
    def view(self):
        if self._get_view is None:
            raise ValueError('View provider not set. Please call bind_view_provider first.')
        base = self._get_view()
        return get_item_recursive(base, self.cfg_subtree) if self.cfg_subtree else base

    def _emit_patch(self, relative_path, value) -> None:
        full_path = list(self.cfg_subtree) + list(relative_path)
        patch = {}
        set_item_recursive(patch, full_path, value)
        if self._apply_patch is not None:
            self._apply_patch(patch)

    def __getattr__(self, item):  # FiXME: use binder.default
        if self.is_simple_attr(item):
            binder = self.params_dict[item]
            widget = binder.widget
            val = self.widget_ops.get(widget)
            return binder.cast_from_ui(val)
        else:
            raise AttributeError(f'{self.__class__.__name__}, unknown attribute "{item}"')

    def __setattr__(self, key, value):
        if self.is_simple_attr(key):
            p_link = self.params_dict[key]
            widget = p_link.widget
            value = p_link.cast_to_ui(value)
            self.widget_ops.set(widget, value, silent=self._painting)
        else:
            object.__setattr__(self, key, value)

    def get(self, item, default_value=None):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            return default_value

    def connect_simple_widgets(self):
        for k in self.params_dict.keys():
            link = self.params_dict[k]
            if (isinstance(link, ParamLink) and link.has_connect_function()) or \
                (self.is_simple_attr(k) and link.connect and not hasattr(self, f'handle_{k}_changed')):
                self.__connect_widget(k)

    def __connect_widget(self, key):
        if key in self._connected_param_keys:
            return  # ensure idempotency
        param_link = self.params_dict[key]
        widget = param_link.widget
        callback = param_link.connect
        if not callable(callback):
            callback = functools.partial(self.handle_widget_changed, attr_name=key)
        disconnector = self.widget_ops.connect(widget, callback)
        param_link.add_disconnector(disconnector)
        if callable(getattr(param_link, 'extra_connect', None)):
            extra_disconnector = param_link.extra_connect(widget, callback)
            param_link.add_disconnector(extra_disconnector)

        self._connected_param_keys.add(key)

    def extend_params_dict(self, extra_param_links: dict[str, ParamLink | list], *, connect_new: bool = True):
        """
        Extend the params_dict with new entries after creation (and potenitally after initaial connexion)

        Parameters
        ----------
        extra_param_links: dict[str, ParamLink | list]
            The new entries to add to the params_dict
        connect_new:

        Returns
        -------

        """
        self.params_dict.update(extra_param_links)
        if connect_new:
            self.connect_simple_widgets()  # (idempotent)

    def handle_widget_changed(self, *_, attr_name='', **__):
        """
        Generic handler for simple widgets that binds them to the config file.
        Calls _emit_patch with the keys obtained from the ParamLink
        with **attr_name** and the current value of the attribute.
        Discards the values of the signal passed by Qt.

        Parameters
        ----------
        attr_name: str
            The name of the attribute in the class
        """
        if self._painting:
            return
        param_link = self.params_dict[attr_name]
        keys = param_link.keys
        if keys is None:  # For display only widgets (e.g. plotting temp settings)
            return
        if not param_link.is_relevant(self.view):
            return
        property_value = getattr(self, attr_name)  # Cast happens here, don't do it twice
        self._emit_patch(keys, property_value)
        if callable(param_link.notify_apply):
            param_link.notify_apply()

    def connect(self):
        """Connect GUI slots here"""
        pass

    def cfg_to_ui(self):
        if not self.params_dict:
            raise NotImplementedError('params_dict not set')

        self._painting = True
        view = attr = keys_list = _UNSET         # Pre-initialize to avoid referencing before initialisation in err
        try:
            view = self.view
            for attr, p_link in self.params_dict.items():
                keys_list = p_link.keys if isinstance(p_link, ParamLink) else p_link
                if keys_list is None:  # For params without cfg
                    continue
                if isinstance(p_link, ParamLink):
                    widget = p_link.widget
                    if widget is None:
                        continue  # TODO: add warning?
                    relevant = p_link.is_relevant(view if isinstance(view, dict) else view.config)
                    # hide when not applicable
                    if not relevant and hasattr(widget, "setVisible"):
                        widget.setVisible(False)
                    if not relevant:
                        continue
                try:
                    val = get_item_recursive(view, keys_list)  # check if we want try_get_item here w/ default
                except (KeyError, ConfigNotFoundError):
                    if isinstance(p_link, ParamLink) and p_link.missing_ok:
                        val = p_link.default
                    else:
                        raise
                # Update the UI
                setattr(self, attr, val)  # comes after the cfg otherwise, key will be missing in the callback
        except Exception as e:
            parts = [f"Error in cfg_to_ui for {self.__class__.__name__}"]
            # Build context defensively — each piece may or may not exist
            for label, obj in [("attr", attr), ("keys", keys_list), ("view", view)]:
                if obj is _UNSET:
                    parts.append(f"{label}=<not yet assigned>")
                else:
                    try:
                        parts.append(f"{label}={obj!r}")
                    except Exception:
                        parts.append(f"{label}=<repr failed: {type(obj).__name__}>")
            print(" | ".join(parts + [f'exception={e!r}']))
            raise
        finally:
            self._painting = False

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

    def str_to_lower(self, val):
        return "" if val is None else str(val).lower()

    def str_to_capitalize(self, val):
        return "" if val is None else str(val).capitalize()

    def sanitize_path_read(self, val):
        return "" if val in (None, 'None') else str(val)

    def sanitize_path_write(self, val):
        return None if val in (None, '', 'None') else str(val)


class ChannelUiParameter(UiParameter, ABC):
    """
    Base class for channel parameters. Each channel parameter is associated with a channel name
    and a widget in the tab.
    """
    def __init__(self, tab: QWidget, channel_name: str, *, event_bus: EventBus,
                 name_widget_name: Optional[str] = None,
                 get_view: Callable[[], Mapping[str, Any]] | None = None,
                 apply_patch: Callable[[dict], None] | None = None):
        tab_widget = tab.channelsParamsTabWidget
        page, self.page_index = tab_widget.get_channel_widget(channel_name, return_idx=True)
        if page is None:
            raise ClearMapValueError(f'Channel {channel_name} not found in tab for {self.__class__.__name__}. '
                                     f'Please add it first.')
        if name_widget_name is not None:
            self.nameWidget = getattr(page, name_widget_name)
            self.nameWidget.setText(channel_name)  # WARNING: needs to come first since used by all the rest
            # self.name = channel_name  # WARNING: needs to come first since used by all the rest
        else:
            self.nameWidget = tab_widget
        super().__init__(page, event_bus=event_bus, get_view=get_view, apply_patch=apply_patch)

    @property
    @abstractmethod
    def cfg_subtree(self) -> List[str]:
        pass

    @cfg_subtree.setter
    def cfg_subtree(self, value: List[str]):
        warnings.warn(f'cfg_subtree should not be set for ChannelParameters. '
                      f'Trying to set it to {value}')

    @property
    def name(self):
        if isinstance(self.nameWidget, QLineEdit):
            return self.nameWidget.text()
        elif isinstance(self.nameWidget, ExtendableTabWidget):
            return self.nameWidget.tabText(self.page_index)
        else:
            raise ValueError(f'Unsupported nameWidget type: {type(self.nameWidget)}')

    @name.setter
    def name(self, value):
        if value == self.name:
            return
        if isinstance(self.nameWidget, QLineEdit):
            self.nameWidget.setText(value)
        elif isinstance(self.nameWidget, ExtendableTabWidget):
            self.nameWidget.setTabText(self.page_index, value)


class UiParameterCollection(BusSubscriberMixin, ABC):
    """
    For multi-section UiParameters that share the same config file.
    This ensures the file remains consistent.
    Although the attributes of contained `UiParameters` are accessible,
    it is recommended to first dereference them to avoid clashes.
    """
    def __init__(self, tab: QWidget, *, pipeline_name: str, event_bus: EventBus,
                 get_view: Callable[[], Mapping[str, Any]] | None = None,
                 apply_patch: Callable[[dict], None] | None = None):
        super().__init__(event_bus)
        self.tab = tab
        self.pipeline_name = pipeline_name
        self._get_view: Callable[[], Mapping[str, Any]] | None = get_view
        self._apply_patch: Callable[[dict], None] | None = apply_patch
        # cfg_subtree is just the section name for single-pipeline collections
        self.cfg_subtree: List[str] = [ALTERNATIVES_REG.pipeline_to_section_name(self.pipeline_name)]
        self.advanced_controls: List[QWidget] = []

    def handle_advanced_state_changed(self, state):
        for ctrl in self.advanced_controls:
            ctrl.setVisible(state)

    def bind_apply_patch(self, fn: Callable[[dict], None]) -> None:
        """Provide a sink to apply a patch (already scoped to the section)."""
        self._apply_patch = fn

    def _update_value(self, keys, value):
        if self._apply_patch is None:
            raise ValueError('Patch sink not set. Please call bind_apply_patch first.')
        patch = {}
        set_item_recursive(patch, self.cfg_subtree + keys, value)
        self._apply_patch(patch)

    def bind_view_provider(self, fn: Callable[[], Mapping[str, Any]]) -> None:
        """Provide a readonly view of the current section (fresh each call)."""
        self._get_view = fn

    @property
    def view(self):
        if self._get_view is None:
            raise ValueError('View provider not set. Please call bind_view_provider first.')
        base = self._get_view()
        if self.cfg_subtree:
            return get_item_recursive(base, self.cfg_subtree)
        else:
            return base

    def _emit_patch(self, relative_path, value) -> None:
        full_path = list(self.cfg_subtree) + list(relative_path)
        patch = {}
        set_item_recursive(patch, full_path, value)
        if self._apply_patch is not None:
            self._apply_patch(patch)

    def pop(self, channel_name):  # FIXME: should be in ChannelsUiParameterCollection. Check calls
        if self._apply_patch is None:
            raise ValueError('Patch sink not set. Please call bind_apply_patch first.')
        if not self.channels:
            warnings.warn(f'Could not pop "{channel_name}" from empty channels')
        if channel_name not in self.channels:
            warnings.warn(f'Could not remove "{channel_name=}" not found in {self.channels=}')
            return None

        channels_before = self.channels[:]

        try:
            self._apply_patch({'channels': {channel_name: DELETE}})
        except Exception as e:
            warnings.warn(f'Could not remove "{channel_name=}" from config: {e}')
            return None

        popped_channel = self.channel_params.pop(channel_name)

        # Ensure all signals get disconnected and resources freed
        try:
            popped_channel.teardown()
        except Exception as e:
            warnings.warn(f'Error during teardown of channel "{channel_name}": {e}')

        tab = self.tab.channelsParamsTabWidget
        tab.remove_channel_widget(channel_name)

        channels_after = self.channels
        self.publish(ChannelsChanged(before=channels_before, after=channels_after))
        return popped_channel

    @property
    def version(self):
        return Version(self.view['clearmap_version'])

    @property
    @abstractmethod
    def params(self) -> list[UiParameter]:
        raise NotImplementedError('Please subclass UiParameterCollection and implement params property')

    def cfg_to_ui(self):
        for param in self.params:
            if param:
                param.cfg_to_ui()
            else:
                print(f'Skipping None param in {self.__class__.__name__}')

    def set_painting(self, painting: bool):
        if hasattr(self, "_painting"):
            self._painting = painting
        for param in self.params:
            param._painting = painting


class ChannelsUiParameterCollection(UiParameterCollection):
    publishes = Publishes(ChannelsChanged)

    def __init__(self, tab: QWidget, *, pipeline_name: str, event_bus: EventBus,
                 get_view: Callable[[], Mapping[str, Any]] | None = None,
                 apply_patch: Callable[[dict], None] | None = None):
        super().__init__(tab, pipeline_name=pipeline_name, event_bus=event_bus,
                         get_view=get_view, apply_patch=apply_patch)
        self._channels = {}

    @property
    def relevant_channels(self):
        return [c for c, v in self._get_view()['sample']['channels'].items() if
                v['data_type'] and CONTENT_TYPE_TO_PIPELINE[v['data_type']] == self.pipeline_name]
    @property
    def channels(self):
        return list(self._channels.keys())

    @property
    def channel_params(self):  # WARNING: this is mutable
        return self._channels

    def get(self, channel, default_value=None):
        return self._channels.get(channel, default_value)

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
