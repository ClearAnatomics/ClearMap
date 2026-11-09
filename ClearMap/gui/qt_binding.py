"""
qt_binding
==========

Choose the Qt binding of the process before any module imports Qt.

ClearMap imports Qt through `qtpy <https://github.com/spyder-ide/qtpy>`_, which supports PyQt5 and PyQt6 (and
PySide, which ClearMap does not support). With several bindings installed, qtpy and pyqtgraph could each pick a
different one: qtpy prefers PyQt5, pyqtgraph PyQt6. So the binding is chosen here, once, and written to the
environment variables they read:

- ``QT_API`` (qtpy), set by the user to choose; otherwise PyQt6 if installed, else PyQt5.
- ``PYQTGRAPH_QT_LIB`` (pyqtgraph), set to the same binding.

Nothing is imported: the bindings are looked for with :func:`importlib.util.find_spec`. Called by the ``__init__`` of
the two packages that may import Qt, ``ClearMap.gui`` and ``ClearMap.Visualization.Qt``, so it runs before their
first Qt import; every other module reaches Qt through them (``tests/compatibility`` checks it). A headless process
never imports them, and its environment is left alone.
"""
import importlib.util
import os
import sys

from ClearMap.Utils.exceptions import ClearMapValueError

QT_API_ENV_VAR = 'QT_API'  # Read by qtpy
PYQTGRAPH_QT_LIB_ENV_VAR = 'PYQTGRAPH_QT_LIB'  # Read by pyqtgraph
# The QT_API value of each supported binding and its module name, in order of preference
SUPPORTED_BINDINGS = {'pyqt6': 'PyQt6', 'pyqt5': 'PyQt5'}


def _is_installed(module_name):
    # An imported module may have no spec (find_spec then raises): imported means installed
    return module_name in sys.modules or importlib.util.find_spec(module_name) is not None


def choose_qt_binding():
    """
    Choose the Qt binding of the process and tell qtpy and pyqtgraph.

    Returns
    -------
    str or None
        The ``QT_API`` value of the binding ('pyqt6' or 'pyqt5'), or None if none is installed (headless).

    Raises
    ------
    ClearMapValueError
        If ``QT_API`` or ``PYQTGRAPH_QT_LIB`` names a binding that ClearMap does not support, or if they disagree.
    """
    requested = os.environ.get(QT_API_ENV_VAR)
    if requested is not None:
        binding = requested.lower()
        if binding not in SUPPORTED_BINDINGS:
            raise ClearMapValueError(f'{QT_API_ENV_VAR}={requested!r} is not a Qt binding that ClearMap supports: '
                                     f'unset it or set it to one of {list(SUPPORTED_BINDINGS)}',
                                     value=requested, expected=list(SUPPORTED_BINDINGS))
    else:
        binding = next((name for name, module in SUPPORTED_BINDINGS.items() if _is_installed(module)), None)
        if binding is None:
            return None
        os.environ[QT_API_ENV_VAR] = binding

    module_name = SUPPORTED_BINDINGS[binding]
    pyqtgraph_binding = os.environ.setdefault(PYQTGRAPH_QT_LIB_ENV_VAR, module_name)
    if pyqtgraph_binding != module_name:
        raise ClearMapValueError(f'{PYQTGRAPH_QT_LIB_ENV_VAR}={pyqtgraph_binding!r} disagrees with '
                                 f'{QT_API_ENV_VAR}={binding!r}: pyqtgraph and the rest of ClearMap would use two Qt '
                                 f'bindings. Unset {PYQTGRAPH_QT_LIB_ENV_VAR} or set it to {module_name!r}',
                                 value=pyqtgraph_binding, expected=module_name)
    return binding
