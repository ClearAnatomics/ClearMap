"""
Which Qt modules a module imports, for the tests of the modules that must import headless.

The module is imported in a new interpreter: in the one of the tests, any test may already have imported Qt.
"""
import subprocess
import sys

# qtpy and every binding it can load, and pyqtgraph, which loads one on its own
QT_MODULES = ('qtpy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'pyqtgraph')
RESULT_MARKER = 'QT_MODULES_IMPORTED:'


def qt_modules_imported_by(module: str) -> list[str]:
    """The Qt modules in sys.modules after importing `module` in a new interpreter"""
    script = (f'import sys, {module}\n'
              f'print({RESULT_MARKER!r} + ",".join(name for name in {QT_MODULES!r} if name in sys.modules))')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'Could not import {module}:\n{result.stderr}')
    # The modules may print when imported: only read the line of the script
    result_line = next(line for line in result.stdout.splitlines() if line.startswith(RESULT_MARKER))
    return [name for name in result_line.removeprefix(RESULT_MARKER).split(',') if name]
