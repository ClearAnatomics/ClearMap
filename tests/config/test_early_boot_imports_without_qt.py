"""
early_boot patches the temporary folder before anything else, for the GUI and for the headless runs (ClearMap.run, on
a cluster node, without display): importing it must not import Qt. Only its prompts use it, imported where they run.

The module is imported in a new interpreter: in this one, any test may already have imported Qt.
"""
import subprocess
import sys

QT_MODULES = ('PyQt5', 'pyqtgraph')
RESULT_MARKER = 'QT_MODULES_IMPORTED:'


def test_early_boot_does_not_import_qt():
    script = ('import sys, ClearMap.config.early_boot\n'
              f'print({RESULT_MARKER!r} + ",".join(name for name in {QT_MODULES!r} if name in sys.modules))')

    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    result_line = next(line for line in result.stdout.splitlines() if line.startswith(RESULT_MARKER))
    assert result_line == RESULT_MARKER, f'early_boot imports {result_line.removeprefix(RESULT_MARKER)}'
