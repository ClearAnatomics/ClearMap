"""
The orchestrators run headless (see the "Contract" in the docstring of
:mod:`ClearMap.pipeline_orchestrators.generic_orchestrators`): importing them must not import Qt.
Only their optional plots use it, imported in the plot methods.

Each module is imported in a new interpreter: in this one, any test may already have imported Qt.
"""
import subprocess
import sys

import pytest

QT_MODULES = ('PyQt5', 'pyqtgraph')
RESULT_MARKER = 'QT_MODULES_IMPORTED:'
HEADLESS_MODULES = ('cell_map', 'colocalization', 'experiment_controller', 'generic_orchestrators',
                    'group_orchestrators', 'registration_orchestrator', 'sample_info_management',
                    'stitching_orchestrator', 'tract_map', 'tube_map')


def _qt_modules_imported_by(module: str) -> list[str]:
    """The Qt modules in sys.modules after importing `module` in a new interpreter"""
    script = (f'import sys, {module}\n'
              f'print({RESULT_MARKER!r} + ",".join(name for name in {QT_MODULES!r} if name in sys.modules))')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'Could not import {module}:\n{result.stderr}')
    # The modules may print when imported: only read the line of the script
    result_line = next(line for line in result.stdout.splitlines() if line.startswith(RESULT_MARKER))
    return [name for name in result_line.removeprefix(RESULT_MARKER).split(',') if name]


@pytest.mark.parametrize('module_name', HEADLESS_MODULES)
def test_orchestrator_module_does_not_import_qt(module_name):
    assert _qt_modules_imported_by(f'ClearMap.pipeline_orchestrators.{module_name}') == []
