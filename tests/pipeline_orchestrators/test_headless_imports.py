"""
The orchestrators run headless (see the "Contract" in the docstring of
:mod:`ClearMap.pipeline_orchestrators.generic_orchestrators`): importing them must not import Qt.
Only their optional plots use it, imported in the plot methods.
"""
import pytest

from tests.qt_imports import qt_modules_imported_by

HEADLESS_MODULES = ('cell_map', 'colocalization', 'experiment_controller', 'generic_orchestrators',
                    'group_orchestrators', 'registration_orchestrator', 'sample_info_management',
                    'stitching_orchestrator', 'tract_map', 'tube_map')


@pytest.mark.parametrize('module_name', HEADLESS_MODULES)
def test_orchestrator_module_does_not_import_qt(module_name):
    assert qt_modules_imported_by(f'ClearMap.pipeline_orchestrators.{module_name}') == []
