"""
early_boot patches the temporary folder before anything else, for the GUI and for the headless runs (ClearMap.run, on
a cluster node, without display): importing it must not import Qt. Only its prompts use it, imported where they run.
"""
from tests.qt_imports import qt_modules_imported_by


def test_early_boot_does_not_import_qt():
    assert qt_modules_imported_by('ClearMap.config.early_boot') == []
