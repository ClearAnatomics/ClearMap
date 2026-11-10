"""
Every designer file of the GUI builds into a widget of its root class with the Qt binding installed (the one chosen
by gui/qt_binding, so run it with QT_API=pyqt5 and QT_API=pyqt6), offscreen, through gui_utils_base.load_ui_into.

The widgets are built in a separate, isolated Python (``-I``: no PYTHONPATH, so no stub of Qt from the test
environment), with the repository first on its path. Skipped when that Python has no Qt.
"""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPOSITORY_FOLDER = Path(__file__).resolve().parents[2]
CREATOR_FOLDER = REPOSITORY_FOLDER / 'ClearMap' / 'gui' / 'creator'
UI_FILES = sorted(CREATOR_FOLDER.glob('*.ui'))
NO_QT_KEY = 'no_qt'

BUILD_EVERY_DESIGNER_FILE = '''
import json, sys
sys.path.insert(0, sys.argv[1])
try:
    from qtpy import QtWidgets
except Exception as error:  # ImportError, or qtpy's error when it finds no binding
    print(json.dumps({"%(no_qt)s": repr(error)}))
    sys.exit(0)
from ClearMap.gui.gui_utils_base import load_ui_into
app = QtWidgets.QApplication([])
results = {}
for ui_name, class_name in json.loads(sys.argv[2]).items():
    try:
        load_ui_into(getattr(QtWidgets, class_name)(), ui_name)
        results[ui_name] = None
    except Exception as error:
        results[ui_name] = f"{type(error).__name__}: {error}"
print(json.dumps(results))
''' % {'no_qt': NO_QT_KEY}


def _root_class(ui_file):
    return ET.parse(ui_file).getroot().find('widget').get('class')


@pytest.fixture(scope='module')
def build_errors():
    root_classes = {ui_file.name: _root_class(ui_file) for ui_file in UI_FILES}
    environment = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
    result = subprocess.run([sys.executable, '-I', '-c', BUILD_EVERY_DESIGNER_FILE, str(REPOSITORY_FOLDER),
                             json.dumps(root_classes)], capture_output=True, text=True, env=environment, timeout=300)
    assert result.returncode == 0, result.stderr
    errors = json.loads(result.stdout.strip().splitlines()[-1])
    if NO_QT_KEY in errors:
        pytest.skip(f'no Qt in {sys.executable}: {errors[NO_QT_KEY]}')
    return errors


@pytest.mark.parametrize('ui_file', UI_FILES, ids=[ui_file.name for ui_file in UI_FILES])
def test_the_designer_file_builds_into_its_root_class(build_errors, ui_file):
    assert build_errors[ui_file.name] is None
