"""
Upgrading a 3.0 experiment folder to 3.1 asks no question: it runs without Qt, e.g. from a script or on a cluster
node (ExperimentController.upgrade_configs). It used to create a QApplication (and the module imported the GUI
dialogs at the top), so it needed Qt and a display.

The upgrade runs in a new interpreter, in which no test has imported Qt yet. The converters of the files are replaced
by recorders: only the decisions of the upgrade are under test here, not the conversion of each file.
"""
import json
import subprocess
import sys

from tests.qt_imports import QT_MODULES

RESULT_MARKER = 'UPGRADE_RESULT:'
EXPERIMENT_FILES = ('sample_params.cfg', 'stitching_params.cfg')  # The 3.0 names of two configs of an experiment

UPGRADE_A_3_0_EXPERIMENT = f'''
import json, sys
import ClearMap.config.convert_config_versions as conversions

converted = []


def record_conversion(config_name, *args, **kwargs):
    converted.append(config_name)


for config_name in ('sample', 'stitching', 'registration', 'cell_map', 'tract_map', 'colocalization', 'vasculature',
                    'batch', 'group_analysis'):
    setattr(conversions, f'convert_{{config_name}}_3_0_to_3_1', record_conversion.__get__(config_name))

conversions.convert_versions('3.0', '3.1', exp_dir=sys.argv[1])
print({RESULT_MARKER!r} + json.dumps({{'converted': converted,
                                       'qt': [name for name in {QT_MODULES!r} if name in sys.modules]}}))
'''


def test_a_3_0_experiment_is_upgraded_to_3_1_without_importing_qt(tmp_path):
    for file_name in EXPERIMENT_FILES:
        (tmp_path / file_name).write_text('')

    result = subprocess.run([sys.executable, '-c', UPGRADE_A_3_0_EXPERIMENT, str(tmp_path)],
                            capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    result_line = next(line for line in result.stdout.splitlines() if line.startswith(RESULT_MARKER))
    upgrade = json.loads(result_line.removeprefix(RESULT_MARKER))
    assert upgrade == {'converted': ['sample', 'stitching'], 'qt': []}
