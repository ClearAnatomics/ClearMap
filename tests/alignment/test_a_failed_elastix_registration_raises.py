"""
A failed elastix registration raises, naming its log (#P0018). The check looked for "fail" in the log only: elastix
stopped by an exception ("Too many samples map outside moving image buffer") ends its log with "Errors occurred!",
exits with an error and writes no result, and the pipeline went on to a "No tiles found" warning about the missing
result image.

A fake elastix writes the log, the transforms and the results it is told to, and exits with the code it is told to.
"""
import sys

import pytest

from ClearMap.Alignment import Elastix as elastix
from ClearMap.Utils.exceptions import ElastixFailedError

LOG_TEXT_VARIABLE = 'FAKE_ELASTIX_LOG_TEXT'
EXIT_CODE_VARIABLE = 'FAKE_ELASTIX_EXIT_CODE'
WRITTEN_STEPS_VARIABLE = 'FAKE_ELASTIX_WRITTEN_STEPS'

FAKE_ELASTIX_SOURCE = f'''#!{sys.executable}
"""Fake elastix: writes its log, the transform and result of the first steps, exits with the given code"""
import os, sys
from pathlib import Path

arguments = sys.argv[1:]
output_directory = Path(arguments[arguments.index('-out') + 1])
(output_directory / 'elastix.log').write_text(os.environ['{LOG_TEXT_VARIABLE}'])
for step in range(int(os.environ['{WRITTEN_STEPS_VARIABLE}'])):
    (output_directory / f'TransformParameters.{{step}}.txt').touch()
    (output_directory / f'result.{{step}}.mhd').touch()
sys.exit(int(os.environ['{EXIT_CODE_VARIABLE}']))
'''

SUCCESSFUL_LOG_END = 'Total time elapsed: 52.1s.\n'
# The end of the log of #P0018
FAILED_LOG_END = ('Description: itk::ERROR: AdvancedMattesMutualInformationMetric(0x13d9c760): '
                  'Too many samples map outside moving image buffer: 10980 / 116812\n'
                  '\n'
                  'Error occurred during actual registration.\n'
                  '\n'
                  'Errors occurred!\n')
PARAMETER_FILE_NAMES = ('align_affine.txt', 'align_bspline.txt')


@pytest.fixture()
def fake_elastix(tmp_path, monkeypatch):
    executable = tmp_path / 'elastix'
    executable.write_text(FAKE_ELASTIX_SOURCE)
    executable.chmod(0o755)
    monkeypatch.setattr(elastix, 'elastix_binary', str(executable))
    monkeypatch.setattr(elastix, 'initialized', True)

    def run(*, log_text, exit_code=0, written_steps=len(PARAMETER_FILE_NAMES)):
        monkeypatch.setenv(LOG_TEXT_VARIABLE, log_text)
        monkeypatch.setenv(EXIT_CODE_VARIABLE, str(exit_code))
        monkeypatch.setenv(WRITTEN_STEPS_VARIABLE, str(written_steps))
        return elastix.align(fixed_image=str(tmp_path / 'fos_resampled.tif'),
                             moving_image=str(tmp_path / 'auto_resampled.tif'),
                             parameter_files=[str(tmp_path / name) for name in PARAMETER_FILE_NAMES],
                             result_directory=str(tmp_path / 'auto_to_fos'), processes=1,
                             check_alignment_success=True)
    return run


def test_a_successful_registration_returns_its_result_directory(tmp_path, fake_elastix):
    assert fake_elastix(log_text=SUCCESSFUL_LOG_END) == str(tmp_path / 'auto_to_fos')


def test_a_registration_whose_log_reports_errors_raises_naming_the_log(tmp_path, fake_elastix):
    with pytest.raises(ElastixFailedError, match='Errors occurred') as error:
        fake_elastix(log_text=FAILED_LOG_END, written_steps=0)

    assert str(tmp_path / 'auto_to_fos' / 'elastix.log') in str(error.value)


def test_a_registration_which_exits_with_an_error_raises_with_the_exit_code(fake_elastix):
    with pytest.raises(ElastixFailedError, match='exited with code 1'):
        fake_elastix(log_text=SUCCESSFUL_LOG_END, exit_code=1)


def test_a_registration_which_leaves_out_the_last_step_raises_naming_its_files(fake_elastix):
    with pytest.raises(ElastixFailedError, match=r'TransformParameters\.1\.txt, result\.1\.mhd'):
        fake_elastix(log_text=SUCCESSFUL_LOG_END, written_steps=1)
