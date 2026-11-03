"""
install_gui.sh takes ClearMap3.yml as the environment file unless -f gives one, whatever the other options (#129:
with -s alone, the folder was opened as the environment file), and stops before doing anything if it is missing.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALL_SCRIPT = Path(__file__).parents[2] / 'install_gui.sh'


def _run_install(arguments, folder):
    """Without conda, python or git on the PATH: should the script go further, it could not install anything"""
    tools = folder.parent / f'{folder.name}_tools'
    tools.mkdir()
    for tool in ('dirname', 'cat'):
        (tools / tool).symlink_to(shutil.which(tool))
    return subprocess.run([shutil.which('bash'), str(INSTALL_SCRIPT), *arguments], cwd=folder, env={'PATH': str(tools)},
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('arguments', [[], ['-s']])
def test_without_f_the_environment_file_is_clearmap3_yml(arguments, tmp_path):
    result = _run_install(arguments, tmp_path)  # A folder without it

    assert result.returncode == 1
    assert "Environment file 'ClearMap3.yml' not found" in result.stdout


@pytest.mark.parametrize('arguments', [['-f'], ['-s', '-f', 'missing.yml']])
def test_a_missing_environment_file_stops_the_install_before_anything_is_done(arguments, tmp_path):
    result = _run_install(arguments, tmp_path)

    assert result.returncode == 1
    assert 'not found' in result.stdout
    assert 'Usage' in result.stderr
    assert list(tmp_path.iterdir()) == []
