"""
Upgrading an experiment converts its config files and renames files, so it is only done with someone who confirms:
the dialogs of the GUI, or a terminal. A batch job (no terminal) refuses. With the terminal, the upgrade runs
without Qt, so that an experiment is upgraded from a shell, e.g. of a cluster node.

The upgrade runs in a new interpreter, in which no test has imported Qt yet, on a folder made of the default 2.1
configs of ClearMap and a few files named as ClearMap 2.1 named them. The terminal is played by a list of answers.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.config.convert_config_versions import ExperimentUpgrade_2_1_to_3_0
from ClearMap.config.upgrade_prompts import (ChannelAnswer, ConsolePrompter, check_channel_answers,
                                             suggested_channel_answers)
from tests.qt_imports import QT_MODULES

RESULT_MARKER = 'UPGRADE_RESULT:'
V2_1_DEFAULTS = Path(__file__).parents[2] / 'ClearMap' / 'config' / 'defaults' / 'v2.1'
OLD_CONFIGS = {'sample_params.cfg': 'default_sample_params.cfg',
               'alignment_params.cfg': 'default_alignment_params.cfg',
               'cell_map_params.cfg': 'default_cell_map_params.cfg'}
OLD_ASSETS = ('stitched.npy', 'resampled.tif', 'resampled_autofluorescence.tif')
OLD_ELASTIX_FOLDERS = ('elastix_auto_to_reference', 'elastix_resampled_to_auto')
SAMPLE_ID = 'S1'

# The questions, in order: the confirmation, the name and type of raw and of autofluorescence, the renames to leave
# out, the confirmation of the renames. Enter takes the suggestion.
ACCEPT_ALL = ['y', '', '', '', '', '', 'y']

UPGRADE_A_2_1_EXPERIMENT = f'''
import json, sys
import ClearMap.config.convert_config_versions as conversions
from ClearMap.config.upgrade_prompts import ConsolePrompter

answers = json.loads(sys.argv[2])


def play_the_terminal(question):
    if not answers:
        raise EOFError
    return answers.pop(0)


conversions.convert_versions('2.1', '3.0', exp_dir=sys.argv[1], create_app=False,
                             prompter=ConsolePrompter(ask=play_the_terminal, say=lambda line: None))
print({RESULT_MARKER!r} + json.dumps({{'qt': [name for name in {QT_MODULES!r} if name in sys.modules]}}))
'''


def make_a_2_1_experiment(folder: Path, *, with_sample_id_prefix: bool = False, aligned: bool = True) -> Path:
    folder.mkdir()
    for config_name, default_name in OLD_CONFIGS.items():
        shutil.copyfile(V2_1_DEFAULTS / default_name, folder / config_name)
    prefix = ''
    if with_sample_id_prefix:
        sample_config = folder / 'sample_params.cfg'
        sample_config.write_text(sample_config.read_text().replace("sample_id = ''", f"sample_id = '{SAMPLE_ID}'")
                                 .replace('use_id_as_prefix = False', 'use_id_as_prefix = True'))
        prefix = f'{SAMPLE_ID}_'
    for asset in OLD_ASSETS:
        (folder / f'{prefix}{asset}').write_text('')
    if aligned:
        for elastix_folder in OLD_ELASTIX_FOLDERS:
            (folder / f'{prefix}{elastix_folder}').mkdir()
            (folder / f'{prefix}{elastix_folder}' / 'result.0.tif').write_text('')
    return folder


def run_the_upgrade(experiment: Path, answers: list[str]) -> subprocess.CompletedProcess:
    command = [sys.executable, '-c', UPGRADE_A_2_1_EXPERIMENT, str(experiment), json.dumps(answers)]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def qt_modules_imported_by(result: subprocess.CompletedProcess) -> list[str]:
    assert result.returncode == 0, result.stderr
    result_line = next(line for line in result.stdout.splitlines() if line.startswith(RESULT_MARKER))
    return json.loads(result_line.removeprefix(RESULT_MARKER))['qt']


def content_of(folder: Path) -> dict[str, bytes]:
    return {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob('*') if path.is_file()}


def test_a_2_1_experiment_is_upgraded_with_the_suggestions_in_a_terminal_without_importing_qt(tmp_path):
    experiment = make_a_2_1_experiment(tmp_path / 'experiment')

    qt_modules = qt_modules_imported_by(run_the_upgrade(experiment, ACCEPT_ALL))

    assert qt_modules == []
    assert (experiment / 'nuclei_stitched.npy').exists()
    assert (experiment / 'autofluorescence_resampled.tif').exists()


def test_a_sample_which_was_not_aligned_has_no_elastix_folder_to_rename(tmp_path):
    experiment = make_a_2_1_experiment(tmp_path / 'experiment', aligned=False)

    qt_modules_imported_by(run_the_upgrade(experiment, ACCEPT_ALL))

    assert (experiment / 'nuclei_stitched.npy').exists()


@pytest.mark.parametrize('with_sample_id_prefix', [False, True])
def test_the_files_of_a_2_1_experiment_are_renamed_by_channel_and_keep_the_sample_id_prefix(tmp_path,
                                                                                            with_sample_id_prefix):
    experiment = make_a_2_1_experiment(tmp_path / 'experiment', with_sample_id_prefix=with_sample_id_prefix)
    prefix = f'{SAMPLE_ID}_' if with_sample_id_prefix else ''

    qt_modules = qt_modules_imported_by(run_the_upgrade(experiment, ACCEPT_ALL))

    assert qt_modules == []  # The suggestions were taken, in a terminal, without Qt
    assert sorted(path.name for path in experiment.iterdir() if path.suffix in ('.tif', '.npy')) == [
        f'{prefix}autofluorescence_resampled.tif', f'{prefix}nuclei_resampled.tif', f'{prefix}nuclei_stitched.npy']


@pytest.mark.parametrize('with_sample_id_prefix', [False, True])
def test_the_elastix_folders_of_a_2_1_experiment_are_named_by_their_channels_whatever_the_sample_id_prefix(
        tmp_path, with_sample_id_prefix):
    experiment = make_a_2_1_experiment(tmp_path / 'experiment', with_sample_id_prefix=with_sample_id_prefix,
                                       aligned=True)

    qt_modules_imported_by(run_the_upgrade(experiment, ACCEPT_ALL))

    # Never prefixed by the sample id: see the asset 'aligned'
    assert sorted(path.name for path in experiment.iterdir() if path.is_dir()) == ['atlas_to_autofluorescence',
                                                                                  'autofluorescence_to_nuclei']


def test_the_names_typed_in_the_terminal_are_used_and_a_wrong_one_is_asked_again(tmp_path):
    experiment = make_a_2_1_experiment(tmp_path / 'experiment')
    answers = ['y',
               'not valid', '', '', '',  # raw is refused: the terminal asks again
               'Fos', 'nuclei', '', '',
               '', 'y']

    qt_modules_imported_by(run_the_upgrade(experiment, answers))

    assert 'Fos' in (experiment / 'sample_params.cfg').read_text()
    assert (experiment / 'Fos_stitched.npy').exists()


def test_the_renames_left_out_in_the_terminal_are_not_done(tmp_path):
    experiment = make_a_2_1_experiment(tmp_path / 'experiment')

    qt_modules_imported_by(run_the_upgrade(experiment, ['y', '', '', '', '', '1', 'y']))  # The first of the renames

    assert sum((experiment / name).exists() for name in OLD_ASSETS) == 1  # It keeps its name, the 2 others do not


@pytest.mark.parametrize('answers', [['n'],  # The upgrade is not confirmed
                                     [],  # No terminal input at all
                                     ['y', 'Fos'],  # Ctrl-D while naming the channels
                                     ['y', '', '', '', '', '', 'n']],  # The renames are not confirmed: the configs
                         ids=['not confirmed', 'no input', 'stopped when naming', 'renames refused'])  # are converted
def test_an_upgrade_which_is_not_confirmed_leaves_the_experiment_as_it_was(tmp_path, answers):
    experiment = make_a_2_1_experiment(tmp_path / 'experiment')
    before = content_of(experiment)

    run_the_upgrade(experiment, answers)

    if answers == ['y', '', '', '', '', '', 'n']:  # Refused at the last question: the files keep their names
        assert all((experiment / name).exists() for name in OLD_ASSETS)
    else:
        assert content_of(experiment) == before


def test_a_batch_job_cannot_upgrade_an_experiment(tmp_path):
    experiment = make_a_2_1_experiment(tmp_path / 'experiment')
    before = content_of(experiment)

    result = subprocess.run([sys.executable, '-c', 'from ClearMap.config.upgrade_prompts import ConsolePrompter; '
                                                   'ConsolePrompter()'],
                            capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False)

    assert result.returncode != 0
    assert 'needs someone to confirm it' in result.stderr
    assert content_of(experiment) == before


def test_the_raw_channel_of_a_sample_with_arteries_is_suggested_as_vessels():
    suggested = suggested_channel_answers(['raw', 'arteries', 'autofluorescence', 'extra'])

    assert suggested == {'raw': ChannelAnswer('vessels', 'vessels'),
                         'arteries': ChannelAnswer('arteries', 'arteries'),
                         'autofluorescence': ChannelAnswer('autofluorescence', 'autofluorescence'),
                         'extra': ChannelAnswer('extra', 'undefined')}


def test_the_raw_channel_of_a_sample_without_arteries_is_suggested_as_nuclei():
    assert suggested_channel_answers(['raw', 'autofluorescence'])['raw'] == ChannelAnswer('nuclei', 'nuclei')


def test_two_channels_cannot_get_the_same_name_whatever_the_case():
    with pytest.raises(ClearMapValueError, match='unique whatever the case'):
        check_channel_answers({'raw': ChannelAnswer('Fos', 'nuclei'), 'arteries': ChannelAnswer('fos', 'arteries')})


def test_an_unknown_content_type_is_refused_naming_the_channel():
    with pytest.raises(ClearMapValueError, match="'raw'.*'nucleus'"):
        check_channel_answers({'raw': ChannelAnswer('Fos', 'nucleus')})


def test_the_terminal_takes_the_suggestions_on_enter():
    prompter = ConsolePrompter(ask=lambda question: '', say=lambda line: None)

    assert prompter.choose_channels(suggested_channel_answers(['raw'])) == {'raw': ChannelAnswer('nuclei', 'nuclei')}


@pytest.mark.parametrize('old_name, new_name', [
    ('resampled_autofluorescence.tif', 'autofluorescence_resampled.tif'),
    ('resampled_autofluorescence_debug.tif', 'autofluorescence_resampled_debug.tif'),  # Was resampled__debug
    ('autofluorescence_resampled.tif', 'autofluorescence_resampled.tif'),  # Was autofluorescence__resampled
    ('autofluorescence.tif', 'autofluorescence.tif'),
    ('stitched_arteries.npy', 'arteries_stitched.npy'),
    ('stitched_arteries_debug.npy', 'arteries_stitched_debug.npy'),
    ('stitched.npy', 'nuclei_stitched.npy')])
def test_the_channel_word_of_an_old_file_name_is_replaced_by_the_channel_without_a_doubled_underscore(
        tmp_path, old_name, new_name):
    upgrader = ExperimentUpgrade_2_1_to_3_0(tmp_path, prompter=ConsolePrompter(ask=input_not_expected))
    upgrader.new_channels = {'raw': ('nuclei', 'nuclei'), 'arteries': ('arteries', 'arteries'),
                             'autofluorescence': ('autofluorescence', 'autofluorescence')}

    assert upgrader._compute_new_file_names({old_name: old_name}) == {old_name: new_name}


def input_not_expected(question):
    raise AssertionError(f'Nobody should be asked {question!r}')
