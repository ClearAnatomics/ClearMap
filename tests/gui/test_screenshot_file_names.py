"""The screenshot tool of the documentation (tests/gui_screenshots.py): the names of the files, its command line"""
import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gui_screenshots import (DEFAULT_WINDOW_SIZE, UNTITLED, ScreenshotNamer, build_argument_parser,
                                   parse_widening_percent, parse_window_size, prepare_output_dir, slugify,
                                   widened_dock_width)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('title, expected', [
    ('Sample Info', 'sample_info'),
    ('&Atlas settings', 'atlas_settings'),  # The mnemonic of Qt
    ('🛠️ Edit workspace folders', 'edit_workspace_folders'),
    ('Résolution (µm)', 'resolution_m'),
    ('  --  ', UNTITLED),
    ('', UNTITLED),
])
def test_a_title_becomes_a_lower_case_ascii_file_name_part(title, expected):
    assert slugify(title) == expected


def test_a_screenshot_is_in_the_folder_of_its_tab_and_named_after_all_the_titles_it_was_reached_by(tmp_path):
    namer = ScreenshotNamer(tmp_path)

    assert namer.next_path(['Sample Info']) == tmp_path / 'sample_info' / 'sample_info.png'
    assert namer.next_path(['Stitching', 'cfos', 'Rigid']) == tmp_path / 'stitching' / 'stitching__cfos__rigid.png'


def test_two_pages_with_the_same_title_do_not_overwrite_each_other(tmp_path):
    namer = ScreenshotNamer(tmp_path)

    paths = [namer.next_path(['Registration', 'Advanced']) for _ in range(3)]

    assert [path.name for path in paths] == ['registration__advanced.png', 'registration__advanced_2.png',
                                             'registration__advanced_3.png']


def test_titles_that_only_differ_by_what_the_file_name_drops_do_not_overwrite_each_other(tmp_path):
    namer = ScreenshotNamer(tmp_path)

    assert namer.next_path(['Cell Counter', '&Run']) != namer.next_path(['Cell Counter', 'Run'])


def test_a_screenshot_without_a_tab_title_is_refused(tmp_path):
    with pytest.raises(ValueError, match='tab'):
        ScreenshotNamer(tmp_path).next_path([])


def test_the_window_size_is_read_as_width_by_height():
    assert parse_window_size('1200x800') == (1200, 800)
    assert parse_window_size(DEFAULT_WINDOW_SIZE) == (1000, 1300)


@pytest.mark.parametrize('text', ['1200', '1200,800', 'x800', '12.5x800', ''])
def test_a_malformed_window_size_is_refused_with_the_expected_format(text):
    with pytest.raises(argparse.ArgumentTypeError, match='WIDTHxHEIGHT'):
        parse_window_size(text)


def test_the_command_line_needs_the_output_folder_and_has_the_default_size(tmp_path):
    arguments = build_argument_parser().parse_args([str(tmp_path)])

    assert arguments.out_dir == tmp_path
    assert arguments.size == parse_window_size(DEFAULT_WINDOW_SIZE)
    assert not arguments.use_real_home
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([])


def test_the_tool_imports_without_qt_and_without_clearmap():
    """The environment (offscreen, home folder) is set up before the first import of Qt and ClearMap"""
    code = ('import sys; import tests.gui_screenshots; '
            'loaded = sorted(name for name in sys.modules if name.split(".")[0] in ("PyQt5", "ClearMap")); '
            'print(loaded); sys.exit(1 if loaded else 0)')

    result = subprocess.run([sys.executable, '-c', code], cwd=REPOSITORY_ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr


def test_an_exception_nothing_catches_is_printed_when_it_happens_and_kept_for_the_count(capsys):
    from tests.gui_screenshots import make_excepthook

    uncaught_exceptions = []
    hook = make_excepthook(uncaught_exceptions)
    try:
        raise RuntimeError('the experiment could not be opened')
    except RuntimeError as error:
        hook(type(error), error, error.__traceback__)

    assert 'RuntimeError: the experiment could not be opened' in capsys.readouterr().err
    assert len(uncaught_exceptions) == 1 and 'could not be opened' in uncaught_exceptions[0]


def test_the_output_folder_is_created_with_its_parents_when_it_can_be_written(tmp_path):
    out_dir = tmp_path / 'not' / 'yet' / 'there'

    assert prepare_output_dir(out_dir) == out_dir
    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []  # The probe leaves nothing behind


def test_an_output_folder_that_cannot_be_created_is_refused_naming_the_folder_before_the_run_starts(tmp_path,
                                                                                                    monkeypatch):
    from ClearMap.Utils.exceptions import ClearMapPermissionError

    def refuse_to_create(self, *args, **kwargs):
        raise PermissionError(13, 'Permission denied', '/data/tmp')

    monkeypatch.setattr(Path, 'mkdir', refuse_to_create)

    with pytest.raises(ClearMapPermissionError, match='/data/tmp/clearmap_shots'):
        prepare_output_dir(Path('/data/tmp/clearmap_shots'))


def test_an_output_folder_that_is_a_file_is_refused_naming_the_folder(tmp_path):
    from ClearMap.Utils.exceptions import ClearMapIoException

    a_file = tmp_path / 'shots'
    a_file.write_text('not a folder')

    with pytest.raises(ClearMapIoException, match='shots'):
        prepare_output_dir(a_file)


def test_the_dock_is_made_narrower_by_what_the_panel_gains():
    assert widened_dock_width(dock_width=800, panel_width=200, widening_percent=30) == 740
    assert widened_dock_width(dock_width=800, panel_width=200, widening_percent=0) == 800


def test_the_dock_width_never_goes_below_zero():
    assert widened_dock_width(dock_width=50, panel_width=400, widening_percent=30) == 0


def test_the_panel_is_widened_by_30_percent_unless_told_otherwise(tmp_path):
    parser = build_argument_parser()

    assert parser.parse_args([str(tmp_path)]).panel_widening == 30.0
    assert parser.parse_args([str(tmp_path), '--panel-widening', '25']).panel_widening == 25.0


@pytest.mark.parametrize('text', ['-5', 'wide', ''])
def test_a_widening_that_is_negative_or_not_a_number_is_refused_with_the_expected_form(text):
    with pytest.raises(argparse.ArgumentTypeError, match='0 or more'):
        parse_widening_percent(text)
