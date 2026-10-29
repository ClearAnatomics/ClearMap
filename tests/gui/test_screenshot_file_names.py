"""The screenshot tool of the documentation (tests/gui_screenshots.py): the names of the files, its command line"""
import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from tests.gui_screenshots import (DEFAULT_WINDOW_SIZE, UNTITLED, ScreenshotNamer, build_argument_parser,
                                   parse_window_size, slugify)

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
    assert parse_window_size(DEFAULT_WINDOW_SIZE) == (1000, 900)


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
