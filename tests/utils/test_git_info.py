import os
import subprocess

import pytest

from ClearMap.Utils.git_info import CommitInfo, get_commit_info, write_commit_stamp, PACKAGE_DIRECTORY, \
    SHORT_HASH_LENGTH, DETACHED_HEAD_LABEL, STAMP_FILE_NAME, ARCHIVE_FILE_NAME

COMMIT_DATE = '2026-11-05'
BRANCH = 'v3.2.x'


def _git(directory, *arguments, **environment):
    subprocess.run(['git', '-C', str(directory), '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
                    *arguments], check=True, capture_output=True,
                   env={'GIT_COMMITTER_DATE': f'{COMMIT_DATE}T12:00:00+00:00', 'PATH': os.environ['PATH'],
                        **environment})


@pytest.fixture
def checkout(tmp_path):
    """The root of a git repository that holds a ClearMap package folder"""
    _git(tmp_path, 'init', '-q', f'--initial-branch={BRANCH}')
    (tmp_path / 'ClearMap').mkdir()
    (tmp_path / 'ClearMap' / 'a.txt').write_text('a')
    _git(tmp_path, 'add', 'ClearMap')
    _git(tmp_path, 'commit', '-q', '-m', 'first')
    return tmp_path


@pytest.fixture
def package(checkout):
    return checkout / 'ClearMap'


def _head(directory):
    return subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()


def test_the_commit_of_a_checkout_is_the_one_that_is_checked_out(checkout, package):
    assert get_commit_info(package) == CommitInfo(_head(checkout), COMMIT_DATE, BRANCH)


def test_the_commit_follows_the_checkout_without_any_file_to_refresh(checkout, package):
    first = get_commit_info(package)
    (package / 'b.txt').write_text('b')
    _git(checkout, 'add', 'ClearMap')
    _git(checkout, 'commit', '-q', '-m', 'second')

    assert get_commit_info(package).commit_hash != first.commit_hash


def test_a_detached_head_has_no_branch(checkout, package):
    _git(checkout, 'checkout', '-q', '--detach')

    info = get_commit_info(package)

    assert info.branch is None
    assert str(info) == f'{_head(checkout)[:SHORT_HASH_LENGTH]} ({DETACHED_HEAD_LABEL}, {COMMIT_DATE})'


def test_the_description_is_the_short_hash_the_branch_and_the_date(checkout, package):
    assert str(get_commit_info(package)) == f'{_head(checkout)[:SHORT_HASH_LENGTH]} ({BRANCH}, {COMMIT_DATE})'


def test_a_folder_that_is_not_a_git_checkout_has_no_commit(tmp_path):
    assert get_commit_info(tmp_path) is None


def test_without_git_installed_there_is_no_commit(package, monkeypatch):
    monkeypatch.setenv('PATH', '')

    assert get_commit_info(package) is None


def test_a_package_installed_inside_another_repository_has_not_the_commit_of_that_repository(checkout):
    installed = checkout / 'site-packages' / 'ClearMap'  # e.g. a conda environment kept in a git repository
    installed.mkdir(parents=True)

    assert get_commit_info(installed) is None


def test_an_installed_package_reports_the_commit_recorded_by_the_installation(tmp_path):
    installed = tmp_path / 'ClearMap'  # Not in a git repository
    installed.mkdir()
    recorded = CommitInfo('0123456789abcdef0123456789abcdef01234567', COMMIT_DATE, BRANCH)
    write_commit_stamp(recorded, installed)

    assert get_commit_info(installed) == recorded


def test_a_detached_head_survives_the_stamp(tmp_path):
    recorded = CommitInfo('0123456789abcdef0123456789abcdef01234567', COMMIT_DATE, None)
    write_commit_stamp(recorded, tmp_path)

    assert get_commit_info(tmp_path).branch is None


def test_the_stamp_wins_over_the_repository_around_it(checkout, package):
    recorded = CommitInfo('0123456789abcdef0123456789abcdef01234567', COMMIT_DATE, BRANCH)
    write_commit_stamp(recorded, package)

    assert get_commit_info(package) == recorded


@pytest.mark.parametrize('content', ['', 'not json', '{"commit_hash": "abc"}', '[1, 2]'])
def test_a_stamp_that_cannot_be_read_is_reported_and_git_is_asked_instead(checkout, package, content, caplog):
    (package / STAMP_FILE_NAME).write_text(content)

    with caplog.at_level('WARNING'):
        info = get_commit_info(package)

    assert info.commit_hash == _head(checkout)
    assert STAMP_FILE_NAME in caplog.text


def test_the_package_directory_is_the_one_of_clearmap():
    assert (PACKAGE_DIRECTORY / 'Utils' / 'git_info.py').exists()


def _archive_of(checkout, destination, revision='HEAD'):
    """What a zip of the commit holds, unpacked: git archive builds it as the download of a commit does"""
    destination.mkdir()
    archive = subprocess.run(['git', '-C', str(checkout), 'archive', revision], check=True, capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(destination)], input=archive, check=True)
    return destination / 'ClearMap'


@pytest.fixture
def checkout_with_archive_file(checkout):
    (checkout / '.gitattributes').write_text(f'ClearMap/{ARCHIVE_FILE_NAME} export-subst\n')
    (checkout / 'ClearMap' / ARCHIVE_FILE_NAME).write_text('$Format:%H$\n$Format:%cs$\n$Format:%D$\n')
    _git(checkout, 'add', '.gitattributes', 'ClearMap')
    _git(checkout, 'commit', '-q', '-m', 'archive file')
    return checkout


def test_the_unpacked_archive_of_a_commit_knows_the_commit_it_was_made_of(checkout_with_archive_file, tmp_path):
    unpacked = _archive_of(checkout_with_archive_file, tmp_path / 'zip')

    info = get_commit_info(unpacked)

    assert info.commit_hash == _head(checkout_with_archive_file)
    assert info.commit_date == COMMIT_DATE
    assert info.branch == BRANCH  # HEAD -> branch, as the archive of a branch gives it


def test_the_unpacked_archive_of_a_tag_knows_the_tag(checkout_with_archive_file, tmp_path):
    _git(checkout_with_archive_file, 'tag', 'v3.2.0')
    _git(checkout_with_archive_file, 'checkout', '-q', '--detach')  # The archive of a tag names no branch

    info = get_commit_info(_archive_of(checkout_with_archive_file, tmp_path / 'zip', revision='v3.2.0'))

    assert info.tag == 'v3.2.0'
    assert str(info) == f'{_head(checkout_with_archive_file)[:SHORT_HASH_LENGTH]} (tag v3.2.0, {COMMIT_DATE})'


def test_the_archive_file_of_a_copy_says_nothing_because_git_did_not_fill_it(tmp_path_factory):
    copy = tmp_path_factory.mktemp('copy')  # Not in a repository, and the placeholders are still there
    (copy / ARCHIVE_FILE_NAME).write_text('$Format:%H$\n$Format:%cs$\n$Format:%D$\n')

    assert get_commit_info(copy) is None


def test_a_checkout_at_a_tag_knows_the_tag(checkout):
    _git(checkout, 'tag', 'v3.2.0')

    assert get_commit_info(checkout / 'ClearMap').tag == 'v3.2.0'


def test_a_tagged_commit_on_a_branch_is_described_by_its_tag():
    # A tag names one commit for good, a branch moves on: the tag says which release it is
    info = CommitInfo(commit_hash='0123456789abcdef', commit_date=COMMIT_DATE, branch=BRANCH, tag='v3.2.0')

    assert str(info) == f'{info.commit_hash[:SHORT_HASH_LENGTH]} (tag v3.2.0, {COMMIT_DATE})'


def test_a_stamp_written_before_tags_were_recorded_is_still_read(tmp_path):
    (tmp_path / STAMP_FILE_NAME).write_text('{"commit_hash": "abc", "commit_date": "2026-11-05", "branch": "v3.1.x"}')

    assert get_commit_info(tmp_path) == CommitInfo('abc', '2026-11-05', 'v3.1.x')
