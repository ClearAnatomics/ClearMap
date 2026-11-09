"""
``python setup.py install`` copies ClearMap to site-packages, where git cannot say which commit it came from: setup.py
records the commit of the checkout in the built package (not in the source tree), and ``get_commit_info`` reads it.
"""
import ast
import os
from dataclasses import asdict
import subprocess

import pytest

from conftest import SETUP_SCRIPT  # pytest puts this directory on the path

import importlib.util

from ClearMap.Utils.git_info import get_commit_info, STAMP_FILE_NAME, PACKAGE_DIRECTORY

BRANCH = 'v3.2.x'


@pytest.fixture(scope='module')
def setup_script():
    spec = importlib.util.spec_from_file_location('clearmap_setup_script', SETUP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(directory, *arguments):
    subprocess.run(['git', '-C', str(directory), '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
                    *arguments], check=True, capture_output=True, env={'PATH': os.environ['PATH']})


@pytest.fixture
def source_checkout(tmp_path):
    """A git checkout that holds the git_info module of this ClearMap, as setup.py loads it"""
    root = tmp_path / 'checkout'
    (root / 'ClearMap' / 'Utils').mkdir(parents=True)
    (root / 'ClearMap' / 'Utils' / 'git_info.py').write_text((PACKAGE_DIRECTORY / 'Utils' / 'git_info.py').read_text())
    _git(root, 'init', '-q', f'--initial-branch={BRANCH}')
    _git(root, 'add', 'ClearMap')
    _git(root, 'commit', '-q', '-m', 'first')
    return root


def test_the_installed_package_knows_the_commit_it_was_built_from(setup_script, source_checkout, tmp_path):
    built = tmp_path / 'build' / 'lib' / 'ClearMap'
    built.mkdir(parents=True)

    stamped = setup_script.stamp_installed_commit(source_checkout, built)

    assert stamped is not None and stamped.branch == BRANCH
    # The built package is not in a repository: only the stamp tells (setup.py has its own copy of the class)
    assert asdict(get_commit_info(built)) == asdict(stamped)


def test_the_source_tree_is_left_alone_by_the_installation(setup_script, source_checkout, tmp_path):
    built = tmp_path / 'build' / 'ClearMap'
    built.mkdir(parents=True)

    setup_script.stamp_installed_commit(source_checkout, built)

    assert not (source_checkout / 'ClearMap' / STAMP_FILE_NAME).exists()
    assert subprocess.run(['git', '-C', str(source_checkout), 'status', '--porcelain'], capture_output=True,
                          text=True, check=True).stdout == ''


def test_installing_from_a_copy_that_is_neither_a_checkout_nor_a_zip_of_a_commit_warns_and_installs_without_a_commit(
        setup_script, tmp_path, capsys):
    root = tmp_path / 'unpacked_zip'
    (root / 'ClearMap' / 'Utils').mkdir(parents=True)
    (root / 'ClearMap' / 'Utils' / 'git_info.py').write_text((PACKAGE_DIRECTORY / 'Utils' / 'git_info.py').read_text())
    built = tmp_path / 'build' / 'ClearMap'
    built.mkdir(parents=True)

    assert setup_script.stamp_installed_commit(root, built) is None

    assert 'WARNING' in capsys.readouterr().out
    assert not (built / STAMP_FILE_NAME).exists()


def test_setup_stamps_the_package_when_it_builds_it():
    tree = ast.parse(SETUP_SCRIPT.read_text(encoding='utf-8'))
    cmdclass = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                and getattr(node.func, 'id', None) == 'setup' for keyword in node.keywords
                if keyword.arg == 'cmdclass']

    assert cmdclass and 'build_py' in [key.value for key in cmdclass[0].keys]


def test_installing_from_the_zip_of_a_commit_records_that_commit(setup_script, source_checkout, tmp_path):
    (source_checkout / '.gitattributes').write_text('ClearMap/_git_archive.txt export-subst\n')
    (source_checkout / 'ClearMap' / '_git_archive.txt').write_text('$Format:%H$\n$Format:%cs$\n$Format:%D$\n')
    _git(source_checkout, 'add', '.gitattributes', 'ClearMap')
    _git(source_checkout, 'commit', '-q', '-m', 'archive file')
    head = subprocess.run(['git', '-C', str(source_checkout), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()
    unpacked = tmp_path / 'zip'
    unpacked.mkdir()
    archive = subprocess.run(['git', '-C', str(source_checkout), 'archive', 'HEAD'], check=True,
                             capture_output=True).stdout
    subprocess.run(['tar', '-x', '-C', str(unpacked)], input=archive, check=True)
    built = tmp_path / 'build' / 'ClearMap'
    built.mkdir(parents=True)

    stamped = setup_script.stamp_installed_commit(unpacked, built)

    assert stamped.commit_hash == head
    assert get_commit_info(built).commit_hash == head  # The installed copy has neither the zip nor git
