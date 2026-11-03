"""
ClearMap ships some files compressed (atlases, look-up tables) and computes others. The folder of the program is often
read-only (a cluster), so such a file is used where it is, else found in a cache folder, else built there atomically.
"""
import bz2
import gzip
import lzma
import zipfile

import pytest

from ClearMap.IO import FileUtils as fu
from ClearMap.Utils.exceptions import ClearMapNotImplementedError

CONTENT = b'look-up table ' * 1000


@pytest.fixture
def folders(tmp_path):
    source_dir, cache_dir = tmp_path / 'program', tmp_path / 'user' / 'cache'
    source_dir.mkdir()
    return source_dir, cache_dir


def write_content(path):
    with open(path, 'wb') as file:
        file.write(CONTENT)


def never_built(path):
    raise AssertionError(f'built {path}, though the file exists')


def test_a_file_next_to_the_code_is_used_and_nothing_is_built(folders):
    source_dir, cache_dir = folders
    (source_dir / 'table.npy').write_bytes(b'prepared')

    path = fu.find_or_build('table.npy', source_dir=source_dir, cache_dir=cache_dir, build=never_built)

    assert path == source_dir / 'table.npy'
    assert not cache_dir.exists()


def test_a_file_of_the_cache_is_used_when_there_is_none_next_to_the_code(folders):
    source_dir, cache_dir = folders
    cache_dir.mkdir(parents=True)
    (cache_dir / 'table.npy').write_bytes(b'cached')

    path = fu.find_or_build('table.npy', source_dir=source_dir, cache_dir=cache_dir, build=never_built)

    assert path == cache_dir / 'table.npy'


def test_a_missing_file_is_built_in_the_cache_and_not_in_the_folder_of_the_code(folders):
    source_dir, cache_dir = folders

    path = fu.find_or_build('table.npy', source_dir=source_dir, cache_dir=cache_dir,
                            build=write_content)

    assert path == cache_dir / 'table.npy'
    assert path.read_bytes() == CONTENT
    assert list(source_dir.iterdir()) == []
    assert [file.name for file in cache_dir.iterdir()] == ['table.npy']  # No temporary file left


def test_a_failing_build_leaves_no_file_and_the_next_call_builds_again(folders):
    source_dir, cache_dir = folders

    def failing_build(tmp_path):
        open(tmp_path, 'wb').write(CONTENT[:10])  # A partial file
        raise RuntimeError('out of memory')

    with pytest.raises(RuntimeError, match='out of memory'):
        fu.find_or_build('table.npy', source_dir=source_dir, cache_dir=cache_dir, build=failing_build)
    assert list(cache_dir.iterdir()) == []

    path = fu.find_or_build('table.npy', source_dir=source_dir, cache_dir=cache_dir,
                            build=write_content)
    assert path.read_bytes() == CONTENT


def compress(path, extension):
    if extension == 'zip':
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('table.npy', CONTENT)
    else:
        {'bz2': bz2, 'gzip': gzip, 'lzma': lzma}[extension].open(path, 'wb').write(CONTENT)


@pytest.mark.parametrize('extension', ['zip', 'bz2', 'gzip', 'lzma'])
def test_decompress_file_writes_the_content_of_each_kind_of_archive(extension, tmp_path):
    compressed_path = tmp_path / f'table.npy.{extension}'
    compress(compressed_path, extension)

    fu.decompress_file(compressed_path, tmp_path / 'out.npy')

    assert (tmp_path / 'out.npy').read_bytes() == CONTENT


def test_decompress_file_refuses_an_unknown_extension(tmp_path):
    (tmp_path / 'table.npy.rar').write_bytes(b'x')

    with pytest.raises(ClearMapNotImplementedError, match='rar'):
        fu.decompress_file(tmp_path / 'table.npy.rar', tmp_path / 'out.npy')
