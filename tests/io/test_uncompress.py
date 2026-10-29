"""
uncompress decompresses next to the file, only when the file is missing. The decompression itself is
decompress_file (streaming), written atomically, so a big file is not read whole in memory.
"""
import bz2
import gzip
import lzma
import zipfile

import pytest

from ClearMap.IO import FileUtils as fu

CONTENT = b'atlas ' * 1000
OPENERS = {'lzma': lzma.compress, 'bz2': bz2.compress, 'gzip': gzip.compress}


def never_read_whole(data):
    raise AssertionError('the compressed file was read whole in memory')


@pytest.mark.parametrize('extension', sorted(OPENERS))
def test_a_missing_file_is_decompressed_next_to_its_archive(tmp_path, extension):
    (tmp_path / f'atlas.tif.{extension}').write_bytes(OPENERS[extension](CONTENT))

    path = fu.uncompress(tmp_path / 'atlas.tif', extension='auto')

    assert path == tmp_path / 'atlas.tif'
    assert path.read_bytes() == CONTENT
    assert sorted(item.name for item in tmp_path.iterdir()) == ['atlas.tif', f'atlas.tif.{extension}']


@pytest.mark.parametrize('module', [lzma, bz2, gzip])
def test_the_archive_is_streamed_not_read_whole_in_memory(tmp_path, monkeypatch, module):
    extension = module.__name__
    (tmp_path / f'atlas.tif.{extension}').write_bytes(OPENERS[extension](CONTENT))
    monkeypatch.setattr(module, 'decompress', never_read_whole)

    fu.uncompress(tmp_path / 'atlas.tif', extension=extension)

    assert (tmp_path / 'atlas.tif').read_bytes() == CONTENT


def test_an_existing_file_is_left_alone(tmp_path):
    (tmp_path / 'atlas.tif').write_bytes(b'prepared')
    (tmp_path / 'atlas.tif.lzma').write_bytes(lzma.compress(CONTENT))

    fu.uncompress(tmp_path / 'atlas.tif', extension='lzma')

    assert (tmp_path / 'atlas.tif').read_bytes() == b'prepared'


def test_a_corrupt_archive_leaves_no_partial_file(tmp_path):
    (tmp_path / 'atlas.tif.lzma').write_bytes(lzma.compress(CONTENT)[:30])

    with pytest.raises((EOFError, lzma.LZMAError)):
        fu.uncompress(tmp_path / 'atlas.tif', extension='lzma')

    assert [item.name for item in tmp_path.iterdir()] == ['atlas.tif.lzma']


def test_a_zip_of_an_image_gives_that_image_only(tmp_path):
    with zipfile.ZipFile(tmp_path / 'atlas.tif.zip', 'w') as archive:
        archive.writestr('atlas.tif', CONTENT)
        archive.writestr('other.txt', b'other')

    path = fu.uncompress(tmp_path / 'atlas.tif')

    assert path.read_bytes() == CONTENT
    assert not (tmp_path / 'other.txt').exists()
    assert sorted(item.name for item in tmp_path.iterdir()) == ['atlas.tif', 'atlas.tif.zip']


def test_a_zip_with_the_file_gives_that_file(tmp_path):
    with zipfile.ZipFile(tmp_path / 'table.npy.zip', 'w') as archive:
        archive.writestr('table.npy', CONTENT)

    path = fu.uncompress(tmp_path / 'table.npy')

    assert path.read_bytes() == CONTENT


def test_a_zip_of_several_files_is_extracted_whole_next_to_it(tmp_path):
    with zipfile.ZipFile(tmp_path / 'tables.npy.zip', 'w') as archive:
        archive.writestr('tables.npy', CONTENT)
        archive.writestr('other.txt', b'other')
    (tmp_path / 'tables.npy').unlink(missing_ok=True)

    fu.uncompress(tmp_path / 'tables.npy')

    assert (tmp_path / 'other.txt').read_bytes() == b'other'
    assert (tmp_path / 'tables.npy').read_bytes() == CONTENT
