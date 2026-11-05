"""
Workspace.info() summarises, for every file type, the range of the tags of the files found (#58).
"""
import pytest

from ClearMap.IO.Workspace import Workspace


def _workspace_with_tiles(tmp_path, *, expression, names):
    for name in names:
        (tmp_path / name).write_text('')
    return Workspace('CellMap', directory=str(tmp_path), raw=expression)


def test_info_gives_the_range_of_integer_tags(tmp_path, capsys):
    ws = _workspace_with_tiles(tmp_path, expression='tile_<X,2>_<Y,2>.tif',
                               names=['tile_01_02.tif', 'tile_03_04.tif'])

    ws.info()

    assert '{2 files, (\'X\', \'Y\'): (1, 2) -> (3, 4)}' in capsys.readouterr().out


def test_info_does_not_fail_on_a_tag_which_is_a_string(tmp_path, capsys):
    ws = _workspace_with_tiles(tmp_path, expression='<Z,S,4>_tile_<X,2>.tif',
                               names=['abcd_tile_01.tif', 'efgh_tile_02.tif'])

    ws.info()

    out = capsys.readouterr().out
    assert '2 files' in out
    assert "('abcd', 1) -> ('efgh', 2)" in out
