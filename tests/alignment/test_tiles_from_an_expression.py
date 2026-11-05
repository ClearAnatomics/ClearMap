"""
The tiles of a stitching are found from a tile pattern. A pattern which matches no file used to end in
"zero-size array to reduction operation maximum which has no identity" (#193). It now says what was looked for.
"""
import numpy as np
import pytest

from ClearMap.Alignment.Stitching import stitching_rigid
from ClearMap.Utils.exceptions import ClearMapFileNotFoundError


def test_a_pattern_which_matches_no_tile_names_the_pattern(tmp_path):
    (tmp_path / 'tile_01_01.tif').write_bytes(b'')  # There are tiles, but not with this extension
    pattern = str(tmp_path / 'tile_<X,2>_<Y,2>.npy')

    with pytest.raises(ClearMapFileNotFoundError, match='No file matches the tile pattern'):
        stitching_rigid._initialize_tiles_from_expression(pattern)


def test_the_error_of_a_pattern_without_tile_gives_the_pattern_it_looked_for(tmp_path):
    with pytest.raises(ClearMapFileNotFoundError) as error:
        stitching_rigid._initialize_tiles_from_expression(str(tmp_path / 'tile_<X,2>_<Y,2>.npy'))

    assert 'tile_<X,I,2>_<Y,I,2>.npy' in str(error.value)
