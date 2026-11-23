"""
A wobbly layout holds, for each pair of tiles, one displacement per slice of their overlap along z. The rigid alignment
moves the tiles along z, so until the wobbly alignment recomputes them, the displacements are those of the overlaps
before: placing such a layout failed with "could not broadcast input array from shape (2433,2) into shape (2430,2)"
(#P0016). The placement now refuses it first, naming the tiles and saying the wobbly alignment did not run.

The rigid alignment runs for real on the tiles of test_stitching_steps.
"""
import pytest

from ClearMap.Alignment.Stitching import stitching_rigid, stitching_wobbly
from ClearMap.Utils.exceptions import ClearMapValueError

from .test_stitching_steps import graph_library_in_python, stitching  # noqa: F401 (fixtures)


@pytest.fixture()
def rigidly_aligned_layout(stitching):
    stitching.align_channel_rigid('main')
    return stitching_rigid.load_layout(stitching.get_path('layout', channel='main', asset_sub_type='aligned_axis'))


def test_the_rigid_alignment_moved_the_tiles_along_z(rigidly_aligned_layout):
    assert len({source.position[rigidly_aligned_layout.axis] for source in rigidly_aligned_layout.sources}) > 1


def test_placing_a_layout_which_is_not_aligned_wobbly_names_the_tiles_and_the_missing_alignment(rigidly_aligned_layout):
    with pytest.raises(ClearMapValueError, match=r'tiles \(0,\) and \(1,\) do not fit their overlap along axis 2: '
                                                 r'24 displacements .* The wobbly alignment of this layout did not run'):
        stitching_wobbly.place_layout(rigidly_aligned_layout, min_quality=-float('inf'), processes=None)
