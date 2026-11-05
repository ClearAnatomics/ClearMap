"""
``Source.offset`` is the distance in bytes between the data of a source and the start of the buffer it is a view of.

It was computed with ``np.byte_bounds``, which NumPy 2 moved to ``np.lib.array_utils`` (AttributeError for any source
over a view of another array, a memory map opened with an offset for instance).
"""
import numpy as np

from ClearMap.IO.source.Source import ArraySource

N_ITEMS = 10
FIRST_ITEM = 3


class _ViewSource(ArraySource):
    """An array source over an in-memory array"""

    def __init__(self, array):
        super().__init__()
        self._array = array

    @property
    def array(self):
        return self._array


def test_the_offset_of_a_source_over_a_view_is_the_distance_to_the_start_of_its_base():
    base = np.arange(N_ITEMS, dtype=np.int32)

    assert _ViewSource(base[FIRST_ITEM:]).offset == FIRST_ITEM * base.itemsize


def test_the_offset_of_a_source_over_a_reversed_view_is_the_distance_to_its_lowest_address():
    # The data of base[::-1] starts at the last item, the buffer of the view ends at the first one
    base = np.arange(N_ITEMS, dtype=np.int32)

    assert _ViewSource(base[::-1]).offset == 0


def test_the_offset_of_a_source_over_an_array_that_owns_its_data_is_zero():
    assert _ViewSource(np.arange(N_ITEMS)).offset == 0
