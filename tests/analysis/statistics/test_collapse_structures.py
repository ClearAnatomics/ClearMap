"""
``collapse_structures()`` merges the counts of the structures mapped to the same id, per hemisphere or for both.

With pandas 2 (needed by NumPy 2), summing the groups also sums the strings of the hemisphere column: collapsing the
hemispheres gave a column of 'LHRH' where pandas 1 dropped it.
"""
import pandas as pd

from ClearMap.Analysis.Statistics.data_frame_operations import collapse_structures

CHILD_ID, PARENT_ID, OTHER_ID = 2, 1, 3
SAMPLE = 'sample_1'


def _counts():
    return pd.DataFrame({'id': [PARENT_ID, CHILD_ID, OTHER_ID], 'hemisphere': ['LH', 'RH', 'LH'], SAMPLE: [1, 2, 3]})


def test_collapsing_the_hemispheres_sums_the_counts_per_structure_without_a_hemisphere_column():
    collapsed = collapse_structures(_counts(), {CHILD_ID: PARENT_ID}, collapse_hemispheres=True)

    assert list(collapsed.columns) == ['id', SAMPLE]
    assert collapsed.set_index('id')[SAMPLE].to_dict() == {PARENT_ID: 3, OTHER_ID: 3}


def test_collapsing_per_hemisphere_keeps_one_row_per_structure_and_hemisphere():
    collapsed = collapse_structures(_counts(), {CHILD_ID: PARENT_ID})

    assert collapsed.set_index(['id', 'hemisphere'])[SAMPLE].to_dict() == {
        (PARENT_ID, 'LH'): 1, (PARENT_ID, 'RH'): 2, (OTHER_ID, 'LH'): 3}
