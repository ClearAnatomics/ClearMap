"""
The names of the coordinate columns of a colocalization channel, given as a tuple (the default) or as a list.

A tuple is a single key for pandas (``df[('x', 'y', 'z')]`` looks for one column named so), and a tuple is never equal
to a list: the names are kept as a list, whatever they were given as.
"""
import numpy as np
import pandas as pd

from ClearMap.Analysis.colocalization.colocalization_channel_mask_components import (
    ColocalizationChannelMaskComponents as Channel)

COORDINATE_NAMES = ['x', 'y', 'z']


def _channel(**kwargs):
    binary = np.zeros((6, 6, 6), dtype=bool)
    binary[1:3, 1:3, 1:3] = True
    binary[4:6, 4:6, 4:6] = True
    dataframe = pd.DataFrame({'x': [1, 4], 'y': [1, 4], 'z': [1, 4]})
    return Channel(binary, dataframe, **kwargs)


def test_the_default_coordinate_names_select_the_columns_of_the_representative_points():
    channel = _channel()

    assert channel.representative_points.to_numpy().tolist() == [[1, 1, 1], [4, 4, 4]]


def test_the_default_coordinate_names_give_the_labels_of_the_representative_points():
    channel = _channel()

    assert len(channel.index_label_correspondence) == 2


def test_the_coordinate_names_are_a_list_whether_they_are_given_as_a_tuple_or_a_list():
    from_a_tuple = _channel(coord_names=('x', 'y', 'z'))
    from_a_list = _channel(coord_names=COORDINATE_NAMES)

    assert from_a_tuple.coord_names == COORDINATE_NAMES
    assert from_a_list.coord_names == COORDINATE_NAMES
    assert from_a_tuple.coord_names == from_a_list.coord_names  # compare() refuses channels with different names
