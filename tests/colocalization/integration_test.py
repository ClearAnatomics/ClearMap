import sys

import numpy as np
import pandas as pd
import skimage
import skimage.measure

sys.path.insert(0, "ClearMap2")

from ClearMap.colocalization import Channel
from ClearMap.colocalization import _naive_bilabel_bincount

from ClearMap.IO.IO import source

shape = (20, 20)
binary_1 = np.zeros(shape, dtype="uint8")
binary_2 = np.zeros(shape, dtype="uint8")

binary_1[3:19, 2:18] = 1


binary_2[:10, :11] = 1
binary_2[:10, 14:] = 1

# binary_1 = source(binary_1)
# binary_2 = source(binary_2)

reps_1 = [(3, 17)]
reps_2 = [(0, 19), (3, 3)]


# hand made computations yield
intersection_size_2 = (10 - 3) * (11 - 2)
intersection_size_1 = (10 - 3) * (18 - 14)

single_rate_1 = intersection_size_1 / ((19 - 3) * (18 - 2))
single_rate_2 = intersection_size_2 / ((19 - 3) * (18 - 2))

global_rate = single_rate_1 + single_rate_2

# We create the two Channel instances
df1 = pd.DataFrame(reps_1, columns=["x", "y"])
df2 = pd.DataFrame(reps_2, columns=["x", "y"])

channel_1 = Channel(binary_1, df1, coord_names=["x", "y"])
channel_2 = Channel(binary_2, df2, coord_names=["x", "y"])

# check centers
# print("channel 1 centers", channel_1.centers)
# print(
#     "channel 1 centers as returned by self.center",
#     *[channel_1.center(index) for index in range(len(channel_1.dataframe))]
# )

# print("channel 2 centers", channel_2.centers)
# print(
#     "channel 2 centers as returned by self.center",
#     *[channel_2.center(index) for index in range(len(channel_2.dataframe))]
# )
assert [channel_1.center(index) for index in range(len(channel_1.dataframe))] == [
    (10.5, 9.5)
]

assert [channel_2.center(index) for index in range(len(channel_2.dataframe))] == [
    (4.5, 16.5),
    (4.5, 5.0),
]

# # check index_label correspondance
# print("channel1 index ot label", channel_1.index_label_correspondance)
# print("channel2 index ot label", channel_2.index_label_correspondance)


# check overlap computations
assert np.allclose(
    channel_1.overlap_rates(channel_2), np.array([global_rate]), atol=1e-05
)


# print(
#     "single_blob..., single_rates:",
#     channel_1.single_blob_overlap_rates(channel_2),
#     np.array([single_rate_1, single_rate_2]),
# )
assert np.allclose(
    channel_1.single_blob_overlap_rates(channel_2, return_max_indices=False),
    np.max(np.array([single_rate_1, single_rate_2])),
    atol=1e-05,
)


# check distances
# print("distances", channel_1.centers_distances(channel_2))
assert np.allclose(
    np.array([[9.219544, 7.5]]), channel_1.centers_distances(channel_2), atol=1e-7
)


max_rates, max_indices = channel_1.single_blob_overlap_rates(
    channel_2, return_max_indices=True
)
assert np.allclose(
    max_rates, np.max(np.array([single_rate_1, single_rate_2])), atol=1e-05
)
assert max_indices == np.array([1])
print("test passed !")
