import sys

# TODO set automatically the correct path
sys.path.insert(0, "ClearMap2")

import numpy as np
import ClearMap.colocalization.channel as channel

import pickle

READ = True

if not READ:

    print("Generating random channels")
    channel_0 = channel.random_channel((200, 200, 200), 500, 5, 10)
    channel_1 = channel.random_channel((200, 200, 200), 500, 5, 10)
    with open("/home/gael.cousin/Documents/my_code/colocalization/ClearMap2/tests/colocalization/c0.pickle", "wb") as f:
        pickle.dump(channel_0, f)

    with open("/home/gael.cousin/Documents/my_code/colocalization/ClearMap2/tests/colocalization/c1.pickle", "wb") as f:
        pickle.dump(channel_1, f)
    print("Random channels generation done.")

else:
    with open("/home/gael.cousin/Documents/my_code/colocalization/ClearMap2/tests/colocalization/c0.pickle", "rb") as f:
        channel_0 = pickle.load(f)
        channel_0.verbose = True

    with open("/home/gael.cousin/Documents/my_code/colocalization/ClearMap2/tests/colocalization/c1.pickle", "rb") as f:
        channel_1 = pickle.load(f)
        channel_1.verbose = True

res_naive = channel_0._naive_compare(channel_1)
print("naive computation done")
res_parallel = channel_0.compare(channel_1, blob_diameter=22, size_min=50, size_max=50, processes=None)


# distance & overlap check
equal_cols = ["max relative blobwise overlap", "closest blob distance"]
assert np.allclose(res_naive[equal_cols].to_numpy(), res_parallel[equal_cols].to_numpy(), atol=10 ** (-4))

# indices check
# even though the min dist (or max overlap) can be realized by two distinct points we check the options
# chosen in res_parallel are correct (coord and corresponding distance)

# for distance
min_dist_index = res_parallel[["closest blob bbox center index"]].to_numpy().flatten()
min_dist_coords = res_parallel[["closest blob center x", "closest blob center y", "closest blob center z"]].to_numpy()
min_dist_vals = res_parallel[["closest blob distance"]].to_numpy()
c1_centers = channel_1.centers_df()

assert np.allclose(c1_centers.iloc[min_dist_index].to_numpy(), min_dist_coords, atol=10 ** (-3))

assert np.allclose(
    channel_0.centers_distances(channel_1)[np.arange(min_dist_index.size), min_dist_index].flatten(),
    min_dist_vals.flatten(),
    atol=10 ** (-3),
)

# for overlap


max_overlap_index = res_parallel[["index of maximizing overlap blob"]].to_numpy().flatten()
max_overlap_coords = res_parallel[
    ["maximizing blob bbox center x", "maximizing blob bbox center y", "maximizing blob bbox center z"]
].to_numpy()
max_overlap_vals = res_parallel[["max blobwise overlap (in voxels)"]].to_numpy()

assert np.allclose(c1_centers.iloc[max_overlap_index].to_numpy(), max_overlap_coords, atol=10 ** (-3))

assert np.allclose(
    channel_0.blobwise_overlaps(channel_1)[np.arange(max_overlap_index.size), max_overlap_index].flatten(),
    max_overlap_vals.flatten(),
    atol=10 ** (-3),
)


print("test passed!")


# res_parallel.to_csv('par_df.csv', index=False)
# res_naive.to_csv('seq_df.csv', index=False)
