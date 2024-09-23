import sys

# TODO set automatically the correct path
sys.path.insert(0, "ClearMap2")

import numpy as np
import pandas as pd
import skimage.morphology
from scipy.spatial.transform import Rotation
import scipy.ndimage as ndi
import ClearMap.colocalization.channel as channel


def random_shape(radius):
    dice = np.random.randint(3)
    if dice == 0:
        print("cube")
        shape = skimage.morphology.cube(2 * int(radius / 1.8) + 1)
    if dice == 1:
        print("octa")
        shape = skimage.morphology.octahedron(radius)
    if dice == 2:
        print("ball")
        shape = skimage.morphology.ball(radius)

    locus = np.where(shape)
    points = np.vstack(locus).transpose()

    rand_rot = Rotation.random()
    new_width = np.ceil((2 * radius + 1) * 1.8).astype("uint8") + 3
    new_center = np.array([new_width // 2] * 3)

    centered = points - np.array([radius] * 3)
    new_centered = rand_rot.apply(centered)
    new_locus = (np.round(new_centered + new_center).astype("uint64")).transpose()
    xs, ys, zs = new_locus[0], new_locus[1], new_locus[2]

    result = np.zeros((new_width,) * 3, dtype="bool")
    result[xs, ys, zs] = 1

    return ndi.binary_closing(result, structure=np.ones((3,) * 3)), new_center


def random_blobs(shape, num_points, min_radius, max_radius):
    if len(shape) != 3:
        raise ValueError("Only 3dim shapes are accepted")
    radii = np.random.randint(min_radius, max_radius, size=(num_points,))
    centers = np.vstack([np.random.randint(shape[i] - 1, size=(num_points,)) for i in range(3)]).transpose()
    margin = 2 * max_radius
    centers += margin
    result = np.zeros(tuple([size + 2 * margin for size in shape]), dtype="bool")
    for i in range(num_points):
        footprint, fp_center = random_shape(radii[i])
        locus = np.vstack(np.where(footprint)).transpose()
        locus = locus - np.array(fp_center) + centers[i]
        locus = locus.transpose()
        xs, ys, zs = locus[0], locus[1], locus[2]
        result[xs, ys, zs] = 1

    return result[margin:-margin, margin:-margin, margin:-margin]


def random_channel(shape, num_points, min_radius, max_radius):
    img = random_blobs(shape, num_points, min_radius, max_radius)
    labels, _ = ndi.label(img)
    vals, indices = np.unique(labels, return_index=True)
    indices = indices[np.where(vals)]
    representatives = np.unravel_index(indices, labels.shape)
    return labels, representatives


labeled_0, reps_0 = random_channel((200, 200, 200), 500, 5, 10)
labeled_1, reps_1 = random_channel((200, 200, 200), 500, 5, 10)
df_0 = pd.DataFrame({c: reps_0[i] for i, c in enumerate("xyz")})
df_1 = pd.DataFrame({c: reps_1[i] for i, c in enumerate("xyz")})

channel_0 = channel.Channel(labeled_0 > 0, df_0)
channel_1 = channel.Channel(labeled_1 > 0, df_1)
