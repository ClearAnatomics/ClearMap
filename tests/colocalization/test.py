import sys

import numpy as np

sys.path.insert(0, "../..")
from ClearMap.Analysis.Measurements.maxima_detection import label_representatives
from ClearMap.colocalization import bilabel_bincount
from ClearMap.colocalization import distances


### for label_representatives


def rand_shape(dim):
    return tuple(np.random.randint(15, size=(dim,)))


def rand_array(dim, high):
    size = rand_shape(dim)
    return np.random.randint(high, size=size)


def elem_assert(array):
    res = label_representatives(array)
    points = [tuple([a[k] for a in res]) for k in range(len(res[0]))]
    vals = np.array([array[point] for point in points])

    labels = np.unique(array)
    if labels.size > 0 and labels[0] == 0:
        labels = labels[1:]
    information = f"labeled_array {array} \n output of label_reps:{res} \n corresponding tuples {points} \n there values {vals} \n vs all the labels  {labels}"
    assert vals.size == labels.size, information
    assert np.all(labels == vals), information


def test_representatives():
    for i in range(100):
        array = rand_array(3, 10)
        elem_assert(array)
        elem_assert(array.transpose())


### for bilabel_bin_count


def test_bilabel_bincount():
    for u in range(100):
        A = np.random.randint(5, size=(20, 20))
        B = np.random.randint(5, size=(20, 20))
        counts = bilabel_bincount(A, B)
        m = A.max() + 1
        n = B.max() + 1
        res = np.zeros((m, n), dtype="uint8")
        for i in range(m):
            for j in range(n):
                res[i, j] = np.count_nonzero((A == i) * (B == j))
        assert np.all(res == counts)


### for distances


def test_distances():
    for i in range(100):
        dim = np.random.randint(1, 10)
        A = np.random.randint(10, size=(30, dim))
        B = np.random.randint(10, size=(30, dim))
        res = np.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                res[i, j] = (np.sum((A[i] - B[j]) ** 2)) ** 0.5
        assert np.all(res == distances(A, B))


# TODO: have this integrated within pytest framework

if __name__ == "__main__":
    print("testing label_representatives")
    test_representatives()
    print("testing bilabel_bincount")
    test_bilabel_bincount()
    print("testing distances")
    test_distances()
    print("The tests were successful!")
