import sys

import numpy as np

sys.path.insert(0, '../..')
from ClearMap.Analysis.Measurements.maxima_detection import label_representatives




def rand_shape(dim):
    return tuple(np.random.randint(15,size=(dim,)))



def rand_array(dim,high):
    size=rand_shape(dim)
    return np.random.randint(high, size=size)


def elem_assert(array):
    res = label_representatives(array)
    points = [tuple([a[k] for a in res]) for k in range(len(res[0]))]
    vals = np.array([array[point] for point in points])

    labels = np.unique(array)
    if labels.size>0 and labels[0]==0:
        labels=labels[1:]
    information = f"labeled_array {array} \n output of label_reps:{res} \n corresponding tuples {points} \n there values {vals} \n vs all the labels  {labels}"
    assert vals.size == labels.size, information
    assert np.all(labels == vals), information

def test_representatives():
    for i in range(100):
        array = rand_array(3,10)
        elem_assert(array)

#TODO: have this integrated within pytest framework

if __name__ == '__main__':
    print('testing label_representatives')
    test_representatives()