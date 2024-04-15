import sys

import graph_tool as gt
import numpy as np

from ClearMap.External import pickle_python_3 as pickle


def pickler(stream, obj):
    sstream = gt.gt_io.BytesIO()
    pickle.dump(obj, sstream)  # ,  gt.gt_io.GT_PICKLE_PROTOCOL)
    stream.write(sstream.getvalue())


def unpickler(stream):
    data = stream.read(buflen=2**31)
    # print('unpickler loaded %d' % len(data))
    sstream = gt.gt_io.BytesIO(data)
    if sys.version_info < (3,):
        return pickle.load(sstream)
    return pickle.load(sstream, encoding="bytes")


def edges_to_connectivity(edges):
    return np.array([[int(e.source()), int(e.target())] for e in edges])
    # return np.array([[e.source(), e.target()] for e in edges], dtype=int)  # TODO: see which is better cpu/RAM wise
