import numpy as np
import graph_tool as gt
import ClearMap.Analysis.Graphs.GraphGt as ggt


scale=1.63
g=ggt.load('/mnt/vol00-renier/Sophie/WhiskerDeprived/data_graph_correcteduniverse.gt')
radii=g.edge_property('radii')
plt.figure()
plt.hist(radii*scale, bins=np.linspace(0, 25, 20))