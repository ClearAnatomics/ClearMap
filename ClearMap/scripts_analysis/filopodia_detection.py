import graph_tool.topology as gtt
import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.IO.IO as io
from multiprocessing import Pool
atlas_path = os.path.join(settings.resources_path, 'Atlas');
import time

def get_length(args):
    i, ind ,coordinates=args
    # print(i)
    diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
    ll=np.sum(np.linalg.norm(diff, axis=1))
    #0.000025

    return ll


signal_val=io.read('/data_2to/p4/filapodia/small_cube.npy')
graph=ggt.load('/data_2to/p4/filapodia/4_graph_reduced.gt')
labels=graph.label_components()
graph=graph.sub_graph(vertex_filter=labels==2)

signal_val=io.read('/data_2to/p4/filapodia/small_cube.npy')
graph_800=ggt.load('/data_2to/p4/no_filapodia/4_graph_reduced.gt')
labels=graph_800.label_components()
graph_800=graph_800.sub_graph(vertex_filter=labels==1)

deg1=np.asarray(graph.vertex_degrees()==1).nonzero()[0]
deg1_800=np.asarray(graph_800.vertex_degrees()==1).nonzero()[0]

coord=graph.vertex_coordinates()
coordinates = graph.edge_geometry_property('coordinates')
indices = graph.edge_property('edge_geometry_indices')

coord_800=graph_800.vertex_coordinates()
rad=50

p = Pool(15)
start = time.time()
length = np.array(
    [p.map(get_length, [(i, ind, coordinates) for i, ind in enumerate(indices)])])
p.close()
end = time.time()
print(end - start)
length=length[0]

graph.add_edge_property('length', length)

matrix=np.ones((np.shape(deg1)[0],np.shape(deg1)[0]))
for il, i in enumerate(deg1):
    for jl, j in enumerate(deg1):
        dist=gtt.shortest_distance(graph.base, weights=graph.base.ep.length,  source=i, target= j)
        print(i, j, dist)
        matrix[il, jl]=dist

from sklearn.cluster import DBSCAN
X = matrix.copy()
X=np.nan_to_num(X, neginf=10000)
clustering = DBSCAN(eps=45, metric='precomputed', min_samples=1).fit(X)
unique_labels=np.unique(clustering.labels_)
# np.unique(clustering.labels_, return_counts=True)

centers=[]
for u in unique_labels[1:]:
    print(u)
    points=np.asarray(clustering.labels_==u).nonzero()[0]
    coo=coord[deg1[points]]

    centers.append(np.mean(coo, axis=0))
centers=np.array(centers)

print(coord_800[deg1_800].shape)
print(centers.shape)

end_point_800=coord_800[deg1_800].copy()
end_point=centers.copy()

import scipy.spatial
tree=scipy.spatial.KDTree(end_point)
assoc=[]
for I1,point in enumerate(end_point_800):
    I2 = tree.query(point,k=1,distance_upper_bound=30)
    assoc.append((I1,I2[1], I2[0]))

assoc_list=np.array(assoc)[np.array(assoc)[:,2]!=np.inf]
assoc_list=assoc_list[:, :-1].astype(int)

associated_pt=[((end_point_800[assoc_list[n, 0]]+end_point[assoc_list[n, 1]])/2).tolist() for n in range(assoc_list.shape[0])]
associated_pt=np.array(associated_pt)


import seaborn as sns
sns.set(style = "whitegrid")
sns.despine()
fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')
ax.set_frame_on(False)
ax.set_facecolor('white')
ax.scatter(centers[:,0],centers[:,1],centers[:,2])
# plt.show()
#
ax.scatter(coord_800[deg1_800][:,0],coord_800[deg1_800][:,1],coord_800[deg1_800][:,2])

ax.scatter(associated_pt[:,0],associated_pt[:,1],associated_pt[:,2], c='red', s=200, alpha=0.3)



### un correlated 800 end points
end_point_800[10]#array([63., 57.,  5.])
end_point_800[12]#array([61., 60.,  6.])

pb_coordinates=end_point_800[29]

x=pb_coordinates[0]
y=pb_coordinates[1]
z=pb_coordinates[2]
rad=35
mins = np.array([x, y, z]) - np.array([rad, rad, rad])
maxs = np.array([x, y, z]) + np.array([rad, rad, rad])
close_edges_centers=extractSubGraph(graph_800.vertex_property('coordinates'), mins, maxs)

g2plot=graph_800.sub_graph(vertex_filter=close_edges_centers)
print(g2plot)
p3d.plot_graph_mesh(g2plot)

close_edges_centers=extractSubGraph(graph.vertex_property('coordinates'), mins, maxs)
g2plot=graph.sub_graph(vertex_filter=close_edges_centers)
print(g2plot)
p3d.plot_graph_mesh(g2plot)