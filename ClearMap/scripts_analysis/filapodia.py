import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Analysis.Graphs.GraphGt as ggt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import ClearMap.IO.IO as io
import time
from multiprocessing import Pool

signal_val=io.read('/data_2to/p4/filapodia/small_cube.npy')
graph=ggt.load('/data_2to/p4/filapodia/4_graph_reduced.gt')

signal_val=io.read('/data_2to/p4/filapodia/small_cube.npy')
graph_800=ggt.load('/data_2to/p4/no_filapodia/4_graph_reduced.gt')

####### associate each vertex from the different graph together
from numpy import linalg as LA
coord=graph.vertex_coordinates()
coord_800=graph_800.vertex_coordinates()
rad=50
matrix=np.ones(graph.n_vertices)
for i in range(graph_800.n_vertices):
    mins = coord_800[i] - np.array([rad, rad, rad])
    maxs = coord_800[i] + np.array([rad, rad, rad])
    print(i,coord_800[i] )
    close_vertex=np.asarray(extractSubGraph(coord, mins, maxs)).nonzero()[0]
    dist=[]
    for v in close_vertex:
        print(i, v)
        dist.append(LA.norm(coord_800[i]-coord[v]))

    closest=close_vertex[np.argmin(np.array(dist))]
    matrix[closest]=0

fil=graph.sub_graph(vertex_filter=matrix)
p3d.plot_graph_mesh(fil)
coordinates = fil.edge_geometry_property('coordinates')
indices = fil.edge_property('edge_geometry_indices')

p = Pool(15)
start = time.time()
length = np.array(
    [p.map(get_length, [(i, ind, coordinates) for i, ind in enumerate(indices)])])
p.close()
end = time.time()
print(end - start)
length=length[0]
indices=[0]
sum=0
geom_coord=fil.edge_geometry()
geom_radii=fil.edge_geometry_property('radii')
for gc in geom_coord:
    indices.append(sum+gc.shape[0])
    sum=sum+gc.shape[0]
    # print(sum, gc.shape[0])
# indices=[gc.shape[0] for gc in geom_coord]
mean_radii=np.asarray([np.median(geom_radii[indices[i]:indices[i+1]-1]) for i in range(len(indices)-1)])

# Calculate the point density
x=length
y=mean_radii


xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100)
plt.show()

######################################################################


geom_coord=graph.edge_geometry()
geom_radii=graph.edge_geometry_property('radii')
geom_length=graph.edge_geometry_lengths()
length=graph.edge_property('length')
coordinates = graph.edge_geometry_property('coordinates')
indices = graph.edge_property('edge_geometry_indices')

p = Pool(15)
start = time.time()
length = np.array(
    [p.map(get_length, [(i, ind, coordinates) for i, ind in enumerate(indices)])])
p.close()
end = time.time()
print(end - start)
length=length[0]

indices=[0]
sum=0
for gc in geom_coord:
    indices.append(sum+gc.shape[0])
    sum=sum+gc.shape[0]
    # print(sum, gc.shape[0])
# indices=[gc.shape[0] for gc in geom_coord]
mean_radii=np.asarray([np.median(geom_radii[indices[i]:indices[i+1]-1]) for i in range(len(indices)-1)])
graph.add_edge_property('mean_radii', mean_radii)

coordinates=graph.vertex_coordinates()
deg1=np.asarray(graph.vertex_degrees()==1).nonzero()[0]
edge_filter=np.zeros(graph.n_edges)
conn=graph.edge_connectivity()

edges_centers = np.array(
    [(coordinates[conn[i, 0]] + coordinates[conn[i, 1]]) / 2 for i in range(conn.shape[0])])
edge_signal=np.array([signal_val[e.astype(int)[0],e.astype(int)[1],e.astype(int)[2]] for e in edges_centers])

plt.figure()
plt.hist(edge_signal, bins=30)

plt.figure()
plt.scatter(length, mean_radii)


from scipy.stats import gaussian_kde

# Calculate the point density
x=length
y=mean_radii
s=edge_signal

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=100)
plt.show()

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca = PCA(n_components=2)

xys = np.vstack([x,y,s])
res=pca.fit_transform(xys.T)

# z = gaussian_kde(res)(res)

fig, ax = plt.subplots()
ax.scatter(res[:,0], res[:,1], c=z, s=100)
plt.show()

res_embedded = TSNE(n_components=2, early_exaggeration=10, learning_rate=100,init='pca', perplexity=50).fit_transform(xys.T)
fig, ax = plt.subplots()
ax.scatter(res_embedded[:,0], res_embedded[:,1], c=z, s=100)
plt.show()

import ClearMap.Visualization.Plot3d as p3d
def extractSubGraph(edges_centers, mins, maxs):
    """
    Extracts the subgraph contained in the cube between mins and maxs coordinates
          6-------7
         /|      /|
        4-------5 |
        | |     | |
        | 2-----|-3
        |/      |/
        0-------1
    """
    isOver = (edges_centers > mins).all(axis=1)
    isUnder = (edges_centers < maxs).all(axis=1)
    return np.logical_and(isOver, isUnder)


filapodia_coordinates=[150, 69, 79]
filapodia_coordinates=[53, 30, 79]
filapodia_coordinates=[155, 151, 99]

x=filapodia_coordinates[0]
y=filapodia_coordinates[1]
z=filapodia_coordinates[2]
rad=40
mins = np.array([x, y, z]) - np.array([rad, rad, rad])
maxs = np.array([x, y, z]) + np.array([rad, rad, rad])
close_edges_centers=extractSubGraph(graph_800.vertex_property('coordinates'), mins, maxs)

g2plot=graph_800.sub_graph(vertex_filter=close_edges_centers)
print(g2plot)
p3d.plot_graph_mesh(g2plot)

rad=30
mins = np.array([x, y, z]) - np.array([rad, rad, rad])
maxs = np.array([x, y, z]) + np.array([rad, rad, rad])
close_edges_centers=extractSubGraph(graph.vertex_property('coordinates'), mins, maxs)
g2plot_0=graph.sub_graph(vertex_filter=close_edges_centers)
print(g2plot_0)
p3d.plot_graph_mesh(g2plot_0)




end_edge=np.zeros(graph.n_edges)
for d in deg1:
    end_edge[np.asarray(conn[:,0]==d).nonzero()[0]]=1
    end_edge[np.asarray(conn[:,1]==d).nonzero()[0]]=1

edge_radii=graph.edge_property('mean_radii')
edge_signal=np.array([signal_val[e.astype(int)[0],e.astype(int)[1],e.astype(int)[2]] for e in edges_centers])
fil_radii=edge_radii[np.asarray(end_edge==1).nonzero()[0]]
plt.figure()
plt.hist(fil_radii)
# vertex_radii=graph.vertex_radii()









g800=ggt.load('/data_2to/p4/no_filapodia/4_graph_reduced.gt')

geom_coord=g800.edge_geometry()
geom_radii=g800.edge_geometry_property('radii')
indices=[0]
sum=0
for gc in geom_coord:
    indices.append(sum+gc.shape[0])
    sum=sum+gc.shape[0]
    # print(sum, gc.shape[0])
# indices=[gc.shape[0] for gc in geom_coord]
mean_radii=np.asarray([np.median(geom_radii[indices[i]:indices[i+1]-1]) for i in range(len(indices)-1)])
g800.add_edge_property('mean_radii', mean_radii)

g800_radii=g800.edge_property('mean_radii')
coord800=g800.vertex_coordinates()
conn800=g800.edge_connectivity()

edges_centers_800 = np.array(
    [(coord800[conn800[i, 0]] + coord800[conn800[i, 1]]) / 2 for i in range(conn800.shape[0])])
edge_signal_800=np.array([signal_val[e.astype(int)[0],e.astype(int)[1],e.astype(int)[2]] for e in edges_centers_800])










plt.figure()

sns.scatterplot(g800_radii, edge_signal_800, alpha=0.5, color='royalblue')
sns.kdeplot(g800_radii, edge_signal_800,  levels=[0.6, 1], fill=True, alpha=0.3, color='royalblue')
sns.kdeplot(g800_radii, edge_signal_800,  levels=[0.5, 0.6, 1], fill=False, alpha=0.3, color='royalblue')


sns.scatterplot(edge_radii[np.asarray(end_edge!=1).nonzero()[0]], edge_signal[np.asarray(end_edge!=1).nonzero()[0]], alpha=0.3, color='cadetblue')
sns.kdeplot(edge_radii[np.asarray(end_edge!=1).nonzero()[0]], edge_signal[np.asarray(end_edge!=1).nonzero()[0]],   levels=[0.75, 1], fill=True,alpha=0.3, color='cadetblue')
sns.kdeplot(edge_radii[np.asarray(end_edge!=1).nonzero()[0]], edge_signal[np.asarray(end_edge!=1).nonzero()[0]],   levels=[0.6, 0.7, 1], fill=False,alpha=0.3, color='cadetblue')

sns.scatterplot(fil_radii, edge_signal[np.asarray(end_edge==1).nonzero()[0]], alpha=0.3, color='indianred')
sns.kdeplot(fil_radii, edge_signal[np.asarray(end_edge==1).nonzero()[0]],  levels=[0.3, 1], fill=True, alpha=0.3, color='indianred')
sns.kdeplot(fil_radii, edge_signal[np.asarray(end_edge==1).nonzero()[0]],  levels=[0.2, 0.25, 1], fill=False, alpha=0.3, color='indianred')








plt.legend(['capillaries', 'filapodia'])

edge_filter=np.logical_and(end_edge, edge_radii<2.2)
np.sum(edge_filter)
vertex_filter=from_e_prop2_vprop(graph,edge_filter)
# vertex_filter=np.logical_and(from_e_prop2_vprop(graph,edge_filter), np.asarray(graph.vertex_degrees()==1))
np.sum(vertex_filter)
filapodia_tips_index=np.asarray(vertex_filter==1).nonzero()[0]
import graph_tool.topology as gtt

distance_matrix=np.ones((filapodia_tips_index.shape[0], filapodia_tips_index.shape[0]))
distance_matrix=100*distance_matrix
for i, ft1 in enumerate(filapodia_tips_index):
    for j, ft2 in enumerate(filapodia_tips_index):
        print(i, j)
        if ft1==ft2:
            distance_matrix[i, i]=0
        if ft1!=ft2:
            paths=gtt.all_shortest_paths(graph.base, ft1, ft2)
            c=0
            # path=np.zeros(100)
            for path in paths:
                if c==0:
                    # print(i, j, len(path))
                    path_temp=path
                    c=1

                    distance_matrix[i, j]=len(path_temp)
                    distance_matrix[j, i]=len(path_temp)

from sklearn.cluster import AgglomerativeClustering
# clustering=DBSCAN(eps=100, metric='precomputed', min_samples=15).fit(distance_matrix)
clustering = AgglomerativeClustering(n_clusters=None, linkage='complete',affinity='precomputed',distance_threshold=5).fit(distance_matrix)
np.unique(clustering.labels_).shape
labels=clustering.labels_
filapodia_ep_coord2=graph.vertex_coordinates()[filapodia_tips_index]

## quick check
for l in np.unique(labels)[0:]:
    X=np.asarray(labels==l).nonzero()[0]
    temp=[]
    for x1 in X:
        for x2 in X:
            temp.append(distance_matrix[x1, x2])
    print(temp)
###


sns.set(style = "whitegrid")
sns.despine
fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')
ax.set_frame_on(False)
ax.set_facecolor('white')
tot=0
for l in np.unique(labels)[1:]:
    X=np.asarray(labels==l).nonzero()[0]
    if X.shape[0]>3:
        ax.scatter(filapodia_ep_coord2[X,0],filapodia_ep_coord2[X,1],filapodia_ep_coord2[X,2])
        tot=tot+1
plt.show()
print(tot)









filapodia_g=graph.sub_graph(edge_filter=edge_filter)
deg0=np.asarray(filapodia_g.vertex_degrees()!=0)
filapodia_g=filapodia_g.sub_graph(vertex_filter=deg0)
p3d.plot_graph_mesh(filapodia_g)

coordinates=filapodia_g.vertex_coordinates()
filapodia_ep_coord=filapodia_g.vertex_coordinates()



clustering2 = DBSCAN(eps=15, min_samples=5).fit(filapodia_ep_coord)
labels2=clustering2.labels_




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set(style = "whitegrid")
sns.despine
fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')
ax.set_frame_on(False)
ax.set_facecolor('white')
for l in np.unique(labels2)[1:]:
    X=np.asarray(labels2==l).nonzero()[0]
    ax.scatter(filapodia_ep_coord[X,0],filapodia_ep_coord[X,1],filapodia_ep_coord[X,2])
plt.show()



rounds=3
clean_graph=graph.copy()
for r in range(rounds):
    print(r)
    # clean_graph=graph.sub_graph(edge_filter=np.logical_not(np.logical_and(end_edge, edge_radii<3.0)))
    # p3d.plot_graph_mesh(clean_graph)
    # clean_graph.save('/data_2to/p4/4_graph_reduced_no_filopedia.gt')

    end_edge=np.zeros(clean_graph.n_edges)
    conn=clean_graph.edge_connectivity()
    deg1=np.asarray(clean_graph.vertex_degrees()==1).nonzero()[0]
    for d in deg1:
        end_edge[np.asarray(conn[:,0]==d).nonzero()[0]]=1
        end_edge[np.asarray(conn[:,1]==d).nonzero()[0]]=1

    edge_radii=clean_graph.edge_property('mean_radii')
    clean_graph=clean_graph.sub_graph(edge_filter=np.logical_not(np.logical_and(end_edge, edge_radii<2.25)))

p3d.plot_graph_mesh(clean_graph)
# edge_rad=clean_graph.vertex_radii()
# filtered_clean_graph=clean_graph.sub_graph(vertex_filter=edge_rad>=3)

clean_graph.save('/data_2to/p4/4_graph_reduced_no_filopedia2.gt')

coordinates=clean_graph_2.vertex_coordinates()
points=coordinates[np.asarray(clean_graph_2.vertex_degrees()==1).nonzero()[0]]



filapodia_g=graph.sub_graph(edge_filter=end_edge)
p3d.plot_graph_mesh(filapodia_g)

edge_rad=filapodia_g.vertex_radii()
filtered_filapodia=filapodia_g.sub_graph(vertex_filter=edge_rad<3)
p3d.plot_graph_mesh(filtered_filapodia)


filapodia=np.logical_and(graph.vertex_radii()<=3, graph.vertex_degrees()==1)