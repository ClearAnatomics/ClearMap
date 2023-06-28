
import os
import numpy as np

import ClearMap.IO.IO as io

import ClearMap.Analysis.Graphs.GraphGt as ggt;
import ClearMap.Visualization.Plot3d as p3d
import ClearMap.ImageProcessing.Topology.Topology3d as t3d

#TODO: cleanup LargeData -> ArrayProcessing
import ClearMap.ParallelProcessing.DataProcessing.LargeData as ld


import ClearMap.Utils.Timer as Timer;

filepath="/mnt/data_SSD_2to/190312-321Lneuron_3.swc"


def fromtestToGgt(filepath):
    f = open(filepath, "r")
    fl = f.readlines()
    n_vertices=len(fl)
    print('n_vertices ', n_vertices)
    g=ggt.Graph(n_vertices=n_vertices-1, directed=False)
    print(g)
    edges_all=np.zeros((0,2), dtype=int)
    vert=np.zeros((0,3), dtype=int)
    radii=np.zeros((0,1), dtype=int)
    for line in fl:
        line=line.split( )
        # print(line, line[0], line[-1])
        if int(line[-1])!=-1:
            edge=(int(line[0])-1, int(line[-1])-1)
            edges_all=np.vstack((edges_all, edge))
            radii=np.vstack((radii, float(line[5])))

        vert=np.vstack((vert, np.array([line[2], line[3],line[4]])))
    # print(edges_all)
    g.add_edge(edges_all)
    #radii=np.ones(edges_all.shape[0])
    print(g)
    g.set_edge_geometry(name='radii', values=radii)

    g.set_vertex_coordinates(vert)
    # g.set_vertex_radius(g.vertices,radii)
    return(g)




directory='/mnt/vol00-renier/Alba/SyGlass/190620Neuron/'
for path, subdirs, files in os.walk(directory):
    for file in files:
        # print(file)
        if '.swc' in file:
            print(file)
            g_test=fromtestToGgt(os.path.join(directory,file))
            print(g_test)
            g_test.save("/mnt/data_SSD_2to/"+file[:-4]+".gt")

filepath="/mnt/vol00-renier/Sophie/SyGlass/190312-321Lneuron (1).gt"
g_test=ggt.load(filepath)
p = p3d.plot_graph_mesh(g_test, n_tube_points=3);
# if __name__ == "__main__":
