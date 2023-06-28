import os
import sys
import numpy as np
import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.Analysis.Graphs.GraphRendering as gr


#vispy.io.mesh.write_mesh('/data_2to/test.obj', p.mesh_data.get_vertices(), p.mesh_data.get_faces(), p.mesh_data.get_face_normals(), None, overwrite=True)

# graph=ggt.load('/data_2to/p4/4_graph_reduced.gt')
graph=ggt.load('/data_2to/p4/4_graph_reduced_no_filopedia2.gt')

n_tube_points=5
if graph.has_edge_geometry():
    #calculate mesh
    coordinates, indices = graph.edge_geometry(name='coordinates', return_indices=True, as_list=False);
    radii = graph.edge_geometry(name='radii', return_indices=False, as_list=False);
else:
    coordinates = graph.vertex_coordinates();
    indices = graph.edge_connectivity();
    coordinates = np.vstack(coordinates[indices]);
    n_edges = len(indices);
    indices = np.array([2*np.arange(0,n_edges), 2*np.arange(1,n_edges+1)]).T;
    radii = np.ones(2*n_edges);
grid, indices, grid_colors = gr.mesh_from_edge_geometry(coordinates, radii, indices, n_tube_points=n_tube_points, edge_colors=None, processes=None);

points=grid

# points = np.load('/mnt/vol00-renier/Sophie/presentation/imageActa2VSiggPodo/iggpodoacta2_capillaries_grid_dnn.npy')
# indices = np.load('/mnt/vol00-renier/Sophie/presentation/imageActa2VSiggPodo/iggpodoacta2_capillaries_indices_dnn.npy')

points = points[:, ]

binary = False
filepath='/data_2to/p4/4_graph_reduced_no_filopedia_370_2.iv'
# filepath='/data_2to/p4/4_graph_reduced_370.iv'
# filepath = '/mnt/vol00-renier/Sophie/presentation/imageActa2VSiggPodo/iggpodoacta2_capillaries_dnn.iv'


def writegrid(pts):
    bool = True
    st = """Coordinate3 { 
               point [
               """

    if binary:
        f = open(filepath, 'a+b')
        f.write(bytearray(st))
    else:
        f = open(filepath, 'a+')
        f.write(st)

    for pt in pts:
        if not np.isnan(pt[0]) and not np.isnan(pt[1]) and not np.isnan(pt[2]):
            if bool == True:
                if binary:
                    f.write(bytearray(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2])))
                else:
                    f.write(str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]))
                bool = False
            else:
                if binary:
                    f.write(bytearray(',' + str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2])))
                else:
                    f.write(',' + str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]))
        else:
            print('nan  grid value !', str(pt[0]) + ' ' + str(pt[1]) + ' ' + str(pt[2]))
            if bool == True:
                if binary:
                    f.write(bytearray(str(0.0) + ' ' + str(0.0) + ' ' + str(0.0)))
                else:
                    f.write(str(0.0) + ' ' + str(0.0) + ' ' + str(0.0))
                bool = False
            else:
                if binary:
                    f.write(bytearray(',' + str(0.0) + ' ' + str(0.0) + ' ' + str(0.0)))
                else:
                    f.write(',' + str(0.0) + ' ' + str(0.0) + ' ' + str(0.0))
    st = """]
         }
       """
    if binary:
        f.write(bytearray(st))
    else:
        f.write(st)
    f.close()


def writeindices(inds):
    bool = True
    st = """IndexedFaceSet { 
               coordIndex [
               """

    if binary:
        f = open(filepath, 'a+b')
        f.write(bytearray(st))
    else:
        f = open(filepath, 'a+')
        f.write(st)

    for i in inds:
        if not np.isnan(i[0]) and not np.isnan(i[1]) and not np.isnan(i[2]):
            if bool == True:
                if binary:
                    f.write(bytearray(str(i[0]) + ',' + str(i[1]) + ',' + str(i[2])))
                else:
                    f.write(str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]))
                bool = False
            else:
                if binary:
                    f.write(bytearray(',-1,' + str(i[0]) + ',' + str(i[1]) + ',' + str(i[2])))
                else:
                    f.write(',-1,' + str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]))
        else:
            print('nan indices !')

    st = """]
         }
       """
    if binary:
        f.write(bytearray(st))
    else:
        f.write(st)

    f.close()


if binary:
    st = """#Inventor V2.0 ascii
            """
else:
    st = """#Inventor V2.0 ascii

        Separator {
            Material {
                diffuseColor 50 50 50
            }
            """

if binary:
    f = open(filepath, 'a+b')
    f.write(bytearray(st))
else:
    f = open(filepath, 'a+')
    f.write(st)

f.close()

writegrid(points)
writeindices(indices)

st = """
     }
     """
if binary:
    f = open(filepath, 'a+b')
    f.write(bytearray(st))
else:
    f = open(filepath, 'a+')
    f.write(st)
f.close()