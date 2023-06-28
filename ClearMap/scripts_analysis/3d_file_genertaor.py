import ClearMap.Alignment.Annotation as ano

import ClearMap.IO.IO as io
import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti
import os
print('TEST')
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt_old as ggto
import graph_tool.centrality as gtc
print('loading...')
import numpy as np
import numexpr as ne
import graph_tool.topology as gtt
from sklearn import preprocessing
# from ClearMap.Visualization.Vispy.sbm_plot import *
import seaborn as sns
from scipy.stats import ttest_ind
import math
pi=math.pi
import pickle
import ClearMap.Settings as settings
import os
import seaborn as sns
import vispy

atlas_path = os.path.join(settings.resources_path, 'Atlas');
anot=io.read(os.path.join(atlas_path, 'annotation_25_full.nrrd'))
#create mask
annotation=np.zeros(anot.shape)

theta_= np.arange(0, 360)*pi/180
lambda_= np.arange(0, 360)*pi/180
R_range=np.arange(25, 30)

x0= 200
y0=200
z0=200

for R in R_range:
    for t in theta_:
        for l in lambda_:
            x=R*np.cos(t)*np.cos(l)
            y=R*np.sin(t)*np.cos(l)
            z=R*np.sin(l)

            annotation[int(x+x0), int(y+y0), int(z+z0)]=1

# p3d.plot(annotation)

#%% annotation
# graph=ggto.load('/data_SSD_2to/191122Otof/2R/data_graph_correcteduniverse.gt')
ano.set_annotation_file(annotation)

def annotate(coordinates):
    label = ano.label_points(coordinates, key='id');
    return label;


graph=ggto.load('/data_SSD_2to/191122Otof/2R/data_graph_correcteduniverse.gt')

graph.transform_properties(annotate,
                                  vertex_properties = {'coordinates_atlas' : 'annotation'},
                                  edge_geometry_properties = {'coordinates_atlas' : 'annotation'});

label = graph.vertex_property('annotation');
vf=label==1

sphere=graph.sub_graph(vertex_filter=vf.astype(int))
p=p3d.plot_graph_mesh(sphere.largest_component(), n_tube_points=5)

vispy.io.mesh.write_mesh('/data_2to/sphere.obj', p.mesh_data.get_vertices(), p.mesh_data.get_faces(), p.mesh_data.get_face_normals(), None, overwrite=True)