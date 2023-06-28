import matplotlib.pyplot as plt

import ClearMap.Analysis.Measurements.Voxelization as vox
import ClearMap.IO.IO as io
import numpy as np
import ClearMap.Analysis.Graphs.GraphGt_old as ggto
import ClearMap.Analysis.Graphs.GraphGt as ggt
# from ClearMap.OtoferliinGraphs import from_e_prop2_vprop, getRadPlanOrienttaion
import ClearMap.Alignment.Annotation as ano
import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt

atlas_path = os.path.join(settings.resources_path, 'Atlas');

work_dirP0='/data_2to/p0/new'
controlsP0=['0a', '0b', '0c', '0d']#['2', '3']

work_dirP5='/data_2to/p5'
controlsP5=['5a', '5b']#['2', '3']

work_dirP1='/data_2to/p1'
controlsP1=['1a', '1b', '1d']#['2', '3']


work_dirP3='/data_2to/p3'
controlsP3=['3a', '3b', '3c', '3d']#['2', '3']

work_dirP7='/data_2to/p7'
controlsP7=['7a', '7b']#['2', '3']

work_dirAdult='/data_2to/earlyDep_ipsi'
controlsAdult=['4', '7', '10', '15']

work_dirP6='/data_2to/p6'
controlsP6=['6a']#['2', '3']

work_dirP14='/data_2to/p14'
controlsP14=['14a', '14b', '14c']#['2', '3']


reg_name=['cortex', 'striatum', 'hippocampus']

reg_name=['brain']
TP=[0, 1, 5, 30, 3, 7, 6, 14]

C=['P0', 'P1', 'P3', 'P5', 'P6', 'P7', 'P14']
workdirs=[work_dirP0, work_dirP1, work_dirP3,work_dirP5, work_dirP6,work_dirP7,work_dirP14]#,work_dirAdult]
controlslist=[controlsP0, controlsP1, controlsP3, controlsP5, controlsP6,controlsP7,controlsP14]#,controlsAdult]

for i, controls in enumerate(controlslist):
    workdir=workdirs[i]

    print(C[i] + ':')
    diff_t_X=[]
    diff_X=[]

    diff_t_Y=[]
    diff_Y=[]

    diff_t_Z=[]
    diff_Z=[]

    for j, control in enumerate(controls):
        G = ggt.load(workdir + '/' + control + '/' + str(control)+'_graph_correcteduniverse.gt')

        xmin=np.min(G.vertex_coordinates()[:, 0])
        xmax=np.max(G.vertex_coordinates()[:, 0])

        xmin_t=np.min(G.vertex_property('coordinates_atlas')[:, 0])
        xmax_t=np.max(G.vertex_property('coordinates_atlas')[:, 0])


        print(control+':', xmax-xmin, xmax_t-xmin_t)
        diff_t_X.append(xmax_t-xmin_t)
        diff_X.append(xmax-xmin)

        xmin=np.min(G.vertex_coordinates()[:, 1])
        xmax=np.max(G.vertex_coordinates()[:, 1])

        xmin_t=np.min(G.vertex_property('coordinates_atlas')[:, 1])
        xmax_t=np.max(G.vertex_property('coordinates_atlas')[:, 1])


        print(control+':', xmax-xmin, xmax_t-xmin_t)
        diff_t_Y.append(xmax_t-xmin_t)
        diff_Y.append(xmax-xmin)

        xmin=np.min(G.vertex_coordinates()[:, 2])
        xmax=np.max(G.vertex_coordinates()[:, 2])

        xmin_t=np.min(G.vertex_property('coordinates_atlas')[:, 2])
        xmax_t=np.max(G.vertex_property('coordinates_atlas')[:, 2])


        print(control+':', xmax-xmin, xmax_t-xmin_t)
        diff_t_Z.append(xmax_t-xmin_t)
        diff_Z.append(xmax-xmin)
    print(C[i],'X :' , np.mean(np.asarray(diff_X))*1.625/np.mean(np.asarray(diff_t_X)))
    print(C[i],'Y :' , np.mean(np.asarray(diff_Y))*1.625/np.mean(np.asarray(diff_t_Y)))
    print(C[i],'Z :' , np.mean(np.asarray(diff_Z))*2/np.mean(np.asarray(diff_t_Z)))





