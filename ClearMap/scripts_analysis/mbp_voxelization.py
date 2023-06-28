import matplotlib.pyplot as plt

import ClearMap.Analysis.Measurements.Voxelization as vox
import ClearMap.IO.IO as io
import numpy as np
import ClearMap.Analysis.Graphs.GraphGt_old as ggto
import ClearMap.Analysis.Graphs.GraphGt as ggt
# from ClearMap.OtoferliinGraphs import from_e_prop2_vprop, getRadPlanOrienttaion
import ClearMap.Alignment.Annotation as ano
template_shape=(320,528,228)
radius=5
work_dir='/data_SSD_2to/mbp/P9/'
mbp_graph=ggt.load(work_dir+'3_graph_uncrusted.gt')

coordinates=mbp_graph.vertex_property('coordinates_atlas')#*1.625/25#coordinates_atlas
v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
io.write(work_dir + '/' + 'vox_mbp'+str(radius)+'.tif', v.astype('float32'))






