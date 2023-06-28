import ClearMap.Alignment.Annotation as ano
import ClearMap.IO.IO as io
import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti
import os

from ClearMap.Visualization.Vispy.PlotGraph3d import plot_graph_mesh

print('TEST')
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt
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
from sklearn.linear_model import LinearRegression

import itertools
import math
import vispy
import ClearMap.Visualization.TurntableCamera as tc
from ClearMap.Visualization.visualUtils import *

# Interpolation functions in NumPy.
def _mix_simple(a, b, x):
    """Mix b (with proportion x) with a."""
    x = np.clip(x, 0.0, 1.0)
    return (1.0 - x)*a + x*b

class graysAlpha(vispy.color.colormap.BaseColormap):
    glsl_map = """
    vec4 grays(float t) {
        return vec4(t, t, t, 0.075 * t);
    }
    """
    def map(self, t):
        if isinstance(t, np.ndarray):
            return np.hstack([t, t, t, 0.075 * t]).astype(np.float32)
        else:
            return np.array([t, t, t, 0.075 * t], dtype=np.float32)



class FireMap(vispy.color.colormap.BaseColormap):
    colors = [(1.0, 1.0, 1.0, 0.0),
              (1.0, 1.0, 0.0, 0.05),
              (1.0, 0.0, 0.0, 0.1)]
    glsl_map = """
    vec4 fire(float t) {
        return mix(mix($color_0, $color_1, t),
                   mix($color_1, $color_2, t*t), t);
    }
    """
    def map(self, t):
        a, b, d = self.colors.rgba
        c = _mix_simple(a, b, t)
        e = _mix_simple(b, d, t**2)
        return _mix_simple(c, e, t)



def plot3d(data, colormap = graysAlpha(), view=None):
    VolumePlot3D = vispy.scene.visuals.create_visual_node(vispy.visuals.VolumeVisual)
    # Add a ViewBox to let the user zoom/rotate
    # build canvas
    if view is None:
        canvas = vispy.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)
        view = canvas.central_widget.add_view(camera=tc.TurntableCamera())
        view.camera = 'turntable'
        view.camera.fov = 0
        view.camera.distance = 7200
        view.camera.elevation = 31
        view.camera.azimuth = 0
        view.camera.depth_value = 100000000
        cc = (np.array(data.shape) // 2)
        view.camera.center = cc
    return VolumePlot3D(data.transpose([2, 1, 0]), method='translucent', relative_step_size=1.5,
                        parent=view.scene, cmap=colormap)



def extractSubGraph(graph, mins, maxs):
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
    isOver = (graph.vertex_coordinates() > mins).all(axis=1)
    isUnder = (graph.vertex_coordinates() < maxs).all(axis=1)
    return graph.sub_graph(vertex_filter=np.logical_and(isOver, isUnder))



def extract_AnnotatedRegion(graph, region, state='dev'):
    order, level = region
    print(state, level, order, ano.find(order, key='order')['name'])

    label = graph.vertex_annotation();
    if state=='dev':
        print('dev')
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter = label_leveled == order;
    elif state=='adult':
        print('adult')
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter = label_leveled == order;

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    return gss4

source='/data_2to/201120/1L/201120-1L_stitched_vasc.tif'
import tifffile
# image=tifffile.imread(source)
image=io.read(source)

graph=ggt.load('/data_2to/201120/1L/data_graph_correcteduniverse.gt')
region=(ano.find(101)['order'],ano.find(101)['level'])
g=extract_AnnotatedRegion(graph, region, 'adult')

work_dir='/data_2to/p3'
control='3a'
graph=ggt.load(work_dir+'/'+control+'/'+control+'_graph.gt')
region=(ano.find(16001)['order'],ano.find(16001)['level'])
g=extract_AnnotatedRegion(graph, region, 'dev')

degrees_m=g.vertex_degrees()
deg4=np.asarray(degrees_m==1).nonzero()[0]
conn = g.edge_connectivity()


d=deg4[6]
center=g.vertex_coordinates()[d]
s=105#25
s2=15
mins=center-np.array([s, s, s])
maxs=center+np.array([s, s, s])
import ClearMap.IO.IO as io
patch = extractSubGraph(g, mins=mins, maxs=maxs)
p3d.plot_graph_mesh(patch)

impatch=image[int(center[0])-s2:int(center[0])+s2, int(center[1])-s2:int(center[1])+s2, int(center[2])-s2:int(center[2])+s2]
# impatch = image.patch(center[0], center[1], center[2], size=(40, 40, 40))

# vb1, vb2 = get_two_views()
view = get_view()
# center=graph.vertex_coordinates()[d]
# center = im_view_center((2800., 1000., 1600.))
# center=(center[0], center[1], center[2])
# plotVessels(patch, radiusScaling=600, view=vb1,  color_map=get_radius_color_map(transparency=0))#center=center,
# plot3d(impatch, colormap=FireMap(), view=vb1)
plot3d(impatch, colormap=FireMap(), view=view)
show()


####





#%% brain regions
import ClearMap.Settings as settings
import os
atlas_path = os.path.join(settings.resources_path, 'Atlas');

ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
               extra_label = None, annotation_file = os.path.join(atlas_path, 'annotation_25_full.nrrd'))


ano.initialize(label_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',
               extra_label = None, annotation_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotationADMBA_thresholded.tif')





import ClearMap.Analysis.Graphs.GraphGt_old as ggto
work_dir='/data_2to/p6'
g='6a'
grt = ggto.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')
# grt = ggto.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse_subregion.gt')
#
work_dir='/data_2to/earlyDep_ipsi'
g='7'

work_dir='/data_SSD_2to/fluoxetine2'
g='10'
grt = ggto.load(work_dir + '/' + g + '/' + 'data_graph.gt')









gs = grt.sub_slice((slice(1,300), slice(50,480), slice(180,210)),coordinates='coordinates_atlas');
gs = grt.sub_slice((slice(1,1000), slice(170,210), slice(1,1000)),coordinates='coordinates_atlas');


# edge_vein_label = gs.edge_property('vein');
edge_artery_label = gs.edge_property('artery')

# edge_filter=np.logical_or(edge_vein_label,edge_artery_label)
edge_filter=edge_artery_label.astype(bool)
gsrt = gs.sub_graph(edge_filter=edge_filter)

# edge_vein_label = gsrt.edge_property('vein');
edge_artery_label = gsrt.edge_property('artery')

vertex_colors = ano.convert_label(gsrt.vertex_annotation(), key='order', value='rgba');
#
connectivity = gsrt.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
edge_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
# edge_colors[edge_vein_label>0] = [0.0,0.0,0.8,1.0]
print('plotting...')
p = p3d.plot_graph_mesh(gsrt, edge_colors=edge_colors, n_tube_points=5);














artery=from_e_prop2_vprop(grt, 'artery')
gs=grt.sub_graph(vertex_filter=artery.astype(bool))
gs=extract_AnnotatedRegion(gs, (52,9), state='adult')
p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3)
# Auditory coronal
# gs = grt.sub_slice((slice(1,270), slice(200,210), slice(1,240)),coordinates='coordinates_atlas');
# gs = grt.sub_slice((slice(1,300), slice(50,480), slice(130,140)),coordinates='coordinates_atlas');


gs = grt.sub_slice((slice(1,300), slice(50,480), slice(60,80)),coordinates='coordinates_atlas');
gs = grt.sub_slice((slice(1,1000), slice(170,190), slice(1,1000)),coordinates='coordinates_atlas');

vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');

connectivity = gs.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
# edge_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
# edge_colors[edge_vein_label  >0] = [0.0,0.0,0.8,1.0]

p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3)


work_dir='/data_2to/p7'
control='7a'
try:
    graph = ggt.load(work_dir + '/' + control + '/' + str(control) + '_graph_correcteduniverse_subregion.gt')
except:
    graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse_subregion.gt')
try:
    artery=from_e_prop2_vprop(graph, 'artery')
except:
    artery=from_e_prop2_vprop(graph, 'artery_binary')
gs=graph.sub_graph(vertex_filter=artery.astype(bool))
gs=extract_AnnotatedRegion(gs, (452,10), state='dev')

# gs = gs.sub_slice((slice(1,1000), slice(100,190), slice(1,1000)),coordinates='coordinates_atlas');

p = p3d.plot_graph_mesh(gs,n_tube_points=3)



label = graph.vertex_annotation();
region=(450, 10)
state='dev'
order, level = region
print(state, level, order, ano.find(order, key='order')['name'])
label = graph.vertex_annotation();
if state=='dev':
    print('dev')
    label_leveled = ano.convert_label(label, key='order', value='id', level=level)
    vertex_filter = label_leveled == order;
gss4 = graph.sub_graph(vertex_filter=vertex_filter)















###
# data=np.load('/data_2to/small_cubes_new_segmentation_method/debug_stitched-striatum_1500-1700_4600-4800_1200-1400.npy')
# data=np.load('/data_2to/small_cubes_new_segmentation_method/debug_stitched_thalamus_2400-2600_4000-4200_1500-1700.npy')
data=np.load('/data_2to/small_cubes_new_segmentation_method/debug_stitched_IC_1000-1200_2000-2200_1600-1800.npy')
# from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt
kwargs = {'sigmas': [1, 2, 5, 6,7,8,9,10], 'alpha': 1, 'beta': 5, 'gamma': 150}
kwargs['black_ridges'] = 0
data_thres=data.copy()
data_thres[np.where(data<=5000)]=0
result = hessian(data_thres, **kwargs)
result_thresh=result.copy()
result_thresh[np.where(result_thresh==1.0)]=0
p3d.plot(result_thresh)

result_thresh[np.where(result_thresh>=1e-9)]=1
p3d.plot(result_thresh)
np.save('/data_2to/small_cubes_new_segmentation_method/IC_binary.npy', result_thresh)
# result_thresh=np.logical_not(result_thresh)
p3d.plot([data,result_thresh])



## voxelisation arteries


# template_shape=(320,528,228)


radius=15
for j, controls in enumerate(controlslist):
    work_dir=workdirs[j]
    template_shape=io.shape('/data_2to/p14/P14_template_halfbrain_scaled.tif')
    vox_shape_c=(template_shape[0], template_shape[1], template_shape[2], len(controls))
    vox_control=np.zeros(vox_shape_c)

    if j==2 or j==0:
        template_shape=(320,528,228)
        vox_shape_c=(320,528,228, len(controls))
        vox_control=np.zeros(vox_shape_c)

    for i,control in enumerate(controls):
        print(control)
        try:
            graph = ggt.load(work_dir + '/' + control + '/' + str(control) + '_graph_correcteduniverse.gt')
        except:
            graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
        try:
            artery=from_e_prop2_vprop(graph, 'artery')
        except:
            artery=from_e_prop2_vprop(graph, 'artery_binary')
        gs=graph.sub_graph(vertex_filter=artery.astype(bool))
        coordinates=gs.vertex_property('coordinates_atlas')#*1.625/25#coordinates_atlas
        v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
        vox_control[:,:,:,i]=v

    for i in range(len(controls)):
        # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_control[:, :, :, i])
        io.write(work_dir + '/' + controls[i] + '/' + 'vox_control'+controls[i]+str(radius)+'.tif', vox_control[:, :, :, i].astype('float32'))

    io.write(work_dir + '/' +'vox_arteries_control'+str(radius)+'.tif', vox_control.astype('float32'))
    vox_control_avg=np.mean(vox_control[:, :, :, :], axis=3)
    io.write(work_dir + '/' +'vox_arteries_control_avg_'+str(radius)+'.tif', vox_control_avg.astype('float32'))




