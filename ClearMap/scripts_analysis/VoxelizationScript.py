import matplotlib.pyplot as plt
import pandas as pd

import ClearMap.Analysis.Measurements.Voxelization as vox
import ClearMap.IO.IO as io
import numpy as np
import ClearMap.Analysis.Graphs.GraphGt_old as ggto
import ClearMap.Analysis.Graphs.GraphGt as ggt
# from ClearMap.OtoferliinGraphs import from_e_prop2_vprop, getRadPlanOrienttaion
import ClearMap.Alignment.Annotation as ano


# def remove_surface(graph, width):
#     distance_from_suface = graph.edge_property('distance_to_surface')
#     ef=distance_from_suface>width
#     g=graph.sub_graph(edge_filter=ef)
#     return g

def remove_surface(graph, width):
    distance_from_suface = graph.vertex_property('distance_to_surface')
    ef=distance_from_suface>width
    g=graph.sub_graph(vertex_filter=ef)
    return g

## for dev brains set annotation
ano.initialize(label_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',
               extra_label = None, annotation_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotationADMBA_thresholded.tif')

ano.set_label_file('/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',extra_label = None)
annotation_file='/data_2to/201120/5annotated/annotation_thresholded_halfbrain__1_2_3__slice_None_None_None__slice_None_None_None__slice_0_246_None__.tif'

if graph.has_vertex_coordinates:
    coordinates = graph.vertex_property('coordinates_atlas');
    annotated = ano.label_points(coordinates,annotation_file=annotation_file, key='order')
    graph.set_vertex_annotation(annotated);

if graph.has_edge_coordinates:
    coordinates = graph.edge_property('coordinates_atlas');
    annotated = ano.label_points(coordinates,annotation_file='/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotationADMBA.tif', key='id')
    graph.set_edge_annotation(annotated);

##


mutants=['5w','6w','7w']
controls=['1k','2k', '3k']
work_dir='/data_SSD_2to/220607-otof-14m-fos'
states=[controls, mutants]


mutants=['2R','3R','5R', '1R']
controls=['7R','8R', '6R']
work_dir='/data_SSD_1to/otof6months'
states=[controls, mutants]

# work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
# controls=['142L','158L','162L', '164L']
work_dir='/data_SSD_2to/whiskers_graphs/fluoxetine'
mutants=['1','2','3', '4', '6', '18']
controls=['21','23', '26', '36']

radius=10

work_dir = '/data_SSD_1to/10weeks'
controls = ['1L', '2L', '3L', '4L']
mutants=['6L', '8L', '9L']
states=[controls, mutants]

# #
controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'
states=[controls, mutants]


work_dir = '/data_SSD_1to/otof1month'
# mutants=['5R', '6R', '7R', '8R','1R', '2R', '3R']\
controls=['7', '9', '11']
mutants=[ '14', '17', '18']
states=[controls, mutants]




work_dir = '/data_2to/fluoxetine'
# mutants=['5R', '6R', '7R', '8R','1R', '2R', '3R']\
controls=['1', '2', '3', '4', '5']
mutants=[ '7', '8']

work_dir = '/data_SSD_2to/fluoxetine2'
controls=['1', '2', '3', '4']#'5
mutants=[ '7', '9', '10', '11']#'8
states=[controls, mutants]

work_dir='/data_2to/earlyDep_ipsi'
controls=['4', '7', '10', '15']
mutants=[ '3', '6', '11', '16']
states=[controls, mutants]

work_dir='/data_2to/201120'
controls=['1', '5', '8']
mutants=[ '2', '3','4', '6', '7']
states=[controls, mutants]

work_dir='/data_2to/otof1M'
controls=[ '1w', '3w', '5w', '6w', '7w']
mutants=['1k', '2k', '3k', '4k']
states=[controls, mutants]

work_dir='/data_2to/otof3M/new_vasc'
# controls=['1w', '2w', '4w', '5w','6w']
# mutants=[ '1k','3k','4k', '5k', '6k']
controls=['2w', '4w', '5w','6w']
mutants=[ '3k', '5k', '6k']

work_dir='/data_SSD_2to/220503_p14_otof'
controls=['1w', '2w', '4w','6w']
mutants=['5k', '6k', '7k', '8k']
states=[controls, mutants]


work_dir='/data_2to/whiskers5M/R'
mutants=['433', '457', '458']#456 not annotated ?
controls=['467', '469']
states=[controls, mutants]


work_dir='/data_2to/dev'
controls=['p2', 'p5', 'p6']

work_dir='/data_2to/p0/new'
controls=['0a', '0b', '0c', '0d']#['2', '3']

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
controlsP14=['14a','14b', '14c']#['2', '3']

work_dirP21 = '/data_SSD_2to/P21'
controlsP21 =['1', '2', '3']

work_dir1M = '/data_2to/DBA2J_new/1m'
controls1M=['21', '22', '24']

work_dir4M = '/data_2to/DBA2J_new/4m'
controls4M = ['1', '2', '4', '7', '8', '9']#['2', '3']

work_dirP10M = '/data_2to/DBA2J_new/10m'
controls10M = ['2', '3', '4', '5', '6']#['2', '3']

work_dirP21 = '/data_SSD_2to/P21'
controlsP21 =['1', '2', '3']




anot_f=['/data_2to/alignement/atlases/new_region_atlases/P1/smoothed_anot_P1.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P3/smoothed_anot_P3_V6.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P7/smoothed_anot_P7.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed_rescale.nrrd']

distance_file = io.read(distance_file)
distance_file_shape = distance_file.shape;

def distance(coordinates):
    c = np.asarray(np.round(coordinates), dtype=int);
    c[c<0] = 0;
    x = c[:,0]; y = c[:,1]; z = c[:,2];
    x[x>=distance_file_shape[0]] = distance_file_shape[0]-1;
    y[y>=distance_file_shape[1]] = distance_file_shape[1]-1;
    z[z>=distance_file_shape[2]] = distance_file_shape[2]-1;
    d = distance_file[x,y,z];
    return d;
#### correction distance to surface alignement if there is a pb
for control in controls:
    graph=ggto.load(work_dir+'/'+control+'/'+control+'_graph.gt')
    # graph.transform_properties(distance,vertex_properties = {'coordinates_atlas' : 'distance_to_surface'},edge_geometry_properties = {'coordinates_atlas' : 'distance_to_surface'});
    # distance_to_surface = graph.edge_geometry('distance_to_surface', as_list=True);
    # distance_to_surface__edge = np.array([np.min(d) for d in distance_to_surface])
    # graph.define_edge_property('distance_to_surface', distance_to_surface__edge)
    # graph.save(work_dir + '/' + control + '/' + control+'_graph.gt')
    graph_crusted=remove_surface(graph, 2)
    # graph=graph.largest_component()
    graph_crusted.save(work_dir + '/' + control + '/' + control+'_graph_correcteduniverse.gt')



import tifffile
template_file=tifffile.imread('/mnt/vol00-renier/Solisa/annotation_thresholded32_halfbrain__1_2_3__slice_None_None_None__slice_None_None_None__slice_0_246_None__.tif')
template_shape=(320,528,228)
vox_shape_m=(template_shape[0],template_shape[1],template_shape[2], len(mutants))
vox_shape_c=(template_shape[0],template_shape[1],template_shape[2], len(controls))
vox_control=np.zeros(vox_shape_c)
vox_mutant=np.zeros(vox_shape_m)
radius=10


##% raph semi correction if dev graphs
for control in controls:
    graph=ggt.load(work_dir+'/'+control+'/'+control+'_graph.gt')
    graph=remove_surface(graph, 2)
    # graph=graph.largest_component()
    graph.save(work_dir + '/' + control + '/' + control+'_graph_correcteduniverse.gt')


##% vox pressure field

for i,control in enumerate(controls):
    print(control)
    graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)
    with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
        sampledict = pickle.load(fp)

    pressure = np.asarray(sampledict['pressure'][0])
    graph.add_vertex_property('pressure', pressure)
    coordinates=graph.vertex_property('coordinates_atlas')#*1.625/25#coordinates_atlas
    # deg0=graph.vertex_degrees()==1
    v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=pressure, radius=(radius, radius, radius),method='sphere');
    w=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
    vox_control[:,:,:,i]=v.array/w.array


io.write(work_dir + '/' +'vox_pressure_control'+str(radius)+'.tif', vox_control.astype('float32'))
vox_control_avg=np.mean(vox_control[:, :, :, :], axis=3)
io.write(work_dir + '/' +'vox_pressure_control_avg_'+str(radius)+'.tif', vox_control_avg.astype('float32'))


#vox cfos
import ClearMap.Analysis.Measurements.Voxelization as vox
import pandas as pd

mutants=['5w','6w','7w']
controls=['1k','2k', '3k']
work_dir='/data_SSD_2to/220607-otof-14m-fos'
states=[controls, mutants]

template_shape=(456,528,320)
# template_shape=io.shape(annotation_file)
vox_shape_c=(template_shape[0],template_shape[1],template_shape[2], len(controls))
vox_control=np.zeros(vox_shape_c)

vox_shape_m=(template_shape[0],template_shape[1],template_shape[2], len(mutants))
vox_mutant=np.zeros(vox_shape_m)

radius=10

for i,control in enumerate(controls):
    print(control)


    coordinates=pd.read_feather(work_dir + '/' + control+'cells.feather')[['xt', 'yt', 'zt']].values

    v = vox.voxelize(coordinates, shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
    vox_control[:,:,:,i]=v

for i in range(len(controls)):
    io.write(work_dir + '/' + controls[i] + 'vox_control'+controls[i]+str(radius)+'.tif', vox_control[:, :, :, i].astype('float32'))


io.write(work_dir + '/' +'vox_control'+str(radius)+'.tif', vox_control.astype('float32'))
vox_control_avg=np.mean(vox_control[:, :, :, :], axis=3)
io.write(work_dir + '/' +'vox_control_avg'+str(radius)+'.tif', vox_control_avg.astype('float32'))



for i,mutant in enumerate(mutants):
    print(mutant)

    coordinates=pd.read_feather(work_dir + '/' + mutant+'cells.feather')[['xt', 'yt', 'zt']].values
    v=vox.voxelize(coordinates, shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
    vox_mutant[:,:,:,i]=v



for i in range(len(mutants)):
    io.write(work_dir + '/' + 'vox_mutant'+mutants[i]+str(radius)+'.tif', vox_mutant[:, :, :, i].astype('float32'))

io.write(work_dir + '/' +'vox_mutant'+str(radius)+'.tif', vox_mutant.astype('float32'))
vox_mutant_avg=np.mean(vox_mutant[:, :, :, :], axis=3)
io.write(work_dir + '/' +'vox_mutant_avg_'+str(radius)+'.tif', vox_mutant_avg.astype('float32'))




vox_control=io.read(work_dir + '/vox_control10.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')
vox_mutant=io.read(work_dir + '/vox_mutant10.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')


from scipy import stats

pcutoff = 0.05
tvals, pvals = stats.ttest_ind(vox_control[:, :, :,:], vox_mutant[:, :, :, :], axis = 3, equal_var = False);
pi = np.isnan(pvals);
pvals[pi] = 1.0;
tvals[pi] = 0;
pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign=np.sign(tvals)
## from sagital to coronal view
pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
# pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
# pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])

pcutoff = 0.01
pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign=np.sign(tvals)
## from sagital to coronal view
pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
pvalscol_01 = colorPValues(pvals2_f, psign_f, positive = [0,255,0], negative = [0,0,255])

pvalscol_f=np.maximum(pvalscol, pvalscol_01)

# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
import tifffile
tifffile.imsave(work_dir+'/pvalcolors_deg1_density_bicol'+str(radius)+'.tif', np.swapaxes(pvalscol_f, 2, 0).astype('uint8'), photometric='rgb',imagej=True)




##%  vox BP density


workdirs=[ work_dirP1, work_dirP5,work_dirP3,work_dirP7,work_dirP6,work_dirP14]
controlslist=[ controlsP1, controlsP5,controlsP3,controlsP7,controlsP6,controlsP14]


import os
import ClearMap.Settings as settings
atlas_path = os.path.join(settings.resources_path, 'Atlas');


for j, controls in enumerate(controlslist):
    print(j,workdirs[j],controls)
    work_dir=workdirs[j]
    annotation_file=anot_f[j]
    ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                   extra_label = None, annotation_file = annotation_file)
    # template_shape=(320,528,228)
    # template_shape=io.shape('/data_2to/p14/annotatn_halfbrain.tif')#(469, 794, 246)
    template_shape=io.shape(annotation_file)
    # vox_shape_c=(469, 794, 246, len(controls))
    vox_shape_c=(template_shape[0],template_shape[1],template_shape[2], len(controls))
    vox_control=np.zeros(vox_shape_c)

    vox_shape_m=(template_shape[0],template_shape[1],template_shape[2], len(mutants))
    vox_mutant=np.zeros(vox_shape_m)



    for i,control in enumerate(controls):
        print(control)
        # graph = ggt.load(work_dir + '/' + control + '/'  + str(control)+ '_graph_correcteduniverse_smoothed.gt')
        graph = ggt.load(work_dir + '/' + control + '/' + str(control) + '_graph_correcteduniverse.gt')
        # graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
        # graph = ggt.load(work_dir + '/' + control + '/' + control+'_graph.gt')
        # BV=graph.vertex_property('radius')>=5
        # graph=graph.sub_graph(vertex_filter=BV)
        # graph = ggt.load('/data_2to/p0/4_graph2.gt')
        coordinates=graph.vertex_property('coordinates_atlas')#*1.625/25#coordinates_atlas
        deg0=graph.vertex_degrees()==1
        # v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
        v=vox.voxelize(coordinates[deg0, :3], shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
        vox_control[:,:,:,i]=v

    # for i in range(len(controls)):
    #     # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_control[:, :, :, i])
    #     io.write(work_dir + '/' + controls[i] + '/' + 'vox_control'+controls[i]+str(radius)+'.tif', vox_control[:, :, :, i].astype('float32'))

    #development
    # io.write(work_dir + '/' +'vox_control'+str(radius)+'.tif', vox_control.astype('float32'))
    io.write(work_dir + '/' +'vox_control_deg1'+str(radius)+'.tif', vox_control.astype('float32'))
    vox_control_avg=np.mean(vox_control[:, :, :, :], axis=3)
    # io.write(work_dir + '/' +'vox_control_avg'+str(radius)+'.tif', vox_control_avg.astype('float32'))
    io.write(work_dir + '/' +'vox_control_avg_deg1_'+str(radius)+'.tif', vox_control_avg.astype('float32'))



    # work_dir='/data_SSD_2to/covid19'
    # mutants=['7', '9']
    # mutants=['9']

    states=[controls, mutants]
    for i,mutant in enumerate(mutants):
        print(mutant)
        graph = ggt.load(work_dir + '/' + mutant + '/' + str(mutant) + '_graph_correcteduniverse.gt')
        # graph = ggt.load(work_dir + '/' + mutant + '/' + 'data_graph_correcteduniverse.gt')
        # graph = ggt.load(work_dir + '/' + control + '/' + control+'_graph.gt')
        # BV=graph.vertex_property('radius')>=5
        # graph=graph.sub_graph(vertex_filter=BV)
        coordinates=graph.vertex_property('coordinates_atlas')#*1.625/25
        deg0=graph.vertex_degrees()==1
        # v=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
        v=vox.voxelize(coordinates[deg0, :3], shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
        vox_mutant[:,:,:,i]=v



    for i in range(len(mutants)):
        # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_mutant[:, :, :, i])
        io.write(work_dir + '/' + mutants[i] + '/' + 'vox_mutant'+mutants[i]+str(radius)+'.tif', vox_mutant[:, :, :, i].astype('float32'))

    # io.write(work_dir + '/' +'vox_mutant'+str(radius)+'.tif', vox_mutant.astype('float32'))
    # vox_mutant_avg=np.mean(vox_mutant[:, :, :, :], axis=3)
    # io.write(work_dir + '/' +'vox_mutant_avg_'+str(radius)+'.tif', vox_mutant_avg.astype('float32'))
    io.write(work_dir + '/' +'vox_mutant_deg1'+str(radius)+'.tif', vox_mutant.astype('float32'))
    vox_mutant_avg=np.mean(vox_mutant[:, :, :, :], axis=3)
    io.write(work_dir + '/' +'vox_mutant_avg_deg1_'+str(radius)+'.tif', vox_mutant_avg.astype('float32'))




    vox_control=io.read(work_dir + '/vox_control10.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')
    vox_mutant=io.read(work_dir + '/vox_mutant10.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')


## mann whitney
# from scipy.stats import mannwhitneyu
# idx = itertools.product(range(vox_control.shape[0]),range(vox_control.shape[1]), range(vox_control.shape[2]))
# def step(args):
#     i, j, k = args
#     try:
#         a, b = mannwhitneyu(vox_control[i, j, k, :], vox_mutant[i, j, k, :])
#     except:
#         a, b= 0,1
#     return [a,b]
#
# res=np.array(list(map(step, idx)))
# pvals = res[:, 1]
# pvals=pvals.reshape(vox_control.shape[:-1])
# tvals = res[:, 0]
# np.save(work_dir+'/pvals_mw.npy', pvals)
# np.save(work_dir+'/tvals_mw.npy', tvals)
# # tvals=np.concatenate(([0,0], tvals))
# # tvals=tvals.reshape(vox_control.shape[:-1])
#
#
#


from scipy import stats

pcutoff = 0.05
tvals, pvals = stats.ttest_ind(vox_control[:, :, :,:], vox_mutant[:, :, :, :], axis = 3, equal_var = False);
pi = np.isnan(pvals);
pvals[pi] = 1.0;
tvals[pi] = 0;
pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign=np.sign(tvals)
## from sagital to coronal view
pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
# pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
# pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])

pcutoff = 0.01
pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign=np.sign(tvals)
## from sagital to coronal view
pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
pvalscol_01 = colorPValues(pvals2_f, psign_f, positive = [0,255,0], negative = [0,0,255])

pvalscol_f=np.maximum(pvalscol, pvalscol_01)

# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
import tifffile
tifffile.imsave(work_dir+'/pvalcolors_BP density_bicol'+str(radius)+'.tif', np.swapaxes(pvalscol_f, 2, 0).astype('uint8'), photometric='rgb',imagej=True)



mutants=['2R','3R','5R', '1R']
controls=['7R','8R', '6R']
work_dir='/data_SSD_1to/otof6months'
states=[controls, mutants]

work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
controls=['142L','158L','162L', '164L']
mutants=['138L','141L', '163L', '165L']
states=[controls, mutants]

# #
controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'
states=[controls, mutants]

work_dir = '/data_SSD_2to/fluoxetine2'
controls=['1', '2', '3', '4','5']
mutants=[ '7', '8', '9', '10', '11']
states=[controls, mutants]



work_dir = '/data_SSD_2to/earlyDep'
controls=['4', '7', '10','12', '15']
mutants=[ '3', '6', '11', '16']#'9',
states=[controls, mutants]


##% total length

for i, g in enumerate(controls):
    print(g)
    graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
    # graph = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph.gt')
    coordinates = graph.edge_geometry_property('coordinates_atlas')
    indices = graph.edge_property('edge_geometry_indices')

    bp = graph.n_vertices

    L = 0
    for i, ind in enumerate(indices):
        diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
        L = L + np.sum(np.linalg.norm(diff, axis=1))
    Lmm = L * 0.000025
    print(Lmm)  # m

##% vessels length radii degree distribution

for i, g in enumerate(controls):
    print(g)
    # graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
    graph = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph.gt')
    coordinates = graph.edge_geometry_property('coordinates_atlas')
    indices = graph.edge_property('edge_geometry_indices')
    degrees=graph.vertex_degrees()
    radii=graph.edge_radii()
    length=[]
    for i, ind in enumerate(indices):
        L = 0
        diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
        length.append(np.sum(np.linalg.norm(diff, axis=1))* 0.000025)

    length=np.array(length)
    print(length.shape)

    plt.figure()
    binarray=np.arange(0, 300, 10)*1e-6
    plt.hist(length, bins=binarray)
    plt.title(g+' length distribution')
    ticks_pos=plt.xticks()[0]
    plt.xticks(np.arange(0, 300*1e-6, 300*1e-6/10), np.arange(0, np.max(binarray*1e6), np.max(binarray*1e6) / 10).astype(int), size='x-large', rotation=20)
    plt.xlabel('length um')

    plt.figure()
    # length=np.array(length)
    plt.hist(radii, bins=[2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    plt.title(g+' radii distribution')


    plt.figure()
    # length=np.array(length)
    plt.hist(degrees,bins=[0,1,2,3,4,5,6,7,8,9,10])
    plt.title(g+' degree distribution')




##%  vox orientation
# work_dir='/data_2to/earlyDep_ipsi'
# controls=['4', '7', '10', '15']

import math
pi=math.pi
from scipy import stats

limit_angle=40
# condition='full_brain'
template_shape=(320,528,228)
vox_shape_c = (320, 528, 228, len(controls))
vox_shape_m = (320, 528, 228, len(mutants))
vox_ori_control_rad = np.zeros(vox_shape_c)
vox_ori_mutant_rad = np.zeros(vox_shape_m)
radius=10
region_list=[(0,0)]#[(0,0)]#[(6,6)]
average=False
# work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
# controls=['142L','158L','162L', '164L']
mode='bigvessels'#'bigvessels
if mode=='bigvessels':
    suffixe='bv'
elif mode=='arteryvein':
    suffixe='av'

for state in states:
    vox_shape_m = (320, 528, 228, len(state))
    vox_ori_control_rad = np.zeros(vox_shape_c)
    for i, g in enumerate(state):

        print(g)
        # graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
        graph = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph.gt')
        # graph=remove_surface(graph, 2)
        # graph=graph.largest_component()
        vertex_filter = np.zeros(graph.n_vertices)
        for j, rl in enumerate(region_list):
            order, level = region_list[j]
            print(level, order, ano.find(order, key='order')['name'])#'order
            label = graph.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter[label_leveled == order] = 1;
        graph = graph.sub_graph(vertex_filter=vertex_filter)

        # graph = ggt.load(work_dir + '/' + g + '/' + '/data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#


        try:
            artery=graph.vertex_property('artery')
            vein=graph.vertex_property('vein')
            # artery=from_e_prop2_vprop(graph, 'artery')
            # vein=from_e_prop2_vprop(graph, 'vein')
        except:
            print('no artery vertex properties')
            artery=np.logical_and(graph.vertex_radii()>=4.8,graph.vertex_radii()<=8)#4
            vein=graph.vertex_radii()>=8
            graph.add_vertex_property('artery', artery)
            graph.add_vertex_property('vein', vein)
            artery=from_v_prop2_eprop(graph, artery)
            graph.add_edge_property('artery', artery)
            vein=from_v_prop2_eprop(graph, vein)
            graph.add_edge_property('vein', vein)

        degrees = graph.vertex_degrees()
        vf = np.logical_and(degrees > 1, degrees <= 4)
        graph = graph.sub_graph(vertex_filter=vf)
        # with open(work_dir + '/' + g + '/sampledict' + g + '.pkl', 'rb') as fp:
        #     sampledict = pickle.load(fp)
        #
        # pressure = np.asarray(sampledict['pressure'][0])
        # graph.add_vertex_property('pressure', pressure)

        label = graph.vertex_annotation();
        # vertex_filter = from_e_prop2_vprop(graph, 'artery')
        # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
        #
        # r, p, N,l = getRadPlanOrienttaion(graph, graph, local_normal=True, calc_art=False)
        # rad = (r / (r + p)) > 0.6
        # plan = (p / (r + p)) > 0.6
        # angle = np.array([math.acos(r[i]) for i in range(r.shape[0])]) * 180 / pi

        # rad=np.load(work_dir + '/' + 'ORI_4' + condition + '_'+g+'.npy', allow_pickle=True)


        angle,graph = GeneralizedRadPlanorientation(graph, g, 4.5, state, mode=mode, average=average)
        dist = graph.edge_property('distance_to_surface')
        rad = angle <= limit_angle  # 40
        planarity = angle >= (90 - limit_angle)  # 60

        connectivity = graph.edge_connectivity()
        coordinates = graph.vertex_property('coordinates_atlas')#*1.625/25
        # coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
        edges_centers = np.array(
            [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])
        # art_coordinates = art_tree.vertex_property('coordinates_atlas')  # *1.625/25
        # print('artBP')#artBP
        # v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
        # vox_art_control[:, :, :, i] = v
        print('rad')
        v = vox.voxelize(edges_centers[rad, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
        w = vox.voxelize(edges_centers[:, :3], shape=template_shape, weights=None, radius=(radius, radius,radius), method='sphere');
        vox_ori_control_rad[:, :, :, i] = v.array / w.array#v/w#
        io.write(work_dir + '/' + control + '/' + 'vox_ori_fi_b'+suffixe+'_rad_'+controls[i]+str(average)+'.tif', (v.array / w.array).astype('float32'))

    if state==controls:
        status='control'
    elif state== mutants:
        status='mutant'
    io.write(work_dir + '/' +'vox_ori_fi_'+suffixe+'_'+status+'_rad'+str(radius)+str(average)+'.tif', vox_ori_control_rad.astype('float32'))
    vox_ori_control_rad_avg=np.nanmean(vox_ori_control_rad[:, :, :,:], axis=3)
    io.write(work_dir + '/' +'vox_ori_fi_'+suffixe+'_'+status+'_rad_avg_'+str(radius)+str(average)+'.tif', vox_ori_control_rad_avg.astype('float32'))
    vox_ori_control_rad=io.read(work_dir + '/' +'vox_ori_fi_'+suffixe+'_'+status+'_rad'+str(radius)+str(average)+'.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')

# print('plan')
    # v = vox.voxelize(edges_centers[plan, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    # # w=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(15,15,15), method = 'sphere');
    # vox_ori_control_plan[:, :, :, i] = v.array / w.array
    #
    # print('rad/Plan')  # artBP
    # vox_art_mutant[:, :, :, i] = vox_ori_mutant_racontrolsd[:, :, :, i]/vox_ori_mutant_plan[:, :, :, i]

# work_dir='/data_SSD_2to/covid19'
# mutants=['7', '9']
# mutants=['9']
for i, g in enumerate(mutants):
    print(g)
    graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
    # graph = ggt.load(work_dir + '/' + g + '/' + '/data_graph.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
    vertex_filter = np.zeros(graph.n_vertices)
    for j, rl in enumerate(region_list):
        order, level = region_list[j]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
    graph = graph.sub_graph(vertex_filter=vertex_filter)



    try:
        artery=graph.vertex_property('artery')
        vein=graph.vertex_property('vein')
        # artery=from_e_prop2_vprop(graph, 'artery')
        # vein=from_e_prop2_vprop(graph, 'vein')
    except:
        try:
            artery=from_e_prop2_vprop(graph , 'artery')
            vein=from_e_prop2_vprop(graph , 'vein')
        except:
            print('no artery vertex properties')
            artery=np.logical_and(graph.vertex_radii()>=4.8,graph.vertex_radii()<=8)#4
            vein=graph.vertex_radii()>=8
            graph.add_vertex_property('artery', artery)
            graph.add_vertex_property('vein', vein)
            artery=from_v_prop2_eprop(graph, artery)
            graph.add_edge_property('artery', artery)
            vein=from_v_prop2_eprop(graph, vein)
            graph.add_edge_property('vein', vein)


    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)
    # with open(work_dir + '/' + g + '/sampledict' + g + '.pkl', 'rb') as fp:
    #     sampledict = pickle.load(fp)
    # #
    # pressure = np.asarray(sampledict['pressure'][0])
    # graph.add_vertex_property('pressure', pressure)

    label = graph.vertex_annotation();
    # vertex_filter = from_e_prop2_vprop(graph, 'artery')
    # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
    # r, p, N,l = getRadPlanOrienttaion(graph, graph, local_normal=True, calc_art=False)
    # rad = (r / (r + p)) > 0.6
    # # plan = (p / (r + p)) > 0.6
    # angle = np.array([math.acos(r[i]) for i in range(r.shape[0])]) * 180 / pi

    # rad = np.load(work_dir + '/' + 'ORI_4' + condition + '_' + g + '.npy', allow_pickle=True)
    angle,graph = GeneralizedRadPlanorientation(graph, g, 4.5, controls, mode=mode, average=average)
    dist = graph.edge_property('distance_to_surface')
    #
    rad = angle <= limit_angle  # 40
    planarity = angle > (90 - limit_angle)  # 60

    connectivity = graph.edge_connectivity()
    coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
    edges_centers = np.array(
        [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])
    # art_coordinates = art_tree.vertex_property('coordinates_atlas')  # *1.625/25
    # print('artBP')  # artBP
    # v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    # vox_art_mutant[:, :, :, i] = v
    print('rad')
    v = vox.voxelize(edges_centers[rad, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    w = vox.voxelize(edges_centers[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    vox_ori_mutant_rad[:, :, :, i] = v.array / w.array# #v/w#
    # print('plan')
    # v = vox.voxelize(edges_centers[plan, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    # # w=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(15,15,15), method = 'sphere');
    # vox_ori_mutant_plan[:, :, :, i] = v.array / w.array
    #
    # print('rad/Plan')  # artBP
    # vox_art_mutant[:, :, :, i] = vox_ori_mutant_rad[:, :, :, i]/vox_ori_mutant_plan[:, :, :, i]


io.write(work_dir + '/' +'vox_ori_fi_'+suffixe+'_control_rad'+str(radius)+str(average)+'.tif', vox_ori_control_rad.astype('float32'))
io.write(work_dir + '/' +'vox_ori_fi_'+suffixe+'_mutant_rad'+str(radius)+str(average)+'.tif', vox_ori_mutant_rad.astype('float32'))
vox_ori_control_rad_avg=np.nanmean(vox_ori_control_rad[:, :, :,:], axis=3)
vox_ori_mutant_rad_avg=np.nanmean(vox_ori_mutant_rad[:, :, :, :], axis=3)
io.write(work_dir + '/' +'vox_ori_fi_'+suffixe+'_mutant_rad_avg_'+str(radius)+str(average)+'.tif', vox_ori_mutant_rad_avg.astype('float32'))
io.write(work_dir + '/' +'vox_ori_fi_'+suffixe+'_control_rad_avg_'+str(radius)+str(average)+'.tif', vox_ori_control_rad_avg.astype('float32'))

#
# io.write(work_dir + '/' +'vox_ori_fi_bv_control_plan'+str(radius)+'.tif', vox_ori_control_rad.astype('float32'))
# vox_ori_control_rad_avg=np.mean(vox_ori_control_rad[:, :, :,:], axis=3)
# io.write(work_dir + '/' +'vox_ori_fi_bv_control_plan_avg_'+str(radius)+'.tif', vox_ori_control_rad_avg.astype('float32'))


for i in range(len(controls)):
    # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_control[:, :, :, i])
    io.write(work_dir + '/' + controls[i] + '/' + 'vox_ori_fi_b'+suffixe+'_rad_'+controls[i]+str(average)+'.tif', vox_ori_control_rad[:, :, :, i].astype('float32'))
for i in range(len(mutants)):
    # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_mutant[:, :, :, i])
    io.write(work_dir + '/' + mutants[i] + '/' + ''+suffixe+'_rad_'+mutants[i]+str(average)+'.tif', vox_ori_mutant_rad[:, :, :, i].astype('float32'))

# work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
vox_ori_control_rad=io.read(work_dir + '/' +'vox_ori_fi_'+suffixe+'_control_rad'+str(radius)+str(average)+'.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')
vox_ori_mutant_rad=io.read(work_dir + '/' +'vox_ori_fi_'+suffixe+'_mutant_rad'+str(radius)+str(average)+'.tif')


pcutoff = 0.05



## t test
tvals, pvals = stats.ttest_ind(vox_ori_control_rad[:, :, :, :], vox_ori_mutant_rad[:, :, :, :], axis = 3, equal_var = True);
pi = np.isnan(pvals);
pvals[pi] = 1.0;
tvals[pi] = 0;
pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign=np.sign(tvals)
## from sagital to coronal view
pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
# pvalscol = colorPValues(pvals2, psign, positive = [255,0,0], negative = [0,255,0])

pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])

pcutoff = 0.01
pvals2 = pvals.copy();
pvals2[pvals2 > pcutoff] = pcutoff;
psign=np.sign(tvals)
## from sagital to coronal view
pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
pvalscol_01 = colorPValues(pvals2_f, psign_f, positive = [0,255,0], negative = [0,0,255])

pvalscol_f=np.maximum(pvalscol, pvalscol_01)

import tifffile
# tifffile.imsave('/data_SSD_1to/otof6months/6Mvs6M/pvalcolors_BP density_10TESTTEST.tif', pvalscol.astype('uint8'), photometric='rgb',imagej=True)
# tifffile.imsave('/data_SSD_1to/otof6months/6Mvs6M/pvalcolors_BP density_10TESTTEST_coro.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)#np.swapaxes(pvalscol, 2, 0)

tifffile.imsave(work_dir+'/pvalcolors_radORI_fi_'+suffixe+'_bicol'+str(radius)+'_'+str(pcutoff)+str(average)+'.tif', np.swapaxes(pvalscol_f, 2, 0).astype('uint8'), photometric='rgb',imagej=True)#np.swapaxes(pvalscol, 2, 0)

# pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
# pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
# io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')


## set mask on voxelization
import tifffile
work_dir='/data_SSD_1to/otof6months'
work_dir='/data_SSD_2to/cfos_otof_1M/vox'
work_dir='/data_SSD_2to/220607-otof-14m-fos'

radius=10
pval=tifffile.imread(work_dir+'/pvalcolors_deg1_density_bicol10.tif')#np.swapaxes(pvalscol, 2, 0)
pval=tifffile.imread('/data_SSD_2to/211019_otof_10m/vox_corrected/vox_mutant_avg_10.tif')#np.swapaxes(pvalscol, 2, 0)

annotation=tifffile.imread('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation.tif')
annotation=annotation.swapaxes(0,1)
annotation=annotation.swapaxes(1,2)
annotation=annotation[:,:, :228]

# np.flip(masked)

order, level= (6,6)
label_leveled = ano.convert_label(annotation, key='id', value='order', level=level)
masked=annotation.copy()
masked[label_leveled != order] = 0;
masked[label_leveled == order] = 1;
# p3d.plot(masked)


masked_pval=pval.copy()

try:
    for c in range(masked_pval.shape[3]):
        masked_pval[:, :, :, c]=masked_pval[:,:,:, c]*masked
except:
    masked_pval=masked_pval*masked

tifffile.imsave(work_dir+'/avgdep'+str(radius)+'_'+str(pcutoff)+'masked.tif', masked_pval.astype('float32'))#np.swapaxes(pvalscol, 2, 0)

tifffile.imsave(work_dir+'/pvalcolors_bicol2'+str(radius)+'_'+str(pcutoff)+'masked.tif', masked_pval.astype('uint8'), photometric='rgb',imagej=True)#np.swapaxes(pvalscol, 2, 0)


### diff voxelisation
import tifffile
work_dir='/data_SSD_2to/191122Otof'
vox_control=io.read(work_dir + '/vox_control_avg_10.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')
vox_mutant=io.read(work_dir + '/vox_mutant_avg_10.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')

work_dir='/data_SSD_2to/211019_otof_10m/vox_corrected'
vox_control2=io.read(work_dir + '/vox_control_avg10.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')
vox_mutant2=io.read(work_dir + '/vox_mutant_avg_10.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')

degradation_controls=(vox_control-vox_control2)/vox_control
degradation_mutants=(vox_mutant-vox_mutant2)/vox_mutant

radius=10
work_dir='/data_2to/devotof'
io.write(work_dir + '/' +'degradation_mutants'+str(radius)+'.tif', degradation_mutants.astype('float32'))
io.write(work_dir + '/' +'degradation_controls'+str(radius)+'.tif', degradation_controls.astype('float32'))