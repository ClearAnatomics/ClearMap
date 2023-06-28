import ClearMap.Alignment.Annotation as ano

import ClearMap.IO.IO as io
import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti
import os
print('TEST')
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import graph_tool.centrality as gtc
import graph_tool.generation as gtg
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

from scipy.stats import wilcoxon
from scipy.stats import ks_2samp
from scipy.stats import brunnermunzel
import pandas as pd

#only useful if you want to compare the interneurons/vessels distances by layers
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

#this file contains the list of all the leaf region according to the annotation file, namely the lowest level of annotation (layers typically)
with open('/data_SSD_2to/191122Otof/reg_list_full.p', 'rb') as fp:
  reg_list = pickle.load(fp)



#shape of the atlas template to which thee brain is aligned
#depends on the orientation
template_shape=(320,528,228)


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
    return(np.asarray(np.logical_and(isOver, isUnder)).nonzero()[0])




def filter(cells, orders, regions):
    cell_filter = np.zeros(cells.shape[0])

    for region_list in regions:
        for i, rl in enumerate(region_list):
            order, level = region_list[i]
            print(level, order, ano.find(order, key='order')['name'])

            label_leveled = ano.convert_label(orders, key='order', value='order', level=level)
            cell_filter[label_leveled == order] = 1;
    return(cell_filter)

import tifffile
distance_file='/home/sophie.skriabine/Projects/clearmapUpdated/ClearMap2/ClearMap/Resources/Atlas/ABA_25um_distance_to_surface.tif'
reference_file='/home/sophie.skriabine/Projects/clearmapUpdated/ClearMap2/ClearMap/Resources/Atlas/ABA_25um_annotation.tif'
# reference_file = tifffile.imread(reference_file)

def dist_surface_cell_filter(coordinates,distance_file):
    distance_file = tifffile.imread(distance_file)
    distance_file_shape = distance_file.shape;
    c = np.asarray(np.round(coordinates), dtype=int);
    c[c<0] = 0;
    x = c[:,0]; y = c[:,1]; z = c[:,2];
    x[x>=distance_file_shape[0]] = distance_file_shape[0]-1;
    y[y>=distance_file_shape[1]] = distance_file_shape[1]-1;
    z[z>=distance_file_shape[2]] = distance_file_shape[2]-1;
    d = distance_file[x,y,z];
    return d;


def id_cell_filter(coordinates,reference_file):
    reference_file = tifffile.imread(reference_file)
    distance_file_shape = reference_file.shape;
    c = np.asarray(np.round(coordinates), dtype=int);
    c[c<0] = 0;
    x = c[:,0]; y = c[:,1]; z = c[:,2];
    x[x>=distance_file_shape[0]] = distance_file_shape[0]-1;
    y[y>=distance_file_shape[1]] = distance_file_shape[1]-1;
    z[z>=distance_file_shape[2]] = distance_file_shape[2]-1;
    d = reference_file[x,y,z];
    return d;


def get_distances(args):
    n, cell, edges_centers, indices, radius = args

    print(n, '/',L )
    x = cell[0]
    y = cell[1]
    z = cell[2]
    c=np.array([x, y, z])
    mins = np.array([x, y, z]) - 50
    maxs = np.array([x, y, z]) + 50
    close_edges_centers=extractSubGraph(edges_centers, mins, maxs)
    close_coordinate = [array[indices[cec][0]:indices[cec][1]] for cec in close_edges_centers]
    # print(close_coordinate)
    print(len(close_coordinate))
    close_radius = [radius[indices[cec][0]:indices[cec][1]] for cec in close_edges_centers]
    if len(close_coordinate)==0:
        print('no vessels found 0')
        return([np.nan, np.nan])
    else:
        close_coordinate=np.vstack(close_coordinate)
        close_distance=[np.linalg.norm(close_coordinate[i, :3]-c) for i in range(close_coordinate.shape[0])]
        close_distance_considering_radius=close_distance-close_radius
        ## here we can choose to take vessel radius in consideration while computing the distance of the interneurons either to the centerline or the surface of the vessel
        i=np.argmin(close_distance)
        print(close_distance[i])
        return([close_distance[i], close_coordinate[i, 3]])


def getlayersIndices(orders, regions):
    cell_filter = filter(orders, regions)
    order_t=orders[cell_filter.astype(bool)]

    layers = ['1', '2/3', '4', '5', '6a', '6b']
    layers_indices=[]
    for b, layer in enumerate(layers):
        layer_filter = np.zeros(order_t.shape[0])
        for region_list in regions:
            for i, rl in enumerate(region_list):
                order, level = region_list[i]
                R = ano.find(order, key='order')['name']
                for r in reg_list.keys():
                    n = ano.find_name(r, key='order')
                    if R in n:
                        for se in reg_list[r]:
                            if layer in ano.find(se, key='order')['name']:
                                l = ano.find(se, key='order')['level']
                                print(ano.find(se, key='order')['name'], se)
                                label_leveled = ano.convert_label(order_t, key='order', value='order', level=l)
                                layer_filter[label_leveled == se] = 1
                                layers_indices.append(layer_filter)
    return(layers_indices)



work_dir='/data_SSD_1to/blinder'#'/data_SSD_2to/whiskers_graphs'
brain='200916'
name='inferior_colliculus'


# regions=[[(6,6)]]#isocortex
if name == 'Auditory_regions':
    regions = [[(142, 8), (149, 8), (128, 8), (156, 8)]]

elif name == 'barrels':
    regions = [[(54, 9)]]#[[(54, 9), (47, 9)]]  # , (75, 9)]  # barrels

elif name == 'motor_primary':
    regions = [[(19, 8)]]#[[(54, 9), (47, 9)]]  # , (75, 9)]  # barrels

elif name == 'hippocampus':
    regions = [[(463, 6)]]

elif name=='inferior_colliculus':
    regions = [[(817,6)]]

graph = ggt.load(work_dir + '/' + brain + '/' + 'data_graph.gt')
cells=np.load(work_dir + '/' + brain + '/' + 'cells.npy')

name='Isocortex'
regions=[[(6,6)]]#isocortex

# name='universe'
# regions=[[(0,0)]]#isocortex



# order, level=region
label = graph.vertex_annotation();

ct_x=[c[5] for c in cells]
ct_y=[c[6] for c in cells]
ct_z=[c[7] for c in cells]
C=[ct_x, ct_y, ct_z]
C=np.array(C)
C=C.T
d=dist_surface_cell_filter(C, distance_file)
cells_no_surface=cells[d>=5]


orders=np.array([cells[i][8] for i in range(cells_no_surface.shape[0])])
cell_filter=filter(cells_no_surface, orders, regions)

# label = graph.vertex_annotation();
# label_leveled = ano.convert_label(orders, key='order', value='order', level=level)
# cell_filter[label_leveled == order] = 1;

cells_t=cells_no_surface[cell_filter.astype(bool)]

connectivity = graph.edge_connectivity()
coordinates = graph.vertex_property('coordinates')  # *1.625/25
edges_centers = np.array(
    [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])

coordinates=graph.edge_geometry_property('coordinates')
indices = graph.edge_property('edge_geometry_indices')
radius = graph.edge_geometry_property('radii')
eradius=radius[:,np.newaxis]
array=np.concatenate((coordinates, eradius), axis=1)

# minDistances=[]
L=cells_t.shape[0]


###### compute the distances between the cell and the closest vessel in a area centered around the cells to reduce the calculations
#  the area size should be estimated large enough to make sure vessels are present, could be for instance the characteristic length
# of a capillary

from multiprocessing import Pool
p = Pool(20)
import time
start = time.time()

minDistances = np.array(
    [p.map(get_distances, [(n, cell,edges_centers, indices, radius) for n, cell in enumerate(cells_t)])])

end = time.time()
print(end - start)

np.save(work_dir+'/'+'mindistances_radii'+name+'.npy',minDistances[0])

# minDistances=np.nan_to_num(minDistances)

test=np.load(work_dir+'/'+'mindistances_radii'+name+'.npy')
dist=test[:, 0]
radius=test[:, 1]

minDistances=np.load('/data_SSD_1to/blinder/mindistances.npy')



#####  histogram  distances
#
plt.figure()
# plt.hist(minDistances, bins=300)
x = minDistances[~np.isnan(minDistances)]
ax = sns.distplot(x)
plt.title('histogram of minimal distances from interneurons to vessels in '+name,size='x-large')
plt.yscale('linear')
plt.xlabel('distance in px', size='x-large')
plt.xticks(size='x-large')
plt.yticks(size='x-large')

#####  histogram  distances < threshold (=3 here)
#
plt.figure()
# plt.hist(minDistances[minDistances<=3], bins=10)
x = minDistances[minDistances<=3][~np.isnan(minDistances[minDistances<=3])]
ax = sns.distplot(x)
plt.title('histogram of minimal distances from interneurons to vessels in '+name,size='x-large')
plt.yscale('linear')
plt.xlabel('distance in px', size='x-large')
plt.xticks(size='x-large')
plt.yticks(size='x-large')

#####  histogram  radius small distance
#
plt.figure()
# plt.hist(minDistances[minDistances<=3], bins=10)
x = radius[dist<=3][~np.isnan(dist[dist<=3])]
sns.kdeplot(x,bw=0.19)
# ax = sns.distplot(x,bins=np.arange(0,15),kde_kws={"color": "cadetblue", "bw_methods": "silverman","lw": 3, "label": "KDE"},)

radgraph=graph.edge_geometry_property('radii')
# sns.distplot(radgraph,bins=np.arange(0,15))
sns.kdeplot(radgraph,bw=0.195)

test=np.load(work_dir+'/'+'randommindistances_'+name+'.npy')
minDistancesRand_dist=test[:, 0]
minDistancesRand_radius=test[:, 1]

x =minDistancesRand_radius[~np.isnan(minDistancesRand_dist)]
ax = sns.kdeplot(x,bw=0.195)

plt.legend(['close PV interneurons-vessels', 'graph radii distribution', 'random distribution'])

plt.title('histogram of radius of vessels at small distance from interneurons '+name,size='x-large')
plt.yscale('linear')
plt.xlabel('radius in px', size='x-large')
plt.xticks(size='x-large')
plt.yticks(size='x-large')


##### generate random distribution
from numpy.random import rand, seed
random_cells=np.zeros((cells_t.shape[0], 3))
ct_x=np.array([c[0] for c in cells_t])
ct_y=np.array([c[1] for c in cells_t])
ct_z=np.array([c[2] for c in cells_t])


seed(1)
valuesX = rand(cells_t.shape[0])
seed(2)
valuesY = rand(cells_t.shape[0])
seed(3)
valuesZ = rand(cells_t.shape[0])
random_cells[:, 0]=(valuesX*(np.max(ct_x)-np.min(ct_x)))+np.min(ct_x)
random_cells[:, 1]=(valuesY*(np.max(ct_y)-np.min(ct_y)))+np.min(ct_y)
random_cells[:, 2]=(valuesZ*(np.max(ct_z)-np.min(ct_z)))+np.min(ct_z)

##### take random cells in cortex

# order, level=region
label = graph.vertex_annotation();
random_cells_atlas=random_cells*1.625/25

ct_x=[c[0] for c in random_cells]
ct_y=[c[1] for c in random_cells]
ct_z=[c[2] for c in random_cells]
C=[ct_x, ct_y, ct_z]
C=np.array(C)
C=C.T
d=dist_surface_cell_filter(C*1.6/25, distance_file)
cells_no_surface=random_cells[d>=5]


# orders=np.array([random_cells[i][8] for i in range(cells_no_surface.shape[0])])
# orders=[ano.find(reference_file[int(C[i, 0]), int(C[i, 1]), int(C[i,2])], key='id') for i in range(C.shape[0])]
ids=id_cell_filter(cells_no_surface*1.6/25, reference_file)
label_leveled = ano.convert_label(ids, key='id', value='order', level=6)
orders=label_leveled#[ano.find(ids[i], key='id')['order'] for i in range(ids.shape[0])]
cell_filter=filter(cells_no_surface, orders, regions)

# label = graph.vertex_annotation();
# label_leveled = ano.convert_label(orders, key='order', value='order', level=level)
# cell_filter[label_leveled == order] = 1;

cells_t=cells_no_surface[cell_filter.astype(bool)]
L=cells_t.shape[0]



from multiprocessing import Pool
p = Pool(20)
import time
start = time.time()

minDistancesRand = np.array(
    [p.map(get_distances, [(n, cell,edges_centers, indices) for n, cell in enumerate(cells_t)])])

end = time.time()
print(end - start)

np.save(work_dir+'/'+'randommindistances_'+name+'.npy',minDistancesRand[0])
np.save(work_dir+'/'+'randomcells_'+name+'.npy', cells_t)
minDistancesRand=np.load('/data_SSD_1to/blinder/randommindistances_'+name+'.npy')
random_cells=np.load('/data_SSD_1to/blinder/randomcells_'+name+'.npy')


##### plot distribution
plt.figure()
x =minDistancesRand[~np.isnan(minDistancesRand)]
ax = sns.distplot(x)
plt.title('histogram of minimal distances from interneurons to vessels in '+name,size='x-large')
plt.yscale('linear')
plt.xlabel('distance in px', size='x-large')
plt.xticks(size='x-large')
plt.yticks(size='x-large')

plt.figure()
x = minDistancesRand[minDistancesRand<=3][~np.isnan(minDistancesRand[minDistancesRand<=3])]
ax = sns.distplot(x)
plt.title('histogram of minimal distances from interneurons to vessels in '+name,size='x-large')
plt.yscale('linear')
plt.xlabel('distance in px', size='x-large')
plt.xticks(size='x-large')
plt.yticks(size='x-large')

##### comparison random VS real cell dist

plt.figure()
# hist, bin=np.histogram(minDistances,bins=np.arange(0,40, 40/40), normed=True)
# plt.plot(hist)
x =minDistancesRand[~np.isnan(minDistancesRand)]
ax = sns.distplot(x)
# plt.hist(minDistancesRand, bins=np.arange(0,40), alpha=0.3)
plt.title('histogram of minimal distances from interneurons to vessels in '+name,size='x-large')
plt.yscale('linear')
plt.xlabel('distance in px', size='x-large')
plt.xticks(size='x-large')
plt.yticks(size='x-large')

# hist, bin=np.histogram(minDistancesRand,bins=np.arange(0,40, 40/40), normed=True)
# plt.plot(hist)
x =minDistances[~np.isnan(minDistances)]
ax = sns.distplot(x)
# plt.hist(minDistances, bins=np.arange(0,40), alpha=0.3)
plt.title('histogram of minimal distances from interneurons to vessels in '+name,size='x-large')
plt.yscale('linear')
plt.xlabel('distance in px', size='x-large')
plt.xticks(np.arange(0, 40,40/10), np.arange(0, 40, 40 / 10).astype(int), size='x-large', rotation=20)
plt.xticks(size='x-large')
plt.yticks(size='x-large')

plt.legend(['random distribution', 'parvalbumine cells distribution'])



plt.figure()
hist, bin=np.histogram(minDistances[minDistances<=3],bins=np.arange(0,3, 0.3), normed=True)
plt.plot(hist)
# plt.hist(minDistancesRand, bins=np.arange(0,40), alpha=0.3)

hist, bin=np.histogram(minDistancesRand[minDistancesRand<=3],bins=np.arange(0,3, 0.3), normed=True)
plt.plot(hist)
# plt.hist(minDistances, bins=np.arange(0,40), alpha=0.3)
plt.title('histogram of minimal distances from interneurons to vessels in '+name,size='x-large')
plt.yscale('linear')
plt.xlabel('distance in px', size='x-large')
plt.xticks(plt.xticks()[0], np.round(np.arange(0, 3, 0.3),2), size='x-large', rotation=20)
plt.yticks(size='x-large')

plt.legend(['random distribution', 'parvalbumine cells distribution'])

##### voxelisation
name='universe'
radius=10

import ClearMap.Analysis.Measurements.Voxelization as vox
coordinates=graph.vertex_property('coordinates_atlas')#*1.625/25#coordinates_atlas
# deg0=graph.vertex_degrees()==1
v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
io.write(work_dir + '/' +'vox_vasculature_universe'+str(radius)+'.tif', v.astype('float32'))



minDistances=np.load(work_dir+'/'+'mindistances_'+name+'.npy')

ct_x=[c[5] for c in cells]
ct_y=[c[6] for c in cells]
ct_z=[c[7] for c in cells]
C=[ct_x, ct_y, ct_z]
C=np.array(C)
C=C.T
short_dist_cells=C[minDistances<=3]
short_dist_cells=C[minDistances<=3]
w=vox.voxelize(C, shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
v=vox.voxelize(short_dist_cells, shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
io.write(work_dir + '/' +'vox_cells_on_vessels_universe'+str(radius)+'.tif', v.astype('float32'))
io.write(work_dir + '/' +'vox_cells_on_vessels_universe_normalized'+str(radius)+'.tif', (v.array / w.array).astype('float32'))


##### voxelisation
name='universe'
minDistancesRand=np.load(work_dir+'/'+'randommindistances_'+name+'.npy')

ct_x=[c[0] for c in random_cells]
ct_y=[c[1] for c in random_cells]
ct_z=[c[2] for c in random_cells]
C=[ct_x, ct_y, ct_z]
C=np.array(C)
C=C.T*1.6/25
short_dist_cells=C[minDistancesRand<=1]
v=vox.voxelize(short_dist_cells, shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
io.write(work_dir + '/' +'vox_randomcells_on_vessels_universe'+str(radius)+'.tif', v.astype('float32'))
# io.write(work_dir + '/' +'vox_cells_on_vessels_universe'+str(radius)+'.tif', v.astype('float32'))




#####  region comparison by layers
#
names=[ '_barrels', '_motor_primary']
reg=[[[(54, 9)]],[[(19, 8)]]]

from scipy import stats

xx = np.linspace(0, 50, 50)

for i, n in enumerate(names):
    plt.figure()
    # sns.set_style(style='white')
    data=np.load(work_dir+'/'+'mindistances'+n+'.npy')
    regions=reg[i]
    layers=getlayersIndices(orders, regions)
    for l in layers:
        layer_data=data[l.astype(bool)]
        kde = stats.gaussian_kde(layer_data)
        plt.plot(xx, kde(xx))
    plt.legend(np.arange(len(layers)))
    plt.title('histogramm of distances per layers im '+n)
    plt.xlabel('distance in px')





#####  region comparison
#
names=['_motor_primary', '_barrels', '','_inferior_colliculus']
plt.figure()
# sns.set_style(style='white')
for n in names:
    data=np.load(work_dir+'/'+'mindistances'+n+'.npy')
    data=data[~np.isnan(data)]
    kde = stats.gaussian_kde(data)
    plt.plot(xx, kde(xx))


plt.legend(labels=['MOp', 'SSp_bfd', 'Isocortex', 'inferior_colliculus'])
# sns.despine()
plt.yscale('linear')




## voxelization cells located close to vessels
import matplotlib.pyplot as plt
import ClearMap.Analysis.Measurements.Voxelization as vox
import ClearMap.IO.IO as io
import numpy as np
import ClearMap.Alignment.Annotation as ano

template_shape=(320,528,228)
radius=10

ct_x=[c[0] for c in cells_t]
ct_y=[c[1] for c in cells_t]
ct_z=[c[2] for c in cells_t]
C=[ct_x, ct_y, ct_z]
C=np.array(C).T
short_dist_cells=C[minDistances<=3]
v=vox.voxelize(short_dist_cells, shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
io.write(work_dir + '/' +'vox_cells_on_vessels'+str(radius)+'.tif', v.astype('float32'))