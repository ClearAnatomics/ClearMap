from sklearn.neighbors import kneighbors_graph
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from sklearn import preprocessing
import pylab as plt
import numpy as np
from numpy import linalg as LA
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import ClearMap.Analysis.Graphs.GraphGt as ggt;
import ClearMap.Visualization.Plot3d as p3d
import os
from multiprocessing import Pool
import time
import ClearMap.Analysis.Graphs.GraphGt_new as ggtn
import ClearMap.Analysis.Graphs.GraphGt_updated as ggtu
import ClearMap.Analysis.Graphs.GraphGt_old as ggto
from scipy.spatial import Voronoi
import graph_tool.inference as gti
import graph_tool.generation as gtg

import seaborn as sns


def generateNeurons(Nb=1000, size=[0,100], weightype='gaussian', a=1, sigma=10):
    xweight = np.ones(size[1]) / size[1]
    zweight = np.ones(size[1]) / size[1]
    if weightype=='random':
        yweight = np.ones(size[1]) / size[1]
        yweight = np.nan_to_num(yweight)

    elif weightype=='gaussian':
        print('gaussian')
        x = np.linspace(0, size[1], size[1])
        yweight = gaussian(x, a, size[1]*30/100, sigma )
        # yweight2 = gaussian(x, a/2, size[1] * 50 / 100, sigma-5)
        # yweight = np.nan_to_num(np.maximum(yweight,yweight2))

    else:
        thresh=2
        reg=5
        try:
            modeldist=np.load(weightype,allow_pickle=True)
            bp_dist=np.array(modeldist[reg])##

        except:
            print('the string provided does not correspond to any file')

        hist_bp_dist, bins_bp_dist = np.histogram(bp_dist[bp_dist>thresh], bins=size[1], normed=True)
        yweight = hist_bp_dist

    zposition = np.random.choice(range(size[1]), int(np.round(Nb)), p=zweight / np.sum(zweight), replace=True)[:,np.newaxis]
    yposition=np.random.choice(range(size[1]), int(np.round(Nb)), p=yweight / np.sum(yweight), replace=True)[:, np.newaxis]
    xposition=np.random.choice(range(size[1]), int(np.round(Nb)), p=xweight / np.sum(xweight), replace=True)[:, np.newaxis]

    neuronposition=np.concatenate((xposition,yposition,zposition), axis=1)
    return neuronposition




directory = "160322"

parent_dir = "/data_SSD_2to/SimulationVasculature3D"
path = os.path.join(parent_dir, directory)
try:
    os.mkdir(path)
    print("Directory '% s' created" % directory)
except:
    print('directory aleready exists')

#%%   test model   ####


a=2
sigma=20#10

neuronpositionTest=generateNeurons(Nb=5000, size=[0,100], weightype='gaussian',a=a, sigma=sigma)#100000
# neuronpositionTest=generateNeurons(Nb=5000, size=[0,100], weightype='/data_SSD_2to/191122Otof/BP_isocortex_5R.npy',a=a, sigma=sigma)#100000
neuronpositionTest=neuronpositionTest[neuronpositionTest[:,1]>10]


plt.figure()
sns.distplot(neuronpositionTest[:,1], bins=10, kde=False)
plt.twinx()
sns.kdeplot(neuronpositionTest[:,1],bw_adjust=1.5)

from scipy.spatial import Voronoi, voronoi_plot_2d
points=neuronpositionTest[np.random.choice(np.arange(neuronpositionTest.shape[0]), int(neuronpositionTest.shape[0]/10), replace=False)]
points=points[:, [1,2]]
vor = Voronoi(points)
fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                        line_width=2, line_alpha=0.3, point_size=5,line_style='-')
plt.scatter(vor.vertices[:, 0], vor.vertices[:,1], c='k', s=5)
plt.show()


vor = Voronoi(neuronpositionTest)
ridges=vor.ridge_vertices
vertices=vor.vertices
E=[]
for faces in ridges:
    for i, node in enumerate(faces):
        try:
            E.append((node, faces[i+1]))
        except:
            print("reached end of array", faces)
            # break
            E.append((node, faces[0]))
            break
E=np.array(E).astype('int32')


# np.save(os.path.join(path, 'cap_vertices.npy'), vertices)
# np.save(os.path.join(path, 'cap_edges.npy'), E)




VecStart_x=vertices[E[:,0]][:,0]
VecEnd_x=vertices[E[:,1]][:,0]

VecStart_y=vertices[E[:,0]][:,1]
VecEnd_y=vertices[E[:,1]][:,1]

VecStart_z=vertices[E[:,0]][:,2]
VecEnd_z=vertices[E[:,1]][:,2]

mins=[5,5,50]
maxs=[100,70,60]

isOverStart = (np.concatenate((np.concatenate((VecStart_x[:, np.newaxis],VecStart_y[:, np.newaxis]), axis=1),VecStart_z[:, np.newaxis]), axis=1) > mins).all(axis=1)
isUnderStart = (np.concatenate((np.concatenate((VecStart_x[:, np.newaxis],VecStart_y[:, np.newaxis]), axis=1),VecStart_z[:, np.newaxis]), axis=1) < maxs).all(axis=1)

isOverEnd = (np.concatenate((np.concatenate((VecEnd_x[:, np.newaxis],VecEnd_y[:, np.newaxis]), axis=1),VecEnd_z[:, np.newaxis]), axis=1) > mins).all(axis=1)
isUnderEnd = (np.concatenate((np.concatenate((VecEnd_x[:, np.newaxis],VecEnd_y[:, np.newaxis]), axis=1),VecEnd_z[:, np.newaxis]), axis=1) < maxs).all(axis=1)

mask=isOverStart * isUnderStart *isOverEnd*isUnderEnd


V_tot_f=vertices
E_cap=E[mask]
isOverV=(vertices > mins).all(axis=1)
isUndeV=(vertices < maxs).all(axis=1)
V=vertices[isOverV*isUndeV]


E_cap=E_cap[np.logical_and(E_cap[:,0]>=0, E_cap[:,1]>=0)]

Etot_f=E#E_cap

ratio=10
graph=fromEVtoGraph(V_tot_f, Etot_f, 0, 0,0,0, radius=1, ratio=ratio)#ratio=30

vertices=graph.vertex_coordinates()
E=graph.edge_connectivity()

mins=[5*ratio,5*ratio,50*ratio]
maxs=[100*ratio,70*ratio,60*ratio]
isOverV=(vertices > mins).all(axis=1)
isUndeV=(vertices < maxs).all(axis=1)
vertex_filter=isOverV*isUndeV
graph=graph.sub_graph(vertex_filter=vertex_filter)

p3d.plot_graph_line(graph)

# signal_val=[gaussian(graph.vertex_coordinates()[n, 1], 100, 400, 150) for n in range(graph.n_vertices)]
#
# import ClearMap.Visualization.Color as col
# colors=col.colormap('cool')
# colorsG=colors(signal_val)
# p3d.plot_graph_mesh(graph, vertex_colors=colorsG)

# gs=graph.sub_slice((slice(300,400),slice(0,1000), slice(0,1000) ));
# p3d.plot_graph_mesh(gs, vertex_colors=colorsG)


graph=graph.largest_component()
import graph_tool.topology as gtt

## choose arteries/veins vertices

art_ain_candidates=np.asarray(graph.vertex_coordinates()[:,1]<np.min(graph.vertex_coordinates()[:,1])+30).nonzero()[0]
art=np.random.choice(art_ain_candidates, 5)
new_art_candidates = np.delete(art_ain_candidates, np.where(art_ain_candidates == art[1]))
new_art_candidates = np.delete(art_ain_candidates, np.where(art_ain_candidates == art[0]))
new_art_candidates = np.delete(art_ain_candidates, np.where(art_ain_candidates == art[2]))

vein=np.random.choice(new_art_candidates, 3)

arteries=np.zeros(graph.n_vertices)
veins=np.zeros(graph.n_vertices)
arteries[art]=1
veins[vein]=1

## tracing

iter=20
i=iter
while i >0:
    print(i, art, vein)
    print(np.sum(arteries), np.sum(veins))
    art_temp=art.copy()
    for j, a in enumerate(art[:5]):
        print(j, a)

        print(np.asarray(arteries==1).nonzero()[0])
        neigh=graph.vertex_neighbours(a)
        print(neigh)
        n2rm=[]
        for r, n in enumerate(neigh):
            # print(n)
            if n in np.asarray(arteries==1).nonzero()[0]:
                n2rm.append(r)
        print(n2rm)
        if len(n2rm)>0:
            neigh=np.delete(neigh, n2rm)
        print(neigh)
        if neigh.shape[0]!=0:
            if neigh.shape[0]==1:
                arteries[neigh[0]]=1
                art_temp[j]=neigh[0]
                print(0,neigh[0])

            else:
                art_pos=graph.vertex_coordinates()[arteries.astype(bool)]
                neigh_pos=graph.vertex_coordinates()[neigh]
                # vegf, o2=vegflevel(neuronpositionTest,art_pos, siga, sigb)
                # signal_val=[np.sum([gaussian(neigh_pos[n], vegf[k], neuronpositionTest[k], sigma) for k in range(neuronpositionTest.shape[0])]) for n in range(neigh.shape[0])]
                signal_val=[np.sum([gaussian(neigh_pos[n], 1, neuronpositionTest[k]*10, sigma) for k in range(neuronpositionTest.shape[0])]) for n in range(neigh.shape[0])]
                arteries[neigh[np.argmax(signal_val)]]=1
                art_temp[j]=neigh[np.argmax(signal_val)]
                print(1,neigh[np.argmax(signal_val)])
    print(art)
    art=art_temp.copy()
    print(art_temp)
    print(np.sum(arteries))

    vein_temp=vein.copy()
    for j, a in enumerate(vein):
        neigh=graph.vertex_neighbours(a)
        n2rm=[]
        for r, n in enumerate(neigh):
            if n in np.asarray(veins==1).nonzero()[0]:
                n2rm.append(r)
        if len(n2rm)>0:
            neigh=np.delete(neigh, n2rm)
        if neigh.shape[0]!=0:
            if neigh.shape[0]==1:
                veins[neigh[0]]=1
                vein_temp[j]=neigh[0]
                print(2,neigh[0])

            else:
                art_pos=graph.vertex_coordinates()[veins.astype(bool)]
                neigh_pos=graph.vertex_coordinates()[neigh]
                vegf, o2=vegflevel(neuronpositionTest,art_pos, siga, sigb)
                signal_val=[np.sum([gaussian(neigh_pos[n], vegf[k], neuronpositionTest[k]*10, sigma) for k in range(neuronpositionTest.shape[0])]) for n in range(neigh.shape[0])]
                veins[neigh[np.argmax(signal_val)]]=1
                vein_temp[j]=neigh[np.argmax(signal_val)]
                print(3,neigh[np.argmax(signal_val)])

    vein=vein_temp.copy()
    print(np.sum(veins))

    i=i-1
    print('############')
    print(i)
    print('############')

graph.add_vertex_property('arteries',arteries)
graph.add_vertex_property('veins',veins)

art_graph=graph.sub_graph(vertex_filter=np.logical_or(arteries, veins))
p3d.plot_graph_line(art_graph)
## remove mutual edges
conn=graph.edge_connectivity()
p = Pool(20)
start = time.time()
e2rm = np.array(
    [p.map(mutualLoopDetection, [(i, conn) for  i in range(graph.n_edges)])])
end = time.time()
print(end - start)
e2rm=e2rm[0][[e2rm[0][i].shape[0]>1 for i in range(e2rm[0].shape[0])]]
e2rm_list=[e2rm[i].tolist() for i in range(e2rm.shape[0])]
L=[]
for i, e in enumerate(e2rm_list):
    print(i)
    if e not in L:
        print('not in list')
        L.append(e)

edge_filter = np.ones(graph.n_edges)
for e in L:
    # print(e)
    try:
        edge_filter[e[1:]] = 0
    except:
        print('no edge to remove')
        print(e)
graphclean = graph.sub_graph(edge_filter=edge_filter)

p3d.plot_graph_line(graphclean)

art_graph=graphclean.sub_graph(vertex_filter=np.logical_or(graphclean.vertex_property('arteries'), graphclean.vertex_property('veins')))
p3d.plot_graph_line(art_graph)


## link together vein and artery in a close graph
gtemp=graphclean.copy()
g=gtemp.base

vertex_path=np.zeros(graphclean.n_vertices)
for a in art:
    for v in vein:
        print(a, v)
        vlist, elist = gtt.shortest_path(gtemp.base, gtemp.base.vertex(a), gtemp.base.vertex(v))
        vlist=[int(str(v)) for v in vlist]
        vertex_path[vlist]=1

edge_path=from_v_prop2_eprop(graphclean, vertex_path)

at_vein=np.logical_or(gtemp.vertex_property('arteries'), gtemp.vertex_property('veins'))
graphclean.add_vertex_property('gtemp',np.logical_or(from_e_prop2_vprop(graphclean, edge_path),at_vein))
at_vein=np.logical_or(from_v_prop2_eprop(gtemp, gtemp.vertex_property('arteries')), from_v_prop2_eprop(gtemp, gtemp.vertex_property('veins')))
gtemp = graphclean.sub_graph(edge_filter=np.logical_or(edge_path,at_vein))


p3d.plot_graph_line(gtemp)
## grow capillaries
g_saved=gtemp.copy()
g_saved.save(os.path.join(path, 'g_saved.gt'))


# fit the nearest neighbours statsmodel
from sklearn.neighbors import NearestNeighbors
coordinates=gtemp.vertex_property('coordinates')
NearestNeighborsModel = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(coordinates)

# select heads:
signal_val=[np.sum([gaussian(coordinates[n], vegf[k], neuronpositionTest[k], sigma) for k in range(neuronpositionTest.shape[0])]) for n in range(gtemp.n_vertices)]
signal_val=np.asarray(signal_val)

signal_val=[gaussian(coordinates[n,1], 20, np.max(coordinates[:,1])*30/100, 200) for n in range(gtemp.n_vertices)]
signal_val=np.asarray(signal_val)
from sklearn.preprocessing import normalize

N=0
degrees=gtemp.vertex_degrees()
# while N<15:
# vegf, o2=vegflevel(neuronpositionTest,coordinates[vessels], siga, sigb)
while np.sum(degrees>=3)/gtemp.n_vertices<0.95:
    print(N, np.sum(degrees>=3)/gtemp.n_vertices)
    degrees=gtemp.vertex_degrees()
    deg0=np.asarray(degrees==0).nonzero()[0]
    deg1=np.asarray(degrees==1).nonzero()[0]
    deg2=np.asarray(degrees==2).nonzero()[0]
    try:
        new_heads=np.random.choice(deg2, 5, p=normalize(signal_val[deg2].reshape(-1,1), norm='l1',axis=0)[:,0])
    except:
        print('no deg2')
        # new_heads=np.array([])
        new_heads=np.random.choice(np.asarray(degrees==0).nonzero()[0], 5, p=normalize(signal_val[deg0].reshape(-1,1), norm='l1',axis=0)[:,0])
    heads=np.concatenate((deg1, new_heads))
    print(heads)
    vessels=np.asarray(degrees>=1).nonzero()[0]

    for h in heads:
        degrees=gtemp.vertex_degrees()
        distances, indices = NearestNeighborsModel.kneighbors([coordinates[h]])
        neigh=indices[0][1:]
        neigh=neigh[degrees[neigh]<3]
        if neigh.shape[0]>0:
            neigh_pos=coordinates[neigh]
            # vegf, o2=vegflevel(neuronpositionTest,coordinates[vessels], siga, sigb)
            # signal_val=[np.sum([gaussian(neigh_pos[n], vegf[k], neuronpositionTest[k], sigma) for k in range(neuronpositionTest.shape[0])]) for n in range(neigh.shape[0])]
            # p=np.array(signal_val)
            p=np.array(signal_val)[neigh]
            # p=np.ones(p.shape)
            if np.sum(p)!=1:
                p=np.ones(p.shape)
            # p[(degrees[neigh]==2)]=np.sum(p)/2#***
            try:
                already_connected=gtemp.vertex_neighbours(h)
                weights=normalize(p.reshape(1,-1),norm='l1')[0]
                # weights=np.ones(weights.shape)
                next=np.random.choice(neigh, 1)#, p=weights)
                while(next in already_connected):
                    # neigh=np.delete(neigh, np.asarray(neigh==next).nonzero())
                    # p=np.array(signal_val)[neigh]
                    # if np.sum(p)==0:
                    #     p=np.ones(p.shape)
                    p[(degrees[neigh]==2)]=np.sum(p)/2
                    p[np.asarray(neigh==next).nonzero()[0]]=0
                    next=np.random.choice(neigh, 1, p=normalize(p.reshape(1,-1),norm='l1')[0])
                    # next=np.random.choice(neigh, 1)
                gtemp.add_edge((h, next))
                print(neigh, next)
            except:
                print('no found neighbours')
        else:
            print('no deg 3 found neighbours')
            neigh=indices[0][1:]
            neigh_pos=coordinates[neigh]
            # vegf, o2=vegflevel(neuronpositionTest,coordinates[vessels], siga, sigb)
            # signal_val=[np.sum([gaussian(neigh_pos[n], vegf[k], neuronpositionTest[k], sigma) for k in range(neuronpositionTest.shape[0])]) for n in range(neigh.shape[0])]
            # p=np.array(signal_val)[neigh]
            # next=neigh[np.argmax(p)]
            next=np.random.choice(neigh, 1)
            gtemp.add_edge((h, next))
            # print(neigh, next)
        # except:
        #     print('no found neighbours')
        #     neigh=indices[0][1:]
        #     gtemp.add_edge((h, neigh[0]))
    N=N+1
    degrees=gtemp.vertex_degrees()



# gtemp.save(os.path.join(path, 'graph_new_model_5000.gt'))

gtemp=ggt.load('/data_SSD_2to/SimulationVasculature3D/160322/g_saved_358.gt')
edge_vein_label = gtemp.vertex_property('veins');
edge_artery_label = gtemp.vertex_property('arteries');
paths=gtemp.vertex_property('gtemp')
# edge_vein_label = vei
# edge_artery_label = art

vertex_colors = np.zeros((gtemp.n_vertices, 4))
vertex_colors[:, 3]=1
vertex_colors[:, 1]=0.8
# connectivity = gtemp.edge_connectivity();
# edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
vertex_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
vertex_colors[edge_vein_label  >0] = [0.0,0.0,0.8,1.0]
vertex_colors[paths  >0] = [0.8,0.0,0.0,1.0]


p = p3d.plot_graph_mesh(gtemp, vertex_colors=vertex_colors, n_tube_points=3);




## check graph
import graph_tool.draw as gtd
pos = gtd.sfdp_layout(gtemp.base)
gtd.graph_draw(gtemp.base, pos=pos, output="/home/sophie.skriabine/graph-draw-sfdp.pdf")



## add radii
refg=ggt.load('/data_SSD_2to/191122Otof/5R/data_graph_correctedIsocortex.gt')
ref_rad=refg.edge_property('radii')
rad=ref_rad[np.random.choice(range(refg.n_edges), gtemp.n_edges)]
gtemp.add_edge_property('radii', rad)
radius=gtemp.edge_property('radii')

plt.figure()
plt.hist(radius)


## filtering out deg0 vertices
vertex_filter=np.ones(gtemp.n_vertices)
vertex_filter[np.asarray(degrees==0).nonzero()[0]]=0
gtemp=gtemp.sub_graph(vertex_filter=vertex_filter)

plt.figure()
plt.hist(gtemp.vertex_degrees()[gtemp.vertex_degrees()>0])


## add edge_geometry
gtemp.save(os.path.join(path, 'graph_new_model_temp.gt'))

gtemp=ggt.load(os.path.join(path, 'graph_new_model_temp.gt'))
res_edge_geom, res_edge_ind,res_edge_rad=assign_edge_greom_from_graph(gtemp, refg)
res_edge_rad=np.array(res_edge_rad)
np.save(os.path.join(path, 'graph_test_edge_geom_v11.npy'),res_edge_geom)
np.save(os.path.join(path, 'graph_test_edge_ind_v11.npy'),res_edge_ind)
np.save(os.path.join(path, 'graph_test_edge_rad_v11.npy'),res_edge_rad)
gtemp.set_edge_geometry(coordinates=res_edge_geom, indices=res_edge_ind, radii=res_edge_rad)

gtemp.save(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_v11.gt'))



#%% plot sub graph wth veins and arteries

#color edges
# gtemp.add_vertex_property()
edge_vein_label = gtemp.vertex_property('veins');
edge_artery_label = gtemp.vertex_property('arteries');

# edge_vein_label = vei
# edge_artery_label = art

vertex_colors = np.zeros((gtemp.n_vertices, 4))
vertex_colors[:, 3]=1
vertex_colors[:, 1]=0.8
# connectivity = gtemp.edge_connectivity();
# edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
vertex_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
vertex_colors[edge_vein_label  >0] = [0.0,0.0,0.8,1.0]

p = p3d.plot_graph_mesh(gtemp, vertex_colors=vertex_colors, n_tube_points=3);
# p3d.plot_graph_mesh(gtemp)

#%% artery/vein tracing

art=gtemp.vertex_property('arteries')
vei=gtemp.vertex_property('veins')




print(np.sum(art), np.sum(vei))
iter=30
head_v=np.asarray(vei==1).nonzero()[0]
head_a=np.asarray(art==1).nonzero()[0]
d2s=gtemp.vertex_coordinates()[:,1]

while iter>0:
    hv=[]
    ha=[]
    print(iter)
    art_pos=np.asarray(art==1).nonzero()[0]
    vei_pos=np.asarray(vei==1).nonzero()[0]
    print(head_a)
    for a in head_a:
        # print(a)
        neigh=gtemp.vertex_neighbours(a)
        d2s_neigh=d2s[neigh]
        n=neigh[np.argmax(d2s_neigh)]
        ha.append(n)
        art[n]=1

    print(head_v)
    for v in head_v:
        # print(v)
        neigh=gtemp.vertex_neighbours(v)
        d2s_neigh=d2s[neigh]
        n=neigh[np.argmax(d2s_neigh)]
        hv.append(n)
        vei[n]=1
        # for n in neigh:
        #     if n not in vei_pos:
        #         if n not in art_pos:
        #             if d2s[n]>d2s[v]:
        #                 vei[n]=1
        #                 hv.append(v)
        #                 # vei[n]=1
        #                 break
    iter=iter-1
    head_a=ha
    head_v=hv

print(np.sum(art), np.sum(vei))

gtemp.add_vertex_property('veins', vei)
gtemp.add_vertex_property('arteries', art)

#%% check out arteries and veins
arteries=gtemp.vertex_property('arteries')
veins=gtemp.vertex_property('veins')

artery=from_v_prop2_eprop(gtemp, arteries)
gtemp.add_edge_property('artery', artery)

vein=from_v_prop2_eprop(gtemp, veins)
gtemp.add_edge_property('vein', vein)


connectivity=gtemp.edge_connectivity()
arteries=gtemp.vertex_property('arteries')
veins=gtemp.vertex_property('veins')

av=np.logical_or(np.logical_and(arteries[connectivity[:,0]],veins[connectivity[:,1]]),np.logical_and(veins[connectivity[:,0]],arteries[connectivity[:,1]]))
av=av.nonzero()[0]

for v in connectivity[av].flatten():
    if veins[v]==1:
        arteries[v]=0
        veins[v]=1

gtemp.add_vertex_property('veins', veins)
gtemp.add_vertex_property('artery', arteries)


artery=from_v_prop2_eprop(gtemp, arteries)
gtemp.add_edge_property('artery', artery)

vein=from_v_prop2_eprop(gtemp, veins)
gtemp.add_edge_property('vein', vein)

gtemp.save(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_v11.gt'))


gtemp=ggt.load(os.path.join(path, 'graph_test_reco_cleaned_deg3_goemetry_v11.gt'))

## Franca Schmidt

directory = "160322"

parent_dir = "/data_SSD_2to/SimulationVasculature3D"
path = os.path.join(parent_dir, directory)
try:
    os.mkdir(path)
    print("Directory '% s' created" % directory)
except:
    print('directory aleready exists')


f, v=computeFlowFranca(path, gtemp, '')

with open(path+'/sampledict.pkl', 'rb') as fp:
    dictio = pickle.load(fp)
    f=dictio['flow']
    v=dictio['v']
    p=dictio['pressure']
    e = 1 - np.exp(-(7.3 / abs(np.asarray(f)[0])))  # abs(f)

## test values

gtemp.add_edge_property('flow', f)
gtemp.add_edge_property('veloc', v)
ps=7.3
# f=np.array(f)
# e = 1 - np.exp(-(ps / abs(f[0])))#abs(f)
# graph.add_edge_property('extracted_frac', e)
import seaborn as sns
plt.figure()
sns.set_style('white')
sns.despine()
plt.hist(p, bins=100)
plt.title('pressure')
plt.yscale('log')

plt.figure()
sns.set_style('white')
sns.despine()
plt.hist(f, bins=100)
plt.title('flow')
plt.yscale('log')

plt.figure()
sns.set_style('white')
sns.despine()
plt.hist(v, bins=100)
plt.title('v')
plt.yscale('log')

e = 1 - np.exp(-(7.3/ abs(np.asarray(f)[0])))#abs(f)
plt.figure()
sns.set_style('white')
sns.despine()
histp, binsp=np.histogram(e, bins=100, normed=True)
plt.hist(e, bins=100)
# plt.bar(binsp[:-1],histp)
plt.title('extracted fraction')
plt.yscale('log')


VecStart_x=VecStart_x[mask]
VecEnd_x=VecEnd_x[mask]
VecStart_y=VecStart_y[mask]
VecEnd_y=VecEnd_y[mask]
VecStart_z=VecStart_z[mask]
VecEnd_z=VecEnd_z[mask]

for i in range(VecStart_x.shape[0]):
    # if (abs(VecStart_x[i])<100 and abs(VecEnd_x[i]<100) and abs(VecStart_y[i]<100)  and abs(VecEnd_y[i]<100)  and abs(VecStart_z[i]<100)  and VecEnd_z[i]<100 ):
    plt.plot([VecStart_x[i] ,VecEnd_x[i]],[VecStart_y[i],VecEnd_y[i]],[VecStart_z[i],VecEnd_z[i]], c='g')

plt.xlim(0,100)
plt.ylim(0,100)
plt.set_zlim(0,100)





directories = ["081221", "101221"]

directories=['160322']
parent_dir = "/data_SSD_2to/SimulationVasculature3D"

hist_rad=[]
hist_plan=[]
hist_bp=[]
hist_iso=[]
hist_iso_rad=[]
bins=0
for i, directory in enumerate(directories):
    # gtemp=ggt.load(parent_dir+'/'+directory+'/'+'graph_new_model_temp.gt')
    gtemp=ggt.load(parent_dir+'/'+directory+'/'+'g_saved_358.gt')



    ## orientation
    import math
    limit_angle=40
    pi=math.pi


    coordinates=gtemp.vertex_coordinates()
    top2bot=np.array([0,1,0])
    x = gtemp.vertex_coordinates()[:, 0]
    y = gtemp.vertex_coordinates()[:, 1]
    z = gtemp.vertex_coordinates()[:, 2]

    connectivity = gtemp.edge_connectivity()
    # lengths = graph.edge_property('length')
    edge_vect = np.array(
        [x[connectivity[:, 1]] - x[connectivity[:, 0]], y[connectivity[:, 1]] - y[connectivity[:, 0]],
         z[connectivity[:, 1]] - z[connectivity[:, 0]]]).T

    normed_edge_vect=np.array([edge_vect[i] / np.linalg.norm(edge_vect[i]) for i in range(edge_vect.shape[0])])
    # N=np.linalg.norm(edge_vect[i])
    # print(N)
    # normed_edge_vect=normed_edge_vect[~np.isnan(normed_edge_vect)]
    rad=np.array([np.dot(top2bot.transpose(), normed_edge_vect[i]) for i in range(edge_vect.shape[0])])
    plan=np.sqrt(1-rad**2)

    r=abs(rad)
    p=abs(plan)

    r = r[~np.isnan(r)]
    p = p[~np.isnan(r)]

    edges_centers = np.array(
        [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])
    dist = edges_centers[:,1][~np.isnan(r)]

    # radiality = (r / (r + p)) > 0.5
    # planarity = (p / (r + p)) > 0.6
    # neutral = np.logical_not(np.logical_or(radiality, planarity))
    angle = np.array([math.acos(r[i]) for i in range(r.shape[0])]) * 180 / pi

    radiality = angle <  limit_angle#40
    planarity = angle >  90-limit_angle#60
    neutral = np.logical_not(np.logical_or(radiality, planarity))

    if i==0:
        radhist, radbins=np.histogram(dist[radiality], bins=10)
        planhist, radbins=np.histogram(dist[planarity], bins=radbins)
        bp_hist, bp_bin=np.histogram(y, bins=radbins)
        bins=radbins
    else:
        radhist, radbins=np.histogram(dist[radiality], bins=bins)
        planhist, radbins=np.histogram(dist[planarity], bins=bins)
        bp_hist, bp_bin=np.histogram(y, bins=bins)

    hist_bp.append(bp_hist)
    hist_rad.append(radhist)
    hist_plan.append(planhist)


    def prop_radial(limit_angle):
        A_tot=4*pi
        A_rad=2*2*pi*(1-np.cos(limit_angle))
        A_plan=(4*pi)-(2*2*pi*(1-np.cos(90 - limit_angle)))
        return A_rad/A_tot, A_plan/A_tot

    angle_range=np.arange(0, 100, 10)
    angle_range=angle_range*pi/180
    rad_prop=[]
    plan_prop=[]
    for theta in angle_range:
        R, P=prop_radial(theta)
        rad_prop.append(R)
        plan_prop.append(P)
    angle_range=np.arange(0, 100, 10)
    hist, binsisotrop=np.histogram(angle,bins=angle_range)
    hist_iso_rad.append(np.diff(rad_prop))
    hist_iso.append(hist)

from sklearn.preprocessing import normalize
import pandas as pd
import seaborn as sns
plt.figure()
Cpd_c = pd.DataFrame(normalize(hist_rad, norm='l2', axis=1)).melt()
Cpd_m = pd.DataFrame(normalize(hist_bp, norm='l2', axis=1)).melt()
Cpd_p = pd.DataFrame(normalize(hist_plan, norm='l2', axis=1)).melt()
sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_p, color='forestgreen', linewidth=2.5)
sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color='indianred', linewidth=2.5)
sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_m, color='cadetblue', linewidth=2.5)
plt.legend(['cross-flow', 'in-flow', 'bp'])

plt.figure()
bins=angle_range
bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
Cpd_m = pd.DataFrame(normalize(hist_iso_rad, norm='l2', axis=1)).melt()
Cpd_p = pd.DataFrame(normalize(hist_iso, norm='l2', axis=1)).melt()
sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_p, color='black', linewidth=2.5)
sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_m, color='indianred', linewidth=2.5)
plt.legend(['model angle distribution', 'isotropic distribution'])

plt.figure()
sns.distplot(dist[radiality], bins=10, kde=True)
sns.distplot(dist[planarity], bins=10, kde=True)
# sns.distplot(dist[neutral], bins=10)
plt.legend(['in-flow', 'cross-flow'])

plt.figure()
radhist, radbins=np.histogram(dist[radiality], bins=10)
planhist, planbins=np.histogram(dist[planarity], bins=radbins)
neuthist, neutbins=np.histogram(dist[neutral], bins=radbins)
Cpd_m = pd.DataFrame(normalize(hist_bp, norm='l2', axis=1)).melt()

sns.lineplot(radbins[:-1], radhist/(radhist+planhist+neuthist), color='cadetblue')
sns.lineplot(radbins[:-1], planhist/(radhist+planhist+neuthist),color='indianred')
# plt.twinx()
sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_m, color='black', linewidth=2.5)
# sns.lineplot(radbins[:-1], neuthist/(radhist+planhist+neuthist),color='forestgreen')
# plt.twinx()
# sns.distplot(y, hist=False, kde=False, bins=10)

plt.legend(['in-flow', 'cross-flow', 'bp'])

plt.figure()
sns.distplot(y, bins=10,kde=False)


## isotropy


def prop_radial(limit_angle):
    A_tot=4*pi
    A_rad=2*2*pi*(1-np.cos(limit_angle))
    A_plan=(4*pi)-(2*2*pi*(1-np.cos(90 - limit_angle)))
    return A_rad/A_tot, A_plan/A_tot

angle_range=np.arange(0, 100, 10)
angle_range=angle_range*pi/180
rad_prop=[]
plan_prop=[]
for theta in angle_range:
    R, P=prop_radial(theta)
    rad_prop.append(R)
    plan_prop.append(P)


plt.figure()
angle_range=np.arange(0, 100, 10)
hist, bins=np.histogram(angle,bins=angle_range)
bins=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
plt.plot(bins,hist/np.sum(hist))
plt.plot(bins, np.diff(rad_prop))
plt.legend(['model angle distribution', 'isotropic distribution'])

## sbm
import graph_tool.inference as gti
base=gtemp.base
state_sbm = gti.minimize_blockmodel_dl(base)
modules = state_sbm.get_blocks().a
gtemp.add_vertex_property('blocks_sbm', modules)
# gss4.add_vertex_property('indices', indices)
Q, Qs = modularity_measure(modules, gtemp, 'blocks_sbm')


b=state_sbm.b

bn= b.get_array()
new_cmap, randRGBcolors = rand_cmap(np.unique(bn).shape[0], type='bright', first_color_black=False,
                                    last_color_black=False, verbose=True)
n = gtemp.n_vertices
colorval = np.zeros((n, 3));
for i in range(bn.size):
    colorval[i] = randRGBcolors[int(bn[i])]
colorval=np.array(colorval)

gtemp.add_vertex_property('vertex_fill_color',colorval)#colorval

p3d.plot_graph_mesh(gtemp, vertex_colors=gtemp.vertex_property('vertex_fill_color'),  n_tube_points=3)

import pickle
with open('/data_SSD_2to/SimulationVasculature3D/160322/sampledict.pkl', 'rb') as fp:
    dictio = pickle.load(fp)
    f = dictio['flow']
    v = dictio['v']
    p = dictio['pressure']
    e = 1 - np.exp(-(7.3 / abs(np.asarray(f)[0])))  # abs(f)

work_dir='/data_SSD_2to/191122Otof'
control='2R'
g_control=ggt.load(work_dir + '/' + control +'/'+'data_graph_correcteduniverse.gt')
with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
    sampledict = pickle.load(fp)
    f_control = sampledict['flow']
    v_control = sampledict['v']
    p_control = sampledict['pressure']
    e_control = 1 - np.exp(-(7.3 / abs(np.asarray(f)[0])))  # abs(f)


degrees = g_control.vertex_degrees()
vf = np.logical_and(degrees > 1, degrees <= 4)
g_control = g_control.sub_graph(vertex_filter=vf)
g_control.add_vertex_property('p_control', np.array(p_control[0]))
g_control.add_vertex_property('f_control', f_control)
g_control.add_vertex_property('v_control', v_control)
# g_control.add_vertex_property('e_control', e_control)

g_aud=extract_AnnotatedRegion(g_control, (142, 8))
g_aud=g_aud.sub_slice((slice(1,3000),slice(3600,3650), slice(0,1000) ))
p_control_aud=g_aud.vertex_property('p_control')

plt.figure()
sns.set_style('white')
sns.kdeplot(np.asarray(p[0]),color='cadetblue', bw_adjust=5)
plt.twinx()
sns.kdeplot(np.asarray(p_control_aud), color='indianred', bw_adjust=5)
sns.despine()

plt.figure()
sns.kdeplot(np.asarray(f[0]),color='cadetblue', bw_adjust=5)
plt.twinx()
sns.kdeplot(np.asarray(f_control[0]), color='indianred', bw_adjust=5)

plt.figure()
sns.kdeplot(np.asarray(v[0]),color='cadetblue', bw_adjust=5)
plt.twinx()
sns.kdeplot(np.asarray(v_control[0]), color='indianred', bw_adjust=5)