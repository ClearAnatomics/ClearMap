import ClearMap.Alignment.Annotation as ano


import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti
import os
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

import math
pi=math.pi
import pickle

work_dir='/data_SSD_2to/191122Otof'#'/data_SSD_2to/whiskers_graphs'
graph_nb=['1R','2R','3R','5R','6R','7R', '8R', '4R']#['44R', '30R']#, '39L']

import pickle
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open('/data_SSD_2to/181002_4/reg_list.p', 'rb') as fp:
  reg_list = pickle.load(fp)

with open('/data_SSD_2to/181002_4/atlas_volume_list.p', 'rb') as fp:
  atlas_list = pickle.load(fp)

regions=['Inferior colliculus','lateral lemniscus', 'Superior olivary complex, lateral part', 'Cochlear nuclei']
controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '8R', '4R']

def from_e_prop2_vprop(graph, property):
    e_prop = graph.edge_property(property)
    v_prop=np.zeros(graph.n_vertices)
    connectivity = graph.edge_connectivity()
    v_prop[connectivity[e_prop==1,0]]=1
    v_prop[connectivity[e_prop == 1,1]] = 1
    # graph.add_vertex_property(property, v_prop)
    return v_prop

def get_nb_radial_vessels(edge_color):
  radial=edge_color[:,2]/(edge_color[:,0]+edge_color[:,1]+edge_color[:,2])
  return(np.sum(radial>0.7))


def get_nb_parrallel_vessels(edge_color):
  planar=(edge_color[:,0]+edge_color[:,1])/(edge_color[:,2]+edge_color[:,0]+edge_color[:,1])
  print(planar.shape)
  return(np.sum(planar>0.7))


def cart2sph(x,y,z, ceval=ne.evaluate):
    """ x, y, z :  ndarray coordinates
        ceval: backend to use:
              - eval :  pure Numpy
              - numexpr.evaluate:  Numexpr """
    if x.shape[0]!=0:
        r = ceval('sqrt(x**2+y**2+z**2)')#sqrt(x * x + y * y + z * z)
        theta = ceval('arccos(z/r)*180')/pi#acos(z / r) * 180 / pi  # to degrees
        phi = ceval('arctan2(y,x)*180')/pi#*180/3.4142
        # azimuth = ceval('arctan2(y,x)')
        # xy2 = ceval('x**2 + y**2')
        # elevation = ceval('arctan2(z, sqrt(xy2))')
        # r = ceval('sqrt(xy2 + z**2)')
        rmax=np.max(r)
    else:
        print('no orientation to compute')
        r=np.array([0])
        theta=np.array([0])
        phi=np.array([0])
        rmax=1
    return phi/180, theta/180, r/rmax#, theta/180, phi/180


def getVesselOrientation(subgraph, graph):
    x = subgraph.vertex_coordinates()[:, 0]
    y = subgraph.vertex_coordinates()[:, 1]
    z = subgraph.vertex_coordinates()[:, 2]

    x_g = graph.vertex_coordinates()[:, 0]
    y_g = graph.vertex_coordinates()[:, 1]
    z_g = graph.vertex_coordinates()[:, 2]

    center = np.array([np.mean(x_g), np.mean(y_g), np.mean(z_g)])
    x = x - np.mean(x_g)
    y = y - np.mean(y_g)
    z = z - np.mean(z_g)

    spherical_coord = np.array(cart2sph(x, y, z, ceval=ne.evaluate)).T
    connectivity = subgraph.edge_connectivity()

    x_s = spherical_coord[:, 0]
    y_s = spherical_coord[:, 1]
    z_s = spherical_coord[:, 2]

    spherical_ori = np.array(
        [x_s[connectivity[:, 1]] - x_s[connectivity[:, 0]], y_s[connectivity[:, 1]] - y_s[connectivity[:, 0]],
         z_s[connectivity[:, 1]] - z_s[connectivity[:, 0]]]).T
    # orientations=preprocessing.normalize(orientations, norm='l2')

    # edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;

    # spherical_ori=np.array(cart2sph(orientations[:, 0],orientations[:, 1],orientations[:, 2], ceval=ne.evaluate)).T
    spherical_ori = preprocessing.normalize(spherical_ori, norm='l2')
    spherical_ori = np.abs(spherical_ori)

    return spherical_ori

list_ori_barrel=[]

## extract density and orientation per regions
graph_nb=['4R']
for g in graph_nb:
  print(g)
  graph=ggt.load(work_dir+'/'+g+'/'+'data_graph.gt')
  label = graph.vertex_annotation();
  brain_barrel=[]
  for e in reg_list.keys():
      reg_name=ano.find_name(e, key='order')
      if 'barrel' in reg_name:#'Primary auditory'#barrel
        print(reg_name)
        label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
        vertex_filter = label_leveled == e  # 54;
        vess_tree=graph.sub_graph(vertex_filter=vertex_filter)
        # edge_filter=vess_tree.edge_property('artery')
        # art_tree=vess_tree.sub_graph(edge_filter=edge_filter)
        art_bp=[]
        vess_bp=[]
        # art_plan=[]
        # art_rad=[]
        # vess_plan=[]
        # vess_rad=[]
        for se in reg_list[e]:
            print(ano.find_name(se, key='order'))
            label_se = vess_tree.vertex_annotation();
            label_leveled_se = ano.convert_label(label_se, key='order', value='order',
                                                 level=ano.find_level(se, key='order'))
            vertex_filter = label_leveled_se == se
            vess_tree_se=vess_tree.sub_graph(vertex_filter=vertex_filter)
            # edge_filter = vess_tree_se.edge_property('artery')
            vertex_filter=from_e_prop2_vprop(vess_tree_se, 'artery')
            art_tree_se = vess_tree_se.sub_graph(vertex_filter=vertex_filter)

            # vess_ori=getVesselOrientation(vess_tree_se, graph)
            # vess_plan.append(get_nb_parrallel_vessels(vess_ori) / vess_tree_se.n_edges)
            # vess_rad.append(get_nb_radial_vessels(vess_ori) / vess_tree_se.n_edges)
            #
            # if art_tree_se.n_vertices==0:
            #     art_rad.append(0)
            # elif art_tree_se.n_edges==0:
            #     art_rad.append(0)
            # else:
            #     art_ori = getVesselOrientation(art_tree_se, graph)
            #     art_plan.append(get_nb_parrallel_vessels(art_ori)/art_tree_se.n_edges)
            #     art_rad.append(get_nb_radial_vessels(art_ori)/art_tree_se.n_edges)



            art_bp.append(art_tree_se.n_vertices)
            vess_bp.append(vess_tree_se.n_vertices)
        brain_barrel.append(art_bp)
        brain_barrel.append(vess_bp)
        # brain_barrel.append(art_plan)
        # brain_barrel.append(art_rad)
        # brain_barrel.append(vess_plan)
        # brain_barrel.append(vess_rad)
  list_barrel.append(brain_barrel)

# print(list_ori_aud)
# print(list_ori_barrel)
print(list_barrel)
# print(list_aud)

with open(work_dir+'/list_barrel.p', 'wb') as fp:
  pickle.dump(list_barrel, fp, protocol=pickle.HIGHEST_PROTOCOL)



with open(work_dir+'/list_aud.p', 'rb') as fp:
  list_aud = pickle.load(fp)



np_layers=6
list_aud=np.array(list_aud)


mutants=[0,4,5,7]
controls=[1,2,3,6]


rad=[1,3]
plan=[0,2]

plt.figure()


vess_controls=list_aud[controls, :, :]

## plot branch point boxplot
vess_controls=list_aud[controls, :, :]


C=[]
for i in range(np_layers):
    c=vess_controls[:,1,i]
    C.append(c.tolist())

box1 = plt.boxplot(C, positions=np.arange(np_layers)+0.80, patch_artist=True, widths=0.4, showfliers=False, showmeans=True, autorange=True, meanline=True)
for patch in box1['boxes']:
    patch.set_facecolor(colors[1])

vess_controls=list_aud[mutants, :, :]

vess_controls_t=np.zeros((4, 4, 6))
for i, e in enumerate(vess_controls):
    for j, f in enumerate(e):
        if len(f)<6:
            f.append(0)
        vess_controls_t[i, j]=f
vess_controls=vess_controls_t

M=[]
for i in range(np_layers):
    c=vess_controls[:,1,i]
    M.append(c.tolist())

box2 = plt.boxplot(M, positions=np.arange(np_layers)+1.20, patch_artist=True, widths=0.4, showfliers=False, showmeans=True, autorange=True, meanline=True)
for patch in box2['boxes']:
    patch.set_facecolor(colors[0])
plt.xticks(np.arange(6), ['','l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
plt.tight_layout()
# plt.legend(['controls', 'mutants'])


plt.figure()
vess_controls=list_aud[controls, :, :]

C=[]
for i in range(np_layers):
    c=vess_controls[:,0,i]
    C.append(c.tolist())

box1 = plt.boxplot(C, positions=np.arange(np_layers)+0.80, patch_artist=True, widths=0.4, showfliers=False, showmeans=True, autorange=True, meanline=True)
for patch in box1['boxes']:
    patch.set_facecolor(colors[1])

vess_controls=list_aud[mutants, :][:]

vess_controls_t=np.zeros((4, 4, 6))
for i, e in enumerate(vess_controls):
    for j, f in enumerate(e):
        if len(f)<6:
            f.append(0)
        vess_controls_t[i, j]=f
vess_controls=vess_controls_t

M=[]
for i in range(np_layers):
    c=vess_controls[:,0,i]
    M.append(c.tolist())

box2 = plt.boxplot(M, positions=np.arange(np_layers)+1.20, patch_artist=True, widths=0.4, showfliers=False, showmeans=True, autorange=True, meanline=True)
for patch in box2['boxes']:
    patch.set_facecolor(colors[0])
plt.xticks(np.arange(6), ['','l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
plt.tight_layout()




## plot orinettaion boxplot

vess_controls=list_barrel[controls, :][:]

vess_controls_t=np.zeros((4, 4, 6))
for i, e in enumerate(vess_controls):
    for j, f in enumerate(e):
        if len(f)<6:
            f.append(0)
        vess_controls_t[i, j]=f
vess_controls=vess_controls_t

C=[]
for i in range(np_layers):
    c=vess_controls[:,3,i]/vess_controls[:,2,i]
    C.append(c.tolist())

box1 = plt.boxplot(C, positions=np.arange(np_layers)+0.80, patch_artist=True, widths=0.4, showfliers=False, showmeans=True, autorange=True, meanline=True)
for patch in box1['boxes']:
    patch.set_facecolor(colors[1])

vess_controls=list_barrel[mutants, :][:]

vess_controls_t=np.zeros((4, 4, 6))
for i, e in enumerate(vess_controls):
    for j, f in enumerate(e):
        if len(f)<6:
            f.append(0)
        vess_controls_t[i, j]=f
vess_controls=vess_controls_t

M=[]
for i in range(np_layers):
    c=vess_controls[:,3,i]/vess_controls[:,2,i]
    M.append(c.tolist())

box2 = plt.boxplot(M, positions=np.arange(np_layers)+1.20, patch_artist=True, widths=0.4, showfliers=False, showmeans=True, autorange=True, meanline=True)
for patch in box2['boxes']:
    patch.set_facecolor(colors[0])
plt.xticks(np.arange(6), ['','l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
plt.tight_layout()
# plt.legend(['controls', 'mutants'])


plt.figure()
vess_controls=list_barrel[controls, :][:]

vess_controls_t=np.zeros((4, 4, 6))
for i, e in enumerate(vess_controls):
    for j, f in enumerate(e):
        if len(f)<6:
            f.append(0)
        vess_controls_t[i, j]=f
vess_controls=vess_controls_t

C=[]
for i in range(np_layers):
    c=vess_controls[:,1,i]/vess_controls[:,0,i]
    C.append(c.tolist())

box1 = plt.boxplot(C, positions=np.arange(np_layers)+0.80, patch_artist=True, widths=0.4, showfliers=False, showmeans=True, autorange=True, meanline=True)
for patch in box1['boxes']:
    patch.set_facecolor(colors[1])

vess_controls=list_barrel[mutants, :][:]

vess_controls_t=np.zeros((4, 4, 6))
for i, e in enumerate(vess_controls):
    for j, f in enumerate(e):
        if len(f)<6:
            f.append(0)
        vess_controls_t[i, j]=f
vess_controls=vess_controls_t

M=[]
for i in range(np_layers):
    c=vess_controls[:,1,i]/vess_controls[:,0,i]
    M.append(c.tolist())

box2 = plt.boxplot(M, positions=np.arange(np_layers)+1.20, patch_artist=True, widths=0.4, showfliers=False, showmeans=True, autorange=True, meanline=True)
for patch in box2['boxes']:
    patch.set_facecolor(colors[0])
plt.xticks(np.arange(6), ['','l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
plt.tight_layout()



## plot raw quantifications for density and orientations
colors=['cadetblue','indianred', 'darkgoldenrod', 'darkorange', 'royalblue', 'blueviolet', 'forestgreen', 'lightseagreen']
plt.figure()
for i, e in enumerate(list_ori_barrel):
    print(i)
    print(e)
    # plt.figure(0)
    # x=[0,1,2,3,4,5]
    # x = np.array(x)
    # # plt.bar(x,e[1])
    # plt.plot(x , e[0], marker='o')
    # # plt.bar(x-width+(i*width),e[0], width=width)
    # plt.xticks(np.arange(6), ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
    # plt.title('arteries' + ' barrel cortex')
    #
    # plt.figure(1)
    # x = [0, 1, 2, 3, 4, 5]
    # x=np.array(x)
    # plt.plot(x, e[1], marker='o')
    # # plt.plot(x-width+(i*width), e[1],width=width)
    # # plt.bar(x, e[0])
    # plt.xticks(np.arange(6), ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
    # plt.title('vessels'+ ' barrel cortex')#graph_nb[i])

    ## orientation plots
    # plt.figure()
    plt.title(graph_nb[i] + ' barrel cortex')
    plt.subplot(211)
    plt.title(' barrel cortex' + ' vessels')
    # plt.title(graph_nb[i]+ ' barrel cortex' + ' vessels')
    x = [0, 1, 2, 3, 4, 5]
    # width=0.3
    x=np.array(x)
    plt.plot(x,np.array(e[3])/np.array(e[2]), marker='o',color=colors[i])#, width=width#np.array(e[3])/np.array(e[2]) e[1]
    plt.xticks(np.arange(6), ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
    plt.subplot(212)
    # plt.title(graph_nb[i] + ' barrel cortex'+ ' arteries')
    plt.title(' barrel cortex' + ' arteries')
    if len(e[0])<6:
        e[0].append(0)
    plt.plot(x, np.array(e[1])/np.array(e[0]), color=colors[i], marker='o')#width=width,#np.array(e[1])/np.array(e[0]) e[0
    plt.xticks(np.arange(6), ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
    plt.tight_layout()


    # plt.figure()
    # plt.title(graph_nb[i] + ' barrel cortex')

    # plt.subplot(221)
    # plt.title(graph_nb[i] + ' barrel cortex' + ' planar vessels')
    # x = [0, 1, 2, 3, 4, 5]
    # width = 0.3
    # x = np.array(x)
    # plt.bar(x, e[2], width=width)
    # plt.xticks(np.arange(6), ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
    #
    # plt.subplot(222)
    # plt.title(graph_nb[i] + ' barrel cortex' + ' radial vessels')
    # plt.bar(x, e[3], width=width)
    # plt.xticks(np.arange(6), ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
    # plt.tight_layout()
    #
    # plt.subplot(223)
    # plt.title(graph_nb[i] + ' barrel cortex' + ' planar arteries')
    # x = [0, 1, 2, 3, 4, 5]
    # width = 0.3
    # x = np.array(x)
    # plt.bar(x, e[0], width=width,  color='indianred')
    # plt.xticks(np.arange(6), ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
    #
    # plt.subplot(224)
    # plt.title(graph_nb[i] + ' barrel cortex' + ' radial arteries')
    # plt.bar(x, e[1], width=width, color='indianred')
    # plt.xticks(np.arange(6), ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'])
    # plt.tight_layout()
plt.legend(graph_nb)

plt.figure(0)
plt.legend(graph_nb)

plt.figure(1)
plt.legend(graph_nb)