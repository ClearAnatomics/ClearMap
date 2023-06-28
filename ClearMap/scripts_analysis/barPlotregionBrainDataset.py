import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt

import pandas as pd
import json
with open('/home/sophie.skriabine/Projects/clearVessel_New/ClearMap/ClearMap/Resources/Atlas/annotation.json') as json_data:
    data_dict = json.load(json_data)['msg']
    print(data_dict)

def parseTree(obj):
    # print(obj)
    # if len(obj["children"]) == 0:
    #     leafArray.append(obj['id'])
    # else:
    leafArray.append((obj['id'], ano.find_level(obj['id'])))

    for child in obj["children"]:
        parseTree(child)

def get_child_tree(data_dict, reg_name):

    for data in data_dict:

        if data['name']==reg_name:
            print(data['name'])
            tree = data  # json.loads(data.strip())
            leafArray.append((tree['id'], ano.find_level(tree['id'])))
            for child in tree["children"]:
                parseTree(child)
        # for child in data["children"]:
        #     print(child)
        get_child_tree(data["children"], reg_name)

def get_volume_region(region_leaves, atlas):
    val=0
    for l in region_leaves:
        id=ano.find(l[0], key='order')['id']
        print(ano.find(l[0], key='order')['name'], 'volume: ',np.sum(atlas==id))
        val=val+(np.sum(atlas==id))
    return val*1.5625e-5 #convrsion from atlas voxel to mm3


def get_e_bp_density(region, graph, volume):
    order, level = region
    # order=ano.find_order(id)
    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter = label_leveled == order;

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    arteries=gss4.edge_property('artery')
    deg=np.sum(gss4.vertex_degrees()>2)
    return deg/volume, gss4.n_edges/volume, np.sum(arteries)/volume

def ttest_DataFrame():
    controls=pd.read_csv('/home/sophie.skriabine/Pictures/general/basics/densities.csv')
    pval=pd.DataFrame()
    pval=pd.merge(controls, datas, left_on = 'orders', right_on = 'orders', how = 'inner')

    import scipy.stats
    pvalues=[]
    ttest=[]
    for i in pval.index:
        ttest.append(stats.ttest_ind_from_stats(pval['vessels mean_x'][i], pval['vessels std_x'][i], 3, pval['vessels std_y'][i],pval['vessels mean_y'][i], 3)[0])
        pvalues.append(stats.ttest_ind_from_stats(pval['vessels mean_x'][i], pval['vessels std_x'][i],3,pval['vessels std_y'][i],pval['vessels mean_y'][i],3)[1])

    pval['ttest']=ttest
    pval['pvalues']=pvalues

    df = pd.DataFrame(ttest, columns=['ttest'])
    sstat = df
    sstat['pvalues']=pvalues
    sstat['acronym']=pval['acronym_x']
    sstat['full_name']=pval['full_name']
    sstat['orders']=pval['orders']
    sstat.to_csv(path_or_buf='/home/sophie.skriabine/Pictures/general/basics/otof_vs_controls_ttest.csv')
    result = sstat.sort_values('pvalues', ascending=1)


import pandas as pd

import json
# st='dev'
# print('dev brain')
# ano.initialize(label_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',
#                extra_label = None, annotation_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotationADMBA_thresholded.tif')

# work_dir='/data_2to/p1'
# controls=['1a', '1b', '1d']#['2', '3']
#
# work_dir='/data_2to/p5'
# controls=['5a', '5b']#['2', '3']

mutants_6M=['2R','3R','5R', '1R']
controls_6M=['7R','8R', '6R']
work_dir_6M='/data_SSD_1to/otof6months'

controls_2M=['2R','3R','5R', '8R']
mutants_2M=['1R','7R', '6R', '4R']
work_dir_2M='/data_SSD_2to/191122Otof'


work_dir_10M='/data_SSD_2to/211019_otof_10m'
mutants_10M=['1k', '2k','3k', '6k']#456 not annotated ?
controls_10M=['7w', '9w', '10w', '12w', '13w']

work_dir_7M='/data_SSD_2to/degradationControls/7M'
controls_7M=['467', '468', '469']

work_dir_2M='/data_SSD_2to/degradationControls/2M'
controls_2M=['3R', '4R', '5R']


work_dir_1M='/data_SSD_2to/fluoxetine2'
controls_1M=['1', '2', '3', '4','5']

work_dir_3M='/data_SSD_2to/whiskers_graphs/new_graphs'
controls_3M=['142L','158L','162L', '164L']

import tifffile
# atlas=tifffile.imread('/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotationADMBA_thresholded.tif')
# atlas=io.read(ano.default_annotation_file)
atlas=io.read('/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd')
leafArray=[]
names=[]
annotated_region=np.unique(atlas)
for ar in annotated_region[2:]:
    leafArray.append(ar)
    names.append(ano.find(ar, key='id')['acronym'])

fuse=['VISC', 'SSp-bfd', 'SSp-n', 'SSs', 'AUD', 'MO', 'VIS', 'GU', 'SSp-ll', 'SSp-m', 'SSp-ul', 'SSp-tr', 'ACA', 'PL', 'ILA',
      'ORB', 'OLF', 'TEa', 'PERI', 'RSP', 'ECT', 'AI']
leafArray_add=[]
names_add=[]

for f in fuse:
    print(f)
    loc=np.asarray([f in n for n in names]).nonzero()[0]
    print(loc)
    leafArray_add.append([leafArray[l] for l in loc])
    print(leafArray_add)
    names_add.append(f)

    leafArray=np.delete(leafArray, loc)
    names=np.delete(names, loc)


leafArray_new=[]


for ar in leafArray:
    leafArray_new.append([[ano.find(ar, key='id')['order'],ano.find(ar, key='id')['level']]])

# for f in fuse:
#     leafArray_new.append([(ano.find(f, key='acronym')['order'],ano.find(f, key='acronym')['level'])])

for la in leafArray_add:
    print(la)
    print([(ano.find(ar, key='id')['order'],ano.find(ar, key='id')['name']) for ar in la])
    leafArray_new.append([[ano.find(ar, key='id')['order'],ano.find(ar, key='id')['level']] for ar in la])

# names=np.array(leafArray_new)[:, 2]

names=names.tolist()
for na in names_add:
    names.append(na)

leafArray=leafArray.tolist()
for ar in annotated_region[2:]:
    leafArray.append(ar)
    names.append(ano.find(ar, key='id')['acronym'])


work_dir=work_dir_3M
controls=controls_3M


leafArray=leafArray_new.copy()

length=[]
L_density=[]
BP_density=[]
volumes=[]
Leafs=[]
subregion=True
for control in controls:
    try:
        gts = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    except:
        gts = ggt.load(work_dir + '/' + control+ '/' + str(control)+'_graph_correcteduniverse.gt')
    # gts = ggt.load(work_dir + '/' + control + '/' + 'data_graph.gt')#data_graph_correcteduniverse
    Nb=0

    for leaf in leafArray:
        vertex_filter = np.zeros(gts.n_vertices)
        volume=0
        L=0
        for l in leaf:
            # try:
            new_vol=get_volume_region([l], atlas)
            volume=volume+new_vol#leafArray/2
            print(leaf, volume)
            if new_vol>0:

                label = gts.vertex_annotation();
                # for reg in [leaf]:
                orderl, levell = l
                idl=ano.find(orderl, key='order')['id']
                print(ano.find(orderl, key='order')['name'])
                label_leveled = ano.convert_label(label, key='order', value='order', level=levell)

                print(levell, orderl, ano.find(orderl, key='order')['name'])
                vertex_filter[label_leveled == orderl]=1;

        gss4 = gts.sub_graph(vertex_filter=vertex_filter)
        # coordinates=gss4.edge_geometry_property('coordinates_atlas')
        # indices=gss4.edge_property('edge_geometry_indices')
        coordinates = gss4.edge_geometry_property('coordinates')
        indices = gss4.edge_property('edge_geometry_indices')
        bp=gss4.n_vertices



        for i, ind in enumerate(indices):
            diff=np.diff(coordinates[ind[0]:ind[1]], axis=0)
            L=L+np.sum(np.linalg.norm(diff, axis=1))
        Lmm=L*0.000025
        print(Lmm)#m
        print(Lmm / volume)#m/mm3
        print(bp / volume)#bp/mm3
        volumes.append(volume)
        length.append(Lmm)
        L_density.append(Lmm/volume)
        BP_density.append(bp/volume)
        # if control=='2R':
        Leafs.append(orderl)
        Nb=Nb+1


            # except:
            #     print('error')
    # Ls.append(length)
# Ls=np.array(Ls)


# np.save('/data_SSD_1to/stata/length.npy', length)
# np.save('/data_SSD_1to/stata/L_density.npy', L_density)
# np.save('/data_SSD_1to/stata/volumes.npy', volumes)
# np.save('/data_SSD_1to/stata/BP_density.npy', BP_density)
# np.save('/data_SSD_1to/stata/leafs.npy', Leafs)
#
# length=np.array(length).reshape((len(controls),Nb))#len(region_list)
# L_density=np.array(L_density).reshape((len(controls),Nb))
# BP_density=np.array(BP_density).reshape((len(controls),Nb))
# volumes=np.array(volumes).reshape((len(controls),Nb))

# np.save(work_dir+'/old2_length.npy', length)
# np.save(work_dir+'/old2_L_density.npy', L_density)
# np.save(work_dir+'/old2_volumes.npy', volumes)
# np.save(work_dir+'/old2_BP_density.npy', BP_density)
# np.save(work_dir+'/old2_leafs.npy', Leafs)

# np.save(work_dir+'/mutants_length.npy', length)
# np.save(work_dir+'/mutants_L_density.npy', L_density)
# np.save(work_dir+'/mutants_volumes.npy', volumes)
# np.save(work_dir+'/mutants_BP_density.npy', BP_density)
# np.save(work_dir+'/mutants_leafs.npy', Leafs)



np.save(work_dir+'/controls_length.npy', length)
np.save(work_dir+'/controls_L_density.npy', L_density)
np.save(work_dir+'/controls_volumes.npy', volumes)
np.save(work_dir+'/controls_BP_density.npy', BP_density)
np.save(work_dir+'/controls_leafs.npy', Leafs)
#







controls_2M=['2R','3R','5R', '8R']
mutants_2M=['1R','7R', '6R', '4R']
work_dir_2M='/data_SSD_2to/191122Otof'

volumes=np.load(work_dir_2M+'/controls_volumes.npy')

work_dir=work_dir_2M
controls=controls_2M
BP_density=np.load(work_dir+'/controls_BP_density.npy')
Leafs=np.load(work_dir+'/controls_leafs.npy')[2:]
Nb=int(len(BP_density)/len(controls))
BP_density=np.array(BP_density).reshape((len(controls),Nb))[:,2:]
correct_values1=np.logical_and(np.mean(BP_density, axis=0)>0 , np.mean(volumes, axis=0)>0.1)#np.where(bp_den<100 and tort<500)
correct_values=np.asarray(correct_values1).nonzero()[0]#np.where(correct_values1==1)
real_BP_density=np.mean(BP_density[:, correct_values], axis=0)
real_BP_density_2M_c=real_BP_density



work_dir=work_dir_10M
controls=controls_10M
BP_density=np.load(work_dir+'/controls_BP_density.npy')
Nb=int(len(BP_density)/len(controls))
BP_density=np.array(BP_density).reshape((len(controls),Nb))
real_BP_density=np.mean(BP_density[:, correct_values], axis=0)
real_BP_density_10M_c=real_BP_density


work_dir=work_dir_10M
controls=mutants_10M
BP_density=np.load(work_dir+'/mutants_BP_density.npy')
Nb=int(len(BP_density)/len(controls))
BP_density=np.array(BP_density).reshape((len(controls),Nb))
real_BP_density=np.mean(BP_density[:, correct_values], axis=0)
real_BP_density_10M_m=real_BP_density


work_dir=work_dir_2M
controls=mutants_2M
BP_density=np.load(work_dir+'/mutants_BP_density.npy')
Nb=int(len(BP_density)/len(controls))
BP_density=np.array(BP_density).reshape((len(controls),Nb))[:,2:]
real_BP_density=np.mean(BP_density[:, correct_values], axis=0)
real_BP_density_2M_m=real_BP_density





#
# work_dir=work_dir_7M
# controls=controls_7M
# BP_density=np.load(work_dir+'/controls_BP_density.npy')
# Nb=int(len(BP_density)/len(controls))
# BP_density=np.array(BP_density).reshape((len(controls),Nb))
# real_BP_density=np.mean(BP_density[:, correct_values], axis=0)
# real_BP_density_7M_c=real_BP_density


real_BP_density_mutants=(real_BP_density_2M_m-real_BP_density_10M_m)/real_BP_density_2M_m

real_BP_density_controls=(real_BP_density_2M_c-real_BP_density_10M_c)/real_BP_density_2M_c

degradation=real_BP_density_controls-real_BP_density_mutants#/real_L_density_controls



length=np.load(work_dir+'/mutants_length.npy')
L_density=np.load(work_dir+'/mutants_L_density.npy')
volumes=np.load(work_dir+'/mutants_volumes.npy')
BP_density=np.load(work_dir+'/mutants_BP_density.npy')
Leafs=np.load(work_dir+'/mutants_leafs.npy')



work_dir=work_dir_7M
controls=controls_7M

length=np.load(work_dir+'/controls_length.npy')
L_density=np.load(work_dir+'/controls_L_density.npy')
volumes=np.load(work_dir+'/controls_volumes.npy')
BP_density=np.load(work_dir+'/controls_BP_density.npy')
Leafs=np.load(work_dir+'/controls_leafs.npy')

Nb=int(len(length)/len(controls))

length=np.array(length).reshape((len(controls),Nb))#len(region_list)
L_density=np.array(L_density).reshape((len(controls),Nb))
BP_density=np.array(BP_density).reshape((len(controls),Nb))
volumes=np.array(volumes).reshape((len(controls),Nb))

#
# for r in range(Ls.shape[1]):
#     print(ano.find_name(region_list[r][0], key='order'), np.mean(Ls[:,r]), '+/-',np.std(Ls[:,r]) )


correct_values1=np.logical_and(np.mean(length, axis=0)>0 , np.mean(volumes, axis=0)>0.1)#np.where(bp_den<100 and tort<500)
# correct_values=np.where(bp_den_c>0)
# correct_values1=np.logical_and(correct_values1,np.mean(L_density,axis=0)<10)
correct_values=np.asarray(correct_values1).nonzero()[0]#np.where(correct_values1==1)
# correct_values=correct_values[0]

real_length_c=np.mean(length[:,correct_values], axis=0)
real_volumes_c=np.mean(volumes[:, correct_values], axis=0)
real_L_density=np.mean(L_density[:,correct_values], axis=0)
real_BP_density=np.mean(BP_density[:, correct_values], axis=0)

std_length_c=np.std(length[:,correct_values], axis=0)
std_volumes_c=np.std(volumes[:, correct_values], axis=0)
std_L_density=np.std(L_density[:,correct_values], axis=0)
std_BP_density=np.std(BP_density[:, correct_values], axis=0)
#
#
# real_L_density_10M_m=real_L_density
# real_L_density_2M_m=real_L_density
# real_L_density_mutants=(real_L_density_2M_m-real_L_density_10M_m)/real_L_density_2M_m


real_L_density_10M_c=real_L_density
real_L_density_2M_c=real_L_density
real_L_density_controls=(real_L_density_2M_c-real_L_density_10M_c)/real_L_density_2M_c

degradation=real_L_density_controls-real_L_density_mutants#/real_L_density_controls


##JSP
# reg_leaves=[]
# for region in region_list:
#     order, level = region
#     # order=ano.find(id, key='id')['order']
#     id = ano.find(order, key='order')['id']
#     print(level, order, id, ano.find(order, key='order')['name'])
#
#     # reg_leaves=[]
#     # del leafArray
#     global leafArray
#     leafArray = []
#
#     # order, level = region
#     name = ano.find_name(order, key='order')
#     get_child_tree(data_dict, name)
#     # reg_leaves.append(leafArray)


#     reg_leaves=[]
#     # vertex_filter = np.zeros(gts.n_vertices)
#     for leaf in leafArray:
#         for reg in [leaf]:
#             order, level = reg
#             # order=ano.find(id, key='id')['order']
#             if order <= 9000:
#                 volume = get_volume_region([leaf], atlas)  # leafArray
#                 # print(volume)
#                 if volume > 0:
#                     reg_leaves.append(order)
#
# print(len(reg_leaves))
# reg_leaves=np.array(reg_leaves)
Leafs=np.array(leafArray)
reg_leaves_corrcted_c=Leafs[correct_values]#Leafs
names_c=np.array(names)[correct_values]
# legend=[]



orders=[]
for i, r in enumerate(names_c):
    orders.append(ano.find(r, key='acronym')['order'])
    print(ano.find(r, key='acronym')['name'])

orders=np.array(orders)

import seaborn as sns


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.set_style('white')
sns.despine()
indices=np.argsort(orders)
# names=[]
names_cs=names_c[indices]
ticks=[]
for i, ind in enumerate(indices):
    # r=reg_leaves_corrcted_c[ind][0][0]
    r=orders[ind]
    print(r,ano.find(r, key='order')['name'],names_c[ind],real_L_density_controls[np.asarray(orders==r).nonzero()[0]])
    # for j, sub_r in enumerate(reg_leaves[i]):
    #     print(i, real_length_c[i], ano.find_color(r))
    plt.bar(i, real_L_density_controls[np.asarray(orders==r).nonzero()[0]], color=ano.find(r, key='order')['rgb'], alpha=1.0)
    ticks.append(ano.find(r, key='order')['name'])
    # names.append(ano.find(r, key='order')['acronym'])

plt.xticks(np.arange(orders.shape[0]), names_cs, rotation=90, fontsize=10)#ticks
plt.title('degradation')

plt.ylim(-5,5)


datas=pd.DataFrame()
# names=[ano.find(region[0], key='order')['name'] for region in region_list]
# names=['brain', 'Cerebellum', 'Straitum','Thalamus','Hypothalamus', 'Midbrain','Hindbrain','Isocortex','Hippocampal formation', 'Olfactory','AUD', 'SSp1', 'Vis', 'motor regions']
# names=['auditory', 'SSp1', 'Vis', 'motor regions']
datas['regions']=names
datas['acronym']=ticks
datas['volume']=np.mean(volumes[:,correct_values], axis=0)[indices]
# datas['volume std']=np.std(volumes, axis=0)

datas['cum_length mean']=np.mean(length[:,correct_values], axis=0)[indices]
datas['cum_length std']=np.std(length[:,correct_values], axis=0)[indices]

datas['length density mean']=np.mean(L_density[:,correct_values], axis=0)[indices]
datas['length density std']=np.std(L_density[:,correct_values], axis=0)[indices]

datas['bp density mean']=np.mean(BP_density[:,correct_values], axis=0)[indices]
datas['bp density std']=np.std(BP_density[:,correct_values], axis=0)[indices]

datas.to_csv(path_or_buf='/home/sophie.skriabine/Documents/erratum_sub_region_old_graphs.csv')

