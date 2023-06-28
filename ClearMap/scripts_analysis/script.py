

control6V = []
control6E = []
mutant6V = []
mutant6E = []

control2V = []
control2E = []
mutant2V = []
mutant2E = []

modules_array2C=[]
modularities2C=[]

modules_array6C=[]
modularities6C=[]

modules_array2M=[]
modularities2M=[]

modules_array6M=[]
modularities6M=[]

mutants=['2R','3R','5R', '1R']
controls=['7R','8R', '6R']
work_dir='/data_SSD_1to/otof6months'
region_list=[(142, 8), (149, 8), (128, 8), (156, 8)]
# region_list = [(54, 9), (47, 9)]  # , (75, 9)]  # barrels
for i, control in enumerate(controls):
    print(control)
    graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    vertex_filter=np.zeros(graph.n_vertices)
    for i, rl in enumerate(region_list):
        order, level = region_list[i]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
    graph = graph.sub_graph(vertex_filter=vertex_filter)
    # g=graph.base
    # state_sbm = gti.minimize_blockmodel_dl(g)
    # modules = state_sbm.get_blocks().a
    # s = np.unique(modules).shape[0]
    # graph.add_vertex_property('blocks', modules)
    # # gss4.add_vertex_property('indices', indices)
    # Q, Qs = modularity_measure(modules, graph, 'blocks')
    # # Q = get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
    # print(s, Q, state_sbm.get_B())
    # modularities6M.append(Q)
    # modules_array6M.append(s)
    control6V.append(graph.n_vertices)
    control6E.append(graph.n_edges)

for i, control in enumerate(mutants):
    print(control)
    graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    vertex_filter = np.zeros(graph.n_vertices)
    for i, rl in enumerate(region_list):
        order, level = region_list[i]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
    graph = graph.sub_graph(vertex_filter=vertex_filter)
    # g = graph.base
    # state_sbm = gti.minimize_blockmodel_dl(g)
    # modules = state_sbm.get_blocks().a
    # s = np.unique(modules).shape[0]
    # graph.add_vertex_property('blocks', modules)
    # # gss4.add_vertex_property('indices', indices)
    # Q, Qs = modularity_measure(modules, graph, 'blocks')
    # # Q = get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
    # print(s, Q, state_sbm.get_B())
    # modularities6C.append(Q)
    # modules_array6C.append(s)
    mutant6V.append(graph.n_vertices)
    mutant6E.append(graph.n_edges)


controls = ['2R', '3R', '5R', '8R']
mutants = ['1R', '7R', '6R', '4R']
work_dir = '/data_SSD_2to/191122Otof'
for i, control in enumerate(controls):
    print(control)
    graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    vertex_filter = np.zeros(graph.n_vertices)
    for i, rl in enumerate(region_list):
        order, level = region_list[i]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
    graph = graph.sub_graph(vertex_filter=vertex_filter)
    # g = graph.base
    # state_sbm = gti.minimize_blockmodel_dl(g)
    # modules = state_sbm.get_blocks().a
    # s = np.unique(modules).shape[0]
    # graph.add_vertex_property('blocks', modules)
    # # gss4.add_vertex_property('indices', indices)
    # Q, Qs = modularity_measure(modules, graph, 'blocks')
    # # Q = get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
    # print(s, Q, state_sbm.get_B())
    # modularities2C.append(Q)
    # modules_array2C.append(s)
    control2V.append(graph.n_vertices)
    control2E.append(graph.n_edges)
for i, control in enumerate(mutants):
    print(control)
    graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    vertex_filter = np.zeros(graph.n_vertices)
    for i, rl in enumerate(region_list):
        order, level = region_list[i]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
    graph = graph.sub_graph(vertex_filter=vertex_filter)
    # g = graph.base
    # state_sbm = gti.minimize_blockmodel_dl(g)
    # modules = state_sbm.get_blocks().a
    # s = np.unique(modules).shape[0]
    # graph.add_vertex_property('blocks', modules)
    # # gss4.add_vertex_property('indices', indices)
    # Q, Qs = modularity_measure(modules, graph, 'blocks')
    # # Q = get_modularity(gss4_mod, gss4_mod.vertex_property('blocks'))
    # print(s, Q, state_sbm.get_B())
    # modularities2M.append(Q)
    # modules_array2M.append(s)
    mutant2V.append(graph.n_vertices)
    mutant2E.append(graph.n_edges)

datas=[np.array(control6V),
np.array(control6E),
np.array(mutant6V),
np.array(mutant6E),
np.array(control2V),
np.array(control2E),
np.array(mutant2V),
np.array(mutant2E)]



Vdata=[np.array(control6V),
np.array(mutant6V),
np.array(control2V),
np.array(mutant2V)]




Edata=[np.array(control6E),
np.array(mutant6E),
np.array(control2E),
np.array(mutant2E)]


ControlData=np.concatenate((np.expand_dims(np.array(control2E)[:3], axis=1),np.expand_dims(np.array(control6E), axis=1)), axis=1)
mutantData=np.concatenate((np.expand_dims(np.array(mutant2E), axis=1),np.expand_dims(np.array(mutant6E), axis=1)), axis=1)
Cdata= pd.DataFrame(ControlData[:3]).melt()
Mdata = pd.DataFrame(mutantData).melt()



#
np.save('/data_SSD_1to/otof6months/ControlDataAUD.npy',ControlData)
np.save('/data_SSD_1to/otof6months/mutantDataAUD.npy',mutantData)

#
# ControlData=np.load('/data_SSD_1to/otof6months/ControlDataSSp.npy')
# mutantData=np.load('/data_SSD_1to/otof6months/mutantDataSSp.npy')

plt.figure()
sns.set_style('white')
sns.despine()
sns.lineplot(x="variable", y="value", err_style='bars', data=Cdata, color="cadetblue", linewidth=2.5)
sns.lineplot(x="variable", y="value", err_style='bars', data=Mdata, color='indianred')
plt.xlim([-0.25, 1.25])
plt.xticks([0,1], ['2 months', '6 months'], size='x-large')
plt.yticks(size='x-large')
plt.title('AUD region', size='x-large')
#
# plt.figure()
# sns.set_style('white')
# ax =sns.boxplot(data=Vdata)
# mybox = ax.artists[0]
# mybox.set_facecolor('cadetblue')
#
# mybox = ax.artists[1]
# mybox.set_facecolor('firebrick')
#
# mybox = ax.artists[2]
# mybox.set_facecolor('lightblue')
#
# mybox = ax.artists[3]
# mybox.set_facecolor('indianred')
#
# plt.xticks([0,1,2,3], ['6months c', '6months m', '2months c', '2months m'])
#
#




plt.figure()
sns.set_style('white')
ax =sns.boxplot(data=Vdata)
mybox = ax.artists[0]
mybox.set_facecolor('cadetblue')

mybox = ax.artists[1]
mybox.set_facecolor('firebrick')

mybox = ax.artists[2]
mybox.set_facecolor('lightblue')

mybox = ax.artists[3]
mybox.set_facecolor('indianred')

plt.xticks([0,1,2,3], ['6months c', '6months m', '2months c', '2months m'])




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
      print(ano.find(l[0], key='id')['name'], 'volume: ',np.sum(atlas==l[0]))
      val=val+(np.sum(atlas==l[0]))
    return val*1.5625e-5 #convrsion from atlas voxel to mm3


def get_e_bp_density(region, graph, volume):
    id, level = region
    order=ano.find_order(id)
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


reg_leaves=[]
region_list=[(6,6), (1006,3), (580,5),(650,5),(724,5),(811,4),(875,4), (463,6),(388,6)]#[(1006,3), (580,5),(650,5),(724,5),(811,4),(875,4), (6,6),(463,6),(388,6)]
for reg in region_list:
    global leafArray
    leafArray = []
    order, level = reg
    name=ano.find_name(order, key='order')
    get_child_tree(data_dict, name)
    reg_leaves.append(leafArray)


import pandas as pd
import json
with open('/home/sophie.skriabine/Projects/clearVessel_New/ClearMap/ClearMap/Resources/Atlas/annotation.json') as json_data:
    data_dict = json.load(json_data)['msg']
    print(data_dict)
leafArray = []

del leafArray
for data in data_dict:
    global leafArray
    leafArray=[]
    tree = data#json.loads(data.strip())
    parseTree(tree)
    #somehow walk through the tree and find leaves
    print ("")
    for each in leafArray:
        print(each)

atlas=io.read('/home/sophie.skriabine/Projects/clearvessel-custom/ClearMap/Resources/Atlas/annotation_25_cropped.nrrd')

controls=['2R','3R','5R', '8R']#['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'
Ls=[]
# brain_list=['190506_6R','190506_3R', '190506_4R']
region_list = [(1,1), (1006, 3), (580, 5), (650, 5), (724, 5), (811, 4), (875, 4), (6, 6), (463, 6), (388, 6)]
region_list = [[(1,1)], [(1006, 3)], [(580, 5)], [(650, 5)], [(724, 5)], [(811, 4)], [(875, 4)], [(6, 6)], [(463, 6)], [(388, 6)],[(127,7)], [(40,8)], [(163,7)], [(13,7)]]
# region_list=[(127,7), (40,8), (163,7), (13,7)]
region_list = [(1006, 3), (580, 5), (650, 5), (724, 5), (811, 4), (875, 4), (6, 6), (463, 6), (388, 6)]
for reg in region_list:
    global leafArray
    leafArray = []
    order, level = reg
    name=ano.find_name(order, key='order')
    get_child_tree(data_dict, name)
    reg_leaves.append(leafArray)

controls=['190506_6R','190506_3R', '190506_4R','190506_7L']
length=[]
L_density=[]
BP_density=[]
volumes=[]
Leafs=[]
subregion=True
for control in controls:
    gts = ggt.load('/data_SSD_2to/' + control + '/data_graph_reduced_transformed.gt')
    # gts = ggt.load(work_dir + '/' + control + '/' + 'data_graph.gt')#data_graph_correcteduniverse
    Nb=0
    for region in region_list:
        order, level = region
        # order=ano.find(id, key='id')['order']
        id = ano.find(order, key='order')['id']
        print(level, order, id, ano.find(order, key='order')['name'])

        # reg_leaves=[]
        # del leafArray
        global leafArray
        leafArray = []

        # order, level = region
        name = ano.find_name(order, key='order')
        get_child_tree(data_dict, name)
        # reg_leaves.append(leafArray)


        if subregion:
            for leaf in leafArray:
                # try:
                volume=get_volume_region([leaf], atlas)#leafArray
                print(leaf, volume)
                if volume>0:
                    vertex_filter = np.zeros(gts.n_vertices)
                    label = gts.vertex_annotation();
                    # for reg in [leaf]:
                    idl, levell = leaf
                    orderl=ano.find(idl, key='id')['order']
                    print(ano.find(idl, key='id')['name'])
                    label_leveled = ano.convert_label(label, key='order', value='order', level=levell)
                    if order<=9000:
                        print(levell, orderl, ano.find(orderl, key='order')['name'])
                        vertex_filter[label_leveled == orderl]=1;

                        gss4 = gts.sub_graph(vertex_filter=vertex_filter)
                        # coordinates=gss4.edge_geometry_property('coordinates_atlas')
                        # indices=gss4.edge_property('edge_geometry_indices')
                        coordinates = gss4.edge_geometry_property('coordinates')
                        indices = gss4.edge_property('edge_geometry_indices')
                        bp=gss4.n_vertices


                        L=0
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
                        if control=='2R':
                            Leafs.append(orderl)
                        Nb=Nb+1

        else:
            volume = get_volume_region(leafArray, atlas)  # leafArray
            print(volume)
            if volume > 0:
                label = gts.vertex_annotation();
                # for reg in leafArray:
                # order, level = reg
                print(ano.find(order, key='order')['name'])
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                if order <= 9000:
                    print(level, order, ano.find(order, key='order')['name'])
                    vertex_filter[label_leveled == order] = 1;

                    gss4 = gts.sub_graph(vertex_filter=vertex_filter)
                    # coordinates = gss4.edge_geometry_property('coordinates_atlas')
                    # indices = gss4.edge_property('edge_geometry_indices')
                    coordinates = gss4.edge_geometry_property('coordinates_atlas')
                    indices = gss4.edge_property('edge_geometry_indices')

                    bp = gss4.n_vertices

                    L = 0
                    for i, ind in enumerate(indices):
                        diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
                        L = L + np.sum(np.linalg.norm(diff, axis=1))
                    Lmm = L * 0.000025
                    print(Lmm)  # m
                    print(Lmm / volume)  # m/mm3
                    print(bp / volume)  # bp/mm3
                    volumes.append(volume)
                    length.append(Lmm)
                    L_density.append(Lmm / volume)
                    BP_density.append(bp / volume)
                    if control == '2R':
                        Leafs.append(orderl)
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

np.save('/data_SSD_1to/stata/old2_length.npy', length)
np.save('/data_SSD_1to/stata/old2_L_density.npy', L_density)
np.save('/data_SSD_1to/stata/old2_volumes.npy', volumes)
np.save('/data_SSD_1to/stata/old2_BP_density.npy', BP_density)
np.save('/data_SSD_1to/stata/old2_leafs.npy', Leafs)



length=np.load('/data_SSD_1to/stata/old2_length.npy')
L_density=np.load('/data_SSD_1to/stata/old2_L_density.npy')
volumes=np.load('/data_SSD_1to/stata/old2_volumes.npy')
BP_density=np.load('/data_SSD_1to/stata/old2_BP_density.npy')
Leafs=np.load('/data_SSD_1to/stata/old2_leafs.npy')

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

reg_leaves=[]
for region in region_list:
    order, level = region
    # order=ano.find(id, key='id')['order']
    id = ano.find(order, key='order')['id']
    print(level, order, id, ano.find(order, key='order')['name'])

    # reg_leaves=[]
    # del leafArray
    global leafArray
    leafArray = []

    # order, level = region
    name = ano.find_name(order, key='order')
    get_child_tree(data_dict, name)
    # reg_leaves.append(leafArray)

    # vertex_filter = np.zeros(gts.n_vertices)
    for leaf in leafArray:
        for reg in [leaf]:
            id, level = reg
            order=ano.find(id, key='id')['order']
            if order <= 9000:
                volume = get_volume_region([leaf], atlas)  # leafArray
                # print(volume)
                if volume > 0:
                    reg_leaves.append(order)

print(len(reg_leaves))
reg_leaves=np.array(reg_leaves)
Leafs=np.array(Leafs)
reg_leaves_corrcted_c=reg_leaves[correct_values]#Leafs

# legend=[]

orders=[]
for i, r in enumerate(reg_leaves_corrcted_c):
    orders.append(ano.find(r, key='order')['order'])
    print(ano.find(r, key='order')['name'])

orders=np.array(orders)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.set_style('white')
sns.despine()
indices=np.argsort(reg_leaves_corrcted_c)
names=[]
ticks=[]
for i, ind in enumerate(indices):
    r=reg_leaves_corrcted_c[ind]
    # for j, sub_r in enumerate(reg_leaves[i]):
    #     print(i, real_length_c[i], ano.find_color(r))
    plt.bar(i, real_BP_density[np.asarray(reg_leaves_corrcted_c==r).nonzero()[0]], color=ano.find(r, key='order')['rgb'], alpha=1.0)
    ticks.append(ano.find(r, key='order')['acronym'])
    names.append(ano.find(r, key='order')['acronym'])
# plt.yscale('linear')
major_ticks = np.arange(0, 150000, 10000)
# minor_ticks = np.arange(0, 150000, 1000)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)
plt.yticks(major_ticks)
# plt.yticks(ax.get_yticks(),visible=True)
plt.xticks(np.arange(orders.shape[0]), ticks, rotation=90, fontsize=10)
plt.title('graph old')



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





import numpy as np
from scipy.spatial import Delaunay

points = np.random.rand(30, 2)
tri = Delaunay(points)

p = tri.points[tri.vertices]

# Triangle vertices
A = p[:,0,:].T
B = p[:,1,:].T
C = p[:,2,:].T

# See http://en.wikipedia.org/wiki/Circumscribed_circle#Circumscribed_circles_of_triangles
# The following is just a direct transcription of the formula there
a = A - C
b = B - C

def dot2(u, v):
    return u[0]*v[0] + u[1]*v[1]

def cross2(u, v, w):
    """u x (v x w)"""
    return dot2(u, w)*v - dot2(u, v)*w

def ncross2(u, v):
    """|| u x v ||^2"""
    return sq2(u)*sq2(v) - dot2(u, v)**2

def sq2(u):
    return dot2(u, u)

cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2*ncross2(a, b)) + C

# Grab the Voronoi edges
vc = cc[:,tri.neighbors]
vc[:,tri.neighbors == -1] = np.nan # edges at infinity, plotting those would need more work...

lines = []
lines.extend(zip(cc.T, vc[:,:,0].T))
lines.extend(zip(cc.T, vc[:,:,1].T))
lines.extend(zip(cc.T, vc[:,:,2].T))

# Plot it
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

lines = LineCollection(lines, edgecolor='k')

plt.figure()
plt.plot(points[:,0], points[:,1], '.', color='r')
plt.plot(cc[0], cc[1], '*')
plt.gca().add_collection(lines)
plt.axis('equal')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()