import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import seaborn as sns
atlas_path = os.path.join(settings.resources_path, 'Atlas');
from sklearn import preprocessing

controls_2M=['2R','3R','5R', '8R']
mutants_2M=['1R','7R', '6R', '4R']
work_dir_2M='/data_SSD_2to/191122Otof'

work_dir_1M='/data_2to/otof1M'
controls_1M=[ '1w', '3w', '5w', '6w', '7w']
mutants_1M=['1k', '2k', '3k', '4k']

mutants_6M=['2R','3R','5R', '1R']
controls_6M=['7R','8R', '6R']
work_dir_6M='/data_SSD_1to/otof6months'



wds=[work_dir_1M,work_dir_2M,work_dir_6M]
ctrls=[controls_1M,controls_2M,controls_6M]
mtts=[mutants_1M, mutants_2M,mutants_6M]

reg_name=['cortex', 'striatum', 'hippocampus']


reg_name=['brain']
TP=['1 months', '2 months', '6 months']

import ClearMap.IO.IO as io
anot=io.read(os.path.join(atlas_path, 'annotation_25_full.nrrd'))
reg_ids=np.unique(anot)

region_list = [(6, 6)]  # isocortex
regions = []
R = ano.find(region_list[0][0], key='order')['name']
main_reg = region_list
sub_region = True
for r in reg_list.keys():
    l = ano.find(r, key='order')['level']
    regions.append([(r, l)])

anot_f= '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd'
reg_ids=np.unique(io.read(anot_f))
for r in reg_ids:
    regions.append([(ano.find(r, key='id')['order'],ano.find(r, key='id')['level'])])

import math
pi=math.pi
from scipy import stats
average=False
limit_angle=40
mode='bigvessels'#'bigvessels
if mode=='bigvessels':
    suffixe='bv'
elif mode=='arteryvein':
    suffixe='av'

D1=pd.DataFrame()
feat_cols = [ 'feat'+str(i) for i in range(20) ]
feat_cols.append('timepoints')
feat_cols.append('region')
feat_cols.append('condition')

normed=True
Table=[]
timepoints_temp=[]
state_temp=[]
reg_temp=[]
for k, state in enumerate([ctrls, mtts]):
    for j, controls in enumerate(state):
        print(j,wds[j],controls)
        for i, g in enumerate(controls):
            print(g)
            work_dir=wds[j]
            try:
                G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                try:
                    G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse_subregion.gt')
                except:
                    G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')


            try:
                # artery=graph.vertex_property('artery')
                # vein=graph.vertex_property('vein')
                artery=from_e_prop2_vprop(G , 'artery')
                vein=from_e_prop2_vprop(G , 'vein')
            except:
                print('no artery vertex properties')
                artery=np.logical_and(G.vertex_radii()>=4.8,G.vertex_radii()<=8)#4
                vein=G.vertex_radii()>=8
                G.add_vertex_property('artery', artery)
                G.add_vertex_property('vein', vein)
                artery=from_v_prop2_eprop(G, artery)
                G.add_edge_property('artery', artery)
                vein=from_v_prop2_eprop(G, vein)
                G.add_edge_property('vein', vein)

            degrees = G.vertex_degrees()
            vf = np.logical_and(degrees > 1, degrees <= 4)
            G = G.sub_graph(vertex_filter=vf)
            label = G.vertex_annotation();
            angle,graph = GeneralizedRadPlanorientation(G, g, 4.5, controls, mode=mode, average=average)

            coordinates=G.vertex_property('coordinates_atlas')
            edge_dist = G.edge_property('distance_to_surface')
            vertex_dist=G.vertex_property('distance_to_surface')
            radiality=angle < limit_angle#40
            G.add_edge_property('radiality', radiality)
            rad_vessels_coords=G.edge_property('distance_to_surface')[np.asarray(radiality==1).nonzero()[0]]

            for region_list in regions:

                print(region_list)
                vertex_filter = np.zeros(G.n_vertices)
                for i, rl in enumerate(region_list):
                    order, level = region_list[i]
                    print(level, order, ano.find(order, key='order')['name'])
                    label = G.vertex_annotation();
                    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                    vertex_filter[label_leveled == order] = 1;

                Table.append([np.sum(vertex_filter),TP[j],order, k])

                edge_filter=from_v_prop2_eprop(G, vertex_filter)

                timepoints_temp.append(TP[j])
                state_temp.append(k)
                reg_temp.append(ano.find(order, key='order')['id'])

                ori_rad=edge_dist[np.asarray(np.logical_and(radiality,edge_filter)==1).nonzero()[0]]
                bp_dist=vertex_dist[np.asarray(vertex_filter==1).nonzero()[0]]
                if normed:
                    histrad=preprocessing.normalize(histrad.reshape(-1, 1), norm='l2', axis=0)
                    histbp=preprocessing.normalize(histbp.reshape(-1, 1), norm='l2', axis=0)
                    Table.append(np.concatenate((np.concatenate((histrad[:,0],histbp[:,0]), axis=0), np.array([TP[j],reg])), axis=0))
                else:
                    Table.append(np.concatenate((np.concatenate((histrad,histbp), axis=0), np.array([TP[j],reg, k])), axis=0))





df = pd.DataFrame(Table,columns=feat_cols)
df['timepoints']=timepoints_temp
df['region']=reg_temp
df['condition']=state_temp
df.to_csv('/data_2to/devotof/TSNE_otof_datas_normed.csv', index=False)




data=df
# data=df[df['timepoints'] != 30]
data['reggion_name'] =  [ano.find(data['region'].values[i], key='id')['name'] for i in range(data.shape[0])]

# data_test=data[['alar' not in rn and 'roof' not in rn and 'basal' not in rn  and 'floor' not in rn for rn in data['reggion_name']]]
data_test=data


data_val=data_test[feat_cols]#[10:]

data_val[feat_cols[10:]]=data_val[feat_cols[10:]].div(data_val[feat_cols[10:]].sum(axis=1), axis=0)
data_val[feat_cols[:10]]=data_val[feat_cols[:10]].div(data_val[feat_cols[:10]].sum(axis=1), axis=0)

data_val=data_val.values


from sklearn.manifold import TSNE
import time

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, early_exaggeration=50, learning_rate=100, perplexity=80, n_iter=300)
tsne_results = tsne.fit_transform(data_val)
# np.save('/data_2to/devotof/TSNE_good_res_30_200_80.npy', tsne_results)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

data_test['tsne-2d-one'] = tsne_results[:,0]
data_test['tsne-2d-two'] = tsne_results[:,1]





data_plot=data_test[data_test["reggion_name"]!='No label']


fig=plt.figure(figsize=(16,10))

palette=sns.color_palette("husl", 3, desat=0.8)
TP=['1 months', '2 months', '6 months']
# TP=[1, 3, 5, 6, 7, 14]
for i, tp in enumerate(TP):
    for j, reg in enumerate(np.unique(data_test["region"])[0:]):
        name=ano.find(data_test['region'].values[j], key='id')['name']
        if name!='universe':
            print(name)
            sns.kdeplot(
                x="tsne-2d-one", y="tsne-2d-two",
                # hue="reggion_name",
                data=data_plot[data_plot["reggion_name"]==name][data_plot["timepoints"]==tp],
                levels=[0.9, 1],
                shade=True,
                bw_adjust=.2,
                # palette=sns.color_palette("hls", 5),
                color=palette[i],
                alpha=0.1
            )
            sns.kdeplot(
                x="tsne-2d-one", y="tsne-2d-two",
                # hue="reggion_name",
                data=data_plot[data_plot["reggion_name"]==name][data_plot["timepoints"]==tp],
                levels=[ 0.9, 0.95, 1],
                shade=False,
                bw_adjust=.2,
                # palette=sns.color_palette("hls", 5),
                color=palette[i],
                alpha=0.1
            )

g_sns=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="timepoints",
    palette=sns.color_palette("husl", 3, desat=0.8),
    data=data_plot,
    legend="full",
    alpha=0.3,
    picker = True
)
# g_sns.legend_.set_picker(True)
fig.set_picker(True)
def onpick(event):
    print("Picked!")
    print(event)
    origin = data_plot.iloc[event.ind[0]]['reggion_name']
    print('Selected item came from {}'.format(origin))
    # plt.gca().set_title('Selected item came from {}'.format(origin))

g_sns.figure.canvas.mpl_connect("pick_event", onpick)




fig=plt.figure(figsize=(16,10))
# legend="full",
# sns.legend_out=True
# nb_reg=len(regions)
nb_reg=np.unique(data_plot[data_plot["reggion_name"]!='No label']['region'].values).shape
# fig=plt.figure()
palette=sns.color_palette("hls", nb_reg[0],desat=0.8)
k=0
for i, reg in enumerate(np.unique(data_plot[data_plot["reggion_name"]!='No label']['region'])[0:]):
    name=ano.find(data_test['region'].values[i], key='id')['name']
    print(name)
    sns.kdeplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # hue="reggion_name",
        data=data_plot[data_plot["reggion_name"]==name],
        levels=[0.7, 1],
        shade=True,
        bw_adjust=.5,
        # palette=sns.color_palette("hls", 5),
        color=palette[k],
        alpha=0.1
    )
    sns.kdeplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # hue="reggion_name",
        data=data_plot[data_plot["reggion_name"]==name],
        levels=[0.7, 0.8, 0.9, 1],
        shade=False,
        bw_adjust=.5,
        # palette=sns.color_palette("hls", 5),
        color=palette[k],
        alpha=0.1
    )
    k=k+1

g_sns1=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="reggion_name",
    palette=sns.color_palette("hls", nb_reg[0], desat=0.8),
    data=data_plot[data_plot["reggion_name"]!='No label'],
    alpha=0.9,
    picker = True
)
# g_sns.legend_.set_picker(True)
fig.set_picker(True)
def onpick(event):
    print("Picked!")
    print(event)
    origin = data_plot.iloc[event.ind[0]]['reggion_name']
    print('Selected item came from {}'.format(origin))
    # plt.gca().set_title('Selected item came from {}'.format(origin))

g_sns1.figure.canvas.mpl_connect("pick_event", onpick)

# plt.show()


fig=plt.figure(figsize=(16,10))
# legend="full",
# sns.legend_out=True

# fig=plt.figure()
palette=sns.color_palette("hls", 2, desat=0.6)

for j, reg in enumerate(np.unique(data_test["region"])[0:]):
    k=0
    name=ano.find(data_test['region'].values[j], key='id')['name']
    for i, cdt in enumerate(np.unique(data_test["condition"])):
        sns.kdeplot(
            x="tsne-2d-one", y="tsne-2d-two",
            # hue="reggion_name",
            data=data_plot[data_plot["condition"]==cdt][data_plot["reggion_name"]==name],
            levels=[0.6, 1],
            bw_adjust=.2,
            shade=True,
            # palette=sns.color_palette("hls", 5),
            color=palette[k],
            alpha=0.1
        )
        sns.kdeplot(
            x="tsne-2d-one", y="tsne-2d-two",
            # hue="reggion_name",
            data=data_plot[data_plot["condition"]==cdt][data_plot["reggion_name"]==name],
            levels=[ 0.6, 0.7, 0.8, 1],
            shade=False,
            bw_adjust=.2,
            # palette=sns.color_palette("hls", 5),
            color=palette[k],
            alpha=0.1
        )
        k=k+1

g_sns2=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="condition",
    palette=sns.color_palette("hls", 2),
    data=data_plot[data_plot["reggion_name"]!='universe'],
    alpha=0.9,
    picker = True
)
# g_sns.legend_.set_picker(True)
fig.set_picker(True)
def onpick(event):
    print("Picked!")
    print(event)
    origin = data_plot.iloc[event.ind[0]]['reggion_name']
    print('Selected item came from {}'.format(origin))
    # plt.gca().set_title('Selected item came from {}'.format(origin))

g_sns2.figure.canvas.mpl_connect("pick_event", onpick)


#region specific plot

fig=plt.figure(figsize=(16,10))
name_choosen=[]
palette=sns.color_palette("husl", 3, desat=0.8)
TP=['1 months', '2 months', '6 months']
palette=sns.color_palette("husl", 3, desat=0.8)
for i, tp in enumerate(TP):
    # for i, tp in enumerate(np.unique(data_test["condition"])):
    for j, reg in enumerate(np.unique(data_test["region"])[0:]):
        name=ano.find(data_test['region'].values[j], key='id')['name']
        if 'Primary visual area' in name:
            name_choosen.append(name)
            print(name)
            sns.kdeplot(
                x="tsne-2d-one", y="tsne-2d-two",
                # hue="reggion_name",
                data=data_plot[data_plot["reggion_name"]==name][data_plot["timepoints"]==tp],
                levels=[0.5, 1],
                shade=True,
                bw_adjust=.5,
                # palette=sns.color_palette("hls", 5),
                color=palette[i],
                alpha=0.1
            )
            sns.kdeplot(
                x="tsne-2d-one", y="tsne-2d-two",
                # hue="reggion_name",
                data=data_plot[data_plot["reggion_name"]==name][data_plot["timepoints"]==tp],
                levels=[ 0.5, 0.8, 1],
                shade=False,
                bw_adjust=.5,
                # palette=sns.color_palette("hls", 5),
                color=palette[i],
                alpha=0.1
            )

for n in name_choosen:
    g_sns=sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="timepoints",
        palette=sns.color_palette("husl", 3, desat=0.8),
        data=data_plot[data_plot["reggion_name"]==n],
        legend="full",
        alpha=0.3,
        picker = True
    )
# g_sns.legend_.set_picker(True)
fig.set_picker(True)
def onpick(event):
    print("Picked!")
    print(event)
    origin = data_plot.iloc[event.ind[0]]['reggion_name']
    print('Selected item came from {}'.format(origin))
    # plt.gca().set_title('Selected item came from {}'.format(origin))

g_sns.figure.canvas.mpl_connect("pick_event", onpick)
