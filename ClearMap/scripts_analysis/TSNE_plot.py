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
controlsP14=['14a','14b', '14c']#['2', '3']


# reg_name=['cortex', 'striatum', 'hippocampus']


reg_name=['brain']
TP=[1, 5, 30, 3, 7, 6, 14]


workdirs=[ work_dirP1, work_dirP5, work_dirAdult,work_dirP3,work_dirP7,work_dirP6,work_dirP14]
controlslist=[ controlsP1, controlsP5, controlsAdult,controlsP3,controlsP7,controlsP6,controlsP14]

anot_f=['/data_2to/alignement/atlases/new_region_atlases/P1/smoothed_anot_P1.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P3/smoothed_anot_P3_V6.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P7/smoothed_anot_P7.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed_rescale.nrrd']

import ClearMap.IO.IO as io
# anot=io.read('/data_2to/pix/anoP4.tif')
# anot=anot-32768
# reg_ids=np.unique(anot)
reg_ids=np.unique(io.read(anot_f[0]))
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



normed=True

D1=pd.DataFrame()
feat_cols = [ 'feat'+str(i) for i in range(20) ]
feat_cols.append('timepoints')
feat_cols.append('region')

get_reg=0
Table=[]
timepoints_temp=[]
reg_temp=[]
for j, controls in enumerate(controlslist):
    print(j,workdirs[j],controls)
    st='dev'
    print('dev brain')
    # ano.initialize(label_file = '/data_2to/pix/region_ids_test_ADMBA_audbarmo.json',
    #                extra_label = None, annotation_file = '/data_2to/pix/anoP4.tif')
    annotation_file=anot_f[j]
    ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                   annotation_file = annotation_file)
    # regions=regions=[(0,0),(451,10), (452, 10), (450,10)]#[(0,0),(449,9)]#, (567,9),(259,9)]
    # if TP[j]==14 or TP[j]==6:
    #     regions=[(0,0)]
    for i, g in enumerate(controls):
        print(g)
        work_dir=workdirs[j]

        try:
            G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse_smoothed.gt')
        except:
            try:
                G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
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

        #extract isocortex
        # if get_reg==0:
        #     order, level=(6,6)
        #     vertex_filter = np.zeros(G.n_vertices)
        #     print(level, order, ano.find(6, key='order')['name'])
        #     label = G.vertex_annotation();
        #     label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        #     vertex_filter[label_leveled == order] = 1;
        #     G = G.sub_graph(vertex_filter=vertex_filter)
        #
        #     label=G.vertex_annotation();
        #     reg_ids=[ano.find(u_l, key='order')['id'] for u_l in np.unique(label)]
        #     get_reg=1


        degrees = G.vertex_degrees()
        # vf = np.logical_and(degrees > 1, degrees <= 4)
        # G = G.sub_graph(vertex_filter=vf)
        label = G.vertex_annotation();
        angle,graph = GeneralizedRadPlanorientation(G, g, 4.5, controls, mode=mode, average=average)

        coordinates=G.vertex_property('coordinates_atlas')
        edge_dist = G.edge_property('distance_to_surface')
        vertex_dist=G.vertex_property('distance_to_surface')
        edge_dist=G.edge_property('distance_to_surface')
        radiality=angle < limit_angle#40
        G.add_edge_property('radiality', radiality)
        rad_vessels_coords=G.edge_property('distance_to_surface')[np.asarray(radiality==1).nonzero()[0]]

        for reg in reg_ids:
            order, level= ano.find(reg, key='id')['order'],ano.find(reg, key='id')['level']
            print(level, order, ano.find(order, key='order')['name'])
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;
            edge_filter=from_v_prop2_eprop(G, vertex_filter)

            timepoints_temp.append(TP[j])
            reg_temp.append(reg)

            ori_rad=edge_dist[np.asarray(np.logical_and(radiality,edge_filter)==1).nonzero()[0]]
            bp_dist=vertex_dist[np.asarray(vertex_filter==1).nonzero()[0]]
            e_dist=edge_dist[np.asarray(edge_filter==1).nonzero()[0]]

            histbp, bins = np.histogram(bp_dist, bins=10)#, normed=normed)
            histrad, bins = np.histogram(ori_rad, bins=10)#, normed=normed)
            histe, bins=np.histogram(e_dist, bins=bins)#, normed=normed)
            histrad=histrad/histe
            histrad=np.nan_to_num(histrad)



            if normed:
                histrad=preprocessing.normalize(histrad.reshape(-1, 1), norm='l2', axis=0)
                histbp=preprocessing.normalize(histbp.reshape(-1, 1), norm='l2', axis=0)
                Table.append(np.concatenate((np.concatenate((histrad[:,0],histbp[:,0]), axis=0), np.array([TP[j],reg])), axis=0))
            else:
                Table.append(np.concatenate((np.concatenate((histrad,histbp), axis=0), np.array([TP[j],reg])), axis=0))




df = pd.DataFrame(Table,columns=feat_cols)
# df['timepoints']=timepoints_temp
# df['region']=reg_temp
df.to_csv('/data_2to/dev/TSNE_datas_new_annot_prop_ori_normed.csv', index=False)
# df.to_csv('/data_2to/dev/TSNE_datas_new_annot_prop_ori_unnormed.csv', index=False)
# df.to_csv('/data_2to/dev/TSNE_isocortex_datas_new_annot_normed.csv', index=False)


df=pd.read_csv('/data_2to/dev/TSNE_datas_new_annot_unnormed.csv')



## evolution od BP and orientation
df=pd.read_csv('/data_2to/dev/TSNE_datas_new_annot_prop_ori_unnormed.csv')


TP_temp=[1, 3, 5, 6, 7, 14, 30]
pal = sns.cubehelix_palette(len(TP_temp), rot=-.25, light=.7)


regions=[ano.find(r, key='order')['id'] for r in  [13, 38, 52, 101, 122,137,186]]

data2plt=df.iloc[:, 10:]#df.iloc[:, [0,1,2,3,4,5,6,7,8,9,-2,-1]]#[:, 10:]#[:, [0,1,2,3,4,5,6,7,8,9,-2,-1]]
# data2plt=data2plt[data2plt['region'].isin(regions)]
normed=False
from sklearn.preprocessing import normalize


for region in regions:
    plt.figure()
    data=data2plt[data2plt['region']==region]

    for i, tp in enumerate(TP_temp):
        d=data[data['timepoints']==tp].iloc[:,:-2]
        Cpd_c = pd.DataFrame(d).melt()
        print(tp, Cpd_c.shape)
        sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color=pal[i], linewidth=2.5)


    plt.title(ano.find(region, key='id')['name'])
    plt.legend(TP_temp)


#plot diff in BP

TP_temp=[1, 3, 5, 6, 7, 14, 30]
D1=data2plt.copy()
DD=D1[D1['region']!= 'Isocortex']
D1_diff=[]#pd.DataFrame()
for k,region in enumerate(regions):
    if reg_name[k]!='Isocortex':
        # D1_diff.append(np.concatenate((np.zeros(10),np.array([0,region])), axis=0))
        for i, tp in enumerate(TP_temp[1:]):
            D=DD[DD['region']== ano.find(reg_name[k], key='name')['id']]
            print(tp, TP_temp[i],reg_name[k])
            diff=np.mean(D[D['timepoints']==tp])-np.mean(D[D['timepoints']==TP_temp[i]])
            val=diff.values[:-2]#np.concatenate((np.array([0]),diff.values[:-2]), axis=0)
            # print(diff)
            # D1_diff=D1_diff.append({'timepoints': TP[i+1], 'diff':diff,'region': reg_name[k]},ignore_index=True)
            D1_diff.append(np.concatenate((val,np.array([TP_temp[i+1],region])), axis=0))

feat_cols = [ 'feat'+str(10+i) for i in range(10) ]
feat_cols.append('timepoints')
feat_cols.append('region')
D1_diff = pd.DataFrame(D1_diff,columns=feat_cols)

TP_temp=[1, 3, 5, 6, 7, 14, 30]
pal = sns.cubehelix_palette(len(TP_temp), rot=-.25, light=.7)
for region in regions:
    plt.figure()
    data=D1_diff[D1_diff['region']==region]

    for i, tp in enumerate(TP_temp):
        d=data[data['timepoints']==tp].iloc[:,:-2]
        Cpd_c = pd.DataFrame(d).melt()
        print(tp, Cpd_c.shape)
        sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color=pal[i], linewidth=2.5)


    plt.title(ano.find(region, key='id')['name'])
    plt.legend(TP_temp)








# data=df[df['timepoints'] != 30]
data=df.copy()
data['reggion_name'] =  [ano.find(data['region'].values[i], key='id')['name'] for i in range(data.shape[0])]

data_test=data[['alar' not in rn and 'roof' not in rn and 'basal' not in rn  and 'floor' not in rn for rn in data['reggion_name']]]




data_val=data_test[feat_cols]
data_val=data_val.values
data_val=np.nan_to_num(data_val)

from sklearn.manifold import TSNE
import time

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, early_exaggeration=12, learning_rate=200, perplexity=80, n_iter=300)
tsne_results = tsne.fit_transform(data_val)
# np.save('/data_2to/dev/TSNE_good_res_12_200_30.npy', tsne_results)
# np.save('/data_2to/dev/TSNE_good_res_12_1000_80.npy', tsne_results)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

data_test['tsne-2d-one'] = tsne_results[:,0]
data_test['tsne-2d-two'] = tsne_results[:,1]





data_plot=data_test[data_test["reggion_name"]!='universe']


fig=plt.figure(figsize=(16,10))

palette=sns.color_palette("hls", 7)
TP=[1, 3, 5, 6, 7, 30, 14]
# TP=[1, 5, 30, 3, 7, 6, 14]#, 14]
# for i, tp in enumerate(TP):
sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="timepoints",
    data=data_plot,#[data_plot["timepoints"]==tp],
    levels=[0.7, 1],
    shade=True,
    bw_adjust=.2,
    palette=palette,#sns.color_palette("hls", 7),#palette[i],
    alpha=0.1
)
sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="timepoints",
    data=data_plot,#data_plot[data_plot["timepoints"]==tp],
    levels=[0.7, 0.85, 1],
    shade=False,
    bw_adjust=.2,
    palette=palette,#sns.color_palette("hls", 7),#palette[i],
    alpha=0.1
)

g_sns=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="timepoints",
    palette=palette,#sns.color_palette("hls", 7),
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



#
# fig=plt.figure(figsize=(16,10))
# # legend="full",
# # sns.legend_out=True
#
# # fig=plt.figure()
# palette=sns.color_palette("hls", 53)
# palette=[]
# k=0
# for i, reg in enumerate(np.unique(data_plot["region"])):
#     name=ano.find(data_plot['region'].values[i], key='id')['name']
#     if name!='Primary somatosensory area, mouth, layer 6b':
#         print(k,name)
#         sns.kdeplot(
#             x="tsne-2d-one", y="tsne-2d-two",
#             # hue="reggion_name",
#             data=data_plot[data_plot["reggion_name"]==name],
#             levels=[0.5, 1],
#             shade=True,
#             bw_adjust=.4,
#             palette=sns.color_palette("hls", 8),
#             # color=ano.find(data_plot['region'].values[i], key='id')['rgb'],
#             alpha=0.1
#         )
#         sns.kdeplot(
#             x="tsne-2d-one", y="tsne-2d-two",
#             # hue="reggion_name",
#             data=data_plot[data_plot["reggion_name"]==name],
#             levels=[0.6, 0.8, 1],
#             shade=False,
#             bw_adjust=.4,
#             palette=sns.color_palette("hls", 8),
#             # color=ano.find(data_plot['region'].values[i], key='id')['rgb'],
#             alpha=0.1
#         )
#         palette.append(ano.find(data_plot['region'].values[i], key='id')['rgb'])
#         k=k+1

palette=[]
for i, reg in enumerate(np.unique(data_plot["region"])):
    reg=ano.find(reg, key='id')['name']
    if reg!='Primary somatosensory area, mouth, layer 6b':
        palette.append(ano.find(reg, key='name')['rgb'])
        print(reg)

# legend="full",
# sns.legend_out=True
# fig=plt.figure()
# palette=sns.color_palette("hls", 52)
fig=plt.figure(figsize=(16,10))
sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="reggion_name",
    data=data_plot[data_plot["reggion_name"]!='Primary somatosensory area, mouth, layer 6b'],
    # levels=[0.3, 1],
    shade=True,
    # bw_adjust=100,
    palette=palette,#sns.color_palette("hls", 8),
    # color=ano.find(data_plot['region'].values[i], key='id')['rgb'],
    alpha=0.1
)
sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="reggion_name",
    data=data_plot[data_plot["reggion_name"]!='Primary somatosensory area, mouth, layer 6b'],
    # levels=[0.1],
    shade=False,
    # bw_adjust=.9,
    palette=palette,#sns.color_palette("hls", 8),
    # color=ano.find(data_plot['region'].values[i], key='id')['rgb'],
    alpha=0.1
)

sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="reggion_name",
    data=data_plot[data_plot["reggion_name"]!='Primary somatosensory area, mouth, layer 6b'],
    levels=[0.1],
    shade=False,
    # bw_adjust=.9,
    palette=palette,#sns.color_palette("hls", 8),
    # color=ano.find(data_plot['region'].values[i], key='id')['rgb'],
    alpha=0.1
)
g_sns1=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="reggion_name",
    palette=palette,#sns.color_palette("hls", 8),#palette,#sns.color_palette("hls", 53),
    data=data_plot[data_plot["reggion_name"]!='Primary somatosensory area, mouth, layer 6b'],
    legend="full",
    alpha=0.5,
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



# filter isocortx regions
regions=[ano.find(r, key='order')['id'] for r in  [13, 38, 52, 101, 122,137,186]]
data_test=data_test[data_test['region'].isin(regions)]




data_val=data_test[feat_cols]
data_val=data_val.values
data_val=np.nan_to_num(data_val)

from sklearn.manifold import TSNE
import time

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, early_exaggeration=30, learning_rate=200, perplexity=60, n_iter=300)
tsne_results = tsne.fit_transform(data_val)
# np.save('/data_2to/dev/TSNE_good_res_12_200_30.npy', tsne_results)
# np.save('/data_2to/dev/TSNE_good_res_12_1000_80.npy', tsne_results)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

data_test['tsne-2d-one'] = tsne_results[:,0]
data_test['tsne-2d-two'] = tsne_results[:,1]





data_plot=data_test[data_test["reggion_name"]!='universe']


fig=plt.figure(figsize=(16,10))

palette=sns.color_palette("hls", 7)
TP=[1, 3, 5, 6, 7, 30, 14]
# TP=[1, 5, 30, 3, 7, 6, 14]#, 14]
# for i, tp in enumerate(TP):
sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="timepoints",
    data=data_plot,#[data_plot["timepoints"]==tp],
    levels=[0.3, 1],
    shade=True,
    bw_adjust=.5,
    palette=palette,#sns.color_palette("hls", 7),#palette[i],
    alpha=0.1
)
sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="timepoints",
    data=data_plot,#data_plot[data_plot["timepoints"]==tp],
    levels=[0.3, 0.6, 1],
    shade=False,
    bw_adjust=.5,
    palette=palette,#sns.color_palette("hls", 7),#palette[i],
    alpha=0.1
)

g_sns=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="timepoints",
    palette=palette,#sns.color_palette("hls", 7),
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


palette=[]
for i, reg in enumerate(np.unique(data_plot["region"])):
    reg=ano.find(reg, key='id')['name']
    if reg!='Primary somatosensory area, mouth, layer 6b':
        palette.append(ano.find(reg, key='name')['rgb'])
        print(reg)

# legend="full",
# sns.legend_out=True
# fig=plt.figure()
palette=sns.color_palette("hls", 7)
fig=plt.figure(figsize=(16,10))
sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="reggion_name",
    data=data_plot[data_plot["reggion_name"]!='Primary somatosensory area, mouth, layer 6b'],
    levels=[0.3, 1],
    shade=True,
    bw_adjust=0.5,
    palette=palette,#sns.color_palette("hls", 8),
    # color=ano.find(data_plot['region'].values[i], key='id')['rgb'],
    alpha=0.1
)
# sns.kdeplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="reggion_name",
#     data=data_plot[data_plot["reggion_name"]!='Primary somatosensory area, mouth, layer 6b'],
#     shade=False,
#     bw_adjust=0.5,
#     palette=palette,#sns.color_palette("hls", 8),
#     # color=ano.find(data_plot['region'].values[i], key='id')['rgb'],
#     alpha=0.1
# )

sns.kdeplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="reggion_name",
    data=data_plot[data_plot["reggion_name"]!='Primary somatosensory area, mouth, layer 6b'],
    levels=[0.3, 0.5, 1],
    shade=False,
    bw_adjust=.5,
    palette=palette,#sns.color_palette("hls", 8),
    # color=ano.find(data_plot['region'].values[i], key='id')['rgb'],
    alpha=0.1
)
g_sns1=sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="reggion_name",
    palette=palette,#sns.color_palette("hls", 8),#palette,#sns.color_palette("hls", 53),
    data=data_plot[data_plot["reggion_name"]!='Primary somatosensory area, mouth, layer 6b'],
    legend="full",
    alpha=0.5,
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