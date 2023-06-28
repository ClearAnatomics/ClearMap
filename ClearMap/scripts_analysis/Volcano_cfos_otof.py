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


work_dir_vasc='/data_2to/otof3M/new_vasc'
work_dir_3M='/data_2to/otof3M/cfos/cells'
# controls=['1w', '2w', '4w', '5w','6w']
# mutants=[ '1k','3k','4k', '5k', '6k']
controls_3M=['2w', '4w', '5w','6w']
mutants_3M=[ '3k', '5k', '6k']

work_dir_1M='/data_SSD_2to/cfos_otof_1M'
controls_1M=['1wt', '6wt','10wt', '13wt']
mutants_1M=[ '2ko', '4ko', '5ko', '7ko']

work_dir_1M_vasc='/data_2to/otof1M'
controls_1M_vasc=[ '1w', '3w', '5w', '6w', '7w']
mutants_1M_vasc=['1k', '2k', '3k', '4k']

wds=[work_dir_1M_vasc]
ctrls=[controls_1M_vasc]
mtts=[mutants_1M_vasc]

# reg_name=['cortex', 'striatum', 'hippocampus']


reg_name=['brain']
TP=['1 months','1 months']

import ClearMap.IO.IO as io
anot=io.read(os.path.join(atlas_path, 'annotation_25_full.nrrd'))
anot_f= '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd'

reg_ids=np.unique(anot)
regions = []
for r in reg_ids:
    try:
        regions.append([(ano.find(r, key='id')['order'],ano.find(r, key='id')['level'])])
    except:
        print(r)
#
# region_list = [(6, 6)]  # isocortex
# regions = []
# R = ano.find(region_list[0][0], key='order')['name']
# main_reg = region_list
# sub_region = True
# for r in reg_list.keys():
#     l = ano.find(r, key='order')['level']
#     regions.append([(r, l)])
#
regions=[]
anot_f= '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd'
reg_ids=np.unique(io.read(anot_f))[2:]
for r in reg_ids:
    print(ano.find(r, key='id')['order'], ano.find(r, key='id')['name'])
    regions.append((ano.find(r, key='id')['order'],ano.find(r, key='id')['level']))

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

# ctrls=[controls_3M]
# mtts=[mutants_3M]
# wds=[work_dir_3M]

atlas=io.read('/data_SSD_2to/cfos_otof_1M/annotation_25_HeadLightOrientation_sagital.tif').swapaxes(0,2)
io.write('/data_SSD_2to/cfos_otof_1M/annotation_25_HeadLightOrientation_sagital_good.tif',atlas)
ano.initialize(annotation_file ='/data_SSD_2to/cfos_otof_1M/annotation_25_HeadLightOrientation_sagital_good.tif')
ano.set_annotation_file(ano.initialize(annotation_file ='/data_SSD_2to/cfos_otof_1M/annotation_25_HeadLightOrientation_sagital.tif'))

for k, state in enumerate([ctrls, mtts]):
    for j, controls in enumerate(state):
        print(j,wds[j],controls)
        for i, g in enumerate(controls):
            there_is_graph=True
            there_is_cells=True

            print(g)
            work_dir=wds[j]
            try:
                G = np.load(work_dir + '/' +g+ '/' + g + 'cells.npy')
                cells_order=[G[i][8] for i in range(G.shape[0])]
                cells_coords=np.asarray([[G[i][5], G[i][6], G[i][7]] for i in range(G.shape[0])])
                cells=ano.label_points(cells_coords, key='order');
                # for i in range(G.shape[0]):
                #     print(i,int(G[i][5]), int(G[i][6]), int(G[i][7]))
                #     try:
                #         cells.append(ano.find(atlas[int(G[i][5]), int(G[i][6]), int(G[i][7])])['order'])
                #     except:
                #         cells.append(0)
                # v = vox.voxelize(cells[:, :3], shape=annotation.shape, weights=None, radius=(radius, radius, radius), method='sphere');
                # tifffile.imsave('/data_SSD_2to/cfos_otof_1M/aud_cells_vox.tif', v.astype('float32'))#np.swapaxes(pvalscol, 2, 0)
                # regions=np.unique(cells)
            except:
                try:
                    print('pb')
                    G = np.load(work_dir + '/' +g+ '/'+ 'cells.npy')
                    cells=[G[i][8] for i in range(G.shape[0])]
                except:
                    print('file missing')
                    there_is_cells=False
            try:
                G = ggt.load(work_dir_vasc + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                try:
                    G = ggt.load(work_dir_vasc + '/' + g + '/' + str(g)+'_graph_correcteduniverse_subregion.gt')
                except:
                    try:
                        G = ggt.load(work_dir_vasc + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')
                    except:
                        print('np graph available')
                        there_is_graph=False

            for region_list in regions:
                order, level=region_list
                if there_is_graph:
                    vertex_filter = np.zeros(G.n_vertices)
                    level = ano.find(order, key='order')['level']

                    label = G.vertex_annotation();
                    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                    vertex_filter[label_leveled == order] = 1;
                    res_bp=np.sum(vertex_filter)
                else:
                    res_bp=np.nan

                if there_is_cells:
                    cells_leveled = ano.convert_label(cells, key='order', value='order', level=level)
                    cells_filtered = np.asarray(cells_leveled==order).nonzero()[0]
                    res_cells = cells_filtered.shape[0]
                else:
                    res_cells=np.nan

                anot_leveled=ano.convert_label(anot, key='id', value='order', level=level)
                volume=np.asarray(anot_leveled.flatten()==order).nonzero()[0].shape[0]


                print(g, level, order, volume, k, ano.find(order, key='order')['name'])
                Table.append([res_bp, res_cells,TP[j],order, volume, k])


df = pd.DataFrame(Table,columns=['bp', 'cfos', 'timepoint', 'region', 'volume', 'condition'])
df.to_csv('/data_SSD_2to/cfos_otof_1M/vasc_otof_datas_metaregions_1M.csv', index=False)
# df.to_csv('/data_SSD_2to/cfos_otof_1M/cfos_otof_datas_metaregions_1M.csv', index=False)
# df.to_csv('/data_2to/devotof/cfos_otof_datas_metaregions.csv', index=False)
# df.to_csv('/data_2to/devotof/cfos_otof_datas.csv', index=False)



df=pd.read_csv('/data_2to/devotof/cfos_otof_datas_metaregions.csv')
tp = '3 months'
df_2m=df[df['cfos']!=np.nan]
df_2m=df[df['timepoint']==tp]
unique=np.unique(df['region'].values)



data2plot=pd.DataFrame()
for u in unique:
    print(ano.find(u, key='order')['name'])
    df_2m_reg=df_2m[df_2m['region']==u]
    cont=df_2m_reg[df_2m_reg['condition']==0]['cfos'].values
    cont = cont[~np.isnan(cont)]
    mut=df_2m_reg[df_2m_reg['condition']==1]['cfos'].values
    mut = mut[~np.isnan(mut)]
    x=np.log2(np.mean(mut))-np.log2(np.mean(cont))
    tvals, pval = stats.ttest_ind(cont, mut, equal_var = False);
    y=-np.log(pval)
    print(cont, mut)
    alpha=0.3
    color=np.array([206/255, 206/255, 206/255])
    if y>3:
        alpha=1
        color=ano.find(u, key='order')['rgb']
    if np.abs(x)>0.5:
        alpha=1
        color=ano.find(u, key='order')['rgb']
    print(x, y, u, ano.find(u, key='order')['acronym'])
    data2plot=data2plot.append({'timepoint': tp, 'x':x,'y': y, 'alpha':alpha,'color':color,'name':ano.find(u, key='order')['acronym']},ignore_index=True)

palette=sns.color_palette("husl", len(unique), desat=0.8)

# palette=[ano.find(u, key='name')['rgb'] for u in data2plot['name'].values]
# palette=data2plot['color'].values


fig=plt.figure()
# g_sns=sns.scatterplot(
#     x='x', y='y',
#     hue='name',
#     palette=palette,
#     # alpha=data2plot['alpha'].values.astype('float'),
#     # color=ano.find(u, key='order')['rgb'],
#     data=data2plot,
#     picker = True
# )
X=data2plot['x'].values
Y=data2plot['y'].values
C=np.array(data2plot['color'].values)
A=data2plot['alpha'].values
N=data2plot['name'].values
for i in range(X.shape[0]):
    plt.scatter(X[i], Y[i], c=C[i], alpha=A[i])
    if A[i]==1:
        plt.text(X[i], Y[i], s=N[i])



fig.set_picker(True)
def onpick(event):
    print("Picked!")
    print(event)
    origin = data2plot.iloc[event.ind[0]]['name']
    print('Selected item came from {}'.format(origin))
    # plt.gca().set_title('Selected item came from {}'.format(origin))

g_sns.figure.canvas.mpl_connect("pick_event", onpick)



###### density cfos VS density vasc
tp = '3 months'

CFos_data=pd.read_csv('/data_2to/devotof/cfos_otof_datas_metaregions.csv')
CFos_data['bp_density']=CFos_data['bp']/CFos_data['volume']
CFos_data['cfos_density']=CFos_data['cfos']/CFos_data['volume']

# CFos_data=CFos_data[CFos_data['bp_density']<10]
# CFos_data=CFos_data[CFos_data['cfos_density']<10]


Isocortex=[6,13,40,54,81,103,127,142,191,388, 464, 564]
Midbrain=[811, 813,817, 826, 841, 856, 859]

CFos_data_Isocortex=CFos_data.loc[CFos_data['region'].isin(Isocortex)]
CFos_data_Midbrain=CFos_data.loc[CFos_data['region'].isin(Midbrain)]


nb_reg=np.unique(CFos_data[CFos_data["condition"]==1]['region']).shape[0]


plt.figure()

un=np.unique(CFos_data[CFos_data["condition"]==1]['region'])
nb_reg=un.shape[0]

palette=[ano.find(u, key='order')['rgb'] for u in un]
# sns.lmplot(x="bp_density", y="cfos_density", data=CFos_data[CFos_data["condition"]==1], color='k');

# sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Isocortex[CFos_data_Isocortex["condition"]==1], color='k');
# palette=sns.color_palette("husl", nb_reg, desat=0.5)
g_sns2=sns.scatterplot(
    x="bp_density", y="cfos_density",
    hue="region",
    palette=palette,
    data=CFos_data[CFos_data["condition"]==0],
    alpha=0.9,
    picker = True
)

# plt.figure()
un=np.unique(CFos_data[CFos_data["condition"]==0]['region'])
nb_reg=u.shape[0]
# sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Isocortex[CFos_data_Isocortex["condition"]==0]);

palette=[ano.find(u, key='order')['rgb'] for u in un]
g_sns2=sns.scatterplot(
    x="bp_density", y="cfos_density",
    hue="region",
    palette=palette,
    data=CFos_data[CFos_data["condition"]==0],
    alpha=0.9,
    picker = True
)




plt.figure()
sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Isocortex[CFos_data_Isocortex["condition"]==0], color='cadetblue');
sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Isocortex[CFos_data_Isocortex["condition"]==1], color='indianred');

reg2plot=[40, 142, 54, 191]
Data2plot=CFos_data.loc[CFos_data['region'].isin(reg2plot)]#CFos_data_Isocortex[CFos_data_Isocortex["condition"]==0]
for r in Data2plot.iterrows():
    if r[1].values[5]==0:
        plt.text(r[1].values[-2], r[1].values[-1], s=ano.find(r[1].values[3], key='order')['acronym'], c='cadetblue')
    elif r[1].values[5]==1:
        plt.text(r[1].values[-2], r[1].values[-1], s=ano.find(r[1].values[3], key='order')['acronym'], c='indianred')
plt.title('Isocortex')



plt.figure()
sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Midbrain[CFos_data_Midbrain["condition"]==0], color='cadetblue');
sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Midbrain[CFos_data_Midbrain["condition"]==1], color='indianred');

reg2plot=[817, 969, 859, 895]
Data2plot=CFos_data.loc[CFos_data['region'].isin(reg2plot)]#CFos_data_Isocortex[CFos_data_Isocortex["condition"]==0]
for r in Data2plot.iterrows():
    if r[1].values[5]==0:
        plt.text(r[1].values[-2], r[1].values[-1], s=ano.find(r[1].values[3], key='order')['acronym'], c='cadetblue')
    elif r[1].values[5]==1:
        plt.text(r[1].values[-2], r[1].values[-1], s=ano.find(r[1].values[3], key='order')['acronym'], c='indianred')
plt.title('Midbrain')



######## vasc density VS cFos density

vasc_Data=pd.read_csv('/data_SSD_2to/cfos_otof_1M/vasc_otof_datas_metaregions_1M.csv')
CFos_data=pd.read_csv('/data_SSD_2to/cfos_otof_1M/cfos_otof_datas_metaregions_1M.csv')


tp = '1 months'

vasc_Data['bp_density']=vasc_Data['bp']/vasc_Data['volume']
CFos_data['cfos_density']=CFos_data['cfos']/CFos_data['volume']

# CFos_data=CFos_data[CFos_data['bp_density']<10]
# CFos_data=CFos_data[CFos_data['cfos_density']<10]


Isocortex=[6,13,40,54,81,103,127,142,191,388, 464, 564]
Midbrain=[811, 813,817, 826, 841, 856, 859]

CFos_data_Isocortex=CFos_data.loc[CFos_data['region'].isin(Isocortex)]
CFos_data_Midbrain=CFos_data.loc[CFos_data['region'].isin(Midbrain)]

nb_reg=np.unique(CFos_data[CFos_data["condition"]==1]['region']).shape[0]


CFos_data_grouped=CFos_data.groupby(['region', 'condition'])
vasc_Data_grouped=vasc_Data.groupby(['region', 'condition'])


groups=np.array(list(CFos_data_grouped.groups.keys()))
data_grouped_mean=CFos_data_grouped.mean()
data_grouped_mean['bp_density']=vasc_Data_grouped.mean()['bp_density']
data_grouped_mean['region']=groups[:,0]
data_grouped_mean['condition']=groups[:,1]





CFos_data=data_grouped_mean.copy()


un=np.unique(CFos_data[CFos_data["condition"]==1]['region'])
nb_reg=un.shape[0]

plt.figure()
palette=[ano.find(u, key='order')['rgb'] for u in un]
# sns.lmplot(x="bp_density", y="cfos_density", data=CFos_data[CFos_data["condition"]==1], color='k');

# sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Isocortex[CFos_data_Isocortex["condition"]==1], color='k');
# palette=sns.color_palette("husl", nb_reg, desat=0.5)
g_sns2=sns.scatterplot(
    x="bp_density", y="cfos_density",
    hue="region",
    palette=palette,
    data=CFos_data[CFos_data["condition"]==1],
    alpha=0.9,
    picker = True
)

# plt.figure()
un=np.unique(CFos_data[CFos_data["condition"]==0]['region'])
nb_reg=un.shape[0]
# sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Isocortex[CFos_data_Isocortex["condition"]==0]);

palette=[ano.find(u, key='order')['rgb'] for u in un]
g_sns2=sns.scatterplot(
    x="bp_density", y="cfos_density",
    hue="region",
    palette=palette,
    data=CFos_data[CFos_data["condition"]==0],
    alpha=0.9,
    picker = True
)



CFos_data_Isocortex=CFos_data.loc[CFos_data['region'].isin(Isocortex)]
CFos_data_Midbrain=CFos_data.loc[CFos_data['region'].isin(Midbrain)]

plt.figure()
sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Isocortex[CFos_data_Isocortex["condition"]==0], color='cadetblue');
sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Isocortex[CFos_data_Isocortex["condition"]==1], color='indianred');

reg2plot=[40, 142, 54, 191]
Data2plot=CFos_data.loc[CFos_data['region'].isin(reg2plot)]#CFos_data_Isocortex[CFos_data_Isocortex["condition"]==0]
for r in Data2plot.iterrows():
    if r[1].values[6]==0:
        plt.text(r[1].values[-3], r[1].values[-4], s=ano.find(r[1].values[5], key='order')['acronym'], c='cadetblue')
    elif r[1].values[6]==1:
        plt.text(r[1].values[-3], r[1].values[-4], s=ano.find(r[1].values[5], key='order')['acronym'], c='indianred')
plt.title('Isocortex')



plt.figure()
sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Midbrain[CFos_data_Midbrain["condition"]==0], color='cadetblue');
sns.regplot(x="bp_density", y="cfos_density", data=CFos_data_Midbrain[CFos_data_Midbrain["condition"]==1], color='indianred');

reg2plot=[817, 969, 859, 895]
Data2plot=CFos_data.loc[CFos_data['region'].isin(reg2plot)]#CFos_data_Isocortex[CFos_data_Isocortex["condition"]==0]
for r in Data2plot.iterrows():
    if r[1].values[6]==0:
        plt.text(r[1].values[-3], r[1].values[-4], s=ano.find(r[1].values[5], key='order')['acronym'], c='cadetblue')
    elif r[1].values[6]==1:
        plt.text(r[1].values[-3], r[1].values[-4], s=ano.find(r[1].values[5], key='order')['acronym'], c='indianred')
plt.title('Midbrain')

