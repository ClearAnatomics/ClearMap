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

work_dir_3M='/data_2to/otof3M/new_vasc'
# controls=['1w', '2w', '4w', '5w','6w']
# mutants=[ '1k','3k','4k', '5k', '6k']
controls_3M=['2w', '4w', '5w','6w']
mutants_3M=[ '3k', '5k', '6k']

work_dir_10M='/data_SSD_2to/211019_otof_10m'
mutants_10M=['1k', '2k','3k', '6k']#456 not annotated ?
controls_10M=['7w', '9w', '10w', '12w', '13w']


wds=[work_dir_1M,work_dir_2M,work_dir_3M, work_dir_6M,work_dir_10M]
ctrls=[controls_1M,controls_2M,controls_3M,controls_6M,controls_10M]
mtts=[mutants_1M, mutants_2M,mutants_3M,mutants_6M,mutants_10M]

# reg_name=['cortex', 'striatum', 'hippocampus']


reg_name=['brain']
TP=['1 months', '2 months',  '3 months', '6 months', '10 months']

import ClearMap.IO.IO as io
anot=io.read(os.path.join(atlas_path, 'annotation_25_full.nrrd'))
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

# ctrls=[controls_10M]
# mtts=[mutants_10M]
# wds=[work_dir_10M]
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


df = pd.DataFrame(Table,columns=['bp', 'timepoint', 'region', 'condition'])
df.to_csv('/data_2to/devotof/TSNE_otof_datas_bp.csv', index=False)

tp = '3 months'
df_2m=df[df['timepoint']==tp]

unique=np.unique(df['region'].values)



data2plot=pd.DataFrame()

for u in unique:
    print(ano.find(u, key='order')['name'])
    df_2m_reg=df_2m[df_2m['region']==u]
    cont=df_2m_reg[df_2m_reg['condition']==0]['bp'].values
    mut=df_2m_reg[df_2m_reg['condition']==1]['bp'].values

    x=np.log2(np.mean(mut))-np.log2(np.mean(cont))
    tvals, pval = stats.ttest_ind(cont, mut, equal_var = False);
    y=-np.log(pval)

    alpha=0.3
    color=np.array([206/255, 206/255, 206/255])
    if y>3:
        alpha=1
        color=ano.find(u, key='order')['rgb']
    if np.abs(x)>0.6:
            alpha=1
            color=ano.find(u, key='order')['rgb']
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
plt.title(tp)


fig.set_picker(True)
def onpick(event):
    print("Picked!")
    print(event)
    origin = data2plot.iloc[event.ind[0]]['name']
    print('Selected item came from {}'.format(origin))
    # plt.gca().set_title('Selected item came from {}'.format(origin))

g_sns.figure.canvas.mpl_connect("pick_event", onpick)



##### bp density evolution


df_2m=df.copy()

unique=np.unique(df['region'].values)
sub_reg=[]
for u in unique:
    if 'SSp-bfd' in ano.find(u, key='order')['acronym']:
        print(ano.find(u, key='order')['acronym'])
        sub_reg.append(u)



data2plot=pd.DataFrame()

for u in [127, 54]:
    print(ano.find(u, key='order')['name'])
    df_2m_reg=df_2m[df_2m['region']==u]
    cont=df_2m_reg[df_2m_reg['condition']==0]
    mut=df_2m_reg[df_2m_reg['condition']==1]

    plt.figure()
    sns.lineplot(x= 'timepoint', y='bp', data=cont,err_style='bars', color='cadetblue')
    sns.lineplot(x= 'timepoint', y='bp', data=mut,err_style='bars', color='indianred')
    plt.title(ano.find(u, key='order')['acronym'][:-1])

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
plt.title(tp)
