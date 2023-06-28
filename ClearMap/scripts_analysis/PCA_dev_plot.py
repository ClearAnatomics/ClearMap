import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.IO.IO as io
import seaborn as sns
atlas_path = os.path.join(settings.resources_path, 'Atlas');
from sklearn import preprocessing

anot=io.read(os.path.join(atlas_path, 'annotation_25_full.nrrd'))
# anot_leveled=ano.convert_label(anot, key='id', value='order', level=level)

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

# work_dirAdult='/data_2to/earlyDep_ipsi'
# controlsAdult=['4', '7', '10', '15']
work_dirAdult='/data_SSD_2to/fluoxetine2'
controlsAdult=['1', '2', '3', '4','5']


work_dirP6='/data_2to/p6'
controlsP6=['6a']#['2', '3']

work_dirP14='/data_2to/p14'
controlsP14=['14a','14b', '14c']#['2', '3']

work_dir_7M='/data_SSD_2to/degradationControls/7M'
controls_7M=['467', '468', '469']

# work_dir_2M='/data_SSD_2to/degradationControls/2M'
# controls_2M=['3R', '4R', '6R']

work_dir_2M='/data_SSD_2to/191122Otof'
controls_2M=['2R', '3R', '5R']

work_dir_P21 = '/data_SSD_2to/P21'
controls_P21 =['1', '2', '3']

work_dir_P12='/data_SSD_2to/p12'
controls_P12 =['1', '2', '3']


work_dir_3M='/data_SSD_2to/whiskers_graphs/new_graphs'
controls_3M=['142L','158L','162L', '164L']



# reg_name=['cortex', 'striatum', 'hippocampus']


reg_name=['brain']
TP=[1, 5, 30, 3, 7, 6, 14, 60, 210, 21, 12, 90]


workdirs=[ work_dirP1, work_dirP5, work_dirAdult,work_dirP3,work_dirP7,work_dirP6,work_dirP14,work_dir_2M,work_dir_7M, work_dir_P21,work_dir_P12,work_dir_3M]
controlslist=[ controlsP1, controlsP5, controlsAdult,controlsP3,controlsP7,controlsP6,controlsP14,controls_2M,controls_7M,controls_P21,controls_P12,controls_3M]

import ClearMap.IO.IO as io
# anot=io.read('/data_2to/pix/anoP4.tif')
# anot=anot-32768
# reg_ids=np.unique(anot)


regions=[]
anot_f= '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd'
reg_ids=np.unique(io.read(anot_f))[2:]
for r in reg_ids:
    print(ano.find(r, key='id')['order'], ano.find(r, key='id')['name'])
    regions.append((r,ano.find(r, key='id')['level']))



anot_f=['/data_2to/alignement/atlases/new_region_atlases/P1/smoothed_anot_P1.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P3/smoothed_anot_P3_V6.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P7/smoothed_anot_P7.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P21/atlasP21.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P10/ano_P10.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd']

# anot_f=[os.path.join(atlas_path, 'annotation_25_full.nrrd'),
#         '/data_2to/alignement/atlases/new_region_atlases/P1/smoothed_anot_P1.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P3/smoothed_anot_P3_V6.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P7/smoothed_anot_P7.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P14/refined/ano_P14_p10_corrected_smoothed_rescale3.nrrd']


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

    controls=controlslist[j]
    print(j,TP[j],workdirs[j],controls)
    st='dev'
    print('dev brain')

    annotation_file=anot_f[j]

    # if TP[j]!=30:
    anot=io.read(annotation_file)
    if TP[j]!=30 and TP[j]!=5 and TP[j]!=60 and TP[j]!=210 and TP[j]!=90:
        # get_reg=0
        print('no extra label')
        ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                       extra_label=None,annotation_file = annotation_file)#extra_label=None,
    else:
        ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                       annotation_file = annotation_file)
    if TP[j]==30:
        print('adult brain')
        st='adult'
        # regions=[(0,0),(52,9), (122, 7), (13,7)]#[(0,0), (6,6)]#, (572, 5), (455,6)]
        # ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
        #                extra_label = None, annotation_file = os.path.join(atlas_path, 'annotation_25_full.nrrd'))

    else:
        st='dev'
        print('dev brain')

    for i, g in enumerate(controls):
        print(g)
        work_dir=workdirs[j]

        try:
            G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse_smoothed.gt')
            print(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse_smoothed.gt')
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
        deg1=degrees==1
        # vf = np.logical_and(degrees > 1, degrees <= 4)
        # G = G.sub_graph(vertex_filter=vf)
        label = G.vertex_annotation();
        angle,graph = GeneralizedRadPlanorientation(G, g, 4.5, controls, mode=mode, average=average)

        # coordinates=G.vertex_property('coordinates_atlas')
        # edge_dist = G.edge_property('distance_to_surface')
        # vertex_dist=G.vertex_property('distance_to_surface')
        # edge_dist=G.edge_property('distance_to_surface')
        radiality=angle < limit_angle#40
        G.add_edge_property('radiality', radiality)
        rad_vessels_coords=G.edge_property('distance_to_surface')[np.asarray(radiality==1).nonzero()[0]]

        for region_list in regions:
            id, level=region_list
            order = ano.find(id, key='id')['order']
            # level = ano.find(order, key='order')['level']
            print(id, level, order, ano.find(order, key='order')['name'])
            if TP[j]==21 or TP[j]==12:
                print('labelled by ID')
                label[label<0]=0
                label_leveled = ano.convert_label(label, key='id', value='id', level=level)
                vertex_filter = label_leveled == id;

            else:
                print('labelled by order')
                label_leveled = ano.convert_label(label, key='order', value='id', level=level)
                vertex_filter = label_leveled == id;
            print(np.sum(vertex_filter))
            edge_filter=from_v_prop2_eprop(G, vertex_filter)

            # timepoints_temp.append(TP[j])
            # reg_temp.append(reg)

            # if TP[j]!=30:
            anot_leveled=ano.convert_label(anot, key='id', value='id', level=level)
            volume=np.sum(np.asarray(anot_leveled==id))

            nb_deg1=np.sum(np.logical_and(vertex_filter,deg1))
            nb_radial_vess=np.sum(np.logical_and(edge_filter,radiality))

            print(level, order, volume, ano.find(order, key='order')['name'])
            Table.append([np.sum(vertex_filter),np.sum(edge_filter), nb_radial_vess,nb_deg1,TP[j],id, volume])






df = pd.DataFrame(Table,columns=['nb_vertices', 'nb_edges', 'nb_radial_vess', 'nb_deg1','timepoint', 'region', 'volume'])

# df.to_csv('/data_2to/dev/PCA_datas_new_annot.csv', index=False)
# df=pd.read_csv('/data_2to/dev/PCA_datas_new_annot.csv')

df.to_csv('/data_2to/dev/PCA_datas_new_annot_all_TP.csv', index=False)
df=pd.read_csv('/data_2to/dev/PCA_datas_new_annot_all_TP.csv')
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# vols=[]
# regs=df['region']
# tps=df['timepoint']
# tp_1=0
# for row in df.iterrows():
#     print(row)
#     tp=row[1]['timepoint']
#     id=row[1]['region']
#     if tp!=tp_1:
#         print('loading anot')
#         j=np.where(TP==tp)[0][0]
#         annotation_file=anot_f[j]
#         anot=io.read(annotation_file)
#         ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
#                        annotation_file = annotation_file)
#     tp_1=tp
#     print(tp, TP, j)
#     level=ano.find(id)['level']
#     anot_leveled=ano.convert_label(anot, key='id', value='id', level=level)
#     volume=np.asarray(anot_leveled.flatten()==id).nonzero()[0].shape[0]
#     vols.append(volume)
#
#
#
# df['volume']=volume

df['v_density']=df['nb_vertices']/df['volume']
df['e_density']=df['nb_edges']/df['volume']
df['rad_vess_density']=df['nb_radial_vess']/df['nb_edges']
df['nb_deg1_density']=df['nb_deg1']/df['volume']

df_no1=df[df['timepoint']!=1]


# df=df[df['timepoint']!=1]
df=df[df['timepoint']!=6]

## prepare mean tables
mean_df=df.groupby(['region','timepoint']).mean()
data=mean_df.copy()
mean_df['timepoint']=[data.index[i][1] for i in range(data.index.shape[0])]
mean_df['region']=[data.index[i][0] for i in range(data.index.shape[0])]

data=mean_df.copy()




X=data[['e_density', 'rad_vess_density', 'nb_deg1_density']]#, 'volume']]
X=data[['e_density', 'nb_deg1_density']]#, 'volume']]
# X=df[['nb_vertices', 'nb_edges', 'nb_radial_vess', 'volume']]
# X=df[['v_density', 'rad_vess_density', 'nb_deg1_density']]
X=df[['e_density', 'rad_vess_density']]#, 'nb_deg1_density']]
y=data['timepoint']


colors = sns.color_palette("hls",12, desat=0.6)
colors=np.asarray(colors)
TP=[1, 5, 30, 3, 7, 6, 14, 60, 210, 21, 12, 90]
TP=[1, 3,5,6,7,12,14, 21,30, 60,90, 210]
table=[]
for i in y.values:
    print(i)
    print(colors[np.where(TP==i)[0][0]])
    table.append(colors[np.where(TP==i)[0][0]])

data['colors']=table



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=data['e_density'], ys=data['rad_vess_density'],zs=data['nb_deg1_density'], c=data['colors'])
ax.set_xlabel('e_density')
ax.set_ylabel('rad_vess_density')
ax.set_zlabel('nb_deg1_density')
plt.xlim(0,0.3)
# plt.set_ylim(1)
ax.set_zlim(0,0.03)


reg=[817, 127, 54, 13, 398]
# plt.figure()
# plt.scatter(x=data['rad_vess_density'], y=data['e_density'], c=data['colors'])
# plt.set_ylabel('e_density')
# plt.set_xlabel('rad_vess_density')
# plt.legend(TP)

# plt.set_ylim(1)



# reg=[52, 122, 101,13]
reg=[817, 127, 54, 13, 398]
# reg=[163, 13, 142, 54]
# reg=np.unique(data['region'].values)
plt.figure()
data2plot=data.copy()
# data2plot=data[data['timepoint']!=1]
# data2plot=data[data['timepoint']!=6]

# data2plot=data.copy()
# data2plot=data2plot[data2plot['timepoint']!=5]
# data2plot=data2plot[data2plot['timepoint']!=14]
# data2plot=data2plot[data2plot['timepoint']!=7]

plt.scatter(x=data2plot['rad_vess_density'],y=data2plot['e_density'], c=data2plot['colors'], alpha=0.5)
sns.kdeplot(
    x="rad_vess_density", y="e_density",
    hue="timepoint",
    data=data2plot,
    levels=[ 0.5,0.6, 0.7, 1],
    shade=False,
    # bw_adjust=1,
    palette=colors[[False, True, False, False, False, False, True]].tolist(),
    alpha=0.25
)
sns.kdeplot(
    x="rad_vess_density", y="e_density",
    hue="timepoint",
    data=data2plot[data2plot['timepoint']==30],
    levels=[ 0.85,0.9, 0.95],
    shade=False,
    # bw_adjust=1,
    palette=colors[[False, False, False, False, False, False, True]].tolist(),
    alpha=0.25
)
sns.kdeplot(
    x="rad_vess_density", y="e_density",
    hue="timepoint",
    data=data2plot[data2plot['timepoint']==30],
    levels=[ 0.95, 1],
    shade=True,
    # bw_adjust=.8,
    palette=colors[[False, False, False, False, False, False, True]].tolist(),
    alpha=0.05
)

sns.kdeplot(
    x="rad_vess_density", y="e_density",
    hue="timepoint",
    data=data2plot,
    levels=[ 0.7, 1],
    shade=True,
    # bw_adjust=.8,
    palette=colors[[False, True, False, False, False, False, True]].tolist(),
    alpha=0.05
)

reg=[817, 127, 54, 13, 398]
reg=[500, 733, 378, 329, 247, 4]
# reg=np.unique(data2plot['region'].values)
i=0
for r in reg:
    plt.figure()
    i=i+1
    # r=ano.find(r, key='order')['id']
    print(r,ano.find(r, key='order')['acronym'])
    tmp=np.unique(data2plot['timepoint'].values)
    print(tmp)
    for t in tmp:
        # if t==30:
        #     ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
        #                    annotation_file = anot_f[0])
        # else:
        #     ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
        #                    annotation_file = anot_f[2])
        id=ano.find(r, key='order')['id']
        print(t)
        d=data2plot[data2plot['timepoint']==t]
        print(len(d))
        d=d[d['region']==id]
        print(len(d))
        s=ano.find(r, key='order')['acronym']
        print(s,d['nb_deg1_density'], d['e_density'])
        plt.scatter(x=d['nb_deg1_density'].values[0], y=d['e_density'].values[0],c=d['colors'].values[0])
        plt.text(x=d['nb_deg1_density'].values[0], y=d['e_density'].values[0], s=s, c=d['colors'].values[0], size='x-large')


plt.xlabel('rad_vess_density')
plt.ylabel('e_density')
plt.xlim(0.2,0.8)
plt.ylim(-0.01,0.2)







plt.figure()
# data=data[data['timepoint']!=1]
# data=data[data['timepoint']!=6]
plt.scatter(x=data['rad_vess_density'],y=data['nb_deg1_density'], c=data['colors'], alpha=0.5)
# sns.kdeplot(
#     x="e_density", y="nb_deg1_density",
#     hue="timepoint",
#     data=data,
#     # levels=[ 0.5,0.6, 0.7, 1],
#     shade=False,
#     # bw_adjust=1,
#     palette=colors[[False, True, True, False, True, True, True]].tolist(),
#     alpha=0.25
# )
plt.xlim(0.2,0.8)
plt.ylim(-0.01,0.2)

plt.xlim(-0.03,0.3)
plt.ylim(-0.005,0.025)

plt.figure()
sns.kdeplot(
    x="e_density",
    y="nb_deg1_density",
    hue="timepoint",
    data=data,
    levels=[ 0.2, 1],
    shade=True,
    # bw_adjust=.8,
    palette=colors[[True, True, True, False, True, True, True,True, True, True,True, True]].tolist(),
    alpha=0.05
)
reg=[500, 733, 378, 329, 247, 4]
reg=[500, 733, 378, 329, 247, 4, 302, 672, 512, 1097, 1080, 698, 803,475]
for r in reg:
    plt.figure()
    # plt.scatter(x=data['e_density'],y=data['nb_deg1_density'], c='gray', alpha=0.5)
    for t in tmp:
        print(t)
        id=r#ano.find(r, key='order')['id']
        d=data[data['timepoint']==t]
        d=d[d['region']==id]
        s=ano.find(r, key='id')['acronym']
        print(s)
        try:
            plt.scatter(d['rad_vess_density'], d['nb_deg1_density'], c=d['colors'].values[0], alpha=1)
            # plt.text(d['e_density'], d['nb_deg1_density'], s=s, fontsize=16,  c=d['colors'].values[0])
        except:
            print('no timepoint in region')

        plt.xlim(0.2,0.8)
        plt.ylim(-0.01,0.03)
        # plt.xlim(-0.03,0.3)
        # plt.ylim(-0.005,0.025)
    plt.title(s)

    plt.xlabel('rad_vess_density')
    plt.ylabel('nb_deg1_density')



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # print(v1, v1_u)
    # print(v2, v2_u)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


reg=[500, 733, 378, 329, 247, 4,302, 672, 512, 1097, 1080, 698,475]
distances=[]
angles=[]

for r in reg:
    i=0
    print(r)
    A=[]
    D=[]
    for t in tmp[:-1]:
        print(t,tmp[i])
        id=r#ano.find(r, key='order')['id']
        d=data[data['timepoint']==tmp[i]]
        d=d[d['region']==id]

        d1=data[data['timepoint']==tmp[i+1]]
        d1=d1[d1['region']==id]

        dist=np.sqrt(np.power(d1['e_density'].values-d['e_density'].values, 2)
                     + np.power(d1['nb_deg1_density'].values-d['nb_deg1_density'].values, 2))
        try:
            d_1=data[data['timepoint']==tmp[i-1]]
            d_1=d_1[d_1['region']==id]
            angle=angle_between(np.array([d['e_density'].values[0]-d_1['e_density'].values[0], d['nb_deg1_density'].values[0]-d_1['nb_deg1_density'].values[0]]),
                                np.array([d1['e_density'].values[0]-d['e_density'].values[0], d1['nb_deg1_density'].values[0]-d['nb_deg1_density'].values[0]]))
            A.append(angle)
            print('angle : ', angle)
        except:
            print('could not compute angle :', i)

        D.append(dist[0])
        print('dist : ', dist)
        i=i+1

    distances.append(D)
    angles.append(A)


distances=np.array(distances)
angles=np.array(angles)


vector_regions=np.concatenate((distances, angles), axis=1)
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(vector_regions, 'ward')#'single'#centroid

labelList = [ano.find(r)['acronym'] for r in reg]

plt.figure()
dendrogram(linked,
           orientation='top',
           labels=labelList,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()

























X=X.fillna(0)

pca = PCA(n_components=2,svd_solver='full')
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)


# Percentage of variance explained for each components
print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
)
print(pca.components_)

plt.figure()

lw = 2

for color, i, target_name in zip(colors, np.unique(data['timepoint'].values), np.unique(data['timepoint'].values)):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
for i, t in enumerate(np.unique(data['timepoint'])):
    sns.kdeplot(data[data['timepoint']==t], color=colors[i])


plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of dev dataset")

plt.figure()
for color, i, target_name in zip(colors, np.unique(df['timepoint'].values), np.unique(df['timepoint'].values)):
    plt.scatter(
        X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of dev dataset")

plt.show()


# Reformat and view results
loadings = pd.DataFrame(pca.components_.T,
                            columns=['PC%s' % _ for _ in range(len(X.columns))],
                            index=X.columns)
print(loadings)

plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Components')