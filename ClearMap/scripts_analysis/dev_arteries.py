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

work_dirP14='/data_2to/dev_arteries/arteries_P14/'#'/data_2to/p14'
controlsP14=['2']#['2', '3']

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

work_dir_P9='/data_2to/dev_arteries/arteries_p9'
controls_P9=['2','3']


reg_name=['cortex', 'striatum', 'hippocampus']


reg_name=['brain']
TP=[1, 5, 30, 3, 7, 6, 14, 60, 210, 21, 12, 90, 9]

# reg_name=['brain','barrels', 'auditory', 'motor']

workdirs=[ work_dirP1, work_dirP5, work_dirAdult,work_dirP3,work_dirP7,work_dirP6,work_dirP14,work_dir_2M,work_dir_7M, work_dir_P21,work_dir_P12,work_dir_3M,work_dir_P9]
controlslist=[ controlsP1, controlsP5, controlsAdult,controlsP3,controlsP7,controlsP6,controlsP14,controls_2M,controls_7M,controls_P21,controls_P12,controls_3M,controls_P9]

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
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P10/ano_P10.nrrd']

reg_ids=[500, 329, 247, 669]
reg_name=['Somatomotor areas',
          'Primary somatosensory area, barrel field','Auditory areas',
          'Visual areas']
reg_ids=[315,1080, 549, 1097, 512, 343, 477, 313]
reg_name=['Isocortex', 'Hippocampal region', 'Thalamus', 'Hypothalamus', 'Cerebellum', 'Brain stem', 'Striatum','Midbrain']

reg_ids=[475, 398, 4,733]
reg_name=['Medial geniculate complex',"Superior olivary complex","Inferior colliculus",'Ventral posteromedial nucleus of the thalamus']
regions=[]
for id in reg_ids:
    regions.append((ano.find(id)['id'],ano.find(id)['level']))

get_reg=1
Table=[]
timepoints_temp=[]
reg_temp=[]

D1=pd.DataFrame()
for j, controls in enumerate(controlslist):
    controls=controlslist[j]
    print(j,TP[j], workdirs[j],controls)
    st='dev'
    print('dev brain')
    # ano.initialize(label_file = '/data_2to/pix/region_ids_test_ADMBA_audbarmo.json',
    #                extra_label = None, annotation_file = '/data_2to/pix/anoP4.tif')
    annotation_file=anot_f[j]
    if TP[j]==6 or TP[j]==9 or TP[j]==14 or TP[j]==30 or TP[j]==60 or TP[j]==210:

        if TP[j]!=30 and TP[j]!=5 and TP[j]!=60 and TP[j]!=210 and TP[j]!=90:
            # get_reg=0
            print('no extra label')
            ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                           extra_label=None,annotation_file = annotation_file)#extra_label=None,
        else:
            ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                           annotation_file = annotation_file)

        # regions=regions=[(0,0),(451,10), (452, 10), (450,10)]#[(0,0),(449,9)]#, (567,9),(259,9)]
        # if TP[j]==14 or TP[j]==6:
        #     regions=[(0,0)]
        for i, g in enumerate(controls):
            print(g)
            work_dir=workdirs[j]
            artere=True
            # if TP[j]==9:
            #     G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_arteries_realigned.gt')

            # else:
            try:
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse_smoothed.gt')
            except:
                try:
                    G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
                except:
                    G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')

                # try:
                #     artery=G.vertex_property('artery')
                #     # vein=G.vertex_property('vein')
                #     # artery=from_e_prop2_vprop(graph, 'artery')
                #     # vein=from_e_prop2_vprop(graph, 'vein')
                # except:
                #     try:
                #         artery=from_e_prop2_vprop(G , 'artery')
                #         # vein=from_e_prop2_vprop(G , 'vein')
                #     except:
            try:
                print('artery binary')
                artery=from_e_prop2_vprop(G, 'artery_binary')
                # vein=G.vertex_property('vein')
            except:
                print('no artery vertex properties')
                artere=False


            # artery_vein=np.logical_or(artery, vein)
            if artere:
                if TP[j]!=9:
                    print(np.sum(artery))
                    G=G.sub_graph(vertex_filter=artery)



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
                # deg1graph=G.sub_graph(vertex_filter=degrees==1)
                vertex_dist=G.vertex_property('distance_to_surface')
                label=G.vertex_annotation();
                # print(np.unique(label))
                # G = G.largest_component()

                for k,region in enumerate(regions):

                    id, level=region
                    order= ano.find(id, key='id')['order']
                    print(id, order, level, ano.find(order, key='order')['name'])
                    if TP[j]==21 or TP[j]==12 or TP[j]==9 or TP[j]==14:
                        print('labelled by ID')
                        graph=extract_AnnotatedRegion(G, region, state='21')
                        # print(np.unique(label))
                    else:
                        print('else')
                        graph=extract_AnnotatedRegion(G, region)


                    if graph.n_vertices!=0:
                        nbdeg1=np.sum(graph.vertex_degrees()==1)
                        r=nbdeg1/graph.n_vertices
                        degrees=graph.vertex_degrees()
                        coordinates = graph.edge_geometry_property('coordinates')
                        indices = graph.edge_property('edge_geometry_indices')

                        L = 0
                        from multiprocessing import Pool
                        p = Pool(15)
                        import time
                        start = time.time()
                        length = np.array(
                            [p.map(get_length, [(i, ind, coordinates) for i, ind in enumerate(indices)])])
                        p.close()
                        end = time.time()
                        print(end - start)
                        length=length[0]
                        L=np.sum(length)*1e3
                        # if TP[j]==60:
                        #     L=L*25/1.6
                        Lmm = L
                        D1=D1.append({'timepoint': TP[j],'nbdeg1': nbdeg1, 'deg1ratio': r, 'lengthtot':Lmm,'region': reg_name[k], 'n_edges':graph.n_edges,'n_vertices':graph.n_vertices, 'brainID':g},ignore_index=True)#([j, Lmm, bp/Lmm])#

                        # vertex_dist=graph.vertex_property('distance_to_surface')
                        # vertex_dist=vertex_dist[vertex_dist<=350]
                        # # bp_dist=vertex_dist[np.asarray(vertex_filter==1).nonzero()[0]]
                        #
                        # histbp, bins = np.histogram(vertex_dist, bins=10)#, normed=normed)
                        #
                        # Table.append(np.concatenate((histbp, np.array([TP[j],reg_name[k]])), axis=0))

D1.to_csv('/data_2to/dev_arteries/BP_smoothed_all_brain_metaregion.csv', index=False)

feat_cols = [ 'feat'+str(i) for i in range(10) ]
feat_cols.append('timepoint')
feat_cols.append('region')
df = pd.DataFrame(Table,columns=feat_cols)
# df['timepoints']=timepoints_temp
# df['region']=reg_temp
df.to_csv('/data_2to/dev_arteries/D1new_annot_arteries_cortex_profile_normed.csv', index=False)




# regions=[ano.find(r, key='order')['id'] for r in  [13, 38, 52, 101, 122,137,186]]

data2plt=df.iloc[:, :12]#df.iloc[:, [0,1,2,3,4,5,6,7,8,9,-2,-1]]#[:, 10:]#[:, [0,1,2,3,4,5,6,7,8,9,-2,-1]]
# data2plt=data2plt[data2plt['region'].isin(regions)]
normed=False
from sklearn.preprocessing import normalize

for col in df.columns[:-1]:
    new_col=df[col].values.astype(int)
    df[col]=new_col
# TP_temp=[1,3,5,7,9, 12,14,21,30,60,90,210]
TP_temp=np.unique(df['timepoint'].values)
pal = sns.cubehelix_palette(len(TP_temp), rot=-.25, light=.7)
# pal=sns.color_palette("icefire", len(TP_temp))

for region in regions:
    plt.figure()
    region_name=ano.find(region[0])['name']
    print(region_name)
    data=df[df['region']==region_name]

    for i, tp in enumerate(TP_temp):
        d=data[data['timepoint']==tp].iloc[:,:-2]
        Cpd_c = pd.DataFrame(d).melt()
        print(tp, Cpd_c.shape)
        col=pal[i]
        # if tp==30:
        #     col='indianred'
        # if tp==210:
        #     col='darkred'
        # if tp==21:
        #     col='darkgoldenrod'
        # if tp==14:
        #     col='goldenrod'
        # if tp==60:
        #     col='olive'
        # if tp==9:
        #     col='forestgreen'
        sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color=col, linewidth=2.5)


    plt.title(region_name)
    plt.legend(TP_temp)

import ClearMap.Analysis.Graphs.GraphGt_old as ggto
graph=ggto.load('/data_2to/dev_arteries/arteries_p9/3/3_graph_correcteduniverse.gt')
# graph=ggto.load('/data_2to/dev_arteries/arteries_P14/2/2_graph_correcteduniverse.gt')
thalamus=[549, 5]
graph=extract_AnnotatedRegion(graph, thalamus, state='21')
# artery=from_e_prop2_vprop(graph, 'artery')
artery=from_e_prop2_vprop(graph, 'artery_binary')
print(np.sum(artery))
graph=graph.sub_graph(vertex_filter=artery)

# gs = graph.sub_slice((slice(1,500), slice(0,500), slice(100,135)),coordinates='coordinates_atlas');
gs = graph.sub_slice((slice(1,500), slice(250,300), slice(0,500)),coordinates='coordinates_atlas');

# gs = graph.sub_slice((slice(1,1000), slice(0,1000), slice(100,200)),coordinates='coordinates_atlas');
# gs = graph.sub_slice((slice(1,1000), slice(200,300), slice(0,1000)),coordinates='coordinates_atlas');

# gs = graph.sub_slice((slice(1,500), slice(0,500), slice(115,145)),coordinates='coordinates_atlas');
# gs = graph.sub_slice((slice(1,500), slice(150,250), slice(0,500)),coordinates='coordinates_atlas');

vertex_colors = ano.convert_label(gs.vertex_annotation(), key='id', value='rgba');

connectivity = gs.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
# edge_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
# edge_colors[edge_vein_label  >0] = [0.0,0.0,0.8,1.0]

p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);