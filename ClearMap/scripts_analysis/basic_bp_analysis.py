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


def extract_AnnotatedRegion(graph, region, state='order', return_graph=True):

    id, level = region
    order= ano.find(id, key='id')['order']
    print(state, level,order, id, ano.find(id, key='id')['name'])

    label = graph.vertex_annotation();

    if state=='order':
        print('order')
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter = label_leveled == order;
    else:
        try:
            label[label<0]=0
            label_leveled = ano.convert_label(label, key='id', value='id', level=level)
            vertex_filter = label_leveled == id;
        except:
            print('could not extract region')

    if return_graph:
        gss4 = graph.sub_graph(vertex_filter=vertex_filter)
        return gss4
    else:
        return vertex_filter


def get_length(args):
    i, ind ,coordinates=args
    # print(i)
    diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
    ll=np.sum(np.linalg.norm(diff, axis=1))
    ll=ll * 1.6*1e-6#0.000025
    # Basicanalysis=Basicanalysis.append({'timepoint': TP[j], 'region':reg_name[k], 'length': ll, 'brainID': g, 'radius': radii[i]},ignore_index=True)
    # if ll<=20:#0.000025#(np.sum(np.linalg.norm(diff, axis=1))* 1.6)
    #     d1=graph.edge(i).target().out_degree()
    #     d2=graph.edge(i).source().out_degree()
    #     svd.append(d1)
    #     svd.append(d2)
    return ll


def remove_surface(graph, width):
    # distance_from_suface = from_v_prop2_eprop(graph, graph.vertex_property('distance_to_surface'))
    # ef=distance_from_suface>width
    vf=graph.vertex_property('distance_to_surface')>width
    g=graph.sub_graph(vertex_filter=vf)
    return g

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

work_dir_P9='/data_SSD_2to/220725_P9'
controls_P9=['2','3','4']


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

#
# work_dirP9='/data_SSD_2to/mbp/P9'
# work_dirP12='/data_SSD_2to/mbp/P12'
# workdirs=[ work_dirP9, work_dirP12]
# controlslist=[['P9'], ['P12']]
# anot_f=['/data_2to/alignement/atlases/new_region_atlases/P7/smoothed_anot_P7.nrrd',
#         '/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed.nrrd']
# TP=[9, 12]
# reg_name=['brain','barrels', 'auditory', 'motor']
get_reg=1
regions=[(3,3)]
reg_ids=[313, 698, 1080, 500, 329, 247, 733]
reg_name=['Midbrain', 'Olfactory areas', 'Hippocampal region', 'Somatomotor areas',
          'Primary somatosensory area, barrel field','Auditory areas',
          'Ventral posteromedial nucleus of the thalamus']

reg_ids=[500, 329, 247, 669]
reg_name=['Somatomotor areas',
          'Primary somatosensory area, barrel field','Auditory areas',
          'Visual areas']
reg_ids=[475, 398, 4]
reg_name=['Medial geniculate complex',"Superior olivary complex","Inferior colliculus"]


reg_ids=[475, 398, 4,733]
reg_name=['Medial geniculate complex',"Superior olivary complex","Inferior colliculus",'Ventral posteromedial nucleus of the thalamus']

# reg_ids=[315, 549, 1097, 512, 343, 477]
# reg_name=['Isocortex', 'Thalamus', 'Hypothalamus', 'Cerebellum', 'Brain stem', 'Striatum']
regions=[]
for id in reg_ids:
    regions.append((ano.find(id)['id'],ano.find(id)['level']))

D1=pd.DataFrame()


for j, controls in enumerate(controlslist):
    # j=j+3
    controls=controlslist[j]
    print(j,workdirs[j],controls, TP[j])
    annotation_file=anot_f[j]

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
        # ano.initialize(label_file = '/data_2to/pix/region_ids_test_ADMBA_audbarmo.json',
        #                extra_label = None, annotation_file = '/data_2to/pix/annotation_halfbrain_with_audbarmot.tif')
        # regions=regions=[(0,0),(451,10), (452, 10), (450,10)]#[(0,0),(449,9)]#, (567,9),(259,9)]
        # if TP[j]==14 or TP[j]==6:
        #     regions=[(0,0)]
    for i, g in enumerate(controls):
        print(g)
        work_dir=workdirs[j]
        try:
            # G = ggt.load(work_dir + '/' + '/' +'3_graph_uncrusted.gt')
            G = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
        except:
            try:
                G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')




        #extract isocortex
        if get_reg==0:
            order, level=(0,0)
            vertex_filter = np.zeros(G.n_vertices)
            # print(level, order, ano.find(6, key='order')['name'])
            label = G.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter[label_leveled == order] = 1;
            G = G.sub_graph(vertex_filter=vertex_filter)

            label=G.vertex_annotation();
            regions=[(u_l, ano.find(u_l, key='order')['level']) for u_l in np.unique(label)][1:]
            reg_name=[ano.find(u_l, key='order')['name'] for u_l in np.unique(label)][1:]
            print(regions)
            print(reg_name)
            get_reg=1


        for k,region in enumerate(regions):
            label=G.vertex_annotation();
            print(np.unique(label))
            G = G.largest_component()
            id, level=region
            order= ano.find(id, key='id')['order']
            print(id, order, level, ano.find(order, key='order')['name'])
            if TP[j]==21 or TP[j]==12 or TP[j]==9:
                print('labelled by ID')
                graph=extract_AnnotatedRegion(G, region, state='21')
                print(np.unique(label))
            else:
                graph=extract_AnnotatedRegion(G, region)
            nbdeg1=np.sum(graph.vertex_degrees()==1)
            r=nbdeg1/graph.n_vertices
            degrees=graph.vertex_degrees()
            # graph=graph.sub_graph(vertex_filter=graph.vertex_degrees()!=1)
            # radii=graph.edge_radii()
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

D1.to_csv('/data_2to/dev/BP_smoothed_all_brain_metaregion.csv', index=False)

D1nodeg1=pd.read_csv('/data_2to/dev/BP_smoothed_all_brain_metaregion_nodeg1.csv')
D1=pd.read_csv('/data_2to/dev/BP_smoothed_all_brain_metaregion.csv')
D1['lengthtot']=D1['lengthtot']*1e3/1.6 #convertion in mm

D1nodeg1=D1nodeg1[D1nodeg1['timepoint']!=6]
D1=D1[D1['timepoint']!=6]

D1['length_unconnected'] = D1['lengthtot'] - D1nodeg1['lengthtot']
D12m['n_vertices/length'] = D12m['n_vertices'] / D12m['lengthtot']

plt.figure()
sns.barplot(data=D12m, x="region", y="n_vertices/length")

plt.figure()
for rs in reg_name:
    if rs=='Brain stem':
        sns.lineplot(x="timepoint", y='lengthtot',err_style='bars', color='darkorchid',
                     data=D1[D1['region']==rs])
    else:
        sns.lineplot(x="timepoint", y='lengthtot',err_style='bars', color=ano.find(rs, key='name')['rgb'],
                 data=D1[D1['region']==rs])

plt.legend(reg_name)
# D1.to_csv('/data_2to/dev/mbp_all_brain_subregion.csv', index=False)
D1.to_csv('/data_2to/dev/BP_smoothed_all_brain_subregion.csv', index=False)
# D1.to_csv('/data_2to/dev/BP_smoothed_cortical_subregion.csv', index=False)
# D1.to_csv('/data_2to/dev/BP_subregion.csv', index=False)
# D1.to_csv('/data_2to/dev/all_subregion.csv', index=False)
D1=pd.read_csv('/data_2to/dev/BP_smoothed_cortical_subregion.csv')
# plt.figure()
# sns.lineplot(x="timepoint", y='n_vertices',err_style='bars',hue='region',data=D1)

D1=pd.read_csv('/data_2to/dev/BP_smoothed_all_brain_subregion.csv')
D1_mbp=pd.read_csv('/data_2to/dev/mbp_all_brain_subregion.csv')

D1['n_vertices/length'] = D1['n_vertices'] / D1['lengthtot']
D1_mbp['n_vertices/length'] = D1_mbp['n_vertices'] / D1_mbp['lengthtot']

D1['nbdeg1/length'] = D1['nbdeg1'] / D1['lengthtot']
D1=D1[D1['timepoint']!=6]
D1brain=D1[D1['region']=='Isocortex']

Rs=['Auditory areas','Caudoputamen', 'Cerebellum','Hindbrain', 'Hippocampal region',
    'Hypothalamus', 'Inferior colliculus', 'Midbrain','Olfactory areas','Primary somatosensory area',
    'Primary somatosensory area, barrel field', 'Primary visual area','Somatomotor areas', 'Thalamus']#np.unique(D1['region'].values)
for rs in Rs:
    plt.figure()
    sns.lineplot(x="timepoint", y='n_vertices/length',err_style='bars', color='indianred',
             data=D1[D1['region']==rs])
    plt.ylim(5000, 15000)
    plt.twinx()
    sns.lineplot(x="timepoint", y='n_vertices/length',err_style='bars', color='cadetblue',
                 data=D1brain)
    plt.title(rs)
    plt.ylim(5000, 15000)


    plt.figure()
    sns.lineplot(x="timepoint", y='nbdeg1/length',err_style='bars', color='indianred',
                 data=D1[D1['region']==rs])
    plt.ylim(0, 5000)
    plt.twinx()
    sns.lineplot(x="timepoint", y='nbdeg1/length',err_style='bars', color='cadetblue',
                 data=D1brain)
    plt.title(rs)
    plt.ylim(0, 5000)

plt.figure()
D1brain=D1[D1.region=='Isocortex']
D1reg=D1[D1.region!='Isocortex']
sns.lineplot(x="timepoint", y='n_vertices',err_style='bars',hue='region',
             data=D1brain)
plt.twinx()

sns.lineplot(x="timepoint", y='n_vertices',err_style='bars',hue='region',
             data=D1reg)

plt.figure()
sns.lineplot(x="timepoint", y='deg1ratio',err_style='bars',hue='region',
             data=D1brain)
plt.twinx()
sns.lineplot(x="timepoint", y='deg1ratio',err_style='bars',hue='region',
             data=D1reg)

plt.figure()
sns.lineplot(x="timepoint", y='lengthtot',err_style='bars',hue='region',
             data=D1brain)
plt.twinx()
sns.lineplot(x="timepoint", y='lengthtot',err_style='bars',hue='region',
             data=D1reg)

plt.figure()
sns.lineplot(x="timepoint", y='n_edges',err_style='bars',hue='region',
             data=D1brain)
plt.twinx()
sns.lineplot(x="timepoint", y='n_edges',err_style='bars',hue='region',
             data=D1reg)


D1brain['n_vertices/length'] = D1brain['n_vertices'] / D1brain['lengthtot']
D1reg['n_vertices/length'] = D1reg['n_vertices'] / D1reg['lengthtot']
plt.figure()
sns.lineplot(x="timepoint", y='n_vertices/length',err_style='bars',hue='region',
             data=D1brain)
plt.twinx()
sns.lineplot(x="timepoint", y='n_vertices/length',err_style='bars',hue='region',
             data=D1reg)


controls_2M=['2R','3R','5R', '8R']
mutants_2M=['1R','7R', '6R', '4R']
work_dir_2M='/data_SSD_2to/191122Otof'
# states=[controls, mutants]

work_dir_1M='/data_2to/otof1M'
controls_1M=[ '1w', '3w', '5w', '6w', '7w']
mutants_1M=['1k', '2k', '3k', '4k']
# states=[controls, mutants]

mutants_6M=['2R','3R','5R', '1R']
controls_6M=['7R','8R', '6R']
work_dir_6M='/data_SSD_1to/otof6months'


work_dir_3M='/data_2to/otof3M/new_vasc'
# controls=['1w', '2w', '4w', '5w','6w']
# mutants=[ '1k','3k','4k', '5k', '6k']
controls_3M=['2w', '4w', '5w','6w']
mutants_3M=[ '3k', '5k', '6k']



wds=[work_dir_1M,work_dir_2M, work_dir_3M, work_dir_6M]
ctrls=[controls_1M,controls_2M,controls_3M, controls_6M]
mtts=[mutants_1M, mutants_2M,mutants_3M, mutants_6M]
TP=['1 months', '2 months', '3 months', '6 months']
region_list = [(142, 8), (149, 8), (128, 8), (156, 8)]#auditory regions






work_dir_ED = '/data_SSD_2to/earlyDep'
controlsED=['4', '7', '10','12', '15']
mutantsED=[ '3', '6', '9', '11', '16']


work_dir_3W='/data_SSD_2to/whiskers_graphs/new_graphs'
controls_3W=['158L','162L', '164L']#'142L',
mutants_3W=['138L','141L', '163L']#, '165L']


work_dir_5M='/data_2to/whiskers5M/R'
mutants_5M=['433', '457', '458']#456 not annotated ?
controls_5M=['467', '469']


TP=['1 months', '3 months', '5 months']#, '6 months']
wds=[work_dir_ED,work_dir_3W, work_dir_5M]#,work_dir_10W]
ctrls=[controlsED,controls_3W,controls_5M]#,controls_10W]
mtts=[mutantsED, mutants_3W,mutants_5M]#,mutants_10W]

region_list = [(54, 9), (47, 9)]

region_list = [(13, 7), (19, 8), (25, 8), (103, 8)]
#[MO, MOs, MOp, SSs]
dt=pd.DataFrame()
for i in range(5):
    work_dir=wds[i]
    print(work_dir)
    ctrl=ctrls[i]
    mtt=mtts[i]
    for control in ctrl:
        graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
        # vertex_filter = np.zeros(graph.n_vertices)
        for j, rl in enumerate(region_list):
            vertex_filter = np.zeros(graph.n_vertices)
            order, level = region_list[j]
            print(level, order, ano.find(order, key='order')['name'])
            label = graph.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter[label_leveled == order] = 1;
            g = graph.sub_graph(vertex_filter=vertex_filter)
            dt=dt.append({'timepoint': TP[i], 'condition':'control', 'n_edges':g.n_edges,'n_vertices':g.n_vertices, 'region':order},ignore_index=True)

    for mutant in mtt:
        graph = ggt.load(work_dir + '/' + mutant + '/' + 'data_graph_correcteduniverse.gt')
        # vertex_filter = np.zeros(graph.n_vertices)
        for j, rl in enumerate(region_list):
            vertex_filter = np.zeros(graph.n_vertices)
            order, level = region_list[j]
            print(level, order, ano.find(order, key='order')['name'])
            label = graph.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter[label_leveled == order] = 1;
            g = graph.sub_graph(vertex_filter=vertex_filter)
            dt=dt.append({'timepoint': TP[i], 'condition': 'mutant', 'n_edges':g.n_edges,'n_vertices':g.n_vertices,'region':order},ignore_index=True)



plt.figure()
sns.lineplot(x="timepoint", y='n_vertices',err_style='bars',hue='condition',
             data=dt)

plt.figure()
sns.lineplot(x="timepoint", y='n_vertices',err_style='bars',hue='condition',
             data=dt[dt['region']==103])
plt.title(ano.find(103, key='order')['name'])



### BP deg 1 profile

get_reg=1
Table=[]
timepoints_temp=[]
reg_temp=[]
for j, controls in enumerate(controlslist):
    controls=controlslist[j]
    print(j,workdirs[j],controls)
    st='dev'
    print('dev brain')
    # ano.initialize(label_file = '/data_2to/pix/region_ids_test_ADMBA_audbarmo.json',
    #                extra_label = None, annotation_file = '/data_2to/pix/anoP4.tif')
    annotation_file=anot_f[j]

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

        try:
            G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse_smoothed.gt')
        except:
            try:
                G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')




        #extract isocortex
        if get_reg==0:
            order, level=(6,6)
            vertex_filter = np.zeros(G.n_vertices)
            print(level, order, ano.find(6, key='order')['name'])
            label = G.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter[label_leveled == order] = 1;
            G = G.sub_graph(vertex_filter=vertex_filter)

            label=G.vertex_annotation();
            reg_ids=[ano.find(u_l, key='order')['id'] for u_l in np.unique(label)]
            get_reg=1


        degrees = G.vertex_degrees()
        # deg1graph=G.sub_graph(vertex_filter=degrees==1)
        vertex_dist=G.vertex_property('distance_to_surface')
        label=G.vertex_annotation();
        # print(np.unique(label))
        G = G.largest_component()

        for k,region in enumerate(regions):

            id, level=region
            order= ano.find(id, key='id')['order']
            print(id, order, level, ano.find(order, key='order')['name'])
            if TP[j]==21 or TP[j]==12 or TP[j]==9:
                print('labelled by ID')
                graph=extract_AnnotatedRegion(G, region, state='21')
                # print(np.unique(label))
            elif TP[j]>=30:
                if id==329:
                    print('adult')
                    vf1=extract_AnnotatedRegion(G, region, return_graph=False)
                    #also extracts nose area in adults
                    vf2=extract_AnnotatedRegion(G, (353, level), return_graph=False)
                    vertex_filter=np.logical_or(vf1, vf2)
                    graph = G.sub_graph(vertex_filter=vertex_filter)
                else:
                    graph=extract_AnnotatedRegion(G, region)
            else:
                graph=extract_AnnotatedRegion(G, region)

            # timepoints_temp.append(TP[j])
            # reg_temp.append(reg_name[k])

            vertex_dist=graph.vertex_property('distance_to_surface')
            vertex_dist=vertex_dist[vertex_dist<=350]
            # bp_dist=vertex_dist[np.asarray(vertex_filter==1).nonzero()[0]]

            histbp, bins = np.histogram(vertex_dist, bins=10)#, normed=normed)

            Table.append(np.concatenate((histbp, np.array([TP[j],reg_name[k]])), axis=0))


feat_cols = [ 'feat'+str(i) for i in range(10) ]
feat_cols.append('timepoint')
feat_cols.append('region')
df = pd.DataFrame(Table,columns=feat_cols)
# df['timepoints']=timepoints_temp
# df['region']=reg_temp
df.to_csv('/data_2to/dev/D1new_annot_BP_cortex_profile_normed.csv', index=False)


TP_temp=[1,3,5,7,9, 12,14,21,30,60,90,210]
pal = sns.cubehelix_palette(len(TP_temp), rot=-.25, light=.7)
pal=sns.color_palette("icefire", len(TP_temp))

# regions=[ano.find(r, key='order')['id'] for r in  [13, 38, 52, 101, 122,137,186]]

data2plt=df.iloc[:, :12]#df.iloc[:, [0,1,2,3,4,5,6,7,8,9,-2,-1]]#[:, 10:]#[:, [0,1,2,3,4,5,6,7,8,9,-2,-1]]
# data2plt=data2plt[data2plt['region'].isin(regions)]
normed=False
from sklearn.preprocessing import normalize

for col in df.columns[:-1]:
    new_col=df[col].values.astype(int)
    df[col]=new_col


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
