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


reg_name=['cortex', 'striatum', 'hippocampus']


reg_name=['brain']
TP=[1, 5, 30, 3, 7, 6, 14]


workdirs=[ work_dirP1, work_dirP5, work_dirAdult,work_dirP3,work_dirP7,work_dirP6,work_dirP14]
controlslist=[ controlsP1, controlsP5, controlsAdult,controlsP3,controlsP7,controlsP6,controlsP14]




def modularity_measure(partition, graph, vertex_prop):
    u, c= np.unique(partition, return_counts=True)
    vp=graph.vertex_property(vertex_prop)
    K=graph.n_edges
    # trash_clusters=u[np.where(c<20)]
    Q=0
    Qs=[]
    for e in u:
        vf=np.zeros(graph.n_vertices)
        vf[np.where(vp==e)[0]]=1
        cluster= graph.sub_graph(vertex_filter=vf)
        ms=cluster.n_edges
        ks=np.sum(cluster.vertex_degrees())
        Q=Q+(ms/K)-((ks/(2*K))**2)
        Qs.append((ms/K)-((ks/(2*K))**2))
    print(Q)
    return Q, Qs










import ClearMap.IO.IO as io
anot=io.read('/data_2to/pix/anoP4.tif')
anot=anot-32768
reg_ids=np.unique(anot)

D1=pd.DataFrame()
for j, controls in enumerate(controlslist):
    print(j,workdirs[j],controls)
    st='dev'
    print('dev brain')
    ano.initialize(label_file = '/data_2to/pix/region_ids_test_ADMBA_audbarmo.json',
                   annotation_file = '/data_2to/pix/anoP4.tif')
    regions=regions=[(0,0),(451,10), (452, 10), (450,10)]#[(0,0),(449,9)]#, (567,9),(259,9)]
    if TP[j]==14 or TP[j]==6:
        regions=[(0,0)]
    for i, g in enumerate(controls):
        print(g)
        work_dir=workdirs[j]
        try:
            G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
        except:
            try:
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse_subregion.gt')
            except:
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')

        modules=np.zeros(G.n_vertices)
        label = G.vertex_annotation();
        val=1
        for reg in reg_ids:
            order, level= ano.find(reg, key='id')['order'],ano.find(reg, key='id')['level']
            print(level, order, ano.find(order, key='order')['name'])
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            modules[np.asarray(vertex_filter==1).nonzero()[0]]=val
            val=val+1

        G.add_vertex_property('modules', modules)
        Q, Qs=modularity_measure(modules, G, 'modules')
        print(Q, Qs)
        D1=D1.append({'timepoint': TP[j],'modularity':Q},ignore_index=True)

D1.to_csv('/data_2to/dev/region_constraint_modularity_dev.csv', index=False)


plt.figure()
sns.set_style(style='white')

sns.lineplot(x="timepoint", y='modularity',err_style='bars',
             data=D1)
sns.despine()