import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt

atlas_path = os.path.join(settings.resources_path, 'Atlas');

# work_dirP0='/data_2to/p0/new'
# controlsP0=['0a', '0b', '0c', '0d']#['2', '3']

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

anot_f=['/data_2to/alignement/atlases/new_region_atlases/P1/smoothed_anot_P1.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P3/smoothed_anot_P3_V6.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P7/smoothed_anot_P7.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed.nrrd']

get_reg=0

def extractSubGraph(edges_centers, mins, maxs):
    """
    Extracts the subgraph contained in the cube between mins and maxs coordinates
          6-------7
         /|      /|
        4-------5 |
        | |     | |
        | 2-----|-3
        |/      |/
        0-------1
    """
    isOver = (edges_centers > mins).all(axis=1)
    isUnder = (edges_centers < maxs).all(axis=1)
    return np.logical_and(isOver, isUnder)


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


def extract_AnnotatedRegion(graph, region, state='dev'):
    order, level = region
    print(state, level, order, ano.find(order, key='order')['name'])

    label = graph.vertex_annotation();
    if state=='dev':
        print('dev')
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter = label_leveled == order;
    elif state=='adult':
        print('adult')
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter = label_leveled == order;

    gss4 = graph.sub_graph(vertex_filter=vertex_filter)
    return gss4

D1=pd.DataFrame()
# D1=pd.read_csv('/data_2to/dev/D1.csv')



get_reg=0
reg_name=['brain','barrels', 'auditory', 'motor']
for j, controls in enumerate(controlslist):
    print(j,workdirs[j],controls)
    work_dir=workdirs[j]
    annotation_file=anot_f[j]
    ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                   extra_label = None, annotation_file = annotation_file)
    if j==2:
        print('adult brain')
        st='adult'
        # regions=[(6,6), (572, 5), (455,6)]
        # regions=[(0,0),(52,9), (122, 7), (13,7)]
        ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                       extra_label = None, annotation_file = os.path.join(atlas_path, 'annotation_25_full.nrrd'))

    else:
        st='dev'
        print('dev brain')
        # ano.initialize(label_file = '/data_2to/pix/region_ids_test_ADMBA_audbarmo.json',
        #                extra_label = None, annotation_file = '/data_2to/pix/annotation_halfbrain_with_audbarmot.tif')

        # regions=[(449,9), (567,9),(259,9)]
        # regions=regions=[(0,0),(451,10), (452, 10), (450,10)]

    for i, g in enumerate(controls):
        try:
            G = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
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
            regions=[(u_l, ano.find(u_l, key='order')['level']) for u_l in np.unique(label)]
            reg_name=[ano.find(u_l, key='order')['name'] for u_l in np.unique(label)]
            get_reg=1


        for k,region in enumerate(regions):
            G = G.largest_component()
            graph=extract_AnnotatedRegion(G, region, st)
            graph = graph.largest_component()
            nb_deg_1=np.sum(graph.vertex_degrees()==1)
            r=nb_deg_1/graph.n_vertices
            D1=D1.append({'timepoint': TP[j], 'nb_deg_1':nb_deg_1, 'deg1ratio': r,'region': reg_name[k]},ignore_index=True)#([j, Lmm, bp/Lmm])#



D1.to_csv('/data_2to/dev/D1_smoothed.csv', index=False)

plt.figure()
sns.lineplot(x="timepoint", y='deg1ratio',err_style='bars',hue='region',
             data=D1)
plt.figure()
sns.lineplot(x="timepoint", y='nb_deg_1',err_style='bars',hue='region',
             data=D1)


D1_diff=pd.DataFrame()

TP=[1,3,5,7,14, 30]
DD=D1[D1['region']!= 'Isocortex']
for k,region in enumerate(regions):
    if reg_name[k]!='Isocortex':
        D1_diff=D1_diff.append({'timepoint': 0, 'diff':0,'region': reg_name[k]},ignore_index=True)
        for i, tp in enumerate(TP[1:]):
            D=DD[DD['region']== reg_name[k]]
            print(tp, TP[i],reg_name[k])
            diff=np.mean(D[D['timepoint']==tp]['nb_deg_1'])-np.mean(D[D['timepoint']==TP[i]]['nb_deg_1'])
            print(diff)
            D1_diff=D1_diff.append({'timepoint': TP[i+1], 'diff':diff,'region': reg_name[k]},ignore_index=True)

plt.figure()
sns.lineplot(x="timepoint", y='diff',err_style='bars',hue='region',
             data=D1_diff)

plt.figure()
ax = D1_diff['diff'].plot.kde()


plt.figure()
sns.kdeplot(x="timepoint", y='nb_deg_1',err_style='bars',hue='region',
             data=D1)


DD=D1[D1.columns[[1,2,3]]]
DD=DD.pivot("timepoint", "region","diff")

D1_diff=D1_diff.pivot("region", "timepoint","diff")
plt.figure()
sns.heatmap(D1_diff)
# D1.to_csv('/data_2to/dev/D1_subregion.csv', index=False)
D1.to_csv('/data_2to/dev/D1_deg1_subregion.csv', index=False)


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

lenghtsTimePoints=[]
featuresTimePoints = pd.DataFrame()#(columns=['timepoint', 'lengthtot', 'bp/length'])#, 'nbMoules', 'modularity'])
SBMTimePoints = pd.DataFrame()
Basicanalysis=pd.DataFrame()
binsL=0
binsD=0
binsR=0
binsS=0
VessellengthDistribution=[]
DegreesdistributionTimePoints=[]
RadiusdistributionTimePoints=[]
ShortVesselsDegree=[]
# reg_name=['brain', 'cortex']
get_reg=1
# reg_name=['brain','barrels', 'auditory', 'motor']
##% total length
state=ctrls
controlslist=state
workdirs=wds
for j, controls in enumerate(controlslist):
    work_dir=wds[j]
    print(j,workdirs[j],controls)
    L_=[]
    D=[]
    R=[]
    S=[]
    regions=[(6,6),(54,9), (127, 7), (13,7)]#[(0,0), (6,6)]#, (572, 5), (455,6)]
    # annotation_file=anot_f[j]
    # ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
    #                extra_label = None, annotation_file = annotation_file)
    # if j==2:
    #     print('adult brain')
    #     st='adult'
        # regions=[(0,0),(52,9), (122, 7), (13,7)]#[(0,0), (6,6)]#, (572, 5), (455,6)]
        # regions=[(0,0)]
        # ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
        #                extra_label = None, annotation_file = os.path.join(atlas_path, 'annotation_25_full.nrrd'))

    # else:
    #     st='dev'
    #     print('dev brain')
        # ano.initialize(label_file = '/data_2to/pix/region_ids_test_ADMBA_audbarmo.json',
        #                extra_label = None, annotation_file = '/data_2to/pix/annotation_halfbrain_with_audbarmot.tif')
        # ano.initialize(label_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',
        #                extra_label = None, annotation_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotationADMBA_thresholded.tif')

        # regions=regions=[(0,0),(451,10), (452, 10), (450,10)]#[(0,0),(449,9)]#, (567,9),(259,9)]
        # regions=[(0,0)]
    for i, g in enumerate(controls):
        L_reg=[]
        D_reg=[]
        R_reg=[]
        S_reg=[]
        print(g)
        work_dir=workdirs[j]
        try:
            G = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
        except:
            try:
                G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')

        if get_reg==0:
            order, level=(6,6)
            vertex_filter = np.zeros(G.n_vertices)
            print(level, order, ano.find(6, key='order')['name'])
            label = G.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter[label_leveled == order] = 1;
            G = G.sub_graph(vertex_filter=vertex_filter)

            label=G.vertex_annotation();
            regions=[(u_l, ano.find(u_l, key='order')['level']) for u_l in np.unique(label)]
            reg_name=[ano.find(u_l, key='order')['name'] for u_l in np.unique(label)]
            get_reg=1

        for k,region in enumerate(regions):
            G = G.largest_component()
            graph=extract_AnnotatedRegion(G, region, st)
            coordinates = graph.edge_geometry_property('coordinates')
            indices = graph.edge_property('edge_geometry_indices')
            degrees=graph.vertex_degrees()
            radii=graph.edge_radii()
            length=[]
            bp = graph.n_vertices
            svd=[]

            L = 0

            from multiprocessing import Pool
            p = Pool(15,maxtasksperchild=1000)
            import time
            start = time.time()

            length = np.array(
                [p.map(get_length, [(i, ind, coordinates) for i, ind in enumerate(indices)])])

            p.close()
            p.join()

            end = time.time()
            print(end - start)
            length=length[0]
            L=np.sum(length)

            # for i, ind in enumerate(indices):
            #     diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
            #     ll=np.sum(np.linalg.norm(diff, axis=1))
            #     L = L + ll
            #     ll=ll * 1.6#0.000025
            #     length.append(ll)
            #     # Basicanalysis=Basicanalysis.append({'timepoint': TP[j], 'region':reg_name[k], 'length': ll, 'brainID': g, 'radius': radii[i]},ignore_index=True)
            #     if ll<=20:#0.000025#(np.sum(np.linalg.norm(diff, axis=1))* 1.6)
            #         d1=graph.edge(i).target().out_degree()
            #         d2=graph.edge(i).source().out_degree()
            #         svd.append(d1)
            #         svd.append(d2)



            Lmm = L * 1.6#0.000025
            print(Lmm)  # m
            try :
                histL, binsL=np.histogram(length, bins=binsL)

            except:
                binsL==0
                histL, binsL=np.histogram(length, bins=np.arange(0,5e-4,5e-6))
            L_reg.append(histL)

            try :
                histR, binsR=np.histogram(radii, bins=binsR)

            except:
                binsR==0
                histR, binsR=np.histogram(radii, bins=10)
            R_reg.append(histR)


            try :
                histD, binsD=np.histogram(degrees, bins=binsD)

            except:
                binsD==0
                histD, binsD=np.histogram(degrees, bins=np.arange(10))
            D_reg.append(histD)

            try :
                histS, binsS=np.histogram(svd, bins=binsS)
            except:
                binsS==0
                histS, binsS=np.histogram(svd, bins=np.arange(10))

            S_reg.append(histS)


            featuresTimePoints=featuresTimePoints.append({'timepoint': TP[j], 'lengthtot':Lmm, 'bp': bp, 'region': reg_name[k]},ignore_index=True)#([j, Lmm, bp/Lmm])#
            # state = gti.minimize_blockmodel_dl(graph.base)
            # modules=state.get_blocks().a
            # s = np.unique(modules).shape[0]
            # graph.add_vertex_property('blocks', modules)

            # Q, Qs = modularity_measure(modules, graph, 'blocks')
            # SBMTimePoints=SBMTimePoints.append({'timepoint': TP[j], 'Nb_modules':s, 'modularity': Q,'region': reg_name[k]},ignore_index=True)#([j, Lmm, bp/Lmm])#

        R.append(R_reg)
        L_.append(L_reg)
        D.append(D_reg)
        # S.append(S_reg)

    RadiusdistributionTimePoints.append(R)
    DegreesdistributionTimePoints.append(D)
    VessellengthDistribution.append(L_)
    ShortVesselsDegree.append(S)
    # RadiusdistributionTimePoints[5]=R
    # DegreesdistributionTimePoints[5]=D
    # VessellengthDistribution[5]=L_
    # ShortVesselsDegree[4]=S
# Basicanalysis.to_csv('/data_2to/dev/Basicanalysis.csv', index=False)
import seaborn as sns
from sklearn.preprocessing import normalize

VessellengthDistribution_t=np.array(VessellengthDistribution)
DegreesdistributionTimePoints_t=np.array(DegreesdistributionTimePoints)
RadiusdistributionTimePoints_t=np.array(RadiusdistributionTimePoints)
ShortVesselsDegree_t=np.array(ShortVesselsDegree)

if state==ctrls:
    # SBMTimePoints.to_csv('/data_2to/dev/new_anot/SBMTimePoints.csv', index=False)
    featuresTimePoints.to_csv(work_dir+'/featuresTimePoints_subregion_control.csv', index=False)#data_2to/dev/new_anot/
    np.save(work_dir+'/RadiusdistributionTimePoints_t_subregion_control.npy',RadiusdistributionTimePoints_t)
    np.save(work_dir+'/DegreesdistributionTimePoints_t_subregion_control.npy',DegreesdistributionTimePoints_t)
    np.save(work_dir+'/VessellengthDistribution_t_subregion_control.npy',VessellengthDistribution_t)
    np.save(work_dir+'/ShortVesselsDegree_t_control.npy',ShortVesselsDegree_t)
    np.save(work_dir+'/new_anot/binsR_control.npy',binsR)
    np.save(work_dir+'/new_anot/binsD_control.npy',binsD)
    np.save(work_dir+'/new_anot/binsS_control.npy',binsS)
    np.save(work_dir+'/new_anot/binsL_control.npy',binsL)
    # df.drop([0, 1])

if state==mtts:
    # SBMTimePoints.to_csv('/data_2to/dev/new_anot/SBMTimePoints.csv', index=False)
    featuresTimePoints.to_csv(work_dir+'/featuresTimePoints_subregion_mutant.csv', index=False)#data_2to/dev/new_anot/
    np.save(work_dir+'/RadiusdistributionTimePoints_t_subregion_mutant.npy',RadiusdistributionTimePoints_t[-5:])
    np.save(work_dir+'/DegreesdistributionTimePoints_t_subregion_mutant.npy',DegreesdistributionTimePoints_t[-5:])
    np.save(work_dir+'/VessellengthDistribution_t_subregion_mutant.npy',VessellengthDistribution_t[-5:])
    np.save(work_dir+'/ShortVesselsDegree_t_mutant.npy',ShortVesselsDegree_t[-5:])
    np.save(work_dir+'/binsR_mutant.npy',binsR[-5:])
    np.save(work_dir+'/binsD_mutant.npy',binsD[-5:])
    np.save(work_dir+'/binsS_mutant.npy',binsS[-5:])
    np.save(work_dir+'/binsL_mutant.npy',binsL[-5:])
    # df.drop([0, 1])

### loading
ShortVesselsDegree_t=np.load('/data_2to/dev/ShortVesselsDegree_t.npy',allow_pickle=True )
VessellengthDistribution_t=np.load('/data_2to/dev/VessellengthDistribution_t.npy',allow_pickle=True )
DegreesdistributionTimePoints_t=np.load('/data_2to/dev/DegreesdistributionTimePoints_t.npy',allow_pickle=True )
RadiusdistributionTimePoints_t=np.load('/data_2to/dev/RadiusdistributionTimePoints_t.npy',allow_pickle=True )

SBMTimePoints=pd.read_csv('/data_2to/dev/new_anot/SBMTimePoints.csv')
featuresTimePoints=pd.read_csv('/data_2to/dev/new_anot/featuresTimePoints_subregion.csv')

featuresTimePoints=pd.read_csv('/data_2to/dev/featuresTimePoints_subregion.csv')
VessellengthDistribution_t=np.load(work_dir+'/VessellengthDistribution_t_subregion_control.npy',allow_pickle=True )
# first_n_column  = featuresTimePoints.iloc[: , :5]
# first_n_column.iloc[36:51+1 , 3]=3
# first_n_column.iloc[20:35+1 , 3]=30
# first_n_column.iloc[12:19+1 , 3]=5
# first_n_column.iloc[0:11+1 , 3]=1

colors_m = ['indianred', 'darkgoldenrod','forestgreen', 'darkblue', 'limegreen', 'skyblue', 'purple']
for i in range(len(reg_name)):
    # plt.figure()
    #
    # for r in range(VessellengthDistribution_t.shape[0]):
    #     Cpd_c = pd.DataFrame(np.array(VessellengthDistribution_t[r])[ :, i, :]).melt()
    #     # Cpd_c = pd.DataFrame(normalize(np.array(ShortVesselsDegree_t[r])[ :, i, :], norm='l2', axis=1)).melt()
    #     sns.lineplot(x="variable", y="value", err_style='bars',  data=Cpd_c, color=colors_m[r], linewidth=2.5)
    # plt.legend(['P0', 'P1', 'P5', '3W', 'P3', 'P7'])
    # plt.yticks(size='x-large')
    # plt.ylabel('short vessels Vessels vertex degrees', size='x-large')
    # # plt.xlabel('cortical depth (um)', size='x-large')
    # plt.xticks(size='x-large')
    # # plt.xticks((1e5*(np.arange(0, np.max(binsS), np.max(binsS) / 10))).astype(int), (1e5*(np.arange(0, np.max(binsS), np.max(binsS) / 10))).astype(int), size='x-large', rotation=20)
    # plt.title(reg_name[i])



    TP=[1,2,3,6,11]
    # TP=[0, 1, 5, 30, 3, 7, 6, 14]
    sortedTP=np.argsort(np.array(TP))
    # TPlabels=[ 'P1','P3', 'P5', 'P6', 'P7', 'P14', 'W4']
    TPlabels=[ '1M','2M', '3M', '6M', '11M']
    fig, axs = plt.subplots(len(TPlabels))
    plt.title(reg_name[i])
    sns.set_style('white',rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(len(TPlabels), rot=-.25, light=.7)
    for r in range(VessellengthDistribution_t.shape[0]):
        # Cpd_c = pd.DataFrame(np.array(VessellengthDistribution_t[r])[ :, i, :]).melt()
        Cpd_c = pd.DataFrame(normalize(np.array(VessellengthDistribution_t[r])[ :, i, :], norm='l2', axis=1)).melt()
        a=np.asarray(sortedTP==r).nonzero()[0][0]
        print(r,a)
        sns.lineplot(ax=axs[a],x="variable", y="value", err_style='bars',  data=Cpd_c, color=pal[a], linewidth=2.5)
        axs[a].set(yticks=[], ylabel="",xlabel="")
        axs[a].set_xlim(0,10)
        axs[a].text(0, .9, TPlabels[a], fontweight="bold", color=pal[a], ha="left", va="center", transform=axs[a].transAxes)

        if a==4:
            plt.xticks((1e5*(np.arange(0, np.max(binsL), np.max(binsL) / 10))).astype(int), (1e5*(np.arange(0, np.max(binsL), np.max(binsL) / 10))).astype(int), size='x-large', rotation=20)
            axs[a].set_xlim(0,10)
            axs[a].text(0, .9, TPlabels[a], fontweight="bold", color=pal[a], ha="left", va="center", transform=axs[a].transAxes)
            axs[a].set(yticks=[], ylabel="", xlabel="")
            sns.despine(bottom=True, left=True)
        else:
            axs[a].set(xticks=[], ylabel="")
            sns.despine(bottom=True, left=True)



    plt.subplots_adjust(hspace=-.25)


    # plt.legend(['P0', 'P1', 'P5', '4W', 'P3', 'P7', 'P14'])
    plt.yticks(size='x-large')
    plt.ylabel('Vessels length', size='x-large')
    # plt.xlabel('cortical depth (um)', size='x-large')
    plt.xticks(size='x-large')
    # plt.xticks((1e5*(np.arange(0, np.max(binsL), np.max(binsL) / 10))).astype(int), (1e5*(np.arange(0, np.max(binsL), np.max(binsL) / 10))).astype(int), size='x-large', rotation=20)



    plt.figure()
    # colors_m = ['indianred', 'darkorange','forestgreen', 'royalblue']

    for r in range(RadiusdistributionTimePoints_t.shape[0]):
        # Cpd_c = pd.DataFrame(np.array(DegreesdistributionTimePoints_t[r])[ :, i, :]).melt()
        Cpd_c = pd.DataFrame(normalize(np.array(RadiusdistributionTimePoints_t[r])[ :, i, :], norm='l2', axis=1)).melt()
        sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color=colors_m[r], linewidth=2.5)
    plt.legend(['P0', 'P1', 'P5', '3W', 'P3', 'P7'])
    plt.yticks(size='x-large')
    plt.ylabel('radius distribution', size='x-large')
    # plt.xlabel('cortical depth (um)', size='x-large')
    plt.xticks(size='x-large')
    plt.xticks((1e5*(np.arange(0, np.max(binsR), np.max(binsR) / 10))).astype(int), (1e5*(np.arange(0, np.max(binsR), np.max(binsR) / 10))).astype(int), size='x-large', rotation=20)
    plt.title(reg_name[i])




plt.figure()
sns.lineplot(x="timepoint", y="modularity", data=SBMTimePoints, color='turquoise',err_style='bars')
plt.twinx()
sns.lineplot(x="timepoint", y="Nb_modules",
             data=SBMTimePoints, color='magenta',err_style='bars')
plt.legend(['modularity', 'Nb_modules'])
#
#
# plt.figure()
# sns.lineplot(x="timepoint", y="lengthtot",
#              data=featuresTimePoints, color='turquoise',err_style='bars')
# plt.twinx()
# sns.lineplot(x="timepoint", y="bp",
#              data=featuresTimePoints, color='magenta',err_style='bars')
# plt.legend(['lengthtot', 'bp'])

plt.figure()
SBMTimePoints['modularity/Nb_modules'] = SBMTimePoints['modularity'] / SBMTimePoints['Nb_modules']
sns.lineplot(x="timepoint", y='modularity/Nb_modules',err_style='bars',hue='region',data=SBMTimePoints)

plt.figure()
featuresTimePoints['bp/length'] = featuresTimePoints['bp'] / featuresTimePoints['lengthtot']*1e-6
sns.lineplot(x="timepoint", y='bp/length',err_style='bars',hue='region',
             data=featuresTimePoints)


plt.figure()
sns.lineplot(x="timepoint", y='bp',err_style='bars',hue='region',
             data=featuresTimePoints)


plt.figure()
sns.lineplot(x="timepoint", y='length',err_style='bars',hue='region',
             data=featuresTimePoints)
#
# plt.figure()
# tps=np.unique(RadiusdistributionTimePoints[0])
# for i in tps:
#     sns.kdeplot(RadiusdistributionTimePoints[1][RadiusdistributionTimePoints[0]==i], bw=0.192)
# plt.title('radius')
# plt.legend(['P0', 'P1', 'P5', '4W'])

plt.figure()
tps=np.unique(DegreesdistributionTimePoints[0])
for i in tps:
    data=DegreesdistributionTimePoints[1][DegreesdistributionTimePoints[0]==i]
    sns.kdeplot(data)

plt.title('degrees')
plt.legend(['P0', 'P1', 'P5', '4W'])

plt.figure()
tps=np.unique(VessellengthDistribution[0])
for i in tps:
    sns.distplot(VessellengthDistribution[1][VessellengthDistribution[0]==i])#, bw=0.192)
plt.title('length')
plt.legend(['P1', 'P3', 'P5','P6','P7', 'P14',  '4W'])


##% vessels length radii degree distribution

for i, g in enumerate(controls):
    print(g)
    # graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
    graph = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')
    coordinates = graph.edge_geometry_property('coordinates_atlas')
    indices = graph.edge_property('edge_geometry_indices')
    degrees=graph.vertex_degrees()
    radii=graph.edge_radii()
    length=[]
    for i, ind in enumerate(indices):
        L = 0
        diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
        length.append(np.sum(np.linalg.norm(diff, axis=1))* 0.000025)

    length=np.array(length)
    print(length.shape)

    plt.figure()
    binarray=np.arange(0, 300, 10)*1e-6
    plt.hist(length, bins=binarray)
    plt.title(g+' length distribution')
    ticks_pos=plt.xticks()[0]
    plt.xticks(np.arange(0, 300*1e-6, 300*1e-6/10), np.arange(0, np.max(binarray*1e6), np.max(binarray*1e6) / 10).astype(int), size='x-large', rotation=20)
    plt.xlabel('length um')

    plt.figure()
    # length=np.array(length)
    plt.hist(radii, bins=[2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    plt.title(g+' radii distribution')


    plt.figure()
    # length=np.array(length)
    plt.hist(degrees,bins=[0,1,2,3,4,5,6,7,8,9,10])
    plt.title(g+' degree distribution')


## LOOPS

def extractCycles(graph, basis, NbLoops, cyclesEPVect,CyclePos, i, Nb_loops_only=False):
    # print('extractFeaturesSubgraphs : ', i)
    NbLoops=0
    cyclesEP = np.zeros((graph.n_edges, len(basis)))
    n = 1

    evect_tot = np.zeros(graph.n_edges)
    # for l in layers:
    #     order=l
    #     level=10
    #     label = g_i.vertex_annotation();
    #     label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    #     vf = label_leveled == order;
    #     g_j=g_i.sub_graph(vertex_filter=vf)
    #     print(level, order, ano.find_name(order, key='order'))
    # cycLen2=[]
    # lengths = g_j.edge_property('length')
    # print(g_j)
    g_j = graph
    print(g_j)
    for i, b in enumerate(basis):
        # if i >= 3:
        res = gtt.subgraph_isomorphism(b.base, g_j.base, induced=True)
        res=checkUnicityOfElement(res)
        evect = np.zeros(g_j.n_edges)
        NbLoops = NbLoops + len(res)
        # print(i, len(res), NbLoops)
        # cycLen=[]
        if not Nb_loops_only:
            for j, r in enumerate(res):
                coordinates = g_j.vertex_property('coordinates_atlas')
                CyclePos.append(np.mean(coordinates[r], axis=0).tolist())
                t = np.zeros(g_j.n_vertices)
                t[r] = 1
                tem = from_v_prop2_eprop(g_j, t)
                # cycLen.append(np.sum(lengths[tem]))
                # print(np.sum(t),np.sum(tem))
                if np.sum(t) == np.sum(tem):
                    evect[tem] = j
                    evect_tot[tem] = 1
                else:
                    print('there is a pb !')
                # n=n+1
            # cycLen2.append(cycLen)
            # cyclesLength30.append(cycLen2)
            # print(len(cyclesLength30))

            # print(evect.shape)
            # print(cyclesEP.shape)
        if not Nb_loops_only:
            cyclesEP[:, i] = (evect)


    if not Nb_loops_only:
        cyclesEPVect.append(cyclesEP)

    if Nb_loops_only:
        return NbLoops
    else:
        return cyclesEPVect, evect_tot, NbLoops,CyclePos

import ClearMap.Analysis.Graphs.GraphGt_old as ggto
def CreateSimpleBasis(n, m):
    simpla_basis=[]
    cycles = np.arange(n,m)#[3,4,5,6,7,8]#,9
    for i in cycles:
        print('cycle', i)
        g = ggto.Graph(n_vertices=i, directed=False)
        edges_all = np.zeros((0, 2), dtype=int)
        for j in range(i):
            if j+1<i:
                edge = (j, j+1)
            else:
                edge = (j, 0)
            edges_all = np.vstack((edges_all, edge))

        # print(edges_all)
        g.add_edge(edges_all)
        simpla_basis.append(g)

    return simpla_basis



def checkUnicityOfElement(liste):
    result=[]
    for i, r in enumerate(liste):
        if i == 0:
            result.append(r.a)
        else:
            bool=False
            for j, elem in enumerate(result):
                if set(elem) == set(r.a):
                    bool=True
                    break
            if not bool:
                result.append(r.a)
                # print(len(result))
    return result


# controlsEmbeddedGraphs=[]
# mutantsEmbeddedGraphs = []
# mutantAggloCycles=[]
# controlAggloCycles=[]
# mutantsNbloops=[]
# controlsNbloops=[]
# controlsCyclesperAggregate=[]
# mutantsCyclesperAggregate=[]


# cluster_max_size=100000#5000
# cluster_min_size=200#100
simpla_basis = CreateSimpleBasis(3, 7)
workdirs=[work_dirP0, work_dirP1, work_dirP5, work_dirAdult]
controlslist=[controlsP0, controlsP1, controlsP5, controlsAdult]


workdirs=[work_dirP3, work_dirP7]
controlslist=[controlsP3, controlsP7]


workdirs=[work_dirP3, work_dirP7]
controlslist=[controlsP3, controlsP7]

get_reg=1
regions=[(315, 6)]
reg_name=['Iscortex']
for nc in range(3,7):
    Positions=[]
    Loops=[]
    EP=[]
    simpla_basis = CreateSimpleBasis(nc, nc+1)
    for j, controls in enumerate(controlslist):
        controls=controlslist[j]
        print(j,workdirs[j],controls)
        Positions_tp=[]
        Loops_tp=[]
        EP_tp=[]
        # regions=[(0,0),(54,9), (127, 7), (13,7)]
        # annotation_file=anot_f[j]
        # ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
        #                extra_label = None, annotation_file = annotation_file)


        if j==3:
            print('adult brain')
            st='adult'
            # regions=[(6,6), (572, 5), (455,6)]
            # regions=[(6,6)]
            # ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
            #                extra_label = None, annotation_file = os.path.join(atlas_path, 'annotation_25_full.nrrd'))

        else:
            st='dev'
            print('dev brain')
            # ano.initialize(label_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',
            #                extra_label = None, annotation_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotationADMBA_thresholded.tif')

            # regions=[(449,9), (567,9),(259,9)]
            # regions=[(449,9)]

        for i, g in enumerate(controls):
            print(g)
            NbLoops=0
            CyclePos=[]
            cyclesEPVect = []
            work_dir=workdirs[j]
            try:
                # G = ggt.load(work_dir + '/' + '/' +'3_graph_uncrusted.gt')
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
            except:
                try:
                    G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
                except:
                    G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')


            label=G.vertex_annotation()




            if get_reg==0:
                order, level=(6,6)
                vertex_filter = np.zeros(G.n_vertices)
                print(level, order, ano.find(6, key='order')['name'])
                label = G.vertex_annotation();
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1;
                G = G.sub_graph(vertex_filter=vertex_filter)

                label=G.vertex_annotation();
                regions=[(u_l, ano.find(u_l, key='order')['level']) for u_l in np.unique(label)]
                reg_name=[ano.find(u_l, key='order')['name'] for u_l in np.unique(label)]
                get_reg=1


            for k,region in enumerate(regions):
                if TP[j]==21 or TP[j]==12 or TP[j]==9:
                    print('labelled by ID')
                    graph=extract_AnnotatedRegion(G, region, state='21')
                    print(np.unique(label))
                else:
                    graph=extract_AnnotatedRegion(G, region)

                coordinates = graph.edge_geometry_property('coordinates_atlas')
                cyclesEPVect, evect_tot, NbLoops,CyclePos = extractCycles(graph, simpla_basis, NbLoops, cyclesEPVect,CyclePos, k)

            Positions_tp.append(CyclePos)
            Loops_tp.append(NbLoops)
            EP_tp.append(cyclesEPVect)

        Positions.append(Positions_tp)
        Loops.append(Loops_tp)
        EP.append(EP_tp)

    np.save('/data_2to/dev/adult/EP_3_7'+reg_name[0]+'_'+str(nc)+'.npy',EP)
    np.save('/data_2to/dev/adult/Loops_3_7'+reg_name[0]+'_'+str(nc)+'.npy',Loops)
    np.save('/data_2to/dev/adult/Positions_3_7'+reg_name[0]+'_'+str(nc)+'.npy',Positions)





TP=['P0', 'P1', 'P5', 'W4','P3', 'P7']

# TP=['P3', 'P7']
reg_name=['Isocortex']
import pandas as pd
import seaborn as sns
loopsdatas=pd.DataFrame()
tot_loops=np.zeros((len(controlslist), 5))

for j, controls in enumerate(controlslist):
    L=0
    for i,control in enumerate(controls):
        for nc in range(3,7):
            if controls in [controlsP3,controlsP7]:
                print(controls)
                Loops=np.load('/data_2to/dev/new_anot/Loops_3_7'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
                L=L+Loops[j-4][i]
            else:
                Loops=np.load('/data_2to/dev/new_anot/Loops_3_7'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
                L=L+Loops[j][i]
        tot_loops[j][i]=L

for nc in range(3,7):
    # Loops=np.load('/data_2to/dev/Loops_3_7'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
    for j, controls in enumerate(controlslist):
        if controls in [controlsP3,controlsP7]:
            print(controls)
            Loops=np.load('/data_2to/dev/Loops_3_7'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
            for i,control in enumerate(controls):
                loopsdatas=loopsdatas.append({'timepoint': TP[j], 'loops_size':nc, 'nb_loopsN': Loops[j-4][i], 'prop_loops': Loops[j-4][i]/tot_loops[j][i]}, ignore_index=True)#([j, Lmm, bp/Lmm])#

        else:
            Loops=np.load('/data_2to/dev/Loops_'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
            for i,control in enumerate(controls):
                loopsdatas=loopsdatas.append({'timepoint': TP[j], 'loops_size':nc, 'nb_loopsN': Loops[j][i], 'prop_loops': Loops[j][i]/tot_loops[j][i]}, ignore_index=True)#([j, Lmm, bp/Lmm])#

loopsdatas.to_csv('/data_2to/dev/loopsdatas.csv', index=False)

plt.figure()
# loopsdatas['loosp_prop'] = loopsdatas['modularity'] / loopsdatas['Nb_modules']
sns.lineplot(x="timepoint", y='prop_loops',err_style='bars',hue='loops_size',data=loopsdatas)






nc=5
EP=np.load('/data_2to/dev/EP_'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
Loops=np.load('/data_2to/dev/Loops_'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
Positions=np.load('/data_2to/dev/Positions_'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)


timepoint=3#1,2,3
work_dir=workdirs[timepoint]
controls=controlslist[timepoint]


template_shape=(320,528,228)
vox_shape_c=(320,528,228, len(controls))
vox_control=np.zeros(vox_shape_c)
radius=10



for i,control in enumerate(controls):
    print(control)
    # try:
    #     G = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    # except:
    #     G = ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph.gt')

    coordinates=np.array(Positions[timepoint][i])
    v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
    vox_control[:,:,:,i]=v

for i in range(len(controls)):
    # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_control[:, :, :, i])
    io.write(work_dir + '/' + controls[i] + '/' + 'vox_loops_control'+controls[i]+str(radius)+'_'+str(nc)+'.tif', vox_control[:, :, :, i].astype('float32'))

io.write(work_dir + '/' +'vox_loops_control'+str(radius)+'_'+str(nc)+'.tif', vox_control.astype('float32'))
vox_control_avg=np.mean(vox_control[:, :, :, :], axis=3)
io.write(work_dir + '/' +'vox_loops_control_avg_'+str(radius)+'_'+str(nc)+'.tif', vox_control_avg.astype('float32'))


## get loops length
LoopsSizeData = pd.DataFrame()


workdirs=[work_dir_P9, work_dir_P12, work_dir_P21,work_dir_2M, work_dir_3M,work_dir_7M]
controlslist=[controls_P9, controls_P12, controls_P21, controls_2M,controls_3M, controls_7M]

anot_f=['/data_2to/alignement/atlases/new_region_atlases/P10/ano_P10.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P10/ano_P10.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P21/atlasP21.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd']
TP=[9, 12, 21, 60, 90 ,210]


# get_reg=0
for j, controls in enumerate(controlslist):
    controls=controlslist[j]
    print(j,workdirs[j],controls)
    annotation_file=anot_f[j]
    ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                   extra_label = None, annotation_file = annotation_file)
    regions=[(315,6)]
    # regions=[(6,6)]
    if j==2:
        print('adult brain')
        st='adult'
        # regions=[(0,0),(52,9), (122, 7), (13,7)]#[(0,0), (6,6)]#, (572, 5), (455,6)]
        # regions=[(0,0)]
        # ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
        #                extra_label = None, annotation_file = os.path.join(atlas_path, 'annotation_25_full.nrrd'))

    else:
        st='dev'
        print('dev brain')

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


        label=G.vertex_annotation()
        for k,region in enumerate(regions):
            if TP[j]==21 or TP[j]==12 or TP[j]==9:
                print('labelled by ID')
                graph=extract_AnnotatedRegion(G, region, state='21')
                print(np.unique(label))
            else:
                graph=extract_AnnotatedRegion(G, region)


            # loops_length=[]
            for nc in [3,4,5,6]:
                # if controls in [controlsP3,controlsP7]:
                print(controls)
                EP=np.load('/data_2to/dev/new_anot/EP_3_7'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
                Positions=np.load('/data_2to/dev/new_anot/Positions_3_7'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
                Position=Positions[j][i]
                ep=EP[j][i][0]#j-4
                # else:
                #     EP=np.load('/data_2to/dev/new_anot/EP_'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
                #     Positions=np.load('/data_2to/dev/new_anot/Positions_'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
                #     Position=Positions[j][i]
                #     EP=EP[j][i][0]
                coordinates = graph.edge_geometry_property('coordinates_atlas')
                indices = graph.edge_property('edge_geometry_indices')
                vessels_length=[]
                vessels_center=[]
                for k, ind in enumerate(indices):
                    diff = np.diff(coordinates[ind[0]:ind[1]], axis=0)
                    vessels_length.append(np.sum(np.linalg.norm(diff, axis=1))* 0.000025)
                    vessels_center.append(np.mean(coordinates[ind[0]:ind[1]], axis=0))

                vessels_length=np.array(vessels_length)
                vessels_center=np.array(vessels_center)
                loops_indices=np.unique(ep)[1:]#remove 0
                for l in loops_indices:
                    L=np.sum(vessels_length[(np.asarray(ep==l).nonzero()[0])])
                    P=np.mean(vessels_center[(np.asarray(ep==l).nonzero()[0])], axis=0)
                    # loops_length.append(L)
                    # LoopsSizeData.append()
                    LoopsSizeData=LoopsSizeData.append({'timepoint': TP[j], 'lengthloop':L, 'bp': nc, 'position':P,'brainID': g},ignore_index=True)#([j, Lmm, bp/Lmm])#


LoopsSizeData.to_csv('/data_2to/dev/adult/LoopsSizeData_new_anot.csv', index=False)


LoopsSizeData=pd.read_csv('/data_2to/dev/LoopsSizeData.csv')
## plot loops examples
import random
import re
import ClearMap.Visualization.Plot3d as p3d
from ast import literal_eval
TP=['P0', 'P1', 'P5', 'W4','P3', 'P7']

tp=2#2
loops_bp=3
l_max=1e-4
l_min=2e-4

work_dir=workdirs[tp]
TP=[9,12, 21]
tp=TP[tp]
loops_pool=LoopsSizeData.loc[lambda df: df['timepoint'] == tp, :]
# loops_pool=loops_pool.loc[lambda df: df['bp'] == loops_bp, :]
# loops_pool=loops_pool.loc[lambda df: df['lengthloop'] <= l_max, :]
loops_pool=loops_pool.loc[lambda df: df['lengthloop'] >= l_min, :]

ind=random.choice(loops_pool['position'].index)
g=LoopsSizeData.iloc[ind]['brainID']
pos=LoopsSizeData.iloc[ind]['position']
pos=re.split(' ',pos[1:-1])
pos=np.delete( np.array(pos),np.asarray(np.array(pos)=='').nonzero()[0]).astype(float)
ll=LoopsSizeData.iloc[ind]['lengthloop']
print(g, ind, pos,ll)

try:
    G = ggt.load(work_dir + '/' + str(g) + '/' + 'data_graph_correcteduniverse.gt')
except:
    G = ggt.load(work_dir + '/' + str(g) + '/' + str(g)+'_graph_correcteduniverse.gt')


x=pos[0]
y=pos[1]
z=pos[2]
rad=2.5
mins = np.array([x, y, z]) - np.array([rad, rad, rad])
maxs = np.array([x, y, z]) + np.array([rad, rad, rad])
close_edges_centers=extractSubGraph(G.vertex_property('coordinates_atlas'), mins, maxs)

g2plot=G.sub_graph(vertex_filter=close_edges_centers)
p3d.plot_graph_mesh(g2plot)



TP=['P9', 'P12', 'P21']#, '3M','7M']
TP=['2M', '3M', '7M']#, '3M','7M']
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# LoopsSizeData=pd.read_csv('/data_2to/dev/LoopsSizeData.csv')
LoopsSizeData_ordered = LoopsSizeData.sort_values(['timepoint']).reset_index(drop=True)

plt.figure()
# %pip install seaborn --upgrade
# sns.displot(data=LoopsSizeData, x="lengthloop", hue="timepoint", kind="kde",hue_norm=(0,1))
# sns.histplot(data=LoopsSizeData, x="lengthloop", hue="timepoint", fill=False, common_norm=False)
# sns.lineplot(x="lengthloop",err_style='bars',hue='timepoint',data=LoopsSizeData)
plt.figure()
sns.histplot(data=LoopsSizeData, x="lengthloop", hue="timepoint", common_norm=False,stat="percent",element='poly', fill=False, bins=50)


sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
# Initialize the FacetGrid object
pal = sns.cubehelix_palette(3, rot=-.25, light=.7)
g = sns.FacetGrid(LoopsSizeData_ordered, row="timepoint", hue="timepoint", aspect=15, height=.5, palette=pal)
# Draw the densities in a few steps
g.map(sns.histplot, "lengthloop", label="timepoint", common_norm=False,stat="percent",element='poly', fill=True, bins=50, clip_on=False,alpha=1, linewidth=1.5)
g.map(sns.histplot, "lengthloop",  label="timepoint", common_norm=False,stat="percent",element='poly', fill=False, bins=50, clip_on=False,alpha=1, linewidth=1.5, color="w")
# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "timepoint")
# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)
# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)


LoopsSizeData_ordered_inv = LoopsSizeData_ordered.sort_values(['bp'], ascending=True).reset_index(drop=True)
plt.figure()
axe=sns.displot(data=LoopsSizeData_ordered, x="timepoint", hue="bp",height=6, common_norm=False, stat="proportion",element='bars', fill=True, multiple='fill')
plt.gca().invert_yaxis()

plt.figure()
sns.histplot(data=LoopsSizeData, x="bp", hue="timepoint", common_norm=False, stat="percent",element='poly', bins=[3,4,5,6,7], fill=False)
sns.displot()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import ClearMap.Analysis.Measurements.Voxelization as vox
LoopsSizeData=pd.read_csv('/data_2to/dev/LoopsSizeData.csv')


timepoint=3#2,3

work_dir=workdirs[timepoint]
controls=controlslist[timepoint]



template_shape=(320,528,228)
vox_shape_c=(320,528,228, len(controls))
vox_control=np.zeros(vox_shape_c)
radius=10

for i,control in enumerate(controls):
    print(control)
    work_dir=workdirs[timepoint]
    controls=controlslist[timepoint]
    # try:
    #     G = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    # except:
    #     G = ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph.gt')

    # for nc in [3,4,5,6]:
        # EP=np.load('/data_2to/dev/EP_'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
        # Loops=np.load('/data_2to/dev/Loops_'+reg_name[0]+'_'+str(nc)+'.npy',allow_pickle=True)
    loopslength = np.array(LoopsSizeData.loc[lambda df: df['brainID'] == control, :]['lengthloop'])
    df= LoopsSizeData.loc[lambda df: df['brainID'] == control, :]['position'].to_numpy()
    coordinates=[re.split(' ',df[k][1:-1]) for k in range(df.shape[0])]
    # coordinates=np.array(coordinates)
    for k in range(len(coordinates)):
        if '' in coordinates[k]:
            coordinates[k]=np.delete( np.array(coordinates[k]),np.asarray(np.array(coordinates[k])=='').nonzero()[0]).astype(float)
            # coordinates[k]=temp
        else:
            coordinates[k]=np.array(coordinates[k]).astype(float)
    coordinates=np.array(coordinates).astype(float)
    # loopslength=loopslength[:, np.newaxis]
    print(nc, loopslength.shape, coordinates.shape)
    loopslength=loopslength*1e4

    v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=loopslength, radius=(radius, radius, radius),method='sphere');
    w = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
    res=v.array.astype(float)/w.array.astype(float)
    # res[np.isnan(res)] = 0
    vox_control[:,:,:,i]=res

for i in range(len(controls)):
    # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_control[:, :, :, i])
    io.write(work_dir + '/' + controls[i] + '/' + 'vox_loops_control'+controls[i]+str(radius)+'_'+str(timepoint)+'.tif', vox_control[:, :, :, i].astype('float32'))

io.write(work_dir + '/' +'vox_loops_length_control'+str(radius)+'_'+str(timepoint)+'.tif', vox_control.astype('float32'))
vox_control_avg=np.nanmean(vox_control[:, :, :, :], axis=3)
io.write(work_dir + '/' +'vox_loops_length_control_avg_'+str(radius)+'_'+str(timepoint)+'.tif', vox_control_avg.astype('float32'))



###  BP and ORI profile:
work_dirP0='/data_2to/p0/new'
controlsP0=['0a', '0b', '0c', '0d']#['2', '3']

work_dirP5='/data_2to/p5'
controlsP5=['5a', '5b']#['2', '3']

work_dirP1='/data_2to/p1'
controlsP1=['1a', '1b', '1d']#['2', '3']

work_dirAdult='/data_2to/earlyDep_ipsi'
controlsAdult=['4', '7', '10', '15']

TP=[0, 1, 5, 30]
TP_name=['P0', 'P1', 'P5', 'P30']

workdirs=[work_dirP0, work_dirP1, work_dirP5, work_dirAdult]
controlslist=[controlsP0, controlsP1, controlsP5, controlsAdult]
reg_name=['cortex']
limit_angle=40

average=False
mode='bigvessels'#'bigvessels
if mode=='bigvessels':
    suffixe='bv'
elif mode=='arteryvein':
    suffixe='av'


for j, controls in enumerate(controlslist):
    print(j,workdirs[j],controls)
    BP=[]
    ORI=[]
    EP=[]
    BP_DEG1=[]
    PROP_ORI=[]

    work_dir=workdirs[j]
    name=TP_name[j]
    annotation_file=anot_f[j]
    ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                   extra_label = None, annotation_file = annotation_file)
    work_dir=workdirs[j]
    # if j==3:
    #     print('adult brain')
    #     st='adult'
    #     # regions=[(6,6), (572, 5), (455,6)]
    #     regions=[(6,6)]
    #     ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
    #                    extra_label = None, annotation_file = os.path.join(atlas_path, 'annotation_25_full.nrrd'))
    #
    # else:
    #     st='dev'
    #     print('dev brain')
    #     ano.initialize(label_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/region_ids_test_ADMBA.json',
    #                    extra_label = None, annotation_file = '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotationADMBA_thresholded.tif')
    #
    #     # regions=[(449,9), (567,9),(259,9)]
    #     regions=[(449,9)]
    #
    # region=regions[0]

    for i, g in enumerate(controls):
        print(g)
        # prop_ori=[]
        # bp_dist_2_surface = []
        # ep_dist_to_surface=[]
        # ori=[]
        # try:
        #     G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
        # except:
        #     G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')

        try:
            G = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
        except:
            try:
                G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')


        # for k,region in enumerate(regions):
        graph=extract_AnnotatedRegion(G, region, st)
        deg1graph=graph.sub_graph(vertex_filter=graph.vertex_degrees()==1)
        print(graph)
        BP_DEG1.append(deg1graph.vertex_property('distance_to_surface'))
        BP.append(graph.vertex_property('distance_to_surface'))
        EP.append(graph.edge_property('distance_to_surface'))
        try:
            artery=graph.vertex_property('artery')
            vein=graph.vertex_property('vein')
        except:
            print('no artery vertex properties')
            artery=np.logical_and(graph.vertex_radii()>=4.8,graph.vertex_radii()<=8)#4
            vein=graph.vertex_radii()>=8
            graph.add_vertex_property('artery', artery)
            graph.add_vertex_property('vein', vein)
            artery=from_v_prop2_eprop(graph, artery)
            graph.add_edge_property('artery', artery)
            vein=from_v_prop2_eprop(graph, vein)
            graph.add_edge_property('vein', vein)


        label = graph.vertex_annotation();

        angle,graph = GeneralizedRadPlanorientation(graph, g, 4.5, controls, mode=mode, average=average)
        print(angle.shape)

        dist = graph.edge_property('distance_to_surface')

        radiality = angle <= limit_angle  # 40
        planarity = angle >= (90 - limit_angle)  # 60
        neutral = np.logical_not(np.logical_or(radiality, planarity))

        ori_prop = np.concatenate((np.expand_dims(dist, axis=1), np.concatenate((np.expand_dims(radiality, axis=1), np.concatenate(
            (np.expand_dims(neutral, axis=1), np.expand_dims(planarity, axis=1)), axis=1)), axis=1)), axis=1)
        PROP_ORI.append(ori_prop)
        ORI.append(radiality)

    np.save(work_dir + '/' + 'BP_DEG1_' + name +'.npy', BP_DEG1)
    np.save(work_dir + '/' + 'PROP_ORI_' + name  + '.npy', PROP_ORI)
    np.save(work_dir + '/' + 'BP_' + name +'.npy', BP)
    np.save(work_dir + '/' + 'ORI_' + name +'.npy', ORI)
    np.save(work_dir + '/' + 'EP_' + name +'.npy', EP)


thresh=2
bin = 10
bin2 = 10
normed = False
from sklearn.preprocessing import normalize

features=[]
features_avg=[]
for j, controls in enumerate(controlslist):

    features_brains=[]

    print(j,workdirs[j],controls)
    work_dir=workdirs[j]
    name=TP_name[j]
    bpdeg1_dist=np.load(work_dir + '/' + 'BP_DEG1_' + name  + '.npy',allow_pickle=True)
    # prop_ori=np.load(work_dir + '/' + 'PROP_ORI_' + name  + '.npy',allow_pickle=True)
    # bp_dist_2_surface=np.load(work_dir + '/' + 'BP_' + name +'.npy',allow_pickle=True)
    # vess_rad_=np.load(work_dir + '/' + 'ORI_' + name +'.npy',allow_pickle=True)
    # ve_ep_dist_2_surface=np.load(work_dir + '/' + 'EP_' + name +'.npy',allow_pickle=True)

    for i, g in enumerate(controls):

        bpdeg1=np.array(bpdeg1_dist[i])
        # ori=np.array(vess_rad_[i])
        # bp_dist=np.array(bp_dist_2_surface[i])##
        # ve_ep=np.array(ve_ep_dist_2_surface[i])##
        # radial_depth=ve_ep[ori>0.6]

        hist_bp_bpdeg1, bins_bp_bpdeg1 = np.histogram(bpdeg1, bins=bin, normed=normed)
        # hist_ve_ep, bins_ve_ep = np.histogram(ve_ep, bins=bin, normed=normed)
        # hist_bp_dist, bins_bp_dist = np.histogram(bp_dist[bp_dist>thresh], bins=bin, normed=normed)
        # #, np.sum(np.mean(H, axis=1)))
        # hist_ori, bins_ori_dist = np.histogram(radial_depth, bins=bin, normed=normed)
        #
        # dist=prop_ori[i][:,0]
        # ori_rad=prop_ori[i][:,1]
        # ori_neutral = prop_ori[i][:,2]
        # ori_plan = prop_ori[i][:,3]
        # histrad, bins = np.histogram(dist[ori_rad.astype(bool)], bins=10)
        # histneut, bins = np.histogram(dist[ori_neutral.astype(bool)], bins=bins)
        # histplan, bins = np.histogram(dist[ori_plan.astype(bool)], bins=bins)
        # R = histrad / (histrad + histneut + histplan)
        # N = histneut / (histrad + histneut + histplan)
        # P = histplan / (histrad + histneut + histplan)

        features_brains.append([hist_bp_bpdeg1])
        # features_brains.append([hist_ve_ep, hist_bp_dist, hist_ori, P, N,R])  # len(shortest_paths_control)
    features_brains=np.array(features_brains)

    features_brain_avg=np.mean(features_brains, axis=0)

    features.append(features_brains)
    features_avg.append(features_brain_avg)

plt.figure()
sns.set_style(style='white')
feat=['BP','PROP_ORI_PLAN', 'PROP_ORI_RAD', 'ORI_RAD']#['ART EP', 'VE EP', 'BP', 'ORI']#'SP len', 'SP step',

for r , feature in enumerate(features):
    ori = pd.DataFrame(normalize(feature[:, 2], norm='l2', axis=1)).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=ori, linewidth=2.5)
plt.title('Orientation profile', size='x-large')
plt.legend(TP_name)
plt.xticks(size='x-large')
plt.xticks(np.arange(0, 10), 25*np.arange(0, np.max(bins), np.max(bins) / 10).astype(int), size='x-large', rotation=20)
plt.xlabel('cortical depth (um)', size='x-large')
plt.yticks(size='x-large')

plt.figure()
sns.set_style(style='white')
for r , feature in enumerate(features):
    bp = pd.DataFrame(normalize(feature[:, 0], norm='l2', axis=1)).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=bp, linewidth=2.5)
plt.title('BP profile', size='x-large')
plt.legend(TP_name)
plt.xticks(size='x-large')
plt.xticks(np.arange(0, 10), 25*np.arange(0, np.max(bins), np.max(bins) / 10).astype(int), size='x-large', rotation=20)
plt.xlabel('cortical depth (um)', size='x-large')
plt.yticks(size='x-large')

plt.figure()
sns.set_style(style='white')
for r , feature in enumerate(features):
    # bp = pd.DataFrame(normalize(feature[:, 0], norm='l2', axis=1)).melt()
    bp = pd.DataFrame(feature[:, 0]).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=bp, linewidth=2.5)
plt.title('BP DEG1 profile', size='x-large')
plt.legend(TP_name)
plt.xticks(size='x-large')
plt.xticks(np.arange(0, 10), 25*np.arange(0, np.max(bins), np.max(bins) / 10).astype(int), size='x-large', rotation=20)
plt.xlabel('cortical depth (um)', size='x-large')
plt.yticks(size='x-large')