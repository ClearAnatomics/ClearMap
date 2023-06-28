import numpy as np
import pandas as pd

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

reg_ids=[500, 329, 247, 669]
reg_name=['Somatomotor areas',
          'Primary somatosensory area, barrel field','Auditory areas',
          'Visual areas']
# reg_ids=[315, 549, 1097, 512, 343, 477]
# reg_name=['Isocortex', 'Thalamus', 'Hypothalamus', 'Cerebellum', 'Brain stem', 'Striatum']
regions=[]
for id in reg_ids:
    regions.append((ano.find(id)['id'],ano.find(id)['level']))

get_reg=1
Tablerad=[]
Tableplan=[]
timepoints_temp=[]
reg_temp=[]
limit_angle=40
compute_prop_ori=True

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

            try:
                r, p, n, l = getLocalNormaleOrienttaion(graph, graph,[id],local_normal=True, verbose=True, calc_art=True) # ,, , calc_art=True)

                if not compute_prop_ori:
                    r = r[~np.isnan(r)]
                    p = p[~np.isnan(r)]
                elif compute_prop_ori:
                    rt = r[~np.isnan(r)]
                    p = p[~np.isnan(r)]
                    dist = graph.edge_property('distance_to_surface')[~np.isnan(r)]

                    # radiality = (r / (r + p)) > 0.5
                    # planarity = (p / (r + p)) > 0.6
                    # neutral = np.logical_not(np.logical_or(radiality, planarity))
                    angle = np.array([math.acos(rt[i]) for i in range(rt.shape[0])]) * 180 / pi

                    radiality = angle <  limit_angle#40
                    planarity = angle >  90-limit_angle#60
                    neutral = np.logical_not(np.logical_or(radiality, planarity))

            except:
                print('indexError')

            # vertex_dist=graph.vertex_property('distance_to_surface')
            rad=dist[radiality]
            plan=dist[planarity]
            # bp_dist=vertex_dist[np.asarray(vertex_filter==1).nonzero()[0]]

            histbprad, bins = np.histogram(rad, bins=10)#, normed=normed)
            histbpplan, bins = np.histogram(plan, bins=bins)#, normed=normed)


            Tablerad.append(np.concatenate((histbprad, np.array([TP[j],reg_name[k]])), axis=0))
            Tableplan.append(np.concatenate((histbpplan, np.array([TP[j],reg_name[k]])), axis=0))


feat_cols = [ 'feat'+str(i) for i in range(10) ]
feat_cols.append('timepoint')
feat_cols.append('region')
dfrad = pd.DataFrame(Tablerad,columns=feat_cols)
dfplan = pd.DataFrame(Tableplan,columns=feat_cols)
# df['timepoints']=timepoints_temp
# df['region']=reg_temp
dfrad.to_csv('/data_2to/dev/D1new_annot_ORIrad_cortex_profile_normed.csv', index=False)
dfplan.to_csv('/data_2to/dev/D1new_annot_ORIplan_cortex_profile_normed.csv', index=False)

dfallvass=pd.read_csv('/data_2to/dev/D1new_annot_BP_cortex_profile_normed.csv')

dfrad=pd.read_csv('/data_2to/dev/D1new_annot_ORIrad_cortex_profile_normed.csv')
dfplan=pd.read_csv('/data_2to/dev/D1new_annot_ORIplan_cortex_profile_normed.csv')



TP_temp=[1,3,5,7,9, 12,14,21,30,60,90,210]
pal = sns.cubehelix_palette(len(TP_temp), rot=-.25, light=.7)
pal=sns.color_palette("icefire", len(TP_temp))

# regions=[ano.find(r, key='order')['id'] for r in  [13, 38, 52, 101, 122,137,186]]

data2pltrad=dfrad.iloc[:, :12]#df.iloc[:, [0,1,2,3,4,5,6,7,8,9,-2,-1]]#[:, 10:]#[:, [0,1,2,3,4,5,6,7,8,9,-2,-1]]
data2pltplan=dfplan.iloc[:, :12]
# data2plt=data2plt[data2plt['region'].isin(regions)]
normed=False
from sklearn.preprocessing import normalize

for col in dfrad.columns[:-1]:
    new_col=dfrad[col].values.astype(int)
    dfrad[col]=new_col

for col in dfplan.columns[:-1]:
    new_col=dfplan[col].values.astype(int)
    dfplan[col]=new_col


for col in dfallvass.columns[:-1]:
    new_col=dfallvass[col].values.astype(int)
    dfallvass[col]=new_col



for region in regions:

    region_name=ano.find(region[0])['name']
    print(region_name)
    datarad=dfrad[dfrad['region']==region_name]
    datarad['sum']=datarad.iloc[:,:-3].sum(1)

    dataplan=dfplan[dfplan['region']==region_name]
    dataplan['sum']=dataplan.iloc[:,:-3].sum(1)

    datavess=dfallvass[dfallvass['region']==region_name]
    datavess['sum']=datavess.iloc[:,:-3].sum(1)*1.5#roughly number of edges

    drp=pd.DataFrame()
    drp['tp']=datarad['timepoint']
    drp['rad']=datarad['sum']
    drp['plan']=dataplan['sum']
    drp['all']=datavess['sum']
    drp['neutral']=datavess['sum']-dataplan['sum']-datarad['sum']
    drp['radnorm']=drp['rad']/datavess['sum']#(drp['plan']+drp['rad'])
    drp['plannorm']=drp['plan']/datavess['sum']#(drp['plan']+drp['rad'])
    drp['ones']=datavess['sum']/datavess['sum']
    drp['planstack']=drp['plan']+drp['rad']

    plt.figure()
    sns.barplot(x='tp', y='all', data=drp, color='beige')
    sns.barplot(x='tp', y='planstack', data=drp, color='forestgreen')
    sns.barplot(x='tp', y='rad', data=drp, color='indianred')

    # plt.figure()
    # sns.pointplot(x="timepoint", y="sum", err_style='bars', data=datarad, color='indianred', linewidth=2.5)
    # sns.pointplot(x="timepoint", y="sum", err_style='bars', data=dataplan, color='forestgreen', linewidth=2.5)

    plt.title(region_name)



for region in regions:
    plt.figure()
    region_name=ano.find(region[0])['name']
    print(region_name)
    dfrad['diff']=dfrad.iloc[:,:-2].max(1)-dfrad.iloc[:,:-2].min(1)
    datarad=dfrad[dfrad['region']==region_name]

    dataplan=dfplan[dfplan['region']==region_name]
    dataplan=dataplan[['feat3','feat1',  'timepoint']]
    dataplan['feat']=dataplan['feat3']-dataplan['feat1']

    sns.pointplot(x="timepoint", y="feat1", err_style='bars', data=datarad, color='indianred', linewidth=2.5)
    sns.pointplot(x="timepoint", y="feat", err_style='bars', data=dataplan, color='forestgreen', linewidth=2.5)
    plt.title(region_name)


for region in regions:
    plt.figure()
    region_name=ano.find(region[0])['name']
    print(region_name)
    dfrad['diff']=dfrad.iloc[:,:-2].max(1)-dfrad.iloc[:,:-2].min(1)
    dfplan['diff']=dfplan[['feat3', 'feat4']].max(1)-dfplan.iloc[:,:5].min(1)
    datarad=dfrad[dfrad['region']==region_name]
    dataplan=dfplan[dfplan['region']==region_name]
    sns.lineplot(x="timepoint", y="diff",  data=datarad,linewidth=2.5)
    sns.lineplot(x="timepoint", y="diff",  data=dataplan,linewidth=2.5)
    plt.title(region_name)

for region in regions:
    plt.figure()
    region_name=ano.find(region[0])['name']
    print(region_name)
    dfrad['diff']=dfrad.iloc[:,:-2].max(1)-dfrad.iloc[:,:-2].min(1)
    dfplan['diff']=dfplan[['feat3', 'feat4']].max(1)-dfplan.iloc[:,:5].min(1)
    datarad=dfrad[dfrad['region']==region_name]
    dataplan=dfplan[dfplan['region']==region_name]
    dataplanallvess=dfallvass[dfallvass['region']==region_name]

    D=[]
    for i, tp in enumerate(TP_temp):
        drad=datarad[datarad['timepoint']==tp].iloc[:,:-2]
        radCpd_c = pd.DataFrame(drad).melt()

        dplan=dataplan[dataplan['timepoint']==tp].iloc[:,:-2]
        planCpd_c = pd.DataFrame(dplan).melt()

        dvess=dataplanallvess[dataplanallvess['timepoint']==tp].iloc[:,:-2]
        vessCpd_c = pd.DataFrame(dvess).melt()


        # Cpd_c=pd.DataFrame()
        Cpd_c['value']=radCpd_c['value']/vessCpd_c['value']
        Cpd_c['variable']=vessCpd_c['variable']
        # Cpd_c['value']=radCpd_c['value']/(radCpd_c['value']+planCpd_c['value'])
        # Cpd_c['variable']=radCpd_c['variable']
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

    #     meanCpdf=Cpd_c.groupby([Cpd_c.index, 'variable'])['value'].first().unstack().mean()
    #     meanCpdf=meanCpdf.values
    #     diff=meanCpdf.max()-meanCpdf.min()
    #     D.append([diff, tp])
    # # D_f=pd.DataFrame(D.transpose()).melt()
    # D=np.asarray(D)
    # sns.lineplot(x=D[:,1], y=D[:,0], err_style='bars', linewidth=2.5)


    plt.title(region_name)
    plt.legend(TP_temp)



for region in regions:
    plt.figure()
    region_name=ano.find(region[0])['name']
    print(region_name)
    dfrad['diff']=dfrad.iloc[:,:-2].max(1)-dfrad.iloc[:,:-2].min(1)
    datarad=dfrad[dfrad['region']==region_name]


    for i, tp in enumerate(TP_temp):
        drad=datarad[datarad['timepoint']==tp].iloc[:,:-3]
        drad['diff']=datarad.iloc[:,:-3].max(1)-datarad.iloc[:,:-3].min(1)
        radCpd_c = pd.DataFrame(drad).melt()



        print(tp, radCpd_c.shape)
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
        sns.lineplot(x="variable", y="value", err_style='bars', data=radCpd_c, color=col, linewidth=2.5)

    plt.title(region_name)
    plt.legend(TP_temp)


for region in regions:
    plt.figure()
    region_name=ano.find(region[0])['name']
    print(region_name)
    dataplan=dfplan[dfplan['region']==region_name]

    for i, tp in enumerate(TP_temp):


        dplan=dataplan[dataplan['timepoint']==tp].iloc[:,:-2]
        planCpd_c = pd.DataFrame(dplan).melt()

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
        sns.lineplot(x="variable", y="value", err_style='bars', data=planCpd_c, color=col, linewidth=2.5)


    plt.title(region_name)
    plt.legend(TP_temp)




[(500, 7), (329, 9), (247, 7), (669, 7)]
work_dir=work_dirAdult
control=controlsAdult[0]
# ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
#                extra_label=None,annotation_file = annotation_file)#extra_label=None,

ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
               annotation_file = annotation_file)

try:
    # G = ggt.load(work_dir + '/' + '/' +'3_graph_uncrusted.gt')
    graph = ggt.load(work_dir + '/' + control + '/' + str(control)+ '_graph_correcteduniverse_smoothed.gt')
except:
    try:
        graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
    except:
        graph = ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph_correcteduniverse.gt')

region=(329, 9)
id=region[0]

graphreg=extract_AnnotatedRegion(graph, region, state='dev')
r, p, n, l = getLocalNormaleOrienttaion(graphreg, graphreg,[id],local_normal=True, verbose=True, calc_art=True) # ,, , calc_art=True)
# rt = r[~np.isnan(r)]
# p = p[~np.isnan(r)]

angle = np.array([math.acos(r[i]) for i in range(r.shape[0])]) * 180 / pi
rad=angle<=limit_angle

graphreg.add_edge_property('rad', rad)
# brad=np.logical_and(graph.edge_property('radii')>6, graph.edge_property('radii')<=8)
# graph.add_edge_property('brad', brad)
gs = graphreg.sub_slice((slice(1,5000), slice(4000,4200), slice(1,5000)));
# edge_artery_label = gs.edge_property('brad')
# edge_filter=edge_artery_label
# gsrt = gs.sub_graph(edge_filter=edge_filter)
gsrt=gs
edge_artery_label = gsrt.edge_property('rad')
vertex_colors =np.zeros((gsrt.n_vertices, 4))
vertex_colors[:, -1]=1
vertex_colors[:, 1]=0.8
#
connectivity = gsrt.edge_connectivity();
edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
edge_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
#edge_colors[edge_vein_label>0] = [0.0,0.0,0.8,1.0]
print('plotting...')
p = p3d.plot_graph_mesh(gsrt, edge_colors=edge_colors, n_tube_points=5);

