import os

import numpy as np
import pandas as pd



def extractSubGraph(graph, mins, maxs):
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
    isOver = (graph.vertex_property('coordinates_atlas') > mins).all(axis=1)
    isUnder = (graph.vertex_property('coordinates_atlas') < maxs).all(axis=1)
    return graph.sub_graph(vertex_filter=np.logical_and(isOver, isUnder))


work_dir1M = '/data_2to/DBA2J_new/1m'
controls1M=['21', '22', '24']

work_dir4M = '/data_2to/DBA2J_new/4m'
controls4M = ['1', '2', '4', '7', '8', '9']#['2', '3']

work_dirP10M = '/data_2to/DBA2J_new/10m'
controls10M = ['2', '3', '4', '5', '6']#['2', '3']


reg_name = ['SSp-bfd', 'AUD']
TP = [1 ,4, 10]

reg_name = ['SSp-bfd', 'AUD', 'SSp', 'MO', 'cube']

mins_cube=(66, 265, 44)
maxs_cube=(143, 286, 71)

reg_name=['SSp-bfd', 'AUD', 'SSp', 'MO', 'cube','AUDp', 'AUDv', 'AUDd', 'AUDpo']

workdirs=[ work_dir1M, work_dir4M, work_dirP10M]
controlslist=[ controls1M, controls4M, controls10M]

D1=pd.DataFrame()
for j, controls in enumerate(controlslist):
    # j=j+3
    print(j,workdirs[j],controls)

    for i, g in enumerate(controls):
        print(g)
        work_dir=workdirs[j]
        graph_path = os.path.join(work_dir, f'{g}_graph.gt')
        try:
            G = ggt.load(graph_path)
            # G = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
        except FileNotFoundError as err:
            print(f'could not load graph from "{graph_path}"')
            raise err

        G = G.largest_component()

        bp_reg=[]
        for reg in reg_name:
            if reg=='cube':
                if TP==10:
                    bp_cube=extractSubGraph(G, mins_cube, maxs_cube).n_vertices
                else:
                    bp_cube=extractSubGraph(G, np.array(mins_cube[::-1]), np.array(maxs_cube[::-1])).n_vertices
                print(bp_cube)
            else:
                region=(ano.find(reg, key='acronym')['order'], ano.find(reg, key='acronym')['level'])
                graph=extract_AnnotatedRegion(G, region)
                bp_reg.append(graph.n_vertices)

        D1=D1.append({'timepoint': TP[j],'n_vertices_SSp_bfd':bp_reg[0],'n_vertices_AUD':bp_reg[1],'n_vertices_SSp':bp_reg[2],'n_vertices_MO':bp_reg[3],'cube':bp_cube,
                      'n_vertices_AUDp':bp_reg[4],'n_vertices_AUDv':bp_reg[5],'n_vertices_AUDd':bp_reg[6],'n_vertices_AUDpo':bp_reg[7], 'brainID':g},ignore_index=True)
        # D1=D1.append({'timepoint': TP[j],'n_vertices_SSp_bfd':bp_reg[0],'n_vertices_AUD':bp_reg[1],'n_vertices_SSp':bp_reg[2],'n_vertices_MO':bp_reg[3],'cube':bp_cube, 'brainID':g},ignore_index=True)





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


workdirs=[ work_dir_1M, work_dir_2M, work_dir_3M, work_dir_6M, work_dir_10M]
controlslist=[ controls_1M,controls_2M, controls_3M,controls_6M, controls_10M]
mutantslist=[ mutants_1M,mutants_2M,mutants_3M,mutants_6M, mutants_10M]

TP=[1,2,3,6,10]

D1_otof_c=pd.DataFrame()
for j, controls in enumerate(controlslist):
    # j=j+3
    print(j,workdirs[j],controls)

    for i, g in enumerate(controls):
        print(g)
        work_dir=workdirs[j]
        graph_path = os.path.join(work_dir, f'{g}_graph.gt')
        try:
            G = ggt.load(work_dir + '/' + '/' +'3_graph_uncrusted.gt')
            # G = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
        except:
            try:
                G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')


        G = G.largest_component()

        bp_reg=[]
        for reg in reg_name:
            if reg=='cube':
                bp_cube=extractSubGraph(G, np.array(mins_cube[::-1]), np.array(maxs_cube[::-1])).n_vertices
                print(bp_cube)
            else:
                region=(ano.find(reg, key='acronym')['order'], ano.find(reg, key='acronym')['level'])
                graph=extract_AnnotatedRegion(G, region)
                bp_reg.append(graph.n_vertices)


        D1_otof_c=D1_otof_c.append({'timepoint': TP[j],'n_vertices_SSp_bfd':bp_reg[0],'n_vertices_AUD':bp_reg[1],'n_vertices_SSp':bp_reg[2],'n_vertices_MO':bp_reg[3],'cube':bp_cube,
                      'n_vertices_AUDp':bp_reg[4],'n_vertices_AUDv':bp_reg[5],'n_vertices_AUDd':bp_reg[6],'n_vertices_AUDpo':bp_reg[7], 'brainID':g},ignore_index=True)
        # D1_otof_c=D1_otof_c.append({'timepoint': TP[j],'n_vertices_SSp_bfd':bp_reg[0],'n_vertices_AUD':bp_reg[1],'n_vertices_SSp':bp_reg[2],'n_vertices_MO':bp_reg[3], 'cube':bp_cube, 'brainID':g},ignore_index=True)
        # D1_otof_c=D1_otof_c.append({'timepoint': TP[j],'n_vertices_AUDp':bp_reg[0],'n_vertices_AUDv':bp_reg[1],'n_vertices_AUDd':bp_reg[2],'n_vertices_AUDpo':bp_reg[3], 'brainID':g},ignore_index=True)





D1_otof_m=pd.DataFrame()
for j, controls in enumerate(mutantslist):
    # j=j+3
    print(j,workdirs[j],controls)

    for i, g in enumerate(controls):
        print(g)
        work_dir=workdirs[j]
        graph_path = os.path.join(work_dir, f'{g}_graph.gt')
        try:
            G = ggt.load(work_dir + '/' + '/' +'3_graph_uncrusted.gt')
            # G = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
        except:
            try:
                G = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                G = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')


        G = G.largest_component()

        bp_reg=[]
        for reg in reg_name:
            if reg=='cube':
                bp_cube=extractSubGraph(G, np.array(mins_cube[::-1]), np.array(maxs_cube[::-1])).n_vertices
                print(bp_cube)
            else:
                region=(ano.find(reg, key='acronym')['order'], ano.find(reg, key='acronym')['level'])
                graph=extract_AnnotatedRegion(G, region)
                bp_reg.append(graph.n_vertices)

        D1_otof_m=D1_otof_m.append({'timepoint': TP[j],'n_vertices_SSp_bfd':bp_reg[0],'n_vertices_AUD':bp_reg[1],'n_vertices_SSp':bp_reg[2],'n_vertices_MO':bp_reg[3],'cube':bp_cube,
                            'n_vertices_AUDp':bp_reg[4],'n_vertices_AUDv':bp_reg[5],'n_vertices_AUDd':bp_reg[6],'n_vertices_AUDpo':bp_reg[7], 'brainID':g},ignore_index=True)

        # D1_otof_m=D1_otof_m.append({'timepoint': TP[j],'n_vertices_AUDp':bp_reg[0],'n_vertices_AUDv':bp_reg[1],'n_vertices_AUDd':bp_reg[2],'n_vertices_AUDpo':bp_reg[3], 'brainID':g},ignore_index=True)
        # D1_otof_m=D1_otof_m.append({'timepoint': TP[j],'n_vertices_SSp_bfd':bp_reg[0],'n_vertices_AUD':bp_reg[1],'n_vertices_SSp':bp_reg[2],'n_vertices_MO':bp_reg[3], 'cube':bp_cube, 'brainID':g},ignore_index=True)






# D1_MO=D1.iloc[14:]
# D1_otof_c_MO=D1_otof_c[21:]
# D1_otof_m_MO=D1_otof_m[19:]
#
# D1_1=D1.iloc[:14]
# D1_otof_c_1=D1_otof_c[:21]
# D1_otof_m_1=D1_otof_m[:19]
#
# D1_1['n_vertices_SSp']=D1_MO['n_vertices_AUD'].values
# D1_1['n_vertices_MO']=D1_MO['n_vertices_SSp_bfd'].values
#
#
# D1_otof_c_1['n_vertices_SSp']=D1_otof_c_MO['n_vertices_AUD'].values
# D1_otof_c_1['n_vertices_MO']=D1_otof_c_MO['n_vertices_SSp_bfd'].values
#
# D1_otof_m_1['n_vertices_SSp']=D1_otof_m_MO['n_vertices_AUD'].values
# D1_otof_m_1['n_vertices_MO']=D1_otof_m_MO['n_vertices_SSp_bfd'].values
#
#
# D1_1['n_vertices_SSp']=D1_MO['n_vertices_AUD'].values
# D1_1['n_vertices_MO']=D1_MO['n_vertices_SSp_bfd'].values


D1_1=D1
# D1_1['ratio_AUDps']=D1_1['n_vertices_AUDp']/(D1_1['n_vertices_AUDv']+D1_1['n_vertices_AUDd']+D1_1['n_vertices_AUDpo'])
# D1_1['ratio_AUD_bfd']=D1_1['n_vertices_AUD']/D1_1['n_vertices_SSp_bfd']
# D1_1['ratio_AUD_SSp']=D1_1['n_vertices_AUD']/D1_1['n_vertices_SSp']
# D1_1['ratio_AUD_MO']=D1_1['n_vertices_AUD']/D1_1['n_vertices_MO']
D1_1['ratio_AUDp_cube']=D1_1['n_vertices_AUDp']/D1_1['cube']
D1_1['ratio_AUDs_cube']=(D1_1['n_vertices_AUDv']+D1_1['n_vertices_AUDd']+D1_1['n_vertices_AUDpo'])/D1_1['cube']
# D1_1['ratio_MO_cube']=D1_1['n_vertices_MO']/D1_1['cube']
D1_1.to_csv('/data_2to/DBA2J_new/comparisonAUDreg_1.csv', index=False)

D1_otof_c_1=D1_otof_c
# D1_otof_c_1['ratio_AUDps']=D1_otof_c_1['n_vertices_AUDp']/(D1_otof_c_1['n_vertices_AUDv']+D1_otof_c_1['n_vertices_AUDd']+D1_otof_c_1['n_vertices_AUDpo'])
# D1_otof_c_1['ratio_AUD_bfd']=D1_otof_c_1['n_vertices_AUD']/D1_otof_c_1['n_vertices_SSp_bfd']
# D1_otof_c_1['ratio_AUD_SSp']=D1_otof_c_1['n_vertices_AUD']/D1_otof_c_1['n_vertices_SSp']
# D1_otof_c_1['ratio_AUD_MO']=D1_otof_c_1['n_vertices_AUD']/D1_otof_c_1['n_vertices_MO']
D1_otof_c_1['ratio_AUDp_cube']=D1_otof_c_1['n_vertices_AUDp']/D1_otof_c_1['cube']
D1_otof_c_1['ratio_AUDs_cube']=(D1_otof_c_1['n_vertices_AUDv']+D1_otof_c_1['n_vertices_AUDd']+D1_otof_c_1['n_vertices_AUDpo'])/D1_otof_c_1['cube']

# D1_otof_c_1['ratio_MO_cube']=D1_otof_c_1['n_vertices_MO']/D1_otof_c_1['cube']
D1_otof_c_1.to_csv('/data_2to/DBA2J_new/comparisonAUDreg_otof_control_1.csv', index=False)

D1_otof_m_1=D1_otof_m
# D1_otof_m_1['ratio_AUDps']=D1_otof_m_1['n_vertices_AUDp']/(D1_otof_m_1['n_vertices_AUDv']+D1_otof_m_1['n_vertices_AUDd']+D1_otof_m_1['n_vertices_AUDpo'])
# D1_otof_m_1['ratio_AUD_bfd']=D1_otof_m_1['n_vertices_AUD']/D1_otof_m_1['n_vertices_SSp_bfd']
# D1_otof_m_1['ratio_AUD_SSp']=D1_otof_m_1['n_vertices_AUD']/D1_otof_m_1['n_vertices_SSp']
# D1_otof_m_1['ratio_AUD_MO']=D1_otof_m_1['n_vertices_AUD']/D1_otof_m_1['n_vertices_MO']
D1_otof_m_1['ratio_AUDp_cube']=D1_otof_m_1['n_vertices_AUDp']/D1_otof_m_1['cube']
D1_otof_m_1['ratio_AUDs_cube']=(D1_otof_m_1['n_vertices_AUDv']+D1_otof_m_1['n_vertices_AUDd']+D1_otof_m_1['n_vertices_AUDpo'])/D1_otof_m_1['cube']

# D1_otof_m_1['ratio_MO_cube']=D1_otof_m_1['n_vertices_MO']/D1_otof_m_1['cube']
D1_otof_m_1.to_csv('/data_2to/DBA2J_new/comparisonAUDreg_otof_mutants_1.csv', index=False)



plt.figure()
sns.lineplot(x="timepoint", y='ratio_AUDs_cube',err_style='bars', data=D1_1, color='forestgreen')
sns.lineplot(x="timepoint", y='ratio_AUDs_cube',err_style='bars', data=D1_otof_m_1, color='indianred')
sns.lineplot(x="timepoint", y='ratio_AUDs_cube',err_style='bars', data=D1_otof_c_1, color='cadetblue')

plt.figure()
sns.lineplot(x="timepoint", y='ratio_AUDp_cube',err_style='bars', data=D1_1, color='forestgreen')
sns.lineplot(x="timepoint", y='ratio_AUDp_cube',err_style='bars', data=D1_otof_m_1, color='indianred')
sns.lineplot(x="timepoint", y='ratio_AUDp_cube',err_style='bars', data=D1_otof_c_1, color='cadetblue')

plt.figure()
sns.lineplot(x="timepoint", y='ratio_AUDps',err_style='bars', data=D1_1, color='forestgreen')
sns.lineplot(x="timepoint", y='ratio_AUDps',err_style='bars', data=D1_otof_m_1, color='indianred')
sns.lineplot(x="timepoint", y='ratio_AUDps',err_style='bars', data=D1_otof_c_1, color='cadetblue')




# df.query('a == 4 & b != 2')
import numpy as np

D1_clean=D1_1[D1_1['brainID']!='9']
D1_clean=D1_clean[D1_clean['brainID']!='1']
filter=np.logical_and((D1_clean['timepoint']==10).values, np.array([i not in ['6', '3', '2'] for i in D1_clean['brainID'].values]))
D1_clean=D1_clean[np.logical_not(filter)]

plt.figure()
sns.lineplot(x="timepoint", y='ratio_AUD_MO',err_style='bars', data=D1_otof_m_1, color='indianred')
sns.lineplot(x="timepoint", y='ratio_AUD_MO',err_style='bars', data=D1_otof_c_1, color='cadetblue')
sns.lineplot(x="timepoint", y='ratio_AUD_MO',err_style='bars', data=D1_clean, color='forestgreen')



plt.figure()
sns.lineplot(x="timepoint", y='ratio_AUD_cube',err_style='bars', data=D1_clean, color='forestgreen')
sns.lineplot(x="timepoint", y='ratio_AUD_cube',err_style='bars', data=D1_otof_m_1, color='indianred')
sns.lineplot(x="timepoint", y='ratio_AUD_cube',err_style='bars', data=D1_otof_c_1, color='cadetblue')



plt.figure()
sns.lineplot(x="timepoint", y='ratio_AUD_bfd',err_style='bars', data=D1_otof_m_1, color='indianred')
sns.lineplot(x="timepoint", y='ratio_AUD_bfd',err_style='bars', data=D1_otof_c_1, color='cadetblue')
sns.lineplot(x="timepoint", y='ratio_AUD_bfd',err_style='bars', data=D1_clean, color='forestgreen')




plt.figure()
sns.lineplot(x="timepoint", y='ratio_AUD_SSp',err_style='bars', data=D1_otof_m_1, color='indianred')
sns.lineplot(x="timepoint", y='ratio_AUD_SSp',err_style='bars', data=D1_otof_c_1, color='cadetblue')
sns.lineplot(x="timepoint", y='ratio_AUD_SSp',err_style='bars', data=D1_clean, color='forestgreen')


