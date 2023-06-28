import numpy as np

import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
atlas_path = os.path.join(settings.resources_path, 'Atlas');


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

work_dir_P14='/data_SSD_2to/220503_p14_otof'
controls_P14=['1w', '2w', '4w']
mutants_P14=['5k', '6k', '7k', '8k']


wds=[work_dir_P14, work_dir_1M,work_dir_2M,work_dir_3M, work_dir_6M,work_dir_10M]
ctrls=[controls_P14, controls_1M,controls_2M,controls_3M,controls_6M,controls_10M]
mtts=[mutants_P14, mutants_1M, mutants_2M,mutants_3M,mutants_6M,mutants_10M]

TP=[0.5, 1, 2, 3, 6, 10]
radius=10
template_shape=(320,528,228)

anot05 = '/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed.nrrd'
anotadult='/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd'

regions=[(247, 7)]
reg_name=['Auditory areas']
limit_angle=40
Table=[]

for i , tp in enumerate(TP):
    work_dir=wds[i]
    controls=ctrls[i]
    mutants=mtts[i]
    print(i,work_dir,controls)

    if tp==0.5:
        ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                       annotation_file = anot05)
    else:
        ano.initialize(label_file = os.path.join(atlas_path, 'annotation.json'),
                       annotation_file = anotadult)

    for j,control in enumerate(controls):
        print(control)
        stt='control'
        try:
            graph=ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph_correcteduniverse.gt')
        except:
            graph=ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')

        # try:
        #     artery=graph.vertex_property('artery')
        #     vein=graph.vertex_property('vein')
        #     # artery=from_e_prop2_vprop(graph, 'artery')
        #     # vein=from_e_prop2_vprop(graph, 'vein')
        # except:
        #     try:
        #         artery=from_e_prop2_vprop(graph , 'artery')
        #         vein=from_e_prop2_vprop(graph , 'vein')
        #     except:
        #         print('no artery vertex properties')
        #         break
        #
        # artery_vein=np.logical_or(artery, vein)
        # art_graph=graph.sub_graph(vertex_filter=artery)
        for k,region in enumerate(regions):

            id, level=region
            order= ano.find(id, key='id')['order']
            print(id, order, level, ano.find(order, key='order')['name'])
            # graph=extract_AnnotatedRegion(art_graph, region)
            graph=extract_AnnotatedRegion(graph, region)


            try:
                r, p, n, l = getLocalNormaleOrienttaion(graph, graph,[id],local_normal=True, verbose=True, calc_art=True) # ,, , calc_art=True)


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

            Table.append(np.concatenate((histbprad/histbpplan, np.array([TP[i],reg_name[k], stt])), axis=0))

            # vertex_dist=graph.vertex_property('distance_to_surface')
            # histbp, bins = np.histogram(vertex_dist, bins=10)#, normed=normed)
            # Table.append(np.concatenate((histbp, np.array([TP[i],reg_name[k], stt])), axis=0))




    for j,mutant in enumerate(mutants):
        stt='mutant'
        print(mutant)
        try:
            graph=ggt.load(work_dir + '/' + mutant + '/' + str(mutant)+'_graph_correcteduniverse.gt')
        except:
            graph=ggt.load(work_dir + '/' + mutant + '/' + 'data_graph_correcteduniverse.gt')

        # try:
        #     artery=graph.vertex_property('artery')
        #     vein=graph.vertex_property('vein')
        #     # artery=from_e_prop2_vprop(graph, 'artery')
        #     # vein=from_e_prop2_vprop(graph, 'vein')
        # except:
        #     try:
        #         artery=from_e_prop2_vprop(graph , 'artery')
        #         vein=from_e_prop2_vprop(graph , 'vein')
        #     except:
        #         print('no artery vertex properties')
        #         break

        # artery_vein=np.logical_or(artery, vein)
        # art_graph=graph.sub_graph(vertex_filter=artery)

        for k,region in enumerate(regions):

            id, level=region
            order= ano.find(id, key='id')['order']
            print(id, order, level, ano.find(order, key='order')['name'])
            # graph=extract_AnnotatedRegion(art_graph, region)
            graph=extract_AnnotatedRegion(graph, region)

            try:
                r, p, n, l = getLocalNormaleOrienttaion(graph, graph,[id],local_normal=True, verbose=True, calc_art=True) # ,, , calc_art=True)


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

            Table.append(np.concatenate((histbprad/histbpplan, np.array([TP[i],reg_name[k], stt])), axis=0))


            # vertex_dist=graph.vertex_property('distance_to_surface')
            # histbp, bins = np.histogram(vertex_dist, bins=10)#, normed=normed)
            # Table.append(np.concatenate((histbp, np.array([TP[i],reg_name[k], stt])), axis=0))


feat_cols = [ 'feat'+str(i) for i in range(10) ]
feat_cols.append('timepoint')
feat_cols.append('region')
feat_cols.append('statue')
df = pd.DataFrame(Table,columns=feat_cols)

df.to_csv('/data_2to/devotof/otof_ORI_table.csv', index=False)

df.to_csv('/data_2to/devotof/otof_BP_table.csv', index=False)
df.to_csv('/data_2to/devotof/otof_arteries_BP_table.csv', index=False)
df=pd.read_csv('/data_2to/devotof/otof_arteries_BP_table.csv')

pal = sns.cubehelix_palette(len(TP), rot=-.25, light=.7)

data2plt=df.iloc[:, :12]


for col in df.columns[:-2]:
    new_col=df[col].values.astype(float)
    df[col]=new_col


for tp in TP:
    plt.figure()
    region_name=ano.find(region[0])['name']
    print(region_name)
    data=df[df['region']==region_name]

    d=data[data['timepoint']==tp]
    Cpd_c = pd.DataFrame(d[d['statue']=='control']).iloc[:,:-2].melt()
    Cpd_m = pd.DataFrame(d[d['statue']=='mutant']).iloc[:,:-2].melt()
    print(tp, Cpd_c.shape)
    col=pal[i]
    sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color='forestgreen', linewidth=2.5)
    sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_m, color='indianred', linewidth=2.5)


    plt.title(tp)













for i, controls in enumerate(ctrls):
    work_dir=wds[i]
    vox_shape_c=(template_shape[0],template_shape[1],template_shape[2], len(controls))
    vox_control=np.zeros(vox_shape_c)

    ###     deg1 voxelization
    for j,control in enumerate(controls):
        print(work_dir, control)
        try:
            graph=ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph_correcteduniverse.gt')
        except:
            graph=ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')

        coordinates = graph.vertex_property('coordinates_atlas') # coordinates_atlas
        # coordinates[:, :2] = coordinates[:, :2] * 1.625 / 25
        # coordinates[:, 2] = coordinates[:, 2] * 2 / 25

        deg0=graph.vertex_degrees()==1
        v = vox.voxelize(coordinates[deg0, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
        vox_control[:,:,:,j]=v

    vox_control_avg=np.mean(vox_control[:, :, :, :], axis=3)
    io.write(work_dir + '/' +'vox_control_deg1'+str(radius)+'.tif', vox_control.astype('float32'))
    io.write(work_dir + '/' +'vox_control_avg_deg1'+str(radius)+'.tif', vox_control_avg.astype('float32'))

for i, mutants in enumerate(mtts):
    work_dir=wds[i]
    vox_shape_m=(template_shape[0],template_shape[1],template_shape[2], len(mutants))
    vox_mutant=np.zeros(vox_shape_m)

    for j,control in enumerate(mutants):
        print(work_dir, control)
        try:
            graph=ggt.load(work_dir + '/' + control + '/' + str(control)+'_graph_correcteduniverse.gt')
        except:
            graph=ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')
        coordinates = graph.vertex_property('coordinates_atlas') # coordinates_atlas
        # coordinates[:, :2] = coordinates[:, :2] * 1.625 / 25
        # coordinates[:, 2] = coordinates[:, 2] * 2 / 25

        deg0=graph.vertex_degrees()==1
        v = vox.voxelize(coordinates[deg0, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
        vox_mutant[:,:,:,j]=v

    vox_mutant_avg=np.mean(vox_mutant[:, :, :, :], axis=3)
    io.write(work_dir + '/' +'vox_mutant_deg1'+str(radius)+'.tif', vox_mutant.astype('float32'))
    io.write(work_dir + '/' +'vox_mutant_avg_deg1'+str(radius)+'.tif', vox_mutant_avg.astype('float32'))









from scipy import stats


for i, controls in enumerate(ctrls):
    work_dir=wds[i]
    vox_control=io.read(work_dir + '/' +'vox_control_deg1'+str(radius)+'.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')
    vox_mutant=io.read(work_dir + '/' +'vox_mutant_deg1'+str(radius)+'.tif')#io.read(work_dir + '/' +'vox_ori4_control_rad'+str(radius)+'.tif')

    print(work_dir)
    pcutoff = 0.05
    tvals, pvals = stats.ttest_ind(vox_control[:, :, :,:], vox_mutant[:, :, :, :], axis = 3, equal_var = False);
    pi = np.isnan(pvals);
    pvals[pi] = 1.0;
    tvals[pi] = 0;
    pvals2 = pvals.copy();
    pvals2[pvals2 > pcutoff] = pcutoff;
    psign=np.sign(tvals)
    ## from sagital to coronal view
    pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
    psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
    # pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
    # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
    pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])

    pcutoff = 0.01
    pvals2 = pvals.copy();
    pvals2[pvals2 > pcutoff] = pcutoff;
    psign=np.sign(tvals)
    ## from sagital to coronal view
    pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
    psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
    pvalscol_01 = colorPValues(pvals2_f, psign_f, positive = [0,255,0], negative = [0,0,255])

    pvalscol_f=np.maximum(pvalscol, pvalscol_01)

    # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
    # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
    import tifffile
    tifffile.imsave(work_dir+'/pvalcolors_deg1_density_bicol'+str(radius)+'.tif', np.swapaxes(pvalscol_f, 2, 0).astype('uint8'), photometric='rgb',imagej=True)




import pandas as pd
region_list = [(13, 7), (19, 8), (25, 8), (103, 8)]

region_list=[(54,9), (127, 7), (6,6), (0,0)]
dt=pd.DataFrame()
for i in range(len(wds)):
    work_dir=wds[i]
    print(work_dir)
    ctrl=ctrls[i]
    mtt=mtts[i]
    for control in ctrl:
        g=control
        try:
            # G = ggt.load(work_dir + '/' + '/' +'3_graph_uncrusted.gt')
            graph = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
        except:
            try:
                graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                graph = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')


        # vertex_filter = np.zeros(graph.n_vertices)
        for j, rl in enumerate(region_list):
            if j==0 and TP[i]!=0.5:
                print(TP[0], j)
                R=[(54, 9),(47,9)]
                vertex_filter = np.zeros(graph.n_vertices)
                for r in R:
                    order, level = r
                    print(level, order, ano.find(order, key='order')['name'])
                    label = graph.vertex_annotation();
                    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                    vertex_filter[label_leveled == order] = 1;
                g = graph.sub_graph(vertex_filter=vertex_filter)
                dt=dt.append({'timepoint': TP[i], 'condition': 'control', 'n_edges':g.n_edges,'n_vertices':g.n_vertices,'region':54},ignore_index=True)

            else:
                print(TP[i], j)
                vertex_filter = np.zeros(graph.n_vertices)
                order, level = region_list[j]
                print(level, order, ano.find(order, key='order')['name'])
                label = graph.vertex_annotation();
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1;
                g = graph.sub_graph(vertex_filter=vertex_filter)
                dt=dt.append({'timepoint': TP[i], 'condition': 'control', 'n_edges':g.n_edges,'n_vertices':g.n_vertices,'region':order},ignore_index=True)

    for mutant in mtt:
        g=mutant
        try:
            # G = ggt.load(work_dir + '/' + '/' +'3_graph_uncrusted.gt')
            graph = ggt.load(work_dir + '/' + g + '/' + str(g)+ '_graph_correcteduniverse_smoothed.gt')
        except:
            try:
                graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')
            except:
                graph = ggt.load(work_dir + '/' + g + '/' + str(g)+'_graph_correcteduniverse.gt')


        # vertex_filter = np.zeros(graph.n_vertices)
        for j, rl in enumerate(region_list):
            if j==0 and TP[i]!=0.5:
                print(TP[0], j)
                R=[(54, 9),(47,9)]
                vertex_filter = np.zeros(graph.n_vertices)
                for r in R:
                    order, level = r
                    print(level, order, ano.find(order, key='order')['name'])
                    label = graph.vertex_annotation();
                    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                    vertex_filter[label_leveled == order] = 1;
                g = graph.sub_graph(vertex_filter=vertex_filter)
                dt=dt.append({'timepoint': TP[i], 'condition': 'mutant', 'n_edges':g.n_edges,'n_vertices':g.n_vertices,'region':54},ignore_index=True)

            else:
                print(TP[i], j)
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
             data=dt[dt['region']==54])
plt.title(ano.find(54, key='order')['name'])


plt.figure()
sns.lineplot(x="timepoint", y='n_vertices',err_style='bars',hue='condition',
             data=dt[dt['region']==127])
plt.title(ano.find(127, key='order')['name'])


plt.figure()
sns.lineplot(x="timepoint", y='n_vertices',err_style='bars',hue='condition',
             data=dt[dt['region']==6])
plt.title(ano.find(6, key='order')['name'])

plt.figure()
sns.lineplot(x="timepoint", y='n_vertices',err_style='bars',hue='condition',
             data=dt[dt['region']==0])
plt.title(ano.find(0, key='order')['name'])


dt.to_csv('/data_2to/devotof/BP_smoothed_all_cortical_and_brain_subregion.csv', index=False)










