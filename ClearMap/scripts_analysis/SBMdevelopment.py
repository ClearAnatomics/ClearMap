import matplotlib.pyplot as plt
import numpy as np
import ClearMap.Settings as settings
import os
import ClearMap.Alignment.Annotation as ano
import graph_tool.inference as gti
import pandas as pd
import graph_tool.topology as gtt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.Visualization.Plot3d as p3d
import seaborn as sns
atlas_path = os.path.join(settings.resources_path, 'Atlas');
import time

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


def compute_sbm(graph):
    state=gti.minimize_nested_blockmodel_dl(graph.base)
    levels = state.get_levels()
    n = 1
    modules = []
    # for l in levels:
    #     blocks_leveled = l.get_blocks().a
    for i in range(n):
        print(i)
        blocks_leveled = levels[i].get_blocks().a
        if i == 0:
            modules = blocks_leveled
        else:
            modules = np.array([blocks_leveled[b] for b in modules])

    print(np.unique(modules))
    return (modules)




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
TP=[1, 5, 30, 3, 7, 6, 14, 21, 12, 9]

# reg_name=['brain','barrels', 'auditory', 'motor']

workdirs=[ work_dirP1, work_dirP5, work_dirAdult,work_dirP3,work_dirP7,work_dirP6,work_dirP14, work_dir_P21,work_dir_P12,work_dir_P9]
controlslist=[ controlsP1, controlsP5, controlsAdult,controlsP3,controlsP7,controlsP6,controlsP14,controls_P21,controls_P12,controls_P9]

anot_f=['/data_2to/alignement/atlases/new_region_atlases/P1/smoothed_anot_P1.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/home/sophie.skriabine/Projects/ClearMap3/ClearMap/ClearMap/Resources/Atlas/annotation_25_full.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P3/smoothed_anot_P3_V6.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P7/smoothed_anot_P7.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P5/smoothed_anot_P5_half.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P14/ano_P14_p10_corrected_smoothed.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P21/atlasP21.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P10/ano_P10.nrrd',
        '/data_2to/alignement/atlases/new_region_atlases/P10/ano_P10.nrrd']



reg_ids=[315, 549, 1097, 512, 672,313, 1065]
reg_name=['Isocortex', 'Thalamus', 'Hypothalamus', 'Cerebellum', 'Striatum', 'Midbrain', 'Hindbrain']
regions=[]
for id in reg_ids:
    regions.append((ano.find(id)['id'],ano.find(id)['level']))

get_reg=1

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
            namme=ano.find(order, key='order')['name']
            print(id, order, level, ano.find(order, key='order')['name'])
            if TP[j]==21 or TP[j]==12 or TP[j]==9:
                print('labelled by ID')
                graph=extract_AnnotatedRegion(G, region, state='21')
                print(np.unique(label))
            else:
                graph=extract_AnnotatedRegion(G, region)

            print('computing sbm...')
            start = time.time()
            modules=compute_sbm(graph)
            end = time.time()
            print(end - start)
            graph.add_vertex_property('blocks', modules)
            np.save('/data_SSD_2to/DEV_SBM/'+str(TP[j])+'_'+str(g)+'_'+str(namme)+'_modules.npy', modules)
            # gss4.add_vertex_property('indices', indices)
            Q, Qs = modularity_measure(modules, graph, 'blocks')

            D1=D1.append({'timepoint': TP[j],'nb_modules': np.unique(modules).shape[0],'modularity':Q, 'region': reg_name[k], 'n_edges':graph.n_edges, 'n_vertices':graph.n_vertices, 'brainID':g},ignore_index=True)#([j, Lmm, bp/Lmm])#

D1.to_csv('/data_SSD_2to/DEV_SBM/SBM_developement.csv', index=False)
import seaborn as sns

plt.figure()
sns.despine()
pal = sns.cubehelix_palette(len(TP), rot=-.25, light=7.)
sns.scatterplot(data=D1, x="region", y="nb_modules", hue="timepoint", palette=pal)

pal = sns.cubehelix_palette(len(TP), rot=-.25, light=7.)
plt.figure()
sns.despine()
sns.barplot(data=D1, x="region", y="nb_modules", hue="timepoint", palette='hls')

plt.figure()
sns.despine()
sns.barplot(data=D1, x="region", y="modularity", hue="timepoint", palette='hls')



for r in reg_name:
    dtemp=D1[D1['region']==r]
    sns.despine()
    sns.barplot(data=dtemp, x="region", y="nb_modules", hue="timepoint")
    # sns.scatterplot(x='modularity', y='nb_modules', hue='region', palette=, data=dtemp)
    plt.title(r)


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        # cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
        #                            boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap, randRGBcolors



bn = np.zeros(modules.shape)
for i in np.unique(modules):
    # print(i)
    bn[np.where(modules == i)] = np.where(np.unique(modules) == i)
new_cmap, randRGBcolors = rand_cmap(len(np.unique(bn)), type='bright', first_color_black=False,
                                    last_color_black=False, verbose=True)
n = len(graph.vertices)
colorval = np.zeros((n, 3));
for i in range(modules.size):
    colorval[i] = randRGBcolors[int(bn[i])]
# colorval = getColorMap_from_vertex_prop(g2plot.vertex_property('artterr'))
p = p3d.plot_graph_mesh(graph, vertex_colors=colorval, n_tube_points=3);