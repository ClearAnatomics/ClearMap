work_dir='/data_SSD_2to/191122Otof'
controls=['2R','3R','5R', '8R']#['3R']#
mutants=['1R','7R', '6R', '4R']
region_list=[(6,6)]#isocortex
brain= '5R'
graph = ggt.load(work_dir + '/' + brain + '/' + 'data_graph.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
vertex_filter = np.zeros(graph.n_vertices)
for i, rl in enumerate(region_list):
    order, level = region_list[i]
    print(level, order, ano.find(order, key='order')['name'])
    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter[label_leveled == order] = 1;
gss4_t = graph.sub_graph(vertex_filter=vertex_filter)

gss4 = gss4_t.copy()
r, p, n, l = getRadPlanOrienttaion(gss4, gss4_t)  # ,, , calc_art=True)
p = np.nan_to_num(p)#[~np.isnan(r)]
r =  np.nan_to_num(r)#r[~np.isnan(r)]


e_dist = gss4.edge_property('distance_to_surface')#[~np.isnan(r)]
v_dist = gss4.vertex_property('distance_to_surface')
length=gss4.edge_property('length')
angle = np.array([math.acos(r[i]) for i in range(r.shape[0])]) * 180 / pi
gss4.add_edge_property('angle',angle)
gss4_t.add_edge_property('angle',angle)
artery = gss4_t.edge_property('artery')
vartery = from_e_prop2_vprop(gss4_t,'artery' )


## full graph
counts, binEdges=np.histogram(angle,bins=90)
assignement = np.digitize(angle, binEdges)

cumLemgth=0
u=np.unique(assignement)
cumLemgth_isocortex=np.array([np.sum(length[assignement<=i]) for i in u])

plt.figure()
x=np.arange(cumLemgth_isocortex.shape[0])
plt.bar(x,cumLemgth_isocortex/np.max(cumLemgth_isocortex))

## surface
radii=gss4.vertex_property('radii')
gss4_surface = gss4.sub_graph(vertex_filter=np.logical_and(v_dist<1, radii>5))

angle=gss4_surface.edge_property('angle')
length=gss4_surface.edge_property('length')
counts, binEdges=np.histogram(angle,bins=90)
assignement = np.digitize(angle, binEdges)

cumLemgth=0
u=np.unique(assignement)
cumLemgth_surface=np.array([np.sum(length[assignement<=i]) for i in u])

plt.figure()
x=np.arange(cumLemgth_surface.shape[0])
plt.bar(x,cumLemgth_surface/np.max(cumLemgth_surface))



## art graph in barrels
# region_list=[(54,9)]
region_list=[(142, 8), (149, 8), (128, 8), (156, 8)]
region_list=[(1183, 4)]


vertex_filter = np.zeros(gss4_t.n_vertices)
for i, rl in enumerate(region_list):
    order, level = region_list[i]
    print(level, order, ano.find(order, key='order')['name'])
    label = gss4_t.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter[label_leveled == order] = 1;

vertex_filter=np.logical_and(vertex_filter, vartery)
vertex_filter=np.logical_and(np.logical_and(vertex_filter, v_dist>5), v_dist<10)
connectivity = gss4_t.edge_connectivity()
vertex_filter = np.logical_and(vertex_filter[connectivity[:, 0]], vertex_filter[connectivity[:, 1]])
gss4_art_barrels = gss4_t.sub_graph(edge_filter=vertex_filter)
dist = gss4_art_barrels.edge_property('distance_to_surface')



art_length=length[vertex_filter]#gss4_art_barrels.edge_property('length')
# art_angle=gss4_art_barrels.edge_property('angle')
# counts, binEdges=np.histogram(art_angle,bins=90)
art_assignement = assignement[vertex_filter]#np.digitize(art_angle, binEdges)

cumLemgth=0
u=np.unique(art_assignement)
cumLemgth_audi=np.array([np.sum(art_length[art_assignement<=i]) for i in u])

plt.figure()
x=np.arange(cumLemgth.shape[0])
plt.bar(x,cumLemgth/np.max(cumLemgth))

plt.figure()
sns.set_style(style='white')
sns.despine()

x=np.arange(cumLemgth_audi.shape[0])
sns.lineplot(x,cumLemgth_audi/np.max(cumLemgth_audi))
x=np.arange(cumLemgth_barrels.shape[0])
sns.lineplot(x,cumLemgth_barrels/np.max(cumLemgth_barrels))
x=np.arange(cumLemgth_isocortex.shape[0])
sns.lineplot(x,cumLemgth_isocortex/np.max(cumLemgth_isocortex))

plt.legend(['AUD','SSp-bfd', 'isocortex'])

## corpus callosum
with open(work_dir + '/' + brain + '/sampledict' + brain + '.pkl', 'rb') as fp:
    sampledict = pickle.load(fp)

degrees = graph.vertex_degrees()
vf = np.logical_and(degrees > 1, degrees <= 4)
graph = graph.sub_graph(vertex_filter=vf)
pressure = np.asarray(sampledict['pressure'][0])
graph.add_vertex_property('pressure', pressure)
angle = GeneralizedRadPlanorientation(graph)
angle=90-angle
graph.add_edge_property('angle',angle)

# region_list=[(1183, 4)]
region_list=[(142, 8), (149, 8), (128, 8), (156, 8)]
vertex_filter = np.zeros(graph.n_vertices)
for i, rl in enumerate(region_list):
    order, level = region_list[i]
    print(level, order, ano.find(order, key='order')['name'])
    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vertex_filter[label_leveled == order] = 1;

gss4_t = graph.sub_graph(vertex_filter=vertex_filter)
angle=gss4_t.edge_property('angle')
length=gss4_t.edge_property('length')
angle=90-angle
# length=length[angle>40]
# angle=angle[angle>40]
# angle=90-angle


counts, binEdges=np.histogram(angle,bins=90)
assignement = np.digitize(angle, binEdges)

cumLemgth=0
u=np.unique(assignement)
cumLemgth_cc=np.array([np.sum(length[assignement<=i]) for i in u])

plt.figure()
sns.set_style(style='white')

x=np.arange(cumLemgth_cc.shape[0])
sns.lineplot(x,cumLemgth_cc/np.max(cumLemgth_cc))
sns.despine()


#################
import json
with open('/home/sophie.skriabine/Projects/clearVessel_New/ClearMap/ClearMap/Resources/Atlas/annotation.json') as json_data:
    data_dict = json.load(json_data)['msg']
print(data_dict)
reg_leaves = []
order=ano.find('OLF', key='acronym')['order']
level=ano.find('OLF', key='acronym')['level']
region_list=[(order,level)]#
reg_leaves=[]
for reg in region_list:
    global leafArray
    leafArray = []
    order, level = reg
    name=ano.find_name(order, key='order')
    get_child_tree(data_dict, name)
    reg_leaves.append(leafArray)

regions=np.array(reg_leaves[0])[np.asarray(np.array([reg_leaves[0][i][1] for i in range(len(reg_leaves[0]))])==7).nonzero()[0]]
regions=[regions]
controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'



#
# region_list = regions # isocortex
# regions = []
# R = ano.find(region_list[0][0], key='order')['name']
# main_reg = region_list
# sub_region = True
# for r in reg_list.keys():
#     n = ano.find_name(r, key='order')
#     if R in n:
#         for se in reg_list[r]:
#             n = ano.find_name(se, key='order')
#             print(n)
#             regions.append(n)

orientation_method = 'flowInterpolation'

ROI=[]
for region_list in regions:

    vertex_filter = np.zeros(graph.n_vertices)
    for i, rl in enumerate(region_list):
        id, level = rl
        ROI.append(ano.find(id, key='id')['acronym'])

for a, control in enumerate(controls):
    ori = []
    bp = []
    prop_ori = []
    bp_dist_2_surface=[]
    graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
    degrees = graph.vertex_degrees()
    vf = np.logical_and(degrees > 1, degrees <= 4)
    graph = graph.sub_graph(vertex_filter=vf)

    if orientation_method == 'flowInterpolation':
        with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
            sampledict = pickle.load(fp)

        pressure = np.asarray(sampledict['pressure'][0])
        graph.add_vertex_property('pressure', pressure)
        for region_list in regions:

            vertex_filter = np.zeros(graph.n_vertices)
            for i, rl in enumerate(region_list):
                id, level = rl
                order=ano.find(id, key='id')['order']
                print(level, order, ano.find(order, key='order')['name'])
                label = graph.vertex_annotation();
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1;

                gss4_t = graph.sub_graph(vertex_filter=vertex_filter)

                gss4 = gss4_t.copy()
                nv = gss4.n_vertices
                ne = gss4.n_edges
                # bp_dist_2_surface.append(gss4.vertex_property('distance_to_surface'))
                bp.append(gss4.vertex_property('distance_to_surface'))
                angle = GeneralizedRadPlanorientation(gss4)
                dist = gss4.edge_property('distance_to_surface')

                radiality = angle < 40  # limit_angle#40
                planarity = angle > 60  # 90-limit_angle#60

                neutral = np.logical_not(np.logical_or(radiality, planarity))

                ori_prop = np.concatenate(
                    (np.expand_dims(dist, axis=1), np.concatenate((np.expand_dims(radiality, axis=1), np.concatenate(
                        (np.expand_dims(neutral, axis=1), np.expand_dims(planarity, axis=1)), axis=1)), axis=1)), axis=1)
                prop_ori.append(ori_prop)

                rad = angle < 40  # limit_angle  # 40
                plan = angle > 60  # 90 - limit_angle  # 60
                ori.append(r)
                print('done')

        
        np.save(work_dir + '/' + 'BP_OLF'  + '_' + control + '.npy', bp)
        np.save(work_dir + '/' + 'ORI_3_OLF'  + '_' + control + '.npy', ori)
        np.save(work_dir + '/' + 'PROP_ORI_3_OLF'  + '_' + control + '.npy', prop_ori)

brains=controls
thresh=2
nb_reg=10
normed=False
bins=10
features = []
regis = []
for control in brains:


    vess_rad_control=np.load(work_dir + '/' + 'ORI_3_OLF' +  '_'+control+'.npy',allow_pickle=True)
    bp_dist_2_surface_control=np.load(work_dir + '/' + 'BP_OLF'  +  '_'+control+'.npy',allow_pickle=True)
    prop_ori_control=np.load(work_dir + '/' + 'PROP_ORI_3_OLF'  + '_' + control + '.npy',allow_pickle=True)
    features_brains = []
    for reg in range(nb_reg):
        regis.append(regions[0][reg])
        dist = prop_ori_control[reg][:, 0]
        ori_rad = prop_ori_control[reg][:, 1]
        ori_neutral = prop_ori_control[reg][:, 2]
        ori_plan = prop_ori_control[reg][:, 3]
        bp_dist = np.array(bp_dist_2_surface_control[reg])  ##
        ori = np.array(vess_rad_control[reg])
        # radial_depth = ve_ep[ori > 0.6]

        histrad, bins = np.histogram(dist[ori_rad.astype(bool)], bins=10)
        histneut, bins = np.histogram(dist[ori_neutral.astype(bool)], bins=bins)
        histplan, bins = np.histogram(dist[ori_plan.astype(bool)], bins=bins)
        hist_bp_dist, bins_bp_dist = np.histogram(bp_dist[bp_dist > thresh], bins=bins, normed=normed)
        # hist_ori, bins_ori_dist = np.histogram(radial_depth, bins=bin, normed=normed)
        R = histrad / (histrad + histneut + histplan)
        N = histneut / (histrad + histneut + histplan)
        P = histplan / (histrad + histneut + histplan)
        features_brains.append(
            [ hist_bp_dist, P, N, R])  # len(shortest_paths_control)
    features.append(features_brains)


bins = 10
shape = (len(brains), len(features_brains), len(features_brains[0]), bins)
# F = np.zeros(shape).astype(float)
# F = F.astype(float)
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         for k in range(shape[2]):
#             F[i, j, k, :] = features[i][j][k].astype(float)
features=np.array(features)
# features = np.nan_to_num(F)
regis = np.array(regis)

if brains == controls:
    features_avg = np.mean(features, axis=0)
    features_avg_c = features_avg
    features_c = features

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize

colors_c = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
colors_m = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
# ROI=['SSp-n']#['SSp-tr', 'MOs', 'MOp', 'SSp-n', 'SSp-bfd','SSs']
norm = False
f = 9
bins = bins_bp_dist
feat = ['BP', 'PROP_ORI_PLAN', 'PROP_ORI_RAD', 'ORI_RAD']  # ['ART EP', 'VE EP', 'BP', 'ORI']#'SP len', 'SP step',
import pandas as pd

for r in range(features.shape[1]):
    l = ano.find(regis[r][0], key='id')['acronym']
    print(l)
    # print(l)
    if l in ROI:
        print(l)
        plt.figure()
        sns.set_style(style='white')

        # for b in range(features.shape[0]):

        ### normal features
        # plt.plot(np.squeeze(normalize(features_c[b, r, f].reshape(-1, 1), norm='l2', axis=0), axis=1), color=colors_c[b])
        # plt.plot(np.squeeze(normalize(features[b, r, f].reshape(-1, 1), norm='l2', axis=0), axis=1), color=colors_m[b])
        norm = False
        f = 3  # 3
        if norm:
            Cpd_c = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
            Cpd_m = pd.DataFrame(normalize(features_c[:, r, f-2], norm='l2', axis=1)).melt()
            # Cpd_m = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
        else:
            Cpd_c = pd.DataFrame(features_c[:, r, f]).melt()
            Cpd_m = pd.DataFrame(features_c[:, r, f-2]).melt()
            # Cpd_m = pd.DataFrame(features_m[:, r, f]).melt()

        f = 0
        norm = False
        if norm:
            bp_c = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
            # bp_m = pd.DataFrame(normalize(features_m[:, r, f], norm='l2', axis=1)).melt()
        else:
            bp_c = pd.DataFrame(features_c[:, r, f]).melt()
            # bp_m = pd.DataFrame(features_m[:, r, f]).melt()

        sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color=colors_c[0], linewidth=2.5)
        sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_m, color=colors_m[1], linewidth=2.5)
        plt.legend(['prop rad', 'prop plan'])#(['prop rad control', 'prop rad mutant'])  # (['prop rad', 'prop plan'])
        plt.yticks(size='x-large')
        plt.ylabel(feat[2], size='x-large')
        plt.twinx()
        sns.lineplot(x="variable", y="value", err_style='bars', data=bp_c, color=colors_c[2], linewidth=2.5)
        # sns.lineplot(x="variable", y="value", err_style='bars', data=bp_m, color=colors_m[3], linewidth=2.5)

        plt.title(l + ' ' + feat[0], size='x-large')
        plt.legend(['bp'])#(['bp control', 'bp mutant'])  # (['bp'])#['control', 'deprived'])
        plt.yticks(size='x-large')
        plt.ylabel(feat[0], size='x-large')
        plt.xlabel('cortical depth', size='x-large')
        plt.xticks(np.arange(0, 10), np.arange(0, np.max(bins), np.max(bins) / 10).astype(int), size='x-large')
        # sns.despine()
        plt.tight_layout()
        # plt.legend(['bp control', 'bp mutants'])

