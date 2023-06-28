import ClearMap.Alignment.Annotation as ano

import ClearMap.IO.IO as io
import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti
import os
print('TEST')
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import graph_tool.centrality as gtc
print('loading...')
import numpy as np
import numexpr as ne
import graph_tool.topology as gtt
from sklearn import preprocessing
# from ClearMap.Visualization.Vispy.sbm_plot import *
import seaborn as sns
from scipy.stats import ttest_ind
import math
pi=math.pi
import pickle
from sklearn.linear_model import LinearRegression
work_dir='/data_SSD_2to/191122Otof'#'/data_SSD_2to/whiskers_graphs'
graph_nb=['1R','2R','3R','5R','6R','7R', '8R', '4R']#['44R', '30R']#, '39L']


work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
graph_nb = ['138L', '141L', '142L', '158L', '163L', '162L', '164L']

controls=['142L','158L','162L', '1vox_shape_c=(320,528,228, len(controls))64L']
mutants=['138L','141L', '163L', '165L']
import pickle
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open('/data_SSD_2to/181002_4/reg_list.p', 'rb') as fp:
  reg_list = pickle.load(fp)

with open('/data_SSD_2to/191122Otof/reg_list_full.p', 'rb') as fp:
  reg_list = pickle.load(fp)

with open('/data_SSD_2to/181002_4/atlas_volume_list.p', 'rb') as fp:
  atlas_list = pickle.load(fp)

regions=['Inferior colliculus','lateral lemniscus', 'Superior olivary complex, lateral part', 'Cochlear nuclei']
# regions=[]
controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']

controls=['142L','158L','162L', '164L']
mutants=['138L','141L', '163L', '165L']

colors = ['cadetblue', 'indianred', 'darkgoldenrod', 'darkorange', 'royalblue', 'blueviolet', 'forestgreen',
          'lightseagreen']

#%%
# def parseTree(obj):
#     # print(obj)
#     # if len(obj["children"]) == 0:
#     #     leafArray.append(obj['id'])
#     # else:
#         leafArray.append((obj['id'], ano.find_level(obj['id'])))
#
#         for child in obj["children"]:
#             parseTree(child)
#
# def get_child_tree(data_dict, reg_name):
#
#     for data in data_dict:
#         print(data['name'])
#         if data['name']==reg_name:
#             tree = data  # json.loads(data.strip())
#             leafArray.append((tree['id'], ano.find_level(tree['id'])))
#             for child in tree["children"]:
#                 parseTree(child)
#         # for child in data["children"]:
#         #     print(child)
#         get_child_tree(data["children"], reg_name)

def get_volume_region(region_leaves, atlas):
    val=0
    for l in region_leaves:
      val=val+np.sum(atlas==l[0])
    return val


def from_v_prop2_eprop(graph, property):
    if isinstance(property, ''.__class__):
        vprop = graph.vertex_property(property)#vprop = property#graph.vertex_property(property)
    else:
        vprop=property
    # e_prop=np.zeros(graph.n_edges)
    connectivity = graph.edge_connectivity()
    e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
    return e_prop

def from_e_prop2_vprop(graph, property):
    if isinstance(property, ''.__class__):
        e_prop=graph.edge_property(property)#e_prop = property#else:
    else:
        e_prop=property
    v_prop=np.zeros(graph.n_vertices)
    connectivity = graph.edge_connectivity()
    v_prop[connectivity[e_prop==1,0]]=1
    v_prop[connectivity[e_prop == 1,1]] = 1
    # graph.add_vertex_property(property, v_prop)
    return v_prop

def get_nb_radial_vessels(edge_color):
  radial=edge_color[:,2]/(edge_color[:,0]+edge_color[:,1]+edge_color[:,2])
  return(np.sum(radial>0.7))


def get_nb_parrallel_vessels(edge_color):
  planar=(edge_color[:,0]+edge_color[:,1])/(edge_color[:,2]+edge_color[:,0]+edge_color[:,1])
  print(planar.shape)
  return(np.sum(planar>0.7))


def getRadPlanOrienttaion(graph, ref_graph, local_normal=False, calc_art=False, verbose=False):

    if local_normal:
        rad_f = np.zeros(graph.n_edges)
        plan_f = np.zeros(graph.n_edges)
        lengths_f = graph.edge_property('length')
        norm_f = np.zeros(graph.n_edges)
        label = graph.vertex_annotation();
        for r in reg_list.keys():
            o = r#ano.find(r, key='id')['order']
            l = ano.find(r, key='id')['level']

            label_leveled = ano.convert_label(label, key='id', value='id', level=l)
            vf = label_leveled == o  # 54;
            ef=from_v_prop2_eprop(graph, vf)
            try:
                # sub_graph=graph.sub_graph(edge_filter=ef)
                sub_graph = graph.sub_graph(vertex_filter=vf)
                if verbose:
                    print(ano.find(o, key='id')['name'], sub_graph)
                r, p, n,l=getRadPlanOrienttaion(sub_graph, sub_graph, local_normal=False, calc_art=calc_art)

                rad_f[ef]=r
                plan_f[ef] = p
                norm_f[ef] = n
            except:
                if verbose:
                    print('problem', ano.find(o, key='id')['name'], ef.shape)

        rad=rad_f
        plan=plan_f
        lengths=lengths_f
        N=norm_f
        # rad = rad_f[~np.isnan(rad_f)]
        # lengths = lengths_f[np.asarray(~np.isnan(rad))]
        # plan = plan_f[~np.isnan(plan_f)]

    else:
        if verbose:
            print(graph, ref_graph)
        x = ref_graph.vertex_coordinates()[:, 0]
        y = ref_graph.vertex_coordinates()[:, 1]
        z = ref_graph.vertex_coordinates()[:, 2]
        dist = ref_graph.vertex_property('distance_to_surface')
        if calc_art:
            ## method 1 : considering arteries orientation
            # art=from_e_prop2_vprop(ref_graph,'artery')
            # print('art', np.sum(art))
            # rad=ref_graph.vertex_property('radii')
            # ef=np.logical_and(art, np.logical_and(dist>0.5, rad>=(np.mean(rad)+np.std(rad))))
            # print(np.sum(ef))
            # g = ref_graph.sub_graph(vertex_filter=ef)
            # x = g.vertex_coordinates()[:, 0]
            # y = g.vertex_coordinates()[:, 1]
            # z = g.vertex_coordinates()[:, 2]
            #
            # connectivity = g.edge_connectivity()
            #
            # edge_vect = np.array(
            #     [x[connectivity[:, 1]] - x[connectivity[:, 0]], y[connectivity[:, 1]] - y[connectivity[:, 0]],
            #      z[connectivity[:, 1]] - z[connectivity[:, 0]]]).T
            # # print(edge_vect.shape)
            # top2bot=np.mean(edge_vect, axis=0)
            # top2bot = top2bot / np.linalg.norm(top2bot)  # preprocessing.normalize(top2bot, norm='l2')
            # print(top2bot)
            m=np.min(dist)+1.5

            x_surface=x[dist<m]

            y_surface = y[dist < m]

            z_surface = z[dist < m]

            coords=np.zeros((np.sum(dist<m), 3))
            coords[:, 0]=x_surface
            coords[:, 1]=y_surface
            coords[:,2]=z_surface
            if verbose:
                print(coords.shape)
            G = coords.sum(axis=0) / coords.shape[0]
            # run SVD
            u, s, vh = np.linalg.svd(coords - G)
            # unitary normal vector
            top2bot = vh[2, :]
            if verbose:
                print(top2bot, np.linalg.norm(top2bot))
        else:
            x = ref_graph.vertex_coordinates()[:, 0]
            y = ref_graph.vertex_coordinates()[:, 1]
            z = ref_graph.vertex_coordinates()[:, 2]

            dist_min=np.min(dist)
            dist_max=np.max(dist)
            top_vertices=dist<=dist_min+1
            bottom_vertices=dist>= dist_max-1

            top2bot=np.array([np.mean(x[bottom_vertices])-np.mean(x[top_vertices]), np.mean(y[bottom_vertices])-np.mean(y[top_vertices]), np.mean(z[bottom_vertices])-np.mean(z[top_vertices])])
            top2bot=top2bot / np.linalg.norm(top2bot)#preprocessing.normalize(top2bot, norm='l2')
            if verbose:
                print(top2bot)

        x = graph.vertex_coordinates()[:, 0]
        y = graph.vertex_coordinates()[:, 1]
        z = graph.vertex_coordinates()[:, 2]

        connectivity = graph.edge_connectivity()
        lengths = graph.edge_property('length')
        edge_vect = np.array(
            [x[connectivity[:, 1]] - x[connectivity[:, 0]], y[connectivity[:, 1]] - y[connectivity[:, 0]],
             z[connectivity[:, 1]] - z[connectivity[:, 0]]]).T

        normed_edge_vect=np.array([edge_vect[i] / np.linalg.norm(edge_vect[i]) for i in range(edge_vect.shape[0])])
        N=np.linalg.norm(edge_vect[i])
        print(N)
        # normed_edge_vect=normed_edge_vect[~np.isnan(normed_edge_vect)]
        rad=np.array([np.dot(top2bot.transpose(), normed_edge_vect[i]) for i in range(edge_vect.shape[0])])
        plan=np.sqrt(1-rad**2)

        # lengths=lengths[np.asarray(~np.isnan(rad))]
        # rad = rad[~np.isnan(rad)]
        # plan = plan[~np.isnan(plan)]
    return abs(rad), abs(plan), N, lengths



def cart2sph(x,y,z, ceval=ne.evaluate):
    """ x, y, z :  ndarray coordinates
        ceval: backend to use:
              - eval :  pure Numpy
              - numexpr.evaluate:  Numexpr """
    if x.shape[0]!=0:
        r = ceval('sqrt(x**2+y**2+z**2)')#sqrt(x * x + y * y + z * z)
        theta = ceval('arccos(z/r)*180')/pi#acos(z / r) * 180 / pi  # to degrees
        phi = ceval('arctan2(y,x)*180')/pi#*180/3.4142
        # azimuth = ceval('arctan2(y,x)')
        # xy2 = ceval('x**2 + y**2')
        # elevation = ceval('arctan2(z, sqrt(xy2))')
        # r = ceval('sqrt(xy2 + z**2)')
        rmax=np.max(r)
    else:
        print('no orientation to compute')
        r=np.array([0])
        theta=np.array([0])
        phi=np.array([0])
        rmax=1
    return phi/180, theta/180, r/rmax#, theta/180, phi/180

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


def getVesselOrientation(subgraph, graph):
    x = subgraph.vertex_coordinates()[:, 0]
    y = subgraph.vertex_coordinates()[:, 1]
    z = subgraph.vertex_coordinates()[:, 2]

    x_g = graph.vertex_coordinates()[:, 0]
    y_g = graph.vertex_coordinates()[:, 1]
    z_g = graph.vertex_coordinates()[:, 2]

    center = np.array([np.mean(x_g), np.mean(y_g), np.mean(z_g)])
    x = x - np.mean(x_g)
    y = y - np.mean(y_g)
    z = z - np.mean(z_g)

    spherical_coord = np.array(cart2sph(x, y, z, ceval=ne.evaluate)).T
    connectivity = subgraph.edge_connectivity()

    x_s = spherical_coord[:, 0]
    y_s = spherical_coord[:, 1]
    z_s = spherical_coord[:, 2]

    spherical_ori = np.array(
        [x_s[connectivity[:, 1]] - x_s[connectivity[:, 0]], y_s[connectivity[:, 1]] - y_s[connectivity[:, 0]],
         z_s[connectivity[:, 1]] - z_s[connectivity[:, 0]]]).T
    # orientations=preprocessing.normalize(orientations, norm='l2')

    # edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;

    # spherical_ori=np.array(cart2sph(orientations[:, 0],orientations[:, 1],orientations[:, 2], ceval=ne.evaluate)).T
    spherical_ori = preprocessing.normalize(spherical_ori, norm='l2')
    spherical_ori = np.abs(spherical_ori)

    return spherical_ori


def plotRelativeValues(bp_layer_artery_control, bp_layer_artery_mutant, condition):
    from sklearn.linear_model import LinearRegression
    plt.figure()
    vess_controls = np.array(bp_layer_artery_control).astype(float)[:, :]
    for i in range(vess_controls.shape[0]):
        m = np.mean(vess_controls[i])
        s = np.std(vess_controls[i])
        vess_controls[i] = (vess_controls[i] - m) / s
        # vess_controls[i]=vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]

    baseline = np.mean(vess_controls, axis=0)
    model = LinearRegression()

    for i in range(vess_controls.shape[0]):
        m = np.mean(vess_controls[i])
        s = np.std(vess_controls[i])
        vess_controls[i] = (vess_controls[i] - m) / s
        # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])  # vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controls[i, 2]
        model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
        a = model.coef_
        b = model.intercept_
        print('intercept:', b)
        print('slope:', a)
        vess_controls[i] = vess_controls[i] - a * baseline - b

    C = []
    colors = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
    for i in range(vess_controls.shape[1]):
        c = vess_controls[:, i]
        C.append(c.tolist())
    for i in range(vess_controls.shape[0]):
        plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')

    if len(np.array(C).shape) == 3:
        C = np.squeeze(np.array(C), axis=2)
    else:
        C = np.array(C)

    dfc = pd.DataFrame(np.array(C).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)

    vess_controls = np.array(bp_layer_artery_mutant).astype(float)[:, :]
    for i in range(vess_controls.shape[0]):
        m = np.mean(vess_controls[i])
        s = np.std(vess_controls[i])
        vess_controls[i] = (vess_controls[i] - m) / s
        # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]
        model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
        a = model.coef_
        b = model.intercept_
        print('intercept:', b)
        print('slope:', a)
        vess_controls[i] = vess_controls[i] - a * baseline - b
    M = []
    colors = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
    for i in range(vess_controls.shape[1]):
        m = vess_controls[:, i]
        M.append(m.tolist())
    for i in range(vess_controls.shape[0]):
        plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')

    if len(np.array(M).shape) == 3:
        M = np.squeeze(np.array(M), axis=2)
    else:
        M = np.array(M)

    dfm = pd.DataFrame(np.array(M).transpose()).melt()
    sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfm)



    for i in range(C.shape[0]):
        gc = np.array(dfc.loc[lambda dfc: dfc['variable'] == i, 'value'])
        gm = np.array(dfm.loc[lambda dfm: dfm['variable'] == i, 'value'])
        st, pval = ttest_ind(gc, gm)
        print(reg, ls[i], pval)
        if pval < 0.06:
            print(reg, ls[i], pval, '!!!!!!!!!!!!!!!!11')
    sns.despine()
    plt.ylabel('nb of branch point in ' + condition, size='x-large')
    plt.xlabel('layers', size='x-large')
    plt.xticks([0,1,2,3,4,5], ['l1', 'l2/3','l4','l5','l6a','l6b'], size='x-large')
    # plt.xticks([0, 1, 2, 3, 4], ['l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
    plt.yticks(size='x-large')
    plt.legend(['controls', 'Deprived'])  # Otof-/-
    plt.title('nb of branch point per layer in ' + condition, size='x-large')
    # plt.legend(*zip(*labels), loc=2)
    plt.legend(['controls', 'Deprived'])  # Otof-/-
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('blue')
    leg.legendHandles[1].set_color('orange')
    plt.tight_layout()







def parrallelGraphFeatureExtraction(args):
    I, graph ,region_list, feature, compute_distance, mode, compute_loops, basis = args

    length_short_path_control = []
    art_ep_dist_2_surface_control = []
    ve_ep_dist_2_surface_control = []

    vertex_filter = np.zeros(graph.n_vertices)
    for i, rl in enumerate(region_list):
        order, level = region_list[i]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
    print('sub_graph')
    graph_t = graphsub(graph.copy(),vertex_filter=vertex_filter)
    print('sub_graph_2')
    if compute_distance:
        distance_control = diffusion_through_penetrating_arteries(graph_t , get_penetration_veins_dustance_surface,
                                                                  get_penetrating_veins_labels, vesseltype='art',
                                                                  graph_dir=work_dir + '/' + control,
                                                                  feature='distance')
        distances_control[I]=distance_control

    if feature == 'vessels':
        gss4 = graph_t.copy()
        bp_dist_2_surface[I]=gss4.vertex_property('distance_to_surface')

    elif feature == 'art_raw_signal':
        artery = from_e_prop2_vprop(gss4_t, 'artery')
        vertex_filter = np.logical_and(artery, gss4_t.vertex_property('artery_binary') > 0)  # np.logical_and()
        gss4 = graphsub(graph_t.copy(),vertex_filter=vertex_filter)

        distance_from_suface = gss4.vertex_property('distance_to_surface')
        # print('distance_to_surface ; ', distance_from_suface.shape)

        penetrating_arteries = distance_from_suface > 1
        print(penetrating_arteries.shape)
        art_grt = graphsub(gss4.copy(),vertex_filter=penetrating_arteries)
        # standardization
        a_r = (art_grt.vertex_property('artery_raw') - np.mean(art_grt.vertex_property('artery_raw'))) / np.std(
            art_grt.vertex_property('artery_raw'))
        a_r = a_r + 10
        print(gss4_t)
        print(art_grt)
        connectivity = art_grt.edge_connectivity()
        length = art_grt.edge_property('length')
        p_a_labels = gtt.label_components(art_grt.base)
        p_a_labels_hist = p_a_labels[1]
        p_a_labels = p_a_labels[0]

        p_a_labels = ggt.vertex_property_map_to_python(p_a_labels, as_array=True)
        u_l, c_l = np.unique(p_a_labels, return_counts=True)
        nb_artery_control.append(u_l.shape[0])
        bp_artery_control.append(c_l)

        raw_signal = []
        lengths = []
        for j, v in enumerate(u_l):
            # print(i, ' / ', len(u_l))
            if c_l[j] > 3:
                vprop = p_a_labels == v
                e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
                r_signal = np.sum(a_r[vprop]) / np.sum(vprop)
                raw_signal.append(r_signal)
                lengths.append(np.sum(length[e_prop]))
        raw_signal_control.append(np.array(raw_signal))
        art_length_control.append(lengths)

    label = gss4.vertex_annotation();
    connectivity = gss4.edge_connectivity()
    NbLoops = 0
    loopsLayer = []
    # mode='layers'
    if mode == 'layers':
        if sub_region:
            layers = ['1', '2/3', '4', '5', '6a', '6b']
            # layers = ['4']
            for b, layer in enumerate(layers):
                loopsLayerLength = []
                vertex_filter = np.zeros(gss4.n_vertices)
                for i, rl in enumerate(region_list):
                    R = ano.find(region_list[i][0], key='order')['name']
                    for r in reg_list.keys():
                        n = ano.find_name(r, key='order')
                        if R in n:
                            for se in reg_list[r]:
                                if layer in ano.find(se, key='order')['name']:
                                    l = ano.find(se, key='order')['level']
                                    print(ano.find(se, key='order')['name'], se)
                                    label_leveled = ano.convert_label(label, key='order', value='order', level=l)
                                    vertex_filter[label_leveled == se] = 1
                vess_tree = graphsub(gss4.copy(), vertex_filter=vertex_filter)
                dts = vess_tree.vertex_property('distance_to_surface')
                plt.figure()
                plt.hist(dts)
                plt.title(layer + ' ' + control)

                connectivity = vess_tree.edge_connectivity()
                lengths = vess_tree.edge_geometry_lengths('length')

                degree = vess_tree.vertex_degrees()
                ep[I]=np.sum(degree == 1)
                bp[I]=np.sum(degree >= 3)

                if feature == 'vessels':
                    r, p, l = getRadPlanOrienttaion(vess_tree, gss4_t, True, True)
                    r_f = r[~np.isnan(r)]
                    # l = l[np.asarray(~np.isnan(r))]
                    p = p[~np.isnan(p)]
                    r = r_f
                    rad = np.sum((r / (r + p)) > 0.5)
                    plan = np.sum((p / (r + p)) > 0.6)
                    # print(rad)
                    # print(plan)
                    # vess_plan.append(plan / vess_tree.n_edges)
                    vess_rad[I]=rad / vess_tree.n_edges

                    ## loops
                    # loops = []
                    # for i, b in enumerate(basis):
                    #     # if i >= 3:
                    #     res = gtt.subgraph_isomorphism(b.base, vess_tree.base, induced=True)
                    #     res = checkUnicityOfElement(res)
                    #     loops.append(len(res))
                    #     NbLoops = NbLoops + len(res)
                    #     lenghtsloops=[]
                    #     # check distance length of loops
                    #     for r in res:
                    #         v_res = np.zeros(vess_tree.n_vertices)
                    #         v_res[r] = 1
                    #         loops_edges = np.logical_and(v_res[connectivity[:, 0]], v_res[connectivity[:, 1]])
                    #         lenghtsloops.append(np.sum(lengths[loops_edges]))
                    #     loopsLayerLength.append(lenghtsloops)
                    #     print(control, layer, i,  len(loopsLayerLength))
                # loopslengthBrain.append(loopsLayerLength)
                # loopsLayer.append(loops)

                # for region in regions:
                #     e = ano.find_order(region, key='name')
                #     label = gss4.vertex_annotation();
                #     print(region, e)
                #     label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
                #     # print(np.unique(label_leveled))
                #     vertex_filter = label_leveled == e  # 54;
                #     print(np.sum(vertex_filter))
                #     vess_tree = gss4.sub_graph(vertex_filter=vertex_filter)
                #     degree = vess_tree.vertex_degrees()
                #     ep.append(np.sum(degree == 1))
                #     bp.append(np.sum(degree >= 3))
            ep_layer_artery_control.append(ep)
            bp_layer_artery_control.append(bp)
            vess_rad_control.append(vess_rad)
            NbLoops_control.append(NbLoops)
            loopsLayerLength_control.append(loopslengthBrain)
            # print(len(loopsLayerLength_control[a][b]))
            loopsLayer_control.append(loopsLayer)

    elif mode == 'clusters':
        print('cluster mode')
        r, p, l = getRadPlanOrienttaion(gss4, gss4_t, local_normal=True, calc_art=True)
        r_f = np.nan_to_num(r)
        d = gss4.edge_property('distance_to_surface')
        # l = l[np.asarray(~np.isnan(r))]
        p = np.nan_to_num(p)
        # print(rad)
        # print(plan)
        # vess_plan.append(plan / vess_tree.n_edges)
        vess_rad[I]=np.concatenate((r_f, d), axis=0)

        loops_length = []
        loops_size = []
        cluster = gss4.vertex_property('cluster')
        artery = from_e_prop2_vprop(gss4, 'artery')
        vein = from_e_prop2_vprop(gss4, 'vein')
        shortest_paths = []
        indices = gss4.vertex_property('overlap')
        u, c = np.unique(indices, return_counts=True)
        n = 0
        j_u = 0
        # for i, e in tqdm(enumerate(u)):
        for i in tqdm(range(u.shape[0])):
            e = u[i]
            j = np.asarray(u == e).nonzero()[0][0]
            print(j.shape)
            vf = np.zeros(gss4.n_vertices)
            vf[np.asarray(indices == e).nonzero()[0]] = 1

            ## get extended overlap
            vf = gss4.expand_vertex_filter(vf, steps=ext_step)
            # print(gss4, vf.shape, 'containing ', np.sum(vf), ' vertices')

            g2plot = graphsub(gss4.copy(),vertex_filter=vf)
            c = g2plot.vertex_property('cluster')
            c = c[0]
            g2plot = g2plot.largest_component()
            if g2plot.n_vertices <= 3000 and g2plot.n_edges >= 5:
                print(g2plot, vf.shape, 'containing ', np.sum(vf), ' vertices')
                connectivity = g2plot.edge_connectivity()
                lengths = g2plot.edge_geometry_lengths('length')
                shortest_path = []
                if compute_loops:
                    for i, b in enumerate(basis):
                        # if i >= 3:
                        res = gtt.subgraph_isomorphism(b.base, g2plot.base, induced=True)
                        res = checkUnicityOfElement(res)

                        # check distance length of loops
                        for r in res:
                            v_res = np.zeros(g2plot.n_vertices)
                            v_res[r] = 1
                            loops_edges = np.logical_and(v_res[connectivity[:, 0]], v_res[connectivity[:, 1]])
                            loops_length.append(np.sum(lengths[loops_edges]))
                            loops_size.append(i + 3)

                ## shortest path

                art = np.logical_and(cluster[:, 0] == c[0], artery)
                ve = np.logical_and(cluster[:, 1] == c[1], vein)
                vf[np.asarray(art == 1).nonzero()[0]] = 1
                vf[np.asarray(ve == 1).nonzero()[0]] = 1
                g2plot = graphsub(gss4.copy(),vertex_filter=vf)

                coord = g2plot.vertex_property('coordinates')
                dts = g2plot.vertex_property('distance_to_surface')
                connectivity = g2plot.edge_connectivity()

                art = from_e_prop2_vprop(g2plot, 'artery')
                ve = from_e_prop2_vprop(g2plot, 'vein')

                label = g2plot.vertex_annotation();
                label_leveled = ano.convert_label(label, key='order', value='order', level=level + 1)

                artery_ep = []
                vein_ep = []
                for i in np.asarray(art == 1).nonzero()[0]:
                    # if '2/3' in ano.find_acronym(label_leveled[i], key='order'):
                    ns = g2plot.vertex_neighbours(i)
                    for n in ns:
                        if art[n] == 0:
                            artery_ep.append(i)
                            art_ep_dist_2_surface_control.append(dts[i])
                            break
                for i in np.asarray(ve == 1).nonzero()[0]:
                    # if '2/3' in ano.find_acronym(label_leveled[i], key='order'):
                    ns = g2plot.vertex_neighbours(i)
                    for n in ns:
                        if ve[n] == 0:
                            vein_ep.append(i)
                            ve_ep_dist_2_surface_control.append(dts[i])
                            break
                # print(len(artery_ep), len(vein_ep))

                path = []
                for a in artery_ep:
                    aep_c = coord[a]
                    for v in vein_ep:
                        l = 0
                        vep_c = coord[v]
                        g = g2plot.base
                        vlist, elist = gtt.shortest_path(g, g.vertex(a), g.vertex(v))
                        start = True
                        v_prop = np.zeros(g2plot.n_vertices)
                        for vl in vlist:
                            dist = dts[int(vl)]
                            path.append([int(vl), dist])
                            v_prop[int(vl)] = 1
                        if len(path) > 0:
                            ed = np.logical_and(v_prop[connectivity[:, 0]], v_prop[connectivity[:, 1]])
                            # print(np.array(path)[:, 1])
                            # print(np.sum(ed), vl, 'max depth :', np.max(np.array(path)[:, 1]))
                            ed = ed.nonzero()[0]
                            # ls = ano.find_acronym(label_leveled[int(vl)], key='order')
                            # layers = ['1', '2/3', '4', '5', '6a', '6b']
                            # for l, la in enumerate(layers):
                            #     if la in ls:
                            #         # print(la, ls)
                            #         path.append([int(vl), l])

                            for el in ed:
                                l = l + g2plot.edge_property('length')[el]

                            length_short_path.append(
                                [l, len(vlist), np.linalg.norm(vep_c - aep_c), np.max(np.array(path)[:, 1])])

            if len(path) > 0:
                u_path = np.unique(path, axis=0)
                # print(len(u_path), len(shortest_paths))
                # shortest_path.append(u_path)
                shortest_paths.append(u_path)
                # coi141.append(e)

        # print(len(loops_length), len(loops_size))
        loops_size_control.append([loops_length, loops_size])
        shortest_paths_control.append(shortest_paths)
        length_short_path_control.append(length_short_path)
    return(i,[loops_length,loops_size],vess_rad,bp_dist_2_surface,art_ep_dist_2_surface_control,ve_ep_dist_2_surface_control)

#%%

#
# def test():
#     #
#     # control_classification_ori=[]
#     # mutants_classification_ori=[]
#     #
#     # control_classification=[]
#     # mutants_classification=[]
#
#
#     control_barrels_ori=[]
#     mutants_barrels_ori=[]
#
#     control_barrels=[]
#     mutants_barrels=[]
#
#
#     control_auditory_ori=[]
#     mutants_auditory_ori=[]
#
#     control_auditory=[]
#     mutants_auditory=[]
#
#     regs=['barrel', 'auditory', 'motor', 'nose']#'auditory',
#     regs=['auditory']
#     # reg='Primary auditory'
#     for reg in regs:
#
#         control_barrels_ori = []
#         mutants_barrels_ori = []
#
#         control_barrels = []
#         mutants_barrels = []
#
#         control_auditory_ori = []
#         mutants_auditory_ori = []
#
#         control_auditory = []
#         mutants_auditory = []
#
#         if reg=='relays':
#             main_reg=(0,0)
#             regions = ['Inferior colliculus', 'lateral lemniscus', 'Superior olivary complex, lateral part', 'Cochlear nuclei']
#
#         elif reg=='auditory':
#             regions=[]
#             R='Primary auditory area'
#             main_reg=(ano.find_order('Primary auditory area', key='name'),ano.find_level('Primary auditory area', key='name'))
#             for r in reg_list.keys():
#                 n=ano.find_name(r, key='order')
#                 if R in n:
#                     for se in reg_list[r]:
#                         n = ano.find_name(se, key='order')
#                         print(n)
#                         regions.append(n)
#
#
#         elif reg=='motor':
#             regions=[]
#             R = 'Primary motor area'
#             main_reg=(ano.find_order('Primary motor area', key='name'),ano.find_level('Primary motor area', key='name'))
#             for r in reg_list.keys():
#                 n=ano.find_name(r, key='order')
#                 if R in n:
#                     for se in reg_list[r]:
#                         n = ano.find_name(se, key='order')
#                         print(n)
#                         regions.append(n)
#
#         elif reg=='nose':
#             regions=[]
#             R='Primary somatosensory area, nose'
#             main_reg=(ano.find_order('Primary somatosensory area, nose', key='name'),ano.find_level('Primary somatosensory area, nose', key='name'))
#             for r in reg_list.keys():
#                 n=ano.find_name(r, key='order')
#                 if R in n:
#                     for se in reg_list[r]:
#                         n = ano.find_name(se, key='order')
#                         print(n)
#                         regions.append(n)
#
#         elif reg == 'barrel':
#             regions = []
#             main_reg = (54, 9)
#             for r in reg_list.keys():
#                 n = ano.find_name(r, key='order')
#                 if reg in n:
#                     for se in reg_list[r]:
#                         n = ano.find_name(se, key='order')
#                         print(n)
#                         regions.append(n)
#
#         import ClearMap.IO.IO as io
#         print(regions)
#         atlas=io.read('/home/sophie.skriabine/Projects/clearvessel-custom/ClearMap/Resources/Atlas/annotation_25_full.nrrd')
#
#         ## extract density and orientation per regions
#         for g in controls:#controls
#           print(g)
#           graph=ggt.load(work_dir+'/'+g+'/'+'data_graph.gt')
#           label = graph.vertex_annotation();
#           label_leveled = ano.convert_label(label, key='order', value='order', level=main_reg[1])
#           vertex_filter = label_leveled == main_reg[0]  # 54;
#           graph = graph.sub_graph(vertex_filter=vertex_filter)
#           brain_barrel_ori=[]
#           brain_barrel = []
#           label = graph.vertex_annotation();
#           print(graph)
#           for region in regions:
#                 e=ano.find_order(region, key='name')
#                 try:
#                     vol=atlas_list[e]
#                 except(KeyError):
#                     sub_r=(e, ano.find_level(e, key='order'))
#                     vol=get_volume_region([sub_r], atlas)
#                     print(region, 'has sub regions')
#
#                 print(region)
#                 print(vol)
#                 label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
#                 vertex_filter = label_leveled == e  # 54;
#                 vess_tree=graph.sub_graph(vertex_filter=vertex_filter)
#                 vertex_filter = from_e_prop2_vprop(vess_tree, 'artery')
#                 art_tree = vess_tree.sub_graph(vertex_filter=vertex_filter)
#
#                 art_bp=art_tree.n_vertices/vol
#                 vess_bp=vess_tree.n_vertices/vol
#                 art_plan=[]
#                 art_rad=[]
#                 vess_plan=[]
#                 vess_rad=[]
#                 # vess_ori=getVesselOrientation(vess_tree, graph)
#                 # vess_plan.append(get_nb_parrallel_vessels(vess_ori) / vess_tree.n_edges)
#                 # vess_rad.append(get_nb_radial_vessels(vess_ori) / vess_tree.n_edges)
#                 r, p, l = getRadPlanOrienttaion(vess_tree, graph)
#                 r_f = r[~np.isnan(r)]
#                 # l = l[np.asarray(~np.isnan(r))]
#                 p = p[~np.isnan(p)]
#                 r =r_f
#                 rad = np.sum((r / (r + p)) > 0.5)
#                 plan = np.sum((p / (r + p)) > 0.6)
#                 # print(rad)
#                 # print(plan)
#                 vess_plan.append(plan / vess_tree.n_edges)
#                 vess_rad.append(rad / vess_tree.n_edges)
#                 print(rad / plan)
#                 if art_tree.n_vertices==0:
#                     art_rad.append(0)
#                 elif art_tree.n_edges==0:
#                     art_rad.append(0)
#                 else:
#                     r, p, l = getRadPlanOrienttaion(art_tree, graph)
#                     r = r[~np.isnan(r)]
#                     # l = l[np.asarray(~np.isnan(r))]
#                     p = p[~np.isnan(p)]
#                     rad = np.sum((r / (r + p)) > 0.5)
#                     plan = np.sum((p / (r + p)) > 0.6)
#                     print(rad/plan)
#                     # print(plan)
#                     art_plan.append(plan / art_tree.n_edges)
#                     art_rad.append(rad / art_tree.n_edges)
#                     # art_ori = getVesselOrientation(art_tree, graph)
#                     # art_plan.append(get_nb_parrallel_vessels(art_ori) / art_tree.n_edges)
#                     # art_rad.append(get_nb_radial_vessels(art_ori) / art_tree.n_edges)
#                 brain_barrel.append(art_bp)
#                 brain_barrel.append(vess_bp)
#                 brain_barrel_ori.append(art_plan)
#                 brain_barrel_ori.append(art_rad)
#                 brain_barrel_ori.append(vess_plan)
#                 brain_barrel_ori.append(vess_rad)
#           control_auditory_ori.append(brain_barrel_ori)
#           control_auditory.append(brain_barrel)
#
#         for g in mutants:
#             print(g)
#             graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph.gt')
#             label = graph.vertex_annotation();
#             label_leveled = ano.convert_label(label, key='order', value='order', level=main_reg[1])
#             vertex_filter = label_leveled == main_reg[0]  # 54;
#             graph = graph.sub_graph(vertex_filter=vertex_filter)
#             brain_barrel_ori = []
#             brain_barrel = []
#             label = graph.vertex_annotation();
#             print(graph)
#             for region in regions:
#                 e = ano.find_order(region, key='name')
#                 try:
#                     vol = atlas_list[e]
#                 except(KeyError):
#                     sub_r = (e, ano.find_level(e, key='order'))
#                     vol = get_volume_region([sub_r], atlas)
#                     print(region, 'has sub regions')
#
#                 print(region)
#                 print(vol)
#                 label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
#                 vertex_filter = label_leveled == e  # 54;
#                 vess_tree = graph.sub_graph(vertex_filter=vertex_filter)
#                 vertex_filter = from_e_prop2_vprop(vess_tree, 'artery')
#                 art_tree = vess_tree.sub_graph(vertex_filter=vertex_filter)
#
#                 art_bp=art_tree.n_vertices/vol
#                 vess_bp=vess_tree.n_vertices/vol
#                 art_plan = []
#                 art_rad = []
#                 vess_plan = []
#                 vess_rad = []
#                 # vess_ori = getVesselOrientation(vess_tree, graph)
#                 # vess_plan.append(get_nb_parrallel_vessels(vess_ori) / vess_tree.n_edges)
#                 # vess_rad.append(get_nb_radial_vessels(vess_ori) / vess_tree.n_edges)
#                 r, p, l = getRadPlanOrienttaion(vess_tree, graph)
#                 r = r[~np.isnan(r)]
#                 # l = l[np.asarray(~np.isnan(r))]
#                 p = p[~np.isnan(p)]
#                 rad = np.sum((r / (r + p)) > 0.5)
#                 plan = np.sum((p / (r + p)) > 0.6)
#                 print(rad/plan)
#                 # print(plan)
#                 vess_plan.append(plan / vess_tree.n_edges)
#                 vess_rad.append(rad / vess_tree.n_edges)
#
#                 if art_tree.n_vertices == 0:
#                     art_rad.append(0)
#                 elif art_tree.n_edges == 0:
#                     art_rad.append(0)
#                 else:
#                     r, p, l = getRadPlanOrienttaion(art_tree,graph)
#                     r = r[~np.isnan(r)]
#                     # l = l[np.asarray(~np.isnan(r))]
#                     p = p[~np.isnan(p)]
#                     rad = np.sum((r / (r + p)) > 0.5)
#                     plan = np.sum((p / (r + p)) > 0.6)
#                     print(rad/plan)
#                     # print(plan)
#                     art_plan.append(plan / art_tree.n_edges)
#                     art_rad.append(rad / art_tree.n_edges)
#                     # art_ori = getVesselOrientation(art_tree, graph)
#                     # art_plan.append(get_nb_parrallel_vessels(art_ori) / art_tree.n_edges)
#                     # art_rad.append(get_nb_radial_vessels(art_ori) / art_tree.n_edges)
#                 brain_barrel.append(art_bp)
#                 brain_barrel.append(vess_bp)
#                 brain_barrel_ori.append(art_plan)
#                 brain_barrel_ori.append(art_rad)
#                 brain_barrel_ori.append(vess_plan)
#                 brain_barrel_ori.append(vess_rad)
#             mutants_auditory_ori.append(brain_barrel_ori)
#             mutants_auditory.append(brain_barrel)
#
#
#         ## plot
#         art_pos=[0,2,4,6]
#         vess_pos=[1,3,5,7]
#
#         control_auditory_ori=np.array(control_auditory_ori)
#         mutants_auditory_ori=np.array(mutants_auditory_ori)
#
#         control_auditory=np.array(control_auditory)
#         mutants_auditory=np.array(mutants_auditory)
#
#         #
#         with open(work_dir+'/control_'+reg+'_ori.p', 'wb') as fp:
#           pickle.dump(control_auditory_ori, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#         with open(work_dir + '/control_'+reg+'.p', 'wb') as fp:
#           pickle.dump(control_auditory, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#         with open(work_dir + '/mutants_'+reg+'_ori.p', 'wb') as fp:
#           pickle.dump(mutants_auditory_ori, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#         with open(work_dir+'/mutants_'+reg+'.p', 'wb') as fp:
#           pickle.dump(mutants_auditory, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
#     import pickle
#
#     work_dir='/data_SSD_2to/191122Otof'#'/data_SSD_2to/whiskers_graphs'
#     # work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#
#
#     # regs=['auditory','barrel', 'motor', 'nose']#'auditory'
#     regs=['auditory']#'auditory'
#     ls=['l1', 'l2/3','l4','l5','l6a','l6b']
#     # reg='Primary auditory'
#     feature='bp'#bp
#
#     for reg in regs:
#
#         with open(work_dir+'/control_'+reg+'_ori.p', 'rb') as fp:
#           control_auditory_ori=pickle.load(fp)
#
#         with open(work_dir + '/control_'+reg+'.p', 'rb') as fp:
#           control_auditory=pickle.load(fp)
#
#         with open(work_dir + '/mutants_'+reg+'_ori.p', 'rb') as fp:
#           mutants_auditory_ori=pickle.load(fp)
#
#         with open(work_dir+'/mutants_'+reg+'.p', 'rb') as fp:
#           mutants_auditory=pickle.load(fp)
#
#
#
#
#         ## plot bpd
#
#         art_pos=[0,2,4,6, 8, 10]
#         vess_pos=[1,3,5,7, 9, 11]
#
#         if feature=='ori':
#             control=control_auditory_ori
#             mutant=mutants_auditory_ori
#
#         if feature=='bp':
#             control = control_auditory
#             mutant = mutants_auditory
#
#
#         if control.shape[1]==10:
#             art_pos =[0,2,4,6, 8]
#             vess_pos = [1,3,5,7, 9]
#
#         pos=vess_pos
#
#         import matplotlib.patches as mpatches
#         import pandas as pd
#         import seaborn as sns
#         labels=[]
#
#
#         ## plot densities all region together
#
#
#         vess_controls=control[:, pos]
#
#         C=[]
#         for i in range(vess_controls.shape[1]):
#             c=vess_controls[:,i]
#             C.append(c.tolist())
#
#
#         if len(np.array(C).shape)==3:
#             C=np.squeeze(np.array(C), axis=2)
#         else:
#             C=np.array(C)
#
#         dfc = pd.DataFrame(np.array(C).transpose()).melt()
#
#         # box1 = plt.boxplot(C, positions=np.arange(vess_controls.shape[1])+0.85, patch_artist=True, widths=0.3, showfliers=False, showmeans=True, autorange=True, meanline=True)
#         # for patch in box1['boxes']:
#         #     patch.set_facecolor(colors[1])
#         # col=box1['boxes'][0].get_facecolor()
#         # labels.append((mpatches.Patch(color=col), 'controls'))
#
#
#         vess_mutants=mutant[:, pos]
#         M=[]
#         for i in range(vess_mutants.shape[1]):
#             m=vess_mutants[:,i]
#             M.append(m.tolist())
#
#         if len(np.array(M).shape)==3:
#             M=np.squeeze(np.array(M), axis=2)
#         else:
#             M=np.array(M)
#
#         dfm = pd.DataFrame(np.array(M).transpose()).melt()
#
#         # plotRelativeValues(C.transpose(), M.transpose(), 'whiskers deprived '+ reg)
#
#
#         from scipy.stats import ttest_ind
#
#         for i in range(C.shape[0]):
#             gc = np.array(dfc.loc[lambda dfc: dfc['variable'] == i, 'value'])
#             gm = np.array(dfm.loc[lambda dfm: dfm['variable'] == i, 'value'])
#             st, pval=ttest_ind(gc, gm)
#             print(reg, ls[i], pval)
#             if pval<0.06:
#                 print(reg, ls[i], pval, '!!!!!!!!!!!!!!!!11')
#         plt.figure()
#         sns.set_style(style='white')
#         sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)
#         sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfm)
#         # box2 = plt.boxplot(M, positions=np.arange(vess_controls.shape[1])+1.15, patch_artist=True, widths=0.3, showfliers=False, showmeans=True, autorange=True, meanline=True)
#         # for patch in box2['boxes']:
#         #     patch.set_facecolor(colors[0])
#         # col=box2['boxes'][0].get_facecolor()
#         # labels.append((mpatches.Patch(color=col), 'mutants'))
#         sns.despine()
#         plt.ylabel('branch point density',size='x-large')
#         plt.xlabel('layers', size='x-large')
#         plt.xticks([0,1,2,3,4,5], ['l1', 'l2/3','l4','l5','l6a','l6b'], size='x-large')
#         plt.yticks(size='x-large')
#         plt.legend(['controls','Deprived'])#Otof-/-
#         plt.title('Branch Point Density '+reg, size='x-large')
#         # plt.legend(*zip(*labels), loc=2)
#         plt.legend(['controls','Deprived'])#Otof-/-
#         plt.tight_layout()
#     ## plot densities in differnet graph
#
#
#     li=[' ','Inferior colliculus','lateral lemniscus', 'Superior olivary complex, lateral part', 'Cochlear nuclei', ' ']
#     LI=[]
#     for l in li:
#         if l!=' ':
#             LI.append(ano.find_acronym(l, key='name'))
#         else:
#             LI.append(l)
#
#
#     plt.xticks([0,1,2,3,4,5], LI, size='medium')
#
#
#
#     vess_controls = control_classification[:, vess_pos]
#     C = []
#     for i in range(control_classification.shape[0]):
#         c = vess_controls[:, i]
#         C.append(c.tolist())
#
#     vess_controls = mutants_classification[:, vess_pos]
#     M = []
#     for i in range(mutants_classification.shape[0]):
#         m = vess_controls[:, i]
#         M.append(m.tolist())
#
#     for i in range(control_classification.shape[0]):
#         plt.figure()
#         box1 = plt.boxplot(C[i], positions=[0 + 0.75], patch_artist=True, widths=0.5, showfliers=False, showmeans=True, autorange=True, meanline=True)
#         box2 = plt.boxplot(M[i], positions=[0+1.25], patch_artist=True, widths=0.5, showfliers=False, showmeans=True, autorange=True, meanline=True)
#         for patch in box2['boxes']:
#             patch.set_facecolor(colors[0])
#         for patch in box1['boxes']:
#             patch.set_facecolor(colors[1])
#         # plt.scatter(np.arange(mutants_classification.shape[0])+1.5, m, color='k', zorder=2, alpha=0.05)
#
#         plt.ylabel('Nb branch points per volume unit')
#         plt.title(li[i+1])
#         plt.xticks([0, 1, 2], [' ',li[i+1], ' '], size='large')
#         plt.yticks(size='large')
#         plt.tight_layout()
#
#
#     ## plot orientations vessels
#
#
#
#     import pickle
#     regs=['auditory']#['auditory','barrel', 'motor', 'nose']#'auditory'
#     ls=['l1', 'l2/3','l4','l5','l6a','l6b']
#     # reg='Primary auditory'
#     for reg in regs:
#
#         with open(work_dir+'/control_'+reg+'_ori.p', 'rb') as fp:
#           control_auditory_ori=pickle.load(fp)
#
#         with open(work_dir + '/control_'+reg+'.p', 'rb') as fp:
#           control_auditory=pickle.load(fp)
#
#         with open(work_dir + '/mutants_'+reg+'_ori.p', 'rb') as fp:
#           mutants_auditory_ori=pickle.load(fp)
#
#         with open(work_dir+'/mutants_'+reg+'.p', 'rb') as fp:
#           mutants_auditory=pickle.load(fp)
#
#
#         control=control_auditory_ori
#         mutant=mutants_auditory_ori
#         #
#
#
#         pos=vess_pos
#
#         art_plan_pos = [0,4,8, 12, 16, 20]
#         art_rad_pos  = [1,5,9, 13, 17, 21]
#         vess_plan_pos= [2,6,10,14, 18, 22]
#         vess_rad_pos = [3,7,11,15, 19, 23]
#
#         if control.shape[1]==20:
#             art_plan_pos = [0, 4, 8, 12, 16]
#             art_rad_pos = [1, 5, 9, 13, 17]
#             vess_plan_pos = [2, 6, 10, 14, 18]
#             vess_rad_pos = [3, 7, 11, 15, 19]
#         # vess_pos=[2,3,6,7, 10, 11, 14, 15]
#
#
#
#         control_=np.zeros(control.shape)
#         for i in range(control.shape[0]):
#             for j in range(control.shape[1]):
#                 if control[i, j]==[]:
#                     control_[i, j]=0
#                 else:
#                     control_[i, j]=control[i, j][0]
#
#
#         mutant_=np.zeros(mutant.shape)
#         for i in range(mutant.shape[0]):
#             for j in range(mutant.shape[1]):
#                 if mutant[i, j]==[]:
#                     mutant_[i, j]=0
#                 else:
#                     mutant_[i, j]=mutant[i, j][0]
#
#         mutant=mutant_
#         control=control_
#
#         import matplotlib.patches as mpatches
#         labels=['controls','Otof-/-']
#
#         plt.figure()
#         sns.set_style(style='white')
#
#         vess_controls=control[:, vess_rad_pos]/control[:, vess_plan_pos]
#         C=[]
#         for i in range(vess_controls.shape[1]):
#             c=vess_controls[:,i]
#             C.append(c.tolist())
#
#
#         if len(np.array(C).shape)==3:
#             C=np.squeeze(np.array(C), axis=2)
#         else:
#             C=np.array(C)
#
#
#         dfc = pd.DataFrame(np.array(C)[:,:].transpose()).melt()
#         p1=sns.lineplot(x="variable", y="value", err_style="bars",ci='sd',data=dfc)
#         # box1 = plt.boxplot(C, positions=np.arange(vess_controls.shape[1])+0.85, patch_artist=True, widths=0.3, showfliers=False, showmeans=True, autorange=True, meanline=True)
#         # for patch in box1['boxes']:
#         #     patch.set_facecolor(colors[1])
#         # col=p1['boxes'][0].get_facecolor()
#         # labels.append((mpatches.Patch(color=col), 'controls'))
#
#         vess_controls=mutant[:, vess_rad_pos]/mutant[:, vess_plan_pos]
#         M=[]
#         for i in range(vess_controls.shape[1]):
#             m=vess_controls[:,i]
#             M.append(m.tolist())
#
#         if len(np.array(M).shape)==3:
#             M=np.squeeze(np.array(M), axis=2)
#         else:
#             M=np.array(M)
#
#         dfm = pd.DataFrame(M[:,:].transpose()).melt()
#         p2=sns.lineplot(x="variable", y="value", err_style="bars",ci='sd',data=dfm)
#         # box2 = plt.boxplot(M, positions=np.arange(vess_controls.shape[1])+1.15, patch_artist=True, widths=0.3, showfliers=False, showmeans=True, autorange=True, meanline=True)
#         # for patch in box2['boxes']:
#         #     patch.set_facecolor(colors[0])
#         # col=p2['boxes'][0].get_facecolor()
#         # labels.append((mpatches.Patch(color=col), 'mutants'))
#
#
#         from scipy.stats import ttest_ind
#
#         for i in range(C.shape[0]):
#             gc = np.array(dfc.loc[lambda dfc: dfc['variable'] == i, 'value'])
#             gm = np.array(dfm.loc[lambda dfm: dfm['variable'] == i, 'value'])
#             st, pval = ttest_ind(gc, gm)
#             print(reg, ls[i], pval)
#             if pval < 0.06:
#                 print(reg, ls[i], pval,  '!!!!!!!!!!!!!!!')
#
#         plt.ylabel('orientation rad/plan',size='x-large')
#         plt.xlabel('layers', size='x-large')
#         ## plot densities in differnet graph
#
#         sns.despine()
#         plt.xticks([0,1,2,3,4,5], ['l1', 'l2/3','l4','l5','l6a','l6b'], size='x-large')
#
#         plt.yticks(size='x-large')
#         plt.legend(['controls','Deprived'])#Otof-/-
#         plt.title('orientation '+reg, size='x-large')
#         plt.tight_layout()
#     # plt.legend(*zip(*labels), loc=2)
#     # new_labels = ['label 1', 'label 2']
#     # for t, l in zip(plt._legend.texts, labels): t.set_text(l)
#
#     li=[' ','Inferior colliculus','lateral lemniscus', 'Superior olivary complex, lateral part', 'Cochlear nuclei', ' ']
#     LI=[]
#     for l in li:
#         if l!=' ':
#             LI.append(ano.find_acronym(l, key='name'))
#         else:
#             LI.append(l)
#
#     # plt.scatter(np.arange(mutants_classification.shape[0])+1.5, m, color='k', zorder=2, alpha=0.05)
#     plt.xticks([0,1,2,3,4,5], LI, size='medium')
#     plt.tight_layout()
#
#     plt.title('vessel_orientation')
#
#
#     ## plot orientations arteries
#     plt.figure()
#     vess_controls=control_classification_ori[:, art_rad_pos, 0]/control_classification_ori[:, art_plan_pos, 0]
#     C=[]
#     for i in range(control_classification_ori.shape[0]):
#         c=vess_controls[:,i]
#         C.append(c.tolist())
#
#     box1 = plt.boxplot(C, positions=np.arange(control_classification_ori.shape[0])+0.75, patch_artist=True, widths=0.5, showfliers=False, showmeans=True, autorange=True, meanline=True)
#     for patch in box1['boxes']:
#         patch.set_facecolor(colors[1])
#
#     # plt.scatter(np.arange(control_classification.shape[0])+0.5, c, color='k', zorder=2, alpha=0.05)
#
#     vess_controls=mutants_classification_ori[:, art_rad_pos, 0]/mutants_classification_ori[:, art_plan_pos, 0]
#     M=[]
#     for i in range(mutants_classification_ori.shape[0]):
#         m=vess_controls[:,i]
#         M.append(m.tolist())
#
#     box2 = plt.boxplot(M, positions=np.arange(mutants_classification_ori.shape[0])+1.25, patch_artist=True, widths=0.5, showfliers=False, showmeans=True, autorange=True, meanline=True)
#     for patch in box2['boxes']:
#         patch.set_facecolor(colors[0])
#
#
#     li=[' ','Inferior colliculus','lateral lemniscus', 'Superior olivary complex, lateral part', 'Cochlear nuclei', ' ']
#     LI=[]
#     for l in li:
#         if l!=' ':
#             LI.append(ano.find_acronym(l, key='name'))
#         else:
#             LI.append(l)
#
#     # plt.scatter(np.arange(mutants_classification.shape[0])+1.5, m, color='k', zorder=2, alpha=0.05)
#     plt.xticks([0,1,2,3,4,5], LI, size='medium')
#     plt.tight_layout()
#
#     plt.title('arteries orientation')
#
#
#     ## heatmaps
#     # controls=['2R','3R','5R', '8R']
#     # mutants=['1R','7R', '6R', '4R']
#     # work_dir='/data_SSD_2to/191122Otof'
#     #
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls=['142L','158L','162L', '164L']
#     mutants=['138L','141L', '163L', '165L']
#
#     import ClearMap.Analysis.Measurements.Voxelization as vox
#     import ClearMap.IO.IO as io
#     template_shape=(320,528,228)
#     vox_shape=(320,528,228, len(controls))
#     vox_control=np.zeros(vox_shape)
#     vox_mutant=np.zeros(vox_shape)
#     radius=5
#
#     for i,control in enumerate(controls):
#         print(control)
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph.gt')
#         coordinates=graph.vertex_property('coordinates_atlas')#*1.625/25
#         v=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
#         vox_control[:,:,:,i]=v
#
#     for i,mutant in enumerate(mutants):
#         print(mutant)
#         graph = ggt.load(work_dir + '/' + mutant + '/' + 'data_graph.gt')
#         coordinates=graph.vertex_property('coordinates_atlas')#*1.625/25
#         v=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
#         vox_mutant[:,:,:,i]=v
#
#
#
#     io.write(work_dir + '/' +'vox_control'+str(radius)+'.tif', vox_control.astype('float32'))
#     io.write(work_dir + '/' +'vox_mutant'+str(radius)+'.tif', vox_mutant.astype('float32'))
#
#     vox_control_avg=np.mean(vox_control[:, :, :, 1:], axis=3)
#     vox_mutant_avg=np.mean(vox_mutant[:, :, :, :-1], axis=3)
#
#
#     io.write(work_dir + '/' +'vox_control_avg_'+str(radius)+'.tif', vox_control_avg.astype('float32'))
#     io.write(work_dir + '/' +'vox_mutant_avg_'+str(radius)+'.tif', vox_mutant_avg.astype('float32'))
#
#     from scipy import stats
#     pcutoff = 0.05
#
#     tvals, pvals = stats.ttest_ind(vox_control[:, :, :, 1:], vox_mutant[:, :, :, :-1], axis = 3, equal_var = False);
#
#     pi = np.isnan(pvals);
#     pvals[pi] = 1.0;
#     tvals[pi] = 0;
#
#     pvals2 = pvals.copy();
#     pvals2[pvals2 > pcutoff] = pcutoff;
#     psign=np.sign(tvals)
#
#
#     ## from sagital to coronal view
#     pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
#     psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
#     # pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
#
#     # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
#     pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])
#
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
#
#     import tifffile
#     tifffile.imsave(work_dir+'/pvalcolors_BP density_'+str(radius)+'.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)
#
#
#     ## arteriesBP
#     #
#     #
#     # controls=['2R','3R','5R', '8R']
#     # mutants=['1R','7R', '6R', '4R']
#     # work_dir='/data_SSD_2to/191122Otof'
#
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls=['142L','158L','162L', '164L']
#     mutants=['138L','141L', '163L', '165L']
#
#     import ClearMap.Analysis.Measurements.Voxelization as vox
#     import ClearMap.IO.IO as io
#     template_shape=(320,528,228)
#     vox_shape=(320,528,228, len(controls))
#     vox_art_control=np.zeros(vox_shape)
#     vox_art_mutant=np.zeros(vox_shape)
#     radius=10
#
#     for i,control in enumerate(controls):
#         print(control)
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         graph=graph.sub_graph(vertex_filter=from_e_prop2_vprop(graph,'artery')>0)
#         coordinates=graph.vertex_property('coordinates_atlas')#*1.625/25
#         v=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
#         vox_art_control[:,:,:,i]=v
#
#     for i,mutant in enumerate(mutants):
#         print(mutant)
#         graph = ggt.load(work_dir + '/' + mutant + '/' + 'data_graph_correctedIsocortex.gt')
#         graph = graph.sub_graph(vertex_filter=from_e_prop2_vprop(graph, 'artery') > 0)
#         coordinates=graph.vertex_property('coordinates_atlas')#*1.625/25
#         v=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(radius,radius,radius), method = 'sphere');
#         vox_art_mutant[:,:,:,i]=v
#
#     vox_control_avg=np.mean(vox_art_control, axis=3)
#     vox_mutant_avg=np.mean(vox_art_mutant, axis=3)
#
#     for i in range(len(controls)):
#         io.write(work_dir +'/' + 'vox_arteries_' + controls[i] + '.tif', vox_art_control[:, :, :, i].astype('float32'))
#
#     for i in range(len(mutants)):
#         io.write(work_dir + '/' + 'vox_arteries_' + mutants[i] + '.tif', vox_art_mutant[:, :, :, i].astype('float32'))
#
#
#
#     io.write(work_dir + '/' +'vox_control_arteries_avg_'+str(radius)+'.tif', vox_control_avg.astype('float32'))
#     io.write(work_dir + '/' +'vox_mutant_arteries_avg_'+str(radius)+'.tif', vox_mutant_avg.astype('float32'))
#
#     from scipy import stats
#     pcutoff = 0.05
#
#     tvals, pvals = stats.ttest_ind(vox_art_control[:, :, :, :-1], vox_art_mutant, axis = 3, equal_var = True);
#
#     pi = np.isnan(pvals);
#     pvals[pi] = 1.0;
#     tvals[pi] = 0;
#
#     pvals2 = pvals.copy();
#     pvals2[pvals2 > pcutoff] = pcutoff;
#     psign=np.sign(tvals)
#
#
#     ## from sagital to coronal view
#     pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
#     psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
#     # pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
#
#     # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
#     pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])
#
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
#
#     import tifffile
#     tifffile.imsave(work_dir+'/pvalcolors_arteries_BP density_'+str(radius)+'.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)
#
#
#     ## artery domains size
#     #
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls=['142L','158L','162L', '164L']
#     mutants=['138L','141L', '163L', '165L']
#
#     # controls=['2R','3R','5R', '8R']
#     # mutants=['1R','7R', '6R', '4R']
#     # work_dir='/data_SSD_2to/191122Otof'
#
#     condition='barrels'
#     art_terr_control=[]
#     art_terr_mutant=[]
#
#
#     for i,control in enumerate(controls):
#         print(control)
#         gss4 = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         art_clusters = np.load(
#             work_dir + '/' + control + '/sbm/diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
#         gss4.add_vertex_property('artterr', art_clusters)
#         if condition == 'Aud_p':
#             region_list = [(142, 8)]  # auditory
#         elif condition == 'Aud':
#             region_list = [(127, 7)]  # barrels
#         elif condition == 'Ssp':
#             region_list = [(40, 8)]  # barrels
#         elif condition == 'barrels':
#             region_list = [(54, 9)]  # barrels
#         else:
#             region_list = [(6, 6)]  # isocortex
#
#         order, level = region_list[0]
#
#         print(level, order, ano.find_name(order, key='order'))
#
#         label = gss4.vertex_annotation();
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         vertex_filter = label_leveled == order;
#
#         gss4 = gss4.sub_graph(vertex_filter=vertex_filter)
#         art_clusters=gss4.vertex_property('artterr')
#         u, c=np.unique(art_clusters, return_counts=True)
#         art_terr_control.append(c)
#
#
#     for i,control in enumerate(mutants):
#         print(control)
#         gss4 = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         art_clusters = np.load(
#             work_dir + '/' + control + '/sbm/diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected.npy')
#         gss4.add_vertex_property('artterr', art_clusters)
#         if condition == 'Aud_p':
#             region_list = [(142, 8)]  # auditory
#         elif condition == 'Aud':
#             region_list = [(127, 7)]  # barrels
#         elif condition == 'Ssp':
#             region_list = [(40, 8)]  # barrels
#         elif condition == 'barrels':
#             region_list = [(54, 9)]  # barrels
#         else:
#             region_list = [(6, 6)]  # isocortex
#
#         order, level = region_list[0]
#
#         print(level, order, ano.find_name(order, key='order'))
#
#         label = gss4.vertex_annotation();
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         vertex_filter = label_leveled == order;
#
#         gss4 = gss4.sub_graph(vertex_filter=vertex_filter)
#         art_clusters=gss4.vertex_property('artterr')
#         u, c=np.unique(art_clusters, return_counts=True)
#         art_terr_mutant.append(c)
#
#
#     art_terr_mutant=np.array(art_terr_mutant)
#     art_terr_control = np.array(art_terr_control)
#
#     np.save(work_dir+'/art_terr_control'+condition+'.npy',art_terr_control)
#     np.save(work_dir+'/art_terr_mutant'+condition+'.npy',art_terr_mutant)
#
#
#     ## PLOT DISTRIBUTION nB OF CYCLES PER AGGREGATE
#     step=100
#     max=1000
#     normed=False
#
#     for a, m in enumerate(art_terr_control):
#         if a==0:
#             hist,bins = np.histogram(np.array(m),bins = np.arange(0, max, step), normed=normed)
#             C=hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             C=np.concatenate((C, hist.reshape((hist.shape[0], 1))), axis=1)
#
#     for a, m in enumerate(art_terr_mutant[1:]):
#         if a == 0:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             M = hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             M = np.concatenate((M, hist.reshape((hist.shape[0], 1))), axis=1)
#     #
#     # M = M.reshape((M.shape[0], 1))
#     # C = C.reshape((C.shape[0], 1))
#     # data=[C, M]
#
#     # data=pd.DataFrame(np.array(data).transpose()).melt()
#     C=pd.DataFrame(np.array(C).transpose()).melt()
#     M=pd.DataFrame(np.array(M).transpose()).melt()
#
#
#     plt.figure()
#     import pandas as pd
#     import seaborn as sns
#
#     sns.set_style(style='white')
#     sns.lineplot(x="variable", y="value", ci='sd', data=C)#, y="normalized count"
#     sns.lineplot(x="variable", y="value", ci='sd', data=M)#, err_style="bars"
#
#     sns.despine()
#
#     plt.legend(['controls', 'mutants'])
#     plt.title('size in BP of artery territories in ' + condition, size='x-large')
#
#     plt.xlabel("size in BP of artery territories")
#     plt.ylabel(" count")
#     plt.xticks(range(np.arange(0, max, step).shape[0]),np.arange(0, max, step))
#     plt.tight_layout()
#
#
#
#
#
#
#
#
#     ## length arteries
#     #
#     #
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls=['142L','158L','162L', '164L']
#     mutants=['138L','141L', '163L', '165L']
#
#     # controls=['2R','3R','5R', '8R']
#     # mutants=['1R','7R', '6R', '4R']
#     # work_dir='/data_SSD_2to/191122Otof'
#
#     condition='barrels'
#     length_art_control=[]
#     length_art_mutant=[]
#
#     length_art_control_BP=[]
#     length_art_mutant_BP=[]
#
#     for i,control in enumerate(controls):
#         print(control)
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         gss4=graph.sub_graph(vertex_filter=from_e_prop2_vprop(graph,'artery')>0)
#         if condition == 'Aud_p':
#             region_list = [(142, 8)]  # auditory
#         elif condition == 'Aud':
#             region_list = [(127, 7)]  # barrels
#         elif condition == 'Ssp':
#             region_list = [(40, 8)]  # barrels
#         elif condition == 'barrels':
#             region_list = [(54, 9)]  # barrels
#         else:
#             region_list = [(6, 6)]  # isocortex
#
#         order, level = region_list[0]
#
#         print(level, order, ano.find_name(order, key='order'))
#
#         label = gss4.vertex_annotation();
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         vertex_filter = label_leveled == order;
#
#         gss4 = gss4.sub_graph(vertex_filter=vertex_filter)
#
#         distance_from_suface = gss4.vertex_property('distance_to_surface')
#         print('distance_to_surface ; ', distance_from_suface.shape)
#
#         penetrating_arteries = distance_from_suface > 1
#         print(penetrating_arteries.shape)
#         art_grt = gss4.sub_graph(vertex_filter=penetrating_arteries)
#         print(graph)
#         print(art_grt)
#         connectivity = art_grt.edge_connectivity()
#         length=art_grt.edge_property('length')
#         p_a_labels = gtt.label_components(art_grt.base)
#         p_a_labels_hist = p_a_labels[1]
#         p_a_labels = p_a_labels[0]
#
#         p_a_labels = ggt.vertex_property_map_to_python(p_a_labels, as_array=True)
#         u_l, c_l = np.unique(p_a_labels, return_counts=True)
#         lengths=[]
#         for j, v in enumerate(u_l):
#             # print(i, ' / ', len(u_l))
#             if c_l[j]>3:
#                 vprop=p_a_labels==v
#                 e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
#                 lengths.append(np.sum(length[e_prop]))
#         u_l, c_l = np.unique(p_a_labels, return_counts=True)
#         print('nb of labelled componenets ; ', np.unique(p_a_labels), p_a_labels.shape)
#         length_art_control_BP.append(c_l[u_l!=-1])
#         length_art_control.append(np.array(lengths))
#
#     for i, control in enumerate(mutants):
#         print(control)
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         gss4 = graph.sub_graph(vertex_filter=from_e_prop2_vprop(graph, 'artery') > 0)
#         if condition == 'Aud_p':
#             region_list = [(142, 8)]  # auditory
#         elif condition == 'Aud':
#             region_list = [(127, 7)]  # barrels
#         elif condition == 'Ssp':
#             region_list = [(40, 8)]  # barrels
#         elif condition == 'barrels':
#             region_list = [(54, 9)]  # barrels
#         else:
#             region_list = [(6, 6)]  # isocortex
#
#         order, level = region_list[0]
#
#         print(level, order, ano.find_name(order, key='order'))
#
#         label = gss4.vertex_annotation();
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         vertex_filter = label_leveled == order;
#
#         gss4 = gss4.sub_graph(vertex_filter=vertex_filter)
#
#         distance_from_suface = gss4.vertex_property('distance_to_surface')
#         print('distance_to_surface ; ', distance_from_suface.shape)
#
#         penetrating_arteries = distance_from_suface > 1
#         print(penetrating_arteries.shape)
#         art_grt = gss4.sub_graph(vertex_filter=penetrating_arteries)
#         print(graph)
#         print(art_grt)
#         connectivity = art_grt.edge_connectivity()
#         length=art_grt.edge_property('length')
#         p_a_labels = gtt.label_components(art_grt.base)
#         p_a_labels_hist = p_a_labels[1]
#         p_a_labels = p_a_labels[0]
#
#         p_a_labels = ggt.vertex_property_map_to_python(p_a_labels, as_array=True)
#         u_l, c_l = np.unique(p_a_labels, return_counts=True)
#         lengths=[]
#         for j, v in enumerate(u_l):
#             # print(i, ' / ', len(u_l))
#             if c_l[j] > 3:
#                 vprop = p_a_labels == v
#                 e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
#                 lengths.append(np.sum(length[e_prop]))
#         u_l, c_l = np.unique(p_a_labels, return_counts=True)
#         print('nb of labelled componenets ; ', np.unique(p_a_labels), p_a_labels.shape)
#         length_art_mutant_BP.append(c_l[u_l!=-1])
#         length_art_mutant.append(np.array(lengths))
#
#     length_art_control=np.array(length_art_control)
#     length_art_mutant = np.array(length_art_mutant)
#
#     length_art_mutant_BP=np.array(length_art_mutant_BP)
#     length_art_control_BP = np.array(length_art_control_BP)
#
#     np.save(work_dir+'/length_art_control_'+condition+'.npy',length_art_control)
#     np.save(work_dir+'/length_art_mutant'+condition+'.npy',length_art_mutant)
#
#     np.save(work_dir+'/length_art_control_BP'+condition+'.npy',length_art_control_BP)
#     np.save(work_dir+'/length_art_mutant_BP'+condition+'.npy',length_art_mutant_BP)
#
#
#
#
#
#     #
#     # condition='Aud'
#     # length_art_control=np.load(work_dir+'/length_art_control_'+condition+'.npy')
#     # length_art_mutant=np.load(work_dir+'/length_art_mutant'+condition+'.npy')
#     #
#     # length_art_control_BP=np.load(work_dir+'/length_art_control_BP'+condition+'.npy')
#     # length_art_mutant_BP=np.load(work_dir+'/length_art_mutant_BP'+condition+'.npy')
#
#     ## PLOT DISTRIBUTION nB OF CYCLES PER AGGREGATE
#     step=100
#     max=1000
#     normed=False
#
#     for a, m in enumerate(length_art_control):
#         if a==0:
#             hist,bins = np.histogram(np.array(m),bins = np.arange(0, max, step), normed=normed)
#             C=hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             C=np.concatenate((C, hist.reshape((hist.shape[0], 1))), axis=1)
#
#     for a, m in enumerate(length_art_mutant[1:]):
#         if a == 0:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             M = hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             M = np.concatenate((M, hist.reshape((hist.shape[0], 1))), axis=1)
#     #
#     # M = M.reshape((M.shape[0], 1))
#     # C = C.reshape((C.shape[0], 1))
#     # data=[C, M]
#
#     # data=pd.DataFrame(np.array(data).transpose()).melt()
#     C=pd.DataFrame(np.array(C).transpose()).melt()
#     M=pd.DataFrame(np.array(M).transpose()).melt()
#
#
#     plt.figure()
#     import pandas as pd
#     import seaborn as sns
#
#     sns.set_style(style='white')
#     sns.lineplot(x="variable", y="value", ci='sd', data=C)#, y="normalized count"
#     sns.lineplot(x="variable", y="value", ci='sd', data=M)#, err_style="bars"
#
#     sns.despine()
#
#     plt.legend(['controls', 'mutants'])
#     plt.title('lengths of penetrating arteries in ' + condition, size='x-large')
#
#     plt.xlabel("lengths of penetrating penetrating arteries")
#     plt.ylabel(" count")
#     plt.xticks(range(np.arange(0, max, step).shape[0]),np.arange(0, max, step))
#     plt.tight_layout()
#     plt.yscale('log')
#
#
#
#
#
#
#
#
#
#     ## ARTRIES RAW VALUES  DISTRIBUTION
#
#     normed=True
#     condition='SSs'
#     EP=True
#     A_r_c=[]
#     # rad_c=[]
#     A_r_d=[]
#     # rad_d=[]
#     bin=np.linspace(9,16,50)
#     for a, control in enumerate(controls):
#         print(control)
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#
#         dist=graph.vertex_property('distance_to_surface')
#         vf=dist>3
#         graph = graph.sub_graph(vertex_filter=vf)
#
#         art_sig_all = graph.vertex_property('artery_raw')
#         if condition == 'Aud_p':
#             region_list = [(142, 8)]  # auditory
#         elif condition == 'Aud':
#             region_list = [(127, 7)]  # barrels
#         elif condition == 'Ssp':
#             region_list = [(40, 8)]  # barrels
#         elif condition == 'barrels':
#             region_list = [(54, 9), (47, 9), (75, 9)]  # barrels
#         elif condition == 'l2 barrels':
#             region_list = [(56, 10), (49, 10), (77, 10)]  # barrels
#         elif condition == 'l4 barrels':
#             region_list = [(58, 10), (51, 10), (79, 10)]  # barrels
#         elif condition == 'SSs':
#             region_list = [(103,8)]  # barrels
#         else:
#             region_list = [(6, 6)]  # isocortex
#         vertex_filter=np.zeros(graph.n_vertices)
#         for i, rl in enumerate(region_list):
#             order, level = region_list[i]
#             print(level, order, ano.find(order, key='order')['name'])
#             label = graph.vertex_annotation();
#             label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#             vertex_filter[label_leveled == order]=1;
#         graph = graph.sub_graph(vertex_filter=vertex_filter)
#
#         if EP:
#             artery=from_e_prop2_vprop(graph, 'artery')
#             art_graph = graph.sub_graph(vertex_filter=artery)
#             art_EP=art_graph.vertex_degrees()==1
#             for i in np.asarray(art_EP==1).nonzero()[0]:
#                 art_EP[art_graph.vertex_neighbours(i)]=1
#             graph = art_graph.sub_graph(vertex_filter=art_EP)
#             artery = from_e_prop2_vprop(graph, 'artery')
#         radii=graph.vertex_radii()
#         a_r = (graph.vertex_property('artery_raw') - np.mean(art_sig_all)) / np.std(art_sig_all)
#         a_r=a_r+10
#         a_r=a_r[artery.astype('bool')]
#         if a==0:
#             hist,bins = np.histogram(a_r,bins = bin, normed=normed)
#             C=hist.reshape((hist.shape[0], 1))
#             A_r_c.append([a_r, radii])
#         else:
#             hist, bins = np.histogram(a_r, bins=bin, normed=normed)
#             C=np.concatenate((C, hist.reshape((hist.shape[0], 1))), axis=1)
#             A_r_c.append([a_r, radii])
#         print(control, bins)
#
#     for a, control in enumerate(mutants):
#         print(control)
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#
#         dist = graph.vertex_property('distance_to_surface')
#         vf = dist > 3
#         graph = graph.sub_graph(vertex_filter=vf)
#
#         art_sig_all = graph.vertex_property('artery_raw')
#         if condition == 'Aud_p':
#             region_list = [(142, 8)]  # auditory
#         elif condition == 'Aud':
#             region_list = [(127, 7)]  # barrels
#         elif condition == 'Ssp':
#             region_list = [(40, 8)]  # barrels
#         elif condition == 'barrels':
#             region_list = [(54, 9), (47, 9), (75, 9)]  # barrels
#         elif condition == 'l2 barrels':
#             region_list = [(56, 10), (49, 10), (77, 10)]  # barrels
#         elif condition == 'l4 barrels':
#             region_list = [(58, 10), (51, 10), (79, 10)]  # barrels
#         elif condition == 'SSs':
#             region_list = [(103,8)]  # barrels
#         else:
#             region_list = [(6, 6)]  # isocortex
#         vertex_filter = np.zeros(graph.n_vertices)
#         for i, rl in enumerate(region_list):
#             order, level = region_list[i]
#             print(level, order, ano.find(order, key='order')['name'])
#             label = graph.vertex_annotation();
#             label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#             vertex_filter[label_leveled == order] = 1;
#         graph = graph.sub_graph(vertex_filter=vertex_filter)
#
#         if EP:
#             artery = from_e_prop2_vprop(graph, 'artery')
#             art_graph = graph.sub_graph(vertex_filter=artery)
#             art_EP = art_graph.vertex_degrees() == 1
#             for i in np.asarray(art_EP == 1).nonzero()[0]:
#                 art_EP[art_graph.vertex_neighbours(i)] = 1
#             graph = art_graph.sub_graph(vertex_filter=art_EP)
#             artery = from_e_prop2_vprop(graph, 'artery')
#         a_r = (graph.vertex_property('artery_raw') - np.mean(art_sig_all)) / np.std(art_sig_all)
#         a_r = a_r + 10
#         a_r = a_r[artery.astype('bool')]
#         radii = graph.vertex_radii()
#         if a == 0:
#             hist, bins = np.histogram(a_r, bins=bin, normed=normed)
#             M = hist.reshape((hist.shape[0], 1))
#             A_r_d.append([a_r, radii])
#         else:
#             hist, bins = np.histogram(a_r, bins=bin, normed=normed)
#             M = np.concatenate((M, hist.reshape((hist.shape[0], 1))), axis=1)
#             A_r_d.append([a_r, radii])
#         print(control, bins)
#
#     # M = M.reshape((M.shape[0], 1))
#     # C = C.reshape((C.shape[0], 1))
#     # data=[C, M]
#
#     # data=pd.DataFrame(np.array(data).transpose()).melt()
#     Cpd=pd.DataFrame(np.array(C).transpose()).melt()
#     Mpd=pd.DataFrame(np.array(M).transpose()).melt()
#
#     #
#     # plt.figure()
#     # for a in A_r_d:
#     #     plt.scatter(a[0], a[1], color='indianred', alpha=0.1)
#     # for a in A_r_c:
#     #     plt.scatter(a[0], a[1], color='cadetblue', alpha=0.1)
#     # plt.legend(['controls', 'mutants'])
#     # plt.title('artery_raw VS radii ' + condition, size='x-large')
#     #
#     # plt.xlabel("artery_raw value", size='x-large')
#     # plt.ylabel(" radius", size='x-large')
#     # plt.yticks(size='x-large')
#     # plt.xticks(np.arange(0,50, 5), np.arange(0, np.max(bins), np.max(bins)/10).astype(int), rotation=45,size='x-large')
#     # plt.tight_layout()
#     # plt.yscale('linear')
#
#
#
#     plt.figure()
#     import pandas as pd
#     import seaborn as sns
#
#     sns.set_style(style='white')
#     sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd)#, y="normalized count"
#     sns.lineplot(x="variable", y="value", err_style='bars', data=Mpd)#, err_style="bars"
#
#     sns.despine()
#
#     plt.legend(['controls', 'mutants'])
#     plt.title('artery_raw ' + condition, size='x-large')
#
#     plt.xlabel("artery_raw value", size='x-large')
#     plt.ylabel(" count", size='x-large')
#     plt.yticks(size='x-large')
#     plt.xticks(np.arange(0,50, 5), np.arange(0, np.max(bins), np.max(bins)/10).astype(int), rotation=45,size='x-large')
#     plt.tight_layout()
#     plt.yscale('linear')
#
#     for c in C.transpose():
#         # cpd=pd.DataFrame(np.array(c).transpose()).melt()
#         # sns.lineplot(x="variable", y="value", ci='sd', style="event", data=cpd, color='cadetblue')
#         plt.plot(np.arange(19), c ,  '--' ,color='cadetblue', alpha=0.3)
#
#     for m in M.transpose():
#         # cpd=pd.DataFrame(np.array(m).transpose()).melt()
#         # sns.lineplot(x="variable", y="value", ci='sd', style="event", data=cpd, color='indianred')
#         plt.plot(np.arange(19), m ,'--', color='indianred', alpha=0.3)
#
#
#     ## PLOT SHORTEST PATHS
#
#     u, c = np.unique(indices, return_counts=True)
#     coi = np.logical_and(np.asarray(c >= 200), np.asarray(c <= 1500)).nonzero()[0]
#     import random
#
#     e = random.choice(coi141)
#     print(e)
#     # print(r)
#     #
#     # e = u[r]
#     # e=coi141[]
#     # print(u[r], c[r])
#     vf = np.zeros(gss4.n_vertices)
#     vf[np.asarray(indices == e).nonzero()[0]] = 1
#     c = g2plot.vertex_property('cluster')
#     c = c[0]
#     ## get extended overlap
#     vf = gss4.expand_vertex_filter(vf, steps=3)
#     print(gss4, vf.shape, 'containing ', np.sum(vf), ' vertices')
#     g2plot = gss4.sub_graph(vertex_filter=vf)
#     g2plot = g2plot.largest_component()
#     art = np.logical_and(cluster[:, 0] == c[0], artery)
#     ve = np.logical_and(cluster[:, 1] == c[1], vein)
#     print(np.sum(art), np.sum(ve))
#     vf = vf * 3
#     vf[np.asarray(art == 1).nonzero()[0]] = 1
#     vf[np.asarray(ve == 1).nonzero()[0]] = 2
#     gss4.add_vertex_property('vf', vf)
#     print(gss4, vf.shape, 'containing ', np.sum(vf > 0), ' vertices + art + vein')
#     g2plot = gss4.sub_graph(vertex_filter=vf > 0)
#     g2plot = g2plot.largest_component()
#     vfilt = g2plot.vertex_property('vf')
#     art = from_e_prop2_vprop(g2plot, 'artery')
#     ve = from_e_prop2_vprop(g2plot, 'vein')
#     label = g2plot.vertex_annotation();
#     label_leveled = ano.convert_label(label, key='order', value='order', level=level + 1)
#     artery_ep = []
#     vein_ep = []
#     for i in np.asarray(art == 1).nonzero()[0]:
#         if '2/3' in ano.find_acronym(label_leveled[i], key='order'):
#             ns = g2plot.vertex_neighbours(i)
#             for n in ns:
#                 if art[n] == 0:
#                     artery_ep.append(i)
#                     break
#     for i in np.asarray(ve == 1).nonzero()[0]:
#         if '2/3' in ano.find_acronym(label_leveled[i], key='order'):
#             ns = g2plot.vertex_neighbours(i)
#             for n in ns:
#                 if ve[n] == 0:
#                     vein_ep.append(i)
#                     break
#     print(len(artery_ep), len(vein_ep))
#     colorVal = np.zeros((vfilt.shape[0], 4))
#     red_blue_map = {1: [1.0, 0, 0, 1.0], 3: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0], 4: [0.0, 0.0, 0.0, 1.0]}
#     for i, c in enumerate(vfilt):
#         # print(j)
#         if c == 1:
#             colorVal[i] = red_blue_map[1]
#         elif c == 2:
#             colorVal[i] = red_blue_map[2]
#         elif c == 3:
#             colorVal[i] = red_blue_map[3]
#         elif c == 4:
#             colorVal[i] = red_blue_map[4]
#     path = []
#     for a in artery_ep:
#         for v in vein_ep:
#             g = g2plot.base
#             vlist, elist = gtt.shortest_path(g, g.vertex(a), g.vertex(v))
#             for vl in vlist:
#                 colorVal[int(vl)] = red_blue_map[4]
#                 path.append([int(vl)])
#
#     vfilt[np.asarray(art == 1).nonzero()[0]] = 1
#     vfilt[np.asarray(ve == 1).nonzero()[0]] = 2
#     for i, c in enumerate(vfilt):
#         # print(j)
#         if c == 1:
#             colorVal[i] = red_blue_map[1]
#         elif c == 2:
#             colorVal[i] = red_blue_map[2]
#     if len(path) > 0:
#         print('plotting')
#         p = p3d.plot_graph_mesh(g2plot, vertex_colors=colorVal)
#     else:
#         print('no art/vein in layer2/3')
#
#
#     ## ARTERIES RAW PER PNETRATING ARTERY DISTRIBUTION
#     #
#     # from tqdm import tqdm
#     from ClearMap.GraphEmbedding import *
#     from ClearMap.DiffusionPenetratingArteriesCortex import diffusion_through_penetrating_arteries,get_penetration_veins_dustance_surface,get_penetrating_veins_labels
#     # from ClearMap.GraphMP import *
#
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls=['142L','158L','162L', '164L']
#     mutants=['138L','141L', '163L', '165L']
#     # #
#     # controls=['2R','3R','5R', '8R']
#     # mutants=['1R','7R', '6R', '4R']
#     # work_dir='/data_SSD_2to/191122Otof'
#     # #
#     work_dir='/data_SSD_2to/whiskers_graphs/fluoxetine'
#     mutants=['1','2','3', '4', '6', '18']
#     controls=['21','22', '23']
#
#
#
#     ls=['l1', 'l2/3','l4','l5','l6a','l6b']
#
#
#     mode='clusters'#'layers'
#     ext_step=7
#     compute_distance=False
#     compute_loops=False
#     compute_path=False
#     basis = CreateSimpleBasis(3, 7)
#     sub_region=False
#     condition='isocortex'#'isocortex'#'isocortex'#'barrel_region'#'Auditory_regions'
#     feature='vessels'#'vessels'#art_raw_signal
#
#
#
#     if condition == 'Aud_p':
#         region_list = [(142, 8)]  # auditory
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             n = ano.find_name(r, key='order')
#             if R in n:
#                 for se in reg_list[r]:
#                     n = ano.find_name(se, key='order')
#                     print(n)
#                     regions.append(n)
#     elif condition == 'Aud':
#         region_list = [(127, 7)]  # barrels
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             n = ano.find_name(r, key='order')
#             if R in n:
#                 for se in reg_list[r]:
#                     n = ano.find_name(se, key='order')
#                     print(n)
#                     regions.append(n)
#     if condition == 'Aud_po':
#         region_list = [(149, 8)]  # auditory
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             n = ano.find_name(r, key='order')
#             if R in n:
#                 for se in reg_list[r]:
#                     n = ano.find_name(se, key='order')
#                     print(n)
#                     regions.append(n)
#     if condition == 'Aud_d':
#         region_list = [(128, 8)]  # auditory
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             n = ano.find_name(r, key='order')
#             if R in n:
#                 for se in reg_list[r]:
#                     n = ano.find_name(se, key='order')
#                     print(n)
#                     regions.append(n)
#     if condition == 'Aud_v':
#         region_list = [(156, 8)]  # auditory
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             n = ano.find_name(r, key='order')
#             if R in n:
#                 for se in reg_list[r]:
#                     n = ano.find_name(se, key='order')
#                     print(n)
#                     regions.append(n)
#     elif condition == 'Ssp':
#         region_list = [(40, 8)]  # barrels
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             n = ano.find_name(r, key='order')
#             if R in n:
#                 for se in reg_list[r]:
#                     n = ano.find_name(se, key='order')
#                     print(n)
#                     regions.append(n)
#     elif condition == 'barrels':
#         region_list = [(54, 9)]  # barrels
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             n = ano.find_name(r, key='order')
#             if R in n:
#                 for se in reg_list[r]:
#                     n = ano.find_name(se, key='order')
#                     print(n)
#                     regions.append(n)
#     elif condition == 'nose':
#         region_list = [(47, 9)]  # barrels
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             n = ano.find_name(r, key='order')
#             if R in n:
#                 for se in reg_list[r]:
#                     n = ano.find_name(se, key='order')
#                     print(n)
#                     regions.append(n)
#     elif condition == 'mouth':
#         region_list = [(75, 9)]  # barrels
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             n = ano.find_name(r, key='order')
#             if R in n:
#                 for se in reg_list[r]:
#                     n = ano.find_name(se, key='order')
#                     print(n)
#                     regions.append(n)
#     elif condition == 'Auditory_regions':
#         regions = [[(142, 8), (149, 8), (128, 8), (156, 8)]]
#         sub_region = True
#     elif condition == 'barrel_region':
#         regions = [[(54, 9), (47, 9)]]  # , (75, 9)]  # barrels
#         sub_region = True
#     elif condition == 'l2 barrels':
#         regions = [(56, 10), (49, 10), (77, 10)]  # barrels
#     elif condition == 'l4 barrels':
#         regions = [(58, 10), (51, 10), (79, 10)]  # barrels
#     elif condition == 'isocortex':
#         region_list = [(6, 6)]  # isocortex
#         regions = []
#         R = ano.find(region_list[0][0], key='order')['name']
#         main_reg = region_list
#         sub_region = True
#         for r in reg_list.keys():
#             l = ano.find(r, key='order')['level']
#             regions.append([(r, l)])
#
#
#
#
#
#     NbLoops_control=[]
#     NbLoops_mutant=[]
#
#     loopsLayer_control=[]
#     loopsLayer_mutant=[]
#
#     raw_signal_control=[]
#     raw_signal_mutant=[]
#     # condition='barrels'
#     nb_artery_control=[]
#     nb_artery_mutant=[]
#
#     bp_artery_control=[]
#     bp_artery_mutant=[]
#
#     bp_layer_artery_control=[]
#     bp_layer_artery_mutant=[]
#
#     ep_layer_artery_control=[]
#     ep_layer_artery_mutant=[]
#
#     vess_rad_control=[]
#     vess_rad_mutant=[]
#
#     art_length_control=[]
#     art_length_mutant=[]
#
#
#     distances_control=[]
#     distances_mutant=[]
#
#     loopsLayerLength_control=[]
#     loopsLayerLength_mutant=[]
#
#
#     loops_size_mutant = []
#     loops_size_control = []
#
#     shortest_paths_control = []
#     shortest_paths_mutant = []
#
#     length_short_path_control_control=[]
#     length_short_path_mutant_mutant=[]
#
#     art_ep_dist_2_surface_control_control=[]
#     ve_ep_dist_2_surface_control_control=[]
#
#     art_ep_dist_2_surface_mutant_mutant=[]
#     ve_ep_dist_2_surface_mutant_mutant=[]
#
#     bp_dist_2_surface_control=[]
#     bp_dist_2_surface_mutant=[]
#
#     shortest_paths_control_control=[]
#     loops_size_control_control=[]
#
#     shortest_paths_mutant_mutant=[]
#     loops_size_mutant_mutant=[]
#
#     # #
#     # def Test(args):
#     #     i=args
#     #     print(i)
#     #     graph = ggt.load(work_dir + '/' + i + '/' + 'data_graph_correctedIsocortex.gt')
#     #     g = graph.copy()
#     #     vertex_filter = np.zeros(graph.n_vertices)
#     #     for i, rl in enumerate(region_list):
#     #         order, level = region_list[i]
#     #         print(level, order, ano.find(order, key='order')['name'])
#     #         label = graph.vertex_annotation();
#     #         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#     #         vertex_filter[label_leveled == order] = 1;
#     #     g.base.set_vertex_filter(vertex_property_map_from_python(vertex_filter, g, dtype=bool))
#     #     # # self.base.set_edge_filter(edge_property_map_from_python(edge_filter, self, dtype=bool))
#     #     print('create graph1')
#     #     g = gt.Graph(g.base, prune=False)
#     #     print('create graph2')
#     #     g = ggt.Graph(base=g);
#     #     try:
#     #         print('resizing')
#     #         resize_edge_geometry(g);
#     #         print('done')
#     #     except(IndexError):
#     #         print(IndexError)
#     #     # g = graphsub(graph.copy(), vertex_filter=vertex_filter, edge_filter=None)
#     #     return g
#
#     # def parrallelGraphFeatureExtraction(args):
#         # i, region_list, graph, feature, compute_distance, mode, compute_loops, basis = args
#
#     import multiprocessing
#     for a, control in enumerate(controls):
#         # i=1
#         # control=controls[i]
#         print(control)
#         length_short_path_control = []
#         ep = []
#         bp = []
#         vess_rad = []
#         loopslengthBrain=[]
#         bp_dist_2_surface = []
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_corrected_Isocortex.gt')
#         diff = np.load(work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
#         graph.add_vertex_property('overlap', diff)
#         cluster = getOverlaps(work_dir, control, graph, save_overlap=False)
#         graph.add_vertex_property('cluster', cluster)
#         art_rp_2_surf = []
#         ve_rp_2_surf = []
#
#         # region_list=regions[0]
#         # ep = [None for _ in range(len(regions))]
#         # bp = [None for _ in range(len(regions))]
#         # vess_rad = [None for _ in range(len(regions))]
#         # loopslengthBrain=[None for _ in range(len(regions))]
#         # bp_dist_2_surface = [None for _ in range(len(regions))]
#         # args=[]
#         # for i,c in enumerate(controls):#enumerate(regions):
#         #     args.append((c))
#         #     # args.append((i, graph, region_list, feature, compute_distance, mode, compute_loops, basis))
#         # pool = multiprocessing.Pool(10)
#         # out=pool.map(Test, [arg for arg in args])#zip(*pool.map(parrallelGraphFeatureExtraction, [arg for arg in args]))#zip(*pool.map(parrallelGraphFeatureExtraction, [arg for arg in args]))
#
#         for region_list in regions:
#             length_short_path = []
#
#             art_ep_dist_2_surface_control = []
#             ve_ep_dist_2_surface_control = []
#             shortest_paths_control=[]
#             loops_size_control=[]
#
#             vertex_filter = np.zeros(graph.n_vertices)
#             for i, rl in enumerate(region_list):
#                 order, level = region_list[i]
#                 print(level, order, ano.find(order, key='order')['name'])
#                 label = graph.vertex_annotation();
#                 label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#                 vertex_filter[label_leveled == order] = 1;
#             gss4_t = graph.sub_graph(vertex_filter=vertex_filter)
#             # gtest=graph_t.graphsub(vertex_filter=vertex_filter)
#
#             if compute_distance:
#                 distance_control = diffusion_through_penetrating_arteries(gss4_t, get_penetration_veins_dustance_surface,
#                                                                            get_penetrating_veins_labels, vesseltype='art',
#                                                                            graph_dir=work_dir + '/' + control, feature='distance')
#                 distances_control.append(distance_control)
#
#
#             if feature=='vessels':
#                 gss4=gss4_t.copy()
#                 bp_dist_2_surface.append(gss4.vertex_property('distance_to_surface'))
#
#             elif feature=='art_raw_signal':
#                 artery = from_e_prop2_vprop(gss4_t, 'artery')
#                 vertex_filter=np.logical_and(artery,gss4_t.vertex_property('artery_binary')>0)#np.logical_and()
#                 gss4 = gss4_t.sub_graph(vertex_filter=vertex_filter)
#
#                 distance_from_suface = gss4.vertex_property('distance_to_surface')
#                 # print('distance_to_surface ; ', distance_from_suface.shape)
#
#                 penetrating_arteries = distance_from_suface > 1
#                 print(penetrating_arteries.shape)
#                 art_grt = gss4.sub_graph(vertex_filter=penetrating_arteries)
#                 # standardization
#                 a_r = (art_grt.vertex_property('artery_raw') - np.mean(art_grt.vertex_property('artery_raw'))) / np.std(art_grt.vertex_property('artery_raw'))
#                 a_r=a_r+10
#                 print(gss4_t)
#                 print(art_grt)
#                 connectivity = art_grt.edge_connectivity()
#                 length = art_grt.edge_property('length')
#                 p_a_labels = gtt.label_components(art_grt.base)
#                 p_a_labels_hist = p_a_labels[1]
#                 p_a_labels = p_a_labels[0]
#
#                 p_a_labels = ggt.vertex_property_map_to_python(p_a_labels, as_array=True)
#                 u_l, c_l = np.unique(p_a_labels, return_counts=True)
#                 nb_artery_control.append(u_l.shape[0])
#                 bp_artery_control.append(c_l)
#
#                 raw_signal = []
#                 lengths = []
#                 for j, v in enumerate(u_l):
#                     # print(i, ' / ', len(u_l))
#                     if c_l[j] > 3:
#                         vprop = p_a_labels == v
#                         e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
#                         r_signal = np.sum(a_r[vprop]) / np.sum(vprop)
#                         raw_signal.append(r_signal)
#                         lengths.append(np.sum(length[e_prop]))
#                 raw_signal_control.append(np.array(raw_signal))
#                 art_length_control.append(lengths)
#
#             label = gss4.vertex_annotation();
#             connectivity = gss4.edge_connectivity()
#             NbLoops = 0
#             loopsLayer=[]
#             # mode='layers'
#             if mode=='layers':
#                 if sub_region:
#                     layers = ['1', '2/3', '4', '5', '6a', '6b']
#                     # layers = ['4']
#                     for b,layer in enumerate(layers):
#                         loopsLayerLength = []
#                         vertex_filter = np.zeros(gss4.n_vertices)
#                         for i, rl in enumerate(region_list):
#                             R = ano.find(region_list[i][0], key='order')['name']
#                             for r in reg_list.keys():
#                                 n = ano.find_name(r, key='order')
#                                 if R in n:
#                                     for se in reg_list[r]:
#                                         if layer in ano.find(se, key='order')['name']:
#                                             l = ano.find(se, key='order')['level']
#                                             print(ano.find(se, key='order')['name'], se)
#                                             label_leveled = ano.convert_label(label, key='order', value='order', level=l)
#                                             vertex_filter[label_leveled == se] = 1
#                         vess_tree = gss4.sub_graph(vertex_filter=vertex_filter)
#                         dts = vess_tree.vertex_property('distance_to_surface')
#                         # plt.figure()
#                         # plt.hist(dts)
#                         # plt.title(layer+' '+control)
#
#
#                         # connectivity = vess_tree.edge_connectivity()
#                         # lengths = vess_tree.edge_geometry_lengths('length')
#                         #
#                         degree = vess_tree.vertex_degrees()
#                         ep.append(np.sum(degree == 1))
#                         bp.append(np.sum(degree >= 3))
#
#                         if feature == 'vessels':
#
#                             r, p, l = getRadPlanOrienttaion(vess_tree, gss4_t, True, True)
#                             r_f = r[~np.isnan(r)]
#                             # l = l[np.asarray(~np.isnan(r))]
#                             p = p[~np.isnan(p)]
#                             r = r_f
#                             rad = np.sum((r / (r + p)) > 0.5)
#                             plan = np.sum((p / (r + p)) > 0.6)
#                             # print(rad)
#                             # print(plan)
#                             # vess_plan.append(plan / vess_tree.n_edges)
#                             vess_rad.append(rad / vess_tree.n_edges)
#
#                             ## loops
#                             # loops = []
#                             # for i, b in enumerate(basis):
#                             #     # if i >= 3:
#                             #     res = gtt.subgraph_isomorphism(b.base, vess_tree.base, induced=True)
#                             #     res = checkUnicityOfElement(res)
#                             #     loops.append(len(res))
#                             #     NbLoops = NbLoops + len(res)
#                             #     lenghtsloops=[]
#                             #     # check distance length of loops
#                             #     for r in res:
#                             #         v_res = np.zeros(vess_tree.n_vertices)
#                             #         v_res[r] = 1
#                             #         loops_edges = np.logical_and(v_res[connectivity[:, 0]], v_res[connectivity[:, 1]])
#                             #         lenghtsloops.append(np.sum(lengths[loops_edges]))
#                             #     loopsLayerLength.append(lenghtsloops)
#                             #     print(control, layer, i,  len(loopsLayerLength))
#                         # loopslengthBrain.append(loopsLayerLength)
#                         # loopsLayer.append(loops)
#
#
#                         # for region in regions:
#                         #     e = ano.find_order(region, key='name')
#                         #     label = gss4.vertex_annotation();
#                         #     print(region, e)
#                         #     label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
#                         #     # print(np.unique(label_leveled))
#                         #     vertex_filter = label_leveled == e  # 54;
#                         #     print(np.sum(vertex_filter))
#                         #     vess_tree = gss4.sub_graph(vertex_filter=vertex_filter)
#                         #     degree = vess_tree.vertex_degrees()
#                         #     ep.append(np.sum(degree == 1))
#                         #     bp.append(np.sum(degree >= 3))
#                     ep_layer_artery_control.append(ep)
#                     bp_layer_artery_control.append(bp)
#                     vess_rad_control.append(vess_rad)
#                     NbLoops_control.append(NbLoops)
#                     loopsLayerLength_control.append(loopslengthBrain)
#                     # print(len(loopsLayerLength_control[a][b]))
#                     loopsLayer_control.append(loopsLayer)
#
#             elif mode=='clusters':
#
#                 r, p, l = getRadPlanOrienttaion(gss4, gss4_t, local_normal=True, calc_art=True)
#                 r_f = np.nan_to_num(r)
#                 d = gss4.edge_property('distance_to_surface')
#                 # l = l[np.asarray(~np.isnan(r))]
#                 p = np.nan_to_num(p)
#                 # print(rad)
#                 # print(plan)
#                 # vess_plan.append(plan / vess_tree.n_edges)
#                 vess_rad.append(np.concatenate((r_f, d), axis=0))
#
#                 loops_length=[]
#                 loops_size=[]
#                 cluster=gss4.vertex_property('cluster')
#                 artery = from_e_prop2_vprop(gss4, 'artery')
#                 vein = from_e_prop2_vprop(gss4, 'vein')
#                 shortest_paths=[]
#                 indices = gss4.vertex_property('overlap')
#                 u, c = np.unique(indices, return_counts=True)
#                 n = 0
#                 j_u = 0
#                 # for i, e in tqdm(enumerate(u)):
#                 for i in tqdm(range(u.shape[0])):
#                     e=u[i]
#                     j = np.asarray(u == e).nonzero()[0][0]
#                     # print(j.shape)
#                     vf = np.zeros(gss4.n_vertices)
#                     vf[np.asarray(indices == e).nonzero()[0]] = 1
#
#                     ## get extended overlap
#                     vf = gss4.expand_vertex_filter(vf, steps=ext_step)
#                     # print(gss4, vf.shape, 'containing ', np.sum(vf), ' vertices')
#
#                     g2plot = gss4.sub_graph(vertex_filter=vf)
#                     c = g2plot.vertex_property('cluster')
#                     c = c[0]
#                     g2plot = g2plot.largest_component()
#                     if g2plot.n_vertices <= 3000 and g2plot.n_edges >= 5:
#                         print(g2plot, vf.shape, 'containing ', np.sum(vf), ' vertices')
#                         connectivity = g2plot.edge_connectivity()
#                         lengths = g2plot.edge_geometry_lengths('length')
#                         # shortest_path = []
#                         if compute_loops:
#                             for i, b in enumerate(basis):
#                                 # if i >= 3:
#                                 res = gtt.subgraph_isomorphism(b.base, g2plot.base, induced=True)
#                                 res = checkUnicityOfElement(res)
#
#                                 # check distance length of loops
#                                 for r in res:
#                                     v_res = np.zeros(g2plot.n_vertices)
#                                     v_res[r] = 1
#                                     loops_edges = np.logical_and(v_res[connectivity[:, 0]], v_res[connectivity[:, 1]])
#                                     loops_length.append(np.sum(lengths[loops_edges]))
#                                     loops_size.append(i+3)
#
#                         ## shortest path
#
#                         art = np.logical_and(cluster[:, 0] == c[0], artery)
#                         ve = np.logical_and(cluster[:, 1] == c[1], vein)
#                         vf[np.asarray(art == 1).nonzero()[0]] = 1
#                         vf[np.asarray(ve == 1).nonzero()[0]] = 1
#                         g2plot = gss4.sub_graph(vertex_filter=vf)
#
#
#                         coord=g2plot.vertex_property('coordinates')
#                         dts=g2plot.vertex_property('distance_to_surface')
#                         connectivity = g2plot.edge_connectivity()
#
#                         art = from_e_prop2_vprop(g2plot, 'artery')
#                         ve = from_e_prop2_vprop(g2plot, 'vein')
#
#                         label = g2plot.vertex_annotation();
#                         label_leveled = ano.convert_label(label, key='order', value='order', level=level + 1)
#
#                         artery_ep = []
#                         vein_ep = []
#                         for i in np.asarray(art == 1).nonzero()[0]:
#                             # if '2/3' in ano.find_acronym(label_leveled[i], key='order'):
#                             ns = g2plot.vertex_neighbours(i)
#                             for n in ns:
#                                 if art[n] == 0:
#                                     artery_ep.append(i)
#                                     art_ep_dist_2_surface_control.append(dts[i])
#                                     break
#                         for i in np.asarray(ve == 1).nonzero()[0]:
#                             # if '2/3' in ano.find_acronym(label_leveled[i], key='order'):
#                             ns = g2plot.vertex_neighbours(i)
#                             for n in ns:
#                                 if ve[n] == 0:
#                                     vein_ep.append(i)
#                                     ve_ep_dist_2_surface_control.append(dts[i])
#                                     break
#                         # print(len(artery_ep), len(vein_ep))
#
#                         path = []
#                         if compute_path:
#                             for a in artery_ep:
#                                 aep_c=coord[a]
#                                 for v in vein_ep:
#                                     l=0
#                                     vep_c = coord[v]
#                                     g = g2plot.base
#                                     vlist, elist = gtt.shortest_path(g, g.vertex(a), g.vertex(v))
#                                     start=True
#                                     v_prop = np.zeros(g2plot.n_vertices)
#                                     for vl in vlist:
#                                         dist = dts[int(vl)]
#                                         path.append([int(vl), dist])
#                                         v_prop[int(vl)] = 1
#                                     if len(path) > 0:
#                                         ed = np.logical_and(v_prop[connectivity[:, 0]], v_prop[connectivity[:, 1]])
#                                         # print(np.array(path)[:, 1])
#                                         # print(np.sum(ed), vl, 'max depth :', np.max(np.array(path)[:, 1]))
#                                         ed = ed.nonzero()[0]
#                                         # ls = ano.find_acronym(label_leveled[int(vl)], key='order')
#                                         # layers = ['1', '2/3', '4', '5', '6a', '6b']
#                                         # for l, la in enumerate(layers):
#                                         #     if la in ls:
#                                         #         # print(la, ls)
#                                         #         path.append([int(vl), l])
#
#                                         for el in ed:
#                                             l = l + g2plot.edge_property('length')[el]
#
#                                         length_short_path.append(
#                                             [l, len(vlist), np.linalg.norm(vep_c - aep_c), np.max(np.array(path)[:, 1])])
#
#                         if len(path) > 0:
#                             u_path = np.unique(path, axis=0)
#                             # print(len(u_path), len(shortest_paths))
#                             # shortest_path.append(u_path)
#                             shortest_paths.append(u_path)
#                             # coi141.append(e)
#
#
#                 # print(len(loops_length), len(loops_size))
#                 loops_size_control.append([loops_length,loops_size])
#                 shortest_paths_control.append(shortest_paths)
#                 length_short_path_control.append(length_short_path)
#                 art_rp_2_surf.append(art_ep_dist_2_surface_control)
#                 ve_rp_2_surf.append(ve_ep_dist_2_surface_control)
#
#
#
#         length_short_path_control_control.append(length_short_path_control)
#         vess_rad_control.append(vess_rad)
#         bp_dist_2_surface_control.append(bp_dist_2_surface)
#         art_ep_dist_2_surface_control_control.append(art_rp_2_surf)
#         ve_ep_dist_2_surface_control_control.append(ve_rp_2_surf)
#         shortest_paths_control_control.append(shortest_paths_control)
#         loops_size_control_control.append(loops_size_control)
#
#         if feature=='art_raw_signal':
#             feat='art_bp_layer_artery'
#         elif feature=='vessels':
#             feat='vess_bp_layer_vessels'
#
#         np.save(work_dir+'/'+feat+'_control_'+condition+'.npy',bp_layer_artery_control)
#         if feature=='art_raw_signal':
#             feat='art_ori_layer_artery'
#         elif feature=='vessels':
#             feat='vess_ori_layer_vessels'
#         np.save(work_dir+'/'+feat+'_control_'+condition+'.npy',vess_rad_control)
#
#         np.save(work_dir + '/' + 'art_ep_dist_2_surface_control' + condition +  '_'+control+'.npy', art_ep_dist_2_surface_control)
#         np.save(work_dir + '/' + 've_ep_dist_2_surface_control' + condition +  '_'+control+'.npy', ve_ep_dist_2_surface_control)
#
#
#
#         np.save(work_dir + '/' + 'length_short_path_control' + condition + '_'+control+ '.npy', length_short_path_control)
#         # np.save(work_dir + '/' + 'length_short_path_mutant' + condition + '_'+control+ '.npy', length_short_path_mutant_mutant)
#
#         np.save(work_dir + '/' + 'shortest_paths_control' + condition + '_'+control+ '.npy', shortest_paths_control)
#         # np.save(work_dir + '/' + 'shortest_paths_mutant' + condition + '_'+control+ '.npy', shortest_paths_mutant_mutant)
#
#         # np.save(work_dir + '/' + 'art_ep_dist_2_surface_mutant' + condition + '.npy', art_ep_dist_2_surface_mutant_mutant)
#         # np.save(work_dir + '/' + 've_ep_dist_2_surface_mutant' + condition + '.npy', ve_ep_dist_2_surface_mutant_mutant)
#         np.save(work_dir + '/' + 'art_ep_dist_2_surface_control' + condition +  '_'+control+'.npy', art_ep_dist_2_surface_control)
#         np.save(work_dir + '/' + 've_ep_dist_2_surface_control' + condition +  '_'+control+'.npy', ve_ep_dist_2_surface_control)
#
#         np.save(work_dir + '/' + feature + 'control_rad_ori' + condition +  '_'+control+'.npy', vess_rad)#_control
#         np.save(work_dir + '/' + 'bp_dist_2_surface_control' + condition + '_'+control+ '.npy', bp_dist_2_surface)#_control
#
#
#
#     for a, control in enumerate(mutants):
#
#
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_corrected_Isocortex.gt')
#         diff = np.load(work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
#         graph.add_vertex_property('overlap', diff)
#         cluster=getOverlaps(work_dir, control, graph, save_overlap=False)
#         graph.add_vertex_property('cluster', cluster)
#         ep = []
#         bp = []
#         vess_rad=[]
#         length_short_path_mutant = []
#
#         loopslengthBrain=[]
#         bp_dist_2_surface=[]
#         art_rp_2_surf=[]
#         ve_rp_2_surf = []
#
#
#         for region_list in regions:
#             length_short_path = []
#             art_ep_dist_2_surface_mutant = []
#             ve_ep_dist_2_surface_mutant = []
#             shortest_paths_mutant=[]
#             loops_size_mutant=[]
#
#             vertex_filter = np.zeros(graph.n_vertices)
#             for i, rl in enumerate(region_list):
#                 order, level = region_list[i]
#                 print(level, order, ano.find(order, key='order')['name'])
#                 label = graph.vertex_annotation();
#                 label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#                 vertex_filter[label_leveled == order] = 1;
#             gss4_t = graph.sub_graph(vertex_filter=vertex_filter)
#
#
#             if compute_distance:
#                 distance_mutant = diffusion_through_penetrating_arteries(graph,
#                                                                           get_penetration_veins_dustance_surface,
#                                                                           get_penetrating_veins_labels, vesseltype='art',
#                                                                           graph_dir=work_dir + '/' + control, feature='distance')
#                 distances_mutant.append(distance_mutant)
#
#             if feature == 'vessels':
#                 gss4 = gss4_t.copy()
#                 bp_dist_2_surface.append(gss4.vertex_property('distance_to_surface'))
#
#             elif feature == 'art_raw_signal':
#                 artery = from_e_prop2_vprop(gss4_t, 'artery')
#                 vertex_filter = np.logical_and(artery, gss4_t.vertex_property('artery_binary') > 0)  # np.logical_and()
#                 gss4 = gss4_t.sub_graph(vertex_filter=vertex_filter)
#
#                 distance_from_suface = gss4.vertex_property('distance_to_surface')
#                 print('distance_to_surface ; ', distance_from_suface.shape)
#
#                 penetrating_arteries = distance_from_suface > 1
#                 print(penetrating_arteries.shape)
#                 art_grt = gss4.sub_graph(vertex_filter=penetrating_arteries)
#                 # standardization
#                 a_r = (art_grt.vertex_property('artery_raw') - np.mean(art_grt.vertex_property('artery_raw'))) / np.std(
#                     art_grt.vertex_property('artery_raw'))
#                 a_r = a_r + 10
#                 print(gss4_t)
#                 print(art_grt)
#                 connectivity = art_grt.edge_connectivity()
#                 length = art_grt.edge_property('length')
#                 p_a_labels = gtt.label_components(art_grt.base)
#                 p_a_labels_hist = p_a_labels[1]
#                 p_a_labels = p_a_labels[0]
#
#                 p_a_labels = ggt.vertex_property_map_to_python(p_a_labels, as_array=True)
#                 u_l, c_l = np.unique(p_a_labels, return_counts=True)
#                 nb_artery_mutant.append(u_l.shape[0])
#                 bp_artery_mutant.append(c_l)
#
#                 raw_signal = []
#                 lengths=[]
#                 for j, v in enumerate(u_l):
#                     # print(i, ' / ', len(u_l))
#                     if c_l[j] > 3:
#                         vprop = p_a_labels == v
#                         e_prop = np.logical_and(vprop[connectivity[:, 0]], vprop[connectivity[:, 1]])
#                         r_signal=np.sum(a_r[vprop])/np.sum(vprop)
#                         raw_signal.append(r_signal)
#                         lengths.append(np.sum(length[e_prop]))
#                 raw_signal_mutant.append(np.array(raw_signal))
#                 art_length_mutant.append(lengths)
#
#             label = gss4.vertex_annotation();
#             connectivity = gss4.edge_connectivity()
#
#             if mode=='layers':
#                 if sub_region:
#                     layers=['1', '2/3', '4', '5', '6a', '6b']
#                     NbLoops=0
#                     loopsLayer = []
#
#                     for layer in layers:
#                         loopsLayerLength = []
#                         vertex_filter = np.zeros(gss4.n_vertices)
#                         for i, rl in enumerate(region_list):
#                             R = ano.find(region_list[i][0], key='order')['name']
#                             for r in reg_list.keys():
#                                 n = ano.find_name(r, key='order')
#                                 if R in n:
#                                     for se in reg_list[r]:
#                                         if layer in ano.find(se, key='order')['name']:
#                                             l = ano.find(se, key='order')['level']
#                                             print(ano.find(se, key='order')['name'], se)
#                                             label_leveled = ano.convert_label(label, key='order', value='order',level=l)
#                                             vertex_filter[label_leveled == se]=1
#                         vess_tree = gss4.sub_graph(vertex_filter=vertex_filter)
#                         dts = vess_tree.vertex_property('distance_to_surface')
#                         # plt.figure()
#                         # plt.hist(dts)
#                         # plt.title(layer + ' ' + control)
#
#
#
#                         # connectivity = vess_tree.edge_connectivity()
#                         # lengths=vess_tree.edge_geometry_lengths('length')
#                         #
#                         degree = vess_tree.vertex_degrees()
#                         ep.append(np.sum(degree == 1))
#                         bp.append(np.sum(degree >= 3))
#
#                         if feature == 'vessels':
#                             r, p, l = getRadPlanOrienttaion(vess_tree, graph, True, True)
#                             r_f = r[~np.isnan(r)]
#                             # l = l[np.asarray(~np.isnan(r))]
#                             p = p[~np.isnan(p)]
#                             r = r_f
#                             rad = np.sum((r / (r + p)) > 0.5)
#                             plan = np.sum((p / (r + p)) > 0.6)
#                             # print(rad)
#                             # print(plan)
#                             # vess_plan.append(plan / vess_tree.n_edges)
#                             vess_rad.append(rad / vess_tree.n_edges)
#                             ## loops
#                             loops=[]
#                             if compute_loops:
#                                 for i, b in enumerate(basis):
#                                     # if i >= 3:
#                                     lenghtsloops=[]
#                                     res = gtt.subgraph_isomorphism(b.base, vess_tree.base, induced=True)
#                                     res = checkUnicityOfElement(res)
#                                     loops.append(len(res))
#                                     NbLoops = NbLoops + len(res)
#                                     #check distance length of loops
#                                     for r in res:
#                                         v_res=np.zeros(vess_tree.n_vertices)
#                                         v_res[r]=1
#                                         loops_edges = np.logical_and(v_res[connectivity[:, 0]], v_res[connectivity[:, 1]])
#                                         lenghtsloops.append(np.sum(lengths[loops_edges]))
#                                     loopsLayerLength.append(lenghtsloops)
#
#                                 loopslengthBrain.append(loopsLayerLength)
#                                 loopsLayer.append(loops)
#
#                         # for region in regions:
#                         #     e = ano.find_order(region, key='name')
#                         #     label = gss4.vertex_annotation();
#                         #     print(region, e)
#                         #     label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
#                         #     # print(np.unique(label_leveled))
#                         #     vertex_filter = label_leveled == e  # 54;
#                         #     print(np.sum(vertex_filter))
#                         #     vess_tree = gss4.sub_graph(vertex_filter=vertex_filter)
#                         #     degree = vess_tree.vertex_degrees()
#                         #     ep.append(np.sum(degree == 1))
#                         #     bp.append(np.sum(degree >= 3))
#                     ep_layer_artery_mutant.append(ep)
#                     bp_layer_artery_mutant.append(bp)
#                     vess_rad_mutant.append(vess_rad)
#                     loopsLayer_mutant.append(loopsLayer)
#                     loopsLayerLength_mutant.append(loopslengthBrain)
#                     NbLoops_mutant.append(NbLoops)
#
#             elif mode == 'clusters':
#                 r, p, l = getRadPlanOrienttaion(gss4, gss4_t, local_normal=True, calc_art=True)
#                 r_f = np.nan_to_num(r)
#                 d = gss4.edge_property('distance_to_surface')
#                 # l = l[np.asarray(~np.isnan(r))]
#                 p = np.nan_to_num(p)
#                 # print(rad)
#                 # print(plan)
#                 # vess_plan.append(plan / vess_tree.n_edges)
#                 vess_rad_mutant.append(np.concatenate((r_f, d), axis=0))
#
#                 loops_length = []
#                 loops_size = []
#                 cluster=gss4.vertex_property('cluster')
#                 artery = from_e_prop2_vprop(gss4, 'artery')
#                 vein = from_e_prop2_vprop(gss4, 'vein')
#                 shortest_paths=[]
#
#                 indices = gss4.vertex_property('overlap')
#                 u, c = np.unique(indices, return_counts=True)
#                 n = 0
#                 j_u = 0
#                 for i, e in tqdm(enumerate(u)):
#                     j = np.asarray(u == e).nonzero()[0][0]
#                     print(j.shape)
#                     vf = np.zeros(gss4.n_vertices)
#                     vf[np.asarray(indices == e).nonzero()[0]] = 1
#
#                     ## get extended overlap
#                     vf = gss4.expand_vertex_filter(vf, steps=ext_step)
#
#
#                     print(gss4, vf.shape, 'containing ', np.sum(vf), ' vertices')
#
#                     g2plot = gss4.sub_graph(vertex_filter=vf)
#                     g2plot = g2plot.largest_component()
#                     c = g2plot.vertex_property('cluster')
#                     c = c[0]
#
#                     if g2plot.n_vertices <= 3000 and g2plot.n_edges >= 5:
#                         shortest_path = []
#                         print(g2plot, vf.shape, 'containing ', np.sum(vf), ' vertices', 'cluster: ', c)
#                         connectivity = g2plot.edge_connectivity()
#                         lengths = g2plot.edge_geometry_lengths('length')
#                         if compute_loops:
#                             for i, b in enumerate(basis):
#                                 # if i >= 3:
#                                 res = gtt.subgraph_isomorphism(b.base, g2plot.base, induced=True)
#                                 res = checkUnicityOfElement(res)
#
#                                 # check distance length of loops
#                                 for r in res:
#                                     v_res = np.zeros(g2plot.n_vertices)
#                                     v_res[r] = 1
#                                     loops_edges = np.logical_and(v_res[connectivity[:, 0]], v_res[connectivity[:, 1]])
#                                     loops_length.append(np.sum(lengths[loops_edges]))
#                                     loops_size.append(i + 3)
#
#                         ## shortest path
#
#                         art = np.logical_and(cluster[:, 0] == c[0], artery)
#                         ve = np.logical_and(cluster[:, 1] == c[1], vein)
#                         vf[np.asarray(art == 1).nonzero()[0]] = 1
#                         vf[np.asarray(ve == 1).nonzero()[0]] = 1
#                         g2plot = gss4.sub_graph(vertex_filter=vf)
#
#                         coord = g2plot.vertex_property('coordinates')
#                         dts = g2plot.vertex_property('distance_to_surface')
#                         connectivity = g2plot.edge_connectivity()
#
#                         art = from_e_prop2_vprop(g2plot, 'artery')
#                         ve = from_e_prop2_vprop(g2plot, 'vein')
#
#                         label = g2plot.vertex_annotation();
#                         label_leveled = ano.convert_label(label, key='order', value='order', level=level + 1)
#
#                         artery_ep = []
#                         vein_ep = []
#                         for i in np.asarray(art == 1).nonzero()[0]:
#                             # if '2/3' in ano.find_acronym(label_leveled[i], key='order'):
#                             ns = g2plot.vertex_neighbours(i)
#                             for n in ns:
#                                 if art[n] == 0:
#                                     artery_ep.append(i)
#                                     art_ep_dist_2_surface_mutant.append(dts[i])
#                                     break
#                         for i in np.asarray(ve == 1).nonzero()[0]:
#                             # if '2/3' in ano.find_acronym(label_leveled[i], key='order'):
#                             ns = g2plot.vertex_neighbours(i)
#                             for n in ns:
#                                 if ve[n] == 0:
#                                     vein_ep.append(i)
#                                     ve_ep_dist_2_surface_mutant.append(dts[i])
#                                     break
#                         print(len(artery_ep), len(vein_ep))
#
#                         path = []
#                         if compute_path:
#                             for a in artery_ep:
#                                 aep_c = coord[a]
#                                 for v in vein_ep:
#                                     l = 0
#                                     vep_c = coord[v]
#                                     g = g2plot.base
#                                     vlist, elist = gtt.shortest_path(g, g.vertex(a), g.vertex(v))
#                                     start = True
#                                     v_prop = np.zeros(g2plot.n_vertices)
#                                     for vl in vlist:
#                                         dist = dts[int(vl)]
#                                         path.append([int(vl), dist])
#                                         v_prop[int(vl)] = 1
#                                     if len(path) > 0:
#                                         ed = np.logical_and(v_prop[connectivity[:, 0]], v_prop[connectivity[:, 1]])
#                                         # print(np.array(path)[:, 1])
#                                         print(np.sum(ed), vl, 'max depth :',np.max(np.array(path)[:, 1]))
#                                         ed = ed.nonzero()[0]
#                                         # ls = ano.find_acronym(label_leveled[int(vl)], key='order')
#                                         # layers = ['1', '2/3', '4', '5', '6a', '6b']
#                                         # for l, la in enumerate(layers):
#                                         #     if la in ls:
#                                         #         # print(la, ls)
#                                         #         path.append([int(vl), l])
#
#                                         for el in ed:
#                                             l = l + g2plot.edge_property('length')[el]
#
#                                         length_short_path.append([l, len(vlist), np.linalg.norm(vep_c - aep_c), np.max(np.array(path)[:, 1])])
#
#                         if len(path)>0:
#                             u_path=np.unique(path, axis=0)
#                             print(len(u_path), len(shortest_paths))
#                             # shortest_path.append(u_path)
#                             shortest_paths.append(u_path)
#
#
#                 print(len(loops_length), len(loops_size))
#                 loops_size_mutant.append([loops_length, loops_size])
#                 shortest_paths_mutant.append(shortest_paths)
#                 length_short_path_mutant.append(length_short_path)
#                 art_rp_2_surf.append(art_ep_dist_2_surface_mutant)
#                 ve_rp_2_surf.append(ve_ep_dist_2_surface_mutant)
#
#         length_short_path_mutant_mutant.append(length_short_path_mutant)
#         vess_rad_mutant.append(vess_rad)
#         bp_dist_2_surface_mutant.append(bp_dist_2_surface)
#         art_ep_dist_2_surface_mutant_mutant.append(art_rp_2_surf)
#         ve_ep_dist_2_surface_mutant_mutant.append(ve_rp_2_surf)
#         shortest_paths_mutant_mutant.append(shortest_paths_mutant)
#         loops_size_mutant_mutant.append(loops_size_mutant)
#
#
#         if feature=='art_raw_signal':
#             feat='art_bp_layer_artery'
#         elif feature=='vessels':
#             feat='vess_bp_layer_vessels'
#         np.save(work_dir+'/'+feat+'_control_'+condition+'.npy',bp_layer_artery_mutant)
#
#         if feature=='art_raw_signal':
#             feat='art_ori_layer_artery'
#         elif feature=='vessels':
#             feat='vess_ori_layer_vessels'
#         np.save(work_dir+'/'+feat+'_control_'+condition+'.npy',vess_rad_mutant)
#
#         np.save(work_dir + '/' + 'art_ep_dist_2_surface_mutant' + condition +  '_'+control+'.npy', art_ep_dist_2_surface_mutant)
#         np.save(work_dir + '/' + 've_ep_dist_2_surface_mutant' + condition +  '_'+control+'.npy', ve_ep_dist_2_surface_mutant)
#
#
#
#         ## brain om;y datas
#
#         # np.save(work_dir + '/' + 'length_short_path_control' + condition + '_'+control+ '.npy', length_short_path_control_control)
#         np.save(work_dir + '/' + 'length_short_path_control' + condition + '_'+control+ '.npy', length_short_path_mutant)
#
#         # np.save(work_dir + '/' + 'shortest_paths_control' + condition + '_'+control+ '.npy', shortest_paths_control_control)
#         np.save(work_dir + '/' + 'shortest_paths_control' + condition + '_'+control+ '.npy', shortest_paths_mutant)
#
#         np.save(work_dir + '/' + 'art_ep_dist_2_surface_mutant' + condition + '_'+control+  '.npy', art_ep_dist_2_surface_mutant)
#         np.save(work_dir + '/' + 've_ep_dist_2_surface_mutant' + condition + '_'+control+  '.npy', ve_ep_dist_2_surface_mutant)
#         # np.save(work_dir + '/' + 'art_ep_dist_2_surface_control' + condition +  '_'+control+'.npy', art_ep_dist_2_surface_control_control)
#         # np.save(work_dir + '/' + 've_ep_dist_2_surface_control' + condition +  '_'+control+'.npy', ve_ep_dist_2_surface_control_control)
#
#         np.save(work_dir + '/' + feature + 'control_rad_ori' + condition +  '_'+control+'.npy', [vess_rad])#_mutant
#         np.save(work_dir + '/' + 'bp_dist_2_surface_control' + condition + '_'+control+ '.npy', bp_dist_2_surface)#_mutant
#
#     if mode=='clusters':
#
#         np.save(work_dir + '/' + 'length_short_path_control' + condition + '.npy', length_short_path_control_control)
#         np.save(work_dir + '/' + 'length_short_path_mutant' + condition + '.npy', length_short_path_mutant_mutant)
#
#         np.save(work_dir + '/' + 'shortest_paths_control' + condition + '.npy', shortest_paths_control)
#         np.save(work_dir + '/' + 'shortest_paths_mutant' + condition + '.npy', shortest_paths_mutant)
#
#         np.save(work_dir + '/' + 'art_ep_dist_2_surface_mutant' + condition + '.npy', art_ep_dist_2_surface_mutant_mutant)
#         np.save(work_dir + '/' + 've_ep_dist_2_surface_mutant' + condition + '.npy', ve_ep_dist_2_surface_mutant_mutant)
#         np.save(work_dir + '/' + 'art_ep_dist_2_surface_control' + condition + '.npy', art_ep_dist_2_surface_control_control)
#         np.save(work_dir + '/' + 've_ep_dist_2_surface_control' + condition + '.npy', ve_ep_dist_2_surface_control_control)
#
#         np.save(work_dir + '/' + feature + 'control_rad_ori' + condition + '.npy', vess_rad_control)
#         np.save(work_dir + '/' + 'bp_dist_2_surface_control' + condition + '.npy', bp_dist_2_surface_control)
#
#         work_dir ='/data_SSD_2to/whiskers_graphs/new_graphs'#'/data_SSD_2to/whiskers_graphs/new_graphs'#'/data_SSD_2to/191122Otof'  #
#         condition =  'barrel_region'#'barrel_region'#'Auditory_regions'
#         length_short_path_control=np.load(work_dir + '/' + 'length_short_path_control' + condition + '.npy')
#         length_short_path_mutant=np.load(work_dir + '/' + 'length_short_path_mutant' + condition + '.npy')
#
#         shortest_paths_control=np.load(work_dir + '/' + 'shortest_paths_control' + condition + '.npy')
#         shortest_paths_mutant=np.load(work_dir + '/' + 'shortest_paths_mutant' + condition + '.npy')
#
#         # art_ep_dist_2_surface_mutant=np.load(work_dir + '/' + 'art_ep_dist_2_surface_mutant' + condition + '.npy')
#         # ve_ep_dist_2_surface_mutant=np.load(work_dir + '/' + 've_ep_dist_2_surface_mutant' + condition + '.npy')
#         # art_ep_dist_2_surface_control=np.load(work_dir + '/' + 'art_ep_dist_2_surface_control' + condition + '.npy')
#         # ve_ep_dist_2_surface_control=np.load(work_dir + '/' + 've_ep_dist_2_surface_control' + condition + '.npy')
#
#         plt.figure()
#         sns.set_style(style='white')
#         sns.despine()
#
#         sns.distplot(art_ep_dist_2_surface_control, color='darkred', bins=50)
#         sns.distplot(ve_ep_dist_2_surface_control, color='darkblue', bins=50)
#
#         sns.distplot(art_ep_dist_2_surface_mutant, color='indianred', bins=50)
#         sns.distplot(ve_ep_dist_2_surface_mutant, color='lightseagreen', bins=50)
#
#         plt.title('art vein end point dist 2 surface control whiskers')
#         plt.legend(['art control', 'vein control', 'art mutant', 'vein_mutant'])
#
#         N=5
#         bin=10
#         normed=True
#         colors_c = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
#
#         colors_m = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
#
#         C_len=[]
#         C_ste = []
#         C_tort = []
#
#         M_len = []
#         M_ste = []
#         M_tort = []
#
#         # plt.figure()
#         for b in range(len(length_short_path_control)):
#             lens=np.array(length_short_path_control[b])[:,0]
#             steps=np.array(length_short_path_control[b])[:,1]
#             torts=np.array(length_short_path_control[b])[:,0]/np.array(length_short_path_control[b])[:,2]
#             torts = torts[~np.isnan(torts)]
#             torts = torts[~np.isinf(torts)]
#
#
#             hist, bins_len = np.histogram(lens, bins=bin, normed=normed)
#             if b==0:
#                 C_len=hist.reshape((hist.shape[0], 1))
#             else:
#                 C_len = np.concatenate((C_len, hist.reshape((hist.shape[0], 1))), axis=1)
#
#             hist, bins_ste = np.histogram(steps, bins=bin, normed=normed)
#             if b == 0:
#                 C_ste = hist.reshape((hist.shape[0], 1))
#             else:
#                 C_ste = np.concatenate((C_ste, hist.reshape((hist.shape[0], 1))), axis=1)
#
#             hist, bins_tort = np.histogram(torts, bins=bin, normed=normed)
#             if b == 0:
#                 C_tort = hist.reshape((hist.shape[0], 1))
#             else:
#                 C_tort = np.concatenate((C_tort, hist.reshape((hist.shape[0], 1))), axis=1)
#
#         Cpd_len = pd.DataFrame(np.array(C_len).transpose()).melt()
#         Cpd_ste = pd.DataFrame(np.array(C_ste).transpose()).melt()
#         Cpd_tort = pd.DataFrame(np.array(C_tort).transpose()).melt()
#
#         plt.figure(N)
#         sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_len)
#         plt.figure(N+1)
#         sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_ste)
#         plt.figure(N+2)
#         sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_tort)
#
#
#
#
#
#
#
#         for b in range(len(length_short_path_mutant)):
#             lens=np.array(length_short_path_mutant[b])[:,0]
#             steps=np.array(length_short_path_mutant[b])[:,1]
#             torts=np.array(length_short_path_mutant[b])[:,0]/np.array(length_short_path_mutant[b])[:,2]
#
#             torts = torts[~np.isnan(torts)]
#             torts = torts[~np.isinf(torts)]
#
#             hist, bins_len = np.histogram(lens, bins=bins_len, normed=normed)
#             if b==0:
#                 M_len=hist.reshape((hist.shape[0], 1))
#             else:
#                 M_len = np.concatenate((M_len, hist.reshape((hist.shape[0], 1))), axis=1)
#
#             hist, bins_ste = np.histogram(steps, bins=bins_ste, normed=normed)
#             if b==0:
#                 M_ste=hist.reshape((hist.shape[0], 1))
#             else:
#                 M_ste = np.concatenate((M_ste, hist.reshape((hist.shape[0], 1))), axis=1)
#
#             hist, bins_tort = np.histogram(torts, bins=bins_tort, normed=normed)
#             if b==0:
#                 M_tort=hist.reshape((hist.shape[0], 1))
#             else:
#                 M_tort = np.concatenate((M_tort, hist.reshape((hist.shape[0], 1))), axis=1)
#
#         Mpd_len = pd.DataFrame(np.array(M_len).transpose()).melt()
#         Mpd_ste = pd.DataFrame(np.array(M_ste).transpose()).melt()
#         Mpd_tort = pd.DataFrame(np.array(M_tort).transpose()).melt()
#
#         plt.figure(N)
#         sns.lineplot(x="variable", y="value", err_style='bars', data=Mpd_len)
#         plt.xticks(np.arange(bin), np.arange(0,np.max(bins_len), np.max(bins_len)/ bin).astype(int))
#         plt.title('length(um)')
#         plt.figure(N+1)
#         sns.lineplot(x="variable", y="value", err_style='bars', data=Mpd_ste)
#         plt.xticks(np.arange(bin), np.arange(0,np.max(bins_ste), np.max(bins_ste)/ bin).astype(int))
#         plt.title('length(bp)')
#         plt.figure(N+2)
#         sns.lineplot(x="variable", y="value", err_style='bars', data=Mpd_tort)
#         plt.xticks(np.arange(bin), np.around(np.arange(0,np.max(bins_tort), np.max(bins_tort)/ bin),1))
#         plt.title('tortuosity')
#
#         for b in range(len(shortest_paths_control)):
#             depth = []
#             for c in range(len(shortest_paths_control[b])):
#                 for p in range(len(shortest_paths_control[b][c])):
#                     depth.append(np.array(shortest_paths_control[b][c][p])[1])
#
#             hist, bins_depth = np.histogram(depth, bins=bin, normed=normed)
#             if b == 0:
#                 C_path = hist.reshape((hist.shape[0], 1))
#             else:
#                 C_path = np.concatenate((C_path, hist.reshape((hist.shape[0], 1))), axis=1)
#         Cpd_path = pd.DataFrame(np.array(C_path).transpose()).melt()
#         plt.figure(N+3)
#         sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_path)
#
#         for b in range(len(shortest_paths_mutant)):
#             depth = []
#             for c in range(len(shortest_paths_mutant[b])):
#                 for p in range(len(shortest_paths_mutant[b][c])):
#                     depth.append(np.array(shortest_paths_mutant[b][c][p])[1])
#
#             hist, bins = np.histogram(depth, bins=bins_depth, normed=normed)
#             if b == 0:
#                 M_path = hist.reshape((hist.shape[0], 1))
#             else:
#                 M_path = np.concatenate((M_path, hist.reshape((hist.shape[0], 1))), axis=1)
#         Mpd_path = pd.DataFrame(np.array(M_path).transpose()).melt()
#         plt.figure(N+3)
#         sns.lineplot(x="variable", y="value", err_style='bars', data=Mpd_path)
#         plt.xticks(np.arange(bin), np.arange(0, np.max(bins_depth), np.max(bins_depth) / bin).astype(int))
#
#         #
#         #
#         # plt.figure(9)
#         # sns.jointplot(x="variable", y="value", err_style='bars', data=Cpd_path)
#
#         work_dir = '/data_SSD_2to/whiskers_graphs/new_graphs'  # '/data_SSD_2to/whiskers_graphs/new_graphs'#'/data_SSD_2to/191122Otof'  #
#         condition = 'barrel_region'  # 'barrel_region'#'Auditory_regions'
#         length_short_path_control = np.load(work_dir + '/' + 'length_short_path_control' + condition + '.npy')
#         length_short_path_mutant = np.load(work_dir + '/' + 'length_short_path_mutant' + condition + '.npy')
#         from scipy.stats import gaussian_kde
#         xedges = np.arange(0, 500, 2)#np.arange(-0.5, 6.5, 1)
#         new_edges = np.arange(0, 500, 2)#np.arange(-0.5, 6.5, 1)
#         yedges = np.arange(10, 30, 1)
#         Xedges, Yedges = np.meshgrid(xedges, yedges)
#         Xedges_new, Yedges_new = np.meshgrid(new_edges, yedges)
#         extent=[0, 500, 10, 30]
#         aspect=13
#
#         Zc = np.zeros((4, len(new_edges), len(yedges)))
#         Zm = np.zeros((4, len(new_edges), len(yedges)))
#
#         import seaborn as sns
#
#
#         # plt.figure()
#         X=[]
#         Y=[]
#         # for b in range(len(shortest_paths_control)-1):
#         #     print(b)
#         #     for p in range(len(shortest_paths_control[b])-1):
#         #         for v in range(len(shortest_paths_control[b][p])-1):
#         #             Y.append(len(shortest_paths_control[b][p]))
#         #             X.append(shortest_paths_control[b][p][v][1])
#         #             print(len(X), len(Y))
#         for b in range(len(length_short_path_control)):
#             steps = np.array(length_short_path_control[b])[:, 0]
#             dep = np.array(length_short_path_control[b])[:, 3]
#             X=steps
#             Y=dep
#
#
#             kde = gaussian_kde(np.vstack([X, Y]))
#             z_c = kde.evaluate(np.vstack([Xedges.ravel(), Yedges.ravel()]))
#             Zc[b, :, :] = z_c.reshape(Xedges.shape).T
#         Zc_mean = np.mean(Zc, axis=0)
#
#         # plt.imshow(Zc_mean.T,extent=extent, cmap='jet', aspect=aspect, origin='lower')
#         # plt.colorbar()
#         # plt.show()
#         # plt.xlabel('step size', size='x-large')
#         # plt.ylabel('depth', size='x-large')
#
#         X = []
#         Y = []
#         # for b in range(len(shortest_paths_mutant)-1):
#         #     print(b)
#         #     for p in range(len(shortest_paths_mutant[b])-1):
#         #         for v in range(len(shortest_paths_mutant[b][p])-1):
#         #             Y.append(len(shortest_paths_mutant[b][p]))
#         #             X.append(shortest_paths_mutant[b][p][v][1])
#         #             print(len(X), len(Y))
#         for b in range(len(length_short_path_mutant)):
#             steps = np.array(length_short_path_mutant[b])[:, 0]
#             dep = np.array(length_short_path_mutant[b])[:, 3]
#             X=steps
#             Y=dep
#
#             kde = gaussian_kde(np.vstack([X, Y]))
#             z_m = kde.evaluate(np.vstack([Xedges.ravel(), Yedges.ravel()]))
#             Zm[b, :, :] = z_m.reshape(Xedges.shape).T
#         Zm_mean = np.mean(Zm, axis=0)
#         # plt.subplot(212)
#         # plt.imshow(Zm_mean.T, extent=extent, cmap='jet', aspect=aspect, origin='lower')
#         # plt.colorbar()
#         # plt.show()
#         # plt.xlabel('step size', size='x-large')
#         # plt.ylabel('depth', size='x-large')
#
#         # plt.figure()
#         from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#
#         import matplotlib.pyplot as plt
#         from matplotlib import cm
#         from matplotlib.ticker import LinearLocator, FormatStrFormatter
#         import numpy as np
#
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         extent = [0, 500, 10, 70]
#         X, Y=np.meshgrid(xedges, np.arange(10, 50, 1))
#         # plt.imshow(np.concatenate((Zc_mean.T , Zm_mean.T), axis=0), extent=extent, cmap='jet', aspect=aspect, origin='lower')
#         surf = ax.plot_surface(X.T, Y.T, np.concatenate((Zc_mean.T , Zm_mean.T), axis=0).T,linewidth=0, cmap=cm.jet, antialiased=False)
#         # plt.imshow(Zc_mean.T - Zm_mean.T, extent=extent, cmap='jet', aspect=aspect, origin='lower')
#         plt.colorbar()
#
#         plt.title('depth vs length heatmap '+ ' in ' + condition, size='x-large')
#         plt.xlabel('length size', size='x-large')
#         plt.ylabel('depth', size='x-large')
#         plt.yticks(np.arange(10, 50, 10), np.concatenate((np.arange(10*15.42,30*15.42,10*15.42), np.arange(20*15.42,30*15.42,10*15.42))).astype(int))
#
#         from scipy import stats
#
#         pcutoff = 0.05
#
#         tvals, pvals = stats.ttest_ind(Zc, Zm, axis=0, equal_var=True);
#
#         pi = np.isnan(pvals);
#         pvals[pi] = 1.0;
#         tvals[pi] = 0;
#
#         pvals2 = pvals.copy();
#         pvals2[pvals2 > pcutoff] = pcutoff;
#         psign = np.sign(tvals)
#         extent = [0, 500, 10, 30]
#         pvalscol = colorPValues(pvals2, psign, positive=[255, 0, 0], negative=[0, 255, 0])
#         plt.figure()
#         plt.imshow(np.swapaxes(pvalscol,0,1), extent=extent, cmap='jet', aspect=aspect, origin='lower')
#         plt.title('depth vs length pvals heatmap ' + ' in ' + condition, size='x-large')
#         plt.xlabel('length size', size='x-large')
#         plt.ylabel('depth', size='x-large')
#         # plt.yticks(np.arange(10, 30, 10),np.arange(10 * 15.42, 30 * 15.42, 10 * 15.42).astype(int))
#         # work_dir =  '/data_SSD_2to/191122Otof'  #'/data_SSD_2to/whiskers_graphs/new_graphs'#'/data_SSD_2to/191122Otof'  #
#         # condition =  'Auditory_regions'#'barrel_region'#'Auditory_regions'
#         # #
#         # shortest_paths_control = np.load(work_dir + '/' + 'shortest_paths_control' + condition + '.npy')
#         # shortest_paths_mutant = np.load(work_dir + '/' + 'shortest_paths_mutant' + condition + '.npy')
#         # from scipy.stats import gaussian_kde
#         #
#         # xedges = np.arange(-0.5, 6.5, 1)
#         # new_edges = np.arange(-0.5, 6.5, 1)
#         # yedges = np.arange(0, 100, 1)
#         # Xedges, Yedges = np.meshgrid(xedges, yedges)
#         # Xedges_new, Yedges_new = np.meshgrid(new_edges, np.arange(0, 100, 5))
#         #
#         # Zc = np.empty((500 , len(np.arange(0, 800, 1))))
#         # Zm = np.empty((500, len(np.arange(0, 800, 1))))
#         # Zc[:]=np.nan
#         # Zm[:] = np.nan
#         # import seaborn as sns
#         #
#         # # plt.figure()
#         # # X = []
#         # # Y = []
#         # for b in range(len(shortest_paths_control) - 1):
#         #     print(b)
#         #     for p in range(len(shortest_paths_control[b]) - 1):
#         #
#         #         for v in range(len(shortest_paths_control[b][p]) - 1):
#         #             Zc[p, v]=shortest_paths_control[b][p][v][1]
#         #
#         # #     kde = gaussian_kde(np.vstack([X, Y]))
#         # #     z_c = kde.evaluate(np.vstack([Xedges.ravel(), Yedges.ravel()]))
#         # #     Zc[b, :, :] = z_c.reshape(Xedges.shape).T
#         # # Zc_mean = np.mean(Zc, axis=0)
#         # # plt.figure()
#         # # plt.imshow(Zc_mean, extent=[ 0, 100,-0.5, 6], cmap='jet', aspect=10, origin='lower')
#         # # plt.colorbar()
#         # # plt.show()
#         # #
#         #
#         # for b in range(len(shortest_paths_mutant) - 1):
#         #     print(b)
#         #     for p in range(len(shortest_paths_mutant[b]) - 1):
#         #
#         #         for v in range(len(shortest_paths_mutant[b][p]) - 1):
#         #             Zm[p, v]=shortest_paths_mutant[b][p][v][1]
#         #
#         # dataC = pd.DataFrame(Zc.transpose()).melt()
#         # dataM = pd.DataFrame(Zm.transpose()).melt()
#         # plt.figure()
#         #
#         # sns.lineplot(x="variable", y="value", ci='sd', data=dataC)
#         # sns.lineplot(x="variable", y="value", ci='sd', data=dataM)
#         # sns.set_style(style='white')
#
#
#
#
#
#         #     kde = gaussian_kde(np.vstack([X, Y]))
#         #     z_m = kde.evaluate(np.vstack([Xedges.ravel(), Yedges.ravel()]))
#         #     Zm[b, :, :] = z_m.reshape(Xedges.shape).T
#         # Zm_mean = np.mean(Zm, axis=0)
#         # plt.figure()
#         # plt.imshow(Zm_mean, extent=[ 0, 100,-0.5, 6], cmap='jet', aspect=10, origin='lower')
#         # plt.colorbar()
#         # plt.show()
#         #
#         # plt.figure()
#         # plt.imshow(Zc_mean - Zm_mean, extent=[ 0, 100,-0.5, 6], cmap='jet', aspect=10, origin='lower')
#         # plt.colorbar()
#         # plt.title('loops step VS length heatmap ' + ' in ' + condition, size='x-large')
#         # plt.xlabel('loops size', size='x-large')
#         # plt.ylabel('loops lengths', size='x-large')
#
#
#     ###### LOOPPS
#         np.save(work_dir+'/'+'loops_size_control'+condition+'.npy',loops_size_control)
#         np.save(work_dir+'/'+'loops_size_mutant'+condition+'.npy',loops_size_mutant)
#
#         work_dir = '/data_SSD_2to/191122Otof'  # '/data_SSD_2to/whiskers_graphs/new_graphs'#
#         condition = 'Auditory_regions'  # 'barrel_region'#'Auditory_regions'
#         #
#         loops_size_control = np.load(work_dir + '/' + 'loops_size_control' + condition + '.npy')
#         loops_size_mutant = np.load(work_dir + '/' + 'loops_size_mutant' + condition + '.npy')
#
#         # from scipy import interpolate
#         # from scipy.stats import gaussian_kde
#         # xedges = np.arange(2.5, 6.5, 1)
#         # new_edges = [3, 4, 5, 6]
#         # yedges = np.arange(0, 150, 5)
#         # Xedges, Yedges = np.meshgrid(xedges, yedges)
#         # Xedges_new, Yedges_new = np.meshgrid(new_edges, np.arange(0, 150, 5))
#         #
#         # Zc = np.zeros((len(loops_size_control), len(new_edges), len(np.arange(0, 150, 5))))
#         # Zm = np.zeros((len(loops_size_mutant), len(new_edges), len(np.arange(0, 150, 5))))
#         #
#         # import seaborn as sns
#         #
#         #
#         # plt.figure()
#         #
#         # for b in range(len(loops_size_control)):
#         #     X=loops_size_control[b][1]
#         #     Y = loops_size_control[b][0]
#         #
#         #
#         #     kde = gaussian_kde(np.vstack([X, Y]))
#         #     z_c = kde.evaluate(np.vstack([Xedges.ravel(), Yedges.ravel()]))
#         #     Zc[b, :, :] = z_c.reshape(Xedges.shape).T
#         # Zc_mean = np.mean(Zc, axis=0)
#         #
#         # for b in range(len(loops_size_mutant)):
#         #     X = loops_size_mutant[b][1]
#         #     Y = loops_size_mutant[b][0]
#         #
#         #     kde = gaussian_kde(np.vstack([X, Y]))
#         #     z_m = kde.evaluate(np.vstack([Xedges.ravel(), Yedges.ravel()]))
#         #     Zm[b, :, :] = z_m.reshape(Xedges.shape).T
#         # Zm_mean = np.mean(Zm, axis=0)
#         #
#         # plt.imshow((Zc_mean - Zm_mean).T, extent=[2.5, 6.5, 0, 150], cmap='jet', aspect=0.01, origin='lower')
#         # plt.colorbar()
#         # plt.title('loops step VS length heatmap '+ ' in ' + condition, size='x-large')
#         # plt.xlabel('loops size', size='x-large')
#         # plt.ylabel('loops lengths', size='x-large')
#
#     raw_signal_control=np.array(raw_signal_control)
#     raw_signal_mutant = np.array(raw_signal_mutant)
#
#     np.save(work_dir+'/'+feature+'_control_standardized_'+condition+'.npy',raw_signal_control)
#     np.save(work_dir+'/'+feature+'_mutant_standardized_'+condition+'.npy',raw_signal_mutant)
#
#     np.save(work_dir+'/'+'loopsLayerLength_control_'+condition+'.npy',loopsLayerLength_control)
#     np.save(work_dir+'/'+'loopsLayerLength_mutant_'+condition+'.npy',loopsLayerLength_mutant)
#
#     np.save(work_dir+'/'+'hist_dist_control_'+condition+'.npy',distances_control)
#     np.save(work_dir+'/'+'hist_dist_mutant_'+condition+'.npy',distances_mutant)
#
#
#     if feature=='vessels':
#         np.save(work_dir + '/' + feature + 'control_rad_ori' + condition + '.npy', vess_rad_control)
#         np.save(work_dir+'/'+feature+'mutant_rad_ori'+condition+'.npy',vess_rad_mutant)
#
#         np.save(work_dir + '/' + feature + 'loopsLayer_control' + condition + '.npy', loopsLayer_control)
#         np.save(work_dir + '/' + feature + 'loopsLayer_mutant' + condition + '.npy', loopsLayer_mutant)
#
#     if feature=='art_raw_signal':
#         feat='art_ep_layer_artery'
#     elif feature=='vessels':
#         feat='vess_ep_layer_vessels'
#
#     np.save(work_dir+'/art_ep_layer_'+feature+'_control_'+condition+'.npy',ep_layer_artery_control)
#     np.save(work_dir+'/art_ep_layer_'+feature+'_mutant_'+condition+'.npy',ep_layer_artery_mutant)
#
#     if feature=='art_raw_signal':
#         feat='art_bp_layer_artery'
#     elif feature=='vessels':
#         feat='vess_bp_layer_vessels'
#
#     np.save(work_dir+'/'+feat+'_control_'+condition+'.npy',bp_layer_artery_control)
#     np.save(work_dir+'/'+feat+'_mutant_'+condition+'.npy',bp_layer_artery_mutant)
#
#     if feature=='art_raw_signal':
#         feat='art_nb_artery'
#
#         np.save(work_dir+'/'+feat+'_control_'+condition+'.npy',nb_artery_control)
#         np.save(work_dir+'/'+feat+'_mutant_'+condition+'.npy',nb_artery_mutant)
#
#         np.save(work_dir + '/length_art_control_' + condition + '.npy', art_length_control)
#         np.save(work_dir + '/length_art_mutant' + condition + '.npy', art_length_mutant)
#
#
#     if feature=='art_raw_signal':
#         feat='art_bp_artery'
#
#         np.save(work_dir+'/'+feat+'_control_'+condition+'.npy',bp_artery_control)
#         np.save(work_dir+'/'+feat+'_mutant_'+condition+'.npy',bp_artery_mutant)
#
#     ##############  PLOTTING
#     #
#     # work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     # condition='barrels'#'Auditory regions'
#     #
#     # nb_artery_control=np.load(work_dir+'/art_nb_artery_control_'+condition+'.npy')
#     # nb_artery_mutant=np.load(work_dir+'/art_nb_artery_mutant_'+condition+'.npy')
#     #
#     # bp_artery_control=np.load(work_dir+'/art_bp_artery_control_'+condition+'.npy')
#     # bp_artery_mutant=np.load(work_dir+'/art_bp_artery_mutant_'+condition+'.npy')
#     #
#     ############ EXTRACT LAYER 4
#     template_shape=(320,528,228)
#     atlas = io.read('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation.tif')[:, :,:228]
#     l4_list = []
#     bool=True
#     for r in reg_list.keys():
#         n = ano.find_name(r, key='order')
#         # print(n)
#
#         for se in reg_list[r]:
#             l = ano.find(se, key='order')['level']
#             n = ano.find(se, key='order')['name']
#             # print(n, se)
#             if '4' in n:
#                 print(ano.find(se, key='order')['name'], se)
#                 # print('4 in n')
#                 seid=ano.find(se, key='order')['id']
#                 l4_list.append(seid)
#             # try:
#             #     bool = True
#             #     reg_list[se]
#             # except:
#             #     print('keyerror')
#             #     bool=False
#             # if bool:
#             #     print(bool)
#             #     for sse in reg_list[se]:
#             #         # l = ano.find(sse, key='order')['level']
#             #         n = ano.find(sse, key='order')['name']
#             #         # print(n, sse)
#             #         if '4' in n:
#             #             # print('4 in n')
#             #             l4_list.append(sse)
#
#
#
#     mask = np.zeros(atlas.shape)
#     for rl in l4_list:
#         coord=np.array(np.asarray(atlas == rl).nonzero()).T
#         # print(coord.shape)
#         for pt in coord:
#             # try:
#             mask[pt[0], pt[1], pt[2]] = 1
#             # except:
#             #     print(template_shape, pt)
#
#     l4_142=np.multiply(np.flip(mask,axis=1),np.nan_to_num(extr_frac[:,:,:,1]))
#     io.write('/data_SSD_2to/whiskers_graphs/new_graphs/vox_control_extracted_fraction_whole_cortex_5_158L.tif',l4_142.astype('float32'))
#
#
#     ############ DISTANCE FROM ARTERIES
#     plt.figure()
#     histc=[]
#     for d in distances_control:
#         h, b=np.histogram(d, np.arange(50))
#         histc.append(h)
#
#     histm=[]
#     for d in distances_mutant:
#         h, b=np.histogram(d, np.arange(50))
#         histm.append(h)
#     histc=np.array(histc)
#     histm=np.array(histm)
#
#     dfc = pd.DataFrame(histc[:, :]).melt()
#     dfm=pd.DataFrame(histm[:, :]).melt()
#
#     for j in range(histc.shape[1]):
#         st, pval = ttest_ind(histc[:, j]*100, histm[:, j]*100, equal_var = False)
#         print(layers[i], condition, j, pval)
#         if pval < 0.06:
#             print(condition, ls[i], pval, '!!!!!!!!!!!!!!!!11')
#
#
#     sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)
#     sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfm)
#
#
#     for j in range(histc.shape[0]):
#         colors = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
#         plt.plot(histc[j, :], '--', color=colors[j], alpha=0.5, label='_nolegend_')
#
#         colors = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
#         plt.plot(histm[j, :], '--', color=colors[j], alpha=0.5, label='_nolegend_')
#
#
#     sns.despine()
#     plt.title('distance from arteries distribution' + ' in '+ condition, size='x-large')
#     plt.xlabel('step', size='x-large')
#     # plt.xticks([0, 1, 2, 3], ['3', '4', '5', '6'], size='x-large')
#     # plt.xticks([0, 1, 2, 3, 4], ['l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
#     plt.yticks(size='x-large')
#     plt.legend(['controls', 'Deprived'])  # Otof-/-
#     plt.ylabel('count', size='x-large')
#     # plt.legend(*zip(*labels), loc=2)
#     plt.legend(['controls', 'Deprived'])  # Otof-/-
#     ax = plt.gca()
#     leg = ax.get_legend()
#     leg.legendHandles[0].set_color('blue')
#     leg.legendHandles[1].set_color('orange')
#     plt.tight_layout()
#
#
#
#
#
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'#'/data_SSD_2to/191122Otof'
#     condition='Auditory_regions'#'barrel_region'#'Auditory_regions'
#     feature='art_raw_signal'#'vessels'#art_raw_signal
#
#
#     if feature=='art_raw_signal':
#         feat='art_bp_layer_artery'
#     elif feature=='vessels':
#         feat='vess_bp_layer_vessels'
#
#     bp_layer_artery_control=np.load(work_dir+'/'+feat+'_control_'+condition+'.npy')
#     bp_layer_artery_mutant=np.load(work_dir+'/'+feat+'_mutant_'+condition+'.npy')
#
#     if feature=='art_raw_signal':
#         feat='art_ep_layer_artery'
#     elif feature=='vessels':
#         feat='vess_ep_layer_vessels'
#
#     ep_layer_artery_control=np.load(work_dir+'/art_ep_layer_'+feature+'_control_'+condition+'.npy')
#     ep_layer_artery_mutant=np.load(work_dir+'/art_ep_layer_'+feature+'_mutant_'+condition+'.npy')
#     #
#     # ep_layer_artery_control=np.load(work_dir+'/art_ep_layer_artery_control_'+condition+'.npy')
#     # ep_layer_artery_mutant=np.load(work_dir+'/art_ep_layer_artery_mutant_'+condition+'.npy')
#     #
#     # raw_signal_control=np.load(work_dir+'/art_raw_signal_control_standardized_'+condition+'.npy')
#     # raw_signal_mutant=np.load(work_dir+'/art_raw_signal_mutant_standardized_'+condition+'.npy')
#     #
#
#     ##KDE loops plot lemgth vs size
#     from scipy import interpolate
#     from scipy.stats import gaussian_kde
#     xedges=np.arange(2.5, 6.5, 1)
#     new_edges=[3,4,5,6]
#     yedges=np.arange(50, 300, 10)
#     Xedges, Yedges = np.meshgrid(xedges, yedges)
#     Xedges_new, Yedges_new = np.meshgrid(new_edges, np.arange(0, 150, 5))
#     #
#     # H, xedges, yedges = np.histogram2d(X, Y, bins=(xedges, yedges))
#     # Z=H.flatten()
#     # f = interpolate.interp2d(Xedges, Yedges, Z, kind='cubic')
#     # z_c=f(Xedges, Yedges)
#     #################
#
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'#'/data_SSD_2to/191122Otof'#
#     condition='barrel_region'#'barrel_region'#'Auditory_regions'
#     #
#     loopsLayerLength_control=np.load(work_dir+'/'+'loopsLayerLength_control_'+condition+'.npy')
#     loopsLayerLength_mutant=np.load(work_dir+'/'+'loopsLayerLength_mutant_'+condition+'.npy')
#
#     layers=['l1','l2/3', 'l4', 'l5', 'l6a', 'l6b']
#     colors_c = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
#     colors_m = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
#     # plt.ion()
#     # plt.show()
#     # plt.pause(0.001)
#
#     # Zc=np.zeros((len(loopsLayerLength_control), len(xedges), len(yedges)))
#     # Zm=np.zeros((len(loopsLayerLength_mutant), len(xedges), len(yedges)))
#
#     Zc=np.zeros((len(loopsLayerLength_control), len(new_edges), len(np.arange(0, 150, 5))))
#     Zm=np.zeros((len(loopsLayerLength_mutant), len(new_edges), len(np.arange(0, 150, 5))))
#
#     import seaborn as sns
#     for i, l in enumerate(layers):
#         print(i)
#         # i=1
#         plt.figure()
#
#         for  b in range(len(loopsLayerLength_control)):
#             X = []
#             Y = []
#             for s in range(len(loopsLayerLength_control[b][i])):
#                 print(s+3)
#                 L = loopsLayerLength_control[b][i][s]
#                 for l in L:
#                     X.append(s+3)
#                     Y.append(l)
#             # print('plotting', controls[b], len(X), len(Y))
#             # sns.kdeplot(X, Y, alpha=0.3, color=colors_c[b])
#             # plt.scatter(X, Y, alpha=0.5, color=colors_c[b])
#             # H =np.histogram2d(X, Y, bins=(xedges, yedges))
#             ## plt.figure()
#             # z_c = plt.imshow(H[0].T, interpolation='bilinear', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect=0.01,cmap='jet').get_array()
#             ## plt.show
#             # Zc[b, :, :] = z_c.reshape(Xedges_new.shape).T
#
#             kde = gaussian_kde(np.vstack([X, Y]))
#             z_c = kde.evaluate(np.vstack([Xedges.ravel(), Yedges.ravel()]))
#             Zc[b, :,:]=z_c.reshape(Xedges.shape).T
#         Zc_mean=np.mean(Zc, axis=0)
#
#
#         for b in range(len(loopsLayerLength_mutant)):
#             X = []
#             Y = []
#             for s in range(len(loopsLayerLength_mutant[b][i])):
#                 print(s+3)
#                 L=loopsLayerLength_mutant[b][i][s]
#                 for l in L:
#                     X.append(s+3)
#                     Y.append(l)
#             # print('plotting', mutants[b], len(X), len(Y))
#             # sns.kdeplot(X, Y, alpha=0.3, color=colors_m[b])
#             # plt.scatter(X, Y, alpha=0.5, color=colors_m[b])
#             # H, xedges, yedges = np.histogram2d(X, Y, bins=(xedges, yedges))
#             # H = plt.hist2d(X, Y, bins=(xedges, yedges))
#             # H = np.histogram2d(X, Y, bins=(xedges, yedges))
#             ## plt.figure()
#             # z_m = plt.imshow(H[0].T, interpolation='bilinear', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],aspect=0.01,cmap='jet').get_array()
#             ## plt.show()
#             # Zm[b, :, :] = z_m.reshape(Xedges_new.shape).T
#             kde = gaussian_kde(np.vstack([X, Y]))
#             z_m = kde.evaluate(np.vstack([Xedges.ravel(), Yedges.ravel()]))
#             Zm[b, :, :] = z_m.reshape(Xedges.shape).T
#         Zm_mean = np.mean(Zm, axis=0)
#
#         plt.imshow((Zc_mean-Zm_mean).T,extent=[2.5, 6.5, 0, 150], cmap='jet',aspect=0.01,origin='lower')
#         plt.colorbar()
#         plt.title('nb of loops per layer in layer '+layers[i] + ' in '+ condition, size='x-large')
#         plt.xlabel('loops size', size='x-large')
#         plt.ylabel('loops lengths', size='x-large')
#
#
#
#     ## PLOT LOOPS SIZE PER LAYER
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     condition='barrel_region'
#     feature='vessels'
#
#     normalized=False
#     baselined=False
#
#     loopsLayer_control=np.load(work_dir + '/' + feature + 'loopsLayer_control' + condition + '.npy')
#     loopsLayer_mutant=np.load(work_dir + '/' + feature + 'loopsLayer_mutant' + condition + '.npy')
#
#     lc=np.array(loopsLayer_control).astype(float)
#     lm=np.array(loopsLayer_mutant).astype(float)
#     layers=['l1','l2/3', 'l4', 'l5', 'l6a', 'l6b']
#
#     for i in range(lc.shape[1]):#layer
#         plt.figure()
#         if normalized:
#             llc=lc[:, i, :]
#             for k in range(lc.shape[0]):
#                 llc[k, :]=llc[k, :]/np.sum(lc[k, i, :])
#             llm = lm[:, i, :]
#             for k in range(llm.shape[0]):
#                 llm[k, :] = llm[k, :] / np.sum(lm[k, i, :])
#
#         if baselined:
#             llc = lc[:, i, :]
#             m = np.mean(llc, axis=0)
#             s = np.std(llc, axis=0)
#             for k in range(lc.shape[0]):
#                 llc[:, k] = (llc[:, k] - m) / s
#
#             llm = lm[:, i, :]
#             m = np.mean(llm, axis=0)
#             s = np.std(llm, axis=0)
#             for k in range(lc.shape[0]):
#                 llm[:, k] = (llm[:, k] - m) / s
#
#             baseline = np.mean(llc, axis=0)
#             model = LinearRegression()
#
#             for k in range(llc.shape[0]):
#                 m = np.mean(llc[k])
#                 s = np.std(llc[k])
#                 llc[k] = (llc[k] - m) / s
#                 # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])  # vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controls[i, 2]
#                 model.fit(baseline.reshape(-1, 1), llc[k].transpose().reshape(-1, 1))
#                 a = model.coef_
#                 b = model.intercept_
#                 print('intercept:', b)
#                 print('slope:', a)
#                 llc[k] = llc[k] - a * baseline - b
#
#
#             for k in range(llm.shape[0]):
#                 # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]
#                 model.fit(baseline.reshape(-1, 1), llm[k].transpose().reshape(-1, 1))
#                 a = model.coef_
#                 b = model.intercept_
#                 print('intercept:', b)
#                 print('slope:', a)
#                 llm[k] = llm[k] - a * baseline - b
#
#         else:
#             llc=lc[:, i, :]
#             llm=lm[:, i, :]
#         dfc = pd.DataFrame(llc[:, :]).melt()
#         dfm=pd.DataFrame(llm[:, :]).melt()
#
#         for j in range(llc.shape[1]):
#             st, pval = ttest_ind(llc[:, j]*100, llm[:, j]*100, equal_var = False)
#             print(layers[i], condition, j, pval)
#             if pval < 0.06:
#                 print(condition, ls[i], pval, '!!!!!!!!!!!!!!!!11')
#
#
#         sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)
#         sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfm)
#
#
#         for j in range(llc.shape[0]):
#             colors = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
#             plt.plot(llc[j, :], '--', color=colors[j], alpha=0.5, label='_nolegend_')
#
#             colors = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
#             plt.plot(llm[j, :], '--', color=colors[j], alpha=0.5, label='_nolegend_')
#
#
#         sns.despine()
#         plt.title('nb of loops per layer in layer '+layers[i] + ' in '+ condition, size='x-large')
#         plt.xlabel('loops size', size='x-large')
#         plt.xticks([0, 1, 2, 3], ['3', '4', '5', '6'], size='x-large')
#         # plt.xticks([0, 1, 2, 3, 4], ['l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
#         plt.yticks(size='x-large')
#         plt.legend(['controls', 'Deprived'])  # Otof-/-
#         plt.ylabel('nb of loops per layer in ' + condition, size='x-large')
#         # plt.legend(*zip(*labels), loc=2)
#         plt.legend(['controls', 'Deprived'])  # Otof-/-
#         ax = plt.gca()
#         leg = ax.get_legend()
#         leg.legendHandles[0].set_color('blue')
#         leg.legendHandles[1].set_color('orange')
#         plt.tight_layout()
#
#
#
#
#
#     ## PLOT LOOPS
#
#
#     plt.figure()
#     sns.despine()
#     colors=['cadetblue', 'indianred']
#     box1 = plt.boxplot(np.array(NbLoops_control), positions=[1], patch_artist=True, widths=0.5, showfliers=False, showmeans=True,
#                        autorange=True, meanline=True)
#     box2 = plt.boxplot(np.array(NbLoops_mutant), positions=[2], patch_artist=True, widths=0.5, showfliers=False, showmeans=True,
#                        autorange=True, meanline=True)
#     for patch in box1['boxes']:
#         patch.set_facecolor(colors[0])
#     for patch in box2['boxes']:
#         patch.set_facecolor(colors[1])
#     plt.ylabel('counts', size='x-large')
#     plt.xticks([1,2], ['controls', 'deprived'], size='x-large')
#     plt.yticks(size='x-large')
#     plt.xlim(0,3)
#     plt.title('Number of loops in '+condition , size='x-large')
#     plt.tight_layout()
#
#
#
#     plt.figure()
#     step=10
#     max=150
#     normed=False
#     # loopsLayer_control=np.array(loopsLayer_control)
#     # loopsLayer_mutant=np.array(loopsLayer_mutant)
#
#     normalized=False
#     baselined=False
#     vess_controls = np.sum(np.array(loopsLayer_control).astype(float)[:, :], axis=2)
#
#     if normalized:
#         for i in range(vess_controls.shape[0]):
#             m = np.mean(vess_controls[i])
#             s = np.std(vess_controls[i])
#             vess_controls[i] = (vess_controls[i] - m) / s
#             # vess_controls[i]=vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]
#
#     if baselined:
#         baseline = np.mean(vess_controls, axis=0)
#         model = LinearRegression()
#
#         for i in range(vess_controls.shape[0]):
#             m = np.mean(vess_controls[i])
#             s = np.std(vess_controls[i])
#             vess_controls[i] = (vess_controls[i] - m) / s
#             # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])  # vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controls[i, 2]
#             model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
#             a = model.coef_
#             b = model.intercept_
#             print('intercept:', b)
#             print('slope:', a)
#             vess_controls[i] = vess_controls[i] - a * baseline - b
#
#     C = []
#     colors = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
#     for i in range(vess_controls.shape[1]):
#         c = vess_controls[:, i]
#         C.append(c.tolist())
#     for i in range(vess_controls.shape[0]):
#         plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')
#
#     if len(np.array(C).shape) == 3:
#         C = np.squeeze(np.array(C), axis=2)
#     else:
#         C = np.array(C)
#
#     dfc = pd.DataFrame(np.array(C).transpose()).melt()
#     sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)
#
#     vess_controls = np.sum(np.array(loopsLayer_mutant).astype(float)[:, :], axis=2)
#     if normalized:
#         for i in range(vess_controls.shape[0]):
#             m = np.mean(vess_controls[i])
#             s = np.std(vess_controls[i])
#             vess_controls[i] = (vess_controls[i] - m) / s
#     if baselined:
#         for i in range(vess_controls.shape[0]):
#             # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]
#             model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
#             a = model.coef_
#             b = model.intercept_
#             print('intercept:', b)
#             print('slope:', a)
#             vess_controls[i] = vess_controls[i] - a * baseline - b
#     M = []
#     colors = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
#     for i in range(vess_controls.shape[1]):
#         m = vess_controls[:, i]
#         M.append(m.tolist())
#     for i in range(vess_controls.shape[0]):
#         plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')
#
#     if len(np.array(M).shape) == 3:
#         M = np.squeeze(np.array(M), axis=2)
#     else:
#         M = np.array(M)
#
#     dfm = pd.DataFrame(np.array(M).transpose()).melt()
#     sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfm)
#
#     from scipy.stats import ttest_ind
#
#     for i in range(C.shape[0]):
#         gc = np.array(dfc.loc[lambda dfc: dfc['variable'] == i, 'value'])
#         gm = np.array(dfm.loc[lambda dfm: dfm['variable'] == i, 'value'])
#         st, pval = ttest_ind(gc, gm)
#         print(condition, ls[i], pval)
#         if pval < 0.06:
#             print(condition, ls[i], pval, '!!!!!!!!!!!!!!!!11')
#     sns.despine()
#     plt.ylabel('nb of loops in ' + condition, size='x-large')
#     plt.xlabel('layers', size='x-large')
#     plt.xticks([0, 1, 2, 3, 4, 5], ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
#     # plt.xticks([0, 1, 2, 3, 4], ['l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
#     plt.yticks(size='x-large')
#     plt.legend(['controls', 'Deprived'])  # Otof-/-
#     plt.title('nb of loops per layer in ' + condition, size='x-large')
#     # plt.legend(*zip(*labels), loc=2)
#     plt.legend(['controls', 'Deprived'])  # Otof-/-
#     ax = plt.gca()
#     leg = ax.get_legend()
#     leg.legendHandles[0].set_color('blue')
#     leg.legendHandles[1].set_color('orange')
#     plt.tight_layout()
#
#     ## PLOT DISTRIBUTION ARTERY LENGTH
#     step = 100
#     max = 1000
#     normed = False
#
#     for a, m in enumerate(art_length_control):
#         if a == 0:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             C = hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             C = np.concatenate((C, hist.reshape((hist.shape[0], 1))), axis=1)
#
#     for a, m in enumerate(art_length_mutant):
#         if a == 0:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             M = hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             M = np.concatenate((M, hist.reshape((hist.shape[0], 1))), axis=1)
#     #
#     # M = M.reshape((M.shape[0], 1))
#     # C = C.reshape((C.shape[0], 1))
#     # data=[C, M]
#
#     # data=pd.DataFrame(np.array(data).transpose()).melt()
#     C = pd.DataFrame(np.array(C).transpose()).melt()
#     M = pd.DataFrame(np.array(M).transpose()).melt()
#
#     plt.figure()
#     import pandas as pd
#     import seaborn as sns
#
#     sns.set_style(style='white')
#     sns.lineplot(x="variable", y="value", ci='sd', data=C)  # , y="normalized count"
#     sns.lineplot(x="variable", y="value", ci='sd', data=M)  # , err_style="bars"
#
#     sns.despine()
#
#     plt.legend(['controls', 'mutants'])
#     plt.title('lengths of penetrating arteries in ' + condition, size='x-large')
#
#     plt.xlabel("lengths of penetrating penetrating arteries")
#     plt.ylabel(" count")
#     plt.xticks(range(np.arange(0, max, step).shape[0]), np.arange(0, max, step))
#     plt.tight_layout()
#     plt.yscale('log')
#
#
#     ## PLOT DISTRIBUTION OF ARTERIES RAW VALUES
#     step=2
#     max=20
#     min=-1
#     normed=True
#
#     for a, m in enumerate(raw_signal_control):
#         if a==0:
#             hist,bins = np.histogram(np.array(m),bins = np.arange(min, max, step), normed=normed)
#             C=hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(min, max, step), normed=normed)
#             C=np.concatenate((C, hist.reshape((hist.shape[0], 1))), axis=1)
#
#     for a, m in enumerate(raw_signal_mutant):
#         if a == 0:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(min, max, step), normed=normed)
#             M = hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(min, max, step), normed=normed)
#             M = np.concatenate((M, hist.reshape((hist.shape[0], 1))), axis=1)
#     #
#     # M = M.reshape((M.shape[0], 1))
#     # C = C.reshape((C.shape[0], 1))
#     # data=[C, M]
#
#     # data=pd.DataFrame(np.array(data).transpose()).melt()
#     C=pd.DataFrame(np.array(C).transpose()).melt()
#     M=pd.DataFrame(np.array(M).transpose()).melt()
#
#
#     plt.figure()
#     import pandas as pd
#     import seaborn as sns
#
#     sns.set_style(style='white')
#     sns.lineplot(x="variable", y="value", ci='sd', data=C)#, y="normalized count"
#     sns.lineplot(x="variable", y="value", ci='sd', data=M)#, err_style="bars"
#
#     sns.despine()
#
#     plt.legend(['controls', 'mutants'])
#     plt.title('arteries raw signal of penetrating arteries in ' + condition, size='x-large')
#
#     plt.xlabel("arteries raw signal of penetrating penetrating arteries", size='x-large')
#     plt.ylabel('count', size='x-large')
#     plt.yticks(size='x-large')
#     plt.xticks(range(np.arange(min, max, step).shape[0]),np.arange(min, max, step),size='x-large')
#     plt.tight_layout()
#     plt.yscale('log')
#
#     ## PLOT ARTERY BP, NB, EP, PER LAYER
#     import seaborn as sns
#
#     plt.figure()
#     sns.despine()
#     colors=['cadetblue', 'indianred']
#     box1 = plt.boxplot(nb_artery_control, positions=[1], patch_artist=True, widths=0.5, showfliers=False, showmeans=True,
#                        autorange=True, meanline=True)
#     box2 = plt.boxplot(nb_artery_mutant, positions=[2], patch_artist=True, widths=0.5, showfliers=False, showmeans=True,
#                        autorange=True, meanline=True)
#     for patch in box1['boxes']:
#         patch.set_facecolor(colors[0])
#     for patch in box2['boxes']:
#         patch.set_facecolor(colors[1])
#     plt.ylabel('counts', size='x-large')
#     plt.xticks([1,2], ['controls', 'deprived'], size='x-large')
#     plt.yticks(size='x-large')
#     plt.xlim(0,3)
#     plt.title('Controls number of artery in '+condition , size='x-large')
#     plt.tight_layout()
#
#
#
#     plt.figure()
#     step=10
#     max=150
#     normed=False
#     for a, m in enumerate(bp_artery_control):
#         if a==0:
#             hist,bins = np.histogram(np.array(m),bins = np.arange(0, max, step), normed=normed)
#             C=hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             C=np.concatenate((C, hist.reshape((hist.shape[0], 1))), axis=1)
#
#     for a, m in enumerate(bp_artery_mutant):
#         if a == 0:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             M = hist.reshape((hist.shape[0], 1))
#         else:
#             hist, bins = np.histogram(np.array(m), bins=np.arange(0, max, step), normed=normed)
#             M = np.concatenate((M, hist.reshape((hist.shape[0], 1))), axis=1)
#     C=pd.DataFrame(np.array(C).transpose()).melt()
#     M=pd.DataFrame(np.array(M).transpose()).melt()
#     import seaborn as sns
#     sns.set_style(style='white')
#     sns.lineplot(x="variable", y="value", ci='sd', data=C)#, y="normalized count"
#     sns.lineplot(x="variable", y="value", ci='sd', data=M)#, err_style="bars"
#     sns.despine()
#     plt.legend(['controls', 'mutants'])
#     plt.title('bp of penetrating arteries in ' + condition, size='x-large')
#     plt.xlabel("bp of penetrating penetrating arteries", size='x-large')
#     plt.ylabel(" count",size='x-large')
#     plt.xticks(range(np.arange(0, max, step).shape[0]),np.arange(0, max, step))
#     plt.tight_layout()
#     plt.yscale('log')
#
#
#
#     ################## GAP BETWEEN L2/3 AND L4 ###########################
#
#     import seaborn as sns
#     control_gap=[]
#     mutant_gap=[]
#
#     vess_controls = np.array(bp_layer_artery_control).astype(float)[:, :]
#     for i in range(vess_controls.shape[0]):
#         m = np.mean(vess_controls[i])
#         s = np.std(vess_controls[i])
#         vess_controls[i] = (vess_controls[i] - m) / s
#         control_gap.append(vess_controls[i,2]-vess_controls[i,1])
#
#     vess_controls = np.array(bp_layer_artery_mutant).astype(float)[:, :]
#     for i in range(vess_controls.shape[0]):
#         m = np.mean(vess_controls[i])
#         s = np.std(vess_controls[i])
#         vess_controls[i] = (vess_controls[i] - m) / s
#         mutant_gap.append(vess_controls[i,2]-vess_controls[i,1])
#
#     plt.figure()
#     sns.despine()
#     colors = ['cadetblue', 'indianred']
#     box1 = plt.boxplot(control_gap, positions=[1], patch_artist=True, widths=0.5, showfliers=False, showmeans=True,
#                        autorange=True, meanline=True)
#     box2 = plt.boxplot(mutant_gap, positions=[2], patch_artist=True, widths=0.5, showfliers=False, showmeans=True,
#                        autorange=True, meanline=True)
#     for patch in box1['boxes']:
#         patch.set_facecolor(colors[0])
#     for patch in box2['boxes']:
#         patch.set_facecolor(colors[1])
#     plt.ylabel('counts', size='x-large')
#     plt.xticks([1, 2], ['controls', 'deprived'], size='x-large')
#     plt.yticks(size='x-large')
#     plt.xlim(0, 3)
#     plt.title('l2/3 - l4 BP gap in ' + condition, size='x-large')
#     plt.tight_layout()
#
#
#
#
#
#
#
#     ################# EP BP #######################################
#     import pandas as pd
#
#     baselined=False
#     normalized=False
#
#     if sub_region:
#         from sklearn.linear_model import LinearRegression
#         plt.figure()
#         vess_controls=np.array(bp_layer_artery_control).astype(float)[:, :]
#
#         if normalized:
#             for i in range(vess_controls.shape[0]):
#                 m=np.mean(vess_controls[i])
#                 s=np.std(vess_controls[i])
#                 vess_controls[i]=(vess_controls[i]-m)/s
#                 # vess_controls[i]=vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]
#
#
#         if baselined:
#             baseline = np.mean(vess_controls, axis=0)
#             model = LinearRegression()
#
#             for i in range(vess_controls.shape[0]):
#                 m=np.mean(vess_controls[i])
#                 s=np.std(vess_controls[i])
#                 vess_controls[i]=(vess_controls[i]-m)/s
#                 # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])  # vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controls[i, 2]
#                 model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
#                 a = model.coef_
#                 b = model.intercept_
#                 print('intercept:', b)
#                 print('slope:', a)
#                 vess_controls[i] = vess_controls[i] - a * baseline - b
#
#
#         C=[]
#         colors = ['royalblue', 'darkblue', 'forestgreen','lightseagreen']
#         for i in range(vess_controls.shape[1]):
#             c=vess_controls[:,i]
#             C.append(c.tolist())
#         for i in range(vess_controls.shape[0]):
#             plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')
#
#
#         if len(np.array(C).shape)==3:
#             C=np.squeeze(np.array(C), axis=2)
#         else:
#             C=np.array(C)
#
#         dfc = pd.DataFrame(np.array(C).transpose()).melt()
#         sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)
#
#         vess_controls=np.array(bp_layer_artery_mutant).astype(float)[:, :]
#         if normalized:
#             for i in range(vess_controls.shape[0]):
#                 m=np.mean(vess_controls[i])
#                 s=np.std(vess_controls[i])
#                 vess_controls[i]=(vess_controls[i]-m)/s
#         if baselined:
#             for i in range(vess_controls.shape[0]):
#                 # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]
#                 model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
#                 a=model.coef_
#                 b=model.intercept_
#                 print('intercept:', b)
#                 print('slope:', a)
#                 vess_controls[i]=vess_controls[i]-a*baseline-b
#         M=[]
#         colors=['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
#         for i in range(vess_controls.shape[1]):
#             m=vess_controls[:,i]
#             M.append(m.tolist())
#         for i in range(vess_controls.shape[0]):
#             plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')
#
#         if len(np.array(M).shape)==3:
#             M=np.squeeze(np.array(M), axis=2)
#         else:
#             M=np.array(M)
#
#         dfm = pd.DataFrame(np.array(M).transpose()).melt()
#         sns.lineplot(x="variable", y="value", err_style="bars",ci='sd',data=dfm)
#
#         from scipy.stats import ttest_ind
#
#         for i in range(C.shape[0]):
#             gc = np.array(dfc.loc[lambda dfc: dfc['variable'] == i, 'value'])
#             gm = np.array(dfm.loc[lambda dfm: dfm['variable'] == i, 'value'])
#             st, pval=ttest_ind(gc, gm)
#             print(condition, ls[i], pval)
#             if pval<0.06:
#                 print(condition, ls[i], pval, '!!!!!!!!!!!!!!!!11')
#         sns.despine()
#         plt.ylabel('nb of branch point in '+condition,size='x-large')
#         plt.xlabel('layers', size='x-large')
#         plt.xticks([0,1,2,3,4,5], ['l1', 'l2/3','l4','l5','l6a','l6b'], size='x-large')
#         # plt.xticks([0, 1, 2, 3, 4], ['l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
#         plt.yticks(size='x-large')
#         plt.legend(['controls','Deprived'])#Otof-/-
#         plt.title('nb of branch point per layer in '+condition, size='x-large')
#         # plt.legend(*zip(*labels), loc=2)
#         plt.legend(['controls','Deprived'])#Otof-/-
#         ax = plt.gca()
#         leg = ax.get_legend()
#         leg.legendHandles[0].set_color('blue')
#         leg.legendHandles[1].set_color('orange')
#         plt.tight_layout()
#
#         #################################
#         plt.figure()
#         vess_controls = np.array(ep_layer_artery_control).astype(float)[:, 1:]
#         if normalized:
#             for i in range(vess_controls.shape[0]):
#                 m=np.mean(vess_controls[i])
#                 s=np.std(vess_controls[i])
#                 vess_controls[i]=(vess_controls[i]-m)/s
#                 # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]
#
#         if baselined:
#             baseline = np.mean(vess_controls, axis=0)
#             model = LinearRegression()
#             for i in range(vess_controls.shape[0]):
#                 m=np.mean(vess_controls[i])
#                 s=np.std(vess_controls[i])
#                 vess_controls[i]=(vess_controls[i]-m)/s
#                 # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])  # vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controls[i, 2]
#                 model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
#                 a = model.coef_
#                 b = model.intercept_
#                 print('intercept:', b)
#                 print('slope:', a)
#                 vess_controls[i] = vess_controls[i] - a * baseline - b
#
#         C = []
#         colors = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
#         for i in range(vess_controls.shape[1]):
#             c = vess_controls[:, i]
#             C.append(c.tolist())
#
#         if len(np.array(C).shape) == 3:
#             C = np.squeeze(np.array(C), axis=2)
#         else:
#             C = np.array(C)
#
#         for i in range(vess_controls.shape[0]):
#             plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')
#
#         dfc = pd.DataFrame(np.array(C).transpose()).melt()
#         sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)
#
#
#         model=LinearRegression()
#         vess_controls = np.array(ep_layer_artery_mutant).astype(float)[:, 1:]
#         if normalized:
#             for i in range(vess_controls.shape[0]):
#                 m=np.mean(vess_controls[i])
#                 s=np.std(vess_controls[i])
#                 vess_controls[i]=(vess_controls[i]-m)/s
#         if baselined:
#             for i in range(vess_controls.shape[0]):
#                 # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controls[i, 2]
#                 model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
#                 a=model.coef_
#                 b=model.intercept_
#                 print('intercept:', b)
#                 print('slope:', a)
#                 vess_controls[i]=vess_controls[i]-a*baseline-b
#         M = []
#         colors = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
#         for i in range(vess_controls.shape[1]):
#             m = vess_controls[:, i]
#             M.append(m.tolist())
#         for i in range(vess_controls.shape[0]):
#             plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')
#
#         if len(np.array(M).shape) == 3:
#             M = np.squeeze(np.array(M), axis=2)
#         else:
#             M = np.array(M)
#
#         dfm = pd.DataFrame(np.array(M).transpose()).melt()
#         sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfm)
#
#         from scipy.stats import ttest_ind
#
#         for i in range(C.shape[0]):
#             gc = np.array(dfc.loc[lambda dfc: dfc['variable'] == i, 'value'])
#             gm = np.array(dfm.loc[lambda dfm: dfm['variable'] == i, 'value'])
#             st, pval = ttest_ind(gc, gm)
#             print(condition, ls[i], pval)
#             if pval < 0.06:
#                 print(condition, ls[i], pval, '!!!!!!!!!!!!!!!!11')
#         sns.despine()
#         plt.ylabel('nb of end point in ' + condition, size='x-large')
#         plt.xlabel('layers', size='x-large')
#         # plt.xticks([0, 1, 2, 3, 4, 5], ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
#         plt.xticks([0, 1, 2, 3, 4], ['l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
#         plt.yticks(size='x-large')
#         plt.legend(['controls', 'Deprived'])  # Otof-/-
#         ax = plt.gca()
#         # leg = ax.get_legend()
#         # leg.legendHandles[0].set_color('blue')
#         # leg.legendHandles[1].set_color('orange')
#         plt.title('nb of end point per layer in '+condition, size='x-large')
#         # plt.legend(*zip(*labels), loc=2)
#         plt.legend(['controls', 'Deprived'])  # Otof-/-
#         plt.tight_layout()
#
#
#         ##### orientation
#         if feature=='vessels':
#             plt.figure()
#             vess_controls = np.array(vess_rad_control).astype(float)[:, :]
#             if normalized:
#                 for i in range(vess_controls.shape[0]):
#                     m = np.mean(vess_controls[i])
#                     s = np.std(vess_controls[i])
#                     vess_controls[i] = (vess_controls[i] - m) / s
#                     # vess_controls[i]=vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]
#             if baselined:
#                 baseline = np.mean(vess_controls, axis=0)
#                 model = LinearRegression()
#
#                 for i in range(vess_controls.shape[0]):
#                     m = np.mean(vess_controls[i])
#                     s = np.std(vess_controls[i])
#                     vess_controls[i] = (vess_controls[i] - m) / s
#                     # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])  # vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controls[i, 2]
#                     model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
#                     a = model.coef_
#                     b = model.intercept_
#                     print('intercept:', b)
#                     print('slope:', a)
#                     vess_controls[i] = vess_controls[i] - a * baseline - b
#
#             C = []
#             colors = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
#             for i in range(vess_controls.shape[1]):
#                 c = vess_controls[:, i]
#                 C.append(c.tolist())
#             for i in range(vess_controls.shape[0]):
#                 plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')
#
#             if len(np.array(C).shape) == 3:
#                 C = np.squeeze(np.array(C), axis=2)
#             else:
#                 C = np.array(C)
#
#             dfc = pd.DataFrame(np.array(C).transpose()).melt()
#             sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc)
#
#             vess_controls = np.array(vess_rad_mutant).astype(float)[:, :]
#             # vess_controls = vess_controls[24:, :]
#             if normalized:
#                 for i in range(vess_controls.shape[0]):
#                     m = np.mean(vess_controls[i])
#                     s = np.std(vess_controls[i])
#                     vess_controls[i] = (vess_controls[i] - m) / s
#             if baselined:
#                 for i in range(vess_controls.shape[0]):
#                     # vess_controls[i] = vess_controls[i] / np.sum(vess_controls[i])#vess_controls[i, 1]#np.sum(vess_controls[i])#vess_controlsvess_controls[i, 2]
#                     model.fit(baseline.reshape(-1, 1), vess_controls[i].transpose().reshape(-1, 1))
#                     a = model.coef_
#                     b = model.intercept_
#                     print('intercept:', b)
#                     print('slope:', a)
#                     vess_controls[i] = vess_controls[i] - a * baseline - b
#             M = []
#             colors = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
#             for i in range(vess_controls.shape[1]):
#                 m = vess_controls[:, i]
#                 M.append(m.tolist())
#             for i in range(vess_controls.shape[0]):
#                 plt.plot(vess_controls[i], '--', color=colors[i], alpha=0.5, label='_nolegend_')
#
#             if len(np.array(M).shape) == 3:
#                 M = np.squeeze(np.array(M), axis=2)
#             else:
#                 M = np.array(M)
#
#             dfm = pd.DataFrame(np.array(M).transpose()).melt()
#             sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfm)
#
#             from scipy.stats import ttest_ind
#
#             for i in range(C.shape[0]):
#                 gc = np.array(dfc.loc[lambda dfc: dfc['variable'] == i, 'value'])
#                 gm = np.array(dfm.loc[lambda dfm: dfm['variable'] == i, 'value'])
#                 st, pval = ttest_ind(gc, gm)
#                 print(condition, ls[i], pval)
#                 if pval < 0.06:
#                     print(condition, ls[i], pval, '!!!!!!!!!!!!!!!!11')
#             sns.despine()
#             plt.ylabel('radial rate ' + condition, size='x-large')
#             plt.xlabel('layers', size='x-large')
#             plt.xticks([0, 1, 2, 3, 4, 5], ['l1', 'l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
#             # plt.xticks([0, 1, 2, 3, 4], ['l2/3', 'l4', 'l5', 'l6a', 'l6b'], size='x-large')
#             plt.yticks(size='x-large')
#             plt.legend(['controls', 'Deprived'])  # Otof-/-
#             plt.title('radial vessels orientation ' + condition, size='x-large')
#             # plt.legend(*zip(*labels), loc=2)
#             plt.legend(['controls', 'Deprived'])  # Otof-/-
#             ax = plt.gca()
#             leg = ax.get_legend()
#             leg.legendHandles[0].set_color('blue')
#             leg.legendHandles[1].set_color('orange')
#             plt.tight_layout()
#
#
#
#
#
#
#
#
#
#
#     ## VOX ARTERY RAW SIGNAL
#
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls=['142L','158L','162L', '164L']
#     mutants=['138L','141L', '163L', '165L']
#
#     template_shape=(320,528,228)
#     vox_shape = (320, 528, 228, len(controls))
#     vox_art_raw_signal_control = np.zeros(vox_shape)
#     vox_art_raw_signal_mutant = np.zeros(vox_shape)
#
#
#     radius=10
#
#
#     for i, g in enumerate(controls):
#         print(g)
#         graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correctedIsocortex.gt')
#         label = graph.vertex_annotation();
#         artery = from_e_prop2_vprop(graph, 'artery')
#         vertex_filter = np.logical_and(artery, graph.vertex_property('artery_raw') > 0)  # np.logical_and()
#         art_grt = graph.sub_graph(vertex_filter=vertex_filter)
#         art_coordinates = art_grt.vertex_property('coordinates_atlas')  # *1.625/25
#         # standardization
#         a_r = (art_grt.vertex_property('artery_raw') - np.mean(art_grt.vertex_property('artery_raw'))) / np.std(
#             art_grt.vertex_property('artery_raw'))
#         a_r = a_r + 10
#         # print('artBP')#artBP
#         v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=a_r, radius=(radius, radius, radius), method='sphere');
#         w = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
#         vox_art_raw_signal_control[:, :, :, i] = v.array / w.array
#
#     for i, g in enumerate(mutants):
#         print(g)
#         graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correctedIsocortex.gt')
#         label = graph.vertex_annotation();
#         artery = from_e_prop2_vprop(graph, 'artery')
#         vertex_filter = np.logical_and(artery, graph.vertex_property('artery_raw') > 0)  # np.logical_and()
#         art_grt = graph.sub_graph(vertex_filter=vertex_filter)
#         art_coordinates = art_grt.vertex_property('coordinates_atlas')  # *1.625/25
#         # standardization
#         a_r = (art_grt.vertex_property('artery_raw') - np.mean(art_grt.vertex_property('artery_raw'))) / np.std(
#             art_grt.vertex_property('artery_raw'))
#         a_r = a_r + 10
#         # print('artBP')#artBP
#         v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=a_r, radius=(radius, radius, radius), method='sphere');
#         w = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
#         vox_art_raw_signal_mutant[:, :, :, i] = v.array / w.array
#
#     io.write(work_dir + '/' +'vox_art_raw_signal_control'+str(radius)+'.tif', vox_art_raw_signal_control.astype('float32'))
#     io.write(work_dir + '/' +'vox_art_raw_signal_mutant'+str(radius)+'.tif', vox_art_raw_signal_mutant.astype('float32'))
#
#     vox_art_raw_signal_control_avg=np.mean(vox_art_raw_signal_control, axis=3)
#     vox_art_raw_signal_mutant_avg=np.mean(vox_art_raw_signal_mutant, axis=3)
#
#
#     io.write(work_dir + '/' +'vox_art_raw_signal_mutant_avg'+str(radius)+'.tif', vox_art_raw_signal_mutant_avg.astype('float32'))
#     io.write(work_dir + '/' +'vox_art_raw_signal_control_avg'+str(radius)+'.tif', vox_art_raw_signal_control_avg.astype('float32'))
#
#
#
#     from scipy import stats
#     pcutoff = 0.05
#
#     tvals, pvals = stats.ttest_ind(vox_art_raw_signal_control, vox_art_raw_signal_mutant, axis = 3, equal_var = True);
#
#     pi = np.isnan(pvals);
#     pvals[pi] = 1.0;
#     tvals[pi] = 0;
#
#     pvals2 = pvals.copy();
#     pvals2[pvals2 > pcutoff] = pcutoff;
#     psign=np.sign(tvals)
#
#
#     ## from sagital to coronal view
#     pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
#     psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
#     # pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
#
#     # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
#     pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])
#
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
#
#     import tifffile
#     tifffile.imsave(work_dir+'/pvalcolors_art_raw_signal_'+str(radius)+'.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)
#
#
#
#
#
#
#
#
#
#
#
#
#     ##heatmaps orientation and arteries
#
#     # controls=['2R','3R','5R', '8R']
#     # mutants=['1R','7R', '6R', '4R']
#     # work_dir='/data_SSD_2to/191122Otof'
#
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls=['142L','158L','162L', '164L']
#     mutants=['138L','141L', '163L', '165L']
#
#     template_shape=(320,528,228)
#     vox_shape = (320, 528, 228, len(controls))
#     vox_ori_control_rad = np.zeros(vox_shape)
#     vox_ori_mutant_rad = np.zeros(vox_shape)
#
#     vox_ori_control_plan = np.zeros(vox_shape)
#     vox_ori_mutant_plan = np.zeros(vox_shape)
#
#     vox_art_control = np.zeros(vox_shape)
#     vox_art_mutant = np.zeros(vox_shape)
#
#     radius=5
#
#
#     for i, g in enumerate(controls):
#         print(g)
#         graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correctedIsocortex.gt')
#         label = graph.vertex_annotation();
#         vertex_filter = from_e_prop2_vprop(graph, 'artery')
#         # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
#         r, p, l = getRadPlanOrienttaion(graph, graph, local_normal=True,  calc_art=True)
#         rad = (r / (r + p)) > 0.6
#         plan = (p / (r + p)) > 0.6
#         connectivity = graph.edge_connectivity()
#         coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
#         edges_centers = np.array(
#             [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])
#         art_coordinates = art_tree.vertex_property('coordinates_atlas')  # *1.625/25
#
#         # print('artBP')#artBP
#         # v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
#         # vox_art_control[:, :, :, i] = v
#
#         print('rad')
#         v = vox.voxelize(edges_centers[rad, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
#         w = vox.voxelize(edges_centers[:, :3], shape=template_shape, weights=None, radius=(radius, radius,radius), method='sphere');
#         vox_ori_control_rad[:, :, :, i] = v.array / w.array
#
#         # print('plan')
#         # v = vox.voxelize(edges_centers[plan, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
#         # # w=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(15,15,15), method = 'sphere');
#         # vox_ori_control_plan[:, :, :, i] = v.array / w.array
#         #
#         # print('rad/Plan')  # artBP
#         # vox_art_mutant[:, :, :, i] = vox_ori_mutant_rad[:, :, :, i]/vox_ori_mutant_plan[:, :, :, i]
#
#
#     for i, g in enumerate(mutants):
#         print(g)
#         graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correctedIsocortex.gt')
#         label = graph.vertex_annotation();
#         vertex_filter = from_e_prop2_vprop(graph, 'artery')
#         # art_tree = graph.sub_graph(vertex_filter=vertex_filter)
#         r, p, l = getRadPlanOrienttaion(graph, graph, local_normal=True, calc_art=True)
#         rad = (r / (r + p)) > 0.6
#         plan = (p / (r + p)) > 0.6
#         connectivity = graph.edge_connectivity()
#         coordinates = graph.vertex_property('coordinates_atlas')  # *1.625/25
#         edges_centers = np.array(
#             [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in range(connectivity.shape[0])])
#         art_coordinates = art_tree.vertex_property('coordinates_atlas')  # *1.625/25
#
#         # print('artBP')  # artBP
#         # v = vox.voxelize(art_coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
#         # vox_art_mutant[:, :, :, i] = v
#
#         print('rad')
#         v = vox.voxelize(edges_centers[rad, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
#         w = vox.voxelize(edges_centers[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
#         vox_ori_mutant_rad[:, :, :, i] = v.array / w.array
#
#         # print('plan')
#         # v = vox.voxelize(edges_centers[plan, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
#         # # w=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(15,15,15), method = 'sphere');
#         # vox_ori_mutant_plan[:, :, :, i] = v.array / w.array
#         #
#         # print('rad/Plan')  # artBP
#         # vox_art_mutant[:, :, :, i] = vox_ori_mutant_rad[:, :, :, i]/vox_ori_mutant_plan[:, :, :, i]
#
#     io.write(work_dir + '/' +'vox_ori_control_rad'+str(radius)+'.tif', vox_ori_control_rad.astype('float32'))
#     io.write(work_dir + '/' +'vox_ori_mutant_rad'+str(radius)+'.tif', vox_ori_mutant_rad.astype('float32'))
#
#     vox_ori_control_rad_avg=np.mean(vox_ori_control_rad, axis=3)
#     vox_ori_mutant_rad_avg=np.mean(vox_ori_mutant_rad, axis=3)
#     vox_ori_control_plan_avg=np.mean(vox_ori_control_plan, axis=3)
#     vox_ori_mutant_plan_avg=np.mean(vox_ori_mutant_plan, axis=3)
#     vox_ori_control_ratio_avg=np.mean(vox_art_control, axis=3)
#     vox_ori_mutant_ratio_avg=np.mean(vox_art_mutant, axis=3)
#
#     io.write(work_dir + '/' +'vox_ori_control_ratio_avg_'+str(radius)+'.tif', vox_ori_control_ratio_avg.astype('float32'))
#     io.write(work_dir + '/' +'vox_ori_mutant_ratio_avg_'+str(radius)+'.tif', vox_ori_mutant_ratio_avg.astype('float32'))
#
#     io.write(work_dir + '/' +'vox_ori_mutant_rad_avg_'+str(radius)+'.tif', vox_ori_mutant_rad_avg.astype('float32'))
#     io.write(work_dir + '/' +'vox_ori_control_rad_avg_'+str(radius)+'.tif', vox_ori_control_rad_avg.astype('float32'))
#
#     io.write(work_dir + '/' +'vox_ori_mutant_plan_avg_'+str(radius)+'.tif', vox_ori_mutant_plan_avg.astype('float32'))
#     io.write(work_dir + '/' +'vox_ori_control_plan_avg_'+str(radius)+'.tif', vox_ori_control_plan_avg.astype('float32'))
#
#
#     for i in range(len(controls)):
#         # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_control[:, :, :, i])
#         io.write(work_dir + '/' + controls[i] + '/' + 'vox_ori_ratio_' + controls[i] + '.tif', vox_art_control[:, :, :, i].astype('float32'))
#         io.write(work_dir + '/' + controls[i] + '/' + 'vox_ori_rad_'+controls[i]+'.tif', vox_ori_control_rad[:, :, :, i].astype('float32'))
#         io.write(work_dir + '/' + controls[i] + '/' + 'vox_ori_plan_'+controls[i]+'.tif', vox_ori_control_plan[:, :, :, i].astype('float32'))
#
#     for i in range(len(mutants)):
#         # np.save(work_dir + '/' + controls[i] + '/' + 'vox_art_'+str(radius)+'.npy', vox_art_mutant[:, :, :, i])
#         io.write(work_dir + '/' + mutants[i] + '/' + 'vox_ori_ratio_' + mutants[i] + '.tif', vox_art_mutant[:, :, :, i].astype('float32'))
#         io.write(work_dir + '/' + mutants[i] + '/' + 'vox_ori_rad_'+mutants[i]+'.tif', vox_ori_mutant_rad[:, :, :, i].astype('float32'))
#         io.write(work_dir + '/' + mutants[i] + '/' + 'vox_ori_plan_'+mutants[i]+'.tif', vox_ori_mutant_plan[:, :, :, i].astype('float32'))
#
#
#
#
#
#     from scipy import stats
#     pcutoff = 0.05
#
#     tvals, pvals = stats.ttest_ind(vox_ori_control_plan, vox_ori_mutant_plan, axis = 3, equal_var = True);
#
#     pi = np.isnan(pvals);
#     pvals[pi] = 1.0;
#     tvals[pi] = 0;
#
#     pvals2 = pvals.copy();
#     pvals2[pvals2 > pcutoff] = pcutoff;
#     psign=np.sign(tvals)
#
#
#     ## from sagital to coronal view
#     pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
#     psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
#     # pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
#
#     # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
#     pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])
#
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
#
#     import tifffile
#     tifffile.imsave(work_dir+'/pvalcolors_planORI_'+str(radius)+'.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)
#
#
#
#     from scipy import stats
#     pcutoff = 0.05
#
#     tvals, pvals = stats.ttest_ind(vox_ori_control_rad, vox_ori_mutant_rad, axis = 3, equal_var = True);
#
#     pi = np.isnan(pvals);
#     pvals[pi] = 1.0;
#     tvals[pi] = 0;
#
#     pvals2 = pvals.copy();
#     pvals2[pvals2 > pcutoff] = pcutoff;
#     psign=np.sign(tvals)
#
#
#     ## from sagital to coronal view
#     pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
#     psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
#     # pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
#
#     # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
#     pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])
#
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
#
#     import tifffile
#     tifffile.imsave(work_dir+'/pvalcolors_radORI_'+str(radius)+'.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)
#
#
#
#
#     tvals, pvals = stats.ttest_ind(vox_art_control, vox_art_mutant, axis = 3, equal_var = True);
#
#     pi = np.isnan(pvals);
#     pvals[pi] = 1.0;
#     tvals[pi] = 0;
#
#     pvals2 = pvals.copy();
#     pvals2[pvals2 > pcutoff] = pcutoff;
#     psign=np.sign(tvals)
#
#
#     ## from sagital to coronal view
#     pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
#     psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
#     # pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
#
#     # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
#     pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])
#
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', np.moveaxis(pvalscol, -1, 0).astype('float32'))
#     # io.write('/data_SSD_2to/191122Otof/pvalcolors.tif', pvalscol.astype('uint8'), photometric='rgb')
#
#     import tifffile
#     tifffile.imsave(work_dir+'/pvalcolors_arteriesBP_'+str(radius)+'.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)
#
#
#
#
#
#     with open('/data_SSD_2to/181002_4/atlas_volume_list.p', 'rb') as fp:
#       atlas_list = pickle.load(fp)
#
#     atlas=io.read('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation.tif')[:, :, :228]
#     atlas=np.flip(2)
#     atlas=np.swapaxes(atlas, 1, 0)
#     atlas_flat=atlas.reshape((atlas.shape[0]*atlas.shape[2]*atlas.shape[1], 1))
#
#     radOri=io.read('/data_SSD_2to/191122Otof/pvalcolors_radORI_'+str(radius)+'.tif')
#     radOri_flat=np.swapaxes(radOri, 1, 3).reshape((radOri.shape[0]*radOri.shape[2]*radOri.shape[3], 3))
#
#     planOri=io.read('/data_SSD_2to/191122Otof/pvalcolors_planORI_'+str(radius)+'.tif')
#     planOri_flat=np.swapaxes(planOri, 1, 3).reshape((planOri.shape[0]*planOri.shape[2]*planOri.shape[3], 3))
#
#     n=0
#     reg_name=[]
#     volumes=[]
#     for i, r in enumerate(reg_list.keys()):
#         name = ano.find(r, key='order')['name']
#         for j, se in enumerate(reg_list[r]):
#             acro=ano.find(se, key='order')['acronym']
#             if '6'not in acro:
#                 reg_name.append(acro)
#                 n = n + 1
#
#     pvalheatMapPos = np.zeros(n)
#     pvalheatMapNeg = np.zeros(n)
#     pvalheatMap=np.zeros(n)
#     k=0
#     for i, r in enumerate(reg_list.keys()):
#         name = ano.find(r, key='order')['name']
#         for j, se in enumerate(reg_list[r]):
#             id=ano.find(se, key='order')['id']
#             acro = ano.find(se, key='order')['acronym']
#             if '6' not in acro:
#                 reg=np.asarray(atlas_flat==id).nonzero()[0]
#                 vol=np.sum(reg)
#                 pvalheatMapPos[k]=np.sum(radOri_flat[reg, 1]>0)#/vol
#                 pvalheatMapNeg[k]=np.sum(radOri_flat[reg, 0] > 0)
#                 pvalheatMap[k]=(np.sum(radOri_flat[reg, 1]>0)-np.sum(radOri_flat[reg, 0]>0))#/vol
#                 k=k+1
#
#
#     import seaborn as sns
#     plt.figure()
#     sns.barplot(np.arange(n), pvalheatMapPos*(pvalheatMapPos>0), palette=sns.color_palette("Blues"))
#     sns.barplot(np.arange(n), -pvalheatMapNeg*(pvalheatMapNeg>0), palette=sns.color_palette("Reds"))
#     plt.xticks(np.arange(n)+0.5,reg_name, size='x-large', rotation=90)
#     sns.despine()
#     pos=pvalheatMap[np.asarray(pvalheatMap>0).nonzero()[0]]
#     neg=pvalheatMap[np.asarray(pvalheatMap<=0).nonzero()[0]]
#
#     posM=np.asarray(pvalheatMap>(np.mean(pos))).nonzero()[0]#+np.std(pos)/5
#     negM=np.asarray(pvalheatMap<=(np.mean(neg))).nonzero()[0]#-np.std(neg)/5
#
#     for p in posM:
#         plt.gca().get_xticklabels()[p].set_color("green")
#     for p in negM:
#         plt.gca().get_xticklabels()[p].set_color("red")
#
#
#     plt.title('pvalues radial vessels controls VS Otof-/-', size='x-large')
#
#     pvalheatMapPos = np.zeros(n)
#     pvalheatMapNeg = np.zeros(n)
#     pvalheatMap = np.zeros(n)
#     k=0
#     for i, r in enumerate(reg_list.keys()):
#         name = ano.find(r, key='order')['name']
#         for j, se in enumerate(reg_list[r]):
#             id=ano.find(se, key='order')['id']
#             acro = ano.find(se, key='order')['acronym']
#             if '6' not in acro:
#                 reg=np.asarray(atlas_flat==id).nonzero()[0]
#                 vol=np.sum(reg)
#                 pvalheatMapPos[k] = np.sum(planOri_flat[reg, 1] > 0)  # /vol
#                 pvalheatMapNeg[k] = np.sum(planOri_flat[reg, 0] > 0)
#                 pvalheatMap[k]=(np.sum(planOri_flat[reg, 1]>0)-np.sum(planOri_flat[reg, 0]>0))#/vol
#                 k=k+1
#
#
#
#     plt.figure()
#     # sns.barplot(np.arange(n), pvalheatMap*(pvalheatMap>0), palette=sns.color_palette("Blues"))
#     # sns.barplot(np.arange(n), pvalheatMap*(pvalheatMap<=0), palette=sns.color_palette("Reds"))
#     sns.barplot(np.arange(n), pvalheatMapPos*(pvalheatMapPos>0), palette=sns.color_palette("Blues"))
#     sns.barplot(np.arange(n), -pvalheatMapNeg*(pvalheatMapNeg>0), palette=sns.color_palette("Reds"))
#     plt.xticks(np.arange(n)+0.5,reg_name, size='x-large', rotation=90)
#     sns.despine()
#     pos=pvalheatMap[np.asarray(pvalheatMap>0).nonzero()[0]]
#     neg=pvalheatMap[np.asarray(pvalheatMap<=0).nonzero()[0]]
#
#     posM=np.asarray(pvalheatMap>(np.mean(pos))).nonzero()[0]#+np.std(pos)/3
#     negM=np.asarray(pvalheatMap<=(np.mean(neg))).nonzero()[0]#-np.std(neg)/3
#
#     for p in posM:
#         plt.gca().get_xticklabels()[p].set_color("green")
#     for p in negM:
#         plt.gca().get_xticklabels()[p].set_color("red")
#
#
#     plt.title('pvalues planar vessels controls VS Otof-/-', size='x-large')
#
#
#
#
#     ### cluster detection + 2d representation of clusters
#
#
#     # Compute DBSCAN
#     rad=np.swapaxes(radOri, 1, 3)
#     neg_rad=np.array(np.asarray(rad[:, :, :, 0]>0).nonzero()).T
#     from sklearn.cluster import DBSCAN
#     X_tofit=neg_rad
#     db = DBSCAN(eps=3, min_samples=10).fit(X_tofit)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#
#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#
#     print('Estimated number of clusters: %d' % n_clusters_)
#     print('Estimated number of noise points: %d' % n_noise_)
#
#
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     X_proj_transformed = pca.fit_transform(X_tofit)#X[:,[0,1,3,4,15,16]])
#
#     #
#     # from sklearn.manifold import TSNE
#     # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#     # X_proj_transformed = tsne.fit_transform(X_tofit)
#
#
#     fig=plt.figure()
#     ax = fig.add_subplot(111)#, projection='3d')
#     u, c = np.unique(labels , return_counts=True)
#     m=np.mean(c)
#     for i, n in enumerate(u):
#         if c[i]>m:
#             indtoplot=np.asarray(labels==n).nonzero()[0]
#             x_m=np.mean(X_proj_transformed[indtoplot,0])
#             y_m=np.mean(X_proj_transformed[indtoplot,1])
#             ax.scatter(x_m ,y_m,  s=c[i]/10, alpha=0.3)# X_proj_transformed[indtoplot,2]
#             regions=[]
#             for x_reg in X_tofit[indtoplot]:
#                 regions.append(atlas[x_reg[0], x_reg[1], x_reg[2]])
#             regions=np.array(regions)
#             u_regions, c_regions=np.unique(regions[regions>0], return_counts=True)
#             try:
#                 c_max=np.max(c_regions)
#                 cs=np.asarray(c_regions>=(c_max-(c_max/4))).nonzero()[0]
#                 u_regions=u_regions[cs]
#                 u_region=[ano.find(r, key='id')['acronym'] for r in u_regions]
#                 s='-'.join(u_region)
#                 plt.text(x_m, y_m, s, fontsize=12)
#                 print(u_region, c_regions)
#             except:
#                 print('non registered region')
#
#     # ax.view_init(30, 185)
#     plt.title('negative pvalues clusters radial vessels controls VS Otof-/-', size='x-large')
#     plt.tight_layout()
#     plt.show()
#
#
#
#     # Compute DBSCAN
#     rad=np.swapaxes(planOri, 1, 3)
#     neg_rad=np.array(np.asarray(rad[:, :, :, 0]>0).nonzero()).T
#     from sklearn.cluster import DBSCAN
#     X_tofit=neg_rad
#     db = DBSCAN(eps=3, min_samples=10).fit(X_tofit)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#
#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#
#     print('Estimated number of clusters: %d' % n_clusters_)
#     print('Estimated number of noise points: %d' % n_noise_)
#
#
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     X_proj_transformed = pca.fit_transform(X_tofit)#X[:,[0,1,3,4,15,16]])
#
#     #
#     # from sklearn.manifold import TSNE
#     # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#     # X_proj_transformed = tsne.fit_transform(X_tofit)
#
#
#     fig=plt.figure()
#     ax = fig.add_subplot(111)#, projection='3d')
#     u, c = np.unique(labels , return_counts=True)
#     m=np.mean(c)
#     for i, n in enumerate(u):
#         if c[i]>m:
#             indtoplot=np.asarray(labels==n).nonzero()[0]
#             x_m=np.mean(X_proj_transformed[indtoplot,0])
#             y_m=np.mean(X_proj_transformed[indtoplot,1])
#             ax.scatter(x_m ,y_m,  s=c[i]/10, alpha=0.3)# X_proj_transformed[indtoplot,2]
#             regions=[]
#             for x_reg in X_tofit[indtoplot]:
#                 regions.append(atlas[x_reg[0], x_reg[1], x_reg[2]])
#             regions=np.array(regions)
#             u_regions, c_regions=np.unique(regions[regions>0], return_counts=True)
#             try:
#                 c_max=np.max(c_regions)
#                 cs=np.asarray(c_regions>=(c_max-(c_max/4))).nonzero()[0]
#                 u_regions=u_regions[cs]
#                 u_region=[ano.find(r, key='id')['acronym'] for r in u_regions]
#                 s='-'.join(u_region)
#                 plt.text(x_m, y_m, s, fontsize=12)
#                 print(u_region, c_regions)
#             except:
#                 print('non registered region')
#
#     # ax.view_init(30, 185)
#     plt.title('negative pvalues clusters planar vessels controls VS Otof-/-', size='x-large')
#     plt.tight_layout()
#     plt.show()
#
#
#
#
#
#     # Compute DBSCAN
#     rad=np.swapaxes(radOri, 1, 3)
#     neg_rad=np.array(np.asarray(rad[:, :, :, 1]>0).nonzero()).T
#     from sklearn.cluster import DBSCAN
#     X_tofit=neg_rad
#     db = DBSCAN(eps=3, min_samples=10).fit(X_tofit)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#
#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#
#     print('Estimated number of clusters: %d' % n_clusters_)
#     print('Estimated number of noise points: %d' % n_noise_)
#
#
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     X_proj_transformed = pca.fit_transform(X_tofit)#X[:,[0,1,3,4,15,16]])
#
#     #
#     # from sklearn.manifold import TSNE
#     # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#     # X_proj_transformed = tsne.fit_transform(X_tofit)
#
#
#     fig=plt.figure()
#     ax = fig.add_subplot(111)#, projection='3d')
#     u, c = np.unique(labels , return_counts=True)
#     m=np.mean(c)
#     for i, n in enumerate(u):
#         if c[i]>m:
#             indtoplot=np.asarray(labels==n).nonzero()[0]
#             x_m=np.mean(X_proj_transformed[indtoplot,0])
#             y_m=np.mean(X_proj_transformed[indtoplot,1])
#             ax.scatter(x_m ,y_m,  s=c[i]/10, alpha=0.3)# X_proj_transformed[indtoplot,2]
#             regions=[]
#             for x_reg in X_tofit[indtoplot]:
#                 regions.append(atlas[x_reg[0], x_reg[1], x_reg[2]])
#             regions=np.array(regions)
#             u_regions, c_regions=np.unique(regions[regions>0], return_counts=True)
#             try:
#                 c_max=np.max(c_regions)
#                 cs=np.asarray(c_regions>=(c_max-(c_max/4))).nonzero()[0]
#                 u_regions=u_regions[cs]
#                 u_region=[ano.find(r, key='id')['acronym'] for r in u_regions]
#                 s='-'.join(u_region)
#                 plt.text(x_m, y_m, s, fontsize=12)
#                 print(u_region, c_regions)
#             except:
#                 print('non registered region')
#
#     # ax.view_init(30, 185)
#     plt.title('positive pvalues clusters radial vessels controls VS Otof-/-', size='x-large')
#     plt.tight_layout()
#     plt.show()
#
#
#
#     # Compute DBSCAN
#     rad=np.swapaxes(planOri, 1, 3)
#     neg_rad=np.array(np.asarray(rad[:, :, :, 1]>0).nonzero()).T
#     from sklearn.cluster import DBSCAN
#     X_tofit=neg_rad
#     db = DBSCAN(eps=3, min_samples=10).fit(X_tofit)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#
#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#
#     print('Estimated number of clusters: %d' % n_clusters_)
#     print('Estimated number of noise points: %d' % n_noise_)
#
#
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     X_proj_transformed = pca.fit_transform(X_tofit)#X[:,[0,1,3,4,15,16]])
#
#     #
#     # from sklearn.manifold import TSNE
#     # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#     # X_proj_transformed = tsne.fit_transform(X_tofit)
#
#
#     fig=plt.figure()
#     ax = fig.add_subplot(111)#, projection='3d')
#     u, c = np.unique(labels , return_counts=True)
#     m=np.mean(c)
#     for i, n in enumerate(u):
#         if c[i]>m:
#             indtoplot=np.asarray(labels==n).nonzero()[0]
#             x_m=np.mean(X_proj_transformed[indtoplot,0])
#             y_m=np.mean(X_proj_transformed[indtoplot,1])
#             ax.scatter(x_m ,y_m,  s=c[i]/10, alpha=0.3)# X_proj_transformed[indtoplot,2]
#             regions=[]
#             for x_reg in X_tofit[indtoplot]:
#                 regions.append(atlas[x_reg[0], x_reg[1], x_reg[2]])
#             regions=np.array(regions)
#             u_regions, c_regions=np.unique(regions[regions>0], return_counts=True)
#             try:
#                 c_max=np.max(c_regions)
#                 cs=np.asarray(c_regions>=(c_max-(c_max/4))).nonzero()[0]
#                 u_regions=u_regions[cs]
#                 u_region=[ano.find(r, key='id')['acronym'] for r in u_regions]
#                 s='-'.join(u_region)
#                 plt.text(x_m, y_m, s, fontsize=12)
#                 print(u_region, c_regions)
#             except:
#                 print('non registered region')
#
#     # ax.view_init(30, 185)
#     plt.title('positive pvalues clusters planar vessels controls VS Otof-/-', size='x-large')
#     plt.tight_layout()
#     plt.show()
#
#
#
#
#
#
#
#     #### test volcano plots data stat representation
#     from scipy import stats
#
#     condition='otof'#'whiskers'#otof
#     datatype='bp'##ori loops bp
#     # datatype='extractedFrac'
#
#     radius=5
#     if condition=='whiskers':
#         work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#         if datatype=='ori':
#             print(datatype, condition)
#             radOri=io.read(work_dir+'/pvalcolors_radORI4_'+str(radius)+'.tif')
#             vox_ori_control_rad=io.read(work_dir + '/' +'vox_ori_control_rad4'+str(radius)+'.tif')
#             vox_ori_mutant_rad=io.read(work_dir + '/' +'vox_ori_mutant_rad4'+str(radius)+'.tif')
#         elif datatype=='loops':
#             print(datatype, condition)
#             radOri = io.read(work_dir + '/pvalcolors_8_loops_cortex' + str(radius) + '.tif')
#             vox_ori_mutant_rad = io.read(work_dir + '/' +'vox_mutant_loops_cortex'+str(radius)+'.tif')
#             vox_ori_control_rad = io.read(work_dir + '/' +'vox_control_loops_cortex'+str(radius)+'.tif')
#         elif datatype=='bp':
#             print(datatype, condition)
#             radOri = io.read(work_dir + '/pvalcolors_BP_cortex_density_no142_165_F_005' + str(radius) + '.tif')
#             vox_ori_mutant_rad = io.read(work_dir + '/' + 'vox_mutant_BP_cortex_' + str(radius) + '.tif')
#             vox_ori_control_rad = io.read(work_dir + '/' + 'vox_control_BP_cortex_' + str(radius) + '.tif')
#         elif datatype=='extractedFrac':
#             print(datatype, condition)
#             radOri=io.read(work_dir+'/pvalcolors_8_extrac6ted_fraction_whole_cortex.tif')
#             vox_ori_mutant_rad = io.read(
#                 work_dir + '/' + 'vox_mutant_extracted_fraction_whole_cortex_' + str(radius) + '.tif')
#             vox_ori_control_rad = io.read(
#                 work_dir + '/' + 'vox_control_extracted_fraction_whole_cortex_' + str(radius) + '.tif')
#         control=vox_ori_control_rad
#         mutant=vox_ori_mutant_rad
#         # control=vox_control #
#         # mutant=vox_mutant #
#         atlas = io.read('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation.tif')[:, :,:230]
#         atlas = np.flip(atlas,axis=1)
#
#     elif condition=='otof':
#         radius = 10
#         work_dir='/data_SSD_1to/otof6months/6Mvs6M'#'/data_SSD_2to/191122Otof'
#         if datatype == 'ori':
#             print(datatype, condition)
#             radOri = io.read(work_dir + '/pvalcolors_radORI4_' + str(radius) + '.tif')
#             vox_ori_control_rad = io.read(work_dir + '/' + 'vox_ori_control_rad4' + str(radius) + '.tif')
#             vox_ori_mutant_rad = io.read(work_dir + '/' + 'vox_ori_mutant_rad4' + str(radius) + '.tif')
#         elif datatype == 'loops':
#             print(datatype, condition)
#             radOri = io.read(work_dir + '/pvalcolors_8_loops_cortex' + str(radius) + '.tif')
#             vox_ori_mutant_rad = io.read(work_dir + '/' + 'vox_mutant_loops_cortex' + str(radius) + '.tif')
#             vox_ori_control_rad = io.read(work_dir + '/' + 'vox_control_loops_cortex' + str(radius) + '.tif')
#         elif datatype == 'bp':
#             print(datatype, condition)
#             radOri = io.read(work_dir + '/pvalcolors_BP density_' + str(radius) + '.tif')
#             vox_ori_mutant_rad = io.read(work_dir + '/' + 'vox_mutant' + str(radius) + '.tif')
#             vox_ori_control_rad = io.read(work_dir + '/' + 'vox_control' + str(radius) + '.tif')
#         elif datatype == 'extractedFrac':
#             print(datatype, condition)
#             radOri = io.read(work_dir + '/pvalcolors_8_extrac6ted_fraction_whole_cortex.tif')
#             vox_ori_mutant_rad = io.read(work_dir + '/' + 'vox_mutant_extracted_fraction_whole_cortex_' + str(radius) + '.tif')
#             vox_ori_control_rad = io.read(work_dir + '/' + 'vox_control_extracted_fraction_whole_cortex_' + str(radius) + '.tif')
#
#         control = vox_ori_control_rad
#         mutant = vox_ori_mutant_rad
#         # control=vox_control #
#         # mutant=vox_mutant #
#         atlas = io.read('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation.tif')[:, :,:230]
#
#
#     pcutoff = 0.05
#     reg_l_int=['AUD', 'VIS', 'bfd', 'IC', 'll', 'CN', 'SOCl', 'SSp-n', 'SSp-m', 'MOs', 'MOp', 'SSp-ul', 'SSp-ll', ]
#
#     # if datatype == 'extractedFrac':
#     #     rad=radOri
#     #     # neg_rad = np.array(np.asarray(radOri[:, :, :, 0] > 0).nonzero()).T
#     #     # pos_rad = np.array(np.array(radOri[:, :, :, 1] > 0).nonzero()).T
#     #
#
#
#     if datatype=='bp':
#         if condition=='whiskers':
#             tvals, pvals = stats.ttest_ind(control[:,:,:,1:], mutant[:, :, :, :-1], axis = 3, equal_var = False);
#         else:
#             tvals, pvals = stats.ttest_ind(control[:, :, :, :], mutant[:, :, :, :], axis=3, equal_var=False);
#     else:
#         tvals, pvals = stats.ttest_ind(control[:,:,:,:], mutant, axis = 3, equal_var = False);
#     # tvals=np.swapaxes(tvals, 1, 0)
#     # pvals=np.swapaxes(pvals, 1, 0)
#     #
#     # pvals=np.swapaxes(pvals, 0, 2)
#     # tvals=np.swapaxes(tvals, 0, 2)
#
#     pi = np.isnan(pvals);
#     pvals[pi] = 1.0;
#     tvals[pi] = 0;
#
#     pvals2 = pvals.copy();
#     pvals2[pvals2 > pcutoff] = pcutoff;
#     psign=np.sign(tvals)
#
#     radOri = colorPValues(pvals2, psign, positive = [255,0,0], negative = [0,255,0])
#     rad=radOri
#     # Compute DBSCAN
#     # rad=np.swapaxes(radOri, 1, 3)
#     # rad=np.swapaxes(rad, 1, 0)
#     neg_rad=np.array(np.asarray(rad[:, :, :, 0]>0).nonzero()).T
#     pos_rad=np.array(np.array(rad[:, :, :, 1]>0).nonzero()).T
#
#     if atlas.shape[:-1]==rad.shape[:-2]:
#         print('True')
#     print(atlas.shape, rad.shape, pvals.shape, tvals.shape)
#
#     from sklearn.cluster import DBSCAN
#     X_tofit=neg_rad
#     db = DBSCAN(eps=3, min_samples=10).fit(X_tofit)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#
#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#
#     print('Estimated number of clusters: %d' % n_clusters_)
#     print('Estimated number of noise points: %d' % n_noise_)
#
#     reg_l_int=['AUD',  'IC',]
#     fig=plt.figure()
#     ax = fig.add_subplot(111)#, projection='3d')
#     u, c = np.unique(labels , return_counts=True)
#     m=1#np.mean(c)
#     for i, n in enumerate(u):
#         # if c[i]>m:
#             indtoplot=np.asarray(labels==n).nonzero()[0]
#             pvals_avg=[]
#             tvals_avg= []
#             regions=[]
#             for x_reg in X_tofit[indtoplot]:
#                 regions.append(atlas[x_reg[0], x_reg[1], x_reg[2]])
#                 pvals_avg.append(pvals[x_reg[0], x_reg[1], x_reg[2]])
#                 tvals_avg.append(tvals[x_reg[0], x_reg[1], x_reg[2]])
#             regions=np.array(regions)
#             u_regions, c_regions=np.unique(regions[regions>0], return_counts=True)
#             x_m = np.log10(np.mean(tvals_avg))
#             y_m = np.log10(c[i])#-np.log(np.mean(pvals_avg))
#
#             try:
#                 color = 'grey'
#                 c_max=np.max(c_regions)
#                 cs=np.asarray(c_regions>=(c_max-(c_max/2))).nonzero()[0]
#                 u_regions=u_regions[cs]
#                 u_region=[ano.find(r, key='id')['acronym'] for r in u_regions]
#                 s='-'.join(u_region)
#                 b = False
#                 for r in reg_l_int:
#                     if r in s:
#                         plt.text(x_m, y_m, s, fontsize=12)
#                         if -np.log(np.mean(pvals_avg)) > 4:
#                             b = True
#                             color = 'indianred'
#
#                 if s != 'NoL':#and '6' not in s:
#                     if c[i] > np.mean(c):
#                         color = 'indianred'
#                         for r in reg_l_int:
#                             if r in s:
#                                 plt.text(x_m, y_m, s, fontsize=12)
#                     if b:
#                         for r in reg_l_int:
#                             if r in s:
#                                 plt.text(x_m, y_m, s, fontsize=12)
#                         color = 'indianred'
#                     if c[i] > np.mean(c) + 1*np.std(c):
#                         for r in reg_l_int:
#                             if r in s:
#                                 plt.text(x_m, y_m, s, fontsize=12)
#                         # color = 'indianred'
#                     if -np.log(np.mean(pvals_avg)) > 4:
#                         for r in reg_l_int:
#                             if r in s:
#                                 plt.text(x_m, y_m, s, fontsize=12)
#                         color = 'indianred'
#                     if abs(x_m) > 1.5:
#                         for r in reg_l_int:
#                             if r in s:
#                                 plt.text(x_m, y_m, s, fontsize=12)
#                         color = 'indianred'
#                 print(u_region, c_regions, x_m, y_m, c[i])
#                 ax.scatter(x_m, y_m, alpha=0.3, color=color, s=100)  # X_proj_transformed[indtoplot,2] # s=c[i] / 10,
#             except:
#                 print('non registered region')
#
#
#
#     reg_l_int=['AUD', 'bfd', 'IC', 'MOs', 'MOp','SSp-n']
#     X_tofit=pos_rad
#     db = DBSCAN(eps=3, min_samples=10).fit(X_tofit)
#     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#     core_samples_mask[db.core_sample_indices_] = True
#     labels = db.labels_
#
#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#
#     print('Estimated number of clusters: %d' % n_clusters_)
#     print('Estimated number of noise points: %d' % n_noise_)
#
#     ax = fig.add_subplot(111)#, projection='3d')
#     u, c = np.unique(labels , return_counts=True)
#     m=1#np.mean(c)
#     for i, n in enumerate(u):
#         if c[i]>m:
#             indtoplot=np.asarray(labels==n).nonzero()[0]
#             pvals_avg=[]
#             tvals_avg= []
#             regions=[]
#             for x_reg in X_tofit[indtoplot]:
#                 regions.append(atlas[x_reg[0], x_reg[1], x_reg[2]])
#                 pvals_avg.append(pvals[x_reg[0], x_reg[1], x_reg[2]])
#                 tvals_avg.append(tvals[x_reg[0], x_reg[1], x_reg[2]])
#             regions=np.array(regions)
#             u_regions, c_regions=np.unique(regions[regions>0], return_counts=True)
#             x_m=-np.log10(-np.mean(tvals_avg))
#             y_m=np.log10(c[i])#-np.log(np.mean(pvals_avg))
#
#             try:
#                 color='grey'
#                 c_max=np.max(c_regions)
#                 cs=np.asarray(c_regions>=(c_max-(c_max/3))).nonzero()[0]
#                 u_regions=u_regions[cs]
#                 u_region=[ano.find(r, key='id')['acronym'] for r in u_regions]
#                 s='-'.join(u_region)
#                 b=False
#                 for r in reg_l_int:
#                     if r in s:
#                         # plt.text(x_m, y_m, s, fontsize=12)
#                         if -np.log(np.mean(pvals_avg)) > 4:
#                             b=True
#
#                 if s != 'NoL':# and '6' not in s:
#                     if c[i] > np.mean(c):
#                         for r in reg_l_int:
#                             if r in s:
#                                 plt.text(x_m, y_m, s, fontsize=12)
#                         color = 'cadetblue'
#                     if b:
#                         plt.text(x_m, y_m, s, fontsize=12)
#                         color = 'cadetblue'
#                     if c[i] > np.mean(c)+np.std(c):
#                         for r in reg_l_int:
#                             if r in s:
#                                 plt.text(x_m, y_m, s, fontsize=12)
#                         color = 'cadetblue'
#                     if -np.log(np.mean(pvals_avg))>4:
#                         for r in reg_l_int:
#                             if r in s:
#                                 plt.text(x_m, y_m, s, fontsize=12)
#                         color = 'cadetblue'
#                     if abs(x_m)>1.5:
#                         for r in reg_l_int:
#                             if r in s:
#                                 plt.text(x_m, y_m, s, fontsize=12)
#                         color = 'cadetblue'
#                 print(u_region, c_regions, x_m, y_m, c[i])
#                 ax.scatter(x_m, y_m, alpha=0.3, color=color, s=100)  # X_proj_transformed[indtoplot,2] # s=c[i] / 10,
#             except:
#                 print('non registered region')
#
#     # ax.view_init(30, 185)
#     if datatype=='ori':
#         s='radial vessels'
#     elif datatype=='loops':
#         s='loops'
#     elif datatype=='bp':
#         s='bp density'
#
#     plt.title(' significative clusters '+s+ ' controls VS '+condition, size='x-large')
#     plt.tight_layout()
#     # plt.xscale('log')
#     # plt.yscale('log')
#     plt.xticks(size='x-large')
#     plt.yticks(size='x-large')
#     plt.ylabel('log cluster size in voxels', size='x-large')
#     plt.xlabel('-log tvals', size='x-large')
#
#     sns.despine()
#
#
#
#
#
#
#
#
#
#     ## RIV
#     controls=['2R','3R','5R', '8R']#['3R']#
#     mutants=['1R','7R', '6R', '4R']
#     work_dir='/data_SSD_2to/191122Otof'
#
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls=['142L','158L','162L', '164L']
#     mutants=['138L','141L', '163L', '165L']
#
#     region_list = [(6,6)]#[(142, 8)]  # auditory
#     from ClearMap.KirschoffAnalysis import *
#
#     Rc=[]
#     Ic=[]
#     Vc=[]
#     Dc=[]
#     Vbc=[]
#     Ec=[]
#     for i,control in enumerate(controls):
#         print(control)
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         omegas_c, currents_c, tension_nodale_c, tension_branche_c, edge_degs_c, currents_branche_c, extracted_fraction_c = getRIV(graph, work_dir + '/' + control, region_list[0])
#         Rc.append(omegas_c)
#         Ic.append(currents_branche_c)
#         Vc.append(tension_nodale_c)
#         Dc.append(edge_degs_c)
#         Vbc.append(tension_branche_c)
#         Ec.append(extracted_fraction_c)
#
#
#     Rm=[]
#     Im=[]
#     Vm=[]
#     Dm=[]
#     Vbm=[]
#     Em=[]
#     for i,control in enumerate(mutants):
#         print(control)
#         graph = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         omegas_m, currents_m, tension_nodale_m, tension_branche_m, edge_degs_m, currents_branche_m, extracted_fraction_m = getRIV(graph, work_dir + '/' + control, region_list[0])
#         Rm.append(omegas_m)
#         Im.append(currents_branche_m)
#         Vm.append(tension_nodale_m)
#         Dm.append(edge_degs_m)
#         Vbm.append(tension_branche_m)
#         Em.append(extracted_fraction_m)
#
#
#     #
#     with open(work_dir+'/control_'+'Rc'+'.p', 'wb') as fp:
#       pickle.dump(Rc, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/control_'+'Ic'+'.p', 'wb') as fp:
#       pickle.dump(Ic, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/control_'+'Vc'+'.p', 'wb') as fp:
#       pickle.dump(Vc, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/control_'+'Dc'+'.p', 'wb') as fp:
#       pickle.dump(Dc, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/control_'+'Vbc'+'.p', 'wb') as fp:
#       pickle.dump(Vbc, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/control_'+'Ec'+'.p', 'wb') as fp:
#       pickle.dump(Ec, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
#     with open(work_dir+'/mutant_'+'Rm'+'.p', 'wb') as fp:
#       pickle.dump(Rm, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/mutant_'+'Im'+'.p', 'wb') as fp:
#       pickle.dump(Im, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/mutant_'+'Vm'+'.p', 'wb') as fp:
#       pickle.dump(Vm, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/mutant_'+'Dm'+'.p', 'wb') as fp:
#       pickle.dump(Dm, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/mutant_'+'Vbm'+'.p', 'wb') as fp:
#       pickle.dump(Vbm, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open(work_dir+'/mutant_'+'Em'+'.p', 'wb') as fp:
#       pickle.dump(Em, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
#
#     np.save(work_dir+'/control_'+'Rc'+'.npy', Rc)
#     np.save(work_dir+'/control_'+'Ic'+'.npy', Ic)
#     np.save(work_dir+'/control_'+'Vc'+'.npy', Vc)
#     np.save(work_dir+'/control_'+'Dc'+'.npy', Dc)
#     np.save(work_dir+'/control_'+'Vbc'+'.npy', Vbc)
#     np.save(work_dir+'/control_'+'Ec'+'.npy', Ec)
#
#     np.save(work_dir+'/control_'+'Rm'+'.npy', Rm)
#     np.save(work_dir+'/control_'+'Im'+'.npy', Im)
#     np.save(work_dir+'/control_'+'Vm'+'.npy', Vm)
#     np.save(work_dir+'/control_'+'Dm'+'.npy', Dm)
#     np.save(work_dir+'/control_'+'Vbm'+'.npy', Vbm)
#     np.save(work_dir+'/control_'+'Em'+'.npy', Em)
#
#
#
#
#
#     Rc=np.load(work_dir+'/control_'+'Rc'+'.npy')
#     Ic=np.load(work_dir+'/control_'+'Ic'+'.npy')
#     Vc=np.load(work_dir+'/control_'+'Vc'+'.npy')
#     Dc=np.load(work_dir+'/control_'+'Dc'+'.npy')
#     Vbc=np.load(work_dir+'/control_'+'Vbc'+'.npy')
#     Ec=np.load(work_dir+'/control_'+'Ec'+'.npy')
#
#     Rm=np.load(work_dir+'/control_'+'Rm'+'.npy')
#     Im=np.load(work_dir+'/control_'+'Im'+'.npy')
#     Vm=np.load(work_dir+'/control_'+'Vm'+'.npy')
#     Dm=np.load(work_dir+'/control_'+'Dm'+'.npy')
#     Vbm=np.load(work_dir+'/control_'+'Vbm'+'.npy')
#     Em=np.load(work_dir+'/control_'+'Em'+'.npy')
#     #
#     # Rc=np.load(work_dir+'/control_'+'Rc'+'.npy',allow_pickle=True)
#     # Ic=np.load(work_dir+'/control_'+'Ic'+'.npy',allow_pickle=True)
#     # Vc=np.load(work_dir+'/control_'+'Vc'+'.npy',allow_pickle=True)
#     # Dc=np.load(work_dir+'/control_'+'Dc'+'.npy',allow_pickle=True)
#     # Vbc=np.load(work_dir+'/control_'+'Vbc'+'.npy',allow_pickle=True)
#     # Ec=np.load(work_dir+'/control_'+'Ec'+'.npy',allow_pickle=True)
#     #
#     # Rm=np.load(work_dir+'/control_'+'Rm'+'.npy',allow_pickle=True)
#     # Im=np.load(work_dir+'/control_'+'Im'+'.npy',allow_pickle=True)
#     # Vm=np.load(work_dir+'/control_'+'Vm'+'.npy',allow_pickle=True)
#     # Dm=np.load(work_dir+'/control_'+'Dm'+'.npy',allow_pickle=True)
#     # Vbm=np.load(work_dir+'/control_'+'Vbm'+'.npy',allow_pickle=True)
#     # Em=np.load(work_dir+'/control_'+'Em'+'.npy',allow_pickle=True)
#
#
#
#
#     with open(work_dir + '/control_' + 'Rc' + '.p', 'rb') as fp:
#         Rc=pickle.load(fp)
#
#     with open(work_dir + '/control_' + 'Ic' + '.p', 'rb') as fp:
#         Ic=pickle.load(fp)
#
#     with open(work_dir + '/control_' + 'Vc' + '.p', 'rb') as fp:
#         Vc=pickle.load(fp)
#
#     with open(work_dir + '/control_' + 'Dc' + '.p', 'rb') as fp:
#         Dc=pickle.load(fp)
#
#     with open(work_dir + '/control_' + 'Vbc' + '.p', 'rb') as fp:
#         Vbc=pickle.load(fp)
#
#     with open(work_dir + '/control_' + 'Ec' + '.p', 'rb') as fp:
#         Ec=pickle.load(fp)
#
#     with open(work_dir + '/mutant_' + 'Rm' + '.p', 'rb') as fp:
#         Rm=pickle.load(fp)
#
#     with open(work_dir + '/mutant_' + 'Im' + '.p', 'rb') as fp:
#         Im=pickle.load(fp)
#
#     with open(work_dir + '/mutant_' + 'Vm' + '.p', 'rb') as fp:
#         Vm=pickle.load(fp)
#
#     with open(work_dir + '/mutant_' + 'Dm' + '.p', 'rb') as fp:
#         Dm=pickle.load(fp)
#
#     with open(work_dir + '/mutant_' + 'Vbm' + '.p', 'rb') as fp:
#         Vbm=pickle.load(fp)
#
#     with open(work_dir + '/mutant_' + 'Em' + '.p', 'rb') as fp:
#         Em=pickle.load(fp)
#
#
#     ## plot graph with extrcated fraction value
#     i=0
#     control=controls[i]
#     graph_c = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#     diff = np.load(work_dir + '/' + control + '/sbm/'+'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
#     graph_c.add_vertex_property('overlap', diff)
#     order, level = region_list[0]
#     print(level, order, ano.find_name(order, key='order'))
#     label = graph_c.vertex_annotation();
#     label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#     vertex_filter = label_leveled == order;
#
#     g_aud = graph_c.sub_graph(vertex_filter=vertex_filter)
#     indices=g_aud.vertex_property('overlap')
#     u, c = np.unique(indices, return_counts=True)
#     medium_cluster = u[np.asarray(np.logical_and(c >= 200, c <= 1200)).nonzero()[0]]
#     print(medium_cluster.shape)
#     r = random.choice(medium_cluster)
#     j=np.asarray(u==r).nonzero()[0][0]
#     print('cluster number : ', r)
#     vf = np.zeros(g_aud.n_vertices)
#     vf[np.asarray(indices == r).nonzero()[0]] = 1
#     print('containing ', np.sum(vf), ' vertices')
#     g2plot = g_aud.sub_graph(vertex_filter=vf)
#     g2plot.add_edge_property('extracted_frac', Ec[i][j])
#     col=getColorMap_from_vertex_prop(g2plot.edge_property('extracted_frac'))
#     v_arteries = g2plot.edge_property('artery')
#     v_veins = g2plot.edge_property('vein')
#     col[v_arteries == 1] = [1., 0.0, 0.0, 1.0]
#     col[v_veins == 1] = [0.0, 0.0, 1.0, 1.0]
#     p = p3d.plot_graph_mesh(g2plot, edge_colors=col, n_tube_points=3)
#
#
#
#     ## voxelizarioin loops and E
#     region_list=[(6,6)]
#     vox_shape=(320,528,228, len(controls))
#     vox_control=np.zeros(vox_shape)
#     vox_mutant=np.zeros(vox_shape)
#     i=0
#     # control=controls[i]
#     for k,control in enumerate(controls):
#         print(k, control)
#         graph_c = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         diff = np.load(work_dir + '/' + control + '/sbm/'+'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
#         graph_c.add_vertex_property('overlap', diff)
#         order, level = region_list[0]
#         # print(level, order, ano.find_name(order, key='order'))
#         label = graph_c.vertex_annotation();
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         vertex_filter = label_leveled == order;
#
#         g_aud = graph_c.sub_graph(vertex_filter=vertex_filter)
#         indices=g_aud.vertex_property('overlap')
#         u, c = np.unique(indices, return_counts=True)
#         n=0
#         j_u=0
#         for i,e in enumerate(u):
#             j = np.asarray(u == e).nonzero()[0][0]
#             print(j.shape)
#             vf = np.zeros(g_aud.n_vertices)
#             vf[np.asarray(indices == e).nonzero()[0]] = 1
#             print(g_aud, vf.shape, 'containing ', np.sum(vf), ' vertices')
#
#             g2plot = g_aud.sub_graph(vertex_filter=vf)
#             g2plot=g2plot.largest_component()
#             if g2plot.n_vertices <= 3000 and g2plot.n_edges >= 5:
#                 connectivity=g2plot.edge_connectivity()
#                 coordinates=g2plot.vertex_property('coordinates_atlas')#*1.625/25
#                 # coordinates_control=gts_control.vertex_property('coordinates')#*1.6/25
#                 print(g2plot, len(Ec[k][j_u]))
#                 if g2plot.n_edges==len(Ec[k][j_u]):
#                     if g2plot.n_vertices!=0:
#                         if g2plot.n_edges!=0:
#                             if n == 0:
#                                 edges_centers = np.array(
#                                     [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in
#                                      range(connectivity.shape[0])])
#                                 E = Ec[k][j_u]
#                                 print(E.shape[0] == edges_centers.shape[0], k, j,j_u, Ec[k][j_u].shape, edges_centers.shape)
#                             else:
#                                 edges_centers = np.concatenate((edges_centers, np.array(
#                                     [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in
#                                      range(connectivity.shape[0])])), axis=0)
#                                 # print(edges_centers.shape)
#                                 print(k, j_u, np.max(j_u))
#                                 try:
#                                     E = np.concatenate((E, Ec[k][j_u]), axis=0)
#                                 except IndexError:
#                                     print(e, u.shape, i, j_u, np.where(u==e))
#                                     # print(E.shape)
#                                     # if (E.shape[0] != edges_centers.shape[0]):
#                                     #     print('False', k, j, Ec[k][j].shape)
#
#                             # print(Ec[k][j_u].shape, E.shape)
#                             j_u = j_u + 1
#                             n=n+1
#             # except:
#             #     print('empty graph')
#
#         vox_data=np.concatenate((edges_centers, np.expand_dims(E, axis=1)), axis=1)
#         v = vox.voxelize(vox_data[:, :3], shape = (320,528,228), weights=vox_data[:, 3], radius=(5,5,5), method = 'sphere');
#         w=vox.voxelize(vox_data[:, :3], shape =  (320,528,228),  weights=None, radius=(15,15,15), method = 'sphere');
#
#         vox_control[:, :, :, k] = v.array/w.array
#
#     for k, control in enumerate(mutants):
#         print(k, control)
#         graph_c = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         diff = np.load(
#             work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
#         graph_c.add_vertex_property('overlap', diff)
#         order, level = region_list[0]
#         # print(level, order, ano.find_name(order, key='order'))
#         label = graph_c.vertex_annotation();
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         vertex_filter = label_leveled == order;
#
#         g_aud = graph_c.sub_graph(vertex_filter=vertex_filter)
#         indices = g_aud.vertex_property('overlap')
#         u, c = np.unique(indices, return_counts=True)
#         n = 0
#         j_u = 0
#         for e in u:
#             j = np.asarray(u == e).nonzero()[0][0]
#             vf = np.zeros(g_aud.n_vertices)
#             vf[np.asarray(indices == e).nonzero()[0]] = 1
#             print(g_aud, vf.shape, 'containing ', np.sum(vf), ' vertices')
#
#             g2plot = g_aud.sub_graph(vertex_filter=vf)
#             g2plot = g2plot.largest_component()
#             if g2plot.n_vertices <= 3000 and g2plot.n_edges >= 5:
#                 connectivity = g2plot.edge_connectivity()
#                 coordinates = g2plot.vertex_property('coordinates_atlas')  # *1.625/25
#                 # coordinates_control=gts_control.vertex_property('coordinates')#*1.6/25
#                 print(g2plot, len(Em[k][j_u]))
#                 if g2plot.n_edges == len(Em[k][j_u]):
#                     if g2plot.n_vertices != 0:
#                         if g2plot.n_edges != 0:
#                             if n == 0:
#                                 edges_centers = np.array(
#                                     [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in
#                                      range(connectivity.shape[0])])
#                                 E = Em[k][j_u]
#                                 print(E.shape[0] == edges_centers.shape[0], k, j, j_u, Ec[k][j_u].shape, edges_centers.shape)
#                             else:
#                                 edges_centers = np.concatenate((edges_centers, np.array(
#                                     [(coordinates[connectivity[i, 0]] + coordinates[connectivity[i, 1]]) / 2 for i in
#                                      range(connectivity.shape[0])])), axis=0)
#                                 # print(edges_centers.shape)
#
#                                 E = np.concatenate((E, Em[k][j_u]), axis=0)
#                                 # print(E.shape)
#                                 # if (E.shape[0] != edges_centers.shape[0]):
#                                 #     print('False', k, j, Em[k][j].shape)
#
#                             print(Em[k][j_u].shape, E.shape)
#                             j_u = j_u + 1
#                             n = n + 1
#             # except:
#             #     print('empty graph')
#
#         vox_data = np.concatenate((edges_centers, np.expand_dims(E, axis=1)), axis=1)
#         v = vox.voxelize(vox_data[:, :3], shape=(320, 528, 228), weights=vox_data[:, 3], radius=(5, 5, 5), method='sphere');
#         w = vox.voxelize(vox_data[:, :3], shape=(320, 528, 228), weights=None, radius=(15, 15, 15), method='sphere');
#
#         vox_mutant[:, :, :, k] = v.array / w.array
#     # p3d.plot(v.array/w.array)
#
#     io.write(work_dir + '/' +'vox_control_extracted_fraction_whole_cortex_'+str(radius)+'.tif', vox_control.astype('float32'))
#     io.write(work_dir + '/' +'vox_mutant_extracted_fraction_whole_cortex_'+str(radius)+'.tif', vox_mutant.astype('float32'))
#
#
#     from scipy import stats
#     pcutoff = 0.05
#
#     tvals, pvals = stats.ttest_ind(vox_control[:, :, :, :-1], vox_mutant, axis = 3, equal_var = True);
#
#     pi = np.isnan(pvals);
#     pvals[pi] = 1.0;
#     tvals[pi] = 0;
#
#     pvals2 = pvals.copy();
#     pvals2[pvals2 > pcutoff] = pcutoff;
#     psign=np.sign(tvals)
#
#
#     ## from sagital to coronal view
#     pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
#     psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
#     # pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
#
#     # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
#     pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])
#
#     import tifffile
#     tifffile.imsave(work_dir + '/pvalcolors_8_extrac6ted_fraction_whole_cortex.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)
#
#
#     vox_control_avg=np.mean(vox_control, axis=3)
#     vox_mutant_avg=np.mean(vox_mutant, axis=3)
#
#     # for i in range(len(controls)):
#     #     io.write(work_dir +'/' + 'vox_arteries_' + controls[i] + '.tif', vox_art_control[:, :, :, i].astype('float32'))
#     #
#     # for i in range(len(mutants)):
#     #     io.write(work_dir + '/' + 'vox_arteries_' + mutants[i] + '.tif', vox_art_mutant[:, :, :, i].astype('float32'))
#     #
#     #
#
#     io.write(work_dir + '/' +'vox_control_extracted_fraction_whole_cortex_avg_'+str(radius)+'.tif', vox_control_avg.astype('float32'))
#     io.write(work_dir + '/' +'vox_mutant_extracted_fraction_whole_cortex_avg_'+str(radius)+'.tif', vox_mutant_avg.astype('float32'))
#
#
#
#
#
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls=['142L','158L','162L', '164L']
#     mutants=['138L','141L', '163L', '165L']#'138L',
#
#     template_shape=(320,528,228)
#     vox_shape=(320,528,228, len(controls))
#     vox_control=np.zeros(vox_shape)
#     vox_mutant=np.zeros(vox_shape)
#
#     radius=5
#
#     condition='cortex'
#     #loops
#     CP=[]
#     for n, cont in enumerate(controls):
#         CyclePos = np.load(work_dir + '/' + cont + '/cyclesPos'+condition+'all_cortex_Extended_Meth.npy')
#         if n==0:
#             CP=CyclePos
#         else:
#             CP=np.concatenate((CP, CyclePos), axis=0)
#
#         i=0
#         # control=controls[i]
#         # CP=np.load(work_dir + '/' + control + '/cyclesPos' + condition + '.npy')
#         vox_data = CP
#         # v = vox.voxelize(vox_data[:, :3], shape = (320,528,228), weights=vox_data[:, 3], radius=(5,5,5), method = 'sphere');
#         w=vox.voxelize(vox_data[:, :3], shape =  (320,528,228),  weights=None, radius=(radius,radius,radius), method = 'sphere');
#         vox_control[:, :, :, n] = w
#
#
#     #loops
#     CP=[]
#     for n, cont in enumerate(mutants):
#         CyclePos = np.load(work_dir + '/' + cont + '/cyclesPos'+condition+'all_cortex_Extended_Meth.npy')
#         if n==0:
#             CP=CyclePos
#         else:
#             CP=np.concatenate((CP, CyclePos), axis=0)
#
#         i=0
#         # control=controls[i]
#         # CP=np.load(work_dir + '/' + control + '/cyclesPos' + condition + '.npy')
#         vox_data = CP
#         # v = vox.voxelize(vox_data[:, :3], shape = (320,528,228), weights=vox_data[:, 3], radius=(5,5,5), method = 'sphere');
#         w=vox.voxelize(vox_data[:, :3], shape =  (320,528,228),  weights=None, radius=(radius,radius,radius), method = 'sphere');
#         vox_mutant[:, :, :, n] = w
#
#
#
#     vox_mutant_avg=np.mean(vox_mutant, axis=3)
#     vox_control_avg=np.mean(vox_control, axis=3)
#
#
#     io.write(work_dir + '/' +'vox_mutant_loops_'+condition+'_avg'+str(radius)+'.tif', np.swapaxes(np.swapaxes(vox_mutant_avg, 0,2), 1,2).astype('float32'))
#     io.write(work_dir + '/' +'vox_control_loops_'+condition+'_avg'+str(radius)+'.tif', np.swapaxes(np.swapaxes(vox_control_avg, 0,2), 1,2).astype('float32'))
#
#     from scipy import stats
#     pcutoff = 0.05
#
#     tvals, pvals = stats.ttest_ind(vox_control, vox_mutant, axis = 3, equal_var = True);#[:, :, :, :-1]
#
#     pi = np.isnan(pvals);
#     pvals[pi] = 1.0;
#     tvals[pi] = 0;
#
#     pvals2 = pvals.copy();
#     pvals2[pvals2 > pcutoff] = pcutoff;
#     psign=np.sign(tvals)
#
#
#     ## from sagital to coronal view
#     pvals2_f=np.swapaxes(np.swapaxes(pvals2, 0,2), 1,2)
#     psign_f=np.swapaxes(np.swapaxes(psign, 0,2), 1,2)
#     # pvals = self.cutoffPValues(pvals, pcutoff = pcutoff);
#
#     # pvals, psign = tTestVoxelization(vox_control_avg, vox_mutant_avg, pcutoff = None, signed = True);
#     pvalscol = colorPValues(pvals2_f, psign_f, positive = [255,0,0], negative = [0,255,0])
#
#     import tifffile
#     tifffile.imsave(work_dir+'/pvalcolors_8_loops_'+condition+str(radius)+'.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)
#
#
#
#
#     p3d.plot(w.array)
#
#
#
#     import seaborn as sns
#
#     ## artery territories size histograms
#     Nc=[]
#     Nm=[]
#     for k,control in enumerate(controls):
#         graph_c = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         diff= np.load(work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
#         graph_c.add_vertex_property('overlap', diff)
#         order, level = region_list[0]
#         # print(level, order, ano.find_name(order, key='order'))
#         label = graph_c.vertex_annotation();
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         vertex_filter = label_leveled == order;
#
#         g_aud = graph_c.sub_graph(vertex_filter=vertex_filter)
#         diff = g_aud.vertex_property('overlap')
#         u, c =np.unique(diff, return_counts=True)
#         if k==0:
#             cont=c
#         else:
#             cont=np.concatenate((cont, c))
#         Nc.append(u.shape[0])
#
#     for k, control in enumerate(mutants):
#         graph_c = ggt.load(work_dir + '/' + control + '/' + 'data_graph_correctedIsocortex.gt')
#         diff = np.load(
#             work_dir + '/' + control + '/sbm/' + 'diffusion_penetrating_vessel_art_end_point_cluster_per_region_iteration_Isocortex_graph_corrected' + '.npy')
#         graph_c.add_vertex_property('overlap', diff)
#         order, level = region_list[0]
#         # print(level, order, ano.find_name(order, key='order'))
#         label = graph_c.vertex_annotation();
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         vertex_filter = label_leveled == order;
#
#         g_aud = graph_c.sub_graph(vertex_filter=vertex_filter)
#         diff = g_aud.vertex_property('overlap')
#         u, c = np.unique(diff, return_counts=True)
#         if k == 0:
#             contm = c
#         else:
#             contm = np.concatenate((contm, c))
#         Nm.append(u.shape[0])
#
#
#     plt.figure()
#     # plt.hist(cont, bins=20, alpha=0.3)
#     sns.distplot(cont, bins=100,  hist_kws={'alpha': 0.0})
#     sns.despine()
#     plt.ylabel('counts', size='x-large')
#     plt.xlabel('artery territories size', size='x-large')
#     plt.xticks(size='x-large')
#     plt.yticks(size='x-large')
#     plt.xlim(0,1500)
#     plt.title('Controls artery territories size' , size='x-large')
#     # plt.legend(*zip(*labels), loc=2)
#     plt.legend(['controls', 'Otof-/-'])
#     plt.tight_layout()
#     plt.yscale('log')
#
#
#
#     from scipy.stats import ttest_ind
#     st, pval=ttest_ind(cont, contm)
#
#     # plt.figure()
#     # plt.hist(contm, bins=20, alpha=0.3)
#     sns.distplot(contm, bins=100,  hist_kws={'alpha': 0.0})
#     sns.despine()
#     plt.ylabel('counts', size='x-large')
#     plt.xlabel('artery territories size', size='x-large')
#     plt.xticks(size='x-large')
#     plt.yticks(size='x-large')
#     plt.xlim(0,1500)
#     plt.title('Mutants artery territories size' , size='x-large')
#     # plt.legend(*zip(*labels), loc=2)
#     plt.legend(['controls', 'Otof-/-'])
#     plt.tight_layout()
#     plt.yscale('log')
#
#     plt.yscale('linear')
#
#     plt.figure()
#     box1 = plt.boxplot(Nc, positions=[1], patch_artist=True, widths=0.5, showfliers=False, showmeans=True,
#                        autorange=True, meanline=True)
#     box2 = plt.boxplot(Nm, positions=[2], patch_artist=True, widths=0.5, showfliers=False, showmeans=True,
#                        autorange=True, meanline=True)
#     for patch in box1['boxes']:
#         patch.set_facecolor(colors[0])
#     for patch in box2['boxes']:
#         patch.set_facecolor(colors[1])
#
#     plt.ylabel('counts', size='x-large')
#     plt.xticks([1,2], ['controls', 'Otof-/-'], size='x-large')
#     plt.yticks(size='x-large')
#     plt.xlim(0,3)
#     plt.title('Controls artery territories size' , size='x-large')
#     # plt.legend(*zip(*labels), loc=2)
#     # plt.legend(['controls', 'Otof-/-'])
#     plt.tight_layout()
#
#
#     # omegas_c=omegas.copy()
#     # currents_c=currents.copy()
#     # tension_nodale_c=tension_nodale.copy()
#     # edge_degs_c=edge_degs.copy()
#
#     V=np.ravel(np.array(Vc))
#     I=np.ravel(np.array(Ic))
#     Vb=np.ravel(np.array(Vbc))
#     E=np.ravel(np.array(Ec))
#     # plt.figure()
#     # plt.hist(V, bins=100)
#
#     del y
#     plt.figure(10)
#     for V_ in V:
#         V_ = np.ravel(np.array(V_))
#         for v in V_:
#             try:
#                 y=np.concatenate((y, np.asarray(v)))
#             except:
#                 print('first')
#                 y=np.asarray(v)
#         y_f = y[~np.isnan(y)]
#     # y=np.squeeze(np.asarray(y), axis=0)
#     plt.hist(y_f, bins=100, alpha=0.3, normed=True, histtype='stepfilled')
#     plt.yscale('log')
#     plt.title('vertex tension')
#     plt.legend(['controls', 'mutants'])
#
#     del x
#     plt.figure(11)
#     for I_ in I:
#         I_ = np.ravel(np.array(I_))
#         for i in I_:
#             try:
#                 x=np.concatenate((x, i))
#             except:
#                 print('first')
#                 x=i
#     x_f = x[~np.isnan(x)]
#     # x = I[~np.isnan(I)]
#     plt.hist(abs(x_f), bins=100, alpha=0.3, normed=True, histtype='stepfilled')
#     plt.yscale('log')
#     plt.title('intensity')
#     plt.legend(['controls', 'mutants'])
#     #
#     # del z
#     # # plt.figure(12)
#     # for v in Vb:
#     #     try:
#     #         z=np.concatenate((z, np.asarray(v)))
#     #     except:
#     #         print('first')
#     #         z=np.asarray(v)
#
#     #
#     # plt.figure(12)
#     # z=z*x
#     # z = z[~np.isnan(z)]
#     # plt.hist(z, bins=100, alpha=0.3, normed=True, histtype='stepfilled')
#     # plt.yscale('log')
#     # plt.title('vessels power')
#     # plt.legend(['controls', 'mutants'])
#
#     del e
#     plt.figure(13)
#     for E_ in E:
#         E_ = np.ravel(np.array(E_))
#         for i in E_:
#             try:
#                 e=np.concatenate((e, i))
#             except:
#                 print('first')
#                 e=i
#     e_f = e[~np.isnan(e)]
#     e_f = e[~np.isinf(e)]
#     # x = I[~np.isnan(I)]
#     plt.hist(abs(e_f), bins=100, alpha=0.3, normed=True, histtype='stepfilled')
#     plt.yscale('log')
#     plt.title('ectracted fraction')
#     plt.legend(['controls', 'mutants'])
#
#
#
#
#     V=np.ravel(Vm)
#     I=np.ravel(Im)
#     Vb=np.ravel(Vbm)
#     E=np.ravel(Em)
#     # plt.figure()
#     # plt.hist(V, bins=100)
#
#
#     del y
#     plt.figure(10)
#     for V_ in V:
#         V_ = np.ravel(np.array(V_))
#         for v in V_:
#             try:
#                 y=np.concatenate((y, np.asarray(v)))
#             except:
#                 print('first')
#                 y=np.asarray(v)
#     y_f = y[~np.isnan(y)]
#     # y=np.squeeze(np.asarray(y), axis=0)
#     plt.hist(y_f, bins=100, alpha=0.3, normed=True, histtype='stepfilled')
#     plt.yscale('log')
#     plt.title('vertex tension')
#     plt.legend(['controls', 'mutants'])
#
#     del x
#     plt.figure(11)
#     for I_ in I:
#         I_ = np.ravel(np.array(I_))
#         for i in I_:
#             try:
#                 x=np.concatenate((x, i))
#             except:
#                 print('first')
#                 x=i
#     x_f = x[~np.isnan(x)]
#     # x = I[~np.isnan(I)]
#     plt.hist(abs(x_f), bins=100, alpha=0.3, normed=True, histtype='stepfilled')
#     plt.yscale('log')
#     plt.title('intensity')
#     plt.legend(['controls', 'mutants'])
#
#     #
#     # del z
#     # # plt.figure(12)
#     # for v in Vb:
#     #     try:
#     #         z=np.concatenate((z, np.asarray(v)))
#     #     except:
#     #         print('first')
#     #         z=np.asarray(v)
#
#     #
#     # plt.figure(12)
#     # z=z*x
#     # z = z[~np.isnan(z)]
#     # plt.hist(z, bins=100, alpha=0.3, normed=True, histtype='stepfilled')
#     # plt.yscale('log')
#     # plt.title('vessels power')
#     # plt.legend(['controls', 'mutants'])
#
#
#     del e
#     plt.figure(13)
#     for E_ in E:
#         E_ = np.ravel(np.array(E_))
#         for i in E_:
#             try:
#                 e=np.concatenate((e, i))
#             except:
#                 print('first')
#                 e=i
#     e_f = e[~np.isnan(e)]
#     e_f = e[~np.isinf(e)]
#     # x = I[~np.isnan(I)]
#     plt.hist(abs(e_f), bins=100, alpha=0.3, normed=True, histtype='stepfilled')
#     plt.yscale('log')
#     plt.title('ectracted fraction')
#     plt.legend(['controls', 'mutants'])
#
#
#     plt.figure()
#     for i in range(10):
#         v=0
#         while v < 300:
#             id = random.choice(range(u.shape[0]))
#             v = c[id]
#         print(id)
#         plt.scatter(edge_degs[id], currents[id])
#     plt.ylabel('current intensity')
#     plt.xlabel('edge vertex degree')
#
#
#     for i,mutant in enumerate(mutants):
#         print(mutant)
#         graph = ggt.load(work_dir + '/' + mutant + '/' + 'data_graph.gt')
#         coordinates=graph.vertex_property('coordinates_atlas')#*1.625/25
#         v=vox.voxelize(coordinates[:, :3], shape=template_shape,  weights=None, radius=(15,15,15), method = 'sphere');
#         vox_mutant[:,:,:,i]=v
#
#
#
#
#     ###########  plot screenshot
#
#
#
#     import ClearMap.Analysis.Graphs.GraphGt_old as ggto
#
#     work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
#     # controls=['142L','158L','162L', '164L']
#     # mutants=['138L','141L', '163L', '165L']
#
#     controls=['2R','3R','5R', '8R']#cpmntrol
#     mutants=['1R','7R', '6R', '4R']#mutant
#
#
#     controls=['2R','3R','5R', '8R']#['2R','3R','5R', '8R']
#     mutants=['1R','7R', '6R', '4R']
#     work_dir='/data_SSD_2to/191122Otof'
#
#     work_dir='/data_SSD_2to/whiskers_graphs/fluoxetine'
#     mutants=['1','2','3', '4', '6', '18']
#     controls=['21','22', '23']
#     # work_dir='/data_SSD_2to/191122Otof'
#
#     mutants = ['1', '2', '4', '7', '8', '9']
#     work_dir = '/data_2to/DBA2J'
#
#     template_shape=(320,528,228)
#     g='9'#'2R'
#     bin=20
#     N=5
#     print(g)
#     rad_r=[]
#     plan_r=[]
#     rad_all=[]
#     # for g in controls:
#         print(g)
#         # graph = ggto.load(work_dir + '/' + g + '/' + 'data_graph_correctedIsocortex.gt')#_correctedIsocortex
#         graph = ggto.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')#_correctedIsocortex
#         graph=ggto.load('/data_2to/p7/4/data_graph.gt')
#
#         graph=ggto.load('/data_2to/p0/4_graph2.gt')
#         graph=ggto.load('/mnt/vol00-renier/Sophie/elisa/210125-3_p7/3_graph.gt')
#         with open(work_dir + '/' + g + '/sampledict' + g + '.pkl', 'rb') as fp:
#             sampledict = pickle.load(fp)
#
#         degrees = graph.vertex_degrees()
#         vf = np.logical_and(degrees > 1, degrees <= 4)
#         graph = graph.sub_graph(vertex_filter=vf)
#         pressure = np.asarray(sampledict['pressure'][0])
#         graph.add_vertex_property('pressure', pressure)
#
#         flow = np.asarray(sampledict['flow'][0])
#         graph.add_vertex_property('flow', flow)
#         ps=7.3
#         e = 1 - np.exp(-(ps / abs(flow)))
#         graph.add_edge_property('extracted_frac', e)
#         # label = graph.vertex_annotation();
#         # order, level = ano.find('MOp', key='acronym')['order'],ano.find('MOp', key='acronym')['level']
#         # print(level, order, ano.find_name(order, key='order'))
#         # label = graph.vertex_annotation();
#         # label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         # region=[(142, 8), (149, 8), (128, 8), (156, 8)]#AUD
#
#         # region
#         # region=[(103, 8)]#SSs
#         region=[(54, 9), (47, 9)]#NOSE
#         vertex_filter=np.zeros(graph.n_vertices)
#         for reg in region:
#             order, level=reg
#             vertex_filter =np.logical_or(vertex_filter, label_leveled == order)
#         #
#         gs=graph.sub_graph(vertex_filter=vertex_filter)
#         # radii=gs.edge_radii()
#         # hist_r, bins_r = np.histogram(radii, bins=bin, normed=normed)
#         # rad_all.append(hist_r)
#
#
#         # # # 285,305#305,315 #360,370(mop)
#         arteries=gs.edge_property('artery')
#         gs=gs.sub_graph(edge_filter=arteries)
#
#
#         # r, p, l=getRadPlanOrienttaion(gs, graph, True, True)
#         # #
#         # artery_color = np.array([[1, 0, 0, 1], [0,1,0, 1]]).astype('float')
#         # edge_colors=artery_color[np.asarray((r/(r+p))>0.6, dtype=int)]
#         # edge_colors[np.asarray((p/(r+p))>0.6, dtype=bool)]=[0,0,1,1]
#
#         # graph.add_edge_property('color', edge_colors)
#         #
#         #
#         # gsr=gs.sub_graph(edge_filter=np.asarray((r/(r+p))>0.6, dtype=bool))
#         # gsp=gs.sub_graph(edge_filter=np.asarray((p/(r+p))>0.6, dtype=bool))
#         # gs = graph.sub_slice((slice(1,320), slice(528-250,528-240), slice(1,228)),coordinates='coordinates_atlas');
#         gss = graph.sub_slice((slice(1,320), slice(300,310), slice(1,228)),coordinates='coordinates_atlas');
#
#         # gs = graph.sub_slice((slice(1,5000), slice(1600,1700), slice(1,5000)))
#         # gs = graph.sub_slice((slice(1,320), slice(267,277), slice(1,228)),coordinates='coordinates_atlas');
#         # gs = graph.sub_slice((slice(1,320), slice(228,243), slice(1,228)),coordinates='coordinates_atlas');
#         # gs = graph.sub_slice((slice(1, 320), slice(327, 337), slice(1, 228)), coordinates='coordinates_atlas');
#         # vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
#
#         colorVal=np.zeros((gss.n_edges, 4))
#         artery=gss.edge_property('artery')
#         vein=gss.edge_property('vein')
#         arteries=np.asarray(artery>0).nonzero()[0]
#         veins=np.asarray(vein>0).nonzero()[0]
#         red_blue_map = {1: [1.0, 0, 0, 1.0], 0: [0.0, 1.0, 0.0, 1.0], 2: [0.0, 0, 1.0, 1.0]}
#         for i in range(gss.n_edges):
#             # print(j)
#             if i in arteries:
#                 colorVal[i] = red_blue_map[1]
#             elif i in veins:
#                 colorVal[i] = red_blue_map[2]
#             else:
#                 colorVal[i] = red_blue_map[0]
#         p = p3d.plot_graph_mesh(gss, edge_colors=colorVal)
#
#         q1 = p3d.plot_graph_mesh(gs, n_tube_points=3,  fov=0);#
#         ##get viridis colormap orientation
#         # r, p, l = getRadPlanOrienttaion(gs, graph, True, True)
#         # rad=np.nan_to_num(r/(r+p))
#         ef=gs.vertex_property('pressure')
#         edge_colors=getColorMap_from_vertex_prop(ef)
#
#         q1 = p3d.plot_graph_mesh(gs, n_tube_points=3, vertex_colors=edge_colors, fov=0);# vertex_colors=vertex_colors,edge_colors=edge_colors
#         ## radius edge_colors
#         # radii=gsr.edge_radii()
#         # hist_r, bins_r = np.histogram(radii, bins=bin, normed=normed)
#         # rad_r.append(hist_r)
#         #
#         # radii = gsp.edge_radii()
#         # hist_r, bins_r = np.histogram(radii, bins=bin, normed=normed)
#         # plan_r.append(hist_r)
#
#
#
#
#
#     work_dir = '/data_SSD_2to/whiskers_graphs/new_graphs'
#     controls = ['142L', '158L', '162L', '164L']
#     mutants = ['7', '9']
#     mutants = ['7']
#     work_dir='/data_SSD_2to/191122Otof'
#     brain='5R'
#
#
#
#     region=[(54, 9), (47, 9)]
#
#     import ClearMap.Analysis.Graphs.GraphGt_old as ggto
#     region=[(19, 8)]#MOp
#     region=[(191, 8)]#VISp
#     region=[(251, 7)]#ILA
#     region=[(0,0)]#all brain
#     graph = ggto.load(work_dir+'/'+brain+'/'+'data_graph_correcteduniverse.gt')
#     from matplotlib import cm
#
#     ### OLD ORIENTATION VERSION
#     # r, p,n, l = getRadPlanOrienttaion(gs, graph,  local_normal=True, calc_art=False, verbose=False)
#     # # rad=np.nan_to_num(r/(r+p))
#     # angle=np.array([math.acos(r[i]) for i in range(r.shape[0])])*180/pi
#
#     with open(work_dir + '/' + brain + '/sampledict' + brain + '.pkl', 'rb') as fp:
#         sampledict = pickle.load(fp)
#
#     degrees = graph.vertex_degrees()
#     vf = np.logical_and(degrees > 1, degrees <= 4)
#     graph = graph.sub_graph(vertex_filter=vf)
#     pressure = np.asarray(sampledict['pressure'][0])
#     graph.add_vertex_property('pressure', pressure)
#     angle = GeneralizedRadPlanorientation(graph)
#
#     rad=np.nan_to_num(angle)
#     artery_color = np.array([[0.82, 0.71, 0.55, 1], [1,0,0, 1]]).astype('float')
#     edge_colors=artery_color[np.asarray(rad<45, dtype=int)]
#     # edge_colors=artery_color[np.asarray(rad>0.6, dtype=int)]
#     # edge_colors[np.asarray((p/(r+p))>0.6, dtype=bool)]=[0,1,0,1]
#     # edge_colors[np.asarray((p/(r+p))>0.75, dtype=bool)]=[0,1,0,1]
#
#     edge_colors[np.asarray(rad>45, dtype=bool)]=[0,1,0,1]
#     graph.add_edge_property('color', edge_colors)
#
#
#     # region=[(6,6)]#NOSE
#     # region=[(0,0)]#all brain
#     # label = graph.vertex_annotation();
#     #
#     # vertex_filter=np.zeros(graph.n_vertices)
#     # for reg in region:
#     #     order, level=reg
#     #     label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#     #     vertex_filter =np.logical_or(vertex_filter, label_leveled == order)
#     # #
#     # gs=graph.sub_graph(vertex_filter=vertex_filter)
#
#     gs=graph.sub_graph(edge_filter=rad>=50)
#
#
#     gs = gs.sub_slice((slice(1, 320), slice(285, 295), slice(1, 228)), coordinates='coordinates_atlas');
#     # gs = gs.sub_slice((slice(1, 320), slice(445, 455), slice(1, 228)), coordinates='coordinates_atlas');
#
#     col= gs.edge_property('color')
#     q1 = p3d.plot_graph_mesh(gs, n_tube_points=3, edge_colors=col)
#
#     import pandas as pd
#     import ClearMap.Analysis.Graphs.GraphGt_old as ggto
#     graph = ggto.load('/data_SSD_2to/covid19/data_graph.gt')
#     covidLevel=graph.vertex_property('artery_raw')
#     # rad=np.nan_to_num(r/(r+p))
#     edge_colors=getColorMap_from_vertex_prop(covidLevel)
#     graph.add_edge_property('color', edge_colors)
#     gs = graph.sub_slice((slice(1, 320), slice(270, 280), slice(1, 228)), coordinates='coordinates_atlas');
#     col= graph.edge_property('color')
#     q2 = p3d.plot_graph_mesh(gs, n_tube_points=3, edge_colors=col,fov=0);# vertex_colors=vertex_colors,
#
#     thresh=0.5
#     bin_nb=10
#     C_bp = []
#     C_ori = []
#     for mut in controls:
#         graph = ggto.load('/data_SSD_2to/whiskers_graphs/new_graphs/' + mut + '/data_graph.gt')
#         label = graph.vertex_annotation();
#         level = 9
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         # region=[(142, 8), (149, 8), (128, 8), (156, 8)]#AUD
#         # region
#         region=[(47, 9)]#NOSE
#         # region=[(54, 9), (47, 9)]#SNOUT
#         vertex_filter = np.zeros(graph.n_vertices)
#         for reg in region:
#             order, level = reg
#             vertex_filter = np.logical_or(vertex_filter, label_leveled == order)
#         #
#         gsnout = graph.sub_graph(vertex_filter=vertex_filter)
#         r, p, l = getRadPlanOrienttaion(gsnout, graph, True, True)
#         rad = np.nan_to_num(r / (r + p))
#         bpd = gsnout.vertex_property('distance_to_surface')
#         bpd = gsnout.vertex_property('distance_to_surface')
#         normed = False
#         hbpd, bins = np.histogram(bpd, bins=bin_nb, normed=normed)
#
#         depth = gsnout.edge_property('distance_to_surface')
#         normed = True
#         hist_ori, bins_ori_dist = np.histogram(depth[rad > thresh], bins=bins, normed=normed)
#         C_bp.append(hbpd)
#         C_ori.append(hist_ori)
#
#
#     M_bp=[]
#     M_ori=[]
#     for mut in mutants:
#         graph = ggto.load('/data_SSD_2to/covid19/'+mut+'/data_graph.gt')
#         label = graph.vertex_annotation();
#         level=9
#         label_leveled = ano.convert_label(label, key='order', value='order', level=level)
#         # region=[(142, 8), (149, 8), (128, 8), (156, 8)]#AUD
#         # region
#         region=[(47, 9)]#NOSE
#         # region=[(54, 9), (47, 9)]#SNOUT
#         vertex_filter=np.zeros(graph.n_vertices)
#         for reg in region:
#             order, level=reg
#             vertex_filter =np.logical_or(vertex_filter, label_leveled == order)
#         #
#         gsnout=graph.sub_graph(vertex_filter=vertex_filter)
#         r, p, l = getRadPlanOrienttaion(gsnout, graph, True, True)
#         rad=np.nan_to_num(r/(r+p))
#         bpd=gsnout.vertex_property('distance_to_surface')
#         bpd=gsnout.vertex_property('distance_to_surface')
#         normed = False
#         hbpd, bins=np.histogram(bpd, bins=bin_nb,normed=normed)
#
#
#
#         depth=gsnout.edge_property('distance_to_surface')
#         normed = True
#         hist_ori, bins_ori_dist = np.histogram(depth[rad>thresh], bins=bins, normed=normed)
#
#     dfc_bp = pd.DataFrame(np.array(C_bp)).melt()
#     dfc_ori = pd.DataFrame(np.array(C_ori)).melt()
#
#
#     plt.figure()
#     import seaborn as sns
#     # sns.lineplot( np.array([(bins[i]+bins[i+1])/2 for i in range(bins.shape[0]-1)]),hbpd)
#     sns.lineplot(np.arange(bin_nb),hbpd)
#     sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc_bp)
#     plt.xticks(np.arange(bin_nb), 25*np.array([(bins[i]+bins[i+1])/2 for i in range(bins.shape[0]-1)]).astype(int))
#     plt.legend(['covid', 'control'])
#     plt.xlabel('depth')
#     plt.title('BP nose')
#
#     plt.figure()
#     # sns.lineplot( np.array([(bins_ori_dist[i]+bins_ori_dist[i+1])/2 for i in range(bins_ori_dist.shape[0]-1)]),hist_ori)
#     sns.lineplot(np.arange(bin_nb),hist_ori)
#     sns.lineplot(x="variable", y="value", err_style="bars", ci='sd', data=dfc_ori)
#     plt.xticks(np.arange(bin_nb),  25*np.array([(bins_ori_dist[i]+bins_ori_dist[i+1])/2 for i in range(bins_ori_dist.shape[0]-1)]).astype(int))
#     plt.legend(['covid', 'control'])
#     plt.xlabel('depth')
#     plt.title('ORI nose')
#     ##plt.figure()
#     # Cpd_rad = pd.DataFrame(normalize(rad_r, norm='l2', axis=1)).melt()
#     # Cpd_plan = pd.DataFrame(normalize(plan_r, norm='l2', axis=1)).melt()
#     # Cpd_all = pd.DataFrame(normalize(rad_all, norm='l2', axis=1)).melt()
#     # sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_plan, color=colors_m[1])
#     # sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_rad, color=colors_c[0])
#     # sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_all, color='g')
#     # plt.title('vessels radii distribution control Nose Regions')
#     # plt.legend(['planar', 'radial', 'all vessels'])
#     # plt.yscale('linear')
#
#
#     # gs = graph.sub_slice((slice(1,320), slice(255,265), slice(1,228)),coordinates='coordinates_atlas');
#
#     # gs = graph.sub_slice((slice(1,320), slice(528-265,528-255), slice(1,228)),coordinates='coordinates_atlas');
#     # vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
#     # connectivity = gs.edge_connectivity();
#     # edge_colors = (vertex_colors[connectivity[:,0]] + vertex_colors[connectivity[:,1]])/2.0;
#     #
#     # r, p, l=getRadPlanOrienttaion(gs, graph, True, True)
#     # #
#     # artery_color = np.array([[1, 0, 0, 1], [0,1,0, 1]]).astype('float')
#     # edge_colors=artery_color[np.asarray((r/(r+p))>0.6, dtype=int)]
#     # edge_colors[np.asarray((p/(r+p))>0.6, dtype=bool)]=[0,0,1,1]
#     #
#     # gs.add_edge_property('color', edge_colors)
#     # gs=gs.sub_graph(edge_filter=np.asarray((r/(r+p))>0.6, dtype=bool))
#     # edge_colors=gs.edge_property('color')
#     # vertex_colors = ano.convert_label(gs.vertex_annotation(), key='order', value='rgba');
#
#     # edge_colors[edge_artery_label>0] = [0.8,0.0,0.0,1.0]
#     # edge_colors[edge_vein_label  >0] = [0.0,0.0,0.8,1.0]
#
#     # p = p3d.plot_graph_mesh(gs, n_tube_points=3);
#
#     # p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);
#
