import numpy as np
import random
from matplotlib import cm
import matplotlib
from ClearMap.DiffusionPenetratingArteriesCortex import diffusion_through_penetrating_arteries,get_penetration_arteries_dustance_surface,get_penetrating_arteries_labels

import ClearMap.Analysis.Graphs.GraphGt_old as ggto
import ClearMap.Alignment.Annotation as ano
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ClearMap.Analysis.Graphs.GraphGt as ggt
from scipy import stats
import tifffile
import ClearMap.Visualization.Plot3d as p3d

def getColorMap_from_vertex_prop(vp, norm=None, cmx=None, cmn=None):
    colors = np.zeros((vp.shape[0], 4));
    print(colors.shape)
    # for i in range(b.size):
    i = 0

    # for index, b in enumerate(np.unique(diff)):
    if cmx==None:
        cmax = np.max(vp)
    else:
        cmax=cmx
    if cmn==None:
        cmin = np.min(vp)
    else:
        cmin=cmn
    # for i in range(gss4.n_vertices):
    #     colors[i, :]=[x[i],0, 1-x[i], 1]
    jet = cm.get_cmap('viridis')#jet_r
    import matplotlib.colors as col
    cNorm = col.Normalize(vmin=cmin, vmax=cmax)
    if norm==None:
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    else:
        scalarMap = cm.ScalarMappable(norm=norm, cmap=jet)
    print(scalarMap.get_clim())
    colorVal = scalarMap.to_rgba(vp)
    return colorVal

def from_e_prop2_vprop(graph, property):
    e_prop = graph.edge_property(property)
    v_prop=np.zeros(graph.n_vertices)
    connectivity = graph.edge_connectivity()
    v_prop[connectivity[e_prop==1,0]]=1
    v_prop[connectivity[e_prop == 1,1]] = 1
    # graph.add_vertex_property(property, v_prop)
    return v_prop

def getangle(c, b,a):
    # print{a}
    # print(b)
    # print(c)
    ba = a - b
    bc = c - b
    # print(ba)
    # print(bc)
    cosine_angle = np.dot(ba, bc.T) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    # print(angle)
    angle=abs(np.degrees(angle))
    if angle>180:
        angle=360-angle
    # print(angle)
    return angle


def stochasticPath(graph, Nb, steps, startPts, mode='pressure'):
    #graph,
    # print('stock path')
    # Nb, steps, startPts=args
    N=Nb
    J=0
    # print('J')
    pressure=graph.vertex_property('pressure')
    radii=graph.vertex_property('radii')
    # print('tradii path')
    coordinates=graph.vertex_property('coordinates')
    # print('coord path')
    V=np.zeros(graph.n_vertices)
    Sp = np.zeros(graph.n_vertices)
    isvein=from_e_prop2_vprop(graph, 'vein')
    isartery = from_e_prop2_vprop(graph, 'artery')
    pathOK=True
    for startPt in startPts:
        J=J+1
        print(startPts.shape[0]-J, '/',startPts.shape[0])
        visited_vertices = np.zeros(graph.n_vertices)
        visited_vertices[startPt] = 1
        sP_vect = np.zeros(graph.n_vertices)
        sP_vect[startPt] = 1
        N=Nb
        while N!=0:
            N = N - 1
            # print(N)
            prev_node=startPt
            node = startPt
            currdiffVal = 0
            pathOK=True
            visited_vertices_temp = np.zeros(graph.n_vertices)
            R=radii[node]
            # visited_vertices[startPt] = visited_vertices[startPt] + 1
            s=steps
            while s>0 and pathOK:
                # print(startPts.shape[0]-J,N,s)
                neighbours=graph.vertex_neighbours(node)
                temp=np.zeros(graph.n_vertices)
                temp[neighbours]=1
                neighbours=np.asarray(np.logical_and(temp, np.logical_not((visited_vertices_temp>0)==1))).nonzero()[0]
                if prev_node in neighbours:
                    i=np.asarray(neighbours==prev_node).nonzero()[0]
                    np.delete(neighbours, i)
                nbn=neighbours.shape[0]

                rads_t=[radii[n]**2 for n in neighbours]
                S=np.sum(rads_t)
                rads=rads_t/S
                if s != steps:
                    if mode=='pressure':
                        press = [pressure[n]-pressure[node] for n in neighbours]
                        press = np.nan_to_num(press).reshape(rads.shape)
                        S = np.sum(press)
                        press = press / S

                        W = press
                    else:
                        ang_t=[getangle(coordinates[n], coordinates[node], coordinates[prev_node]) for n in neighbours]
                        ang_t=np.nan_to_num(ang_t).reshape(rads.shape)
                        S=np.sum(ang_t)
                        angs = ang_t / S
    
                        W=angs*rads
                else:
                    # print('first step')
                    W=rads

                # print(W)
                # print('node ', node, neighbours)
                if nbn<1:
                    pathOK=False
                    break
                elif nbn>1:
                    # n=random.choice(range(nbn))
                    n = random.choices(range(nbn), weights=W)
                    visited_vertices_temp[neighbours[n]] =  1
                    visited_vertices[neighbours[n]] = visited_vertices[neighbours[n]] + 1
                    prev_node=node
                    node = neighbours[n]
                elif nbn==1:
                    n=0
                    visited_vertices_temp[neighbours[n]] =  1
                    visited_vertices[neighbours[n]] = visited_vertices[neighbours[n]] + 1
                    prev_node=node
                    node = neighbours[n]
                else:
                    visited_vertices_temp[neighbours[0]] = 1
                    visited_vertices[neighbours[0]] = visited_vertices[neighbours[0]] + 1
                    prev_node = node
                    node=neighbours[0]
                if isvein[node]==1:
                    s=0
                if not isartery[node]:
                    s = s - 1
        # currdiffVal=diff_val[prev_node]
        # print(currdiffVal)
        # visited_vertices[(visited_vertices_temp>0).astype(bool)]=1
        V=V+visited_vertices
        Sp=Sp+sP_vect
    return V#, Sp





def stochasticPathEmbedding(graph, Nb, steps, startPts):
    #graph,
    # print('stock path')
    # Nb, steps, startPts=args
    N=Nb
    J=0
    color = graph.vertex_property('plot_color')
    artery=graph.vertex_property('artery')
    # print('J')
    # radii=graph.vertex_property('radii')
    # print('tradii path')
    # coordinates=graph.vertex_property('coordinates')
    # print('coord path')
    V=np.zeros(graph.n_vertices)

    isvein=color==[0,0,1]
    isartery = graph.vertex_property('artery')
    pathOK=True
    L=[]
    for startPt in startPts:
        J=J+1
        print(startPts.shape[0]-J, '/',startPts.shape[0])
        visited_vertices = np.zeros(graph.n_vertices)
        visited_vertices[startPt] = 1
        sP_vect = np.zeros(graph.n_vertices)
        sP_vect[startPt] = 1
        N=Nb
        while N!=0:
            N = N - 1
            # print(N)
            prev_node=startPt
            node = startPt
            currdiffVal = 0
            pathOK=True
            visited_vertices_temp = np.zeros(graph.n_vertices)
            # R=radii[node]
            # visited_vertices[startPt] = visited_vertices[startPt] + 1
            s=steps
            while s>0 and pathOK:
                # print(startPts.shape[0]-J,N,s)
                neighbours=graph.vertex_neighbours(node)
                temp=np.zeros(graph.n_vertices)
                temp[neighbours]=1
                neighbours=np.asarray(np.logical_and(temp, np.logical_not((visited_vertices_temp>0)==1))).nonzero()[0]
                if prev_node in neighbours:
                    if artery[prev_node]!=0:
                        i=np.asarray(neighbours==prev_node).nonzero()[0]
                        np.delete(neighbours, i)
                nbn=neighbours.shape[0]


                # print(W)
                # print('node ', node, neighbours)
                if nbn<1:
                    pathOK=False
                    break
                elif nbn>1:
                    # n=random.choice(range(nbn))
                    n = random.choices(range(nbn))
                    visited_vertices_temp[neighbours[n]] =  1
                    visited_vertices[neighbours[n]] = visited_vertices[neighbours[n]] + 1
                    prev_node=node
                    node = neighbours[n]
                elif nbn==1:
                    n=0
                    visited_vertices_temp[neighbours[n]] =  1
                    visited_vertices[neighbours[n]] = visited_vertices[neighbours[n]] + 1
                    prev_node=node
                    node = neighbours[n]
                else:
                    visited_vertices_temp[neighbours[0]] = 1
                    visited_vertices[neighbours[0]] = visited_vertices[neighbours[0]] + 1
                    prev_node = node
                    node=neighbours[0]
                if (isvein[node]==[1,1,1]).all():
                    s=0
                if not isartery[node].all():
                    s = s - 1
            L.append(steps-s)
        # currdiffVal=diff_val[prev_node]
        # print(currdiffVal)
        # visited_vertices[(visited_vertices_temp>0).astype(bool)]=1
        V=V+visited_vertices
    return V,L#, Sp



try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open('/data_SSD_2to/181002_4/reg_list.p', 'rb') as fp:
  reg_list = pickle.load(fp)

with open('/data_SSD_2to/191122Otof/reg_list_full.p', 'rb') as fp:
  reg_list = pickle.load(fp)


if __name__ == "__main__":

    work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
    controls=['142L','158L','162L', '164L']
    mutants=['138L','141L','163L', '165L']#

    # controls=['2R','3R','5R', '8R']#cpmntrol
    # mutants=['1R','7R', '6R', '4R']#mutant

    # work_dir='/data_SSD_2to/191122Otof'
    condition='isocortex'

    if condition == 'isocortex':
        region_list = [(6, 6)]  # isocortex
        regions = []
        R = ano.find(region_list[0][0], key='order')['name']
        main_reg = region_list
        sub_region = True
        for r in reg_list.keys():
            l = ano.find(r, key='order')['level']
            regions.append([(r, l)])

    regions=np.array(regions)

    template_shape=(320,528,228)
    # controls = ['2R', '3R', '5R', '8R','1R','7R', '6R', '4R']
    controls = ['142L','158L','162L', '164L', '138L','141L','163L', '165L']
    # condition='all_cortex'
    regions=[[(6,6)]]
    for g in controls:
        # g='142L'
        bin=20
        N=5
        print(g)
        rad_r=[]
        plan_r=[]
        rad_all=[]
        # for g in controls:
        print(g)
        graph = ggto.load(work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')#data_graph_correctedIsocortex.gt')#_correctedIsocortex
        degrees = graph.vertex_degrees()
        vf = np.logical_and(degrees > 1, degrees <= 4)
        graph = graph.sub_graph(vertex_filter=vf)
        with open(work_dir + '/' + g + '/sampledict' + g + '.pkl', 'rb') as fp:
            sampledict = pickle.load(fp)

        pressure = np.asarray(sampledict['pressure'][0])
        graph.add_vertex_property('pressure', pressure)
        V=[]
        D=[]
        C=[]
        # graph = ggto.load(work_dir + '/' + g + '/' + 'data_graph.gt')#_correctedIsocortex
        for r,region_list in enumerate(regions):
            try:
                vertex_filter = np.zeros(graph.n_vertices)
                for i, rl in enumerate(region_list):
                    order, level = region_list[i]
                    print(level, order, ano.find(order, key='order')['name'])
                    label = graph.vertex_annotation();
                    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                    vertex_filter[label_leveled == order] = 1;
                gss4_t = graph.sub_graph(vertex_filter=vertex_filter)
                arteries=from_e_prop2_vprop(gss4_t, 'artery')
                d2s=gss4_t.vertex_property('distance_to_surface')
                # penetrating_arteries = np.logical_and(d2s >= 3, arteries)
                # startPts=np.asarray( np.logical_and(np.logical_and(d2s <= 3.3, d2s >= 3), arteries)).nonzero()[0]

                aepl=get_penetrating_arteries_labels( gss4_t, graph, get_penetration_arteries_dustance_surface)
                u, c=np.unique(aepl, return_counts=True)
                startPts=[]
                for a in u[1:]:
                    print(a)

                    pt=np.asarray(np.logical_and(aepl==a, d2s==np.min(d2s[aepl==a]))).nonzero()[0][0]
                    startPts.append(pt)

                startPts=np.array(startPts)
                print(startPts, startPts.shape)
                visited_vertices = stochasticPath(gss4_t, 200, 15 , startPts)

                V.append(visited_vertices)
                D.append(gss4_t.vertex_property('distance_to_surface'))
                C.append(gss4_t.vertex_property('coordinates_atlas'))
            except(IndexError):
                print('no arteries in ', ano.find(order, key='order')['name'])
                # regions = np.delete(regions, r)
                # V.append([])
                # D.append([])
                C.append([])


        np.save(work_dir + '/' + g + '/' + 'pressurestochasticPath3_'+condition+'.npy',V)
        np.save(work_dir + '/' + g + '/' + 'distance_to_surface3_'+condition+'.npy', D)
        np.save(work_dir + '/' + g + '/' + 'coordinates3_'+condition+'.npy', C)

    controls = ['2R', '3R', '5R', '8R']  # cpmntrol
    mutants = ['1R', '7R', '6R', '4R']  # mutant
    controls=['142L','158L','162L', '164L']
    mutants=['138L','141L', '163L', '165L']
    region_list= [[(54, 9), (47, 9)]]
    regions = [[(6, 6)]]
    # region_list=[[(142, 8), (149, 8), (128, 8), (156, 8)]]
    condition='isocortex'#'all_cortex'
    for j,g in enumerate(controls):
        print(g)
        graph = ggto.load(
            work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_correctedIsocortex.gt')#_correctedIsocortex
        degrees = graph.vertex_degrees()
        vf = np.logical_and(degrees > 1, degrees <= 4)
        graph = graph.sub_graph(vertex_filter=vf)
        vv = np.load(work_dir + '/' + g + '/' + 'pressurestochasticPath3_' + condition + '.npy')[0]
        
        for r, reg in enumerate(regions):
            vertex_filter = np.zeros(graph.n_vertices)
            for i, rl in enumerate(reg):
                order, level = reg[i]
                print(level, order, ano.find(order, key='order')['name'])
                label = graph.vertex_annotation();
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1
        gss4_t = graph.sub_graph(vertex_filter=vertex_filter)

        gss4_t.add_vertex_property('stochasticPath', vv)
        d2s = gss4_t.vertex_property('distance_to_surface')
        
        gss4_t.add_vertex_property('distance_to_surface3', d2s)
        #remove values on arteries from the pool beacause there are always high regardless the topology of the capillary bed
        artery = from_e_prop2_vprop(gss4_t, 'artery')
        vertex_filter = np.zeros(gss4_t.n_vertices)
        for i, rl in enumerate(region_list[0]):
            order, level = region_list[0][i]
            print(level, order, ano.find(order, key='order')['name'])
            label = gss4_t.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter[label_leveled == order] = 1;
        gss4_t = gss4_t.sub_graph(vertex_filter=vertex_filter)
        artery=from_e_prop2_vprop(gss4_t, 'artery')
        gss4_t = gss4_t.sub_graph(vertex_filter=np.logical_not(artery))
        vv=gss4_t.vertex_property('stochasticPath')
        d2s = gss4_t.vertex_property('distance_to_surface')
        # vf=np.logical_and(vertex_filter, np.logical_not(artery))
        # vv = vv[:, np.where(vf != 0)[0]]
        # d2s = d2s[:, np.where(vf != 0)[0]]
        if j==0:
            Pc=vv
            Dc=d2s
        else:
            Pc=np.concatenate((Pc,vv), axis=0)
            Dc = np.concatenate((Dc, d2s), axis=0)
    condition = 'isocortex'
    for j,g in enumerate(mutants):
        print(g)
        graph = ggto.load(
            work_dir + '/' + g + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_correctedIsocortex.gt')#_correctedIsocortex
        degrees = graph.vertex_degrees()
        vf = np.logical_and(degrees > 1, degrees <= 4)
        graph = graph.sub_graph(vertex_filter=vf)
        vv = np.load(work_dir + '/' + g + '/' + 'pressurestochasticPath3_' + condition + '.npy')[0]
        
        for r, reg in enumerate(regions):
            vertex_filter = np.zeros(graph.n_vertices)
            for i, rl in enumerate(reg):
                order, level = reg[i]
                print(level, order, ano.find(order, key='order')['name'])
                label = graph.vertex_annotation();
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1
        gss4_t = graph.sub_graph(vertex_filter=vertex_filter)
        gss4_t.add_vertex_property('stochasticPath', vv)
        d2s = gss4_t.vertex_property('distance_to_surface')        
        gss4_t.add_vertex_property('distance_to_surface3', d2s)
        
        # remove values on arteries from the pool beacause there are always high regardless the topology of the capillary bed
        artery = from_e_prop2_vprop(gss4_t, 'artery')
        vertex_filter = np.zeros(gss4_t.n_vertices)
        for i, rl in enumerate(region_list[0]):
            order, level = region_list[0][i]
            print(level, order, ano.find(order, key='order')['name'])
            label = gss4_t.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter[label_leveled == order] = 1;
        gss4_t = gss4_t.sub_graph(vertex_filter=vertex_filter)
        artery = from_e_prop2_vprop(gss4_t, 'artery')
        gss4_t = gss4_t.sub_graph(vertex_filter=np.logical_not(artery))
        vv = gss4_t.vertex_property('stochasticPath')
        d2s = gss4_t.vertex_property('distance_to_surface')
        # vf=np.logical_and(vertex_filter, np.logical_not(artery))
        # vv = vv[:, np.where(vf != 0)[0]]
        # d2s = d2s[:, np.where(vf != 0)[0]]
        if j==0:
            Pm=vv
            Dm=d2s
        else:
            Pm=np.concatenate((Pm,vv), axis=0)
            Dm = np.concatenate((Dm, d2s), axis=0)

    # plt.figure()
    data= {'depth':np.concatenate((Dc, Dm), axis=0), 'path':np.concatenate((Pc, Pm), axis=0), 'condition':np.concatenate((np.zeros(Pc.shape[0]),np.ones(Pm.shape[0])), axis=0)}
    df = pd.DataFrame(data)

    control = df.loc[df.condition == 0.0]
    mutant = df.loc[df.condition == 1.0]

    # sns.set_style(style='white')
    # sns.despine()
    g = sns.JointGrid('depth', 'path', data=df)
    # plt.figure()
    sns.set_style(style='white')
    sns.despine()
    ax=sns.kdeplot(mutant.depth, mutant.path, cmap="Reds",
            shade=False, shade_lowest=False, ax=g.ax_joint)
    sns.distplot(mutant.depth, color="r", ax=g.ax_marg_x,bins=50)
    ax=sns.distplot(mutant.path,kde=True,  color="r", ax=g.ax_marg_y, vertical=True,bins=200)
    ax.set_yscale('log')
    # sns.scatterplot(mutant.depth,mutant.path,size=0.1, alpha=0.001,color="r")
    # sns.distplot(Dm, kde=True, hist=False, color="r", ax=g.ax_marg_x)
    # sns.distplot(Pm, kde=True, hist=False, color="r", ax=g.ax_marg_y, vertical=True)

    ax=sns.kdeplot(control.depth, control.path, cmap="Blues",
            shade=False, shade_lowest=False, ax=g.ax_joint)
    sns.distplot(control.depth,color="b", ax=g.ax_marg_x,bins=50)
    ax=sns.distplot(control.path,  kde=True, color="b", ax=g.ax_marg_y, vertical=True,bins=200)
    g.ax_marg_y.set_yscale('linear')
    # sns.scatterplot(control.depth,control.path,size=0.1, alpha=0.0001,color="b")
    # sns.distplot(Dc, kde=True, hist=False, color="b", ax=g.ax_marg_x)
    # sns.distplot(Pc, kde=True, hist=False, color="b", ax=g.ax_marg_y, vertical=True)
    sns.set_style(style='white')
    sns.despine()


    ## voxelize
    import ClearMap.Analysis.Measurements.Voxelization as vox
    import ClearMap.IO.IO as io

    work_dir='/data_SSD_2to/whiskers_graphs/new_graphs'
    controls=['142L','158L','162L', '164L']
    mutants=['138L','141L', '163L', '165L']

    template_shape=(320,528,228)
    vox_shape_c = (320, 528, 228, len(controls))
    vox_shape_m = (320, 528, 228, len(mutants))
    vox_art_raw_signal_control = np.zeros(vox_shape_c)
    vox_art_raw_signal_mutant = np.zeros(vox_shape_m)


    radius=10


    for i, g in enumerate(controls):
        vv = np.load(work_dir + '/' + g + '/' + 'stochasticPath3_'+condition+'.npy')[0]
        coo = np.load(work_dir + '/' + g + '/' + 'coordinates3_'+condition+'.npy')[0]
        v = vox.voxelize(coo, shape=template_shape, weights=vv, radius=(radius, radius, radius), method='sphere');
        w = vox.voxelize(coo, shape=template_shape, weights=None, radius=(radius, radius, radius),method='sphere');
        vox_art_raw_signal_control[:, :, :, i] = v.array / w.array

    io.write(work_dir + '/' +'vox_stochasticPath3_control'+str(radius)+'.tif', vox_art_raw_signal_control.astype('float32'))
    vox_art_raw_signal_control_avg=np.mean(vox_art_raw_signal_control, axis=3)
    io.write(work_dir + '/' +'vox_stochasticPath3_control_avg'+str(radius)+'.tif', vox_art_raw_signal_control_avg.astype('float32'))



    for i, g in enumerate(mutants):
        vv = np.load(work_dir + '/' + g + '/' + 'stochasticPath3_'+condition+'.npy')[0]
        coo = np.load(work_dir + '/' + g + '/' + 'coordinates3_'+condition+'.npy')[0]
        v = vox.voxelize(coo, shape=template_shape, weights=vv, radius=(radius, radius, radius), method='sphere');
        w = vox.voxelize(coo, shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
        vox_art_raw_signal_mutant[:, :, :, i] = v.array / w.array

    io.write(work_dir + '/' +'vox_stochasticPath3_mutant'+str(radius)+'.tif', vox_art_raw_signal_mutant.astype('float32'))
    vox_art_raw_signal_mutant_avg=np.mean(vox_art_raw_signal_mutant, axis=3)
    io.write(work_dir + '/' +'vox_stochasticPath3_mutant_avg'+str(radius)+'.tif', vox_art_raw_signal_mutant_avg.astype('float32'))




    pcutoff = 0.05

    tvals, pvals = stats.ttest_ind(vox_art_raw_signal_control, vox_art_raw_signal_mutant, axis = 3, equal_var = True);

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


    tifffile.imsave(work_dir+'/pvalcolors_stochasticPath3_'+str(radius)+'.tif', np.swapaxes(pvalscol, 2, 0).astype('uint8'), photometric='rgb',imagej=True)


    plt.figure(3)
    plt.title('Random walk: vertices Depth VS path occurence in Controls')
    plt.xlabel('depth')
    plt.ylabel('number of paths')


    graph.add_vertex_property('visited_vertices', visited_vertices)
    graph.add_vertex_property('sP_vect', Sp)
    gs = graph.sub_slice((slice(1,320), slice(228,243), slice(1,228)),coordinates='coordinates_atlas');
    arteries=from_e_prop2_vprop(gs, 'artery')
    gs=gs.sub_graph(vertex_filter=arteries==0)
    # stocha_path = gss4.sub_graph(vertex_filter=visited_vertices > 0)
    norm=matplotlib.colors.LogNorm()#cNorm#
    colorval = getColorMap_from_vertex_prop(gs.vertex_property('visited_vertices'), norm=None,cmx=10, cmn=1)
    v_arteries = from_e_prop2_vprop(gs, 'artery')
    v_veins = from_e_prop2_vprop(gs, 'vein')
    # colorval[v_arteries == 1] = [1., 0.0, 0.0, 1.0]
    # colorval[v_veins == 1] = [0.0, 1.0, 0.0, 1.0]
    colorval[gs.vertex_property('sP_vect') == 1] = [0.0, 0.0, 0.0, 1.0]
    p = p3d.plot_graph_mesh(gs, vertex_colors=colorval, n_tube_points=3);

    ##
    controls=['142L','158L','162L', '164L']
    mutants=['138L','141L', '163L', '165L']
    # controls=['2R','3R','5R', '8R']#cpmntrol
    # mutants=['1R','7R', '6R', '4R']#mutant
    g='1R'
    condition='isocortex'
    region_list= [[(54, 9), (47, 9)]]
    region_list=[[(142, 8), (149, 8), (128, 8), (156, 8)]]

    graph = ggt.load(work_dir + '/' + g + '/' + 'data_graph_correctedIsocortex.gt')  # _correctedIsocortex
    vv = np.load(work_dir + '/' + g + '/' + 'stochasticPath3_' + condition + '.npy')
    # graph.add_vertex_property('sP', vv)
    vertex_filter = np.zeros(graph.n_vertices)
    for i, rl in enumerate(region_list[0]):
        order, level = region_list[0][i]
        print(level, order, ano.find(order, key='order')['name'])
        label = graph.vertex_annotation();
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        vertex_filter[label_leveled == order] = 1;
    gs=graph.sub_graph(vertex_filter=vertex_filter)
    # vv=gs.vertex_property('sP')

    vv=vv[:, vertex_filter.astype('bool')]
    v_arteries = from_e_prop2_vprop(gs, 'artery')
    vf=np.logical_or(vv>=6, v_arteries)
    gs = gs.sub_graph(vertex_filter=vf)

    # vv = vv[:, np.where(vertex_filter != 0)[0]][0]
    # norm=matplotlib.colors.LogNorm()#cNorm#

    colorval=np.zeros((gs.n_vertices, 4))
    colorval[:, 1]=np.ones(gs.n_vertices)
    colorval[:, 3]=np.ones(gs.n_vertices)

    colorval = getColorMap_from_vertex_prop(vv, norm=None,cmx=10, cmn=1)
    v_arteries = from_e_prop2_vprop(gs, 'artery')
    v_veins = from_e_prop2_vprop(gs, 'vein')
    colorval[v_arteries == 1] = [1., 0.0, 0.0, 1.0]
    colorval[v_veins == 1] = [0.0, 1.0, 0.0, 1.0]
    # colorval[gs.vertex_property('sP_vect') == 1] = [0.0, 0.0, 0.0, 1.0]
    p = p3d.plot_graph_mesh(gs, vertex_colors=colorval, n_tube_points=3);