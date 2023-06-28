import numpy as np

import ClearMap.Settings as settings
import ClearMap.IO.IO as io

import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skl

import ClearMap.Analysis.Graphs.GraphProcessing as gp

import ClearMap.Alignment.Annotation as ano

import ClearMap.Analysis.Measurements.MeasureRadius as mr
import ClearMap.Analysis.Measurements.MeasureExpression as me
import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti

print('TEST')
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import graph_tool.centrality as gtc
print('loading...')
import numpy as np
from numpy import arctan2, sqrt
import numexpr as ne
from sklearn import preprocessing
import math
import multiprocessing as mp #Semaphore
from scipy import stats
mutex = None
import graph_tool.topology as gtt
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score
import seaborn as sns; sns.set()
import scipy.cluster.hierarchy as hierarchy
import seaborn as sns
import graph_tool.draw as gtd
import ClearMap.Gt2Nx as Gt2Nx
import networkx as nx
import graph_tool as gt
import networkx.algorithms.similarity as nxas
# from ClearMap.Visualization.Vispy.sbm_plot import initialize_brain_graph
from scipy.cluster.hierarchy import ward, dendrogram, linkage
import random

def getWienerIndex(graph):
    dist = ggt.vertex_property_map_to_python(gtt.shortest_distance(graph.base, source=None), as_array=True)
    print('shortest path shape :', dist.shape)
    nb_val= dist.shape[0]*dist.shape[1]
    print(nb_val)
    res=np.sum(dist)/nb_val#for norma;ization
    print(res)
    return res

def getdeltaIndex(graph):
    dist = ggt.vertex_property_map_to_python(gtt.shortest_distance(graph.base, source=None), as_array=True)
    return dist

def getWienerIndices(region_list, brain_list, sbm_level):
    WienerIndices = []
    for brainnb in brain_list:
        print(brainnb)

        g, gts, base = initialize_brain_graph(brainnb)
        modules = []
        n = sbm_level
        # for i in range(n):
        #     print(i)
        blocks = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_' + str(n-1) + '.npy')
        # if i == 0:
        modules = blocks
        # else:
        #     modules = np.array([blocks[b] for b in modules])

        gts.add_vertex_property('blocks', modules)

        for regions in region_list:
            order, level = regions
            print(level, order, ano.find_name(order, key='order'))

            label = gts.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            gss4 = gts.sub_graph(vertex_filter=vertex_filter)

            block = gss4.vertex_property('blocks')
            for b in np.unique(block):
                if (np.sum(block==b)>100):
                    sub_gss4=gss4.sub_graph(vertex_filter=block==b)
                    WienerIndices.append((getWienerIndex(sub_gss4), order))
    WienerIndices=np.array(WienerIndices)
    if len(brain_list)>1:
        np.save('/mnt/data_SSD_2to/WienerIndices.npy', WienerIndices)
    else:
        np.save('/mnt/data_SSD_2to/' + brainnb + '/sbm/WienerIndices.npy', WienerIndices)
    return WienerIndices

def getdeltaIndices(region_list, brain_list, sbm_level):
    deltaIndices=[]
    for brainnb in brain_list:
        print(brainnb)

        # g, gts, base = initialize_brain_graph(brainnb)
        modules = []
        n = sbm_level
        # for i in range(n):
        #     print(i)
        blocks = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_' + str(n-1) + '.npy')
        # if i == 0:
        modules = blocks
        # else:
        #     modules = np.array([blocks[b] for b in modules])

        gts.add_vertex_property('blocks', modules)

        for regions in region_list:
            order, level = regions
            print(level, order, ano.find_name(order, key='order'))

            label = gts.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            gss4 = gts.sub_graph(vertex_filter=vertex_filter)

            block = gss4.vertex_property('blocks')
            for b in np.unique(block):
                if (np.sum(block==b)>100):
                    sub_gss4=gss4.sub_graph(vertex_filter=block==b)
                    deltaIndices.append((getdeltaIndex(sub_gss4), order))
    deltaIndices = np.array(deltaIndices)
    if len(brain_list) > 1:
        import pickle
        with open('/mnt/data_SSD_2to/deltaIndices.p', 'wb') as fp:
            pickle.dump(deltaIndices, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        import pickle
        with open('/mnt/data_SSD_2to/' + brainnb + '/sbm/deltaIndices.p', 'wb') as fp:
            pickle.dump(deltaIndices, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return deltaIndices



def computedeltaProduct(args):
    mat_i, mat_j, i, j=args
    print(i, j)

    ii=mat_i*mat_i.T
    jj=mat_j*mat_j.T
    ij=mat_i*mat_j.T
    # i, counts_i=np.unique(mat_i, return_counts=True)
    # j, counts_j=np.unique(mat_j, return_counts=True)
    # i = np.array(i)
    # j = np.array(j)
    # result=0
    # for k in range(i.shape[0]):
    #     result=result+counts_i[k]*np.sum(np.where(mat_j)==i[k])
    # for k in range(j.shape[0]):
    #     result = result + counts_j[k] * np.sum(np.where(mat_i) == j[k])
    result=np.sum(np.where(ii==ij)+np.where(jj==ii))
    return result


def getDeltaKernelproductMatrix(region_list, brain_list,sbm_level, N):
    import random
    train_data = []


    # deltas=getdeltaIndices(region_list, brain_list, sbm_level)
    import pickle
    with open('/mnt/data_SSD_2to/' + brainnb + '/sbm/deltaIndices.p', 'rb') as fp:
        deltas = pickle.load(fp)
    # KernelProductMatrix = np.zeros((len(deltas), len(deltas)))
    # for i in range(len(deltas)):
    #     for j in range(len(deltas)):
    #         order=deltas[i,1]
    #         labels.append(order)
    #         KernelProductMatrix[i, j]=computedeltaProduct(deltas[i,0], deltas[i,0])
    labels = deltas[:, 1]
    p=mp.Pool(20)
    import itertools
    ans=[p.map(computedeltaProduct, [(deltas[i,0], deltas[i,0], i, j) for i, j in itertools.product(range(len(deltas)), range(len(deltas)))])]
    ans=np.array(ans[0])
    ans=ans.astype(int)
    print(ans.shape)
    np.save('/mnt/data_2to/DelataKernelProductMatrix.npy', ans)
    np.save('/mnt/data_2to/Delatalabels.npy', labels)
    return ans, labels


def getKernelproductMatrix(region_list, brain_list,sbm_level, N, linear):
    import random
    train_data=[]
    # for brainnb in brain_list:
    if linear:
        WienerIndices=np.load('/mnt/data_SSD_2to/WienerIndices.npy')
        kpm = np.load('/mnt/data_SSD_2to/' + '190408_38L' + '/sbm/WienerIndices.npy')
        WienerIndices=np.concatenate((WienerIndices, kpm), axis=0)
        kpm = np.load('/mnt/data_SSD_2to/' + '190408_39L' + '/sbm/WienerIndices.npy')
        WienerIndices = np.concatenate((WienerIndices, kpm), axis=0)
        # WienerIndices=getWienerIndices(region_list, brain_list, sbm_level)
    from sklearn.preprocessing import StandardScaler
    print(WienerIndices.shape)
    WienerIndices_std = StandardScaler().fit_transform(WienerIndices[:, 0].reshape(-1, 1))
    WienerIndices[:, 0]=WienerIndices_std[:,0]
    l=len(region_list)
    for r in region_list:
        order, level = r
        list=np.where(WienerIndices[:, 1] == order)[0]
        random.shuffle(list)
        n_max=min(N, list.shape[0])
        for ind in list[:n_max]:
            train_data.append((WienerIndices[ind]))
            print(len(train_data))
    train_data=np.array(train_data)
    np.save('/mnt/data_SSD_2to/' + 'traindata.npy', train_data)
    print(train_data.shape)
    KernelProductMatrix=np.zeros((train_data.shape[0],train_data.shape[0]))
    labels=[]
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[0]):
            KernelProductMatrix[i, j]=train_data[i,0]*train_data[j,0]
        labels.append(train_data[i, 1])
    if len(brain_list) > 1:
        print('>1')
        # np.save('/mnt/data_SSD_2to/KernelProductMatrix.npy', KernelProductMatrix)
        # np.save('/mnt/data_SSD_2to/Klabels.npy', labels)
    else:
        brainnb=brain_list[0]
        # np.save('/mnt/data_SSD_2to/' + brainnb + '/sbm/KernelProductMatrix.npy',KernelProductMatrix)
        # np.save('/mnt/data_SSD_2to/' + brainnb + '/sbm/Klabels.npy', labels)
    return KernelProductMatrix, labels


def SVMvalidationTest(model, KernelProductMatrix, labels):
    # KernelProductMatrix= np.load('/mnt/data_SSD_2to/KernelProductMatrix.npy')#
    # labels = np.load('/mnt/data_SSD_2to/Klabels.npy')
    print('labels shape', labels.shape)
    # print(labels)
    print(KernelProductMatrix)
    y_valid_pred = model.predict(KernelProductMatrix)

    print('score', (y_valid_pred == labels).sum() / len(labels))

def SVMprobFunc(model, KernelProductMatrix, labels):
    # KernelProductMatrix= np.load('/mnt/data_SSD_2to/KernelProductMatrix.npy')#
    # labels = np.load('/mnt/data_SSD_2to/Klabels.npy')
    print('labels shape', labels.shape)
    # print(labels)
    print(KernelProductMatrix)
    proba = model.predict_proba(KernelProductMatrix)

    return proba


def SVMModel(region_list, brain_list,sbm_level, N, linear):
    # if linear:
    #     KernelProductMatrix, labels=getKernelproductMatrix(region_list, brain_list,sbm_level, N, linear)
    # else:
    #     print('delta')
    #     KernelProductMatrix, labels=getDeltaKernelproductMatrix(region_list, brain_list,sbm_level, N)
    # import pickle
    # with open('/mnt/data_SSD_2to/KernelProductMatrix.p', 'wb') as fp:
    #     pickle.dump(KernelProductMatrix, fp, protocol=pickle.HIGHEST_PROTOCOL)
    KernelProductMatrix= np.load('/mnt/data_2to/DelataKernelProductMatrix.npy')#
    labels= np.load('/mnt/data_2to/Delatalabels.npy').astype(int)
    print('labels shape', labels.shape)
    # print(labels)
    ####################scaling
    from sklearn.preprocessing import StandardScaler
    print(KernelProductMatrix.shape)
    KernelProductMatrix = StandardScaler().fit_transform(KernelProductMatrix.reshape(-1, 1))#.reshape(-1, 1)
    KernelProductMatrix=KernelProductMatrix.reshape((labels.shape[0], labels.shape[0]))
    #####################
    print(KernelProductMatrix.shape)
    from sklearn import svm
    lin_clf = svm.SVC(kernel='precomputed', tol=1e-7, probability=True)
    lin_clf.fit(KernelProductMatrix, labels)
    SVMvalidationTest(lin_clf, KernelProductMatrix, labels)
    proba=SVMprobFunc(lin_clf, KernelProductMatrix, labels)
    return lin_clf, proba, labels, KernelProductMatrix



def getTestData(brain_list, region_list, N):
    import random
    testData = []
    WienerIndices = getWienerIndices(region_list, brain_list, 1)
    from sklearn.preprocessing import StandardScaler
    print(WienerIndices[:, 0].shape)
    WienerIndices_std = StandardScaler().fit_transform(WienerIndices[:, 0].reshape(-1, 1))
    WienerIndices[:, 0] = WienerIndices_std[:, 0]
    l = len(region_list)
    for r in region_list:
        order, level = r
        list = np.where(WienerIndices[:, 1] == order)[0]
        random.shuffle(list)
        n_max = min(N, list.shape[0])
        for ind in list[:n_max]:
            testData.append((WienerIndices[ind]))
            print(len(testData))
    train_data = np.array(testData)
    np.save('/mnt/data_SSD_2to/' + 'testData.npy', testData)
    return testData

def SVMTest(model, brain_list, region_list, N):
    train_data = np.load('/mnt/data_SSD_2to/' + 'traindata.npy')
    testdata=getTestData(brain_list, region_list, N)
    print(train_data.shape)
    KernelProductMatrix = np.zeros((train_data.shape[0], train_data.shape[0]))
    labels = []
    for i in range(testdata.shape[0]):
        for j in range(train_data.shape[0]):
            KernelProductMatrix[i, j] = testdata[i, 0] * train_data[j, 0]
            labels.append(testdata[i, 1])
    y_valid_pred = model.predict(KernelProductMatrix)
    print('score', (y_valid_pred == labels).sum() / len(labels))


def getShortestPathgraphs(graph, N, subgraphs, order, orders):
    # if subgraphs_array==[]:
    #     subgraphs=[]
    # else subgraphs_array=subgraphs
    index_vertex_to_check = np.where(graph.vertex_property('vf') == 1)[0]  # to remove border effects
    print(graph.n_vertices, np.max(index_vertex_to_check))
    dist = ggt.vertex_property_map_to_python(gtt.shortest_distance(graph.base, source=None), as_array=True)
    print(dist.shape)

    for i in index_vertex_to_check:#range(dist.shape[0]):
        filter=np.zeros(graph.n_vertices)
        # filter[np.argsort(dist[i])[:N]]=1
        # print(i, np.where(dist[i]<=4),np.where(dist[i]<=4))
        filter[np.where(dist[i]<=4)]=1
        print(i,np.sum(filter) )
        sg=graph.sub_graph(vertex_filter=filter)
        # sg = sg.sub_graph(vertex_filter=sg.vertex_degrees() != 2)
        sg = sg.largest_component()
        if sg.n_vertices>=9:
            subgraphs.append(sg)
            orders.append(order)
    return subgraphs, orders

def getshortestpathssubgraphs(brain_list, region_list, sbm_level, N):
    subgraphs=[]
    orders=[]
    for brainnb in brain_list:
        print(brainnb)

        # g, gts, base = initialize_brain_graph(brainnb)
        modules = []
        n = sbm_level
        for i in range(n):
            print(i)
            blocks = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_' + str(i) + '.npy')
            if i == 0:
                modules = blocks
            else:
                modules = np.array([blocks[b] for b in modules])

        gts_filtered.add_vertex_property('blocks', modules)
        print('gts vertex degree 2 ', np.sum(gts.vertex_degrees() == 2))
        for regions in region_list:
            order, level = regions
            print(level, order, ano.find_name(order, key='order'))

            label = gts_filtered.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            gss4 = gts_filtered.sub_graph(vertex_filter=vertex_filter)
            print('gss4 vertex degree 2 ',np.sum(gss4.vertex_degrees() == 2))
            block = gss4.vertex_property('blocks')
            for b in np.unique(block)[:1]:
                if (np.sum(block==b)>100):
                    vf=block==b
                    vf_expanded=gss4.expand_vertex_filter(vf, steps=4)
                    gss4.add_vertex_property('vf', vf)
                    sub_gss4=gss4.sub_graph(vertex_filter=vf_expanded)

                    # sub_gss4 = sub_gss4.sub_graph(vertex_filter=sub_gss4.vertex_degrees()>=3)
                    sub_gss4 = sub_gss4.largest_component()
                    print('sub_gss4 vertex degree 2 ', np.sum(sub_gss4.vertex_degrees()==2))
                    # vf=sub_gss4.vertex_property('vf')
                    # sub_gss4.set_vertex_property('vf', vf)
                    subgraphs, orders =getShortestPathgraphs(sub_gss4, N, subgraphs, order, orders)
                    # subgraphs i in temp:
                    #     subgraphs.append(i)
                    #     orders.append(order)

    return subgraphs, orders


def diffusion(args):
    startindex, ind, step = args
    print(ind)
    # coordinateG.append(coordinates[startindex])
    visited_vertices = np.zeros(gss4.n_vertices)
    visited_vertices[startindex] = 1
    n = 0
    while n < step:
        # print(n)
        neigh = np.zeros(gss4.n_vertices)
        for i in range(gss4.n_vertices):
            if visited_vertices[i] > 0:
                # print(gss4.vertex_neighbours(i))
                neigh[gss4.vertex_neighbours(i)] = visited_vertices[i]
                # if visited_vertices[gss4.vertex_neighbours(i)]==0:
        update = np.logical_and(neigh > 0, visited_vertices == 0)
        visited_vertices[update] = neigh[update]
        n = n + 1

    return visited_vertices



def extractRandomSubgraph(brain_list, region_list, N_step, Nb ):
    subgraphs=[]
    coordinateG=[]
    orders=[]
    Nb=100000


    for brainnb in brain_list:
        print(brainnb)
        # g, gts, base = initialize_brain_graph(brainnb)
        # modules = []
        # n = 7
        # for i in range(n):
        #     print(i)
        #     blocks = np.load('/mnt/data_SSD_2to/' + brainnb + '/sbm/blockstate_full_brain_levelled_' + str(i) + '.npy')
        #     if i == 0:
        #         modules = blocks
        #     else:
        #         modules = np.array([blocks[b] for b in modules])
        #
        # gts.add_vertex_property('blocks', modules)

        for reg in region_list:
            order, level = reg
            print(level, order, ano.find_name(order, key='order'))

            label = gts.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            gss4 = gts.sub_graph(vertex_filter=vertex_filter)



            vert = np.arange(gss4.n_vertices)
            random.shuffle(vert)
            vert = vert[:int(gss4.n_vertices/20)]
            coordinateG.append(coordinates[vert])
            import timeit
            p = mp.Pool(20)
            filters = [p.map(diffusion,[(i, ind, N_step) for ind,i in enumerate(range(100))])]#vert
            print('done')
            # for ind,i in enumerate(vert):  # range(dist.shape[0]):
            for filter in filters[0][:50000]:
                # filter=diffusion(i, gss4, N_step)
                # print(ind, np.sum(filter))
                sg = gss4.sub_graph(vertex_filter=filter)
                # sg = sg.sub_graph(vertex_filter=sg.vertex_degrees() != 2)
                sg = sg.largest_component()
                if sg.n_vertices >= 9:
                    subgraphs.append(sg)
                    # orders.append(order)
    return subgraphs

def check_isomorphisms(subgraphs_list, size):
    subs=subgraphs_list
    subs=np.array(subs)
    canon=[]
    n=0
    for i in range(len(subgraphs_list)):
        print(i)
        if canon==[]:
            if subgraphs_list[i].n_vertices>=size:
                sub = subgraphs_list[i]
                sub=sub.largest_component()
                canon.append(sub)
                pos = gtd.sfdp_layout(sub.base)
                gtd.graph_draw(sub.base, pos=pos,
                               output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/graph_draw_sfdp_' + str(n) + '.pdf')
                n=n+1
        else:
            if i<len(subs):
                if subs[i].n_vertices >= size:
                    print(i, '/', len(subs))
                    sub=subs[i].largest_component()
                    canon.append(sub)
                    pos = gtd.sfdp_layout(sub.base)
                    gtd.graph_draw(sub.base, pos=pos,
                                   output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/graph_draw_sfdp_' + str(n) + '.pdf')
                    n = n + 1
            else:
                break
        temp=subs
        index_to_delete=[]
        if subgraphs_list[i].n_vertices >= size:#9, 13
            for j in range(subs.shape[0]):
                # print(i, j)
                is_iso1=gtt.subgraph_isomorphism(sub.base, subs[j].base, max_n=1, induced=True)
                # is_iso2 = gtt.subgraph_isomorphism(subs[j].base, sub.base, max_n=1, induced=True)
                print(i, j, '/', len(subs), is_iso1, len(is_iso1))
                # if (len(is_iso1)+len(is_iso2)>0):
                if len(is_iso1)>0:
                    index_to_delete.append(j)
                    # print(temp.shape)
                    # print(temp.shape)

            temp=np.delete(temp, index_to_delete, 0)
        subs=temp
    return canon

def check_isomorphisms_full(subgraphs_list, size):
    subs=subgraphs_list
    subs=np.array(subs)
    canon=[]
    n=0
    for i in range(len(subgraphs_list)):
        print(i)
        if canon==[]:
            if subgraphs_list[i].n_vertices>=size:
                sub = subgraphs_list[i]
                sub=sub.largest_component()
                canon.append(sub)
                pos = gtd.sfdp_layout(sub.base)
                gtd.graph_draw(sub.base, pos=pos,
                               output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/graph_draw_sfdp_' + str(n) + '.pdf')
                n=n+1
        else:
            if i<len(subs):
                if subs[i].n_vertices >= size:
                    print(i, '/', len(subs))
                    sub=subs[i].largest_component()
                    canon.append(sub)
                    pos = gtd.sfdp_layout(sub.base)
                    gtd.graph_draw(sub.base, pos=pos,
                                   output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/graph_draw_sfdp_' + str(n) + '.pdf')
                    n = n + 1
            else:
                break
        temp=subs
        index_to_delete=[]
        if subgraphs_list[i].n_vertices >= size:#9, 13
            for j in range(subs.shape[0]):
                # print(i, j)
                is_iso1=gtt.subgraph_isomorphism(sub.base, subs[j].base, max_n=1, induced=True)
                is_iso2 = gtt.subgraph_isomorphism(subs[j].base, sub.base, max_n=1, induced=True)
                print(i, j, '/', len(subs), is_iso1, is_iso2, len(is_iso1)+len(is_iso2))
                if (len(is_iso1)+len(is_iso2)>0):
                # if len(is_iso1)>0:
                    index_to_delete.append(j)
                    # print(temp.shape)
                    # print(temp.shape)

            temp=np.delete(temp, index_to_delete, 0)
        subs=temp
    return canon


def subgraph_isomorphism(args):
    sub, g, i = args
    print(i)
    res=gtt.subgraph_isomorphism(sub, g)
    return res


def getSubgraphDecomposition(args):
    print('getSubgraphDecomposition')
    e, graph=args
    simple_basis=np.array(simpla_basis)
    # g, graph, base = initialize_brain_graph(b)
    # e, graph = args
    mode = 'non para'
    label = graph.vertex_annotation();
    label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
    vertex_filter = label_leveled == e  # 54;
    print(ano.find_name(e, key='order'))
    region_name = ano.find_name(e, key='order')
    vertex_filter = graph.expand_vertex_filter(vertex_filter, steps=2)
    grt = graph.sub_graph(vertex_filter=vertex_filter)
    grt=grt.base
    iso=[]
    n=0
    if mode=='para':
        print('mode para')
        p = mp.Pool(5)
        res = [p.map(subgraph_isomorphism, [(simple_basis[i].base, grt.copy(), i) for i in range(simple_basis.shape[0])])]
        iso=[len(res[0][i]) for i in range(len(res[0]))]
    else:
        print('mode non para')
        for i in range(simple_basis.shape[0]):
            c=simple_basis[i]
            print(n)
            n=n+1
            res = gtt.subgraph_isomorphism(c.base, grt.base, induced=True)
            iso.append(len(res))
    return iso, region_name


def normalisation(projections):
    proj_array=[]
    for i in range(projections.shape[0]):
        norm2 = normalize(np.array(projections[i])[:, np.newaxis], axis=0, norm='l2').ravel()
        # norm2=np.array(projections[i])
        proj_array.append(norm2)
    proj_array=np.array(proj_array)
    return(proj_array)


def llf(id):
  # labels = np.load('/mnt/data_SSD_2to/190408-44L/labelsAcronymMat.npy')
  # labels = np.load('/mnt/data_SSD_2to/190408_38L/labelsAcronymMat.npy')
  return labels[id]


def filter_graph(graph):
    # get auto loops
    connectivity = graph.edge_connectivity()
    temp = sort_tuples([(connectivity[i, 0], connectivity[i, 1]) for i in range(connectivity.shape[0])])
    test = np.empty((len(connectivity, )), dtype=object)
    test[:] = [tempi for tempi in temp]
    # temp=np.array(temp, dtype=object)
    u, counts = np.unique(test, return_counts=True)

    duplica = u[np.where(counts > 1)]
    print(duplica.shape)

    doubles = [temp.index(duplica[i]) for i in range(duplica.shape[0])]
    loops = np.where(connectivity[:, 0] == connectivity[:, 1])

    # doubles=np.where(connectivity[:, 0]==connectivity[:, 1])
    lengths_doubles = gts.edge_geometry_lengths()[doubles]
    radius_doubles = gts.edge_geometry_property('radii')[doubles]

    single_edge_filter = np.ones(gts.n_edges)
    single_edge_filter[doubles] = 0
    single_edge_filter[loops] = 0
    gts_filtered = gts.sub_graph(edge_filter=single_edge_filter)
    return gts_filtered


def getSimilarityCanonGraph(canon):
    # get similarity heatmap between graphlets of the canon basis

    hm = np.zeros((len(canon), len(canon)))
    for i, c1 in enumerate(canon):
        for j, c2 in enumerate(canon):
            res = gtt.similarity(c1.base, c2.base)
            hm[i, j] = res
            print(i, j)

    import seaborn as sns;
    sns.set()
    import scipy.cluster.hierarchy as hierarchy
    z = hierarchy.linkage(hm, 'ward')
    ax = sns.clustermap(hm, row_linkage=z, col_linkage=z)
    return hm



def sort_tuples(alistoftuples):
    return [tuple(sorted(k)) for k in alistoftuples]



def CreateSimpleBasis():
    simpla_basis=[]

    #intersections
    inersect=[5,6,7]
    for i in inersect:
        print('intersect', i)
        g=ggt.Graph(n_vertices=i, directed=False)
        edges_all=np.zeros((0,2), dtype=int)
        for j in range(1, i):
            edge=(0, j)
            edges_all=np.vstack((edges_all, edge))

        # print(edges_all)
        g.add_edge(edges_all)
        simpla_basis.append(g)

        # cycles
    cycles = [3,4,5,6,7,8,9]
    for i in cycles:
        print('cycle', i)
        g = ggt.Graph(n_vertices=i, directed=False)
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

def createKirstMotif():
    i=4
    mot=[]
    g = ggt.Graph(n_vertices=4, directed=False)
    edges_all = np.zeros((0, 2), dtype=int)
    for j in range(1, 4):
        edge = (0, j)
        edges_all = np.vstack((edges_all, edge))

    # print(edges_all)
    g.add_edge(edges_all)
    mot.append(g)

    i=3
    g = ggt.Graph(n_vertices=4, directed=False)
    edges_all = np.zeros((0, 2), dtype=int)
    for j in range(3):
        if j + 1 < i:
            edge = (j, j + 1)
        else:
            edge = (j, 0)
        edges_all = np.vstack((edges_all, edge))
    edge = (3, 0)
    edges_all = np.vstack((edges_all, edge))
    # print(edges_all)
    g.add_edge(edges_all)
    mot.append(g)


    i=4
    g = ggt.Graph(n_vertices=4, directed=False)
    edges_all = np.zeros((0, 2), dtype=int)
    for j in range(4):
        if j + 1 < i:
            edge = (j, j + 1)
        else:
            edge = (j, 0)
        edges_all = np.vstack((edges_all, edge))

    # print(edges_all)
    g.add_edge(edges_all)
    mot.append(g)

    return mot

def plotStatAnalysis(projections, clusters):
    cond_means = []
    for u in np.unique(clusters):
        if u !=-1:
            temp = projections[np.where(clusters == u), :]
            print(temp.shape)
            temsp_means = np.mean(temp, axis=1)
            print(temsp_means.shape)
            cond_means.append(temsp_means)

    cond_means = np.array(cond_means)
    print(cond_means.shape)
    pred_proj = [cond_means[clusters[n] - 1] for n in range(projections.shape[0])]
    pred_proj = np.array(pred_proj)
    mains_clusters = [n for n in range(clusters.shape[0])]# if clusters[n] not in [4, 5, 6]]
    pred_proj = pred_proj[mains_clusters]
    p = np.squeeze(pred_proj, axis=1)
    coefficient_of_dermination = []
    for i in range(p.shape[1]):
        coefficient_of_dermination.append(r2_score(projections[mains_clusters, i], p[:, i]))

    coefficient_of_dermination = np.array(coefficient_of_dermination)
    print(coefficient_of_dermination)
    impacting_graphlets = np.array(np.where(abs(coefficient_of_dermination) > 0.0))[0]
    label = impacting_graphlets
    # labels=np.arange(69)
    # labels=np.array([1,3,4,38,39,47,65,49])
    statis = np.squeeze(cond_means[:, :, label], axis=1)

    angles = np.linspace(0, 2 * np.pi, label.shape[0], endpoint=False)[np.newaxis, :]

    print(statis.shape, angles.shape)
    # close the plot

    for i in range(statis.shape[0]):
        angles = np.concatenate((angles, [angles[0]]), axis=0)
    statis = np.concatenate((statis, [statis[0]]))

    print(statis.shape, angles.shape)
    import seaborn as sns

    fig = plt.figure()
    for i in range(statis.shape[0]):
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles[i], statis[i], 'o-', linewidth=2)#, color=colors_dict[i + 1]
        ax.fill(angles[i], statis[i], alpha=0.25)#color=colors_dict[i + 1]
        # ax.set_title([df.loc[386,"Name"]])
        ax.grid(True)
    ax.set_thetagrids(angles[0] * 180 / np.pi, label)

    import scipy.stats as ss
    dll = 6
    vtest = np.zeros((cond_means.shape[0], projections.shape[1]))
    mean = np.mean(projections, axis=0)
    n = projections.shape[0]
    sigma = np.var(projections, axis=0)
    for i in range(cond_means.shape[0]):
        for j in range(projections.shape[1]):
            print(i, j)
            cluster = np.where(clusters == i + 1)[0]
            ng = cluster.shape[0]
            print(projections[cluster, j][:].shape)
            vt = (cond_means[i, 0, j] - mean[j]) / (np.sqrt(((n - ng) / (n - 1)) * (sigma[j] / ng)))
            print(vt)
            vtest[i, j] = vt

    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)[np.newaxis, :]  # 69

    print(vtest.shape, angles.shape)
    # close the plot

    for i in range(vtest.shape[0]):
        angles = np.concatenate((angles, [angles[0]]), axis=0)
    vtest = np.concatenate((vtest, [vtest[0]]))

    print(vtest.shape, angles.shape)
    import seaborn as sns
    labels = np.arange(10)  # 69
    fig = plt.figure()
    for i in range(vtest.shape[0]):  # vtest.shape[0]):
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles[i], vtest[i], 'o-', linewidth=2)#, color=colors_dict[i + 1]
        ax.fill(angles[i], vtest[i], alpha=0.25)#, color=colors_dict[i + 1]
        # ax.set_title([df.loc[386,"Name"]])
        ax.grid(True)
    ax.set_thetagrids(angles[0] * 180 / np.pi, labels)

def featuresClusters(projections):
    projections=np.array(projections)
    labels = projections[:, 1]
    projections = projections[:, 0]

    def llf(id):
        # labels = np.load('/mnt/data_SSD_2to/190408-44L/labelsAcronymMat.npy')
        # labels = np.load('/mnt/data_SSD_2to/190408_38L/labelsAcronymMat.npy')
        return labels[id]


    projections = normalisation(projections)
    from scipy.cluster.hierarchy import ward, dendrogram, linkage
    plt.figure()
    Z = ward(projections)
    dn = dendrogram(Z, orientation='right', leaf_label_func=llf, leaf_font_size=12)
    plt.show()

    from scipy.cluster.hierarchy import fcluster
    max_d = 0.6
    clusters = fcluster(Z, max_d, criterion='distance')

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj_transformed = pca.fit_transform(projections)
    print(pca.explained_variance_ratio_)

    colors_dict = {0: 'deepskyblue', 1: 'forestgreen', 2: 'firebrick', 3: 'royalblue', 4: 'plum', 5: 'orange',
                   6: 'grey'}

    plt.figure()
    for n in range(proj_transformed.shape[0]):
        plt.scatter(proj_transformed[n][0], proj_transformed[n][1], color=colors_dict[clusters[n]-1])
    plt.show()

    plotStatAnalysis(projections, clusters)


def getMeanDeg(graph):
        md = np.sum(graph.vertex_degrees()) / graph.n_vertices
        return md

def getArtRatio(graph):
    arteries=graph.edge_property('artery')
    return np.sum(arteries)/graph.n_edges

def getNonArtRatio(graph):
    arteries=graph.edge_property('artery')
    return np.sum(np.logical_not(arteries))/graph.n_edges


def isflat(x, y, z):
    res=False
    if (x*3<=y and 3*x<=z):
        print(1)
        res=True
    if (3*y<=x and 3*y<=z):
        print(2)
        res=True
    if (3*z<=x and 3*z<=y):
        print(3)
        res=True
    return res


def istube(x, y, z):
    res=False
    if (x>=3*y and x>=3*z):
        res=True
    if (y>=3*x and y>=3*z):
        res=True
    if (z>=3*x and z>=3*y):
        res=True
    return res


def getGraphVolume(graph):
    coordinates=np.array(graph.edge_geometry('coordinates'))
    for i,coord in enumerate(coordinates):
        if i==0:
            arr=coord
        else:
            arr=np.concatenate((arr, coord), axis=0)

    X = np.max(arr[:,0])-np.min(arr[:,0])
    Y = np.max(arr[:, 1]) - np.min(arr[:, 1])
    Z = np.max(arr[:, 2]) - np.min(arr[:, 2])

    if istube(X, Y, Z):
        shape=1
    if isflat(X, Y, Z):
        shape=2
    else:
        shape=3


    return X*Y*Z,shape



def extractFeaturesSubgraph(graph):
    vol, sha = getGraphVolume(graph)
    f=[getMeanDeg(graph), getArtRatio(graph),getNonArtRatio(graph),  vol, sha]
    return f

def extractFeaturesSubgraphs(graphlist, basis):
    nb_feat=5
    featVect=[]#np.zeros(graphlist.shape[0])
    cyclesVPVect = []
    for i, g in enumerate(graphlist):
        vf=gss4.vertex_property('overlap')==g
        g=gss4.sub_graph(vertex_filter=vf)
        print(g)
        print('extractFeaturesSubgraphs : ', i)
        nbCycles = 0
        if g.n_edges>0:
            f=extractFeaturesSubgraph(g)
            cyclesVP=np.zeros((g.n_vertices ,7))
            for i, b in enumerate(basis):
                res = gtt.subgraph_isomorphism(b.base, g.base, induced=True)
                f.append(len(res))
                if i >= 3:
                    vect = np.zeros(g.n_vertices)
                    if len(res) > 0:
                        nbCycles = nbCycles + 1
                    for j, r in enumerate(res):
                        temp = ggt.vertex_property_map_to_python(r, as_array=True)
                        vect[temp] = 1
                        # print(vect.shape)
                        # print(cyclesVP.shape)
                    cyclesVP[:, i - 5] = vect
            avg_l=np.sum(g.edge_geometry_lengths())/g.n_edges
            nbStuckCycles=np.max(np.sum(cyclesVP!=0, axis=1))
            f.append(nbStuckCycles/nbCycles)
            f.append(nbCycles/g.n_vertices)
            f.append(avg_l)
            featVect.append(f)
            cyclesVPVect.append(cyclesVP)
    print(graphlist.shape, len(featVect))
    return featVect, cyclesVPVect



def extractGraphPerArteryDistance(brain_list, region_list):
    distances_subgraphs=[]
    distances_gt=[]
    for brainnb in brain_list:
        print(brainnb)
        for region in region_list:
            order, level = region
            print(level, order, ano.find_name(order, key='order'))

            label = gts.vertex_annotation();
            label_leveled = ano.convert_label(label, key='order', value='order', level=level)
            vertex_filter = label_leveled == order;

            gss4 = gts.sub_graph(vertex_filter=vertex_filter)
            diff = np.load(
                '/mnt/data_SSD_2to/' + brainnb + '/sbm/diffusion_cluster_per_region_iteration_' + ano.find_name(order,
                                                                                                                key='order') + '.npy')
            gss4.add_vertex_property('diff_val', diff)
            distances, dist_c=np.unique(diff, return_counts=True)
            print(distances)
            print(dist_c)
            for j,d in enumerate(distances):
                if dist_c[j]> 1000:
                    print(d)
                    indices=np.where(diff==d)[0]
                    print(np.array(indices).shape)
                    random.shuffle(indices)
                    indices=indices[:1000]
                    p = mp.Pool(20)
                    filters = [p.map(diffusion, [(i, ind, N_step) for ind, i in enumerate(indices)])]
                    for filter in filters[0]:
                        # filter=diffusion(i, gss4, N_step)
                        # print(ind, np.sum(filter))
                        sg = gss4.sub_graph(vertex_filter=filter)
                        # sg = sg.sub_graph(vertex_filter=sg.vertex_degrees() != 2)
                        sg = sg.largest_component()
                        if sg.n_vertices >= 9:
                            distances_subgraphs.append(sg)
                            distances_gt.append(d)

    return distances_subgraphs, distances_gt


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
    coordinates=graph.vertex_property('coordinates')
    isOver = ( coordinates > mins).all(axis=1)
    isUnder = (coordinates < maxs).all(axis=1)
    return graph.sub_graph(vertex_filter=isOver*isUnder)


# def getmeanFeaturesperClusters(dendogram):
#
# def getvariancExplained(dendogram, mean_values):
#
# def getSpiderNetPlot(dendogram, graphlets, thresholdvalue):



#### get kirst motives
brain_list = ['190506_6R']
region_list=[(1006,3), (580,5),(650,5),(724,5),(811,4),(875,4), (6,6),(463,6),(388,6)]
kmot=createKirstMotif()
g, gts, base = initialize_brain_graph('190506_6R')
N_step=6
gl_test=extractRandomSubgraph(brain_list,region_list, N_step, 100 )

n=0
for b in kmot:
    pos = gtd.sfdp_layout(b.base)
    gtd.graph_draw(b.base, pos=pos,
                   output='/home/sophie.skriabine/Pictures/paperreviews/kmot/kmot_sfdp_' + str(n) + '.pdf')
    n=n+1

motives=[]
for i, b in enumerate(kmot):
    print(i)
    arr=np.random.shuffle(subgraphs)
    for g in subgraphs:
        # print(g)
        res = gtt.subgraph_isomorphism(b.base, g.base, induced=True)
        if len(res)>1:
            motives.append((res[0], g, b))
            print('bingo')
            break

red_blue_map = {1: [1.0, 0, 0, 1.0], 0: [1.0, 1.0, 0.0]}
for i, b in enumerate(motives):
    g = b[1]
    vp=ggt.vertex_property_map_to_python(b[0], as_array=True)
    vertex_filter=np.zeros(g.n_vertices)
    vertex_filter[vp]=1
    g.add_vertex_property('vf', vertex_filter)
    vertex_filter=g.base.vertex_properties['vf']
    v_color = g.base.new_vertex_property('vector<double>')

    v_color = g.base.new_vertex_property('vector<double>')
    for v in g.base.vertices():
        res = 0
        if vertex_filter[v]:
            res = 1
        else:
            res = 0
        v_color[v] = red_blue_map[res]


    g.base.vertex_properties['v_color'] = v_color
    vertex_filtrer=[np.array(red_blue_map[v]) for v in vertex_filter]
    vertex_filtrer = np.array(vertex_filtrer)
    m=b[2]
    # s_g=g.sub_graph(vertex_filter=vertex_filter)
    p3d.plot_graph_mesh(g, vertex_colors=g.vertex_property('v_color'))





import ClearMap.IO.IO as io

##### test
brain_list = ['190506_6R']
g, gts, base = initialize_brain_graph('190506_6R')
region_list=[(1006,3), (580,5),(650,5),(724,5),(811,4),(875,4), (6,6),(463,6),(388,6)]
N_step=4
simpla_basis = CreateSimpleBasis()
# region_list=[(1,1)]
gl_test=extractRandomSubgraph(brain_list,region_list, N_step, 100000 )


for i in range(len(coordinateG)):
    if i==0:
        Gcoordinate = np.array(coordinateG[i])
    else:
        Gcoordinate=np.concatenate((Gcoordinate, coordinateG[i]))

import ClearMap.Analysis.Measurements.Voxelization as vox


import ClearMap.IO.IO as io
#voxelisation stuff
X_tofit=X[:,[0, 1, 3, 4 ,15, 16, 17]]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_proj_transformed = pca.fit_transform(X_tofit[np.where(labels!=-1)])
labels=labels[np.where(labels!=-1)]
Gcoordinate=Gcoordinate[:labels.shape[0]]
fig=plt.figure()
ax = fig.add_subplot(111)
import scipy.misc
for n in np.unique(labels):
    print(n)
    indtoplot = np.where(labels == n)
    # ax.scatter(X_proj_transformed[indtoplot, 0], X_proj_transformed[indtoplot, 1],
    #            color=colors_dict[n + 1])  # X_proj_transformed[indtoplot,2]
    v = vox.voxelize(Gcoordinate[:], shape=(320, 528, 228), weights=None, radius=(5, 5, 5),
                     method='sphere');
    np.save('/home/sophie.skriabine/Pictures/graphlet/dbscanclusters/cluser'+str(n)+'.npy',v)
    # io.convert_files('/home/sophie.skriabine/Pictures/graphlet/dbscanclusters/cluser'+str(n)+'.npy','/home/sophie.skriabine/Pictures/graphlet/dbscanclusters/cluser'+str(n)+'.tif' )
    p3d.plot(v)
plt.show()


# 2d hhistogram


# edge length voxelization + heatmaps
control='190506_6R'
brain= 'mesospim'#'190912_SC3'#
gts = ggt.load('/data_SSD_2to/' + brain + '/data_graph_annotated.gt')
gts = ggt.load('/data_SSD_2to/191122Otof/0_9NA/data_graph_annotated.gt')
gss=gts.largest_component()

rad=gss.edge_property('radii')
plt.figure()
plt.hist(rad, bins=100)
plt.yscale('log')
# gts_control= ggt.load('/mnt/data_SSD_2to/' + control + '/data_graph_reduced_transformed.gt')

lengths=np.expand_dims(gss.edge_geometry_lengths(), axis=1)#*1.625/25
connectivity=gss.edge_connectivity()
coordinates=gss.vertex_property('coordinates_atlas')#*1.625/25
# coordinates_control=gts_control.vertex_property('coordinates')#*1.6/25

high_degree=np.expand_dims(gss.vertex_degrees()>=3, axis=1)
high_degree_control=np.expand_dims(gss.vertex_degrees()==3, axis=1)

edges_centers=np.array([(coordinates[connectivity[i,0]]+coordinates[connectivity[i,1]])/2 for i in range(connectivity.shape[0])])
#
plt.figure()
plt.hist(edges_centers[:,2], bins=100)



vox_data=np.concatenate((edges_centers, lengths), axis=1)
# vox_data=np.concatenate((coordinates, high_degree), axis=1)
vox_data_control=np.concatenate((coordinates, high_degree_control), axis=1)
# np.save('/home/sophie.skriabine/Desktop/vox_data.npy', vox_data)
# vox_data=vox_data
#(320,528,228)
#shape = (3736, 7135, 2403)
# (239.104, 456.64, 153.792)
# (239, 456, 153)
# (320, 528, 456)
v = vox.voxelize(vox_data[:, :3], shape = (528, 456, 320), weights=vox_data[:, 3], radius=(5,5,5), method = 'sphere');
# v_control = vox.voxelize(vox_data_control[:, :3], shape = (528, 456, 320), weights=vox_data_control[:, 3], radius=(5,5,5), method = 'sphere');

w=vox.voxelize(vox_data[:, :3], shape =  (528, 456, 320),  weights=None, radius=(15,15,15), method = 'sphere');
# w_control=vox.voxelize(vox_data_control[:, :3], shape =  (528, 456, 320),  weights=None, radius=(5,5,5), method = 'sphere');
# res=np.divide(v,w)
p3d.plot(v.array/w.array)
# w_control[np.asarray(w_control==0).nonzero()[0]]=1
# p3d.plot(w)
p3d.plot([v.array/w.array, v_control.array/w_control.array])

np.save('/data_SSD_2to/mesospim/data_voxelized_deg_4.npy', v)
np.save('/data_SSD_2to/mesospim/bp_density.npy', w)

np.save('/data_SSD_2to/mesospim/data_voxelized_deg_3_.npy', v_control)
np.save('', w_control)




egi=gss.edge_property('edge_geometry_indices')
eradii=gss.edge_geometry_property('radii')
radii=gss.edge_property('radii')

edge_radii=[(eradii[egi[i, 1]-1]+eradii[egi[i, 1]-1])/2 for i in range(egi.shape[0])]
edge_radii=np.array(edge_radii)#*25/1.625
# H, xedges, yedges = np.histogram2d(radii,edges_centers[:,1], bins=(100, 100))
import matplotlib as mpl
plt.figure()
# connectivity=gss.edge_connectivity()
# coordinates=gss.vertex_property('coordinates')#*1.625/25
# edges_centers=np.array([(coordinates[connectivity[i,0]]+coordinates[connectivity[i,1]])/2 for i in range(connectivity.shape[0])])
edges2plot=edges_centers[np.asarray(edges_centers[:,0]>17).nonzero()[0]]
radii2plot=radii[np.asarray(edges_centers[:,0]>17).nonzero()[0]]
#[edges2plot]
degrees_m = gss.vertex_degrees()
deg4 = np.asarray(degrees_m == 4).nonzero()[0]
print('deg4 init ', deg4.shape[0]/gss.n_vertices)
plt.hist2d(edges2plot[:,1], radii2plot, bins=(100, 100),cmap='jet')#,norm=mpl.colors.LogNorm()), bins=(100, 100)

degrees_m = gss.vertex_degrees()
deg4 = np.asarray(degrees_m==4).nonzero()[0]
print('deg4 init ', deg4.shape[0]/gss.n_vertices)
plt.figure()
x=edges_centers[:,0][deg4][np.asarray(edges_centers[:,0][deg4]>500).nonzero()[0]]
plt.hist(x, bins=500)


io.write('/mnt/data_SSD_2to/190912_SC3/spinal_cord_bp.npy', w.astype('uint8'))
m=io.read('/mnt/data_SSD_2to/190912_SC3/spinal_cord_bp.npy')
io.write('/mnt/data_SSD_2to/190912_SC3/spinal_cord_bp.tif',m)

res=v_control/w_control
res=np.nan_to_num(res)
M=np.max(res)
res=res*255/M
io.write('/mnt/data_SSD_2to/AlbaNA_06/control_res_edge_length.npy', res.astype('uint8'))
m=io.read('/mnt/data_SSD_2to/AlbaNA_06/control_res_edge_length.npy')
io.write('/mnt/data_SSD_2to/AlbaNA_06/control_res_edge_length.tif',m)






test = []  # np.zeros(graphlist.shape[0])
for i, g in enumerate(gl):
    print('extractFeaturesSubgraphs : ', i)
    v, x, y, z = getGraphVolume(g)
    test.append([x, y, z])

test= np.array(test)


cyclesVPVect=[]
for i, g in enumerate(gl):
    print('extractFeaturesSubgraphs : ', i)
    cyclesVP=np.zeros((g.n_vertices ,7))
    n=1
    for i, b in enumerate(simpla_basis):
        if i >= 3:
            res = gtt.subgraph_isomorphism(b.base, g.base, induced=True)
            vect=np.zeros(g.n_vertices)
            for j, r in enumerate(res):
                temp=ggt.vertex_property_map_to_python(r, as_array=True)
                vect[temp]=j
                # n=n+1
            print(vect.shape)
            print(cyclesVP.shape)
            cyclesVP[:, i-4]=(vect)
    cyclesVPVect.append(cyclesVP)




##### end test

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test[:,0], test[:,1], test[:,2])
ax.view_init(30, 180)
ax.set_xlabel('deg')
ax.set_ylabel('art')
ax.set_zlabel('volume')




for i, n in enumerate(simpla_basis):
    pos = gtd.sfdp_layout(n.base)
    gtd.graph_draw(n.base, pos=pos,
                   output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/simple_basis/simple_basis_sfdp_' + str(
                       i) + '.pdf')


ans = [getSubgraphDecomposition((e, gts_filtered)) for e in reg_list.keys()]


import pickle
with open('/mnt/data_SSD_2to/canongraphlet_190506_6R_new/simple_basis/190506_6R_simple_basis.p', 'wb') as fp:
    pickle.dump(ans, fp, protocol=pickle.HIGHEST_PROTOCOL)


projections=np.array(ans)

featuresClusters(ans)

gl=subgraphs
distances_subgraphs, distances_gt
distances_gt=np.array(distances_gt)

class1=np.where(distances_gt<=3)
class2=np.where(np.logical_and(distances_gt<=6, distances_gt>3))
class3=np.where(np.logical_and(distances_gt<=9, distances_gt>6))
class4=np.where(distances_gt>9)
new_distance_gt=np.ones(distances_gt.shape[0])
new_distance_gt[class2]=2
new_distance_gt[class3]=3
new_distance_gt[class4]=4


overlap=np.load('/data_SSD_2to/' + brainnb + '/sbm/diffusion_penetrating_vessel_overlap_end_point_cluster_per_region_iteration_Isocortex.npy')
graph.add_vertex_property('overlap', overlap)
gss4=graph.largest_component()
overlap_u, c = np.unique(overlap, return_counts=True)

featVect, cyclesVPVect=extractFeaturesSubgraphs(overlap_u, simpla_basis)#distances_subgraphs
featVect=np.array(featVect)
featVect=np.nan_to_num(featVect)

region_list=[(1006,3), (580,5),(650,5),(724,5),(811,4),(875,4), (6,6),(463,6),(388,6)]
level_range=[3,4,5,6]
order_range=[[1006],[811,875],[580,650,724],[6,463,388]]

##### get brain region as label
# to do ; over isocortex sub regions
#####

overlap=graph.vertex_property('overlap')

for u in np.unique(overlap):
    s_g=graph.sub_graph(vertex_filter=overlap==u)
    label = s_g.vertex_annotation();
    print(s_g)
    reg = [0, 0, 0]
    reg = np.array(reg)
    l=0
    for level in level_range:
        label_leveled = ano.convert_label(label, key='order', value='order', level=level)
        for order in order_range[l]:
            n=np.sum(label_leveled==order)
            if n>0:
                b=[order, level, n]
                reg=np.row_stack((reg, b))
                print(reg)
        l=l+1
        print(reg)
    r= np.argmax(reg[:, :, 2]
    reg_inex=np.where(region_list==(reg[])))


#extract isocoetex subregion:
subreg_label=np.zeros(gss4.n_vertices)

for e in reg_list.keys():
    n = n + 1
    print(str(n) + '/' + str(len(reg_list.keys())))
    label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
    vertex_filter = label_leveled == e  # 54;
    subreg_label[vertex_filter]=e
    print(ano.find_name(e, key='order'))

gss4.add_vertex_property('subreg_label', subreg_label)
overlap = gss4.vertex_property('overlap')

labels=[]
for u in np.unique(overlap):
    s_g = gss4.sub_graph(vertex_filter=overlap == u)
    if s_g.n_edges>0:
        print(s_g)
        u, c=np.unique(s_g.vertex_property('subreg_label'), return_counts=True )
        print(ano.find_name(u[0], key='order'))
        labels.append(u[0])

labels=np.array(labels)
    print(labels.shape)

X = StandardScaler().fit_transform(featVect)
from sklearn.decomposition import PCA
X_tofit=X[:,[0, 1, 3, 4 ,15, 16, 17]]
pca = PCA(n_components=2)
X_proj_transformed = pca.fit_transform(X_tofit)#X[:,[0,1,3,4,15,16]])

fig = plt.figure()
ax = fig.add_subplot(111)  # , projection='3d')

for n in np.unique(labels):
    indtoplot = np.where(labels == n)[0]
    ax.scatter(X_proj_transformed[indtoplot, 0], X_proj_transformed[indtoplot, 1],
               color=ano.find_color(n, key='order'), alpha=0.5)  # X_proj_transformed[indtoplot,2]
# ax.view_init(30, 185)
plt.show()



def llf(id):
  return ano.find_name(labels[id], key='order')

Z = ward(X_tofit)

x_un, x_ct=np.unique(labels, return_counts=True)

plt.figure()
dn = dendrogram(Z, orientation='right', leaf_font_size=12, leaf_label_func=llf, color_threshold=45)
plt.show()

from scipy.cluster.hierarchy import fcluster
assignments = fcluster(Z,45,'distance')
print(np.unique(assignments))
#check asignements clusters population
for u in np.unique(assignments):
    clusters_names=[]
    cluster_ids=np.where(assignments==u)[0]
    # for id in cluster_ids:
    #     clusters_names.append(ano.find_name(labels[id], key='order'))

    un, ct=np.unique(labels[cluster_ids], return_counts=True)
    un_name=np.array([ano.find_name(ui, key='order') for ui in un])
    # un_name=np.unique(clusters_names)
    ct_tot=[]
    for n in un:
        ct_tot.append(x_ct[np.where(x_un==n)[0]])
    plt.figure()

    arr=ct/np.array(ct_tot).squeeze(axis=1)

    m=np.mean(arr)
    std=np.std(arr)
    # plt.bar(range(un.shape[0]), ct / np.array(ct_tot).squeeze(axis=1))  #
    # plt.xticks(range(un.shape[0]), un_name.tolist(), rotation='vertical')
    for i in range(arr.shape[0]):
        color='skyblue'
        if arr[i]>=m+std/2:
            color = 'indianred'
        plt.bar(i, arr[i], color=color)  #
    plt.xticks(range(un.shape[0]), un_name.tolist(), rotation=45, ha='right')
    plt.title('cluster' + str(u))
    print(u,un_name, ct/np.array(ct_tot).squeeze(axis=1) )









from sklearn.preprocessing import StandardScaler
#RandomForest
from sklearn.model_selection import train_test_split
X = StandardScaler().fit_transform(featVect)
X_train, X_valid, y_train, y_valid=train_test_split(X, new_distance_gt, test_size=0.3, random_state=42, stratify=new_distance_gt)
X_train.shape , y_train.shape, X_valid.shape, y_valid.shape

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=10, min_samples_split=100 , min_samples_leaf=100 )
rf.fit(X_train, y_train)

y_valid_pred=rf.predict(X_valid)
print(y_valid[:10])
print(y_valid_pred[:10])

from sklearn.metrics import accuracy_score
acc_test=accuracy_score(y_valid, y_valid_pred)
acc_train=accuracy_score(y_train, rf.predict(X_train))
print(acc_test, acc_train)





Z = ward(featVect[:, [0,1,3]])

plt.figure()
dn = dendrogram(Z, orientation='right', leaf_font_size=12)
plt.show()

max_d = 19000
clusters = fcluster(Z, max_d, criterion='distance')

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
proj_transformed = pca.fit_transform(featvect[:, [0,1,3, 15, 16]])
print(pca.explained_variance_ratio_)

colors_dict = {0: 'deepskyblue', 1: 'forestgreen', 2: 'firebrick', 3: 'royalblue', 4: 'plum', 5: 'orange',
               6: 'grey'}

from sklearn import metrics
from sklearn.cluster import DBSCAN
X = StandardScaler().fit_transform(featVect)#[:, [0, 1, 3, 4 ,15, 16]]


# Compute DBSCAN

#get capillaries beds
# u,c=np.unique(labels, return_counts=True)
# biggest_cluster=X[np.where(labels==u[np.argmax(c)])]

X_tofit=X[:,[0, 1, 3, 4 ,15, 16, 17]]
db = DBSCAN(eps=0.62, min_samples=10).fit(X_tofit)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)



fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
for n in np.unique(labels)[1:]:
    indtoplot=np.where(labels==n)[0]
    ax.scatter(X[indtoplot,0], X[indtoplot,1], X[indtoplot,2] , alpha=0.1)
ax.view_init(30, 185)
ax.set_xlabel('deg')
ax.set_ylabel('art')
ax.set_zlabel('volume')
# plt.show()

X_tofit=X[:,[0, 1, 3, 4 ,15, 16, 17]]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_proj_transformed = pca.fit_transform(X_tofit)#X[:,[0,1,3,4,15,16]])


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_proj_transformed = tsne.fit_transform(X_tofit)

# import matplotlib
# matplotlib.use('Qt5Agg',warn=False, force=True)
# from matplotlib import pyplot as plt
# plt.switch_backend('Qt5Agg')
labels=new_distance_gt

fig=plt.figure()
ax = fig.add_subplot(111)#, projection='3d')
for n in np.unique(labels)[1:]:
    indtoplot=np.where(labels==n)[0]
    ax.scatter(X_proj_transformed[indtoplot,0], X_proj_transformed[indtoplot,1], color=colors_dict[n+1])# X_proj_transformed[indtoplot,2]
# ax.view_init(30, 185)
plt.show()


def getVeins(g):
    red_blue_map = {1: [1.0, 0, 0, 1.0], 0: [1.0, 1.0, 0.0]}
    radii = g.edge_geometry('radii', as_list=False, return_indices=False);

    huge_radius = 1.5
    big_radius = 0.4

    edge_artery_label = g.edge_property('artery')
    coordinates, edge_indices = g.edge_geometry(as_list=False, return_indices=True)

    huge_vessels_label = np.array([np.max(radii[ei[0]:ei[1]]) >= huge_radius for ei in edge_indices], dtype='int');
    big_vessels_label = np.array([np.max(radii[ei[0]:ei[1]]) >= big_radius for ei in edge_indices], dtype='int');
    edge_vein_label = np.logical_or(huge_vessels_label,
                                    np.logical_and(big_vessels_label, np.logical_not(edge_artery_label)))

    # g.add_edge_property('vein', edge_vein_label);
    # edge_vein_label = g.edge_property('vein');
    vertex_rad=np.zeros(sub.n_vertices)
    con=sub.edge_connectivity()
    for i,e in enumerate(sub.edges):
        # print(i, e)
        # res = edge_vein_label[i]
        vertex_rad[con[i, 0]] = edge_vein_label[i]
        vertex_rad[con[i, 1]] = edge_vein_label[i]

    return vertex_rad

# plt.switch_backend('Agg')
# fig=plt.figure()
veins_stat_vect=[]
art_stat_vect=[]
for n in np.unique(labels):#[1:]:
    print(n)
    veins_stat = []
    art_stat = []
    indtoplot=np.where(labels==n)[0]
    random.shuffle(indtoplot)
    # ind=indtoplot[0]
    for ind in indtoplot:
        sub=distances_subgraphs[ind]#[0]#gl
        print(ind)#[0]
        # pos = gtd.sfdp_layout(sub.base)
        # red_blue_map = {1: [1.0, 0, 0, 1.0], 0: [1.0, 1.0, 0.0], 2:[0,0, 1.0, 1.0]}
        # v_color = sub.base.new_vertex_property('vector<double>')
        # j = 0
        artery=sub.edge_property('artery')
        art=np.where(artery==1)
        vein=getVeins(sub)
        sumV=np.sum(vein)
        sumA = np.sum(artery)
        veins_stat.append(sumV)
        art_stat.append(sumA)
        # print(art)
        # print(np.where(vein==1))
        # lp=np.zeros(sub.n_vertices)
        #
        # art_connectivity=sub.edge_connectivity()[art].flatten()
        # lp[art_connectivity]=1
        #
        # sub.add_vertex_property('artery',lp)
        # sub.add_vertex_property('vein', vein)
        # lp=sub.base.vertex_properties['artery']
        # vein=sub.base.vertex_properties['vein']
    veins_stat_vect.append(veins_stat)
    art_stat_vect.append(art_stat)


fig=plt.figure()

for i,stat in enumerate(veins_stat_vect):

    # fig.add_subplot(2,1,1)
    vein_v=np.array(veins_stat_vect[i])>0
    mean=np.mean(vein_v)
    std=np.var(vein_v)
    plt.bar(3*(i-1),mean, yerr=std,color='b')
    plt.errorbar(3*(i-1), std)
    # plt.xlim(0, 1)
    # fig.add_subplot(2, 1, 2)
    art_v = np.array(art_stat_vect[i]) > 0
    mean = np.mean(art_v)
    std = np.var(art_v)
    plt.bar(3*(i-1)+1,mean, yerr=std,color='r')
    plt.errorbar(3*(i-1)+1, std)
    # plt.xlim(0, 1)
plt.xticks([3*(i-1) for i in range(5)], ['group 0', 'group 1', 'group 2', 'group 3'])#, 'group 4'

red_blue_map = {1: [1.0, 0, 0, 1.0], 0: [1.0, 1.0, 0.0], 2:[0,0, 1.0, 1.0]}
for n in np.unique(labels):#[1:]:
    print(n)
    veins_stat = []
    art_stat = []
    indtoplot=np.where(labels==n)[0]
    random.shuffle(indtoplot)
    ind=indtoplot[0]
    sub=gl[ind]#[0]#gl
    print(ind)#[0]
    lp=np.zeros(sub.n_vertices)
    artery = sub.edge_property('artery')
    art = np.where(artery == 1)
    vein = getVeins(sub)
    art_connectivity=sub.edge_connectivity()[art].flatten()
    lp[art_connectivity]=1

    sub.add_vertex_property('artery',lp)
    sub.add_vertex_property('vein', vein)
    lp=sub.base.vertex_properties['artery']
    vein=sub.base.vertex_properties['vein']
    v_color = sub.base.new_vertex_property('vector<double>')
    for v in sub.base.vertices():
        res = 0
        if lp[v]:
            res = 1
        elif vein[v]:
            res = 2
        v_color[v] = red_blue_map[res]

    pos = gtd.sfdp_layout(sub.base)
    sub.base.vertex_properties['v_color'] = v_color
    gtd.graph_draw(sub.base, pos=pos,vertex_fill_color=sub.base.vertex_properties['v_color'], output='/home/sophie.skriabine/Pictures/graphlet/clusterswlength/clusters_graph_1' + str(n) + '.pdf')
    p3d.plot_graph_mesh(sub, vertex_colors=sub.vertex_property('v_color'))




colors_dict = {0: 'deepskyblue', 1: 'forestgreen', 2: 'firebrick', 3: 'royalblue', 4: 'plum', 5: 'orange',6: 'grey'}

cols=['forestgreen','firebrick', 'royalblue', 'plum','orange']
for i in range(len(region_list)):
    fig = plt.figure()
    graphs_cat=labels[1000*i:1000*(i+6)]
    u, c=np.unique(graphs_cat, return_counts=True)
    plt.bar(u,c, color=cols)







from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(featvect[:,0], featvect[:,1], featvect[:,3], c='skyblue', s=60, alpha=0.1)
ax.view_init(30, 185)
ax.set_xlabel('deg')
ax.set_ylabel('art')
ax.set_zlabel('volume')
plt.show()



def plotSpideNet(projections, clusters):
    cond_means = []
    cond_stds = []
    groups_nb=0
    U,c=np.unique(clusters, return_counts=True)
    for i, u in enumerate(U):#U[1:]
        if u !=-1:
            if c[i]>10:#100
                groups_nb=groups_nb+1
                temp = projections[np.where(clusters == u), :]
                # print(temp.shape)
                temsp_means = np.mean(temp, axis=1)
                temsp_stds = np.std(temp, axis=1)
                # print(temsp_means.shape)
                cond_means.append(temsp_means)
                cond_stds.append(temsp_stds)

    print(groups_nb)
    cond_means = np.array(cond_means)
    cond_stds = np.array(cond_stds)
    print(cond_means.shape)
    # pred_proj = [cond_means[clusters[n] - 1] for n in range(projections.shape[0])]
    # pred_proj = np.array(pred_proj)
    # mains_clusters = [n for n in range(clusters.shape[0])]# if clusters[n] not in [4, 5, 6]]
    # pred_proj = pred_proj[mains_clusters]
    # p = np.squeeze(pred_proj, axis=1)
    # coefficient_of_dermination = []
    # for i in range(p.shape[1]):
    #     coefficient_of_dermination.append(r2_score(projections[mains_clusters, i], p[:, i]))
    #
    # coefficient_of_dermination = np.array(coefficient_of_dermination)
    # print(coefficient_of_dermination)
    # impacting_graphlets = np.array(np.where(abs(coefficient_of_dermination) > 0.0))[0]
    # label = impacting_graphlets
    # labels=np.arange(69)
    # labels=np.array([1,3,4,38,39,47,65,49])
    statis = np.squeeze(cond_means[:, :, :], axis=1)
    errors = np.squeeze(cond_stds[:, :, :], axis=1)

    angles = np.linspace(0, 2 * np.pi, projections.shape[1], endpoint=False)[np.newaxis, :]

    print(statis.shape, angles.shape)
    # close the plot

    for i in range(statis.shape[0]):
        angles = np.concatenate((angles, [angles[0]]), axis=0)
    statis = np.concatenate((statis, [statis[0]]))
    errors = np.concatenate((errors, [errors[0]]))
    print(statis.shape, angles.shape)
    import seaborn as sns
    fig = plt.figure()
    for i in np.unique(clusters):#[1:]:
        # for i in range(statis.shape[0]):
        i=int(i)
        print(i)
        print(statis[i])
        print(errors[i])
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles[i], statis[i], 'o-', linewidth=2, color=colors_dict[i+1])#, color=colors_dict[i + 1]
        # ax.fill(angles[i], statis[i], alpha=0.25)#color=colors_dict[i + 1]
        ax.fill_between(angles[i], statis[i] - errors[i], statis[i] + errors[i],alpha=0.25, color=colors_dict[i+1])
        # ax.set_title([df.loc[386,"Name"]])
        ax.grid(True)
        ax.set_thetagrids(angles[0] * 180 / np.pi, range(projections.shape[1]))


plt.figure()
sns.jointplot(featvect[:,:1], featvect[:,1:2    ], kind='kde')

plotSpideNet(X, labels)#[:,[0,1,3,4,15,16]]

plotSpideNet(X_tofit, labels)
plotSpideNet(X[:,[5,6,7,8,9,10,11,12,13,14]], labels)

plotStatAnalysis(X_tofit[:,-10:], labels)#featVect




#get capillaries beds
u,c=np.unique(labels, return_counts=True)
X_tofit=X[:,[5:14]]
biggest_cluster=X_tofit[np.where(labels==u[np.argmax(c)])]
# biggest_cluster_all_feat=featvect[np.where(labels==u[np.argmax(c)])]

db_cap = DBSCAN(eps=0.115, min_samples=20).fit(biggest_cluster)
core_samples_mask = np.zeros_like(db_cap.labels_, dtype=bool)
core_samples_mask[db_cap.core_sample_indices_] = True
labels = db_cap.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)



fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
for n in np.unique(labels)[1:]:
    indtoplot=np.where(labels==n)[0]
    ax.scatter(biggest_cluster[indtoplot,0], biggest_cluster[indtoplot,1], biggest_cluster[indtoplot,2] , alpha=0.1)
ax.view_init(-91, 88)
ax.set_xlabel('deg')
ax.set_ylabel('art')
ax.set_zlabel('volume')
# plt.show()

#keep the biggest clusters
u,c=np.unique(labels, return_counts=True)
labels_big_clusters=u[np.where(c>100)][1:]
labels_bc=labels[np.logical_and(labels>=0, labels<=5)]

plotStatAnalysis(biggest_cluster_all_feat[np.logical_and(labels>=0,labels<=5),-10:], labels_bc)




brain_list=['190506_6R']#['190408_39L','190506_6R', '190506_3R']#'190408_38L'
brainnb=brain_list[0]
region_list=[(1006,3), (580,5),(650,5),(724,5),(811,4),(875,4), (6,6),(463,6),(388,6)]
model_delsta, proba, labels, KernelProductMatrix = SVMModel(region_list, brain_list, 1, 2500, False)

import pickle
with open('/mnt/data_SSD_2to/svcmodel_delsta.p', 'wb') as fp:
  pickle.dump(model_delsta, fp, protocol=pickle.HIGHEST_PROTOCOL)

#[   6.,  388.,  463.,  580.,  650.,  724.,  811.,  875., 1006.]
classe=1006
KPM=KernelProductMatrix[np.where(labels==classe)]
SVMvalidationTest(model_delsta, KPM, labels[np.where(labels==classe)])


SVMTest(model, ['190408_44L'], region_list, 1000)

import pickle
with open('/mnt/data_SSD_2to/svcmodel.p', 'rb') as fp:
  model=pickle.load(fp)

g, gts, base = initialize_brain_graph('190506_6R')
brain_list=['190506_6R']#['190408_39L','190506_6R', '190506_3R']#'190408_38L'
brainnb=brain_list[0]
region_list=[(6,6)]#[(1006,3), (580,5),(650,5),(724,5),(811,4),(875,4), (6,6),(463,6),(388,6)]

gts_filtered=filter_graph(gts)

test=getshortestpathssubgraphs(brain_list, region_list, 7, 13)#22,13, graphs_list

gl=test[0]
orders=test[1]
canon=check_isomorphisms(gl, 18)
canon_isofree=check_isomorphisms_full(canon, 13)

import pickle
with open('/mnt/data_SSD_2to/190428_6R/canonbase4_new.p', 'wb') as fp:
  pickle.dump(canon, fp, protocol=pickle.HIGHEST_PROTOCOL)
# g, gts, base = initialize_brain_graph('190506_6R')


var=[]
small_graph=[]
for i,c in enumerate(canon):
    deg=np.sum(c.vertex_degrees())/c.n_vertices
    print(deg)
    if deg<1.9:
        small_graph.append(i)
    var.append(deg)
print(len(small_graph))
print(small_graph)
g=sns.distplot(var)



canon_filtered=np.delete(np.array(canon), small_graph)
hm=getSimilarityCanonGraph(canon_filtered)

l=np.where(hm>=0.55)
L=(sort_tuples([(l[0][i], l[1][i]) for i in range(l[0].shape[0]) if l[1][i]!=l[0][i]]))
L=list(set(L))
print(len(L))
print(L)

#sorting
for n in small_graph:
    sub=canon[n].base
    pos = gtd.sfdp_layout(sub.base)
    gtd.graph_draw(sub.base, pos=pos,
                   output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/test/linear/graph_draw_sfdp_' + str(n) + '.pdf')

for n,sub in enumerate(canon):
    if n not in small_graph:
        pos = gtd.sfdp_layout(sub.base)
        gtd.graph_draw(sub.base, pos=pos,
                       output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/test/cyclic/graph_draw_sfdp_' + str(n) + '.pdf')




def gt2nx(gt_g):
    G = nx.Graph()
    G.add_node(gt_g.n_vertices)
    connectivity=gt_g.edge_connectivity()
    G.add_edges_from(connectivity)
    return G


def GED(g1, g2):
    ged=nxas.graph_edit_distance(g1, g2)
    return ged

graph_edit_distace_mat=np.zeros((canon.shape[0],canon.shape[0] ))
for i, cf1 in enumerate(canon):
    nx_c1 = gt2nx(cf1)
    for j, cf2 in enumerate(canon):
        print(i, j)
        nx_c2 = gt2nx(cf2)
        for k,v in enumerate(nxas.optimize_graph_edit_distance(nx_c1, nx_c2)):
            print(v)
            if k==0:
                res=v
                break
        # res=GED(nx_c1, nx_c2)
        print(res)
        graph_edit_distace_mat[i, j]=res



z=hierarchy.linkage(graph_edit_distace_mat, 'ward')
ax = sns.clustermap(graph_edit_distace_mat,  row_linkage=z, col_linkage=z)

test=graph_edit_distace_mat
for i in range(test.shape[0]):
    test[i, i]=100

pairs=np.array([[i, np.argmin(test[i]), test[i, np.argmin(test[i])]] for i in range(test.shape[0])])
plt.figure()
g=sns.distplot(pairs[:,2])

for n,sub in enumerate(canon):
        pos = gtd.sfdp_layout(sub.base)
        gtd.graph_draw(sub.base, pos=pos,
                       output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/test/filtered/all_bis/graph_draw_sfdp_' + str(n) + '.pdf')

similar_pairs=pairs[np.where(pairs[:,2]<22)][:, :2]#15
print(similar_pairs.shape)
L=sort_tuples(similar_pairs)#[(l[0][i], l[1][i]) for i in range(l[0].shape[0]) if l[1][i]!=l[0][i]]))
L=list(set(L))
print(len(L))
print(L)
u, c=np.unique(similar_pairs, return_counts=True)
ind_to_delete=[similar_pairs[i, np.argmax([c[np.where(u==similar_pairs[i][0])], c[np.where(u==similar_pairs[i][1])]])] for i in range(similar_pairs.shape[0])]
canon_filtered_mega=np.delete(np.array(canon_filtered), ind_to_delete)

ar=[]
small_graph=[]
for i,c in enumerate(canon_filtered_mega):
    deg=np.sum(c.vertex_degrees())/c.n_vertices
    print(deg)
    if deg<1.9:
        small_graph.append(i)
    var.append(deg)
print(len(small_graph))
print(small_graph)
# plt.figure()
# g=sns.distplot(var)
k=0
for n,sub in enumerate(canon_filtered_mega):
    if n not in small_graph:
        pos = gtd.sfdp_layout(sub.base)
        gtd.graph_draw(sub.base, pos=pos, output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/test/filtered/filtered_mega_bis/graph_draw_sfdp_' + str(k) + '.pdf')
        k=k+1


import pickle
with open('/mnt/data_SSD_2to/190428_6R/canonbase4_new.p', 'rb') as fp:
  canon = pickle.load(fp)



with open('/mnt/data_SSD_2to/181002_4/reg_list.p', 'rb') as fp:
  reg_list = pickle.load(fp)

# n=0
# for graph in canon:
#     if n==69:
#         p=p3d.plot_graph_mesh(graph)
#     n=n+1
#
n=0
projection_on_basis=[]
region=[]

b='190408_38L'#'190506_6R'

import ipyparallel as ipp
rc=ipp.Client()
p = rc[:20]
ans = [getSubgraphDecomposition((e, gts_filtered)) for e in reg_list.keys()]
# p = mp.Pool(20)
args=[(e, b) for e in reg_list.keys()]
print('args')

ans = [p.map(getSubgraphDecomposition, args)]


import pickle
with open('/mnt/data_SSD_2to/190506_6R_nonpara.p', 'wb') as fp:
  pickle.dump(ans, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('/mnt/data_SSD_2to/190506_6R_nonpara.p', 'rb') as fp:
  projections = pickle.load(fp)

projections=np.array(ans)
labels=projections[:, 1]
projections=projections[:, 0]

projections=normalisation(projections)
from scipy.cluster.hierarchy import ward, dendrogram, linkage
Z=ward(projections)
dn = dendrogram(Z, orientation='right', leaf_label_func=llf,leaf_font_size=12)
plt.show()

from scipy.cluster.hierarchy import fcluster
max_d = 0.6
clusters = fcluster(Z, max_d, criterion='distance')


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
proj_transformed=pca.fit_transform(projections)
print(pca.explained_variance_ratio_)

colors_dict={0:'deepskyblue', 1:'forestgreen', 2:'firebrick', 3:'royalblue', 4:'plum', 5:'orange', 6:'grey'}

plt.figure(2)
for n in range(proj_transformed.shape[0]):
    plt.scatter(proj_transformed[n][0],proj_transformed[n][1], color=colors_dict[clusters[n]])
plt.show()

cond_means=[]
for u in np.unique(clusters):
    temp=projections[np.where(clusters==u), :]
    print(temp.shape)
    temsp_means=np.mean(temp, axis=1)
    print(temsp_means.shape)
    cond_means.append(temsp_means)

cond_means=np.array(cond_means)
print(cond_means.shape)
pred_proj=[cond_means[clusters[n]-1] for n in range(projections.shape[0])]
pred_proj=np.array(pred_proj)
mains_clusters=[n  for n in range(clusters.shape[0]) if clusters[n] not in [4,5,6]]
pred_proj=pred_proj[mains_clusters]
p=np.squeeze(pred_proj, axis=1)
coefficient_of_dermination=[]
for i in range(p.shape[1]):
    coefficient_of_dermination.append(r2_score(projections[mains_clusters, i], p[:, i]))

coefficient_of_dermination=np.array(coefficient_of_dermination)

impacting_graphlets=np.array(np.where(coefficient_of_dermination>0.25))[0]
labels=impacting_graphlets
# labels=np.arange(69)
# labels=np.array([1,3,4,38,39,47,65,49])
statis=np.squeeze(cond_means[:,:,labels], axis=1)

angles=np.linspace(0, 2*np.pi, labels.shape[0], endpoint=False)[np.newaxis, :]

print(statis.shape, angles.shape)
# close the plot

for i in range(statis.shape[0]):
    angles=np.concatenate((angles,[angles[0]]), axis=0)
statis=np.concatenate((statis,[statis[0]]))

print(statis.shape, angles.shape)
import seaborn as sns

fig = plt.figure()
for i in range(statis.shape[0]):
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles[i], statis[i], 'o-', linewidth=2)
    ax.fill(angles[i], statis[i], alpha=0.25)
    # ax.set_title([df.loc[386,"Name"]])
    ax.grid(True)
ax.set_thetagrids(angles[0] * 180/np.pi, labels)


import scipy.stats as ss
dll=6
vtest=np.zeros((cond_means.shape[0], projections.shape[1]))
mean=np.mean(projections, axis=0)
n=projections.shape[0]
sigma=np.var(projections, axis=0)
for i in range(cond_means.shape[0]):
    for j in range(projections.shape[1]):
        print(i, j)
        cluster=np.where(clusters==i+1)[0]
        ng=cluster.shape[0]
        print(projections[cluster, j][:].shape)
        vt=(cond_means[i, 0, j]-mean[j])/(np.sqrt(((n-ng)/(n-1))*(sigma[j]/ng)))
        print(vt)
        vtest[i, j]=vt


angles=np.linspace(0, 2*np.pi, 10, endpoint=False)[np.newaxis, :]#69

print(vtest.shape, angles.shape)
# close the plot

for i in range(vtest.shape[0]):
    angles=np.concatenate((angles,[angles[0]]), axis=0)
vtest=np.concatenate((vtest,[vtest[0]]))

print(vtest.shape, angles.shape)
import seaborn as sns
labels=np.arange(10)#69
fig = plt.figure()
for i in range(vtest.shape[0]):#vtest.shape[0]):
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles[i], vtest[i], 'o-', linewidth=2, color=colors_dict[i+1])
    ax.fill(angles[i], vtest[i], alpha=0.25, color=colors_dict[i+1])
    # ax.set_title([df.loc[386,"Name"]])
    ax.grid(True)
ax.set_thetagrids(angles[0] * 180/np.pi, labels)


#draw canon base
import graph_tool.draw as gtd
n=0
for i,c in enumerate(canon):
    print(n)
    pos = gtd.sfdp_layout(c.base)
    gtd.graph_draw(c.base, pos=pos, output='/mnt/data_SSD_2to/canongraphlet_190506_6R_new/graph_draw_sfdp_'+str(n)+'.pdf')
    n=n+1


#get similarity heatmap between graphlets of the canon basis

hm=np.zeros((len(canon), len(canon)))
for i, c1 in enumerate(canon):
    for j, c2 in enumerate(canon):
        res=gtt.similarity(c1.base, c2.base)
        hm[i, j]=res
        print(i, j)

import seaborn as sns; sns.set()
import scipy.cluster.hierarchy as hierarchy
z=hierarchy.linkage(hm, 'ward')
ax = sns.clustermap(hm,  row_linkage=z, col_linkage=z)


l=np.where(hm>=0.60)
L=(sort_tuples([(l[0][i], l[1][i]) for i in range(l[0].shape[0]) if l[1][i]!=l[0][i]]))
L=list(set(L))
print(L)

#get auto loops
connectivity = gts.edge_connectivity()
temp=sort_tuples([(connectivity[i, 0],connectivity[i, 1]) for i in range(connectivity.shape[0])])
test=np.empty((len(connectivity,)), dtype=object)
test[:]=[tempi for tempi in temp]
# temp=np.array(temp, dtype=object)
u, counts=np.unique(test, return_counts=True)


duplica=u[np.where(counts>1)]
print(duplica.shape)

doubles=[temp.index(duplica[i]) for i in range(duplica.shape[0])]
loops=np.where(connectivity[:, 0]==connectivity[:, 1])

# doubles=np.where(connectivity[:, 0]==connectivity[:, 1])
lengths_doubles=gts.edge_geometry_lengths()[doubles]
radius_doubles=gts.edge_geometry_property('radii')[doubles]

single_edge_filter=np.ones(gts.n_edges)
single_edge_filter[doubles]=0
single_edge_filter[loops]=0
gts_filtered=gts.sub_graph(edge_filter=single_edge_filter)


# doubles=np.where(connectivity[:, 0]==connectivity[:, 1])
lengths_doubles=gts.edge_geometry_lengths()[doubles]
radius_doubles=gts.edge_geometry_property('radii')[doubles]



g = sns.jointplot(lengths_doubles, radius_doubles, kind="kde", space=0, color="g")
# g=sns.kdeplot(lengths_doubles, radius_doubles,cmap="Blues", shade=True, bw=.15)
ax = g.ax_joint
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('lengths')
# ax.set_ylabel('radius')
plt.colorbar(g)
plt.show()

for e in reg_list.keys():
    n = n + 1
    print(str(n) + '/' + str(len(reg_list.keys())))
    region_projection_on_basis, reg_name=getSubgraphDecomposition(e, gts, canon)
    projection_on_basis.append(region_projection_on_basis)
    region.append(reg_name)

iso=[]
for c in canon:
    res=gtt.subgraph_isomorphism(c.base, base)
    iso.append(res)

import pickle
    with open('/mnt/data_SSD_2to/graphs_list.p', 'wb') as fp:
        pickle.dump(graphs_list, fp, protocol=pickle.HIGHEST_PROTOCOL)

# for label in np.unique(labels):
    temp = proba[np.where(labels==388)]
    import matplotlib.pyplot as plt
    import seaborn as sns


    # sns.clustermap(temp)
    # sns.heatmap(temp)

    g = sns.heatmap(temp)
