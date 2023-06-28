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
import graph_tool.topology as gtt
import pandas as pd
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

def getNbEdges(graph_reduced):
    return graph_reduced.n_edges

def getNbBranchPt(graph):
    d=graph.vertex_degrees()
    bp=d>=3
    return np.sum(bp)

def getNbEndPt(graph):
    d=graph.vertex_degrees()
    ep=d==1
    return np.sum(ep)

def getLengthbranches(graph_reduced):
    # vessel_length=[]
    lengths=graph_reduced.edge_geometry_lengths()
    print(lengths)
    # length=0
    # for edge in graph_reduced.edge_geometry_property('coordinates'):
    #     for i in edge
    #     length = length+
    # vessel_length.append(graph_reduced.edge_geometry_property('coordinates').shape[0] / len(graph_reduced.edges))
    return lengths

def getNbCrossedRegions(graph_transformed):
    labels = graph_transformed.vertex_annotation()
    return np.unique(labels).shape[0]

def getBranchVertexLoc(graph_transformed):
    d=graph_transformed.vertex_degrees()
    bp = d>=3
    labels = graph_transformed.vertex_annotation()[bp]
    return labels

def getEndVertexLoc(graph_transformed):
    d=graph_transformed.vertex_degrees()
    ep = d==1
    labels = graph_transformed.vertex_annotation()[ep]
    return labels

def getSomaLoc(graph_transformed):
    soma=getSomas(graph_transformed)
    labels = graph_transformed.vertex_annotation()[soma]
    return labels

def extract_features_neurons(neuron_name, neuron_graph_transformed, neuron_graph_reduced):
    depth, width=graphDepthWidth(neuron_graph_reduced)
    features=(neuron_name, getNbBranchPt(neuron_graph_reduced), getNbEndPt(neuron_graph_transformed), getLengthbranches(neuron_graph_reduced), getSomaLoc(neuron_graph_transformed), getBranchVertexLoc(neuron_graph_transformed), getEndVertexLoc(neuron_graph_transformed), getNbCrossedRegions(neuron_graph_transformed), depth, width )
    # features.append(neuron_name)
    # features.append(getNbBranchPt(neuron_graph_reduced))
    # features.append(getNbEndPt(neuron_graph_transformed))
    # features.append(getLengthbranches(neuron_graph_reduced))
    # features.append(getBranchVertexLoc(neuron_graph_transformed))
    # features.append(getEndVertexLoc(neuron_graph_transformed))
    return [features]

def getSomas(graph):
    # print(graph.vertex_property('identity'))
    id=graph.vertex_property('identity')
    somas = id==1
    return somas.reshape((id.shape[0])).astype(int)

def diffusion(graph, start_vertex):
    print(' nb start_vertex ', np.sum(start_vertex), start_vertex.shape, graph.n_vertices)
    visited_vertices = start_vertex  # .copy()
    new_sum = 0
    n = 0
    while 0 in visited_vertices:
        print(n)
        neigh = np.zeros(graph.n_vertices)
        for i in range(graph.n_vertices):
            if visited_vertices[i] > 0:
                # print(gss4.vertex_neighbours(i))
                neigh[graph.vertex_neighbours(i)] = visited_vertices[i]+1
                # if visited_vertices[gss4.vertex_neighbours(i)]==0:
        update = np.logical_and(neigh > 0, visited_vertices == 0)
        # print(update)
        visited_vertices[update] = neigh[update]
        # visited_vertices = np.minimum(visited_vertices, neigh)
        # print(visited_vertices, neigh)
        print(np.sum(visited_vertices > 0), np.sum(neigh))
        if len(np.unique(neigh)) == 0:
            print('no arteries detected')
            break
        if not new_sum < np.sum(visited_vertices > 0):
            print('converged', new_sum, np.sum(visited_vertices > 0))
            break
        new_sum = np.sum(visited_vertices > 0)
        n = n + 1
    return visited_vertices

def graphDepthWidth(graph):
    start_vertex=getSomas(graph)
    levels=diffusion(graph, start_vertex)
    depth=np.max(levels)
    width=np.max(np.array([np.sum(levels==n) for n in np.unique(levels)]))
    return depth, width


def parseTree(obj):
    # print(obj)
    # if len(obj["children"]) == 0:
    #     leafArray.append(obj['id'])
    # else:
        leafArray.append(obj['id'])

        for child in obj["children"]:
            parseTree(child)

def getleaves(jsonpath='/home/sophie.skriabine/Projects/clearVessel_New/ClearMap/ClearMap/Resources/Atlas/annotation.json'):
    import json
    with open(jsonpath) as json_data:
        data_dict = json.load(json_data)['msg']
        # print(data_dict)
    for data in data_dict:
        global leafArray
        leafArray = []
        tree = data  # json.loads(data.strip())
        parseTree(tree)
        # somehow walk through the tree and find leaves
        # print("")
        # for each in leafArray:
            # print(each)
    return leafArray

def FromLoctovector(locArray):
    leaves=getleaves()
    res=np.zeros((len(leaves)))
    for l in locArray:
        index= np.where(leaves==l)[0]
        res[index]=res[index]+1
    print(res)
    return res

def FromArrayToHist(array):
    # print(array)
    # array=pd.to_numeric(array)#.astype(str).astype(int)
    # print(array)
    hist, bin_edges = np.histogram(array, bins=40, range=(0,200))
    print(hist)
    return hist

def formatline(pd):
    res=[]
    res.append(pd['NbBP'])
    res.append(pd['NbEP'])
    res=np.array(res)
    print(res)
    res=np.concatenate((res,FromArrayToHist(pd['EdgeLength'])), axis=None)
    print(res)
    res = np.concatenate((res, pd['SomaLoc']), axis=None)
    print(res.shape)
    res=np.concatenate((res,FromLoctovector(pd['BPloc'])), axis=None)
    print(res.shape)
    res=np.concatenate((res, FromLoctovector(pd['EPloc'])), axis=None)
    print(res.shape)
    res=np.concatenate((res,pd['NbRegion']), axis=None)
    res=np.concatenate((res,pd['depth']), axis=None)
    res=np.concatenate((res, pd['width']), axis=None)
    return res

def formatData(pandapath):
    formattedData=[]
    if os.path.isfile(pandapath):
        # data=pd.read_csv(pandapath)

        def converter(instr):
            return np.fromstring(instr[1:-1], sep=' ')

        data = pd.read_csv(pandapath, converters={'EdgeLength': converter, 'BPloc':converter, 'EPloc':converter})
        # print(data.dtypes)
        for index, row in data.iterrows():
            print(index)
            formattedData.append(formatline(row))
    return formattedData



def addNeuronToDatabase(pandapath, dir):
    print(pandapath, dir)
    csv=False
    if os.path.isfile(pandapath):
        data=pd.read_csv(pandapath)
        csv=True

    for path, subdirs, files in os.walk(os.path.join(dir, 'reduced')):
        for file in files:
            print(file)
            if 'gt' in file:
                filename=file[:-17]
                print(filename)
                if csv:
                    if file not in data['name']:
                        gt=ggt.load(os.path.join(os.path.join(dir, 'transformed'), file[:-17]+'_graph_reduced_transformed.gt'))
                        gr = ggt.load(
                            os.path.join(os.path.join(dir, 'reduced'), file[:-17] + '_graph_reduced.gt'))
                        features=extract_features_neurons(file, gt, gr)
                        print(features)
                        print(data.columns, len(features))
                        data = data.append(pd.Series(features[0], index=data.columns), ignore_index=True)#, index=data.columns
                else:
                    csv=True
                    gt = ggt.load(os.path.join(os.path.join(dir, 'transformed'), file[:-17] + '_graph_reduced_transformed.gt'))
                    gr = ggt.load(
                        os.path.join(os.path.join(dir, 'reduced'), file[:-17] + '_graph_reduced.gt'))
                    features = extract_features_neurons(filename, gt, gr)
                    print(features)
                    data = pd.DataFrame(features, columns = ['name' , 'NbBP', 'NbEP' , 'EdgeLength', 'SomaLoc', 'BPloc', 'EPloc', 'NbRegion', 'depth', 'width'])#, index=['a', 'b', 'c' , 'd' , 'e' , 'f']

    data.to_csv(pandapath)
    return data


##### Create Database
data=addNeuronToDatabase('/mnt/data_SSD_2to/test/test4.csv', '/mnt/data_SSD_2to/test/')
##### Formatting data to feed model
formattedData=np.array(formatData('/mnt/data_SSD_2to/test/test3.csv'))
##### transforming data to feed model
## truncated SVD to reduce dimensionnality
svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
transformed_formattedData=svd.fit_transform(formattedData)
## tSNE
## clustering - ward, random forest, svc
from scipy.cluster.hierarchy import ward, dendrogram, linkage
Z=ward(transformed_formattedData)
dn = dendrogram(Z, orientation='right', leaf_font_size=24)
plt.show()