import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
print(os.getcwd())
import multiprocessing as mp
# from ArtificialData.DataGeneration import *
import graph_tool as gt
from numba import njit, prange
import ClearMap.IO.IO as io
# g = gt.collection.data["polblogs"]
# L = gt.laplacian(g)
# ew, ev = scipy.linalg.eig(L.todense())

savedir = '/mnt/data_2to/190506_6R_dataseg/'#''/mnt/data_2to/deatasetGraphReduced/'
name='190506_6R'
# workdir = '/media/sophie.skriabine/TOSHIBA EXT/Sophie/graphVasculature/graph/originalData'
# impath = '/mnt/raid/UnsupSegment/180509_IgG_10-43-24/10-43-24_IgG_UltraII[01 x 08]_C00.ome.tif'
patchShape = (100, 100, 100)
# graphfile = '/mnt/data_2to/190123_7/190123_7data_graph_reduced.gt'#''/mnt/data_2to/190123_7/results/arteries_gpu400_94400_dense_walls_ratio4_0_8_graph_cleaned.gt'#'/mnt/data_2to/190123_7/data_graph_cleaned.gt'
#
# graph = gph.load(graphfile)  # os.path.join(workdir, file))

# artery=graph.vertexProperty('artery')
# large_ids = graph.vertexRadius() > 1800;
# d = graph.degrees() < 5
# final=np.logical_and(large_ids, d)#np.logical_or(artery, large_ids), d)
# print(len(final))
# final = d  # np.logical_and(large_ids, d)
# graph = graph.subGraph(vertexFilter=final)
# print(graph)


groundtruth=io.read('/mnt/data_2to/190506_6R_dataseg/data_binary_final.npy')
patches=io.read('/mnt/data_2to/190506_6R_dataseg/data_stitched.npy')
xmax = patches.shape[0]
xmin = 0
ymax = patches.shape[1]
ymin = 0
zmax = patches.shape[2]
zmin = 0


def createDataset(args):
    i, j, k = args
    print(i, j, k)
    # mins = np.array([i, j, k])  # Minimum coordinates of the cube
    # maxs = np.array([i + patchShape[0], j + patchShape[1], k + patchShape[2]])
    # if (np.sum(patches[i:i+ patchShape[0], j:j + patchShape[1], k:k+ patchShape[2]]) > 100):
    print('saving...')

    io.write(os.path.join(savedir + '/patches' + '/' + name + '/','graph_cleaned' + '_patch_' + str(i) + '_' + str(j) + '_' + str(k) + '.npy'),patches[i:i+ patchShape[0]+10, j:j + patchShape[1]+10, k:k+ patchShape[2]+10])
    print('patches  done, gt...')
    io.write(os.path.join(savedir + 'groundtruth' + '/' + name + '/','graph_cleaned' + '_GT_' + str(i) + '_' + str(j) + '_' + str(k) + '.npy'),groundtruth[i:i+ patchShape[0]+10, j:j + patchShape[1]+10, k:k+ patchShape[2]+10])
    print('gt done')

def from_gt_to_patch(xmax, ymax, zmax):


    cpt = 0
    args = []
    jobs=[]

    for i in prange(3000, xmax - 500 - patchShape[0], patchShape[0]):
        for j in prange(700, ymax - 500 - patchShape[1], patchShape[1]):
            for k in prange(700, zmax - 500 - patchShape[2], patchShape[2]):
                # job=mp.Process(target=extraction,  args=(i, j, k))
                # jobs.append(job)
                # createDataset((i, j, k))
                args.append((i, j, k))
                # if i<int(xmax/2):
                #     extraction((i, j, k))
                    # args.append((i, j, k, sg1))
                # else:
                # extraction((i, j, k))
                # args.append((i, j, k))
                #
                # extractGraphData((i, j, k))
                # cpt += 1
                # print(i, j, k)
                # mins = np.array([i, j, k])  # Minimum coordinates of the cube
                # maxs = np.array([i+ patchShape[0], j+ patchShape[1], k+ patchShape[2]])  # Maximum coordinates of the cube
                # #print(len(subgraph.vertexProperty('x')))
                # subgraph = vs.extractSubGraph(graph, mins, maxs)
                # if (len(subgraph.vertexProperty('x'))>1):
                #     np.save(os.path.join(savedir, file[:-3], file[:-3] + '_patch_' + str(i) + '_' + str(j) + '_' + str(k) + '.npy'),
                #             vs.createArtificialData(subgraph, mins, patchShape).M)
                #     np.save(os.path.join(savedir, file[:-3], file[:-3] + '_GT_' + str(i) + '_' + str(j) + '_' + str(k) + '.npy'),
                #             vs.createGTData(subgraph, mins, patchShape))
                #     # print(cpt, 'patch saved over', (image.shape[0] // patchShape[0]) *
                #     #       (image.shape[1] // patchShape[1]) *
                #     #       (image.shape[2] // patchShape[2]))
    # return jobs
    return args
# for file in os.listdir(workdir):
#     if '.gt' in file:
#         print(file[:-3])
#
#         if not os.path.exists(os.path.join(savedir, file[:-3])):
#             os.makedirs(os.path.join(savedir, file[:-3]))



if __name__ == '__main__':



    parnb=10
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    args=from_gt_to_patch(xmax, ymax, zmax)
    # print(len(args))
    # createDataset(args)
    # for i in tqdm(range(0, len(jobs), parnb)):
    #     for j in jobs[i:i+parnb]:
    #         j.start()
    #
    #     for j in jobs[i:i+parnb]:
    #         j.join()
    with mp.Pool(10) as p:
        p.map(createDataset, args)