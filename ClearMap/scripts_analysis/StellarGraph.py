import networkx as nx
from stellargraph.mapper import GraphWaveGenerator
from stellargraph import StellarGraph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from ClearMap.Gt2Nx import *
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt
import ClearMap.Alignment.Annotation as ano
from time import time
controls=['2R','3R','5R', '8R']
mutants=['1R','7R', '6R', '4R']
work_dir='/data_SSD_2to/191122Otof'
states=[controls, mutants]
# states=[controls]

condition='mouth'#'Auditory_regions'
control='5R'

if condition == 'Auditory_regions':
    regions = [[(142, 8), (149, 8), (128, 8), (156, 8)]]
    sub_region = True
elif condition == 'barrel_region':
    regions = [[(54, 9), (47, 9)]]  # , (75, 9)]  # barrels
elif condition== 'mouth':
    regions = [[(75,9)]]
N=0
Embeds=[]
Embeds_indices=[]
for state in states:
    for control in state:



        graph = ggt.load(
            work_dir + '/' + control + '/' + 'data_graph_correcteduniverse.gt')  # data_graph_corrected_Isocortex.gt')#/data_graph_correcteduniverse.gt')#
        degrees = graph.vertex_degrees()
        vf = np.logical_and(degrees > 1, degrees <= 4)
        graph = graph.sub_graph(vertex_filter=vf)

        for region_list in regions:

            vertex_filter = np.zeros(graph.n_vertices)
            for i, rl in enumerate(region_list):
                order, level = region_list[i]
                print(level, order, ano.find(order, key='order')['name'])
                label = graph.vertex_annotation();
                label_leveled = ano.convert_label(label, key='order', value='order', level=level)
                vertex_filter[label_leveled == order] = 1;
            # gss4_t = graph.sub_graph(vertex_filter=vertex_filter)

        gss4 = graph.copy()
        gss4 = gss4.sub_graph(vertex_filter=vertex_filter)
        gss4 = gss4.largest_component()


        gnx = gt2nx(gss4, weight=None)

        # m1 = 9
        # m2 = len(gnx.nodes)#11
        # classes = [0,] * len(gnx.nodes)
        # # number of nodes with a non-zero class (the path, plus the nodes it connects to on either end)
        # nonzero = m2 + 2
        # # count up to the halfway point (rounded up)
        # first = range(1, (nonzero + 1) // 2 + 1)
        # # and down for the rest
        # second = reversed(range(1, nonzero - len(first) + 1))
        # classes[m1 - 1 : (m1 + m2) + 1] = list(first) + list(second)
        m2 = len(gnx.nodes)#11
        classes = np.array([0,] * len(gnx.nodes))
        artery=from_e_prop2_vprop(gss4, 'artery').astype(bool)
        vein=from_e_prop2_vprop(gss4, 'vein').astype(bool)
        d2s=gss4.vertex_property('distance_to_surface')
        classes[vein]=1
        classes[artery]=2
        classes=classes.tolist()

        counts, binEdges=np.histogram(d2s,bins=10)
        assignement = np.digitize(d2s, binEdges)
        classes=assignement.tolist()

        G = StellarGraph.from_networkx(gnx)
        sample_points = np.linspace(0, 100, 50).astype(np.float32)
        degree = 20
        scales = np.arange(3,15).tolist()

        generator = GraphWaveGenerator(G, scales=scales, degree=degree)

        embeddings_dataset = generator.flow(
            node_ids=G.nodes(), sample_points=sample_points, batch_size=1, repeat=False
        )

        embeddings = [x.numpy() for x in embeddings_dataset]

        np.save(work_dir + '/' + control + '/' +condition+ '_graphWaveEmbedding.npy', embeddings)

        if N==0:
            Embeds=np.array(embeddings)
            Embeds_indices=np.zeros(np.array(embeddings).shape[:2])
        else:
            Embeds=np.concatenate((Embeds, np.array(embeddings)), axis=0)
            Embeds_indices = np.concatenate((Embeds_indices,N*np.ones(np.array(embeddings).shape[:2])), axis=0)

        N=N+1
np.save(work_dir + '/' +condition+ '_graphWaveEmbedding.npy', Embeds)
np.save(work_dir + '/' +condition+ '_graphWaveEmbedding_Indices.npy', Embeds_indices)




### load
Embeds=np.load(work_dir + '/' +condition+ '_graphWaveEmbedding.npy')
Embeds_indices=np.load(work_dir + '/' +condition+ '_graphWaveEmbedding_Indices.npy')


## PCA transformation
# plt.figure()
# trans_emb = PCA(n_components=2).fit_transform(np.vstack(embeddings))
# plt.scatter(
#     trans_emb[:, 0], trans_emb[:, 1], c=classes, cmap="jet", alpha=0.2,
# )
# plt.colorbar()
# plt.show()


## TSNE transformation
# plt.figure()
trans_emb = TSNE(n_components=2,perplexity=40, learning_rate=200).fit_transform(np.vstack(Embeds))

#
# plt.scatter(
#     trans_emb[:, 0], trans_emb[:, 1], c=classes, cmap="jet", alpha=0.2,
# )
# plt.colorbar()
# plt.show()
np.save(work_dir + '/' +condition+ '_graphWaveTransformed_embeddingd.npy', trans_emb)

## TSNE + clustering
trans_emb=np.load(work_dir + '/' +condition+ '_graphWaveTransformed_embeddingd.npy')


e2rm=random.sample(range(trans_emb.shape[0]), k=int(trans_emb.shape[0]/2))

trans_emb=trans_emb[e2rm, :]

linkage='ward'
clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)

t0 = time()
clustering.fit(trans_emb)
print("%s :\t%.2fs" % (linkage, time() - t0))
labels=clustering.labels_

np.save(work_dir  + '/' +condition+ '_graphWaveEmbedding_labels.npy', labels)
# np.save(work_dir + '/' + control + '/' +condition+ '_graphWaveEmbedding_labels.npy', labels)


plt.figure()
plt.scatter(
    trans_emb[:, 0], trans_emb[:, 1], c=labels, cmap="jet", alpha=0.2,
)
plt.colorbar()
plt.show()


Embeds_indices=Embeds_indices[e2rm, :]

C=[]
M=[]
all_sample=[]
for i in range(len(controls)):
    brain_labels=labels[np.asarray(Embeds_indices==i).nonzero()[0]]
    counts, binEdges = np.histogram(brain_labels, bins=10)
    assignement = np.digitize(brain_labels, binEdges)
    C.append(counts/np.sum(counts))
    all_sample.append(counts/np.sum(counts))
Cnp=np.stack(C, axis=0)

for j in range(len(mutants)):
    brain_labels=labels[np.asarray(Embeds_indices==i+j+1).nonzero()[0]]
    counts, binEdges = np.histogram(brain_labels, bins=10)
    assignement = np.digitize(brain_labels, binEdges)
    M.append(counts/np.sum(counts))
    all_sample.append(counts/np.sum(counts))
Mnp = np.stack(M, axis=0)

plt.figure()
PCA_brains=PCA(n_components=2)
trans_controls = PCA_brains.fit(all_sample)
trans_controls = PCA_brains.transform(Cnp)
plt.scatter(
    trans_controls[:, 0], trans_controls[:, 1], c='cadetblue',
)

trans_mutants = PCA_brains.transform(Mnp)
plt.scatter(
    trans_mutants[:, 0], trans_mutants[:, 1], c='indianred',
)
plt.legend(['controls', 'otof'])
plt.title('t-SNE+PCA decomposition of otoferlin mice using graph wavelet', size='x-large')
plt.show()


vertex_colors=getColorMap_from_vertex_prop(labels)
p = p3d.plot_graph_mesh(gss4, vertex_colors=vertex_colors, n_tube_points=3)