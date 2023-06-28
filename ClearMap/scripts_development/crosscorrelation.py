import os

import numpy as np
import pyqtgraph as pg

from ClearMap.IO.elastix_config import ElastixParser
from ClearMap.gui.widgets import LandmarksSelectorDialog, Scatter3D
from ClearMap.Environment import *  # analysis:ignore

from scipy import signal
import matplotlib.pyplot as plt

def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


#test

img1=io.read('/data_2to/p7/a/avox_controla10.tif')
img2=io.read('/data_2to/p7/b/bvox_controlb10.tif')

#correlation (computes shift)
corr = signal.correlate(img1, img2, mode='same')
corr /= np.max(corr)

#SSIM
from skimage.metrics import structural_similarity
import cv2
import numpy as np

(score, diff) = structural_similarity(img1, img2, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))

diff = (diff * 255).astype("uint8")
diff=np.invert(diff)
# p3d.plot([diff,img1, img2])

#test dataset
# work_dir='/data_2to/otof1M'
# controls=[ '1w', '3w', '5w', '6w', '7w']
# mutants=['1k', '2k', '3k', '4k']
# states=[controls, mutants]
# brains=[ '1w', '3w', '5w', '6w', '7w','1k', '2k', '3k', '4k']

#
# work_dir='/data_2to/otof3M/new_vasc'
# controls=['2w', '4w', '5w','6w']
# mutants=[ '3k', '5k', '6k']
# states=[controls, mutants]
# brains=[ '2w', '4w', '5w','6w', '3k', '5k', '6k']


work_dir='/data_SSD_2to/211019_otof_10m'
mutants=['1k', '2k','3k', '6k']#456 not annotated ?
controls=['7w', '9w', '10w', '12w', '13w']
brains=['7w', '9w', '10w', '12w', '13w', '1k', '2k','3k', '6k']

img=io.read(work_dir+'/'+brains[0]+'/'+'vox_control'+brains[0]+'10.tif')

diffs=np.zeros((img.shape[0],img.shape[1],img.shape[2], len(brains)))
corr_matrix=np.zeros((len(brains), len(brains)))

for i, b1 in enumerate(brains):
    for j, b2 in enumerate(brains):
        print(b1, b2)
        try:
            img1=io.read(work_dir+'/'+b1+'/'+'vox_mutant'+b1+'10.tif')
        except:
            img1=io.read(work_dir+'/'+b1+'/'+'vox_control'+b1+'10.tif')
        try:
            img2=io.read(work_dir+'/'+b2+'/'+'vox_mutant'+b2+'10.tif')
        except:
            img2=io.read(work_dir+'/'+b2+'/'+'vox_control'+b2+'10.tif')
        (score, diff) = structural_similarity(img1, img2, full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))
        # diff = (diff * 255).astype("uint8")
        # diff=np.invert(diff)

        # diffs[:,:,:,i]=diff
        corr_matrix[i, j]=score

import seaborn as sns
sns.heatmap(corr_matrix, cmap="crest", xticklabels=brains, yticklabels=brains)

N = len(corr_matrix)

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(corr_matrix, 'ward')

res_order = seriation(Z, N, N + N - 2)
seriated_dist = np.zeros((N, N))
a, b = np.triu_indices(N, k=1)
seriated_dist[a, b] = corr_matrix[[res_order[i] for i in a], [res_order[j] for j in b]]
seriated_dist[b, a] = seriated_dist[a, b]


fig = plt.figure(11)
plt.subplot(1, 2, 1)
sns.heatmap(seriated_dist, cmap="crest")
sns.despine()
# plt.pcolormesh(seriated_dist)
plt.xlim([0,N])
plt.ylim([0,N])


plt.subplot(1, 2, 2)
Z1 =dendrogram(Z, labels=brains, orientation='right')

sns.despine()



### test diff controls vs mutants
diffs=np.zeros((img.shape[0],img.shape[1],img.shape[2], len(controls)*len(controls)))
k=0
for i, b1 in enumerate(controls):
    for j, b2 in enumerate(controls):
        print(b1, b2)
        try:
            img1=io.read(work_dir+'/'+b1+'/'+'vox_mutant'+b1+'10.tif')
        except:
            img1=io.read(work_dir+'/'+b1+'/'+'vox_control'+b1+'10.tif')
        try:
            img2=io.read(work_dir+'/'+b2+'/'+'vox_mutant'+b2+'10.tif')
        except:
            img2=io.read(work_dir+'/'+b2+'/'+'vox_control'+b2+'10.tif')
        (score, diff) = structural_similarity(img1, img2, full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))
        diff = (diff * 255).astype("uint8")
        diff=np.invert(diff)
        #
        diffs[:,:,:,k]=diff
        k=k+1

diffs_avg=np.mean(diffs, axis=3)
# p3d.plot(diffs_avg)

diffs_avg_mvs_c=diffs_avg.copy()

p3d.plot(diffs_avg_mvs_c-diffs_avg)
io.write('/data_SSD_2to/testSSI/diff_otof10M.tif',(diffs_avg_mvs_c-diffs_avg).astype('float32'))

### test Lasso linear models between vox maps

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
# X=io.read('/data_2to/otof3M/new_vasc/vox_control_avg_10.tif').astype('float16')
# Y=io.read('/data_2to/otof3M/new_vasc/vox_mutant_avg_10.tif').astype('float16')
# x=X.flatten().reshape(-1, 1).T
# y=Y.flatten().reshape(-1, 1).T

X=io.read('/data_SSD_2to/211019_otof_10m/vox_control_avg_10.tif').astype('float16')
Y=io.read('/data_SSD_2to/211019_otof_10m/vox_mutant_avg_10.tif').astype('float16')
Z=io.read('/data_2to/DBA2J_OLD-RESTAIN/vox_control_avg_bp_10.tif').astype('float16')


# x=X[:, :, :].flatten().reshape(-1, 1)
# y=Y[:, :, :].flatten().reshape(-1, 1)

## remove background
# x=X[:, :, :].flatten()
# y=Z[:, :, :].flatten()
#
# y_=y[y!=0]
# x_=x[y!=0]
#
# x=x_.reshape(-1, 1)
# y=y_.reshape(-1, 1)


## set isocortical mask
import tifffile
annotation=tifffile.imread('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation.tif')
annotation=annotation.swapaxes(0,1)
annotation=annotation.swapaxes(1,2)
annotation=annotation[:,:, :228]
# np.flip(masked)
order, level= (6,6)
label_leveled = ano.convert_label(annotation, key='id', value='order', level=level)
masked=annotation.copy()
masked[label_leveled != order] = 0;
masked[label_leveled == order] = 1;
masked=masked.swapaxes(0,1)

X=X*masked
Y=Y*masked
x=X[:, :, :].flatten().reshape(-1, 1)
y=Y[:, :, :].flatten().reshape(-1, 1)


clf = linear_model.LinearRegression()
clf.fit(x, y)
y_=clf.predict(x)
dif=y_-y
dif=dif.reshape((320, 528,228))
# plt.figure()
# plt.imshow(dif)
io.write('/data_SSD_2to/testSSI/linreg_otof10M_control_vs_mutants.tif',(dif).astype('float32'))

import random
sample=random.sample(np.arange(len(x)).tolist(), 300000)
plt.figure()
plt.scatter(x[sample], y[sample], alpha=0.1)
plt.plot(x[sample], y_[sample], color='k')





## check interindividuality
import tifffile
annotation=tifffile.imread('/home/sophie.skriabine/Documents/ClearMap_Ressources/annotation_25_HeadLightOrientation.tif')
annotation=annotation.swapaxes(0,1)
annotation=annotation.swapaxes(1,2)
annotation=annotation[:,:, :228]


work_dir='/data_SSD_2to/211019_otof_10m'
mutants=['1k', '2k','3k', '6k']#456 not annotated ?
controls=['7w', '9w', '10w', '12w', '13w']
brains=['7w', '9w', '10w', '12w', '13w', '1k', '2k','3k', '6k']

## if voxelization is required
import ClearMap.Analysis.Graphs.GraphGt as ggt
# atlas='/data_SSD_2to/interindividual_variability/p3/ano_full_P3.tif'
# work_dir='/data_SSD_2to/interindividual_variability/p3'
# controls=['2', '3', '4']
# atlas='/data_SSD_2to/interindividual_variability/p5/P5_full_annotation_sagital.tif'
# atlas_array=io.read(atlas)
# atlas_array=np.swapaxes(np.swapaxes(atlas_array, 1,2), 0,2)
# work_dir='/data_SSD_2to/interindividual_variability/p5'
controls=['3a', '3b', '3c', '3d', '1', '2']
# template_shape=[io.shape(atlas_array)[0], io.shape(atlas_array)[1],io.shape(atlas_array)[2]]
# template_shape_=template_shape
controls=['7R','8R', '6R']
work_dir='/data_SSD_1to/otof6months'
template_shape=(320,528,228)

radius=10

for i,control in enumerate(controls):
    print(control)
    graph=ggt.load(work_dir + '/' + str(control)+'/'+'data_graph_correcteduniverse.gt')
    coordinates = graph.vertex_property('coordinates_atlas') # coordinates_atlas
    # coordinates[:, :2] = coordinates[:, :2] #* 1.625 / 25
    # coordinates[:, 2] = coordinates[:, 2] #* 2 / 25

    v = vox.voxelize(coordinates[:, :3], shape=template_shape, weights=None, radius=(radius, radius, radius), method='sphere');
    io.write(work_dir + '/' + str(control)+ 'vox_bp_'+str(radius)+'.tif', v.astype('float32'))




#correlation

img=io.read(work_dir + '/' + str(controls[0])+ 'vox_bp_'+str(radius)+'.tif')

corr_matrix=np.zeros((len(controls), len(controls)))
diffs=np.zeros((img.shape[0],img.shape[1],img.shape[2], len(controls)*len(controls)))
k=0

for i, b1 in enumerate(controls):
    for j, b2 in enumerate(controls):
        print(b1, b2)
        # try:
        #     x=io.read(work_dir+'/'+b1+'/'+'vox_mutant'+b1+'10.tif')
        # except:
        #     x=io.read(work_dir+'/'+b1+'/'+'vox_control'+b1+'10.tif')
        # try:
        #     y=io.read(work_dir+'/'+b2+'/'+'vox_mutant'+b2+'10.tif')
        # except:
        #     y=io.read(work_dir+'/'+b2+'/'+'vox_control'+b2+'10.tif')
        x=io.read(work_dir + '/' + str(b1)+ 'vox_bp_'+str(radius)+'.tif')
        y=io.read(work_dir + '/' + str(b2)+ 'vox_bp_'+str(radius)+'.tif')

        # order, level= (1,1)
        # label_leveled = ano.convert_label(annotation, key='id', value='order', level=level)
        # masked=annotation.copy()
        # masked[label_leveled != order] = 0;
        # masked[label_leveled == order] = 1;
        # masked=masked.swapaxes(0,1)
        #
        # x=x*masked
        # y=y*masked
        x=x[:, :, :].flatten().reshape(-1, 1)
        y=y[:, :, :].flatten().reshape(-1, 1)

        clf = linear_model.LinearRegression()
        clf.fit(x, y)
        score=clf.score(x, y)
        print(score)
        y_=clf.predict(x)
        dif=y_-y
        dif=dif.reshape(template_shape)
        diffs[:,:,:,k]=dif
        corr_matrix[i, j]=score
        k=k+1


import seaborn as sns
plt.figure()
sns.heatmap(corr_matrix, cmap="crest", xticklabels=controls, yticklabels=controls)

N = len(corr_matrix)

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(corr_matrix, 'ward')

res_order = seriation(Z, N, N + N - 2)
seriated_dist = np.zeros((N, N))
a, b = np.triu_indices(N, k=1)
seriated_dist[a, b] = corr_matrix[[res_order[i] for i in a], [res_order[j] for j in b]]
seriated_dist[b, a] = seriated_dist[a, b]


# fig = plt.figure()
plt.subplot(1, 2, 1)
sns.heatmap(seriated_dist, cmap="crest")
sns.despine()
# plt.pcolormesh(seriated_dist)
plt.xlim([0,N])
plt.ylim([0,N])


plt.subplot(1, 2, 2)
Z1 =dendrogram(Z, labels=controls, orientation='right')

sns.despine()


avg_diff=np.mean(diffs, axis=3)
avg_std=np.std(diffs, axis=3)
p3d.plot([-avg_diff, avg_std])

# work_dir='/media/sophie.skriabine/sophie/otof6M'
io.write(work_dir+'/avg_diff_otof_control_allbatch.tif',avg_diff)
io.write(work_dir+'/avg_std_otof_control_allbatch.tif',avg_std)