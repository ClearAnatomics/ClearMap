
import numpy as np

import ClearMap.Settings as settings
import ClearMap.IO.IO as io

import ClearMap.Alignment.Annotation as ano

import ClearMap.Analysis.Measurements.MeasureExpression as me

import ClearMap.Visualization.Plot3d as p3d


import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt as ggt


import numpy as np
from numpy import arctan2, sqrt
import numexpr as ne
from sklearn import preprocessing
import math


from scipy.optimize import linprog
from scipy.stats import wasserstein_distance, energy_distance
from scipy.cluster.hierarchy import ward, dendrogram, linkage, fcluster


pi=math.pi


def get_Aeq_matrix(imax, jmax):
  Aeq = np.zeros((imax + jmax, imax * jmax), dtype=np.bool)

  # Upper part of the matrix
  for i in range(imax):
    Aeq[i, jmax * i: jmax * i + jmax] = 1

  # Lower part of the matrix
  for j in range(jmax):
    for k in range(0, imax * jmax, jmax):
      Aeq[imax + j, j + k] = 1
  return Aeq

def euclidianDist(vect1, vect2):
    return np.linalg.norm(vect1 - vect2)

def fillDistanceMatix(features):
  imax = features.shape[0] * features.shape[2]
  jmax = features.shape[0] * features.shape[2]
  C = np.zeros((imax, jmax), dtype=np.float32)
  pos=np.array([0,1,2,3,4,5])
  Xr1_pos=np.array([pos, pos]).flatten()
  Xr2_pos=np.array([pos, pos]).flatten()
  for i in range(imax):
    for j in range(jmax):
      C[i, j] = euclidianDist(Xr1_pos[i], Xr2_pos[j])

  return C


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


def get_nb_clique(graph):
  A=graph.adjacency()
  # A=torch.tensor(A).cuda()
  # a=torch.matrix_power(A,3)
  a = A ** 3
  return(np.trace(a)/6)

def get_nb_radial_vessels(edge_color):
  radial=edge_color[:,2]/(edge_color[:,0]+edge_color[:,1]+edge_color[:,2])
  return(np.sum(radial>0.7))


def get_nb_parrallel_vessels(edge_color):
  planar=(edge_color[:,0]+edge_color[:,1])/(edge_color[:,2]+edge_color[:,0]+edge_color[:,1])
  print(planar.shape)
  return(np.sum(planar>0.7))



def get_edges_from_vertex_filter(prev_graph,vertex_filter):
  connectivity=prev_graph.edge_connectivity()
  edges=np.logical_and(vertex_filter[connectivity[:,0]], vertex_filter[connectivity[:,1]])
  return(edges)


def cart2sph(x,y,z, ceval=ne.evaluate):
    """ x, y, z :  ndarray coordinates
        ceval: backend to use:
              - eval :  pure Numpy
              - numexpr.evaluate:  Numexpr """
    r = ceval('sqrt(x**2+y**2+z**2)')#sqrt(x * x + y * y + z * z)
    theta = ceval('arccos(z/r)*180')/pi#acos(z / r) * 180 / pi  # to degrees
    phi = ceval('arctan2(y,x)*180')/pi#*180/3.4142
    # azimuth = ceval('arctan2(y,x)')
    # xy2 = ceval('x**2 + y**2')
    # elevation = ceval('arctan2(z, sqrt(xy2))')
    # r = ceval('sqrt(xy2 + z**2)')
    rmax=np.max(r)
    return phi/180, theta/180, r/rmax#, theta/180, phi/180


def llf(id):
  # labels = np.load('/mnt/data_SSD_2to/190408-44L/labelsAcronymMat.npy')
  labels = np.load('/mnt/data_SSD_2to/190408_38L/labelsAcronymMat.npy')
  return labels[id]

import os
def wait():
    raw_input("Press Enter to continue...")




############################ WASSERSTEIN OTOFERLIN + WHISKERS DEPRIVED #################################################

work_dir ='/data_SSD_2to/whiskers_graphs/new_graphs'  #'/data_SSD_2to/whiskers_graphs/new_graphs'#'/data_SSD_2to/191122Otof'  #
condition = 'isocortex' #'barrel_region'#'Auditory_regions'

brains=['138L', '165L', '163L', '141L']#mutant
# brains=['142L', '158L','162L', '164L']#control

# brains=['2R','3R','5R', '8R']#cpmntrol
# brains=['1R','7R', '6R', '4R']#mutant
# work_dir='/data_SSD_2to/191122Otof'


work_dir='/data_SSD_2to/whiskers_graphs/fluoxetine'
# brains=['1','2','3', '4', '6', '18']
brains=['21','22', '23']


control='controls'
feature='vessels'
nb_reg=48
N = 5
thresh=2
bin = 10
normed = False
regis=[]
features=[]#np.array((len(brains),nb_reg))
first=True

grouped=True
aud=True
vis=True
rsp=True
nose=True
trunk=True

reg_lis=True

for control in brains:

  length_short_path_control=np.load(work_dir + '/' + 'length_short_path_control' + condition +'_' + control +  '.npy')


  # shortest_paths_control=np.load(work_dir + '/' + 'shortest_paths_control' + condition + '_' + control + '.npy')


  vess_rad_control=np.load(work_dir + '/' + feature + 'control_rad_ori' + condition + '_' + control + '.npy')
  bp_dist_2_surface_control=np.load(work_dir + '/' + 'bp_dist_2_surface_control' + condition + '_' + control + '.npy')

  art_ep_dist_2_surface_control=np.load(work_dir + '/' + 'art_ep_dist_2_surface_mutant' + condition +  '_'+control+'.npy')
  ve_ep_dist_2_surface_control=np.load(work_dir + '/' + 've_ep_dist_2_surface_mutant' + condition +  '_'+control+'.npy')
  features_brains = []
  for reg in range(nb_reg):
      # depth = []
      order, level = regions[reg][0]
      n = ano.find(order, key='order')['acronym']
      id=ano.find(order, key='order')['id']
      print(id,(n != 'NoL' or id==182305712))
      if (n != 'NoL' or id==182305712):
        if '6' not in n:
          if reg_lis:
            regis.append(regions[reg])
          try:
            # lens = np.array(length_short_path_control[0][reg])[:, 0]
            # steps = np.array(length_short_path_control[0][reg])[:, 1]
            bp_dist=np.array(bp_dist_2_surface_control[0][reg])##
            art_ep=np.array(art_ep_dist_2_surface_control[1][reg])##
            ve_ep=np.array(ve_ep_dist_2_surface_control[1][reg])##
            ori=np.array(vess_rad_control[0][reg])
            radial_ori=ori[:int(len(ori)/2)]
            radial_depth=ori[int(len(ori)/2):]
            # if first:
            #   first=False
            # H, xedges, yedges = np.histogram2d(radial_depth[radial_depth>thresh], radial_ori[radial_depth>thresh], bins=bin)
            # radial_depth=radial_depth[radial_depth > thresh]
            # radial_ori=radial_ori[radial_depth > thresh]
            radial_depth=radial_depth[radial_ori>0.7]
            # H = H.T  # Let each row list bins with common y range.

            # for c in range(len(shortest_paths_control[0][reg])):
            #     for p in range(len(shortest_paths_control[0][reg][c])):
            #         depth.append(np.array(shortest_paths_control[0][reg][c][p])[1])

            # hist_depth, bins_depth = np.histogram(depth, bins=bin, normed=normed)
            # hist_len, bins_len = np.histogram(lens, bins=bin, normed=normed)
            # hist_ste, bins_ste = np.histogram(steps, bins=bin, normed=normed)
            hist_art_ep, bins_art_ep = np.histogram(art_ep, bins=bin, normed=normed)
            hist_ve_ep, bins_ve_ep = np.histogram(ve_ep, bins=bin, normed=normed)
            hist_bp_dist, bins_bp_dist = np.histogram(bp_dist[bp_dist>thresh], bins=bin, normed=normed)
            print(reg)#, np.sum(np.mean(H, axis=1)))
            hist_ori, bins_ori_dist = np.histogram(radial_depth, bins=bin, normed=normed)
            # hist_ori = np.mean(H, axis=1)/np.sum(np.mean(H, axis=1))
            # else:
            #   H, xedges, yedges = np.histogram2d(radial_depth, radial_ori, bins=bin)
            #   # H = H.T  # Let each row list bins with common y range.
            #
            #   # for c in range(len(shortest_paths_control[0][reg])):
            #   #     for p in range(len(shortest_paths_control[0][reg][c])):
            #   #         depth.append(np.array(shortest_paths_control[0][reg][c][p])[1])
            #
            #   # hist_depth, bins_depth = np.histogram(depth, bins=bin, normed=normed)
            #   hist_len, bins_len = np.histogram(lens, bins=bins_len, normed=normed)
            #   hist_ste, bins_ste = np.histogram(steps, bins=bins_ste, normed=normed)
            #   hist_art_ep, bins_art_ep = np.histogram(art_ep, bins=bins_art_ep, normed=normed)
            #   hist_ve_ep, bins_ve_ep = np.histogram(ve_ep, bins=bins_ve_ep, normed=normed)
            #   hist_bp_dist, bins_bp_dist = np.histogram(bp_dist, bins=bins_bp_dist, normed=normed)
            #   hist_ori = np.mean(H, axis=1)/np.sum(np.mean(H, axis=1))
            # [hist_len, hist_ste,
            features_brains.append([hist_art_ep,hist_ve_ep,hist_bp_dist,hist_ori])#len(shortest_paths_control)
          except:
            print(reg,'no data')
            features_brains.append([np.zeros(10),np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)])

  features.append(features_brains)
  reg_lis=False


features=np.array(features)
features=np.nan_to_num(features)
regis=np.array(regis)

if grouped:

  print(features.shape, regis.shape)
  if aud:
    print('aud')
    inds=[]
    for i, r in enumerate(regis):
      order, level = r[0]
      n = ano.find(order, key='order')['acronym']
      if 'AUD' in n:
        inds.append(i)
    aud_reg=np.mean(features[:, inds, :, :], axis=1)
    aud_reg=np.expand_dims(aud_reg, axis=1)
    features=np.delete(features, inds, axis=1)
    features=np.concatenate((features, aud_reg), axis=1)
    regis=np.delete(regis, inds, axis=0)
    regis=np.concatenate((regis, np.expand_dims(np.array([[127, 7]]), axis=0)))#auditory areas
    aud=False

  print(features.shape, regis.shape)
  if vis:
    print('vis')
    inds = []
    for i, r in enumerate(regis):
      order, level = r[0]
      n = ano.find(order, key='order')['acronym']
      if 'VIS' in n:
        inds.append(i)
    vis_reg = np.mean(features[:, inds, :, :], axis=1)
    vis_reg = np.expand_dims(vis_reg, axis=1)
    features=np.delete(features, inds, axis=1)
    features=np.concatenate((features, vis_reg), axis=1)
    regis=np.delete(regis, inds, axis=0)
    regis=np.concatenate((regis, np.expand_dims(np.array([[163, 7]]), axis=0)))  # auditory areas
    vis = False

  print(features.shape, regis.shape)
  if rsp:
    print('rsp')
    inds = []
    for i, r in enumerate(regis):
      order, level = r[0]
      n = ano.find(order, key='order')['acronym']
      if 'RSP' in n:
        inds.append(i)
    rsp_reg = np.mean(features[:, inds, :, :], axis=1)
    rsp_reg = np.expand_dims(rsp_reg, axis=1)
    features=np.delete(features, inds, axis=1)
    features=np.concatenate((features, rsp_reg), axis=1)
    regis = np.delete(regis, inds, axis=0)
    regis=np.concatenate((regis, np.expand_dims(np.array([[303, 7]]), axis=0)))  # auditory areas
    rsp = False

  print(features.shape, regis.shape)
  if nose:
    print('nose')
    inds = []
    for i, r in enumerate(regis):
      order, level = r[0]
      n = ano.find(order, key='order')['acronym']
      if 'bfd' in n:
        inds.append(i)
      if 'SSp-n' in n:
        inds.append(i)
    nos_reg = np.mean(features[:, inds, :, :], axis=1)
    nos_reg = np.expand_dims(nos_reg, axis=1)
    features=np.delete(features, inds, axis=1)
    features=np.concatenate((features, nos_reg), axis=1)
    regis=np.delete(regis, inds, axis=0)
    regis=np.concatenate((regis, np.expand_dims(np.array([[47, 9]]), axis=0)) ) # auditory areas
    nose = False

  print(features.shape, regis.shape)
  if trunk:
    print('nose')
    inds = []
    for i, r in enumerate(regis):
      order, level = r[0]
      n = ano.find(order, key='order')['acronym']
      if 'SSp-ll' in n:
        inds.append(i)
      if 'SSp-tr' in n:
        inds.append(i)
    nos_reg = np.mean(features[:, inds, :, :], axis=1)
    nos_reg = np.expand_dims(nos_reg, axis=1)
    features = np.delete(features, inds, axis=1)
    features = np.concatenate((features, nos_reg), axis=1)
    regis = np.delete(regis, inds, axis=0)
    regis = np.concatenate((regis, np.expand_dims(np.array([[89, 9]]), axis=0)))  # auditory areas
    trunk = False




features_avg=np.mean(features, axis=0)
features_avg_c=features_avg
features_c=features



def kolmogorov_distance(v1, v2):
  return np.linalg.norm(v1-v2)


feat_list=[1,2,3,4, 5, 6,7]

# feat_list=[2, 3]
from sklearn.preprocessing import normalize
D=np.zeros((features_avg.shape[0],features_avg.shape[0]))
# w=[]
for r1 in range(features_avg.shape[0]):
  for r2 in range(features_avg.shape[0]):
    w=[]
    for f in range(features_avg.shape[1]):
      if f in feat_list:
        # mr1=np.sum(features_avg[r1, f])
        # mr2 = np.sum(features_avg[r2, f])
        n1=np.squeeze(normalize(features_avg[r1, f].reshape(-1, 1), norm='l2', axis=0), axis=1)
        n2=np.squeeze(normalize(features_avg[r2, f].reshape(-1, 1), norm='l2', axis=0), axis=1)
        w.append(kolmogorov_distance(n1,n2))#kolmogorov_distance#wasserstein_distance
        # print(f)
        # w.append(energy_distance(features_avg[r1, f],features_avg[r2, f]))

    D[r1,r2]=np.linalg.norm(np.array(w))


from scipy.stats import wasserstein_distance, energy_distance
from scipy.cluster.hierarchy import ward, dendrogram, linkage, fcluster, centroid
Z=centroid(D)#ward#centroid
N = len(D)
res_order = seriation(Z, N, N + N - 2)
seriated_dist = np.zeros((N, N))
a, b = np.triu_indices(N, k=1)
seriated_dist[a, b] = D[[res_order[i] for i in a], [res_order[j] for j in b]]
seriated_dist[b, a] = seriated_dist[a, b]


# np.save('/mnt/data_SSD_2to/190408-44L/labelsMat.npy', labels)
# Z_c=Z


fig = plt.figure()
plt.subplot(1, 2, 1)
plt.pcolormesh(seriated_dist)
plt.xlim([0,N])
plt.ylim([0,N])
# plt.yticks(y_pos, layer_nb)

def lqbeling(id):
  order, level=regis[id][0]
  return ano.find(order, key='order')['acronym']

plt.subplot(1, 2, 2)
# fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, orientation='right',leaf_label_func=lqbeling, leaf_font_size=12, color_threshold=1.5)
plt.title('mutant BP ORI')
# plt.show()
#
# for id in range(len(regis)):
#   order, level = regis[id][0]
#   n = ano.find(order, key='order')['acronym']
#   print(n, id)
#   if n == 'SSp-bfd':
#     print('bfd', id)
#   if n == 'SSs':
#     print('sss', id)
#
f=5
for b in range(features.shape[0]):
  plt.figure()
  plt.plot(features[b][1][f])
  plt.plot(features[b][2][f])
  plt.plot(features[b][4][f])
  plt.plot(features[b][9][f])
  plt.legend(['mop', 'mos','bfd', 'sss'])
  plt.title(brains[b])
#
#

f=3
plt.figure()
plt.plot(np.squeeze(normalize(features_avg_c[1, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
plt.plot(np.squeeze(normalize(features_avg_c[2, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
plt.plot(np.squeeze(normalize(features_avg_c[4, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
plt.plot(np.squeeze(normalize(features_avg_c[9, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
plt.plot(np.squeeze(normalize(features_avg_c[33, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
plt.plot(np.squeeze(normalize(features_avg_c[26, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
plt.plot(np.squeeze(normalize(features_avg_c[7, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
plt.plot(np.squeeze(normalize(features_avg_c[15, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
plt.plot(np.squeeze(normalize(features_avg_c[19, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
plt.plot(np.squeeze(normalize(features_avg_c[22, f].reshape(-1, 1), norm='l2', axis=0), axis=1))
# plt.plot(features_avg[1, 5])
# plt.plot(features_avg[2, 5])
# plt.plot(features_avg[4][5])
# plt.plot(features_avg[9][5])
plt.legend(['mop', 'mos','bfd', 'sss', 'rspd', 'orbl', 'sspul','audv', 'visp', 'acad'])
plt.title('control ori')

############################ PCA ##############################

## fit control
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


pca = PCA(n_components=2)#PCA(n_components=2)
X=features_avg_c[:, 2:, :]
X=X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
pca.fit(X)
X_transform=pca.transform(X)

col_dict={1:'r', 2:'c', 3:'g', 4:'m', 5:'y'}
clusters = fcluster(Z, 1.5, criterion='distance')

plt.figure()
sns.set_style(style='white')


for i, x in enumerate(X_transform):
  plt.scatter(x[1], x[0], c=col_dict[clusters[i]])
  label=ano.find(regis[i][0][0], key='order')['acronym']
  plt.text(x[1], x[0],label)  # horizontal alignment can be left, right or center
plt.title('cluster control')
sns.despine()

## transform deprived

Y=features_avg[:, 2:, :]
Y=Y.reshape((Y.shape[0], Y.shape[1]*Y.shape[2]))
Y_transform=pca.transform(Y)

# col_dict={1:'r', 2:'g', 3:'c'}
clusters = fcluster(Z, 1.5, criterion='distance')
plt.figure()
sns.set_style(style='white')
for i, x in enumerate(Y_transform):
  plt.scatter(x[1], x[0], c=col_dict[clusters[i]])
  label=ano.find(regis[i][0][0], key='order')['acronym']
  plt.text(x[1], x[0],label) # horizontal alignment can be left, right or center

# plt.scatter(Y_transform[:, 0], Y_transform[:, 1], c=clusters)
plt.title('cluster mutant')
sns.despine()
##########  plot comparison

ROI=['MOs', 'MOp', 'Alp', 'VIS', 'AUDp', 'AUDd', 'SSp-ul',  'SSp-ll', 'AUD', 'SSp-n', 'SSp-bfd', 'SSs','ILA','RSP' ]

plt.figure()
sns.set_style(style='white')
# col_dict={1:'skyblue', 2:'salmon', 3:'limegreen'}
col_dict={1:'r', 2:'c', 3:'g', 4:'m', 5:'y'}
clusters = fcluster(Z_c, 1.5, criterion='distance')
for i, x in enumerate(X_transform):
  label = ano.find(regis[i][0][0], key='order')['acronym']
  if label in ROI:
    plt.scatter(x[1], x[0], c=col_dict[clusters[i]], alpha=1)
    plt.text(x[1], x[0],label,bbox=dict(facecolor=col_dict[clusters[i]], alpha=0.5))  # horizontal alignment can be left, right or center
  else:
    plt.scatter(x[1], x[0], c=col_dict[clusters[i]], alpha=0.3)
    # plt.text(x[0], x[1], label)  # horizontal alignment can be left, right or center

# col_dict={1:'firebrick', 2:'forestgreen', 3:'royalblue'}
# clusters = fcluster(Z, 3, criterion='distance')
for i, x in enumerate(Y_transform):
  label = ano.find(regis[i][0][0], key='order')['acronym']
  if label in ROI:
    plt.scatter(x[1], x[0], c=col_dict[clusters[i]], alpha=1)
    plt.text(x[1], x[0], label, bbox=dict(facecolor=col_dict[clusters[i]], alpha=0.5))  # horizontal alignment can be left, right or center
  else:
    plt.scatter(x[1], x[0], c=col_dict[clusters[i]], alpha=0.3)
    # plt.text(x[0], x[1], label)  # horizontal alignment can be left, right or center

plt.title('cluster comparison')
sns.despine()

################ plot individual values for ROI
colors_c = ['royalblue', 'darkblue', 'forestgreen', 'lightseagreen']
colors_m = ['darkred', 'indianred', 'darkgoldenrod', 'darkorange']
ROI=['AUD']#['SSp-n']#['SSp-tr', 'MOs', 'MOp', 'SSp-n', 'SSp-bfd','SSs']
norm=False
f=7
bins=bins_bp_dist
feat=['PROP_ORI_PLAN', 'PROP_ORI_RAD']#['ART EP', 'VE EP', 'BP', 'ORI']#'SP len', 'SP step',
import pandas as pd
for  r in range(features.shape[1]):
  l = ano.find(regis[r][0][0], key='order')['acronym']
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
    if norm:
      Cpd_c = pd.DataFrame(normalize(features_c[:, r, f], norm='l2', axis=1)).melt()
      Cpd_m = pd.DataFrame(normalize(features[1:, r, f], norm='l2', axis=1)).melt()
    else:
      Cpd_c = pd.DataFrame(features_c[:, r, f]).melt()
      Cpd_m = pd.DataFrame(features[1:, r, f]).melt()
    sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_c, color=colors_c[0], linewidth=2.5)
    sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_m, color=colors_m[3], linewidth=2.5)

    plt.title(l + ' ' + feat[f],size='x-large')
    plt.legend(['control', 'deprived'])
    plt.yticks(size='x-large')
    plt.ylabel(feat[f], size='x-large')
    plt.xlabel('cortical thickness', size='x-large')
    plt.xticks(np.arange(0, 10), np.arange(0, np.max(bins), np.max(bins) / 10).astype(int), size='x-large')
    sns.despine()
    plt.tight_layout()

    ## art/vein ep features
    # if norm:
    #   Cpd_art_c = pd.DataFrame(normalize(np.array(features_c[:, r, 0]), norm='l2', axis=1)).melt()
    #   Cpd_art_m = pd.DataFrame(normalize(np.array(features[:, r, 0]), norm='l2', axis=1)).melt()
    #   Cpd_ve_c = pd.DataFrame(normalize(np.array(features_c[:, r, 1]), norm='l2', axis=1)).melt()
    #   Cpd_ve_m = pd.DataFrame(normalize(np.array(features[:, r, 1]), norm='l2', axis=1)).melt()
    # else:
    #   Cpd_art_c = pd.DataFrame(np.array(features_c[:, r, 0])).melt()
    #   Cpd_art_m = pd.DataFrame(np.array(features[:, r, 0])).melt()
    #   Cpd_ve_c = pd.DataFrame(np.array(features_c[:, r, 1])).melt()
    #   Cpd_ve_m = pd.DataFrame(np.array(features[:, r, 1])).melt()

    sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_art_c, color=colors_m[0])
    sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_art_m, color=colors_m[3])
    sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_ve_c, color=colors_c[0])
    sns.lineplot(x="variable", y="value", err_style='bars', data=Cpd_ve_m, color=colors_c[3])
    plt.title(l+' '+'ART/VEIN EP')
    plt.legend(['art_c', 'art_v', 've_c', 've_m'])
    sns.despine()



############################ WASSERSTEIN OLD ###########################################################################





##% load grphs
gri=ggt.load('/mnt/data_SSD_2to/190408_38L/data_graph_reduced.gt')
grti=ggt.load('/mnt/data_SSD_2to/190408_38L/data_graph_reduced_transformed.gt')




##% get vessels orientation

# gs = gri.sub_slice((slice(0,4000), slice(0,7000), slice(2000,2150)));
# gs = grti.sub_slice((slice(0,300), slice(0,700), slice(60,70)));
label = grti.vertex_annotation();
label_leveled = ano.convert_label(label, key='order', value='order', level=4)#2
vertex_filter = label_leveled == 1183#1093;
vertex_filter = gri.expand_vertex_filter(vertex_filter, steps=2)
# abel_leveled_ci = ano.convert_label(label, key='order', value='order', level=3)
# medial_bundle=abel_leveled_ci == 1230
# label_leveled_ci = ano.convert_label(label, key='order', value='order', level=5)
# internal_capsul=label_leveled_ci == 1193
# cccapsule=label_leveled_ci == 1186
# gs = gri.sub_graph(vertex_filter=vertex_filter)#np.logical_and(vertex_filter, np.logical_not(np.logical_or(internal_capsul, np.logical_or(medial_bundle, cccapsule)))))

# edge_colors_s=edge_colors[vertex_filter]
# gs = gs.sub_slice((slice(0,4000), slice(0,7000), slice(2000,2150)));

# gs=grti
gs = gri.sub_graph(vertex_filter=vertex_filter)
radii=gri.edge_geometry_property('radii')
gri.set_edge_geometry(name='radii', values=radii*4)

x=gs.vertex_coordinates()[:, 0]
y=gs.vertex_coordinates()[:, 1]
z=gs.vertex_coordinates()[:, 2]


x_g=gri.vertex_coordinates()[:, 0]
y_g=gri.vertex_coordinates()[:, 1]
z_g=gri.vertex_coordinates()[:, 2]

center=np.array([np.mean(x_g),np.mean(y_g), np.mean(z_g)])
x=x-np.mean(x_g)
y=y-np.mean(y_g)
z=z-np.mean(z_g)

spherical_coord=np.array(cart2sph(x,y,z, ceval=ne.evaluate)).T
connectivity = gs.edge_connectivity()

x_s=spherical_coord[:, 0]
y_s=spherical_coord[:, 1]
z_s=spherical_coord[:, 2]

spherical_ori=np.array([x_s[connectivity[:, 1]]-x_s[connectivity[:, 0]], y_s[connectivity[:, 1]]-y_s[connectivity[:, 0]], z_s[connectivity[:, 1]]-z_s[connectivity[:, 0]]]).T
# orientations=preprocessing.normalize(orientations, norm='l2')

# edge_colors = (vertex_colors[connectivity[:, 0]] + vertex_colors[connectivity[:, 1]]) / 2.0;


# spherical_ori=np.array(cart2sph(orientations[:, 0],orientations[:, 1],orientations[:, 2], ceval=ne.evaluate)).T
spherical_ori=preprocessing.normalize(spherical_ori, norm='l2')
spherical_ori=np.abs(spherical_ori)
edge_colors=np.insert(spherical_ori, 3, 1.0, axis=1)


print('plotting...')


p = p3d.plot_graph_mesh(gs, edge_colors=edge_colors, n_tube_points=3);


#%% get atlas region list

print('extracting sub region...')
label = grti.vertex_annotation();

layer_order={}
layer_order.update({'1':0})
layer_order.update({'2/3':1})
layer_order.update({'4':2})
layer_order.update({'5':3})
layer_order.update({'6a':4})
layer_order.update({'6b':5})

#get sur layer region id
reg_list={}
for l in np.unique(label):
  if 'ayer' in ano.find_name(l, key='order'):
    print(ano.find_name(l, key='order'))
    print(ano.find_parent(l, key='order'))
    if ano.find_order(ano.find_parent(l, key='order')) not in reg_list.keys():
          reg_list.update( {ano.find_order(ano.find_parent(l, key='order'),key='id'): []} )
          reg_list[ano.find_order(ano.find_parent(l, key='order'))].append(l)
    else:
          reg_list[ano.find_order(ano.find_parent(l, key='order'))].append(l)
          print(reg_list[ano.find_order(ano.find_parent(l, key='order'))])

  else:
    print(ano.find_name(l, key='order'))
    if ano.find_order(l, key='order') not in reg_list.keys():
          reg_list.update( {l: []} )
          reg_list[l].append(l)
    else:
          reg_list[l].append(l)
          print(reg_list[l])

atlas_list={}
atlas=io.read('/home/sophie.skriabine/Projects/clearvessel-custom/ClearMap/Resources/Atlas/annotation_25_full.nrrd')
for l in np.unique(atlas):
  val=np.sum(atlas==l)
  # print(l, np.sum(atlas==l))
  atlas_list.update({ano.find_order(l,key='atlas_id'): val } )
  if ano.find_order(l,key='atlas_id')==930:
    print('cochlear nuclei')


# print(reg_list)
for parent in reg_list.keys():
  print(ano.find_name(parent, key='order'))


# import cloudpickle as pickle
import pickle
with open('/mnt/data_SSD_2to/181002_4/reg_list.p', 'wb') as fp:
    pickle.dump(reg_list, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('/mnt/data_SSD_2to/181002_4/atlas_volume_list.p', 'wb') as fp:
  pickle.dump(atlas_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
#%% save and load dictionnary

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

with open('/mnt/data_SSD_2to/181002_4/reg_list.p', 'rb') as fp:
  reg_list = pickle.load(fp)

with open('/mnt/data_SSD_2to/181002_4/atlas_volume_list.p', 'rb') as fp:
  atlas_list = pickle.load(fp)


##% easserstein


n=0
shap=int(math.sqrt(len(reg_list.keys())))+1
print(shap)
plt.figure(1)

#%% extract radius histograms per sur layer region
for e in reg_list.keys():
  n=n+1
  print(str(n)+'/'+str(len(reg_list.keys())))
  label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
  vertex_filter = label_leveled == e#54;
  print(ano.find_name(e, key='order'))
  region_name=ano.find_name(e, key='order')
  vertex_filter = grti.expand_vertex_filter(vertex_filter, steps=2)

  grt = grti.sub_graph(vertex_filter=vertex_filter)
  gr = gri.sub_graph(vertex_filter=vertex_filter)

  print('get arteries...')
  # radii=gr.graph_property('edge_geometry_radii_erosion')
  #
  # gr.set_edge_geometry(name='radii', values=radii/600)
  radii=gr.graph_property('edge_geometry_radii')
  print(radii.size)
  radiimax=np.mean(radii)
  print(radiimax)

  coordinates, edge_indices = gr.edge_geometry(as_list=False, return_indices=True);
  expression = me.measure_expression(binary_arteries, coordinates, radiimax, method='max');
  edge_artery_label = np.array([np.sum(expression[ei[0]:ei[1]]) >= (ei[1]-ei[0])/2.0 for ei in edge_indices], dtype='int');
  grt.add_edge_property('artery', edge_artery_label);

  edge_artery_label = gr.edge_property('artery')


  print('filtering out low degree vertices...')
  # gs = grt.sub_graph(vertex_filter=vertex_filter)
  gs=gr
  degrees=gs.vertex_degrees()
  vertex_filter=degrees>1
  gs = gs.sub_graph(vertex_filter=vertex_filter)
  print(gs)

  edge_artery_label = gs.edge_property('artery')



  edge_filter=edge_artery_label#np.logical_or(edge_vein_label,edge_artery_label)
  gsrt = gs.sub_graph(edge_filter=edge_filter)

  radii_art = gsrt.graph_property('edge_geometry_radii')
  print(radii_art.size)

  num_bins = 50

  plt.subplot(shap, shap, int(n))
  plt.title(str(region_name)+' all radius histogram')
  # plt.title('all radius histogram')
  no, bins, patches = plt.hist(radii,range = (0, 20), bins=num_bins, facecolor='blue', alpha=0.5)
  # plt.subplot(2, 1, 2)
  # plt.title('arteries radius histogram')
  plt.twinx()
  no, bins, patches = plt.hist(radii_art,range = (0, 20), bins=num_bins, facecolor='red', alpha=0.25)
  # plt.tight_layout()
  plt.show()
# plt.tight_layout()
plt.show()


labels=[]

n=0

branch_pt=np.zeros((len(reg_list.keys()), 6))
art_branch_pt=np.zeros((len(reg_list.keys()), 6))
art_end_pt=np.zeros((len(reg_list.keys()), 6))
clique=np.zeros((len(reg_list.keys()), 6))
plan_vess=np.zeros((len(reg_list.keys()), 6))
rad_vess=np.zeros((len(reg_list.keys()), 6))
volumes=np.zeros((len(reg_list.keys()), 6))

branch_pt_n=np.zeros((len(reg_list.keys()), 6))
art_branch_pt_n=np.zeros((len(reg_list.keys()), 6))
art_end_pt_n=np.zeros((len(reg_list.keys()), 6))
clique_n=np.zeros((len(reg_list.keys()), 6))
plan_vess_n=np.zeros((len(reg_list.keys()), 6))
rad_vess_n=np.zeros((len(reg_list.keys()), 6))



#%% end point info from layered region/branching point
for e in reg_list.keys():

  n = n + 1
  print(str(n) + '/' + str(len(reg_list.keys())))
  label_leveled = ano.convert_label(label, key='order', value='order', level=ano.find_level(e, key='order'))
  vertex_filter = label_leveled == e  # 54;
  print(ano.find_name(e, key='order'))
  region_name = ano.find_name(e, key='order')
  titlr_size=len(region_name)
  # print(titlr_size)
  vertex_filter = grti.expand_vertex_filter(vertex_filter, steps=2)

  grt = grti.sub_graph(vertex_filter=vertex_filter)
  gr = gri.sub_graph(vertex_filter=vertex_filter)

  # grt = grti.sub_graph(vertex_filter=np.logical_and(vertex_filter,high_list_end_point))
  # gr = gri.sub_graph(vertex_filter=np.logical_and(vertex_filter,high_list_end_point))

  print('get arteries...')
  radii = gr.graph_property('edge_geometry_radii')
  radiimax = np.mean(radii)
  # print(radiimax)

  # coordinates, edge_indices = gr.edge_geometry(as_list=False, return_indices=True);
  # expression = me.measure_expression(binary_arteries, coordinates, radiimax, method='max');
  # edge_artery_label = np.array([np.sum(expression[ei[0]:ei[1]]) >= (ei[1] - ei[0]) / 2.0 for ei in edge_indices],
  #                          dtype='int');
  # grt.add_edge_property('artery', edge_artery_label);
  edge_artery_label= grt.edge_property('artery')
  edge_filter = edge_artery_label#edge_artery_label  # np.logical_or(edge_vein_label,edge_artery_label)
  edge_colors_e=edge_colors[get_edges_from_vertex_filter(grti,vertex_filter)]
  print(np.sum(edge_filter))

  #%% quick hack to get rif of non arteries vertices:
  art_grt = grt.sub_graph(edge_filter=edge_filter)
  connectivity = art_grt.edge_connectivity();
  vertices=art_grt.vertex_degrees()<0
  vertices[connectivity[:, 0]]=1
  vertices[connectivity[:, 1]] = 1
  art_grt = art_grt.sub_graph(vertex_filter=vertices)
  vess_grt=grt


  # art_grt=grt

  high_list_end_point = art_grt.vertex_degrees() == 1
  art_branching_point = art_grt.vertex_degrees() > 2
  branching_point=vess_grt.vertex_degrees()>2
  end_point = vess_grt.vertex_degrees() == 1
  nb_ep_tot=np.sum(vertex_filter)
  # layers_ep=[]
  # layers_ep = []

  layers_clique_ep = []
  layers_art_ep_ep = []
  layers_art_bp_ep = []
  layers_rad_vess_ep = []
  layers_plan_vess_ep = []
  layers_bp_ep=[]

  layers_names = []
  layer_nb=[]
  layer_vol=[]
  #get sub region
  for se in reg_list[e]:
    # print(se)
    print(ano.find_name(se, key='order'))
    label_se = vess_grt.vertex_annotation();
    label_leveled_se = ano.convert_label(label_se, key='order', value='order', level=ano.find_level(se, key='order'))
    vertex_filter = label_leveled_se == se

    art_label_se = art_grt.vertex_annotation();
    art_label_leveled_se = ano.convert_label(art_label_se, key='order', value='order', level=ano.find_level(se, key='order'))
    art_vertex_filter = art_label_leveled_se == se

    # print(np.sum(np.logical_and(vertex_filter,high_list_end_point)))
    # print(np.unique(np.logical_and(vertex_filter,high_list_end_point)))
    # print((np.logical_and(vertex_filter,high_list_end_point)).size)
    # if np.sum(np.logical_and(art_vertex_filter,high_list_end_point))>0:
    # try:
    #   print(np.logical_and(vertex_filter,branching_point))
    art_vertex_filter_se_endpoint=np.logical_and(art_vertex_filter,high_list_end_point)
    art_vertex_filter_se_branchpoint = np.logical_and(art_vertex_filter, art_branching_point)
    vertex_filter_se_branchpoint=np.logical_and(vertex_filter,branching_point)
    edge_color_filter=np.logical_and(vertex_filter,branching_point)

    layer = vess_grt.sub_graph(vertex_filter=vertex_filter_se_branchpoint)
    art_ep_layer= art_grt.sub_graph(vertex_filter=art_vertex_filter_se_endpoint)
    art_bp_layer = art_grt.sub_graph(vertex_filter=art_vertex_filter_se_branchpoint)

    print(layer)
    # end_point_se = layer.vertex()

    edge_color_layer = edge_colors_e[get_edges_from_vertex_filter(vess_grt, edge_color_filter)]
    # print(edge_color_layer.shape)

    nb_clique_lay = get_nb_clique(layer)
    nb_art_ep_lay = np.sum(art_vertex_filter_se_endpoint>0)
    nb_art_bp_lay = np.sum(art_vertex_filter_se_branchpoint > 0)
    np_bp_lay=np.sum(vertex_filter_se_branchpoint > 0)
    nb_rad_vess_lay = get_nb_radial_vessels(edge_color_layer)
    nb_plan_vess_lay = get_nb_parrallel_vessels(edge_color_layer)



    layers_clique_ep.append(nb_clique_lay)#*100/nb_ep_tot)
    layers_art_ep_ep.append(nb_art_ep_lay)
    layers_art_bp_ep.append(nb_art_bp_lay)
    layers_bp_ep.append(np_bp_lay)
    layers_rad_vess_ep.append(nb_rad_vess_lay)
    layers_plan_vess_ep.append(nb_plan_vess_lay)

    # print(nb_ep_lay)#*100/nb_ep_tot)
    layer_name=ano.find_name(se, key='order')
    layers_names.append(layer_name)
    layer_vol.append(atlas_list[se])
    layer_nb.append(layer_name[titlr_size+8:])
    # vertex_colors = ano.convert_label(layer.vertex_annotation(), key='order', value='rgba');

    # except(IndexError):
    #   print('index error')
    #   print(IndexError)
    #   break
    # else:
    #   break
  if len(layer_nb)>3:
    for i in range(len(layer_nb)):
      if layer_nb[i] in layer_order.keys():
        branch_pt[n-1][layer_order[layer_nb[i]]]=layers_bp_ep[i]
        art_branch_pt[n - 1][layer_order[layer_nb[i]]] = layers_art_bp_ep[i]
        art_end_pt[n - 1][layer_order[layer_nb[i]]] = layers_art_ep_ep[i]
        clique[n - 1][layer_order[layer_nb[i]]] = layers_clique_ep[i]
        rad_vess[n - 1][layer_order[layer_nb[i]]] = layers_rad_vess_ep[i]
        plan_vess[n - 1][layer_order[layer_nb[i]]] = layers_plan_vess_ep[i]
        volumes[n - 1][layer_order[layer_nb[i]]] = layer_vol[i]
        if ano.find_acronym(e, key='order') not in labels:
          labels.append(ano.find_acronym(e, key='order'))


  num_bins = len(layers_names)
  # print(len(range(num_bins)))
  # print(len(layers_ep))
  # layers_ep = np.array(layers_ep).astype(float)
  # layer_vol=np.array(layer_vol).astype(float)
  # print(layer_vol)
  # print(layers_ep)

  # layer_vol=np.sum(layer_vol)/layer_vol
  # layers_ep=np.multiply(layers_ep,layer_vol)
  # layers_ep=layers_ep*100/np.sum(layers_ep)
  # print(layer_vol)
  # print(layers_ep)

  fig = plt.figure(6)
  plt.subplot(shap, shap, int(n))
  plt.title(str(region_name)+' vess branch pt histogram')
  # plt.title('all radius histogram')
  layers_bp_ep = np.array(layers_bp_ep).astype(float)
  layer_vol=np.array(layer_vol).astype(float)
  # layer_vol = np.sum(layer_vol) / layer_vol
  layers_bp_ep = np.multiply(layers_bp_ep, layer_vol)
  layers_bp_ep = layers_bp_ep * 100 / np.sum(layers_bp_ep)
  if len(layer_nb) > 3:
    for i in range(len(layer_nb)):
      if layer_nb[i] in layer_order.keys():
        branch_pt_n[n-1, layer_order[layer_nb[i]]] = layers_bp_ep[i]
  plt.bar(range(num_bins),layers_bp_ep)
  y_pos = np.arange(len(layer_nb))
  plt.xticks(y_pos, layer_nb)
  plt.show()


  fig = plt.figure(1)
  plt.subplot(shap, shap, int(n))
  plt.title(str(region_name)+' art branch pt histogram')
  # plt.title('all radius histogram')
  layers_art_bp_ep = np.array(layers_art_bp_ep).astype(float)
  layer_vol = np.array(layer_vol).astype(float)
  # layer_vol = np.sum(layer_vol) / layer_vol
  layers_art_bp_ep = np.multiply(layers_art_bp_ep, layer_vol)
  layers_art_bp_ep = layers_art_bp_ep * 100 / np.sum(layers_art_bp_ep)
  if len(layer_nb) > 3:
    for i in range(len(layer_nb)):
      if layer_nb[i] in layer_order.keys():
        art_branch_pt_n[n-1, layer_order[layer_nb[i]]] = layers_art_bp_ep[i]
  plt.bar(range(num_bins),layers_art_bp_ep)
  y_pos = np.arange(len(layer_nb))
  plt.xticks(y_pos, layer_nb)
  plt.show()

  fig = plt.figure(2)
  plt.subplot(shap, shap, int(n))
  plt.title(str(region_name)+' art end pt histogram')
  # plt.title('all radius histogram')
  layers_art_ep_ep = np.array(layers_art_ep_ep).astype(float)
  layer_vol = np.array(layer_vol).astype(float)
  # layer_vol = np.sum(layer_vol) / layer_vol
  layers_art_ep_ep = np.multiply(layers_art_ep_ep, layer_vol)
  layers_art_ep_ep = layers_art_ep_ep * 100 / np.sum(layers_art_ep_ep)
  if len(layer_nb) > 3:
    for i in range(len(layer_nb)):
      if layer_nb[i] in layer_order.keys():
        art_end_pt_n[n-1, layer_order[layer_nb[i]]] = layers_art_ep_ep[i]
  plt.bar(range(num_bins),layers_art_ep_ep)
  y_pos = np.arange(len(layer_nb))
  plt.xticks(y_pos, layer_nb)
  plt.show()

  fig = plt.figure(3)
  plt.subplot(shap, shap, int(n))
  plt.title(str(region_name)+' clique histogram')
  # plt.title('all radius histogram')
  layers_clique_ep = np.array(layers_clique_ep).astype(float)
  layer_vol = np.array(layer_vol).astype(float)
  # layer_vol = np.sum(layer_vol) / layer_vol
  layers_clique_ep = np.multiply(layers_clique_ep, layer_vol)
  layers_clique_ep = layers_clique_ep * 100 / np.sum(layers_clique_ep)
  if len(layer_nb) > 3:
    for i in range(len(layer_nb)):
      if layer_nb[i] in layer_order.keys():
        clique_n[n-1, layer_order[layer_nb[i]]] = layers_clique_ep[i]
  plt.bar(range(num_bins),layers_clique_ep)
  y_pos = np.arange(len(layer_nb))
  plt.xticks(y_pos, layer_nb)
  plt.show()

  fig = plt.figure(4)
  plt.subplot(shap, shap, int(n))
  plt.title(str(region_name)+' rad vess histogram')
  # plt.title('all radius histogram')
  layers_rad_vess_ep = np.array(layers_rad_vess_ep).astype(float)
  # layer_vol = np.array(layer_vol).astype(float)
  layer_vol = np.sum(layer_vol) / layer_vol
  layers_rad_vess_ep = np.multiply(layers_rad_vess_ep, layer_vol)
  layers_rad_vess_ep = layers_rad_vess_ep * 100 / np.sum(layers_rad_vess_ep)
  if len(layer_nb) > 3:
    for i in range(len(layer_nb)):
      if layer_nb[i] in layer_order.keys():
        rad_vess_n[n - 1, layer_order[layer_nb[i]]] = layers_rad_vess_ep[i]
  plt.bar(range(num_bins),layers_rad_vess_ep)
  y_pos = np.arange(len(layer_nb))
  plt.xticks(y_pos, layer_nb)
  plt.show()

  fig = plt.figure(5)
  plt.subplot(shap, shap, int(n))
  plt.title(str(region_name)+' plan vess histogram')
  # plt.title('all radius histogram')
  layers_plan_vess_ep = np.array(layers_plan_vess_ep).astype(float)
  layer_vol = np.array(layer_vol).astype(float)
  # layer_vol = np.sum(layer_vol) / layer_vol
  layers_plan_vess_ep = np.multiply(layers_plan_vess_ep, layer_vol)
  layers_plan_vess_ep = layers_plan_vess_ep * 100 / np.sum(layers_plan_vess_ep)
  if len(layer_nb) > 3:
    for i in range(len(layer_nb)):
      if layer_nb[i] in layer_order.keys():
        plan_vess_n[n - 1 ,layer_order[layer_nb[i]]] = layers_plan_vess_ep[i]
  plt.bar(range(num_bins),layers_plan_vess_ep)
  y_pos = np.arange(len(layer_nb))
  plt.xticks(y_pos, layer_nb)
  plt.show()


plt.show()


# labels=np.unique(np.array(labels))
# print(labels.size)
np.save('/mnt/data_SSD_2to/181002_4/labelsAcronymMat.npy', labels)

# m_nonzero_branch_pt = branch_pt[[i for i, x in enumerate(branch_pt) if x.any()]]
#%%check how arteries are filtered plz
np.save('/mnt/data_SSD_2to/181002_4/VessbranchHist.npy', branch_pt)
np.save('/mnt/data_SSD_2to/181002_4/ArtBranchHist.npy', art_branch_pt)
np.save('/mnt/data_SSD_2to/181002_4/ArtEndPointHist.npy', art_end_pt)
np.save('/mnt/data_SSD_2to/181002_4/cliqueHist.npy', clique)
np.save('/mnt/data_SSD_2to/181002_4/radialHist.npy', rad_vess)
np.save('/mnt/data_SSD_2to/181002_4/planarHist.npy', plan_vess)
np.save('/mnt/data_SSD_2to/181002_4/volumes.npy', volumes)

np.save('/mnt/data_SSD_2to/181002_4/VessbranchHist_normalized.npy', branch_pt_n)
np.save('/mnt/data_SSD_2to/181002_4/ArtBranchHist_normalized.npy', art_branch_pt_n)
np.save('/mnt/data_SSD_2to/181002_4/ArtEndPointHist_normalized.npy', art_end_pt_n)
np.save('/mnt/data_SSD_2to/181002_4/cliqueHist_normalized.npy', clique_n)
np.save('/mnt/data_SSD_2to/181002_4/radialHist_normalized.npy', rad_vess_n)
np.save('/mnt/data_SSD_2to/181002_4/planarHist_normalized.npy', plan_vess_n)




#%% preprocessing duplicates 2/3 to 4 if no 4 and 6a to 6b if no 6b + removes 0 rows(i.e deleted region)
# branch=np.load('/mnt/data_SSD_2to/190408-44L/branchPtMatHist.npy')
folders=['7r_test', '190428_8R', '190428_6R']#['190408_38L', '190408-44L', '190408_39L']#['7r_test']#['181002_4']#
n=0
BPmat=[]
AEPmat=[]
ABPmat=[]
CLmat=[]
Rmat=[]
PLmat=[]
Vmat=[]

for folder in folders:

  branch=np.load('/mnt/data_SSD_2to/'+folder+'/VessbranchHist_normalized.npy')
  for i in range(branch.shape[0]):
    if (branch[i, 2]==0):
      branch[i, 2]=branch[i, 1]
    if (branch[i, 5]==0):
      branch[i, 5]=branch[i, 4]
  branch = branch[[i for i, x in enumerate(branch) if x.any()]]
  # end=np.load('/mnt/data_SSD_2to/'+folder+'/endPtMatHist.npy')
  # for i in range(end.shape[0]):
  #   if (end[i, 2]==0):
  #     end[i, 2]=end[i, 1]
  #   if (end[i, 5]==0):
  #     end[i, 5]=end[i, 4]
  # end = end[[i for i, x in enumerate(end) if x.any()]]
  # branch_a=np.load('/mnt/data_SSD_2to/190408-44L/arteriesbranchPTMatHist.npy')
  branch_a=np.load('/mnt/data_SSD_2to/'+folder+'/ArtBranchHist_normalized.npy')
  for i in range(branch_a.shape[0]):
    if (branch_a[i, 2]==0):
      branch_a[i, 2]=branch_a[i, 1]
    if (branch_a[i, 5]==0):
      branch_a[i, 5]=branch_a[i, 4]
  branch_a = branch_a[[i for i, x in enumerate(branch_a) if x.any()]]
  # end_a=np.load('/mnt/data_SSD_2to/190408-44L/arteriesendPTMatHist.npy')
  end_a=np.load('/mnt/data_SSD_2to/'+folder+'/ArtEndPointHist_normalized.npy')
  for i in range(end_a.shape[0]):
    if (end_a[i, 2]==0):
      end_a[i, 2]=end_a[i, 1]
    if (end_a[i, 5]==0):
      end_a[i, 5]=end_a[i, 4]
  end_a = end_a[[i for i, x in enumerate(end_a) if x.any()]]
  cliques=np.load('/mnt/data_SSD_2to/'+folder+'/cliqueHist_normalized.npy')
  for i in range(cliques.shape[0]):
    if (cliques[i, 2]==0):
      cliques[i, 2]=cliques[i, 1]
    if (cliques[i, 5]==0):
      cliques[i, 5]=cliques[i, 4]
  cliques = cliques[[i for i, x in enumerate(cliques) if x.any()]]
  # planar=np.load('/mnt/data_SSD_2to/190408-44L/planarVessNBMatHist.npy')
  planar=np.load('/mnt/data_SSD_2to/'+folder+'/radialHist_normalized.npy')
  for i in range(planar.shape[0]):
    if (planar[i, 2]==0):
      planar[i, 2]=planar[i, 1]
    if (planar[i, 5]==0):
      planar[i, 5]=planar[i, 4]
  planar = planar[[i for i, x in enumerate(planar) if x.any()]]
  # radial=np.load('/mnt/data_SSD_2to/190408-44L/radialVessNBMatHist.npy')
  radial=np.load('/mnt/data_SSD_2to/'+folder+'/planarHist_normalized.npy')
  for i in range(radial.shape[0]):
    if (radial[i, 2]==0):
      radial[i, 2]=radial[i, 1]
    if (radial[i, 5]==0):
      radial[i, 5]=radial[i, 4]
  radial = radial[[i for i, x in enumerate(radial) if x.any()]]
  volumes = np.load('/mnt/data_SSD_2to/' + folder + '/volumes.npy')
  for i in range(radial.shape[0]):
    if (volumes[i, 2] == 0):
      volumes[i, 2] = volumes[i, 1]
    if (volumes[i, 5] == 0):
      volumes[i, 5] = volumes[i, 4]
  volumes = volumes[[i for i, x in enumerate(volumes) if x.any()]]
  if n==0:
    BP=branch
    AEP=end_a
    ABP=branch_a
    CL=cliques
    R=radial
    PL=planar
    V=volumes

  else:
    BP=BP+branch
    AEP = AEP+end_a
    ABP = ABP+branch_a
    CL = CL+cliques
    R = R+radial
    PL = PL+planar
    V=V+volumes

  BPmat.append(branch)
  AEPmat.append(end_a)
  ABPmat.append(branch_a)
  CLmat.append(cliques)
  Rmat.append(radial)
  PLmat.append(planar)
  Vmat.append(volumes)

  print(n)
  n=n+1

fig = plt.figure(20)
t=[' Branch Pt histogram', ' AE Pt histogram', ' AB Pt histogram', ' clique histogram', ' Planar histogram',  ' Radial histogram']
regions=['MOp', 'SSp-bfd', 'RSPd']
features=[BP, AEP, ABP, PL, CL, R]
featuresmat=np.array([BPmat, AEPmat, ABPmat, CLmat, Rmat, PLmat])
labels=np.array(labels)
n=0
g=0
colors={}
colors.update({'MOp':'lightcoral'})#'[1, 0.1, 0.1, 0.5]})
colors.update({'SSp-bfd':'mediumseagreen'})#[0.1, 1, 0.2, 0.5]})
colors.update({'RSPd':'cornflowerblue'})#[0.1, 0.1, 1, 0.5]})
for r in regions:
  # n=n+1
  ind= np.where(labels==r)[0][0]
  print(r, ind)
  col=colors[r]
  for f in range(featuresmat.shape[0]):
    # feat=features[n, :, ind, :]
    print(t[f])
    g = g + 1
    print(g - 1 + n)
    featmat=featuresmat[f, :, ind, :]
    if r =='MOp':
      print(featmat)
      featmat=np.delete(np.array(featmat),2,1)
      print(featmat)
    print(featmat)
    num_bins = featmat.shape[1]
    plt.subplot(3, 6, int(g))
    plt.title(str(labels[ind]) + t[f])
    pt1 = []
    pt2 = []
    for i in range(featmat.shape[1]):
      # plt.title('all radius histogram')
      # mi.append(np.min(featmat[:, i]))
      # ma.append(np.max(featmat[:, i]))
      for j in range(featmat.shape[0]):
        pt1.append(featmat[j, i])
        pt2.append(i)
      c=col
      # c[3]=1
    # yerr=np.array([(ma[k]-mi[k])/2 for k in range(featmat.shape[1])]),
    plt.bar(np.arange(num_bins), np.mean(featmat, axis=0), color=col, alpha=0.8)

    y_pos = np.arange(num_bins)
    if r == 'MOp':
      plt.xticks(y_pos, ['1', '2/3', '5', '6a', '6b'])
    else:
      plt.xticks(y_pos, ['1', '2/3', '4', '5', '6a', '6b'])
    plt.scatter(pt2, pt1, color='k',  zorder=2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    axes = plt.gca()
    axes.set_ylim([0, 60])
    plt.show()




features=[BP, ABP,PL,R]#,AEP, CL]#BP, ABP, AEP, PL,R, CL]##BP]# PL,R]#[BP, ABP]#AEP, CL,, PL,R
features=np.array(features)[:, :, :]
# features=np.delete(np.array(features),2,2)
print(features.shape)
D =  np.zeros((35,35))

cpt = 0
for r1 in range(features.shape[1]):
  for r2 in range(features.shape[1]):
    D[r1, r2]=wasserstein_distance(features[:,r1, :].flatten(), features[:,r2, :].flatten())
#
#     imax = features.shape[0]*features.shape[2]
#     jmax = features.shape[0]*features.shape[2]
#
#     C = fillDistanceMatix(features)
#     C = C.reshape(-1)
#     bounds = (0, None)
#     b_eq = np.concatenate((features[:,r1, :].flatten(), features[:,r2, :].flatten()), axis=0)
#
#     A_eq = get_Aeq_matrix(imax, jmax)
#
#     try:
#         res = linprog(C, method='interior-point',
#                       options={"maxiter": 100})#, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
#
#         if res.success:
#             # print(r1, r2, "Distance:", res.fun)
#             D[r1, r2] = res.fun
#             cpt += 1
#         else:
#             print("linprog failed:", res.message)
#     except np.linalg.LinAlgError:
#         print("SVD did not converge")
#         D[r1, r2] = -1
#         cpt += 1
#
print('saving matrix...')
# np.save('/mnt/data_SSD_2to/190408-44L/WassersteinDisttest', D)
np.save('/mnt/data_SSD_2to/190408_38L/WassersteinDisttest', D)
N = len(D)
Z=ward(D)
# np.save('/mnt/data_SSD_2to/190408-44L/WassersteinDisttestWardClustering', Z)
np.save('/mnt/data_SSD_2to/190408_38L/WassersteinDisttestWardClustering', Z)





res_order = seriation(Z, N, N + N - 2)
seriated_dist = np.zeros((N, N))
a, b = np.triu_indices(N, k=1)
seriated_dist[a, b] = D[[res_order[i] for i in a], [res_order[j] for j in b]]
seriated_dist[b, a] = seriated_dist[a, b]


# np.save('/mnt/data_SSD_2to/190408-44L/labelsMat.npy', labels)



fig = plt.figure(11)
plt.subplot(1, 2, 1)
plt.pcolormesh(seriated_dist)
plt.xlim([0,N])
plt.ylim([0,N])
# plt.yticks(y_pos, layer_nb)


plt.subplot(1, 2, 2)
# fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z, orientation='right',leaf_label_func=llf, leaf_font_size=24, color_threshold=0.04)
plt.show()