import os.path
import sys
sys.path.append('/home/sophie.skriabine/Projects/Keops/keops')
import pykeops
pykeops.clean_pykeops()
pykeops.test_numpy_bindings()
import ClearMap.Alignment.Annotation as ano
import ClearMap.IO.IO as io
import ClearMap.Visualization.Plot3d as p3d
import graph_tool.inference as gti
import os
print('TEST')
import math
import matplotlib.pyplot as plt
import ClearMap.Analysis.Graphs.GraphGt_new as grpn
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
import graph_tool.spectral as gts
import scipy
import pykeops.torch
from pykeops.numpy import LazyTensor
from pykeops.numpy.utils import IsGpuAvailable
from time import time
from scipy.sparse.linalg import eigsh
import pickle
from scipy.optimize import curve_fit
from scipy.stats import poisson

use_cuda = IsGpuAvailable()
cuda=torch.cuda.is_available()
print(cuda)

control='163L'
work_dir='/data_SSD_2to/whiskers_graphs/new_graphs/'
graph=grpn.load('/data_SSD_2to/whiskers_graphs/new_graphs/' + control + '/data_graph_correcteduniverse.gt')
degrees = graph.vertex_degrees()
vf = np.logical_and(degrees > 1, degrees <= 4)
graph = graph.sub_graph(vertex_filter=vf)
with open(work_dir + '/' + control + '/sampledict' + control + '.pkl', 'rb') as fp:
    sampledict = pickle.load(fp)

f = np.asarray(sampledict['flow'][0])
v = np.asarray(sampledict['v'][0])
graph.add_edge_property('flow', f)
graph.add_edge_property('veloc', v)


region=[(6,6)]#subependymal zone, #caudoputamen, #cortex

def expfunc(x, a, b, c):
    return a * np.exp(-b * x) + c

def gaussianfunc(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)

def fit_function(x, amp, lamb):
    '''poisson function, parameter lamb is the fit parameter'''
    return amp*poisson.pmf(x, lamb)

for reg in region:
    vertex_filter = np.zeros(graph.n_vertices)
    order, level = reg

    if ano.find(order, key='order')['name']!='subependymal zone':
        # ano.set_annotation_file(annotatin_real)
        # coordinates = graph_reduced.vertex_property('coordinates_atlas')
        # label = annotation(coordinates)
        # print('take real annotation')
        label = graph.vertex_annotation();
    else:
        label=labels



    label_leveled = ano.convert_label(label, key='order', value='order', level=level)
    vf =np.logical_or(vertex_filter, label_leveled == order)
    if ano.find(order, key='order')['name']=='subependymal zone':
        # remove plexus choroid
        coordinates_y=graph.vertex_property('coordinates_atlas')[:,1]
        vf=np.logical_and(vf, coordinates_y>270)
    print(ano.find(order, key='order')['name'],np.sum(vf.astype(int)))

    g=graph.sub_graph(vertex_filter=vf)

    # vertex_colors = ano.convert_label(g.vertex_annotation(), key='order', value='rgba');
    # p = p3d.plot_graph_mesh(g,vertex_colors=vertex_colors, n_tube_points=3);

    # g.save('/data_SSD_2to/whiskers_graphs/new_graphs/163L/data_graph_subependymal_annotated_Andromachi.gt')




    f=g.edge_property('flow')

    ## get vessels length distribution
    plt.figure()
    bins=np.arange(0,150, 2)
    hist, b=np.histogram(g.edge_geometry_lengths(), bins=bins)
    xdata=(b[1:] + b[:-1])/2
    popt, pcov = curve_fit(gaussianfunc, xdata, hist, bounds=([45000, 10, 400],[55000, 15, 1000]))
    plt.plot(xdata, hist,label='data')
    plt.plot(xdata, gaussianfunc(xdata, *popt), 'r-',
             label='fit: amp=%5.3f, cent=%5.3f, wid=%5.3f' % tuple(popt))
    popt, pcov = curve_fit(expfunc, xdata, hist, bounds=([80000, 0.020, 0],[100000, 0.2, 1000]))
    plt.plot(xdata, expfunc(xdata, *popt), 'g-',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.title('edge length distribution in '+ano.find(order, key='order')['name'])
    plt.legend()

    plt.figure()
    bins = np.arange(0, 20, 0.5)
    hist, b=np.histogram(g.edge_radii(), bins=bins)
    xdata = (b[1:] + b[:-1]) / 2
    popt, pcov = curve_fit(gaussianfunc, xdata, hist, bounds=([300000, 0, 0],[350000, 5, 5]))
    plt.plot(xdata, hist,label='data')
    plt.plot(xdata, gaussianfunc(xdata, *popt), 'r-',
             label='fit: amp=%5.3f, cent=%5.3f, wid=%5.3f' % tuple(popt))
    plt.title('edge radii distribution in ' + ano.find(order, key='order')['name'])
    plt.legend()

    plt.figure()
    bins = np.arange(0, 150, 2)
    hist, b=np.histogram(f, bins=bins)
    xdata = (b[1:] + b[:-1]) / 2
    popt, pcov = curve_fit(expfunc, xdata, hist)
    plt.plot(xdata, hist,label='data')
    plt.plot(xdata, expfunc(xdata, *popt), 'r-',
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.title('edge flow distribution in ' + ano.find(order, key='order')['name'])
    plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')











g=graph.base
H = gts.hashimoto(g)
Klo=scipy.sparse.linalg.aslinearoperator(H)
# Hlt=LazyTensor(Klo)
start = time()

from scipy.sparse.linalg import *

eigenvalues = eigs(Klo, k=3, which='LM', return_eigenvectors=False)  # Largest 10 eigenvalues/vectors
eigenvalues2 = eigs(Klo, k=3, which='SM', return_eigenvectors=False)  # Largest 10 eigenvalues/vectors
ew = np.concatenate((eigenvalues, eigenvalues2))
print("Largest eigenvalues of the normalized graph Laplacian, computed in {:.3f}s ".format(time() - start) \
      + "on a cloud of {:,} points in dimension {}:".format(graph.n_vertices, graph.n_edges))

print("Largest eigenvalues:", eigenvalues)


plt.figure(figsize=(8, 4))
plt.scatter(np.real(ew), np.imag(ew), c=np.sqrt(abs(ew)), linewidths=0, alpha=0.6)
plt.xlabel(r"$\operatorname{Re}(\lambda)$")
plt.ylabel(r"$\operatorname{Im}(\lambda)$")
plt.tight_layout()


N=100
N_display = 10000 if cuda else N
indices_display = np.random.randint(0, N, N_display)

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), subplot_kw=dict(projection='3d'))
x_ = x[indices_display, :]
ax.scatter3D(x_[:, 0], x_[:, 1], x_[:, 2],
             c=t[indices_display],
             cmap=plt.cm.Spectral)
ax.set_title("{:,} out of {:,} points in our source point cloud".format(N_display, N))
plt.show()
#
# N = 2 * g.num_edges()
# ew1 = scipy.sparse.linalg.eigs(Hlt, k=N//2, which="LR", return_eigenvectors=False)
# ew2 = scipy.sparse.linalg.eigs(Hlt, k=N-N//2, which="SR", return_eigenvectors=False)
# ew = np.concatenate((ew1, ew2))
