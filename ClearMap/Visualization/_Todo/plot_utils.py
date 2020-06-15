# -*- coding: utf-8 -*-
"""
Fast plotting of large matrices via pyqtgraph

This is module provides scripts to plot large matrices or point clouds in a fast way
using pyqtgraph and opengl.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import numpy as np

import matplotlib.pyplot as plt;

import pyqtgraph as pg
import pyqtgraph.exporters as pgexp

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

#import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib as mpl
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'viridis'
mpl.rcParams['figure.autolayout'] = True;
mpl.rcParams['figure.figsize'] = (21,14);


def colormap_lut(color = 'viridis', ncolors = None):
   # build lookup table
  if color == 'r': 
    pos = np.array([0.0, 1.0])
    color = np.array([[0,0,0,255], [255,0,0,255]], dtype=np.ubyte)
    ncolors = 512;
  elif color =='g':
    pos = np.array([0.0, 1.0])
    color = np.array([[0,0,0,255], [0,255,0,255]], dtype=np.ubyte)
    ncolors = 512;
  elif color =='b':
    pos = np.array([0.0, 1.0])
    color = np.array([[0,0,0,255], [0,0,255,255]], dtype=np.ubyte)
    ncolors = 512;
  else:
    #pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    #color = np.array([[0,0,255,255], [0,255,255,255],  [0,255,0,255], [255,255,0,255], [255,0,0,255]], dtype=np.ubyte)
    #color = np.array([[0,0,128,255], [0,255,255,255],  [0,255,0,255], [255,255,0,255], [128,0,0,255]], dtype=np.ubyte)
    cmap = cm.get_cmap(color);
    if ncolors is None:
      ncolors = cmap.N;
    pos = np.linspace(0.0, 1.0, ncolors);
    color = cmap(pos, bytes = True);
  
  cmap = pg.ColorMap(pos, color)
  return cmap.getLookupTable(0.0, 1.0, ncolors);


class plot_array():
  """Plots a 2d matrix"""
  def __init__(self, data, title = None, color = 'viridis', ncolors = None):
    #self.app = pg.QtGui.QApplication([])
    #self.win = pg.GraphicsLayoutWidget()
    #self.win.resize(1200, 800)
    lut = colormap_lut(color, ncolors);
 
    self.img = pg.ImageItem()
    self.img.setLookupTable(lut)
    self.img.setLevels([0,1])        
    
    #self.plot = self.win.addPlot()
    self.plot = pg.plot(title = title);
    self.plot.addItem(self.img)
    
    #self.timer = QtCore.QTimer()
    #self.timer.timeout.connect(self.check_for_new_data_and_replot)
    #self.timer.start(100)
    self.img.setImage(data.T)
    #self.win.show()
    
    

import pyqtgraph.opengl as gl

class plot3d():
  """Plots a 3d cloud of points"""
  def __init__(self, points, title = None):
    pg.mkQApp();
    self.w = gl.GLViewWidget()
    self.w.opts['distance'] = 20
    self.w.show()
    self.w.setWindowTitle(title)

    self.g = gl.GLGridItem()
    self.w.addItem(self.g)
    self.sp = gl.GLScatterPlotItem(pos=points, color=(1,1,1,1), pxMode= True)
    self.w.addItem(self.sp);
    #self.plot.addItem(self.w);
      
    #
    ### create three grids, add each to the view
    #xgrid = gl.GLGridItem()
    #ygrid = gl.GLGridItem()
    #zgrid = gl.GLGridItem()
    #view.addItem(xgrid)
    #view.addItem(ygrid)
    #view.addItem(zgrid)
    #
    ### rotate x and y grids to face the correct direction
    #xgrid.rotate(90, 0, 1, 0)
    #ygrid.rotate(90, 1, 0, 0)
    #
    ### scale each grid differently
    #xgrid.scale(0.2, 0.1, 0.1)
    #ygrid.scale(0.2, 0.1, 0.1)
    #zgrid.scale(0.1, 0.2, 0.1)


def savefig(plot, filename, width = None):
  """Export plot to file"""
  exporter = pgexp.ImageExporter(plot.img);
  if width is not None:
    exporter.parameters()['width'] = width   # (note this also affects height parameter)
  
  # save to file
  exporter.export(filename)



def plot_trace(xy, ids = None, depth = 0, colormap = 'rainbow', line_color = 'k', line_width = 1, point_size = 5, title = None):
  """Plot trajectories with positions color coded according to discrete ids"""
  
  #if ids is not None:
  uids = np.unique(ids);
  
  cmap = cm.get_cmap(colormap);
  n = len(uids);
  colors = cmap(range(n), bytes = True);
  
  #lines
  if line_width is not None:
    #plt.plot(xy[:,0], xy[:,1], color = lines);    
    plot = pg.plot(xy[:,0], xy[:,1], pen = pg.mkPen(color = line_color, width = line_width))    
  else:
    plot = pg.plot(title = title);
    
  if ids is None:
    sp = pg.ScatterPlotItem(pos = xy, size=point_size, pen=pg.mkPen(colors[0])); #, pxMode=True);
  else:
    sp = pg.ScatterPlotItem(size=point_size); #, pxMode=True);
    spots = [];
    for j,i in enumerate(uids):
      idx = ids == i;
      spots.append({'pos': xy[idx,:].T, 'data': 1, 'brush':pg.mkBrush(colors[j])}); #, 'size': point_size});
    sp.addPoints(spots)
  
  plot.addItem(sp);
  
  return plot;

  
#  legs = [];
#  for k,i in enumerate(uids):
#    ii = np.where(ids == i)[0];
#    if depth > 0:
#      ii = [ii-d for d in range(depth)];
#      ii = np.unique(np.concatenate(ii));
#    
#    plt.plot(data[ii, 0], data[ii, 1], '.', color = color[k]);
#
#    legs.append(mpatches.Patch(color=color[k], label= str(i)));
#  
#  plt.legend(handles=legs);


def plot_image_array(arrays, names = None, order = None, cmap = 'viridis', invert_y = False, nplots = None, vmin = None, vmax = None):
  if nplots is None:
    nplots = len(arrays);
    
  if vmax is all:
    vmax = max([a.max() for a in arrays]);    
  if vmin is all:
    vmin = min([a.min() for a in arrays]);        
    
  for i,d in enumerate(arrays):
    plt.subplot(nplots,1,i+1);
    if order is not None:
      dpl = d[order];
    else:
      dpl = d;
    if invert_y:
      #e = (0, dpl.shape[1], 0, dpl.shape[0]);
      o = 'lower';
    else:
      #e = None;
      o = 'upper';
    plt.imshow(dpl, interpolation = 'none', aspect = 'auto', cmap = cmap, origin = o, vmax = vmax)
    plt.colorbar(fraction = 0.01, pad = 0.01)
    if names is not None:
      plt.title(names[i]);
  plt.tight_layout();
  plt.show();




from scipy.stats import gaussian_kde

def plot_distributions(data, cmap = plt.cm.Spectral_r, percentiles =  [5, 25, 50, 75, 95], percentiles_colors = ['gray', 'gray', 'red', 'gray', 'gray']):
  """Plots the data point color as local density"""
  npoints, ntimes = data.shape;
  
  for t in range(ntimes):
    cols = gaussian_kde(data[:,t])(data[:,t]);
    idx = cols.argsort();
    
    plt.scatter(np.ones(npoints)*t, data[idx,t], c=cols[idx], s=30, edgecolor='face', cmap = cmap)
    
  pct = np.percentile(data, percentiles, axis = 0); 
  for i in range(len(percentiles)):
    #plt.plot(iqr[s0][i,:],  c = plt.cm.Spectral(i/5.0), linewidth = 3);
    plt.plot(pct[i,:], c = percentiles_colors[i], linewidth = 2);



import scipy.cluster.hierarchy as sch

def plot_hierarchical_cluster(distances, clustering = None, linkage_kwargs = {'method' : 'single'}, dendogram_kwargs = { 'link_color_func' : lambda x: 'k'},
                              padding = [0.01, 0.01], dendogram_size = [0.2, 0.2], cmap = plt.cm.viridis, label = None, colorbar_width = 0.075, stride_line_kwargs = {'c' :'k'}):
  """Plot a correlation matrix with hierachical clustering"""
  
  ds = dendogram_size;
  if ds[0] is None:
    ds[0] = 0;
  if ds[1] is None:
    ds[1] = 0;
  
  if clustering is None:
    Y = sch.linkage(distances, **linkage_kwargs)
  else:
    Y = sch.linkage(clustering, **linkage_kwargs);
    
  cbw = colorbar_width;
  if cbw is None:
    cbw = 0;
  else:
    cbw = cbw + padding[0];

  fig = plt.gcf();
  fig.patch.set_facecolor('white')

  # Plot distance matrix.
  ax = fig.add_axes([padding[0] + ds[0], padding[1], 1 - cbw - 2 * padding[1] - ds[0], 1 - 2 * padding[1] - ds[1]])
  
  if dendogram_size[0] is not None:
    spec = [padding[0], padding[1], ds[0], 1 - 2 * padding[1] - ds[1]];
    ax1 = fig.add_axes(spec); # , sharey = ax)
    z1 = sch.dendrogram(Y, orientation = 'left', **dendogram_kwargs)
    ax1.axis('off')
    #ax1.set_xticks([])
    #ax1.set_yticks([])
    #ax1.get_yaxis().set_visible(False)
    #ax1.get_xaxis().set_visible(False)
  
  if dendogram_size[1] is not None:
    spec = [padding[0] + ds[0], 1 - padding[1] - ds[1], 1 - cbw - 2* padding[1] - ds[0], ds[1]];
    ax2 = fig.add_axes(spec); #, sharex = ax)
    z2 = sch.dendrogram(Y, **dendogram_kwargs);
    #ax2.set_xticks([])
    #ax2.set_yticks([])
    ax2.axis('off')
  
  idx = z1['leaves'];
  D = distances.copy();
  n = len(D); ni = len(idx);
  stride = n/ni;
  if stride > 1:
    idx2 = np.array([i * stride + np.arange(stride) for i in idx]).flatten();
  else:
    idx2 = idx;
  D = D[np.ix_(idx2,idx2)];
  
  im = ax.imshow(D.max() - D, aspect='auto', origin='lower',interpolation = 'none', cmap= cmap)
  
  if stride > 1 and stride_line_kwargs is not None:
    for i in range(1,ni):
      ax.plot([i *stride-0.5, i * stride-0.5],[-0.5, n-0.5], **stride_line_kwargs);
      ax.plot([-0.5, n-0.5], [i *stride-0.5, i * stride-0.5], **stride_line_kwargs);
    ax.set_xlim(-0.5,n-0.5); ax.set_ylim(-0.5,n-0.5);
  ax.set_xticks([])
  ax.set_yticks([])
  if label is not None:
    xt = np.linspace(0,len(idx2),len(label)+1)[:-1];
    xt = xt + (xt[1] -1)/2.0;
    ax.set_xticks(xt);
    ax.set_xticklabels([label[i] for i in idx])
    #ax.set_yticks(range(len(idx)));
    #ax.set_yticklabels([label[i] for i in idx])

  
  # Plot colorbar
  if colorbar_width is not None:
    axcolor = fig.add_axes([1-cbw, padding[1], cbw - padding[0], 1 - 2 * padding[1] - ds[1]])
    plt.colorbar(im, cax=axcolor)













from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D

def plot_pca(data, analyse = True):
  """Performs PCA and plots an overview of the results"""
  if analyse:
    results = PCA(data);
  else:
    results = data;
  
  #results = PCA(X);
  pcs = results.Y;
  
  plt.subplot(1,3,1);
  plt.imshow(pcs, interpolation = 'none', aspect = 'auto', cmap = 'viridis')
  plt.colorbar(pad = 0.01,fraction = 0.01)
  plt.title('pca components');
  plt.subplot(2,3,2);
  plt.imshow(results.Wt, cmap = 'magma', interpolation = 'none' );
  plt.colorbar(pad = 0.01,fraction = 0.01)
  plt.title('pca vectors')
  ax = plt.gcf().add_subplot(2,3,5, projection = '3d');
  ax.plot(pcs[:,0], pcs[:,1], pcs[:,2], 'k');
  ax.scatter(pcs[:,0], pcs[:,1], pcs[:,2], 'bo', c = range(len(pcs[:,0])), cmap = plt.cm.Spectral );
  plt.xlabel('PCA1'); plt.ylabel('PCA2');
  ax.set_zlabel('PCA3');
  plt.subplot(2,3,3);
  plt.plot(results.mu)
  plt.title('mean');
  plt.subplot(2,3,6);
  plt.plot(np.cumsum(results.fracs), 'r')
  plt.title('variance explained')
  plt.tight_layout();

  return results;



import sklearn.manifold as sl;

def plot_tsne(data, analyse = True, n_components = 2, precomputed = False):
  """Perform t-SNE and plot overview of the results"""
  if precomputed:
    metric = 'precomputed';
  else:
    metric = 'euclidean';
  
  if analyse:
    tsne = sl.TSNE(n_components=n_components, init = 'pca', random_state = 0, metric = metric)
    Y = tsne.fit_transform(data)
  else:
    Y = data;
  
  if n_components == 1:
    plt.plot(Y);
  elif n_components == 2:
    plt.scatter(Y[:,0], Y[:,1], c = range(len(Y[:,0])), cmap = plt.cm.Spectral);
  else:
    ax = plt.gcf().add_subplot(1, 1, 1, projection = '3d');
    ax.scatter(Y[:, 0], Y[:, 1], Y[:,2], c = range(len(Y[:,0])), cmap=plt.cm.Spectral)
  plt.title("t-SNE")
  plt.tight_layout();

  return Y;


import sklearn.decomposition as sd

def plot_nmf(data, analyse = True, n_components = 2):
  """Perform NMF and plot overview of the results"""
  
  if analyse:
    nmf = sd.NMF(n_components=n_components, init = 'nndsvdar', random_state = 0, solver = 'cd')
    Y = nmf.fit_transform(data)
  else:
    Y = data;
    nmf = None;    
  
  if n_components is None:
    n_components = 3;
  
  if n_components == 1:
    plt.subplot(1,3,1);  
    plt.plot(Y);
  elif n_components == 2:
    plt.subplot(1,3,1); 
    plt.scatter(Y[:,0], Y[:,1], c = range(len(Y[:,0])), cmap = plt.cm.Spectral);
  else:
    ax = plt.gcf().add_subplot(1,3,1, projection = '3d');
    ax.scatter(Y[:, 0], Y[:, 1], Y[:,2], c = range(len(Y[:,0])), cmap=plt.cm.Spectral)
  plt.title("nmf")
  
  if nmf is not None:
    feat = nmf.components_;
    plt.subplot(1,3,2);
    plt.imshow(feat, interpolation = 'none', aspect = 'auto', cmap = 'viridis')
    plt.colorbar(pad = 0.01,fraction = 0.01)
    plt.title('features');
    
  plt.subplot(1,3,3);
  plt.imshow(Y, interpolation = 'none', aspect = 'auto', cmap = 'viridis')
  plt.colorbar(pad = 0.01,fraction = 0.01)
  plt.title('amplitudes');
  
  plt.tight_layout();
  


import time

def plot_embedding(Y, subplot = None, title  = None, color = None, cmap = plt.cm.spectral_r):
  """Plot data from an embedding / dim reduction"""
  fig  = plt.gcf();
  n_components = Y.shape[1];
  if n_components == 2:
    if subplot is not None:
      ax = fig.add_subplot(subplot[0],subplot[1],subplot[2]);
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=cmap)
  else:
    if subplot is None:
      ax = plt.gca();
    else:
      ax = fig.add_subplot(subplot[0],subplot[1],subplot[2], projection = '3d');
    ax.scatter(Y[:, 0], Y[:, 1], Y[:,2], c = color, cmap=cmap)
  plt.title(title);
 
 
import scipy.stats as st
 
def plot_embedding_contours(Y, contours = 10, cmap = plt.cm.plasma, xmin = None, xmax = None, ymin = None, ymax = None, npts = 100, density = False):
  """Plot a 2d density map of the embedding Y"""
    
  if xmin is None:
    xmin = np.min(Y[:,0]);
  if xmax is None:
    xmax = np.max(Y[:,0]);
  if ymin is None:
    ymin = np.min(Y[:,1]);
  if ymax is None:
    ymax = np.max(Y[:,1]);   
  
  #print xmin,xmax,ymin,ymax
  dx = float(xmax-xmin) / npts;
  dy = float(ymax-ymin) / npts;
  xx, yy = np.mgrid[xmin:xmax:dx, ymin:ymax:dy]
  positions = np.vstack([xx.ravel(), yy.ravel()])
  kernel = st.gaussian_kde(Y.T);
  #print xx.shape
  #print positions.shape
  #print Y.shape
  #print kernel(positions).shape
  f = kernel(positions)
  f = np.reshape(f, xx.shape)
  #print f.shape
  
  ax = plt.gca()
  ax.set_xlim(xmin, xmax)
  ax.set_ylim(ymin, ymax)
  # Contourf plot
  if density:
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
  ## Or kernel density estimate plot instead of the contourf plot
  #ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
  # Contour plot
  if contours is not None:
    cset = ax.contour(xx, yy, f, contours, cmap=cmap)
  # Label plot
  #ax.clabel(cset, inline=1, fontsize=10)
  ax.set_xlabel('Y0')
  ax.set_ylabel('Y1')
  
  return (kernel, f)
      


def plot_manifold_embeddings(X, n_neighbors = 10, n_components = 2, precomputed = False, label = None):
  """Performs a variety of manifold learing on the data and plots the results"""

  fig = plt.gcf(); fig.clf();
  
  methods = ['standard', 'ltsa', 'hessian', 'modified']
  labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']
  res = {};
  
  if precomputed:
    metric = 'precomputed';
  else:
    metric = None;
  
  if label is None:
    color = range(len(X[:,0]));
  else:
    color = label;
  cmap = plt.cm.Spectral;
  
  def plotY(Y, subplot, title):
    plot_embedding(Y, subplot = subplot, title  = title, color = color, cmap = cmap);

  if precomputed is False:
    for i, method in enumerate(methods):
      t0 = time.time()
      Y = sl.LocallyLinearEmbedding(n_neighbors, n_components,
                                          eigen_solver='auto',
                                          method=method).fit_transform(X)
      t1 = time.time()
      print("%s: %.2g sec" % (methods[i], t1 - t0))
      plotY(Y, (2, 4, 1 + i), "%s" % labels[i])
      res[method] = Y;
     
    t0 = time.time();
    Y = sl.Isomap(n_neighbors, n_components).fit_transform(X);
    t1 = time.time();
    print("Isomap: %.2g sec" % (t1 - t0));
    plotY(Y, (2, 4, 8), "Isomap") 
    res["Isomap"] = Y;
    
    nplot0 = 2;
    nplot1 = 4;  
  else:
    nplot0 = 1;
    nplot1 = 3;
    
  t0 = time.time()
  mds = sl.MDS(n_components, max_iter=100, n_init=1, dissimilarity = metric)
  Y = mds.fit_transform(X)
  t1 = time.time()
  print("MDS: %.2g sec" % (t1 - t0));
  plotY(Y, (nplot0,nplot1,(nplot0-1)*nplot1+1), "MDS")
  res["MDS"] = Y;
  
  t0 = time.time()
  se = sl.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors, affinity = metric)
  Y = se.fit_transform(X)
  t1 = time.time()
  print("SpectralEmbedding: %.2g sec" % (t1 - t0));
  plotY(Y, (nplot0,nplot1,(nplot0-1)*nplot1+2), "SpectralEmbedding")
  res["SpectralEmbedding"] = Y;

  t0 = time.time()
  if precomputed:
    init = 'random';
  else:
    init = 'pca';
  tsne = sl.TSNE(n_components=n_components, init=init, random_state=0, metric = metric)
  Y = tsne.fit_transform(X)
  t1 = time.time()
  print("t-SNE: %.2g sec" % (t1 - t0))
  plotY(Y, (nplot0,nplot1,(nplot0-1)*nplot1+3), "t-SNE")
  plt.tight_layout();
  res["t-SNE"] = Y;
  
  return res;
  

from matplotlib.collections import LineCollection

def plot_colored_line(x,y, color, cmap = 'spectral_r', line_width = 2):
  """Plot a line with color code"""
  segments = np.array([x, y]).T.reshape(-1, 1, 2);
  segments = np.concatenate([segments[:-1], segments[1:]], axis=1);
  lc = LineCollection(segments, cmap=cmap);
  lc.set_array(np.array(color));
  #lc.set_linewidth(line_width);
  plt.gca().add_collection(lc)


def test():
  import numpy as np
  import plot as p;
  reload(p)
  data = np.random.rand(100,200);
  p.plot(data, 'test')
  
  reload(p)
  pts  = np.random.rand(10000,3);
  p.plot3d(pts);
  
if __name__ == '__main__':
  test();