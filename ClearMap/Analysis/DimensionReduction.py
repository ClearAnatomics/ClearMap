# -*- coding: utf-8 -*-
"""
DimensionReduction
==================

Module hosting dimension reduction and embedding routines.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

import matplotlib.pyplot as plt;

import sklearn.manifold as sl;

import ClearMap.Visualization.Plot3d as p3d

###############################################################################
### TSNE
###############################################################################

def tsne(data, n_components = 2, precomputed = False, 
               perplexity=30.0, random_state = 0, init = 'pca',
               early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, **kwargs):
  """Perform t-SNE"""
  
  if precomputed:
    metric = 'precomputed';
  else:
    metric = None;
  
  tsne = sl.TSNE(n_components = n_components, init = init, random_state = random_state, 
                 metric = metric, perplexity = perplexity, early_exaggeration = early_exaggeration,
                 learning_rate = learning_rate, n_iter = n_iter, **kwargs);
  return tsne.fit_transform(data);


#TODO: for millions of points -> use vispy / backend option
def plot_tsne(data, cmap = plt.cm.Spectral, colors = None, title = 't-SNE', backend = 'cm'):
  if data.ndim == 1:
    n_components = 1;
  else:
    n_components = data.shape[1];
    
  if colors is None:
    colors = np.arange(len(data[:,0]));
  
  if backend == 'plt':
    if n_components == 1:
      plt.plot(data);
    elif n_components == 2:
      plt.scatter(data[:,0], data[:,1], c = colors, cmap = cmap);
    else:
      ax = plt.gcf().add_subplot(1, 1, 1, projection = '3d');
      ax.scatter(data[:, 0], data[:, 1], data[:,2], c = colors, cmap = cmap)
    plt.title(title)
    plt.tight_layout();
  else:
    raise NotImplementedError();
