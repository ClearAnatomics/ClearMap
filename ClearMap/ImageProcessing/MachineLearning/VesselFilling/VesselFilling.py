# -*- coding: utf-8 -*-
"""
VesselFilling
=============

This module uses a convolutionary neuronal network to fill empty tubes 
and vessels.
"""
__author__    = 'Sophie Skriabin, Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'



import os
import gc
import numpy as np
import torch


import ClearMap.IO.IO as io

import ClearMap.ParallelProcessing.BlockProcessing as bp
import ClearMap.ParallelProcessing.DataProcessing.ArrayProcessing as ap


import ClearMap.ImageProcessing.MachineLearning.Torch as tor
import ClearMap.ImageProcessing.MachineLearning.VesselFilling.VesselFillingNetwork as vfn

import ClearMap.Utils.Timer as tmr;


###############################################################################
### Locations of trained networks
###############################################################################

resources_path = os.path.join(os.path.abspath(os.path.split(__file__)[0]), 'Resources');
"""Path to the trained neuronal networks."""

network_binary_vessel_filling_filename = os.path.join(resources_path, 'cnn_binary_vessel_filling.pth');
"""Filename of the default neuronal network to use for binary hollow vessel filling."""


###############################################################################
### Default processing parameter
###############################################################################

default_fill_vessels_processing_parameter = dict(
    overlap = 100,
    axes = [2],
    size_max = None,
    optimization = False,
    processes = 1);


###############################################################################
### Tube filling
###############################################################################

def vessel_filling_network(network = None, dtype = 'float16', cuda = None):
  """Initialize vessel filling network.
  
  Arguments
  ---------
  network : str, Model or None
    The network speicifcation. If None the default trained network is used.
  dtype : str
    The dtype to use for the network. See 
    :func:`ClearMap.ImageProcessing.MachineLearning.Torch.to` for details.
  cuda : bool or None
    If True, use gpu processing. If None, automatically detect gpu.
  
  Returns
  -------
  network : Model
    The neural network model.
  """
  if network is None:
    network = network_binary_vessel_filling_filename;
  if isinstance(network, str):
    network = vfn.VesselFillingNetwork(load=network);
  network = tor.to(network, dtype=dtype);
  
  if cuda is None:
    cuda = torch.cuda.is_available();
  if cuda:
    network = network.cuda();
  
  return network;

    
def fill_vessels(source, sink, 
                 resample = None, threshold = 0.5,
                 network = None, dtype = 'float16', cuda = None,
                 processing_parameter = None,
                 verbose = False):
  """Fill hollow tubes via a neural network.
  
  Arguments
  ---------
  source : str or Source
    The binary data source to fill hollow tubes in.
  sink : str or Source.
    The binary sink to write data to. sink is created if it does not exists.
  resample : int or None
    If int, downsample the data by this factor, apply network and upsample.
  threshold : float or None
    Apply a threshold to the result of the cnn. If None, the probability of
    being foreground is returned.
  network : str, Model or None
    The network speicifcation. If None, the default trained network is used.
  dtype : str
    The dtype to use for the network. See 
    :func:`ClearMap.ImageProcessing.MachineLearning.Torch.to` for details.
  cuda : bool or None
    If True, use gpu processing. If None, automatically detect gpu.
  processing_parameter : dict or None
    Parameter to use for block processing.
  verbose : bool
    If True, print progress.
  
  Returns
  -------
  network : Model
    The neural network model.
  """
  if verbose:
    timer = tmr.Timer();

  #cuda
  if cuda is None:
    cuda = torch.cuda.is_available();
    
  #initialize network
  network = vessel_filling_network(network=network, dtype=dtype, cuda=cuda);
  if not cuda:  #some functions only work as float on CPU
    network = network.float();
  if verbose:
    timer.print_elapsed_time('Vessel filling: neural network initialized')
    print(network);
    print('Vessel filling: using %s' % (('gpu' if cuda else 'cpu'),))
  
  #initialize source
  source = io.as_source(source);
 
  if verbose:
    timer.print_elapsed_time('Vessel filling: source loaded');
    
  #initialize sink
  if threshold:
    sink_dtype = bool;
  else:
    sink_dtype = dtype;
  sink, sink_shape = ap.initialize_sink(sink=sink, shape=source.shape, dtype=sink_dtype, order=source.order, return_buffer=False, return_shape=True);
  
  #resampling
  if resample is not None:
    maxpool = torch.nn.MaxPool3d(kernel_size=resample)
    upsample = torch.nn.Upsample(mode="trilinear", scale_factor=resample, align_corners=False);
    
    if cuda:
      maxpool = maxpool.cuda();
      upsample = upsample.cuda();
      if dtype is not None:
        maxpool  = tor.to(maxpool, dtype);
        upsample = tor.to(upsample, dtype);
    else:
      maxpool = maxpool.float();
      upsample = upsample.float();
  
  #processing
  if processing_parameter is None:
    processing_parameter = default_fill_vessels_processing_parameter
  if processing_parameter:
    processing_parameter = processing_parameter.copy();
    processing_parameter.update(optimization=False);
    if 'size_max' not in processing_parameter or processing_parameter['size_max'] is None:
      processing_parameter['size_max'] = np.max(source.shape);
    if 'size_min' not in processing_parameter:
      processing_parameter['size_min'] = None;    
    blocks = bp.split_into_blocks(source, **processing_parameter);  
  else:
    blocks = [source];
  
  #process blocks
  for block in blocks:
    if verbose:
      timer_block = tmr.Timer();
      print('Vessel filling: processing block %s' % (block.info()));
    
    #load data
    data = np.array(block.array);
    if data.dtype == bool:
      data = data.astype('uint8');
    data = torch.unsqueeze(torch.from_numpy(data), 0)
    if cuda:
      data = tor.to(data, dtype=dtype);
      data = data.cuda();
    else:
      data = data.float();
    if verbose:
      print('Vessel filling: loaded data: %r' % (tuple(data.shape),));
    
    #down sampleprocessing_parameter
    if resample:        
      result = maxpool(data);
    else:
      result = data; 
    result = torch.unsqueeze(result, 1);  
    if verbose:
      print('Vessel filling: resampled data: %r' % (tuple(result.shape),));
    
    #fill
    result = network(result);
    if verbose:
      print('Vessel filling: network %r' % (tuple(result.shape),));
    
    #upsample
    if resample:
      result = upsample(result)
    if verbose:
      print('Vessel filling: upsampled %r' % (tuple(result.shape),));
      
    #write data
    sink_slicing = block.slicing;
    result_shape = result.shape;
    result_slicing = tuple(slice(None,min(ss, rs)) for ss,rs in zip(sink_shape, result_shape[2:]));
    data_slicing = (0,) + tuple(slice(None, s.stop) for s in result_slicing);
    sink_slicing = bp.blk.slc.sliced_slicing(result_slicing, sink_slicing, sink_shape);  
    result_slicing = (0,0) + result_slicing;

    #print('result', result.shape, result_slicing, 'data', data.shape, data_slicing, 'sink', sink_shape, sink_slicing)
    
    if threshold:
      sink_prev = torch.from_numpy(np.asarray(sink[sink_slicing], dtype='uint8'));
    else:
      sink_prev = torch.from_numpy(sink[sink_slicing]);
    
    if cuda:
      sink_prev = sink_prev.cuda();
      sink_prev = tor.to(sink_prev, dtype=dtype);
    else:
      sink_prev  = sink_prev.float();

    #print('slices:', result[result_slicing].shape, data[data_slicing].shape, sink_prev.shape)
    
    result = torch.max(torch.max(result[result_slicing], data[data_slicing]), sink_prev);
    if threshold:
      result = result >= threshold;
    if verbose:
      print('Vessel filling: thresholded %r' % (tuple(result.shape),));
    
    if cuda:
      sink[sink_slicing] = result.data.cpu();
    else:
      sink[sink_slicing] = result.data;
    
    if verbose:
      print('Vessel filling: result written to %r' % (sink_slicing,));

    del data, result, sink_prev;
    gc.collect();
    
    if verbose:
      timer_block.print_elapsed_time('Vessel filling: processing block %s' % (block.info()));
   
  if verbose:
    timer.print_elapsed_time('Vessel filling');
  
  return sink;

###############################################################################
### Tube filling training
###############################################################################

def train(network, data):
  pass






#%%############################################################################
### Tests
###############################################################################

def _test():
  import ClearMap.ImageProcessing.MachineLearning.VesselFilling.VesselFilling as vf
  
  
  


