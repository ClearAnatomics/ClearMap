#  -*- coding: utf-8 -*-
"""
VesselFillingNetwork
====================

This module implements the vessel filling neuronal network in PyTorch.
"""
__author__    = 'Sophie Skriabin, Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import torch
import torch.nn as nn


###############################################################################
### Convolutional neural network architecture
###############################################################################

class DeSepConv3d(nn.Module):
  """Depthwise separable convolutional layer."""
  def __init__(self, nin, nout):
      super(DeSepConv3d, self).__init__()
      self.depthwise = nn.Conv3d(nin, nin,  kernel_size=3, padding=(1,1,1), groups=nin)
      self.pointwise = nn.Conv3d(nin, nout, kernel_size=1)

  def forward(self, x):
      out = self.depthwise(x)
      out = self.pointwise(out)
      return out


class VesselFillingNetwork(nn.Module):
  """Vessel filling neuronal network."""
  
  def __init__(self, load = None):
    super(VesselFillingNetwork, self).__init__()
    
    #architecture 
    self.conv1      = nn.Conv3d(1 , 16, 7, stride=2, padding=(3,3,3), dilation=1)
    self.conv2      = nn.Conv3d(16, 32, 5, stride=1, padding=(2,2,2), dilation=1)
    self.desepconv4 = DeSepConv3d(32, 32)
    self.convbin1   = nn.Conv3d(32, 32, 3, stride=1, padding=(1,1,1), dilation=1)
    self.conv3      = nn.ConvTranspose3d(32, 16, 7, stride=2, padding=(3,3,3), dilation=1)
    self.conv4      = nn.Conv3d(17, 2, 3, stride=1, padding=(1,1,1), dilation=1)
    
    # non-linearities
    self.maxpool  = nn.MaxPool3d(kernel_size=2)
    self.relu     = nn.ReLU(True)
    self.dropout  = nn.Dropout(p=0.15)
    self.softmax  = nn.Softmax(dim=1)
    self.out_act  = nn.Sigmoid()
    self.upsample = nn.Upsample(mode="trilinear", scale_factor=2, align_corners=False)
    
    if load is not None:
      map_location = None if torch.cuda.is_available() else torch.device('cpu');
      self.load_state_dict(torch.load(load, map_location=map_location));
  
  def encode(self, inp):
    skip1 = inp
    inp = self.maxpool(inp)
    x = self.relu(self.conv1(inp))
    x = self.dropout(x)
    x = self.maxpool(x)
    x = self.relu(self.conv2(x))
    x = self.dropout(x)
    x = self.desepconv4(x)
    x = self.relu(self.convbin1(x))
    x = self.dropout(x)
    x = self.upsample(x)
    x = self.relu(self.conv3(x))
    x = self.dropout(x)
    x = self.upsample(x)
    slicing = (slice(None), slice(None)) + tuple(slice(None,min(skips, xs)) for skips,xs in zip(skip1.shape[2:], x.shape[2:]));
    x = torch.cat((skip1[slicing], x[slicing]), 1)
    x = self.out_act(4 * (self.conv4(x) - 0.5))
    return self.softmax(x)[:,[0]];

  def decode(self, x):
    return x

  def forward(self, x):
    x = self.encode(x)
    return x
