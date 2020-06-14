#TODO: integrate fully in ClearMap

__author__    = 'Sophie Skriabin, Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader  # , Dataset
from vesselSegmentation.dataset import Dataset
from vesselSegmentation.visualize import plot3d, show
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from vesselSegmentation.gaussianKernel import get_gaussian_filter
logdir = 'logs'


class Contrast_loss(nn.Module):

    def __init__(self):
        super(Contrast_loss, self).__init__()

    def forward(self, latent):
        contrast=np.amax(latent.cpu().numpy())-np.amin(latent.cpu().numpy())
        # print(contrast)
        return torch.sum(torch.exp((latent - 0.5)**2 / 0.5)*(1/contrast))#, torch.tensor(contrast))


if __name__ == "__main__":

    ones = np.ones((1, 2, 20, 40, 40))
    zeros = np.zeros((1, 2, 20, 40, 40))
    latent = np.concatenate((ones, ones*0.5), axis=2)
    latent = torch.from_numpy(latent).cuda().float()

    criterion2 = Contrast_loss()

    contrast_loss = criterion2(latent)

    print(contrast_loss)