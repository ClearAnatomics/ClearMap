#TODO: integrate fully in ClearMap

__author__    = 'Sophie Skriabin, Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


#import os
import torch
from torch import nn
#from torch.autograd import Variable
#from torch.utils.data import DataLoader  # , Dataset
#from vesselSegmentation.dataset import Dataset
# from vesselSegmentation.visualize import plot3d, show
import numpy as np
#from tensorboardX import SummaryWriter
#import torch.nn.functional as F
# from vesselSegmentation.gaussianKernel import get_gaussian_filter
logdir = 'logs'

def testClassBalanceLoss(output, groundtruth, normalized=True, cuda=True):
    # if cuda:
    #     ones = torch.from_numpy(np.ones((1, input.shape[1], input.shape[2], input.shape[3],
    #                                      input.shape[4]))).cuda().float()
    #     res = torch.from_numpy(np.zeros(1)).cuda().float()
    # else:
    #     ones = torch.from_numpy(np.ones((1, input.shape[1], input.shape[2], input.shape[3], input.shape[4]))).float()
    #     res = torch.from_numpy(np.zeros(1)).float()
    # print("res", res)
    y_plus=output[groundtruth>0]
    y_minus=1-output[groundtruth==0]
    # print(len(y_plus))
    # print(len(y_minus))
    pos=torch.mul(groundtruth == 0,  output > 0.5)
    neg=torch.mul(groundtruth > 0, output <= 0.5)
    # print(pos)
    # print(neg)
    false_pos=1-output[pos]
    false_neg = output[neg]
    # print(false_pos.shape)
    # print(false_neg.shape)
    L1=cross_entropy(output, groundtruth, y_plus, y_minus, cuda)
    L2=rate_correction(output, groundtruth, false_pos, false_neg, cuda)
    result=L1+L2
    print("balance loss", result)
    return result.cuda().float()


def cross_entropy(output, groundtruth, y_plus, y_minus, cuda):
    if cuda:
        res1 = torch.sum(torch.log(y_plus).cuda().float())
        res2 = torch.sum(torch.log(y_minus).cuda().float())
    else:
        res1 = torch.sum(torch.log(torch.from_numpy(y_plus).float()))
        res2 = torch.sum(torch.log(torch.from_numpy(y_minus).float()))
    # print(res1)
    # print(res2)
    # res1 = torch.sum(torch.log(torch.from_numpy(y_plus).cuda().float()))
    # res2 = torch.sum(torch.log(torch.from_numpy(y_minus).cuda().float()))
    l1=len(y_plus)
    l2=len(y_minus)
    if l1==0:
        l1=1
    if l2==0:
        l2=1
    res = -((1/(l1))*res1)-((1/(l2))*res2)
    return res


def rate_correction(output, groundtruth, false_pos, false_neg, cuda):
    if cuda:
        res1 = torch.sum(torch.log(false_pos).cuda().float())
        res2 = torch.sum(torch.log(false_neg).cuda().float())
    else:
        res1 = torch.sum(torch.log(torch.from_numpy(false_pos).float()))
        res2 = torch.sum(torch.log(torch.from_numpy(false_neg).float()))

    l1 = len(false_pos)
    l2 = len(false_neg)
    if l1 == 0:
        l1 = 1
    if l2 == 0:
        l2 = 1

    g1 = 0.5 + ((1 / (l1)) * torch.sum(false_pos - 0.5))
    g2 = 0.5 + ((1 / (l2)) * torch.sum(false_neg - 0.5))

    res=-((g1/(l1))*res1)-((g2/(l2))*res2)
    return res


class Class_balancing_loss(nn.Module):

    def __init__(self, cuda=True):
        super(Class_balancing_loss, self).__init__()
        self.cuda = cuda

    def forward(self, output, groundtruth):
        return testClassBalanceLoss(output, groundtruth, normalized=False, cuda=self.cuda)


# class Normalized_soft_cut_loss(nn.Module):
#
#     def __init__(self, cuda=True):
#         super(Normalized_soft_cut_loss, self).__init__()
#         self.cuda = cuda
#
#     def forward(self, x, kernel):
#         return normalizedSoftCutLoss(x, kernel, cuda=self.cuda)


if __name__ == "__main__":

    ones = np.ones((1, 2, 20, 40, 40))
    zeros = np.zeros((1, 2, 20, 40, 40))
    latent = np.concatenate((ones*0.5, ones*0.5), axis=2)
    latent = torch.from_numpy(latent).cuda().float()

    criterion2 = Class_balancing_loss()
F
    # print(np.unique(latent[0, 1, :, :, 0]))

    sigma = 4
    kernel_size = 11
    # kernel = get_gaussian_filter(sigma, kernel_size)

    soft_loss = criterion2(latent, kernel)

    print(soft_loss)