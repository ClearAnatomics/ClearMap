#TODO: integrate fully in ClearMap

__author__    = 'Sophie Skriabin, Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'



from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torch import nn
import torch
#from vesselSegmentation.centerLineExtractor import preprocessNegChannel
from vesselSegmentation.visualize import plot3d, show
from vesselSegmentation.visualize import twoClassesMap, get_two_views

logdir="/home/sophie.skriabine/Bureau/"


def map_float_to_int(x):
    out = x*255 / np.max(x)
    return out


class Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, workdir):

        """
        Args:
            workdir (string): Directory with all the images.
        """
        self.load_dir = workdir
        data = []
        n=0
        for path, subdirs, files in os.walk(workdir):
            for subdir in subdirs:
                # if not 'dense' in subdir:
                #     for path, subdirs, files in os.walk(os.path.join(workdir, subdir)):
                        gt_dir = workdir[:-8] + 'groundtruth/'+str(subdir)+'/'
                        print(gt_dir)
                        print(os.path.join(workdir,subdir))
                        for  path, subdirs, files in os.walk(os.path.join(workdir,subdir)):
                                for name in files:
                                    if '.npy' in name:
                                        id=name[20:]#20#18
                                        # print(id)
                                        gtname=name[:14]+'GT_'+id#14#12
                                        if ('hessian' in gt_dir):#or ('3R' in gt_dir) or ('4R' in gt_dir) or ('4R_t' in gt_dir) or ('2R' in gt_dir)
                                            if n<1:
                                                data.append([os.path.join(path, name), os.path.join(gt_dir, gtname)])
                                                n=n+1
                                        # print(os.path.join(path, name),os.path.join(gt_dir, gtname))
                                        else:
                                            data.append([os.path.join(path, name), os.path.join(gt_dir, gtname)])
                                    # for path, subdirs, files in os.walk(gt_dir):
                                    #     for file in files:
                                    #         #print(file)
                                    #         if id in file:
                                    #             gtname=file
                                    #             print(gtname, name)
                                                #data.append([os.path.join(path, name),os.path.join(gt_dir, gtname)])
        print('data size : '+str(len(data)))
        self.data = data

    def __len__(self):
        return len(self.data)



    def make_filled_cylinder(self, array):

        return array


    def __getitem__(self, idx):
        try:
            image = np.load(os.path.join(self.load_dir, self.data[idx][0]))#.astype(float)
            groundtruth = np.load(os.path.join(self.load_dir, self.data[idx][1]))
            # print(np.unique(groundtruth))
            # np.save(os.path.join(logdir, 'orig'), image)
            # image = map_float_to_int(image).astype('uint8')
            #neg_image = -image
            if 'dense'in self.data[idx][0]:
                data = np.concatenate((image[np.newaxis, :],groundtruth[np.newaxis, :]), axis=0)#image[np.newaxis, :]+
            else:
            #     data = np.concatenate((image[np.newaxis, :], groundtruth[np.newaxis, :]),axis=0)  # image[np.newaxis, :]+
                data = np.concatenate((image[np.newaxis, :], np.logical_or(image[np.newaxis, :],groundtruth[np.newaxis, :])), axis=0)#image[np.newaxis, :]+
            # image = image.reshape((1,) + image.shape)
            # print(data.shape)
            # np.save(os.path.join(logdir, 'pre_process'), data[0])
            # image=preprocessNegChannel(image)
            # np.save(os.path.join(logdir, 'pre_process_clahe'), image[0])
        except ValueError:
            print(str(self.data[idx][0]))
        return data.astype(float)



if __name__ == '__main__':
    workdir = '/mnt/data_SSD_1to/Dataset/xR_vessels/patches/'  # ordi fixe
    gt_dir = workdir[:-14] + 'groundtruth'
    for path, subdirs, files in os.walk(workdir):
        for subdir in subdirs:
            if subdir=='dense':
                gt_dir = workdir[:-8] + 'groundtruth/' + str(subdir) + '/'
                print(gt_dir)
                # print(os.path.join(workdir, subdir))
                for path, subdirs, files in os.walk(os.path.join(workdir, subdir)):
                    for name in files:
                        if '.npy' in name:
                            id = name[20:]
                            print('data id : ' + id)
                            # print(id)
                            gtname = name[:14] + 'GT_' + id
                            arr2= np.load(os.path.join(path, name))
                            arr1= np.load(os.path.join(gt_dir, gtname))
                            # print(os.path.join(path, name),os.path.join(gt_dir, gtname))
                            vb1, vb2 = get_two_views()
                            plot3d(arr2, view=vb2)
                            plot3d(arr1, view=vb1)
                            show()                # for path, subdirs, files in os.walk(gt_dir):
                            #     for file in files:
                            #         #print(file)
                            #         if id in file:
                            #             gtname=file
                            #             print(gtname, name)
                            # data.append([os.path.join(path, name),os.path.join(gt_dir, gtname)])
                            print('data size : ' + str(len(arr1)))