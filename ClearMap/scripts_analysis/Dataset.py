from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


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



    def __getitem__(self, idx):
        try:
            image = np.load(os.path.join(self.load_dir, self.data[idx][0]))#.astype(float)
            groundtruth = np.load(os.path.join(self.load_dir, self.data[idx][1]))

            if 'dense'in self.data[idx][0]:
                data = np.concatenate((image[np.newaxis, :],groundtruth[np.newaxis, :]), axis=0)#image[np.newaxis, :]+
            else:
            #     data = np.concatenate((image[np.newaxis, :], groundtruth[np.newaxis, :]),axis=0)  # image[np.newaxis, :]+
                data = np.concatenate((image[np.newaxis, :], np.logical_or(image[np.newaxis, :],groundtruth[np.newaxis, :])), axis=0)#image[np.newaxis, :]+

        except ValueError:
            print(str(self.data[idx][0]))
        return data.astype(float)

