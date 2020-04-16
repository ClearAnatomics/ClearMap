import geodesic_distance
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import nibabel

def load_nifty_volume_as_array(filename):
    # input shape [W, H, D]
    # output shape [D, H, W]
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    return data

def save_array_as_nifty_volume(data, filename):
    # numpy data shape [D, H, W]
    # nifty image shape [W, H, W]
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)

def geodesic_distance_3d(I, S, lamb, iter):
    '''
    get 3d geodesic disntance by raser scanning.
    I: input image
    S: binary image where non-zero pixels are used as seeds
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return geodesic_distance.geodesic3d_raster_scan(I,S, lamb, iter)

def test_geodesic_distance3d():
    I = load_nifty_volume_as_array("data/img3d.nii")
    I = np.asarray(I, np.float32)
    I = I[18:38, 63:183, 93:233 ]
    S = np.zeros_like(I, np.uint8)
    S[10][60][70] = 1
    t0 = time.time()
    D1 = geodesic_distance.geodesic3d_fast_marching(I,S)
    t1 = time.time()
    D2 = geodesic_distance_3d(I,S, 1.0, 4)
    dt1 = t1 - t0
    dt2 = time.time() - t1
    print "runtime(s) fast marching {0:}".format(dt1)
    print "runtime(s) raster scan   {0:}".format(dt2)

    D1 = D1*255/D1.max()
    D1 = np.asarray(D1, np.uint8)
    save_array_as_nifty_volume(D1, "data/image3d_dis1.nii")

    D2 = D2*255/D2.max()
    D2 = np.asarray(D2, np.uint8)
    save_array_as_nifty_volume(D2, "data/image3d_dis2.nii")
    
    I = I*255/I.max()
    I = np.asarray(I, np.uint8)
    save_array_as_nifty_volume(I, "data/image3d_sub.nii")

    I_slice = I[10]; D1_slice = D1[10]; D2_slice = D2[10]
    plt.subplot(1,3,1); plt.imshow(I_slice, cmap='gray')
    plt.autoscale(False);  plt.plot([70], [60], 'ro')
    plt.axis('off'); plt.title('input image')
    
    plt.subplot(1,3,2); plt.imshow(D1_slice)
    plt.axis('off'); plt.title('fast marching')
    
    plt.subplot(1,3,3); plt.imshow(D2_slice)
    plt.axis('off'); plt.title('ranster scan')
    plt.show()

if __name__ == '__main__':
    test_geodesic_distance3d()
