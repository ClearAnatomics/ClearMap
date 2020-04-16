import geodesic_distance
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
def test_geodesic_distance2d():
    I = np.asarray(Image.open('data/img2d.png').convert('L'), np.float32)
    S = np.zeros_like(I, np.uint8)
    S[100][100] = 1
    t0 = time.time()
    D1 = geodesic_distance.geodesic2d_fast_marching(I,S)
    t1 = time.time()
    D2 = geodesic_distance.geodesic2d_raster_scan(I,S)
    dt1 = t1 - t0
    dt2 = time.time() - t1
    print "runtime(s) of fast marching {0:}".format(dt1)
    print "runtime(s) of raster  scan  {0:}".format(dt2)
    plt.subplot(1,3,1); plt.imshow(I, cmap='gray')
    plt.autoscale(False);  plt.plot([100], [100], 'ro')
    plt.axis('off'); plt.title('input image')
    
    plt.subplot(1,3,2); plt.imshow(D1)
    plt.axis('off'); plt.title('fast marching')
    
    plt.subplot(1,3,3); plt.imshow(D2)
    plt.axis('off'); plt.title('ranster scan')
    plt.show()

if __name__ == '__main__':
    test_geodesic_distance2d()
