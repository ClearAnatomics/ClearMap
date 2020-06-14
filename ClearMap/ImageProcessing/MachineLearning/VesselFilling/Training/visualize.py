#TODO: integrate fully in ClearMap

__author__    = 'Sophie Skriabin, Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import vispy
import vispy.color
import vispy.scene
from vispy import scene
import vispy.app
import vispy.visuals
import numpy as np
import os
import vesselSegmentation.TurntableCamera as tc
import vesselSegmentation.IO.MMP as mp

# from skimage.measure import block_reduce

class twoClassesMap(vispy.color.colormap.BaseColormap):
    glsl_map = """
    vec4 threeclasses(float t) {
        if(t == 0){
            return vec4(0, 0, 0, 0);
        }
        else if (t == 1) {
            return vec4(0.8, 0.1, 0.1, 0.3);
        }
        else {
            return vec4(0.8, 0.1, 0.1, 0.3);
        }
    }
    """
    """
    t should take values in 0, 1, 2
    """

    def map(self, t):
        if isinstance(t, np.ndarray):
            return np.hstack([t, 0.1, 1-t, 1]).astype(np.float32)
        else:
            return np.array([t, 0.1, 1-t, 1], dtype=np.float32)


class FireMap(vispy.color.colormap.BaseColormap):
    colors = [(1.0, 1.0, 1.0, 0.0),
              (1.0, 1.0, 0.0, 0.05),
              (1.0, 0.0, 0.0, 0.1)]

    glsl_map = """
    vec4 fire(float t) {
        return mix(mix($color_0, $color_1, t),
                   mix($color_1, $color_2, t*t), t);
    }
    """


def show():
    vispy.app.run()


def get_two_views():
    """
    Get two views in order to plot two graphs/images in a consistent manner
    """
    canvas = vispy.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)
    vb1 = scene.widgets.ViewBox(border_color='yellow', parent=canvas.scene, camera=tc.TurntableCamera())
    vb2 = scene.widgets.ViewBox(border_color='blue', parent=canvas.scene, camera=tc.TurntableCamera())

    grid = canvas.central_widget.add_grid()
    grid.padding = 6
    grid.add_widget(vb1, 0, 0)
    grid.add_widget(vb2, 0, 1)

    for view in vb1, vb2:
        view.camera = 'turntable'
        view.camera.fov = 100
        view.camera.distance = 0
        view.camera.elevation = 0
        view.camera.azimuth = 0

    vb1.camera.aspect = vb2.camera.aspect = 1
    vb1.camera.link(vb2.camera)

    return vb1, vb2



def plot3d(data, colormap = FireMap(), view=None):
    VolumePlot3D = vispy.scene.visuals.create_visual_node(vispy.visuals.VolumeVisual)
    # Add a ViewBox to let the user zoom/rotate
    # build canvas
    if view is None:
        canvas = vispy.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)
        view = canvas.central_widget.add_view(camera=tc.TurntableCamera())
        view.camera = 'turntable'
        view.camera.fov = 0
        view.camera.distance = 7200
        view.camera.elevation = 31
        view.camera.azimuth = 0
        view.camera.depth_value = 100000000

    cc = (np.array(data.shape) // 2)
    view.camera.center = cc

    return VolumePlot3D(data.transpose([2, 1, 0]), method='translucent', relative_step_size=1.5,
                        parent=view.scene, cmap=colormap)


def reconstructionView(readdir, N):
    img = np.load(os.path.join(readdir, 'input_' + str(N) + '.npy'))
    output = np.load(os.path.join(readdir, 'output_' + str(N) + '.npy'))
    gt = np.load(os.path.join(readdir, 'groundtruth_' + str(N) + '.npy'))

    print("img", np.unique(img))
    print("output", np.unique(output))

    vb1, vb2 = get_two_views()
    plot3d(img, view=vb1)
    plot3d(gt+img, view=vb2)
    show()


def viewFile(filepath):
    img = np.load(filepath)

    # print("img", np.unique(img))
    # np.save('/mnt/vol00-renier/Sophie/DataToPlot/resDNN/outputcrustfull06thresh.npy', img>0.6)
    super_threshold_indices = img <= 0.6
    img[super_threshold_indices] = 0
    vb1, vb2 = get_two_views()
    plot3d(img, view=vb1)
    show()


if __name__ == '__main__':
    #
    # readdir = 'logs/training240419_sparseMidVessTest2'
    readdir = 'logs/training050719_test'
    #'logs/training060319_axons_recog_test_1'
    N = 15100#49800#41850
    #59200#133150#117300 #139250#
    #40
    reconstructionView(readdir, N)
    # #
    # input = np.load(os.path.join(readdir, 'output_' + str(N) + '.npy'))
    # output = np.load(os.path.join(readdir, 'groundtruth_' + str(N) + '.npy'))
    # #
    # #
    # input=np.load('/mnt/data_2to/Axons/data/test2/data_binary_center.npy')
    # output = np.load('/mnt/data_2to/Axons/data/res_test2.npy')
    # input = np.load('/mnt/data_SSD_2to/190123_7/inputinput.npy')
    # output =np.load('/mnt/data_SSD_2to/190123_7/outputoutput.npy')
    # # input, output[1500:1700, 3750:3950, 2380:2580]
    # # # output=block_reduce(output, block_size=(2, 2, 2), func=np.maximum).astype(int)
    # vb1, vb2 = get_two_views()
    # # # plot3d(arr, view=vb2)
    # # # show()
    # # # print(np.unique(0.47-latent1))
    # plot3d(input>0.5, view=vb1)
    # plot3d(output, view=vb2)
    # show()
    # # viewFile('/media/sophie.skriabine/TOSHIBA EXT/Sophie/teryData/arteryData/data_arteries_crust.npy')
    # arr=np.load('/mnt/vol00-renier/Nicolas/data_binary.npy')
    # plot3d(arr)
    # show()GXFGv