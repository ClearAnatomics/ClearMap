import os

import matplotlib
import numpy as np
from matplotlib.colors import hsv_to_rgb
from skimage import transform as sk_transform

from PyQt5 import QtGui
from PyQt5.QtGui import QColor

from ClearMap import Settings
from ClearMap.IO import TIF


def np_to_qpixmap(img_array, alpha):
    img_array = img_array.astype(np.float64, copy=True)
    rng = img_array.min(), img_array.max()
    dynamic_range = rng[1] - rng[0]

    if dynamic_range == 0:
        img_array[:] = 0.0  # avoid divide-by-zero
    else:
        img_array = (img_array - rng[0]) / dynamic_range  # normalise to 0..1

    # img = np.uint8(matplotlib.cm.Blues_r(img_array)*255)
    rgba = (matplotlib.cm.copper(img_array) * 255).astype(np.uint8)  # (H,W,4)

    # Replace alpha channel with provided alpha and ensure range is 0..255
    if alpha.dtype == bool:
        scaled_alpha = np.where(alpha, 255, 0).astype(np.uint8)
    else:
        scaled_alpha = alpha.astype(np.uint8)
    rgba[..., 3] = scaled_alpha

    h, w, _ = rgba.shape
    qimg = QtGui.QImage(rgba.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
    # ensure data stays alive: deep-copy to an owned QImage
    qimg = qimg.copy()
    return QtGui.QPixmap.fromImage(qimg)

# def __np_to_qpixmap(img_array, fmt=QtGui.QImage.Format_Indexed8):
#     img = QtGui.QImage(
#         img_array.data.tobytes(),
#         img_array.shape[0],
#         img_array.shape[1],
#         img_array.strides[0],
#         fmt
#     )
#     return QtGui.QPixmap.fromImage(img)
def project(img, axis, invalid_val=np.nan):
    mask = img != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def surface_project(img):
    proj = project(img, 2, invalid_val=0.0)
    proj -= np.min(proj[np.nonzero(proj)])
    proj[proj < 0] = np.nan  # or -np.inf
    mask = ~np.isnan(proj).T
    proj *= 255.0 / np.nanmax(proj)
    proj = 255 - proj  # invert
    proj = proj.astype(np.uint8).T
    return mask, proj


def setup_mini_brain(atlas_base_name, mini_brain_scaling=(5, 5, 5)):  # TODO: scaling in prefs
    """
    Create a downsampled version of the Allen Brain Atlas for the mini brain widget

    Parameters
    ----------
    mini_brain_scaling : tuple(int, int, int)
        The scaling factors for the mini brain. Default is (5, 5, 5)

    Returns
    -------
    tuple(scale, downsampled_array)
    """
    atlas_path = os.path.join(Settings.atlas_folder, f'{atlas_base_name}_annotation.tif')
    arr = TIF.Source(atlas_path).array
    return mini_brain_scaling, sk_transform.downscale_local_mean(arr, mini_brain_scaling)


def get_current_res(app):
    screen = app.primaryScreen()
    size = np.array((screen.size().width(), screen.size().height()))

    hd_res = np.array((1920, 1080))
    four_k_res = np.array((3840, 2160))

    if (size < hd_res).any():
        res = 'sd'
    elif (size < four_k_res).any():
        res = 'hd'
    else:
        res = '4k'
    return res


def pseudo_random_rgb_array(n_samples):
    if n_samples == 0:
        return None
    hues = np.random.rand(n_samples)
    saturations = np.random.rand(n_samples) / 2 + 0.5
    values = np.random.rand(n_samples) / 2 + 0.5
    hsvs = np.vstack((hues, saturations, values))
    rgbs = np.apply_along_axis(hsv_to_rgb, 0, hsvs)
    return rgbs.T


def get_random_color():
    return QColor(*np.random.randint(0, 255, 3))


def get_pseudo_random_color(format='rgb'):
    """
    Return a pseudo random colour. The hue is random but
    The saturation and the value are kept in the upper half interval

    Returns
    -------

    """
    rand_gen = np.random.default_rng()
    hsv = (rand_gen.uniform(0, 1),
           rand_gen.uniform(0.5, 1),
           rand_gen.uniform(0.5, 1))
    if format == 'hsv':
        return hsv
    elif format == 'rgb':
        return hsv_to_rgb(hsv)
    elif format == 'qcolor':
        return QColor(*[int(c * 255) for c in hsv_to_rgb(hsv)])
    else:
        raise ValueError(f'Unknown format "{format}"')


def is_dark(color):
    """

    Parameters
    ----------
    color QColor

    Returns
    -------


    """
    return color.getHsl()[2] < 128
