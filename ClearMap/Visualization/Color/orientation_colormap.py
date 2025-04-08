"""
Orientation Color Map
=====================

An accurate and intuitive orientation color map.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Christoph Kirst'


import numpy as np

WHITE = np.array([1, 1, 1])
BLACK = np.array([0, 0, 0])
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])
CYAN = np.array([0, 1, 1])

COLOR_Z = WHITE
COLOR_XY = (RED, BLUE)
COLOR_M = (CYAN, MAGENTA, GREEN, YELLOW)

# alternative coloring
# COLOR_Z = WHITE
# COLOR_XY = (RED, CYAN)
# COLOR_M = (BLUE, GREEN, MAGENTA, YELLOW)


def zero_to_one(x, s=0.1, e=0.5, k=1):
    """Scale x from 0 to one."""
    zo = np.zeros(x.shape)

    ids = x > e
    zo[ids] = 1

    ids = np.logical_and(s < x, x < e)
    zo[ids] = (0.5 * (np.sin(np.pi * ((x[ids] - s) / (e - s) - 0.5)) + 1)) ** k

    return zo


def deform_1(x, s=0.25, e=0.75, h=0.9, k=1):
    """Deform x for more uniform perception of color gradients."""
    xd = np.array(x)

    ids = x > s
    x_ids = x[ids]

    a = zero_to_one(x_ids, s=s, e=e, k=k)

    xd[ids] = (1 - a) * x_ids + a * ((1 - h) * (x_ids - s) / (1 - s) + h)

    return xd


def deform_xyz(x, y, z, s=0.6, e=0.8, h=0.95):
    """Deform xyz for more uniform perception of color gradients."""
    theta = np.arctan2(y, x)
    phi = (np.pi / 2 - np.arccos(z)) / (np.pi / 2)
    phi = deform_1(phi, s=s, e=e, h=h)
    phi = np.pi / 2 * (1 - phi)
    z = np.cos(phi)
    sin_phi = np.sin(phi)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)

    return x, y, z


def generate_color_squares(color_z=COLOR_Z, color_xy=COLOR_XY, color_m=COLOR_M):
    """Generate colors for each square from the color settings."""

    # colors for each square: 0,1,2,3 upper squares, 4,5,6,7 center squares
    color_squares = np.zeros((8, 2, 2, 3))
    for i in range(4):
        color_squares[i][1, 1] = color_z
        color_squares[i][0, 0] = color_xy[i % 2]
        color_squares[i][0, 1] = color_m[(i - 1) % 4]
        color_squares[i][1, 0] = color_m[i]
    for i in range(4, 8):
        k = i - 4
        color_squares[i][0, 0] = color_xy[k % 2]
        color_squares[i][1, 1] = color_xy[(k + 1) % 2]
        color_squares[i][0, 1] = color_m[k]
        color_squares[i][1, 0] = color_m[(k + 2) % 4]

    return color_squares


def orientation_color(xyz, normalize=False, deform=None, colors=(COLOR_Z, COLOR_XY, COLOR_M), verbose=False):
    """Orientation color for the given orientation vectors."""
    color_squares = generate_color_squares(color_z=colors[0], color_xy=colors[1], color_m=colors[2])

    x, y, z = np.array(xyz[..., 0]), np.array(xyz[..., 1]), np.array(xyz[..., 2])

    # align to positive z
    invert = z < 0
    x[invert] *= -1
    y[invert] *= -1
    z[invert] *= -1

    # normalize to length 1
    if normalize:
        norm = np.sqrt(x * x + y * y + z * z)
        x = x / norm
        y = y / norm
        z = z / norm

    # deform for more uniform color perception
    if deform is True:
        deform = deform_xyz

    if deform:
        if verbose:
            print('deforming!')
        x, y, z = deform(x, y, z)

    theta = np.arctan2(y, x) / (2 * np.pi) + 0.5
    if np.any(np.isnan(theta)):
        if verbose:
            print('encountered nans!')
        ids = np.isnan(theta)
        print(x[ids], y[ids])

    theta = np.mod(theta, 1)
    sin_phi = 1 - z * z
    if verbose and np.any(sin_phi < 0):
        print('invalid 1-z2: ', np.sum(sin_phi))
    sin_phi[sin_phi < 0] = 0
    sin_phi = np.sqrt(sin_phi)

    # rotate to first quadrant
    q = np.array(4 * theta, dtype=int)
    theta_t = (4 * theta - q)
    octant = np.array(theta_t >= 0.5, dtype=int)

    theta_t = theta_t * np.pi / 2
    x_t = np.cos(theta_t) * sin_phi
    y_t = np.sin(theta_t) * sin_phi

    # interpolation values in center squares
    ds = 2 * np.sqrt(1 - y_t * z + x_t * (y_t + z))
    s = 0.5 - 3 / np.pi * (np.arcsin((x_t + z) / ds) + np.arcsin((z - y_t) / ds))

    dt = 2 * np.sqrt(1 - x_t * z + (x_t + z) * y_t)
    t = -1 + 3 / np.pi * (np.arcsin((y_t + z) / dt) + np.arccos((x_t - z) / dt))

    # decide upper or center squares
    st = np.array([s, 1 - t])
    upper = np.take_along_axis(st, octant[np.newaxis, :], axis=0)[0] < 0

    # upper 2nd quadrant rotate by -Pi/2
    upper_octant2 = np.logical_and(upper, octant == 1)

    theta_t[upper_octant2] -= np.pi / 2
    x_t[upper_octant2] = np.cos(theta_t[upper_octant2]) * sin_phi[upper_octant2]
    y_t[upper_octant2] = np.sin(theta_t[upper_octant2]) * sin_phi[upper_octant2]

    # calculate interpolation values in upper squares
    x_t_u = x_t[upper]
    y_t_u = y_t[upper]
    z_u = z[upper]

    dv = 2 * np.sqrt(1 - y_t_u * z_u + x_t_u * (y_t_u + z_u))
    v = 0.5 - 3 / np.pi * (np.arcsin((y_t_u - z_u) / dv) + np.arcsin((x_t_u + y_t_u) / dv))

    du = 2 * np.sqrt(1 - x_t_u * y_t_u + (x_t_u + y_t_u) * z_u)
    u = -1 + 3 / np.pi * (np.arcsin((y_t_u + z_u) / du) + np.arccos((x_t_u - y_t_u) / du))

    # calculate color square id
    c_id = q + (1 - upper) * 4
    c_id[upper_octant2] = np.mod(c_id[upper_octant2] + 1, 4)
    # return c_id

    # interpolate colors
    a = s
    b = t
    a[upper] = u
    b[upper] = v
    a = a[..., np.newaxis]
    b = b[..., np.newaxis]

    colors = color_squares[c_id]  # shape (...,2,2,3)
    c = colors[..., 0, 0, :] * (1 - a) * (1 - b) + colors[..., 1, 0, :] * a * (1 - b) + colors[..., 0, 1, :] * (
            1 - a) * b + colors[..., 1, 1, :] * a * b

    # numeric rounding will be slightly off
    c[c > 1] = 1
    c[c < 0] = 0

    return c


_colormap_cache = None


def orientation_colormap_cache(n=200, **kwargs):
    global _colormap_cache

    import os
    cache_file = os.path.join(os.path.split(__file__)[0], 'omap.npy')

    if _colormap_cache is None:
        if os.path.isfile(cache_file):
            _colormap_cache = np.load(cache_file)

    if _colormap_cache is not None and n == _colormap_cache.shape[0]:
        return _colormap_cache

    x_lin, y_lin, z_lin = [np.linspace(-1, 1, n + 1) for i in range(3)]
    x, y, z = np.meshgrid(x_lin, y_lin, z_lin)

    # zero
    zero = np.logical_and(x == 0, np.logical_and(y == 0, z == 0))
    x[zero] = 1

    xyz = np.array([x, y, z]).transpose((1, 2, 3, 0))

    col = orientation_color(xyz, normalize=True, **kwargs)

    _colormap_cache = col

    np.save(cache_file, _colormap_cache)

    return col


def orientation_color_cached(xyz, cache=None):
    """Fast no interpolation between colors."""
    if cache is None:
        cache = _colormap_cache
        if cache is None:
            cache = orientation_colormap_cache()

    nx, ny, nz = np.array(cache.shape[:3]) - 1

    ix = np.array(xyz[..., 0] * (nx / 2) + nx / 2, dtype=int)
    iy = np.array(xyz[..., 1] * (ny / 2) + ny / 2, dtype=int)
    iz = np.array(xyz[..., 2] * (nz / 2) + nz / 2, dtype=int)

    colors = cache[ix, iy, iz]

    return colors


# util
#
# def orientation_vectors(shape, max_radius=None, min_radius=None, reshape=True, orientation=True, rescale=True):
#     """Generate vectors to all pixels in a box neighborhood."""
#
#     # center
#     center = np.array([s // 2 - 1 if s % 2 == 0 else s // 2 for s in shape])
#     ranges = [np.arange(-c, c + 1) for c in center]
#
#     if max_radius == 'sphere':
#         max_radius = np.min(center)
#
#     # all vectors
#     vectors = np.array(np.meshgrid(*ranges, indexing='ij'), dtype=float)
#
#     if reshape:
#         vectors = vectors.reshape(len(shape), -1).T
#     else:
#         vectors = vectors.transpose(tuple(d for d in range(1, len(shape) + 1)) + (0,))
#
#     # length weights # is this a good measure?
#     lengths = np.linalg.norm(vectors, axis=-1)
#
#     # valid
#     if min_radius is None:
#         min_radius = 0
#     valid = lengths > min_radius
#     if max_radius is not None:
#         valid = np.logical_and(valid, lengths <= max_radius)
#
#     # orientation
#     if orientation:
#         # invert z < 0
#         vectors[vectors[..., 2] < 0, :] *= -1
#         # invert z = 0, y < 0
#         ids0 = vectors[..., 2] == 0
#         vectors[np.logical_and(ids0, vectors[..., 1] < 0), :] *= -1
#         # invert z = 0, y = 0, x < 0
#         ids0 = np.logical_and(ids0, vectors[..., 1] == 0)
#         vectors[np.logical_and(ids0, vectors[..., 0] < 0), :] *= -1
#
#     # rescale vectors
#     if rescale is not None:
#         if rescale:
#             rescale = lambda x: x
#         vectors[valid] = (vectors[valid] / rescale(lengths[valid][:, np.newaxis]))
#
#     return vectors, lengths, valid
