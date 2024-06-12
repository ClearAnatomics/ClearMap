"""
geometry_utils.py

This module contains utility functions for performing geometric calculations related to the
ClearMap vasculature analysis.
Generally 3D coordinates are represented as a 3-column array, with each row representing a
different point. For example, a set of 3 points in 3D space would be represented as:

        [[x1, y1, z1],
        [x2, y2, z2],
        [x3, y3, z3]]

Functions:
----------
angle_between_vectors(vectors_1, vectors_2, in_degrees=True):
    Compute the angle between two sets of vectors. By default, the angle is returned in degrees.

cartesian_to_polar(x, y, ceval=numexpr.evaluate):
    Transform from cartesian to polar coordinates. It takes in x and y coordinates and a backend to use for computation (either `eval` for pure Numpy or `numexpr.evaluate` for Numexpr).

f_min(X, p):
    Calculate the minimum distance from a set of points X to a plane defined by p. The plane is defined by a vector (the first three elements of `p`) and a scalar `p[3]`. The vector represents the normal to the plane, and `p[3]` is the distance from the origin to the plane along this normal. The function calculates the distance from each point in `X` to the plane, and then normalizes these distances by the length of the normal vector.
"""


import math

import numexpr
import numpy as np
from scipy.interpolate import LinearNDInterpolator


def angle_between_vectors(vectors_1, vectors_2, in_degrees=True):
    """
    Compute the angle between two sets of vectors.
    By default, the angle is returned in degrees.
    
    Parameters
    ----------
    vectors_1 : ndarray
       The first set of vectors
    vectors_2
         The second set of vectors
    in_degrees : bool
        Whether to return the angle in degrees or radians

    Returns
    -------

    """
    dot_product = np.sum(vectors_1 * vectors_2, axis=1).astype('float64')  # Because norms is upcast to float64
    norms = np.linalg.norm(vectors_1, axis=1) * np.linalg.norm(vectors_2, axis=1)
    cos_angle = np.abs(np.divide(dot_product, norms, out=np.full_like(dot_product, np.nan),
                                 where=norms != 0))  # Avoid division when one norms is zero
    angle2normal = np.arccos(np.clip(cos_angle, -1, 1))  # Clipping avoids error in arccos

    # Convert to degrees if necessary
    if in_degrees:
        angle2normal = np.degrees(angle2normal)
    return angle2normal


# noinspection PyUnusedLocal
def cartesian_to_polar(x, y, ceval=numexpr.evaluate):
    """
    Transform from cartesian to polar
    x:  ndarray coordinates
    y:  ndarray coordinates
    ceval: backend to use:
    - eval :  pure Numpy
    - numexpr.evaluate:  Numexpr

    Returns
    """
    if x.shape[0]:
        r = ceval('sqrt(x**2+y**2)')
        phi = ceval('arctan2(y,x)*180') / math.pi
        r_max = np.max(r)
    else:
        print('cart2sph, no orientation to compute')
        r = np.zeros(1)
        phi = np.zeros(1)
        r_max = 1
    return phi / 180, r / r_max


def f_min(X, p):
    """
    Calculate the minimum distance from a set of points X to a plane defined by p.

    Parameters
    ----------
    X : ndarray
        The coordinates of the points.
    p : ndarray
        The parameters defining the plane. The first three elements are the normal to the plane,
        and the last element is the distance from the origin to the plane along this normal.

    Returns
    -------
    ndarray
        The normalized distances from the points to the plane.
    """
    plane_xyz = p[:3]
    dist_from_origin = p[3]
    distance = (plane_xyz * X.T).sum(axis=1) + dist_from_origin
    return distance / np.linalg.norm(plane_xyz)


def compute_grid(coordinates, grid_n_pts=100):  # TODO: would be nice to have grid_n_pts be a tuple representing the number of points for each axis
    """
    Compute a grid of coordinates from the source coordinates to interpolate
    the vectors (e.g. flow vectors) onto

    Parameters
    ----------
    coordinates : np.array
        The coordinates of the edges
    grid_n_pts : int
        The number of points to use for each axis of the grid
        This determine the granularity of the grid

    Returns
    -------

    """
    n_axes = coordinates.shape[1]
    # Generate grid for each axis
    grid_axes = [np.linspace(0, np.max(coordinates[:, i]), grid_n_pts)
                 for i in range(n_axes)]
    # Create meshgrid and reshape
    grid = np.array(np.meshgrid(*grid_axes)).reshape((n_axes, grid_n_pts**n_axes)).T
    return grid


def interpolate_vectors(vectors, source_coordinates, interpolated_coordinates):
    """
    Interpolate the vectors from source_coordinates to interpolated_coordinates

    Parameters
    ----------
    vectors : np.array
        The vectors to interpolate. The shape should be (n_points, 3)
    source_coordinates : np.array
        The coordinates of the source points. The shape should be (n_points, 3)
    interpolated_coordinates : np.array
        The coordinates of the points to interpolate the vectors onto.
        The shape should be (n_points, 3)

    Returns
    -------

    """
    # TEST: we could probably use only one interpolator for all 3 axes
    NNDI_x = LinearNDInterpolator(source_coordinates, vectors[:, 0])
    NNDI_y = LinearNDInterpolator(source_coordinates, vectors[:, 1])
    NNDI_z = LinearNDInterpolator(source_coordinates, vectors[:, 2])

    flow_vectors_x = NNDI_x(interpolated_coordinates)
    flow_vectors_y = NNDI_y(interpolated_coordinates)
    flow_vectors_z = NNDI_z(interpolated_coordinates)

    # FIXME: check if we need to precise the stacking axis
    flow_vectors = np.stack([flow_vectors_x, flow_vectors_y, flow_vectors_z]).T
    return flow_vectors
