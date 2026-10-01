# -*- coding: utf-8 -*-
"""
StructureElement
================

Routines to generate structure elements for filters.

The public entry point is :func:`structure_element`. The shape of an element can be given as
an integer (combined with ``ndim``) or as any 1D sequence of integers
(tuple, list, range, 1D array, ...).
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


import numpy as np


def _normalize_shape(shape, ndim=None):
    """Convert a shape specification to a 1D integer array.

    Arguments
    ---------
    shape : int or iterable of int
        Size of the element along each axis. A scalar is treated as a 1D shape of that size.
        Any 1D iterable of integers is accepted (tuple, list, range, 1D array, ...).
    ndim : int or None
        If not None, the shape is forced to this number of dimensions: a scalar (or a shorter shape)
        is repeated cyclically, a longer shape is truncated.

    Returns
    -------
    shape : 1D array of int
        Validated shape.

    Raises
    ------
    ValueError
        If the shape is empty, has more than one dimension (e.g. an already built element was passed
        instead of a shape) or contains sizes smaller than 1.
    TypeError
        If the sizes are not integers.
    """
    shape = np.asarray(shape)
    if shape.ndim > 1:
        raise ValueError(f'The shape must be an int or a 1D sequence of ints, got an array of shape '
                         f'{shape.shape!r}. If you already have a structure element, use it directly '
                         f'instead of calling structure_element.')
    shape = shape.reshape(-1)  # scalar -> (1,)
    if shape.size == 0:
        raise ValueError('The shape of a structure element cannot be empty!')
    if not np.issubdtype(shape.dtype, np.integer):
        raise TypeError(f'The shape of a structure element must contain integers, got dtype {shape.dtype}!')
    if (shape < 1).any():
        raise ValueError(f'The shape of a structure element must be strictly positive, got {shape.tolist()!r}!')
    shape = shape.astype(int)
    if ndim is not None:
        shape = np.resize(shape, int(ndim))  # cyclic repeat or truncation
    return shape


def disk(shape=(3, 3)):
    """Disk (or ball in 3D) structuring element, as a boolean array."""
    return sphere(shape=shape) > 0


def sphere(shape=(3, 3)):
    """Spherical structuring element with weights decreasing from the center.

    The weights are normalised to sum to 1. Only the non-zero support is used by the rank filters.
    """
    shape = _normalize_shape(shape)
    offsets = structure_element_offsets(shape)
    mesh = [range(-o[0], o[1]) for o in offsets]
    mesh = np.array(np.meshgrid(*mesh, indexing='ij'), dtype=float)

    add = ((shape + 1) % 2) / 2.0
    nrm = np.max(offsets, axis=1)
    for d in range(len(add)):
        mesh[d] = (mesh[d] + add[d]) / nrm[d]

    r = 1 - np.sum(mesh * mesh, axis=0)
    r[r < 0] = 0
    r /= r.sum()
    return r


def cube(shape=(3, 3)):
    """Cube (or rectangle) structuring element, as a boolean array."""
    shape = _normalize_shape(shape)
    return np.ones(shape, dtype=bool)


_FORMS = {
    'disk': disk, 'd': disk,
    'sphere': sphere, 's': sphere,
    'cube': cube, 'c': cube, 'rectangle': cube, 'r': cube,
}


def structure_element(shape=(3, 3), form='Disk', ndim=None):
    """Creates a structuring element of a given form and shape.

    Arguments
    ---------
    shape : int or iterable of int
        Shape of the structure element (tuple, list, 1D array, ...).
        A single int is repeated along ``ndim`` axes (a single axis if ``ndim`` is None).
    form : str
        Structure element type (case insensitive):
        'disk' ('d'), 'sphere' ('s'), 'cube' ('c', 'rectangle', 'r').
    ndim : int or None
        Number of dimensions of the structuring element. If None, it is the length of ``shape``.
        Otherwise ``shape`` is repeated cyclically or truncated to match.

    Returns
    -------
    element : array
        The structure element: boolean for 'disk' and 'cube', float weights for 'sphere'.

    Raises
    ------
    ValueError
        If the form is unknown or the shape is invalid.
    """
    try:
        builder = _FORMS[str(form).lower()]
    except KeyError:
        raise ValueError(f'Form {form!r} for structuring element not valid, '
                         f'expected one of {sorted(_FORMS)}!') from None
    return builder(shape=_normalize_shape(shape, ndim=ndim))


def structure_element_offsets(shape):
    """Calculates offsets to center for a structural element given its shape.

    Arguments
    ---------
    shape : int or iterable of int
        Shape of the structure element.

    Returns
    -------
    offsets : array
        Offsets to center taking care of even/odd number of elements.
        One row per axis: (elements before the center, elements from the center on).
    """
    shape = _normalize_shape(shape)
    off = shape // 2
    return np.array([off, shape - off]).T


########################################################################################################################
# Tests
########################################################################################################################

def _test():
    import ClearMap.ImageProcessing.Filter.StructureElement as se

    from importlib import reload
    reload(se)

    element = se.sphere((150, 50, 10))

    import ClearMap.Visualization.Plot3d as p3d
    p3d.plot(element)
