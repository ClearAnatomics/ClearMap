import numpy as np


def format_orientation(orientation, inverse=False, default=None):
    """Convert orientation to standard format.

    Arguments
    ---------
    orientation : tuple, str or None
      The orientation specification.
    inverse : bool
       If True, invert orientation.
    default : object
       The default value if orientation is None;

    Returns
    -------
    orientation : tuple of ints
      The orientation sequence.

    See Also
    --------
    :mod:`~ClearMap.Alignment.orientation`
    """
    if orientation is None:
        return default

    # convert named representations
    if orientation == 'left':
        # orientation = (1,2,3);
        orientation = None
    elif orientation == 'right':
        orientation = (-1, 2, 3)

    if orientation is not None and len(orientation) != 3:
        raise ValueError(
            "orientation should be 'left', 'right' or a tuple of 3 signed integers from 1 to 3, found %r" % (
                orientation,))

    if inverse:
        orientation = invert_orientation(orientation)

    return orientation


def invert_orientation(orientation):
    """Returns the inverse orientation taking axis inversions into account.

    Arguments
    ---------
    orientation : tuple, str or None
      The orientation specification.

    Returns
    -------
    orientation : tuple
      The inverse orientation sequence.

    See Also
    --------
    :mod:`~ClearMap.Alignment.orientation`
    """
    orientation = format_orientation(orientation)

    if orientation is None:
        return None

    # orientation is defined as permuting the axes and then inverrting the axis
    inverse = list(orientation)
    for i, o in enumerate(orientation):
        if o < 0:
            inverse[int(abs(o) - 1)] = -(i + 1)
        else:
            inverse[int(abs(o) - 1)] = (i + 1)

    return tuple(inverse)


def orientation_to_transposition(orientation, inverse=False):
    """Extracts the transposition permutation from an orientation.

    Arguments
    ---------
    orientation : tuple or str
      The orientation specification.
    inverse : bool
      If True, return inverse permutation.

    Returns
    -------
    permutation : tuple of ints
      The permutation sequence.

    See Also
    --------
    :mod:`~ClearMap.Alignment.orientation`
    """
    orientation = format_orientation(orientation, inverse=inverse)
    if orientation is None:
        return 0, 1, 2
    else:
        return tuple(int(abs(i)) - 1 for i in orientation)


def orient_resolution(resolution, orientation, inverse=False):
    """Permutes a resolution tuple according to the given orientation.

    Arguments
    ---------
      resolution : tuple
        The resolution specification.
      orientation : tuple or str
        The orientation specification.
      inverse : bool
        If True, invert the orientation.

    Returns
    -------
      resolution : tuple
        The re-oriented resolution sequence.

    See Also
    --------
    :mod:`~ClearMap.Alignment.orientation`
    """
    if orientation is None:
        return resolution

    axes = orientation_to_transposition(orientation, inverse=inverse)
    return tuple(resolution[a] for a in axes)


def orient_shape(shape, orientation, inverse=False):
    """Permutes a shape according to the given orientation.

    Arguments
    ---------
    shape : tuple
      The shape specification.
    orientation : tuple or str
      The orientation specification.
    inverse : bool
      If True, invert the orientation.

    Returns
    -------
    shape : tuple
      The oriented shape tuple.

    See Also
    --------
    :mod:`~ClearMap.Alignment.orientation`
    """
    return orient_resolution(shape, orientation, inverse=inverse)


def orient(data, orientation, inverse=False):
    """
    Orients a data array according to the given orientation.

    Arguments
    ---------
    data : array or Source
     The data to orient.
    orientation : tuple or str
     The orientation specification.
    inverse : bool
     If True, invert the orientation.

    Returns
    -------
    oriented : array
     The oriented data array.

    See Also
    --------
    :mod:`~ClearMap.Alignment.orientation`
    """
    orientation = format_orientation(orientation)

    oriented = data

    if orientation is not None:
        # reverse
        reverse = np.any([o < 0 for o in orientation])

        if inverse and reverse:
            slicing = tuple(slice(None, None, -1) if o < 0 else slice(None) for o in orientation)
            oriented = oriented[slicing]

        # re-orient
        oriented = oriented.transpose(orientation_to_transposition(orientation, inverse=inverse))

        # reverse
        if not inverse and reverse:
            slicing = tuple(slice(None, None, -1) if o < 0 else slice(None) for o in orientation)
            oriented = oriented[slicing]

    return oriented


def orient_points(points, orientation, shape=None, inverse=False):
    """Orients an array of coordinates according to the given orientation.

       Arguments
       ---------
       points : array or Source
         The data points to orient, as nxd array where d is dimension.
       orientation : tuple or str
         The orientation specification.
       shape : tuple
         The shape of the data array before reorientation needed in case some axes are reversed.
       inverse : bool
         If True, invert the orientation.

       Returns
       -------
       oriented : array
         The oriented data array.

       See Also
       --------
       :mod:`ClearMap.Alignment.orientation`
       """
    # reorient points
    orientation = format_orientation(orientation)

    oriented = points

    if orientation is not None:
        # reverse
        reverse = np.any([o < 0 for o in orientation])
        if reverse and shape is None:
            raise ValueError('Cannot invert orientation without data shape.')

        if reverse and inverse:
            shape_oriented = orient_shape(shape, orientation)
            for d, o in enumerate(orientation):
                if o < 0:
                    oriented[..., d] = shape_oriented[d] - 1 - oriented[..., d]

        # permute
        oriented = oriented[..., orientation_to_transposition(orientation, inverse=inverse)]

        if reverse and not inverse:
            shape_oriented = orient_shape(shape, orientation)
            for d, o in enumerate(orientation):
                if o < 0:
                    oriented[..., d] = shape_oriented[d] - 1 - oriented[..., d]

    return oriented
