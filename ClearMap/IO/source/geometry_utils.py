import numpy as np

from ClearMap.Utils.Formatting import ensure
from ClearMap.Utils.exceptions import ClearMapValueError


def _normalise_order(value):
    value = ensure(value, str).upper()
    if value not in ('C', 'F'):
        raise ClearMapValueError('Invalid order.', value=value, expected=('C', 'F'))
    return value


_NORMALISERS = {'shape': tuple, 'dtype': np.dtype, 'order': _normalise_order}


def order(array):
    """Returns the contiguous order of an array.

    Arguments
    ---------
    array : ndarray or Source

    Returns
    -------
    order : 'C', 'F', None
        None if the array is not contiguous, or if *array* is not something whose order
        can be determined. Note that for shapes with at most one axis longer than 1 both
        orders hold and 'C' is returned arbitrarily; use ``order_is_ambiguous`` before
        treating a mismatch as meaningful.
    """
    if isinstance(array, np.ndarray):
        if array.flags['C_CONTIGUOUS']:
            return 'C'
        elif array.flags['F_CONTIGUOUS']:
            return 'F'
        else:
            return None
    else:
        value = getattr(array, 'order', None)
        return _normalise_order(value) if value is not None else None


def order_is_ambiguous(shape):
    """Whether C and F order are indistinguishable for this shape."""
    return sum(n > 1 for n in tuple(shape)) <= 1


def properties_match(source, **properties):
    """True if *source* matches every recognised property given.

    Unrecognised keys and ``None`` values are ignored, so a kwargs bag can be passed
    straight in. Values are normalised before comparison, so ``[10] == (10,)`` and
    ``'f4' == float32``. ``order`` is skipped where the shape makes both orders equally
    true, so a length-1 or 1-d source is never reported as mismatched on order alone.
    """
    for key, normalise in _NORMALISERS.items():
        requested = properties.get(key)
        if requested is None:  # not asked about
            continue
        if key == 'order':
            if order_is_ambiguous(source.shape):
                continue
            actual = order(source)
            if actual is None:  # non-contiguous: no order to compare against
                raise ClearMapValueError(f'Cannot compare order of non-contiguous {source!r}.',
                                         value=source, expected='a contiguous source')
        else:
            actual = normalise(getattr(source, key))
        if normalise(requested) != actual:
            return False
    return True


def resolve_geometry(shape=None, dtype=None, order_=None, *,
                     array=None, like=None, default_order=None):
    """Resolve normalised shape, dtype and order from explicit values and templates."""
    if array is not None:
        if shape is None:
            shape = getattr(array, 'shape', None)
        if dtype is None:
            dtype = getattr(array, 'dtype', None)
        if order_ is None:
            order_ = order(array)

    if like is not None:
        if shape is None:
            shape = getattr(like, 'shape', None)
        if dtype is None:
            dtype = getattr(like, 'dtype', None)
        if order_ is None:
            order_ = order(like)

    if order_ is None:
        order_ = default_order

    if shape is not None:
        shape = tuple(shape)
    if dtype is not None:
        dtype = np.dtype(dtype)
    if order_ is not None:
        order_ = _normalise_order(order_)

    return shape, dtype, order_
