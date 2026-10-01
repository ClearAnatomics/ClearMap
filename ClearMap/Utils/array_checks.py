"""
array_checks
============

Validation and coercion of numpy arrays before they cross into compiled (Cython) code.

The Cython kernels declare strict typed memoryviews (e.g. ``const uint8_t[:, :, :]``) and raise
cryptic ``Buffer dtype mismatch`` errors. The thin Python wrapper of each compiled module is the
place that decides what is accepted and how it is coerced; this module provides the shared
building blocks so that the wrappers stay short and consistent.

Conventions
-----------
* Nothing is ever modified in place and read-only inputs are fine, the kernels declare them ``const``.
* A copy is made only when the dtype has to change, never silently for data that is written to
  (see :func:`check_dtype`, which only validates and never converts sinks).
* Errors name the offending argument.
"""
import numpy as np

__all__ = ['check_dtype', 'bool_as_uint8', 'as_uint8_flags', 'as_index_array', 'as_dtype',
           'pad_to_ndim', 'scratch_parameters']


def _dtype_names(dtypes):
    return [np.dtype(d).name for d in dtypes]


def check_dtype(array, allowed, name='array', allow_bool=False):
    """Check that the dtype of ``array`` is one of ``allowed`` without converting it.

    Use this for arrays written by the kernel (sinks), where a silent conversion would write the
    result to a temporary copy.

    Arguments
    ---------
    array : array
      The array to check.
    allowed : sequence of dtype
      The accepted dtypes.
    name : str
      Name of the argument used in the error message.
    allow_bool : bool
      Also accept bool (for arrays that the caller views as uint8 afterwards).

    Returns
    -------
    array : array
      The input array.
    """
    dtype = np.dtype(array.dtype)
    if dtype in [np.dtype(d) for d in allowed] or (allow_bool and dtype == np.bool_):
        return array
    extra = ' or bool' if allow_bool else ''
    raise TypeError(f'{name} has dtype {dtype.name}, expected one of {_dtype_names(allowed)}{extra}!')


def bool_as_uint8(array):
    """View bool arrays as uint8 (same values, no copy), return any other array unchanged.

    Unlike :func:`as_uint8_flags` this never changes the values of numeric data, use it for
    data (sources, kernels, sinks) rather than for flags.
    """
    array = np.asarray(array)
    return array.view(np.uint8) if array.dtype == np.bool_ else array


def as_uint8_flags(array, name='array'):
    """Return ``array`` as a uint8 array of 0/1 flags (mask, structuring element, background, ...).

    * bool arrays are viewed as uint8, without copy (and, being a view, read-only stays read-only);
    * uint8 arrays are returned unchanged, so their values are used as given;
    * any other numeric dtype is converted with ``array != 0``.

    Note
    ----
    Never use ``array.view('uint8')`` for dtypes that are not 1 byte wide: it reinterprets the
    memory and changes the number of elements instead of converting the values.
    """
    array = np.asarray(array)
    if array.dtype == np.bool_:
        return array.view(np.uint8)
    if array.dtype == np.uint8:
        return array
    if array.dtype.kind in 'biufc':  # bool, signed/unsigned int, float, complex
        return (array != 0).view(np.uint8)
    raise TypeError(f'{name} has dtype {array.dtype.name} which cannot be interpreted as flags!')


def as_index_array(values, name='indices', dtype=np.intp, ndim=None):
    """Return ``values`` as an integer array of type ``dtype`` (``Py_ssize_t`` by default).

    Floats are rejected instead of being silently truncated.

    Arguments
    ---------
    values : array-like
      Integer valued indices or coordinates.
    name : str
      Name of the argument used in the error message.
    dtype : dtype
      Target integer dtype.
    ndim : int or None
      If given, the required number of dimensions.

    Returns
    -------
    indices : array
      The array itself if it already has the right dtype, a converted copy otherwise.
    """
    values = np.asarray(values)
    if values.dtype.kind not in 'iu':  # int (signed/unsigned)
        raise TypeError(f'{name} must contain integers, found dtype {values.dtype.name}!')
    if ndim is not None and values.ndim != ndim:
        raise ValueError(f'{name} must be {ndim:d} dimensional, found {values.ndim:d} dimensions!')
    return values.astype(dtype, copy=False)


def as_dtype(array, dtype, name='array', casting='same_kind'):
    """Return ``array`` with the given dtype (no copy if it already has it).

    Arguments
    ---------
    array : array-like
      Input array.
    dtype : dtype
      Target dtype.
    name : str
      Name of the argument used in the error message.
    casting : str
      numpy casting rule, ``'same_kind'`` refuses e.g. float -> int.
    """
    array = np.asarray(array)
    if not np.can_cast(array.dtype, dtype, casting=casting):
        raise TypeError(f'{name} has dtype {array.dtype.name} which cannot be converted '
                        f'to {np.dtype(dtype).name} ({casting} casting)!')
    return array.astype(dtype, copy=False)


def pad_to_ndim(array, ndim=3):
    """View of ``array`` with trailing singleton axes added to reach ``ndim`` dimensions (no copy)."""
    if array.ndim > ndim:
        raise ValueError(f'Cannot pad an array with {array.ndim:d} dimensions to {ndim:d}!')
    return array[(Ellipsis,) + (None,) * (ndim - array.ndim)]


def scratch_parameters(values, dtype):
    """Flat, *writable*, fresh copy of ``values`` (scalar, sequence or None) for kernel parameters.

    Some kernels use their parameter arrays as scratch space, hence they must be private writable
    copies and cannot be declared ``const`` in the Cython code.
    """
    return np.array(() if values is None else values, dtype=dtype).ravel()
