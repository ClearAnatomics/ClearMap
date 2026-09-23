import pathlib
import warnings

import numpy as np

from ClearMap.IO import io_ops
from ClearMap.IO.source import source_modes, Source as source_mod, geometry_utils
from ClearMap.IO.source.backends import sma_backend, npy_backend
from ClearMap.IO.FileUtils import abspath, normalize_location_spec
from ClearMap.IO.io_ops import peek_into
from ClearMap.IO.dispatch import location_to_module, as_source
from ClearMap.Utils import tag_expression as te
from ClearMap.Utils.exceptions import (ClearMapException, AssetNotFoundError, SourceNotFoundError,
                                       ClearMapRuntimeError, ClearMapValueError, IncompatibleSource)


def initialize(source_=None, shape_=None, dtype_=None,
               order_=None, location_=None, memory_=None, like=None, hint=None, **kwargs):
    """
    Initialize (open to edit or create if missing) a source with specified properties.

    Note
    ----
    The source is created on disk or in memory if it does not exist so processes
    can start writing into it.

    .. WARNING::
        In the case of table type sources, they can be written but not initialized

    Parameters
    ----------
    source_ : str, array, Source class
        The source to write data to.
    shape_ : tuple or None
        The desired shape of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid shape shapes are tested to match.
    dtype_ : type, str or None
        The desired dtype of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid dtype the types are tested to match.
    order_ : 'C', 'F' or None
        The desired order of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid order the orders are tested to match.
    location_ : str or None
        The desired location of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid location the locations need to match.
    memory_ : 'shared' or None
        The memory type of the source. If 'shared' a shared array is created.
    like : str, array or Source class
        Infer the source parameter from this source.
    hint : str, array or Source class
        If parameters for source creation are missing use the ones from this
        hint source.

    Returns
    -------
    source: Source class
        The initialized source.
    """
    source_ = normalize_location_spec(source_)

    if isinstance(source_, (str, te.Expression)):  # If the source is a path (location)
        location_ = source_
        source_ = None

    if like is not None:
        shape_, dtype_, order_ = _from_like(like, shape_, dtype_, order_)

    if source_ is None:
        if location_ is None:  # No source and no path: array in memory, regular or shared
            shape_, dtype_, order_ = _from_hint(hint, shape_, dtype_, order_)
            if memory_ in ['shared', 'automatic']:
                return sma_backend.create(shape=shape_, dtype=dtype_, order=order_, **kwargs)
            else:
                return npy_backend.create(shape=shape_, dtype=dtype_, order=order_, **kwargs)
        else:  # No source but a path
            # Before try because missing module != missing file so shouldn't fall through to creation.
            mod = location_to_module(location_)
            try:  # First, attempt to open the existing source in 'edit' mode
                if hasattr(mod, 'edit'):
                    source_ = io_ops.edit(location_)
                else:
                    source_ = as_source(location_, mode=source_modes.DEFAULT_EDIT_MODE)
            except AssetNotFoundError:  # workspace-level failure is not a reason to create a file
                raise  # WARNING: must stay before FileNotFoundError
            except (SourceNotFoundError, FileNotFoundError): # TODO: narrow to SourceNotFoundError once every IO module raises it.
                source_ = None
            # Anything else (IncompleteSourceSpecError, corrupt header, shape mismatch) != missing file -> propagates

            if source_ is None:  # Opening existing failed -> creation path
                if isinstance(location_, str) and pathlib.Path(location_).is_file():  # Just belt and braces
                    raise ClearMapRuntimeError(f'{location_} exists but could not be opened for editing; refusing overwrite.')
                shape_, dtype_, order_ = _from_hint(hint, shape_, dtype_, order_)
                try:
                    return io_ops.create(location=location_, shape=shape_, dtype=dtype_, order=order_, **kwargs)
                except ClearMapException:  # Catch and raise specific to avoid generic path
                    raise
                except Exception as error:
                    raise ClearMapRuntimeError(f'Cannot initialize source for location {location_}') from error

    if isinstance(source_, np.ndarray):
        source_ = as_source(source_)

    # ######## Exception handling ##############
    if not isinstance(source_, source_mod.Source):
        raise ClearMapValueError(f'Source specification {source_} not a valid location, array or Source class!')

    current_vars = locals()
    for attr in ('shape_', 'dtype_', 'order_'):
        base_attr = attr[:-1]
        if current_vars.get(attr) is not None and current_vars[attr] != getattr(source_, base_attr, None):
            raise IncompatibleSource(source_, attr, current_vars)

    if location_ is not None and abspath(location_) != abspath(source_.location):
        raise IncompatibleSource(source_, 'location', current_vars)
    if memory_ == 'shared' and not sma_backend.is_shared(source_):
        raise ClearMapValueError(f'Incompatible memory type, the source {source_} is not shared!')

    return source_


def _from_like(like, shape, dtype, order):
    """Resolve geometry from a template (``like``) source.

    Parameters
    ----------
    like : object or None
        Source, source location, or other object from which missing geometry
        properties can be inferred.
    shape : tuple-like or None
        Requested shape. If ``None``, infer it from ``like``.
    dtype : dtype-like or None
        Requested data type. If ``None``, infer it from ``like``.
    order : {'C', 'F'} or None
        Requested memory order. If ``None``, infer it from ``like``.

    Returns
    -------
    shape : tuple-like or None
        Explicitly supplied or inferred shape.
    dtype : numpy.dtype or None
        Explicitly supplied or inferred data type.
    order : {'C', 'F'} or None
        Explicitly supplied or inferred memory order.

    Notes
    -----
    Explicit values take precedence over values inferred from ``like``.
    The source is opened read-only for metadata inspection and any temporary
    source opened for that purpose is closed before returning.
    """
    if like is None:
        return shape, dtype, order
    else:
        with peek_into(like) as source:
            return geometry_utils.resolve_geometry(shape=shape, dtype=dtype, order_=order, like=source)


def _from_hint(hint, shape, dtype, order):
    """Best-effort geometry inference from an initialization hint.
    Contrary to _from_like, this does not Raise

    Parameters
    ----------
    hint : object or None
        Source, source location, or other object from which missing geometry
        properties should be inferred.
    shape : tuple-like or None
        Requested shape, or ``None`` to infer it from ``hint``.
    dtype : dtype-like or None
        Requested data type, or ``None`` to infer it from ``hint``.
    order : {'C', 'F'} or None
        Requested memory order, or ``None`` to infer it from ``hint``.

    Returns
    -------
    shape : tuple-like or None
        Explicitly supplied or inferred shape.
    dtype : numpy.dtype or None
        Explicitly supplied or inferred data type.
    order : {'C', 'F'} or None
        Explicitly supplied or inferred memory order.

    Notes
    -----
    This is a non-raising wrapper around :func:`_from_like`. If inference
    fails for any reason, a warning is emitted and the original values are
    returned unchanged.

    Explicit values take precedence over values inferred from ``hint``.
    """
    try:
        return _from_like(hint, shape, dtype, order)
    except Exception as err:
        warnings.warn(f'Cannot infer shape, dtype and order from hint {hint}, keeping defaults; {err}', stacklevel=2)
        return shape, dtype, order

# FIXME: unused
def initialize_buffer(source_, shape=None, dtype=None,
                      order=None, location=None, memory=None, like=None, **kwargs):
    """
    Initialize a buffer with specific properties.

    Parameters
    ----------
    source_ : str, array, Source class
        The source to write data to.
    shape : tuple or None
        The desired shape of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid shape shapes are tested to match.
    dtype : type, str or None
        The desired dtype of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid dtype the types are tested to match.
    order : 'C', 'F' or None
        The desired order of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid order the orders are tested to match.
    location : str or None
        The desired location of the source.
        If None, inferred from existing file or from the like parameter.
        If not None and source has a valid location the locations need to match.
    memory : 'shared' or None
        The memory type of the source. If 'shared' a shared array is created.
    like : str, array or Source class
        Infer the source parameter from this source.

    Returns
    -------
    buffer : array
        The initialized buffer to use tih e.g. cython.

    Note
    ----
    The buffer is created if it does not exist.
    """
    source_ = initialize(source_, shape_=shape, dtype_=dtype, order_=order, location_=location, memory_=memory, **kwargs)
    return source_.as_buffer()
