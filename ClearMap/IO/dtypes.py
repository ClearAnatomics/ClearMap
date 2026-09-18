import pathlib

import numpy as np

from ClearMap.IO import Source as source_mod
from ClearMap.IO.IO import dtype


def get_info(d_type):
    """
    Get the numpy info object for a data type. (automatically determines if integer or float)

    Parameters
    ----------
    d_type: dtype
        The data type to get the info for.

    Returns
    -------
    info: numpy info object
        The info object for the data type.
    """
    try:
        return np.iinfo(d_type)
    except ValueError:
        return np.finfo(d_type)


def get_value(source_, value_type):  # REFACTOR: should be moved to io_utils or Source module
    """
    Get the minimal or maximal value of a source data type.

    Parameters
    ----------
    source_: str, array, dtype or Source
        The source specification.
    value_type: str
        The value type to get, either 'min' or 'max'.

    Returns
    -------
    value: number
        The value of the data type.
    """
    if isinstance(source_, pathlib.Path):
        source_ = str(source_)

    if value_type not in ['min', 'max']:
        raise ValueError(f'Unknown value type {value_type}, accepted Parameters are "min" and "max"!')

    if isinstance(source_, (source_mod.Source, np.ndarray)):
        source_ = source_.dtype

    if isinstance(source_, str):
        try:
            source_ = np.dtype(source_)
        except TypeError:
            pass

    if not isinstance(source_, (type, np.dtype)):
        source_ = dtype(source_)

    try:
        info = get_info(source_)
        return getattr(info, value_type)
    except ValueError as e:
        raise ValueError(f'Cannot determine the {value_type} value for the type {source_}!; {e}')


def min_value(source_):
    """
    Returns the minimal value of a source data type.

    Parameters
    ----------
    source_ : str, array, dtype or Source
        The source specification.

    Returns
    -------
    min_value : number
        The minimal value for the data type of the source
    """
    return get_value(source_, 'min')


def max_value(source_):
    """
    Returns the maximal value of a source data type.

    Parameters
    ----------
    source_ : str, array, dtype or Source
        The source specification.

    Returns
    -------
    max_value : number
       The maximal value for the data type of the source
    """
    max_value = get_value(source_, 'max')
    return max_value
