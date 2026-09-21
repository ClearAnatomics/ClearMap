from ClearMap.IO.io_ops import open_ro


def ndim(source_):
    """
    Returns number of dimensions of a source.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    ndim : int
        The number of dimensions in the source.
    """
    source_ = open_ro(source_)
    return source_.ndim


def shape(source_):
    """
    Returns shape of a source.

    Parameters
    ----------
    source_: str, array or Source
       The source specification.

    Returns
    -------
    shape : tuple of ints
       The shape of the source.
    """
    source_ = open_ro(source_)
    return source_.shape


def size(source_):
    """
    Returns size of a source.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    size : int
        The size of the source.
    """
    source_ = open_ro(source_)
    return source_.size


def dtype(source_):
    """
    Returns dtype of a source.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    dtype : dtype
        The data type of the source.
    """
    source_ = open_ro(source_)
    return source_.dtype


def order(source_):
    """
    Returns order of a source.

    Parameters
    ----------
    source_ : str, array or Source
        The source specification.

    Returns
    -------
    order : 'C', 'F', or None
        The order of the source data items.
    """
    source_ = open_ro(source_)
    return source_.order


def element_strides(source_):
    """
    Returns the strides of the data array of a source.

    Parameters
    ----------
    source_ : str, array, dtype or Source
        The source specification.

    Returns
    -------
    strides : tuple of int
        The strides of the source.
    """
    try:
        source_ = open_ro(source_)
        strides = source_.element_strides
    except Exception as e:
        raise ValueError(f'Cannot determine the strides for the source!; {e}')

    return strides
