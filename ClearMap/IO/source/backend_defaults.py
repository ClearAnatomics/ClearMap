"""Default implementations of the backend module API.

Every format backend in :mod:`ClearMap.IO` exposes ``create``, ``open_ro``,
``read`` and ``write`` at module level. Backends that do not support one of
these import the corresponding default from here, so that the failure is a
``ClearMapNotImplementedError`` with a consistent message rather than an
``AttributeError`` from the dispatch layer.

Note
----
This module imports nothing from the source class hierarchy, by design.
"""
from ClearMap.Utils.exceptions import ClearMapNotImplementedError

def create(location=None, shape=None, dtype=None, order=None, mode=None,
           array=None, as_source=True, **kwargs):
    """Create a source.

    This generic implementation exists as the default operation for backend
    modules that do not support source creation.
    """
    raise ClearMapNotImplementedError('Creating sources is not implemented by this backend.',
                                      operation='create')


def open_ro(source_, **kwargs):
    """Open a source read-only.

    Backends that cannot provide read-only access should inherit this default
    implementation.
    """
    raise ClearMapNotImplementedError('Opening sources read-only is not implemented by this backend.',
                                      operation='open_ro')


def read(source_, slicing=None, **kwargs):
    """Read data from a source.

    Backends that do not support reading should inherit this default
    implementation.
    """
    raise ClearMapNotImplementedError('Reading sources is not implemented by this backend.',
                                      operation='read')


def write(sink, data=None, slicing=None, overwrite=False, **kwargs):
    """Write data to a source.

    Backends that do not support writing should inherit this default
    implementation.
    """
    raise ClearMapNotImplementedError('Writing sources is not implemented by this backend.',
                                      operation='write')


def unsupported(operation, backend):
    """Build a stub that names its backend.

    Use where the generic message is too vague, e.g.::

        write = unsupported('write', backend='NRRD')
    """
    def _unsupported(*args, **kwargs):
        raise ClearMapNotImplementedError(
            f'{backend} does not support {operation}.', operation=operation, backend=backend)
    _unsupported.__name__ = operation
    _unsupported.__doc__ = f'Not supported by {backend}.'
    return _unsupported
