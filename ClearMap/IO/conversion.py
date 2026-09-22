import functools
import math
import multiprocessing as mp
import pathlib

from ClearMap.IO import FileUtils as fu, io_ops
from ClearMap.IO.io_ops import open_ro, write
from ClearMap.IO.dispatch import source_to_module
from ClearMap.ParallelProcessing import ParallelTraceback as ptb
from ClearMap.Utils import Timer as tmr
from ClearMap.Utils.utilities import CancelableProcessPoolExecutor


def convert(source_, sink, processes=None, verbose=False, **kwargs):
    """
    Transforms a source into another format.

    Parameters
    ----------
    source_ : source specification
        The source or list of sources.
    sink : source specification
        The sink or list of sinks.

    Returns
    -------
    sink : sink specification
        The sink or list of sinks.
    """
    sink = fu.normalize_location_spec(sink)
    source_ = open_ro(source_)
    if verbose:
        print(f'converting {source_} -> {sink}')
    mod = source_to_module(source_)
    if hasattr(mod, 'convert'):
        return mod.convert(source_, sink, processes=processes, verbose=verbose, **kwargs)
    else:
        return write(sink, source_)


def convert_files(filenames, extension=None, path=None, processes=None, verbose=False, workspace=None, verify=False):
    """
    Transforms list of files to their sink format in parallel.

    Parameters
    ----------
    filenames : list of str | list of pathlib.Path
        The filenames to convert
    extension : str
        The new file format extension.
    path : str or None
        Optional path specification.
    processes : int, 'serial' or None
        The number of processes to use for parallel conversion.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    filenames : list of str
        The new file names.
    """
    if extension.startswith('.'):  # FIXME: downstream code should handle extension with or without dot
        extension = extension[1:]
    if not isinstance(filenames, (tuple, list)):
        filenames = [filenames]
    if len(filenames) == 0:
        return []
    n_files = len(filenames)

    if path is not None:
        filenames = [fu.join(path, fu.split(f)[1]) for f in filenames]  # TODO: replace with pathlib
    sinks = [str(pathlib.Path(f).with_suffix('.'+extension)) for f in filenames]

    if verbose:
        timer = tmr.Timer()
        print(f'Converting {n_files} files to {extension}!')

    if not isinstance(processes, int) and processes != 'serial':
        processes = mp.cpu_count()

    # print(n_files, extension, filenames, sinks)
    _convert = functools.partial(_convert_files, n_files=n_files, extension=extension, verbose=verbose, verify=verify)

    if processes == 'serial':
        [_convert(source_, sink, i) for i, source_, sink in zip(range(n_files), filenames, sinks)]
    else:
        with CancelableProcessPoolExecutor(processes) as executor:
            results = executor.map(_convert, filenames, sinks, range(n_files))
            if workspace is not None:
                workspace.executor = executor
            _ = list(results)  # to catch exceptions
        if workspace is not None:
            workspace.executor = None

    if verbose:
        timer.print_elapsed_time(f'Converting {n_files} files to {extension}')

    return sinks


@ptb.parallel_traceback
def _convert_files(source_, sink, fid, n_files, extension, verbose, verify=False):
    source_ = open_ro(source_)
    if verbose:
        print(f'Converting file {fid}/{n_files} {source_} -> {sink}')
    io_ops.write(sink, source_)
    if verify:
        src_mean = source_.array.mean()
        sink_mean = io_ops.read(sink).mean()
        if not math.isclose(src_mean, sink_mean, rel_tol=1e-5):
            raise RuntimeError(f"Conversion of {source_} to {sink} failed, means differ")
