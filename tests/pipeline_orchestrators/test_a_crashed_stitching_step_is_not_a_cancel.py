"""
When the process pool of a stitching step breaks, the step was either canceled (the Abort button: stop_process sets
stopped and shuts the pool down) or crashed (a worker died, e.g. killed for lack of memory). The steps treated both as
a cancel without setting stopped, so after a crash of the wobbly alignment the stitching went on: it saved the layout
which was not aligned and placed it, which failed with "could not broadcast input array from shape (2433,2) into shape
(2430,2)" (#P0016). A crash now stops the stitching with an error naming the step; a cancel stays a cancel.

The tiles are stitched for real (the fixtures of test_stitching_steps); only the step that breaks is replaced.
"""
from concurrent.futures.process import BrokenProcessPool

import pytest

from ClearMap.Alignment.Stitching import stitching_rigid, stitching_wobbly
from ClearMap.Utils.exceptions import WorkerProcessDiedError

from .test_stitching_steps import graph_library_in_python, stitching  # noqa: F401 (fixtures)

STEPS = pytest.mark.parametrize('module, function_name, step', [
    (stitching_rigid, 'align_layout_rigid_mip', 'rigid alignment'),
    (stitching_wobbly, 'align_layout', 'wobbly alignment'),
    (stitching_wobbly, 'place_layout', 'wobbly placement'),
    (stitching_wobbly, 'stitch_layout', 'stitching'),
], ids=['rigid alignment', 'wobbly alignment', 'wobbly placement', 'stitching'])


def break_the_pool_of(monkeypatch, module, function_name, *, worker=None):
    """The step's pool breaks; with a worker, the Abort button was pressed first (stop_process sets stopped)"""
    def broken_pool(*args, **kwargs):
        if worker is not None:
            worker.stopped = True
        raise BrokenProcessPool('A process in the process pool was terminated abruptly')
    monkeypatch.setattr(module, function_name, broken_pool)


@STEPS
def test_a_step_whose_worker_died_stops_the_stitching_with_an_error_naming_the_step(stitching, monkeypatch,
                                                                                     module, function_name, step):
    break_the_pool_of(monkeypatch, module, function_name)

    with pytest.raises(WorkerProcessDiedError, match=f'The {step} stopped: one of its worker processes died'):
        stitching.stitch_channel('main')

    assert not stitching.get('stitched', channel='main').exists


def test_after_a_crash_of_the_wobbly_alignment_the_layout_is_neither_saved_as_aligned_nor_placed(stitching,
                                                                                                monkeypatch):
    break_the_pool_of(monkeypatch, stitching_wobbly, 'align_layout')

    with pytest.raises(WorkerProcessDiedError):
        stitching.stitch_channel('main')

    assert not stitching.get('layout', channel='main', asset_sub_type='aligned').exists
    assert not stitching.get('layout', channel='main', asset_sub_type='placed').exists


@STEPS
def test_a_step_canceled_with_the_abort_button_stops_the_stitching_without_error(stitching, monkeypatch,
                                                                                 module, function_name, step):
    break_the_pool_of(monkeypatch, module, function_name, worker=stitching)

    stitching.stitch_channel('main')

    assert stitching.stopped
    assert not stitching.get('stitched', channel='main').exists
