"""
The blocks are sized for the processes that will run them, counted like everywhere else (``sanitize_n_processes``).

``block_sizes`` used to turn anything but a positive integer into 1: with ``processes=-1`` (all the CPUs but one, as
the configurations say it) the blocks were sized for 1 process, 10 blocks of 100 for 7 processes instead of 7 of 142.
"""
import pytest

from ClearMap.Utils import utilities

CPUS_OF_THE_JOB = 8
SIZE = 1000
BLOCK_LIMITS = dict(size_max=100, size_min=10, overlap=0)


@pytest.fixture
def block_processing(monkeypatch):
    def cpus_of_the_job():
        return CPUS_OF_THE_JOB

    monkeypatch.setattr(utilities, 'available_cpu_count', cpus_of_the_job)
    return pytest.importorskip('ClearMap.ParallelProcessing.BlockProcessing')


def _number_of_blocks(block_processing, processes):
    n_blocks, _, _ = block_processing.block_sizes(SIZE, processes=processes, **BLOCK_LIMITS)
    return n_blocks


@pytest.mark.parametrize('processes, same_as', [(None, CPUS_OF_THE_JOB), (-1, CPUS_OF_THE_JOB - 1), (0, 1)])
def test_the_blocks_are_sized_for_the_processes_that_will_run_them(block_processing, processes, same_as):
    assert _number_of_blocks(block_processing, processes) == _number_of_blocks(block_processing, same_as)


def test_the_number_of_processes_changes_the_blocks(block_processing):
    assert _number_of_blocks(block_processing, CPUS_OF_THE_JOB - 1) != _number_of_blocks(block_processing, 1)
