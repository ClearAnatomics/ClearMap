"""
``processes=0`` means no subprocess: the calling process does the work, one item after the other (to debug).

It is not the same as 1, a pool of one subprocess. The code that can run in the calling process (stitching,
conversion, block processing, resampling, the file lists...) asks
``sanitize_n_processes(processes, can_run_in_calling_process=True)`` and gets 0; the code that only needs a count
(the threads of compiled code...) gets 1. ``'serial'`` used to mean it, but the resampling counted the processes
before it looked for ``'serial'``, so it never ran in the calling process.
"""
import ast
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from ClearMap.Utils import utilities
from ClearMap.Utils.exceptions import ClearMapTypeError, ClearMapValueError

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[2] / 'ClearMap'
THE_ONE_PLACE = PACKAGE_DIRECTORY / 'Utils' / 'utilities.py'
NOT_MAINTAINED_DIRECTORIES = ('External', '_Old', '_Todo')
CPUS_OF_THE_JOB = 8


@pytest.fixture
def job_of_8_cpus(monkeypatch):
    def cpus_of_the_job():
        return CPUS_OF_THE_JOB

    monkeypatch.setattr(utilities, 'available_cpu_count', cpus_of_the_job)


def test_zero_is_the_calling_process_where_the_code_can_run_there():
    assert utilities.sanitize_n_processes(0, can_run_in_calling_process=True) == 0


def test_zero_is_one_worker_for_the_code_that_only_needs_a_count():
    assert utilities.sanitize_n_processes(0) == 1


@pytest.mark.parametrize('processes, expected', [(None, CPUS_OF_THE_JOB), (3, 3), (-2, CPUS_OF_THE_JOB - 2),
                                                 (-CPUS_OF_THE_JOB, 1), (-100, 1)])
def test_a_number_relative_to_the_cpus_never_means_the_calling_process(job_of_8_cpus, processes, expected):
    assert utilities.sanitize_n_processes(processes, can_run_in_calling_process=True) == expected


@pytest.mark.parametrize('spelling', ('serial', 'Serial', 'SERIAL'))
def test_the_former_serial_is_the_calling_process_and_is_deprecated(spelling):
    with pytest.warns(DeprecationWarning):
        assert utilities.sanitize_n_processes(spelling, can_run_in_calling_process=True) == 0
    with pytest.warns(DeprecationWarning):
        assert utilities.sanitize_n_processes(spelling) == 1


def test_the_former_not_serial_is_all_but_one_and_never_the_calling_process(monkeypatch):
    def one_cpu():
        return 1

    monkeypatch.setattr(utilities, 'available_cpu_count', one_cpu)
    with pytest.warns(DeprecationWarning):
        assert utilities.sanitize_n_processes('!serial', can_run_in_calling_process=True) == 1


def test_another_string_is_refused():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        with pytest.raises(ClearMapValueError):
            utilities.sanitize_n_processes('many', can_run_in_calling_process=True)


@pytest.mark.parametrize('not_a_number', (2.5, [1]))
def test_anything_else_is_refused(not_a_number):
    with pytest.raises(ClearMapTypeError):
        utilities.sanitize_n_processes(not_a_number, can_run_in_calling_process=True)


def process_id_of_the_block(block):
    return os.getpid()


@pytest.fixture
def block_processing():
    return pytest.importorskip('ClearMap.ParallelProcessing.BlockProcessing')


def _process_ids(block_processing, processes):
    data = np.zeros((4, 4, 8), dtype='uint8')
    process_ids = block_processing.process(process_id_of_the_block, data, axes=[2], size_max=2, size_min=2,
                                           overlap=0, function_type='block', return_result=True, processes=processes)
    assert len(process_ids) == 4 and None not in process_ids, process_ids
    return process_ids


def test_block_processing_with_zero_runs_every_block_in_the_calling_process(block_processing):
    assert set(_process_ids(block_processing, 0)) == {os.getpid()}


def test_block_processing_with_one_runs_the_blocks_in_a_subprocess(block_processing):
    process_ids = set(_process_ids(block_processing, 1))
    assert len(process_ids) == 1 and os.getpid() not in process_ids


def _comparisons_with_serial(tree):
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and any(isinstance(operand, ast.Constant) and operand.value == 'serial'
                    for operand in [node.left, *node.comparators])]


def test_no_module_looks_for_serial_itself():
    offenders = []
    for path in sorted(PACKAGE_DIRECTORY.rglob('*.py')):
        if path == THE_ONE_PLACE or set(path.relative_to(PACKAGE_DIRECTORY).parts) & set(NOT_MAINTAINED_DIRECTORIES):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                tree = ast.parse(path.read_text(encoding='utf-8'))
            except SyntaxError:
                continue
        offenders += [f'{path.relative_to(PACKAGE_DIRECTORY.parent)}:{line}' for line in _comparisons_with_serial(tree)]
    assert not offenders, ('Ask sanitize_n_processes(processes, can_run_in_calling_process=True) and compare with 0:\n'
                           + '\n'.join(offenders))
