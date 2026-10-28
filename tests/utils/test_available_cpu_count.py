"""
The number of CPUs ClearMap may use is ``available_cpu_count()``, not ``multiprocessing.cpu_count()``, which is the
number of cores of the machine: on a cluster a job gets a few cores of a node, and starting a process for each core
of the node makes it fight with itself and with the jobs of the others.
"""
import os

import pytest

from ClearMap.Utils import utilities

CORES_OF_THE_NODE = 64
ALLOWED_CORES = 8


@pytest.fixture
def node_of_64_cores(monkeypatch):
    """The machine has 64 cores, the process may run on all of them and no Slurm variable is set"""
    def all_cores_of_the_node(_pid):
        return set(range(CORES_OF_THE_NODE))

    def cores_of_the_node():
        return CORES_OF_THE_NODE

    monkeypatch.setattr(os, 'sched_getaffinity', all_cores_of_the_node, raising=False)
    monkeypatch.setattr(os, 'cpu_count', cores_of_the_node)
    for name in utilities.CPU_LIMIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_without_any_limit_it_is_the_cores_of_the_machine(node_of_64_cores):
    assert utilities.available_cpu_count() == CORES_OF_THE_NODE


def test_the_cpus_the_process_may_run_on_are_counted_not_the_cores_of_the_node(node_of_64_cores, monkeypatch):
    def allowed_cores(_pid):
        return set(range(ALLOWED_CORES))

    monkeypatch.setattr(os, 'sched_getaffinity', allowed_cores, raising=False)
    assert utilities.available_cpu_count() == ALLOWED_CORES


@pytest.mark.parametrize('name', utilities.CPU_LIMIT_ENV_VARS)
def test_a_slurm_limit_lower_than_the_cores_is_used(node_of_64_cores, monkeypatch, name):
    monkeypatch.setenv(name, str(ALLOWED_CORES))
    assert utilities.available_cpu_count() == ALLOWED_CORES


@pytest.mark.parametrize('name', utilities.CPU_LIMIT_ENV_VARS)
def test_a_slurm_limit_higher_than_the_cpus_the_process_may_use_does_not_raise_them(node_of_64_cores, monkeypatch, name):
    def allowed_cores(_pid):
        return set(range(ALLOWED_CORES))

    monkeypatch.setattr(os, 'sched_getaffinity', allowed_cores, raising=False)
    monkeypatch.setenv(name, str(CORES_OF_THE_NODE))
    assert utilities.available_cpu_count() == ALLOWED_CORES


def test_the_lowest_of_the_limits_wins(node_of_64_cores, monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '16')
    assert utilities.available_cpu_count() == 4


@pytest.mark.parametrize('not_a_limit', ('', ' ', 'many', '0', '-3', '2.5', '8(x2)'))
def test_a_slurm_value_that_is_not_a_positive_integer_is_ignored(node_of_64_cores, monkeypatch, not_a_limit):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', not_a_limit)
    assert utilities.available_cpu_count() == CORES_OF_THE_NODE


def test_where_the_system_does_not_say_which_cpus_the_process_may_use_it_is_the_cores_of_the_machine(
        node_of_64_cores, monkeypatch):
    monkeypatch.delattr(os, 'sched_getaffinity')
    assert utilities.available_cpu_count() == CORES_OF_THE_NODE


def test_it_is_at_least_one_even_if_the_system_cannot_count(node_of_64_cores, monkeypatch):
    def cannot_count():
        return None

    monkeypatch.delattr(os, 'sched_getaffinity')
    monkeypatch.setattr(os, 'cpu_count', cannot_count)
    assert utilities.available_cpu_count() == 1


@pytest.fixture
def job_of_8_cores(node_of_64_cores, monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', str(ALLOWED_CORES))


def test_no_number_of_processes_is_all_the_cpus_of_the_job_not_of_the_node(job_of_8_cores):
    assert utilities.sanitize_n_processes(None) == ALLOWED_CORES


@pytest.mark.parametrize('processes, expected', [(-1, ALLOWED_CORES - 1), (-2, ALLOWED_CORES - 2), (3, 3),
                                                 (-ALLOWED_CORES, 1), (-100, 1), (0, 1)])
def test_negative_numbers_of_processes_are_taken_from_the_cpus_of_the_job(job_of_8_cores, processes, expected):
    assert utilities.sanitize_n_processes(processes) == expected


def test_not_serial_leaves_one_cpu_of_the_job(job_of_8_cores):
    with pytest.warns(DeprecationWarning):
        assert utilities.sanitize_n_processes('!serial') == ALLOWED_CORES - 1


@pytest.mark.parametrize('processes', (ALLOWED_CORES + 1, 23, CORES_OF_THE_NODE, 10 ** 6))
def test_more_processes_than_the_cpus_of_the_job_are_the_cpus_of_the_job(job_of_8_cores, processes):
    """The defaults of the configuration are written for a large workstation: a small job does not fight with itself"""
    assert utilities.sanitize_n_processes(processes) == ALLOWED_CORES


def test_as_many_processes_as_the_cpus_of_the_job_are_kept(job_of_8_cores):
    assert utilities.sanitize_n_processes(ALLOWED_CORES) == ALLOWED_CORES


@pytest.mark.parametrize('can_run_in_calling_process, expected', [(True, 0), (False, 1)])
def test_the_cap_does_not_touch_the_calling_process(job_of_8_cores, can_run_in_calling_process, expected):
    assert utilities.sanitize_n_processes(0, can_run_in_calling_process=can_run_in_calling_process) == expected
