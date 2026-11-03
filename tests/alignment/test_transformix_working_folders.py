"""
Each transformix run keeps its input and output in a folder of its own, in the temp folder. With the former fixed
names (``elastix_input.*``, ``elastix_output``), two jobs or processes running at the same time on a node overwrote
each other's points.
"""
import multiprocessing
import shutil
from pathlib import Path

import numpy as np
import pytest

from ClearMap.Alignment import Elastix as elastix
from ClearMap.Utils.exceptions import ClearMapRuntimeError
from .conftest import FAKE_TRANSFORMIX_SLEEP_VARIABLE

POINTS = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

N_CONCURRENT_RUNS = 6
CONCURRENT_RUN_DURATION_SECONDS = '0.5'
"""Long enough for every run to write its input before any of them has read it"""


def test_two_point_transformations_use_different_input_and_output_paths(temp_folder, transformix_runs,
                                                                        transform_directory):
    for _ in range(2):
        transformed = elastix.transform_points(POINTS, transform_directory=str(transform_directory), binary=True)
        np.testing.assert_array_equal(transformed, POINTS)

    first, second = transformix_runs()
    assert first['-def'] != second['-def']
    assert first['-out'] != second['-out']


def test_a_point_transformation_leaves_nothing_in_the_temp_folder(temp_folder, transformix_runs, transform_directory,
                                                                  tmp_path):
    result_directory = tmp_path / 'results'
    elastix.transform_points(POINTS, transform_directory=str(transform_directory), binary=True)
    elastix.transform_points(POINTS, transform_directory=str(transform_directory), binary=True,
                             result_directory=str(result_directory))

    assert list(temp_folder.iterdir()) == []
    assert (result_directory / 'outputpoints.bin').exists()


def test_a_refused_point_source_leaves_nothing_in_the_temp_folder(temp_folder, transformix_runs, transform_directory):
    with pytest.raises(ClearMapRuntimeError, match='source not string or array'):
        elastix.transform_points([1.0, 2.0, 3.0], transform_directory=str(transform_directory))

    assert list(temp_folder.iterdir()) == []


def test_two_image_transformations_use_different_input_and_output_paths(temp_folder, transformix_runs,
                                                                        transform_directory):
    result_files = [elastix.transform(np.zeros((4, 5, 6), dtype=np.uint16), sink='transformix',
                                      transform_directory=str(transform_directory)) for _ in range(2)]

    first, second = transformix_runs()
    assert first['-in'] != second['-in']
    assert Path(result_files[0]).parent != Path(result_files[1]).parent
    assert not any(path.name.startswith('elastix_input') for path in temp_folder.rglob('*'))  # the inputs are gone


def test_two_deformation_fields_use_different_output_paths(temp_folder, transformix_runs, transform_directory):
    result_files = [elastix.deformation_field(sink='transformix', transform_directory=str(transform_directory))
                    for _ in range(2)]

    assert Path(result_files[0]).parent != Path(result_files[1]).parent


def test_an_image_transformation_into_a_result_directory_leaves_nothing_in_the_temp_folder(
        temp_folder, transformix_runs, transform_directory, tmp_path):
    result_directory = tmp_path / 'results'

    result_file = elastix.transform(np.zeros((4, 5, 6), dtype=np.uint16), sink='transformix',
                                    transform_directory=str(transform_directory),
                                    result_directory=str(result_directory))

    assert Path(result_file).parent == result_directory
    assert list(temp_folder.iterdir()) == []


def test_a_transformix_working_folder_is_deleted_with_its_content(temp_folder):
    working_directory = Path(elastix.tempfile.mkdtemp(prefix=elastix.TRANSFORMIX_WORKING_DIRECTORY_PREFIX))
    (working_directory / 'outputpoints.txt').touch()

    elastix._remove_working_directory(working_directory)

    assert not working_directory.exists()


def test_a_folder_that_is_not_a_transformix_working_folder_is_never_deleted(tmp_path):
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'result.mhd').touch()

    with pytest.raises(ClearMapRuntimeError, match='Refusing to delete'):
        elastix._remove_working_directory(results)

    assert (results / 'result.mhd').exists()


def transform_points_when_every_process_is_ready(all_ready, run, transform_directory, results):
    """Run in a process of its own: the processes share nothing but the temp folder, as in tract_map"""
    try:
        points = POINTS + run
        all_ready.wait()
        results.put((run, elastix.transform_points(points, transform_directory=str(transform_directory),
                                                   binary=True).tolist()))
    except Exception as error:  # reported to the test, which compares what each process got back
        results.put((run, repr(error)))


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(),
                    reason='the processes inherit the fake transformix by fork')
def test_processes_transforming_points_at_the_same_time_each_get_their_own_points_back(
        temp_folder, transformix_runs, transform_directory, tmp_path, monkeypatch):
    monkeypatch.setenv(FAKE_TRANSFORMIX_SLEEP_VARIABLE, CONCURRENT_RUN_DURATION_SECONDS)
    context = multiprocessing.get_context('fork')
    all_ready = context.Barrier(N_CONCURRENT_RUNS)
    results = context.Queue()
    # Each run has its own copy of the transform files, as tract_map does: transformix runs edit them in place
    transform_directories = [shutil.copytree(transform_directory, tmp_path / f'transform_{run}')
                             for run in range(N_CONCURRENT_RUNS)]
    processes = [context.Process(target=transform_points_when_every_process_is_ready,
                                 args=(all_ready, run, transform_directories[run], results))
                 for run in range(N_CONCURRENT_RUNS)]
    for process in processes:
        process.start()
    outcomes = dict(results.get(timeout=60) for _ in processes)
    for process in processes:
        process.join(timeout=60)

    assert outcomes == {run: (POINTS + run).tolist() for run in range(N_CONCURRENT_RUNS)}
    assert list(temp_folder.iterdir()) == []
