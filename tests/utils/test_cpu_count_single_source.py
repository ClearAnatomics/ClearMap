"""
How many CPUs ClearMap may use is asked to ``ClearMap.Utils.utilities`` (``available_cpu_count`` or, for a number
relative to it, ``sanitize_n_processes``), and nowhere else.

``multiprocessing.cpu_count()``, ``os.cpu_count()`` and ``psutil.cpu_count()`` are the cores of the machine: on a cluster
a job gets a few cores of a node, and every place that asks them starts as many processes as the node has cores, on
the few cores of the job. One of them is enough to bring a pipeline back to fighting with itself, so none is left.
"""
import ast
import re
import warnings
from pathlib import Path

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[2] / 'ClearMap'
THE_ONE_PLACE = PACKAGE_DIRECTORY / 'Utils' / 'utilities.py'
NOT_MAINTAINED_DIRECTORIES = ('External', '_Old', '_Todo')  # Vendored code, and code that is not imported
COUNTING_THE_CORES = 'cpu_count'


def _sources(suffix):
    return [path for path in sorted(PACKAGE_DIRECTORY.rglob(f'*{suffix}'))
            if path != THE_ONE_PLACE
            and not set(path.relative_to(PACKAGE_DIRECTORY).parts) & set(NOT_MAINTAINED_DIRECTORIES)]


def _lines_that_count_the_cores(tree):
    """The lines of a call to ``cpu_count`` (``mp.cpu_count()``, ``os.cpu_count()``, ``cpu_count()``...) or of its import"""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', None)
            if name == COUNTING_THE_CORES:
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and any(alias.name == COUNTING_THE_CORES for alias in node.names):
            lines.append(node.lineno)
    return lines


def test_no_python_module_counts_the_cores_of_the_machine():
    offenders = []
    for path in _sources('.py'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Old files with invalid escape sequences
            try:
                tree = ast.parse(path.read_text(encoding='utf-8'))
            except SyntaxError:  # Python 2 files that are not imported
                continue
        offenders += [f'{path.relative_to(PACKAGE_DIRECTORY.parent)}:{line}'
                      for line in _lines_that_count_the_cores(tree)]
    assert not offenders, ('Ask ClearMap.Utils.utilities.available_cpu_count() or sanitize_n_processes() instead:\n'
                           + '\n'.join(offenders))


def test_no_cython_module_counts_the_cores_of_the_machine():
    offenders = [str(path.relative_to(PACKAGE_DIRECTORY.parent)) for path in _sources('.pyx')
                 if re.search(rf'\b{COUNTING_THE_CORES}\b', path.read_text(encoding='utf-8'))]
    assert not offenders, f'Ask ClearMap.Utils.utilities.available_cpu_count() instead: {offenders}'


def test_the_scan_sees_the_calls_it_looks_for():
    tree = ast.parse('import os\nimport multiprocessing as mp\nfrom multiprocessing import cpu_count\n'
                     'a = mp.cpu_count()\nb = cpu_count()\nc = os.cpu_count() - 1\nd = len(os.sched_getaffinity(0))\n')
    assert _lines_that_count_the_cores(tree) == [3, 4, 5, 6]
