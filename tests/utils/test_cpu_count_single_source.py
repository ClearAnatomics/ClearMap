"""
How many CPUs ClearMap may use is asked to ``ClearMap.Utils.utilities`` (``available_cpu_count`` or, for a number
relative to it, ``sanitize_n_processes``), and nowhere else.

``multiprocessing.cpu_count()``, ``os.cpu_count()`` and ``psutil.cpu_count()`` are the cores of the machine: on a
cluster a job gets a few cores of a node, and every place that asks them starts as many processes as the node has
cores, on the few cores of the job. One of them is enough to bring a pipeline back to fighting with itself, so none
is left.

A pool given no number of workers (or None) counts them the same way, inside the standard library: every pool is
given a number that was counted by those functions (or a fixed one).
"""
import ast
import re
import warnings
from pathlib import Path

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[2] / 'ClearMap'
THE_ONE_PLACE = PACKAGE_DIRECTORY / 'Utils' / 'utilities.py'
NOT_MAINTAINED_DIRECTORIES = ('External', '_Old', '_Todo')  # Vendored code, and code that is not imported
COUNTING_THE_CORES = 'cpu_count'
POOLS = ('Pool', 'ThreadPool', 'ProcessPoolExecutor', 'ThreadPoolExecutor', 'CancelableProcessPoolExecutor')
NUMBER_OF_WORKERS_KEYWORDS = ('processes', 'max_workers')
COUNTING_FUNCTIONS = ('available_cpu_count', 'sanitize_n_processes')
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _sources(suffix):
    return [path for path in sorted(PACKAGE_DIRECTORY.rglob(f'*{suffix}'))
            if path != THE_ONE_PLACE
            and not set(path.relative_to(PACKAGE_DIRECTORY).parts) & set(NOT_MAINTAINED_DIRECTORIES)]


def _parsed_python_sources():
    """(path, syntax tree) of the python modules (the Python 2 files, that are not imported, are skipped)"""
    for path in _sources('.py'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Old files with invalid escape sequences
            try:
                yield path, ast.parse(path.read_text(encoding='utf-8'))
            except SyntaxError:
                continue


def _called_name(call):
    function = call.func
    return function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', None)


def _lines_that_count_the_cores(tree):
    """The lines of a call to ``cpu_count`` (``mp.cpu_count()``, ``os.cpu_count()``...) or of its import"""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _called_name(node) == COUNTING_THE_CORES:
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and any(alias.name == COUNTING_THE_CORES for alias in node.names):
            lines.append(node.lineno)
    return lines


def test_no_python_module_counts_the_cores_of_the_machine():
    offenders = []
    for path, tree in _parsed_python_sources():
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


def _calls_a_counting_function(expression):
    return any(isinstance(node, ast.Call) and _called_name(node) in COUNTING_FUNCTIONS for node in ast.walk(expression))


def _counted_names(scope):
    """The names the scope (a function, or the module) gives a counted or a fixed number of workers"""
    names = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            fixed = isinstance(node.value, ast.Constant) and isinstance(node.value.value, int)
            if fixed or _calls_a_counting_function(node.value):
                names |= {target.id for target in node.targets if isinstance(target, ast.Name)}
    return names


def _number_of_workers(pool_call):
    for keyword in pool_call.keywords:
        if keyword.arg in NUMBER_OF_WORKERS_KEYWORDS:
            return keyword.value
    return pool_call.args[0] if pool_call.args else None


def _is_counted(number, counted_names):
    if number is None or (isinstance(number, ast.Constant) and number.value is None):
        return False  # The pool counts the cores of the machine itself
    if isinstance(number, ast.Constant):
        return isinstance(number.value, int)
    return _calls_a_counting_function(number) or any(
        isinstance(node, ast.Name) and node.id in counted_names for node in ast.walk(number))


def _uncounted_pools(node, counted_names):
    """The lines of the pools under the node given no number of workers, None, or a number that neither a counting
    function nor a constant gave (in the function, or at the level of the module)"""
    for child in ast.iter_child_nodes(node):
        names = counted_names | _counted_names(child) if isinstance(child, FUNCTIONS) else counted_names
        if isinstance(child, ast.Call) and _called_name(child) in POOLS \
                and not _is_counted(_number_of_workers(child), names):
            yield child.lineno
        yield from _uncounted_pools(child, names)


def _lines_of_uncounted_pools(tree):
    module_level = ast.Module(body=[statement for statement in tree.body if isinstance(statement, ast.Assign)],
                              type_ignores=[])
    return sorted(_uncounted_pools(tree, _counted_names(module_level)))


def test_every_pool_is_given_a_counted_number_of_workers():
    offenders = []
    for path, tree in _parsed_python_sources():
        offenders += [f'{path.relative_to(PACKAGE_DIRECTORY.parent)}:{line}'
                      for line in _lines_of_uncounted_pools(tree)]
    assert not offenders, ('Give the pool sanitize_n_processes(...) (or available_cpu_count()):\n'
                           + '\n'.join(offenders))


def test_the_pool_scan_sees_what_it_looks_for():
    tree = ast.parse(
        'import multiprocessing as mp\n'
        'from concurrent.futures import ProcessPoolExecutor\n'
        'FIXED = 4\n'
        'def counted(processes):\n'
        '    processes = sanitize_n_processes(processes)\n'
        '    a = mp.Pool(processes)\n'
        '    b = ProcessPoolExecutor(max_workers=min(processes, 3))\n'
        '    c = mp.Pool(processes=FIXED)\n'
        '    d = ProcessPoolExecutor(sanitize_n_processes(-1))\n'
        'def not_counted(processes, other):\n'
        '    e = mp.Pool()\n'
        '    f = mp.Pool(processes=None)\n'
        '    g = ProcessPoolExecutor(processes)\n'
        '    h = ProcessPoolExecutor(max_workers=other - 1)\n')
    assert _lines_of_uncounted_pools(tree) == [11, 12, 13, 14]
