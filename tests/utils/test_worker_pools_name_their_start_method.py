"""
Python 3.14 makes ``forkserver`` the default start method of multiprocessing on Linux. The workers of ClearMap rely on
what a forked process inherits (module globals, open memory maps, local classes), so every pool is created in the
context of ``worker_pool_context`` (``fork`` on Linux), and no pool is left to the default of the Python in use.

The behaviour is tested in a new interpreter that sets ``forkserver`` as its default, as Python 3.14 does.
"""
import ast
import subprocess
import sys

import pytest

from ClearMap.Utils.utilities import worker_pool_context
from tests.source_scan import maintained_syntax_trees

INHERITED_VALUE = 'set by the parent after the import'
RESULT_MARKER = 'WORKER_SAW:'
# Demonstrations of the shared arrays, in the test functions at the end of the modules (they run nothing by themselves)
DEMONSTRATION_MODULES = ('ClearMap/ParallelProcessing/SharedMemoryArray.py',
                         'ClearMap/ParallelProcessing/SharedMemoryManager.py')

PROBE_MODULE = f'''
MARK = None


def read_mark():
    return MARK
'''

RUN_A_POOL_WITH_FORKSERVER_AS_DEFAULT = f'''
import multiprocessing
import sys
sys.path.insert(0, sys.argv[1])
import pool_probe
from ClearMap.Utils.utilities import CancelableProcessPoolExecutor

multiprocessing.set_start_method('forkserver', force=True)  # The default of Python 3.14 on Linux
pool_probe.MARK = {INHERITED_VALUE!r}
with CancelableProcessPoolExecutor(1) as executor:
    print({RESULT_MARKER!r} + repr(executor.submit(pool_probe.read_mark).result()))
'''


def _called_name(call):
    function = call.func
    return function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', '')


def _is_a_pool_of_the_default_context(call):
    """``mp.Pool(...)``, ``multiprocessing.Pool(...)``, ``smm.mp.Pool(...)``: the pools made without a context"""
    function = call.func
    if not (isinstance(function, ast.Attribute) and function.attr == 'Pool'):
        return False
    owner = function.value
    owner_name = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, 'id', '')
    return owner_name in ('mp', 'multiprocessing')


def test_the_context_of_the_workers_forks_on_linux():
    if not sys.platform.startswith('linux'):
        pytest.skip('only Linux is told to fork')

    assert worker_pool_context().get_start_method() == 'fork'


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='only Linux is told to fork')
def test_a_worker_inherits_the_state_of_its_parent_even_when_the_default_start_method_is_forkserver(tmp_path):
    (tmp_path / 'pool_probe.py').write_text(PROBE_MODULE)

    result = subprocess.run([sys.executable, '-c', RUN_A_POOL_WITH_FORKSERVER_AS_DEFAULT, str(tmp_path)],
                            capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    seen = next(line for line in result.stdout.splitlines() if line.startswith(RESULT_MARKER))
    assert seen == RESULT_MARKER + repr(INHERITED_VALUE)


def test_every_process_pool_of_clearmap_is_given_a_context():
    without_context = []
    for path, tree in maintained_syntax_trees(allowed_modules=DEMONSTRATION_MODULES):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _called_name(node) == 'ProcessPoolExecutor' and not any(
                    keyword.arg == 'mp_context' for keyword in node.keywords) and len(node.args) < 2:
                without_context.append(f'{path}:{node.lineno} ProcessPoolExecutor')
            if _is_a_pool_of_the_default_context(node):
                without_context.append(f'{path}:{node.lineno} multiprocessing Pool')

    assert without_context == [], 'create the pools with worker_pool_context(): ' + ', '.join(without_context)
