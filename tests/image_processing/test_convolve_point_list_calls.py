"""
The convolutions at points (ConvolvePointList) are compiled and cannot be run everywhere, and their wrappers have
no **kwargs. skeletonize(radii=True) called convolve_3d_indices with out_dtype=, which the function never had (it is
sink_dtype), and failed with "unexpected keyword argument" (#75). This reads the source: every call of a function
of the module, through the alias under which it is imported, uses arguments that the function has.
"""
import ast
from pathlib import Path

import pytest

CLEARMAP_FOLDER = Path(__file__).parents[2] / 'ClearMap'
CONVOLVE_MODULE = 'ClearMap.ParallelProcessing.DataProcessing.ConvolvePointList'
CONVOLVE_MODULE_FILE = CLEARMAP_FOLDER / 'ParallelProcessing' / 'DataProcessing' / 'ConvolvePointList.py'


def _signatures_of_the_module():
    """{function name: (names of the arguments, accepts any keyword)} of the functions defined at the top level"""
    signatures = {}
    for node in ast.parse(CONVOLVE_MODULE_FILE.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            arguments = node.args
            names = {a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
            signatures[node.name] = (names, arguments.kwarg is not None)
    return signatures


def _lines_of_functions_which_import_the_alias_from_elsewhere(tree, aliases):
    """The lines of the functions which bind the alias to another module than ConvolvePointList (stale scratch code)"""
    lines = set()
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef):
            rebinds = any(isinstance(node, ast.Import) and any(a.asname in aliases and a.name != CONVOLVE_MODULE
                                                              for a in node.names)
                          for node in ast.walk(function))
            if rebinds:
                lines.update(range(function.lineno, function.end_lineno + 1))
    return lines


def _calls_through_the_alias(source_file):
    """[(line, function name, keywords given)] of the calls of the module under the alias it has in the file"""
    text = source_file.read_text()
    if CONVOLVE_MODULE not in text:  # Also spares the files which do not parse (python 2 leftovers)
        return []
    tree = ast.parse(text)
    aliases = {alias.asname for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names if alias.name == CONVOLVE_MODULE and alias.asname}
    other_module_lines = _lines_of_functions_which_import_the_alias_from_elsewhere(tree, aliases)
    return [(node.lineno, node.func.attr, [k.arg for k in node.keywords if k.arg is not None])
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id in aliases
            and node.lineno not in other_module_lines]


def _all_calls():
    # Not the module itself: its test() is scratch code which calls functions that no longer exist
    return [(source_file, line, name, keywords) for source_file in sorted(CLEARMAP_FOLDER.rglob('*.py'))
            if source_file != CONVOLVE_MODULE_FILE
            for line, name, keywords in _calls_through_the_alias(source_file)]


def test_the_calls_of_the_convolutions_are_found():
    """If this finds nothing the test below checks nothing (the alias or the module moved)"""
    called = {name for _, _, name, _ in _all_calls()}

    assert 'convolve_3d_indices' in called


def test_every_call_of_a_convolution_uses_keywords_that_the_function_has():
    signatures = _signatures_of_the_module()
    wrong = []
    for source_file, line, name, keywords in _all_calls():
        names, accepts_any_keyword = signatures[name]
        unknown = [k for k in keywords if k not in names]
        if unknown and not accepts_any_keyword:
            wrong.append(f'{source_file.relative_to(CLEARMAP_FOLDER.parent)}:{line}: {name}({", ".join(unknown)}=...)')

    assert wrong == []
