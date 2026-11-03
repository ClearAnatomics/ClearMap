"""
The scripts of new_api stitch and register with the methods of the processors (stitch and register): they had their
own copies of the steps (align_new_api.stitch, register), which forced the rigid and the wobbly alignments whatever
the skip flags said. What the scripts import from align_new_api must exist (the scripts need a display and the
data to run: this reads their source).
"""
import ast
from pathlib import Path

import pytest

SCRIPTS_FOLDER = Path(__file__).parents[1] / 'ClearMap' / 'Scripts'
SCRIPTS = ('cell_map_new_api', 'tube_map_new_api', 'tract_map', 'colocalization')


def _tree_of(script_name):
    return ast.parse((SCRIPTS_FOLDER / f'{script_name}.py').read_text())


def _names_defined_by(script_name):
    return {node.name for node in _tree_of(script_name).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def _called_attributes(script_name):
    return {node.func.attr for node in ast.walk(_tree_of(script_name))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}


@pytest.mark.parametrize('script_name', SCRIPTS)
def test_the_names_imported_from_align_new_api_exist(script_name):
    imported = {alias.name for node in ast.walk(_tree_of(script_name))
                if isinstance(node, ast.ImportFrom) and node.module == 'ClearMap.Scripts.align_new_api'
                for alias in node.names}
    assert imported <= _names_defined_by('align_new_api')


@pytest.mark.parametrize('script_name', SCRIPTS)
def test_the_scripts_stitch_and_register_with_the_processors(script_name):
    assert {'stitch', 'register'} <= _called_attributes(script_name)
