"""
The annotators belong to the registration processor, which tells what is wrong when a channel has none
(``annotator_of``). Indexing ``annotators`` elsewhere gives a KeyError, or an AttributeError on None, for the same cause.
"""
import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2] / 'ClearMap'
OWNER = PACKAGE / 'pipeline_orchestrators' / 'registration_orchestrator.py'


def _annotators_indexed_outside_the_registration_processor():
    found = []
    for path in sorted(PACKAGE.rglob('*.py')):
        if path == OWNER:
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except SyntaxError:  # Legacy files kept for reference (python 2)
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                    and node.value.attr == 'annotators'):
                found.append(f'{path.relative_to(PACKAGE)}:{node.lineno}')
    return found


def test_the_annotators_are_only_indexed_by_the_registration_processor():
    assert _annotators_indexed_outside_the_registration_processor() == []
