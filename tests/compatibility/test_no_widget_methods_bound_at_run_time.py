"""
The widgets of the GUI get their methods from classes (compound_boxes.NumberListBox, a QDialogButtonBox is connected
by its standard buttons), not from functions bound to widgets at run time by their name: a widget without its methods
is then an error at its creation, not at its first use. A scan of the syntax trees of the maintained modules.
"""
import ast

from tests.source_scan import maintained_syntax_trees

REMOVED_BUTTON_BOX_METHODS = ('connectApply', 'connectClose', 'connectSave', 'connectOpen', 'connectOk', 'connectCancel')
BINDING_FUNCTION_NAME = 'MethodType'


def _binds_a_function_to_an_instance(node):
    return isinstance(node, ast.Call) and getattr(node.func, 'attr', getattr(node.func, 'id', None)) == BINDING_FUNCTION_NAME


def _uses_a_button_box_shortcut(node):
    return isinstance(node, ast.Attribute) and node.attr in REMOVED_BUTTON_BOX_METHODS


def _offences(is_offence):
    return [f'{path}:{node.lineno}' for path, tree in maintained_syntax_trees()
            for node in ast.walk(tree) if is_offence(node)]


def test_no_maintained_module_binds_a_function_to_an_instance_with_methodtype():
    offences = _offences(_binds_a_function_to_an_instance)

    assert not offences, f'Give the widget a class (a promoted widget in the designer file) at {offences}'


def test_no_maintained_module_calls_the_shortcuts_of_the_button_boxes():
    offences = _offences(_uses_a_button_box_shortcut)

    assert not offences, f'Connect button(QDialogButtonBox.<Button>).clicked at {offences}'
