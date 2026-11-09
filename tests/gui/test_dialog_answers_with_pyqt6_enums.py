"""
The answer of a dialog is read the same way with PyQt5 and PyQt6.

``exec()`` gives an int (1 accepted, 0 rejected; for a message box, the int of the button). PyQt6 enums are Python
enums, and an int can no longer be relied on to compare equal to one: ``dialog.exec() == QDialog.Accepted`` may be
False there, so accepting a dialog would cancel it, and choosing Retry in the VRAM popup would abort. ClearMap compares
with DIALOG_ACCEPTED (an int) and reads a message box through standardButton(). The fakes below behave as PyQt6 does.
"""
import ast
import enum

from ClearMap.gui import dialog_helpers
from tests.source_scan import maintained_syntax_trees

EXEC_METHOD_NAMES = ('exec', 'exec_')
DIALOG_CODE_CONSTANT_NAMES = ('DIALOG_ACCEPTED', 'DIALOG_REJECTED')
ENTERED_VALUE = 'sample_1'


class StandardButton(enum.Flag):  # As in PyQt6 (hypothesis: enum.Flag, not IntFlag): not an int
    Abort = 0x00040000
    Retry = 0x00080000


class FakeMessageBox:
    """A message box whose user clicks `clicked_button`"""
    Warning = Abort = StandardButton.Abort
    Retry = StandardButton.Retry
    clicked_button = None

    def __getattr__(self, name):  # setIcon, setText...: no effect here
        return _do_nothing

    def exec(self):
        return self.clicked_button.value  # An int, as with PyQt6

    def clickedButton(self):
        return self.clicked_button

    def standardButton(self, button):
        return button


class FakeInputDialog:
    TextInput = IntInput = DoubleInput = 'mode'
    accepted = True

    def __getattr__(self, name):
        return _do_nothing

    def exec(self):
        return int(self.accepted)

    def textValue(self):
        return ENTERED_VALUE


def _do_nothing(*_, **__):
    pass


def test_retry_in_the_abort_retry_popup_is_read_as_retry(monkeypatch):
    monkeypatch.setattr(dialog_helpers, 'QMessageBox', FakeMessageBox)
    monkeypatch.setattr(FakeMessageBox, 'clicked_button', StandardButton.Retry)

    assert dialog_helpers.abort_retry_popup('Not enough VRAM', 'details') is True


def test_abort_in_the_abort_retry_popup_is_read_as_abort(monkeypatch):
    monkeypatch.setattr(dialog_helpers, 'QMessageBox', FakeMessageBox)
    monkeypatch.setattr(FakeMessageBox, 'clicked_button', StandardButton.Abort)

    assert dialog_helpers.abort_retry_popup('Not enough VRAM', 'details') is False


def test_an_accepted_input_dialog_gives_the_value_entered(monkeypatch):
    monkeypatch.setattr(dialog_helpers, 'QInputDialog', FakeInputDialog)

    assert dialog_helpers.input_dialog('Sample', 'Name of the sample') == ENTERED_VALUE


def test_a_rejected_input_dialog_gives_nothing(monkeypatch):
    monkeypatch.setattr(dialog_helpers, 'QInputDialog', FakeInputDialog)
    monkeypatch.setattr(FakeInputDialog, 'accepted', False)

    assert dialog_helpers.input_dialog('Sample', 'Name of the sample') is None


def _is_an_exec_call(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in EXEC_METHOD_NAMES


def _is_a_dialog_code_constant(node):
    return isinstance(node, ast.Name) and node.id in DIALOG_CODE_CONSTANT_NAMES


def _compares_exec_with_something_else_than_the_dialog_codes(node):
    if not isinstance(node, ast.Compare):
        return False
    operands = [node.left, *node.comparators]
    return any(map(_is_an_exec_call, operands)) and not all(
        _is_an_exec_call(operand) or _is_a_dialog_code_constant(operand) for operand in operands)


def test_no_maintained_module_compares_the_int_of_exec_with_anything_but_the_dialog_codes():
    comparisons = [f'{path}:{node.lineno}' for path, tree in maintained_syntax_trees() for node in ast.walk(tree)
                   if _compares_exec_with_something_else_than_the_dialog_codes(node)]

    assert not comparisons, f'Compare with DIALOG_ACCEPTED or read a button with standardButton() at {comparisons}'
