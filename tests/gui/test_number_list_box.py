"""
The compound boxes of the designer files (singlets, doublets, triplets): the order of their spin boxes, what they
read and write, and what they tell when the controls are disabled. The spin boxes and the check box are fakes of the
Qt classes (the sandbox has no Qt): the box is built into a real one by test_designer_files_build_with_the_qt_binding.
"""
import pytest
from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QSpinBox

from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.gui.compound_boxes import NumberListBox

NO_SPECIAL_VALUE_TEXT = ''


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class FakeSpinMixin:
    def init_fake(self, name, value=0, minimum=-1, special_value_text=NO_SPECIAL_VALUE_TEXT):
        self._name, self._value, self._minimum = name, value, minimum
        self._special_value_text = special_value_text
        self._enabled = True
        self.blocked = False
        self.values_set = []  # (value, were the signals blocked)
        self.keyboard_tracking = True
        self.editingFinished = FakeSignal()
        return self

    def objectName(self):
        return self._name

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value
        self.values_set.append((value, self.blocked))

    def minimum(self):
        return self._minimum

    def specialValueText(self):
        return self._special_value_text

    def blockSignals(self, blocked):
        self.blocked = blocked

    def isEnabled(self):
        return self._enabled

    def setKeyboardTracking(self, tracking):
        self.keyboard_tracking = tracking


class FakeSpinBox(FakeSpinMixin, QSpinBox):
    pass


class FakeDoubleSpinBox(FakeSpinMixin, QDoubleSpinBox):
    pass


class FakeCheckBox(QCheckBox):
    def __init__(self, checked=False):
        super().__init__()
        self._checked = checked
        self.toggled = FakeSignal()

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        self._checked = checked


class FakeNumberListBox(NumberListBox):
    """A box whose children are given, in the order of the designer file, as findChildren answers them"""
    def __init__(self, children, name='fooDoublet'):
        super().__init__()
        self._children, self._name = children, name

    def objectName(self):
        return self._name

    def findChildren(self, widget_class, *args):
        return [child for child in self._children if isinstance(child, widget_class)]


def spin(name, **kwargs):
    return FakeSpinBox().init_fake(name, **kwargs)


def double_spin(name, **kwargs):
    return FakeDoubleSpinBox().init_fake(name, **kwargs)


def values_in_order(box):
    return [spin_box.objectName() for spin_box in box.spinBoxes()]


@pytest.mark.parametrize('names, ordered', [
    (['spinBox_10', 'spinBox_2', 'spinBox_7'], ['spinBox_2', 'spinBox_7', 'spinBox_10']),  # By number, not by text
    (['rangeMax', 'rangeMin'], ['rangeMin', 'rangeMax']),
    (['spinBox_3'], ['spinBox_3']),
])
def test_the_spin_boxes_are_ordered_by_the_number_or_the_min_max_suffix_of_their_names(names, ordered):
    box = FakeNumberListBox([spin(name) for name in names])

    assert values_in_order(box) == ordered


@pytest.mark.parametrize('names', [['spinBox_a', 'spinBox_b', 'spinBox_c'], ['lower', 'upper']])
def test_spin_boxes_whose_order_cannot_be_told_are_refused_naming_the_box(names):
    box = FakeNumberListBox([spin(name) for name in names], name='fooDoublet')

    with pytest.raises(ClearMapValueError, match='fooDoublet'):
        box.spinBoxes()


def test_the_spin_boxes_are_the_double_spin_boxes_when_there_is_no_int_one():
    box = FakeNumberListBox([double_spin('spinBox_1'), double_spin('spinBox_0')])

    assert values_in_order(box) == ['spinBox_0', 'spinBox_1']


def test_a_triplet_gives_the_list_of_its_values_and_a_singlet_the_value():
    triplet = FakeNumberListBox([spin('s_0', value=1), spin('s_1', value=2), spin('s_2', value=3)])
    singlet = FakeNumberListBox([spin('s_0', value=5)])

    assert (triplet.getValue(), singlet.getValue()) == ([1, 2, 3], 5)


def test_unchecked_controls_give_none_whatever_the_spin_boxes_hold():
    box = FakeNumberListBox([FakeCheckBox(checked=False), spin('s_0', value=4), spin('s_1', value=6)])

    assert box.controlsEnabled() is False and box.getValue() is None


def test_without_a_check_box_the_box_is_enabled_when_its_first_spin_box_is():
    box = FakeNumberListBox([spin('s_0'), spin('s_1')])
    box.spinBoxes()[0]._enabled = False

    assert box.controlsEnabled() is False and box.getValue() is None


def test_setting_values_checks_the_box_and_converts_them_to_int_for_int_spin_boxes():
    check_box = FakeCheckBox(checked=False)
    first, second = spin('s_0'), spin('s_1')
    box = FakeNumberListBox([check_box, first, second])

    box.setValue([3.0, 2])

    assert check_box.isChecked()
    assert [first.value(), second.value()] == [3, 2] and isinstance(first.value(), int)


def test_setting_values_converts_them_to_float_for_double_spin_boxes():
    first, second = double_spin('s_0'), double_spin('s_1')
    box = FakeNumberListBox([first, second])

    box.setValue([3, 2])

    assert [first.value(), second.value()] == [3.0, 2.0] and isinstance(first.value(), float)


def test_only_the_last_spin_box_signals_when_values_are_set():
    first, second, third = spin('s_0'), spin('s_1'), spin('s_2')
    box = FakeNumberListBox([first, second, third])

    box.setValue([1, 2, 3])

    assert [spin_box.values_set for spin_box in (first, second, third)] == [[(1, True)], [(2, True)], [(3, False)]]
    assert not any(spin_box.blocked for spin_box in (first, second, third))


def test_a_scalar_sets_the_only_spin_box_of_a_singlet():
    only = spin('s_0')
    box = FakeNumberListBox([only])

    box.setValue(7)

    assert only.value() == 7


def test_setting_none_unchecks_the_box_and_shows_the_special_value_text_without_signals():
    check_box = FakeCheckBox(checked=True)
    with_text = spin('s_0', value=9, minimum=-1, special_value_text='auto')
    without_text = spin('s_1', value=9)
    box = FakeNumberListBox([check_box, with_text, without_text])

    box.setValue(None)

    assert not check_box.isChecked()
    assert with_text.values_set == [(-1, True)] and not with_text.blocked
    assert without_text.values_set == []


def test_a_list_of_the_wrong_length_is_refused_naming_the_box():
    box = FakeNumberListBox([spin('s_0'), spin('s_1')], name='projectionDoublet')

    with pytest.raises(ClearMapValueError, match='projectionDoublet has 2 spin boxes, got 3 values'):
        box.setValue([1, 2, 3])


def test_user_edits_and_the_check_box_call_back_and_typing_does_not():
    check_box = FakeCheckBox()
    first, second = spin('s_0'), spin('s_1')
    box = FakeNumberListBox([check_box, first, second])
    calls = []

    box.valueChangedConnect(calls.append)

    assert first.editingFinished.callbacks == second.editingFinished.callbacks == [calls.append]
    assert check_box.toggled.callbacks == [calls.append]
    assert not first.keyboard_tracking and not second.keyboard_tracking
