# -*- coding: utf-8 -*-
"""
compound_boxes
==============

The compound boxes of the designer files: a ``QFrame`` which holds one to three spin boxes and, optionally, a check
box which enables them (a singlet, a doublet or a triplet), used for the settings made of a few numbers (an overlap, a
range, a resolution).

In Qt Designer, such a box is a ``QFrame`` promoted to :class:`NumberListBox` (header ``ClearMap.gui.compound_boxes``,
see ``creator/README.rst``), so that the widget which the loader creates has the methods below.
The spin boxes are ordered by the number at the end of their object name (``spinBox_2`` before ``spinBox_10``), or by
the suffix ``min`` and ``max`` for two of them.
"""
import operator
from typing import Callable, Iterable, List, Union

from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QFrame, QSpinBox

from ClearMap.Utils.exceptions import ClearMapValueError

__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2026 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'

NAME_INDEX_SEPARATOR = '_'  # objectName of a spin box: <name>_<index>
MINIMUM_NAME_SUFFIX = 'min'  # The spin boxes of a doublet may also end in min and max
MAXIMUM_NAME_SUFFIX = 'max'
MINIMUM_SPIN_BOX_INDEX, MAXIMUM_SPIN_BOX_INDEX = 0, 1  # Position of the min and max spin boxes of a doublet


class NumberListBox(QFrame):
    """
    A box of one to three spin boxes (int or float), with or without a check box which enables them.

    Without a check box, the box is enabled when its first spin box is. With one, the value of an unchecked box is
    None, and individual sentinel values (-1) are left as is for the caller to interpret (e.g. as 'auto').

    The box is disabled as a whole: there is no per-spin-box disabling here. Showing and converting the sentinel of
    individual values is the job of ``VectorLink`` (``ui_sentinel``, ``enforce_sentinel_min``).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cached_spin_boxes = None

    def getCheckBox(self):
        """The check box which enables the spin boxes, or None"""
        check_boxes = self.findChildren(QCheckBox)
        return check_boxes[0] if check_boxes else None

    def controlsEnabled(self) -> bool:
        """Whether the check box is checked or, without one, the first spin box is enabled"""
        check_box = self.getCheckBox()
        if check_box is not None:
            return check_box.isChecked()
        return self.spinBoxes()[0].isEnabled()

    def enableControls(self) -> None:
        check_box = self.getCheckBox()
        if check_box is not None:
            check_box.setChecked(True)

    def disableControls(self) -> None:
        check_box = self.getCheckBox()
        if check_box is not None:
            check_box.setChecked(False)

    def spinBoxes(self) -> List[Union[QSpinBox, QDoubleSpinBox]]:
        """
        The spin boxes of the box, in the order of their values (see the module docstring)

        Raises
        ------
        ClearMapValueError
            If the order of a spin box cannot be told from its name.
        """
        if self._cached_spin_boxes is None:
            self._cached_spin_boxes = self._sorted_spin_boxes()
        return self._cached_spin_boxes

    def getValue(self):
        """
        The value of the spin box, or the list of the values of the spin boxes (None if the controls are disabled)
        """
        if not self.controlsEnabled():
            return None
        values = [spin_box.value() for spin_box in self.spinBoxes()]
        return values[0] if len(values) == 1 else values  # A single value for a singlet

    def setValue(self, values) -> None:
        """
        Set the value(s) of the spin boxes.

        Parameters
        ----------
        values : Iterable, int, float or None
            The value(s). None disables the controls and shows the special value text of each spin box.

        Raises
        ------
        ClearMapValueError
            If there is a different number of values than spin boxes.
        """
        spin_boxes = self.spinBoxes()
        if values is None:
            self.disableControls()
            for spin_box in spin_boxes:  # Display the text of the "special" value
                if spin_box.specialValueText():
                    spin_box.blockSignals(True)
                    spin_box.setValue(spin_box.minimum())
                    spin_box.blockSignals(False)
            return
        if isinstance(values, (list, tuple)) and len(values) != len(spin_boxes):
            raise ClearMapValueError(f'{self.objectName()} has {len(spin_boxes)} spin boxes, got {len(values)} values',
                                     value=values, expected=f'{len(spin_boxes)} values')
        if not isinstance(values, Iterable) or isinstance(values, str):
            values = [values]

        self.enableControls()
        for i, (value, spin_box) in enumerate(zip(values, spin_boxes)):
            spin_box.blockSignals(i != len(values) - 1)  # Only the last one signals, to call back once
            if isinstance(spin_box, QSpinBox):
                spin_box.setValue(int(value))
            elif isinstance(spin_box, QDoubleSpinBox):
                spin_box.setValue(float(value))
            spin_box.blockSignals(False)

    def valueChangedConnect(self, callback: Callable) -> None:
        """Call callback when the user finishes editing a spin box or toggles the check box"""
        for spin_box in self.spinBoxes():
            spin_box.setKeyboardTracking(False)
            spin_box.editingFinished.connect(callback)
        check_box = self.getCheckBox()
        if check_box is not None:
            check_box.toggled.connect(callback)

    def _sorted_spin_boxes(self) -> List[Union[QSpinBox, QDoubleSpinBox]]:
        """
        Find the spin boxes and sort them by the index in their name (uncached).

        Note
        ----
        Not meant to be called, except by :meth:`spinBoxes`, which caches the result. Use :meth:`spinBoxes`, also
        from inside this class, because looking the children up and sorting them at each call is wasteful.
        """
        spin_boxes = self.findChildren(QSpinBox) or self.findChildren(QDoubleSpinBox)
        indices = [self._spin_box_index(spin_box, n_spin_boxes=len(spin_boxes)) for spin_box in spin_boxes]
        return [spin_box for _, spin_box in sorted(zip(indices, spin_boxes), key=operator.itemgetter(0))]

    def _spin_box_index(self, spin_box, *, n_spin_boxes: int) -> int:
        name = spin_box.objectName().lower()
        try:
            return int(name.split(NAME_INDEX_SEPARATOR)[-1])
        except ValueError:
            pass
        if n_spin_boxes == 2:
            if name.endswith(MINIMUM_NAME_SUFFIX):
                return MINIMUM_SPIN_BOX_INDEX
            if name.endswith(MAXIMUM_NAME_SUFFIX):
                return MAXIMUM_SPIN_BOX_INDEX
        raise ClearMapValueError(f'Could not tell the order of the spin box "{spin_box.objectName()}" in '
                                 f'"{self.objectName()}": its name must end with _<index>'
                                 f'{" or min / max" if n_spin_boxes == 2 else ""}',
                                 value=spin_box.objectName(), expected='a name ending with _<index>')
