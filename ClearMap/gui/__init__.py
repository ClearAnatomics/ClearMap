"""
This subpackage contains the graphical user interface for ClearMap.

The GUI is used to run the ClearMap pipelines without much customisation
and to visualize the results.

For a more advanced usage, the ClearMap classes can be used directly. See
:mod:`cell_map_new_api` and :mod:`tube_map_new_api` for more information.
"""

from ClearMap.gui.qt_binding import choose_qt_binding

choose_qt_binding()  # Before the package imports Qt: qtpy and pyqtgraph read the binding when first imported
