gui
===

This package implements the ClearMap Graphical User Interface, built on Qt through qtpy (PyQt6 by default, PyQt5
also works).
It provides a frontend to all ClearMap pipelines — CellMap, TubeMap, TractMap,
Colocalization, and group-level statistics — without requiring any scripting.

The GUI backend is provided by the pipeline workers in
:mod:`ClearMap.pipeline_orchestrators`. The GUI can also be bypassed entirely
for scripted or batch workflows; see :mod:`ClearMap.Scripts` for entry points.

Configuration is managed through YAML files under ``~/.clearmap/`` and in
the experiment directory. These files are created automatically on first run
and are best edited through the **Preferences** dialog or the per-pipeline
tabs in the GUI.


Installation
============

.. code-block:: bash

    git clone https://github.com/ClearAnatomics/ClearMap.git
    cd ClearMap
    chmod +x install_gui.sh
    ./install_gui.sh


Running
=======

.. code-block:: bash

    conda activate ClearMap3.2
    clearmap-ui

The GUI will guide you and create the required config files with sensible defaults that
you can edit through the widgets


Qt binding and theme
====================

ClearMap imports Qt only through `qtpy <https://github.com/spyder-ide/qtpy>`_. The binding is chosen once, at the
start, by :mod:`ClearMap.gui.qt_binding`: the one named by the ``QT_API`` environment variable (``pyqt6`` or
``pyqt5``), otherwise PyQt6 if it is installed, else PyQt5. PySide is not supported.

The environment file ``ClearMap3.yml`` installs PyQt6. To use PyQt5 instead, replace ``pyqt6`` by ``pyqt=5`` in it
before running ``install_gui.sh`` (the installer then also drops the ``qt6-main`` line), or set ``QT_API=pyqt5`` if
both bindings are installed.

The icons are compiled at the first start by the ``rcc`` of the loaded Qt (:mod:`ClearMap.gui.qt_resources`), into
``ClearMap/gui/creator/*.qt5.rcc`` or ``*.qt6.rcc``; nothing to do by hand.

The theme is a preference (**Preferences**, applied at the next start), stored as ``theme`` in the machine
configuration: ``qdarkstyle`` (the default, the look ClearMap is tested with), ``qdarkstyle-light``, or ``system``
(the style of the desktop).

Structure
=========

app
    Main window and application entry point.  Owns the
    ``ExperimentController`` and ``GuiController`` and wires together
    all tabs and dialogs.

tabs
    Concrete tab classes for each pipeline step (Sample info, Stitching,
    Registration, CellMap, TubeMap, TractMap, Colocalization, Group analysis,
    Batch processing).

tabs_interfaces
    Abstract base classes for tabs (``GenericTab``, ``ExperimentTab``,
    ``PipelineTab``, ``PreProcessingTab``, ``PostProcessingTab``,
    ``BatchTab``).

params
    Parameter-link objects that bind GUI widgets to YAML config values for
    each pipeline section.

params_interfaces
    Core UI–config binding machinery: ``UiParameter``, ``ParamLink``,
    ``VectorLink``, ``WidgetOps``, and related helpers.

params_mixins
    Reusable mixin behaviours for parameter classes (e.g. ortho-viewer
    slicing).

widgets
    Custom Qt widgets (``Scatter3D``, ``ExtendableTabWidget``,
    ``BlockProcessingWidget``, ``NProcessesWidget``, progress watcher, etc.).

pipeline_model
    Data model for the configurable binarization pipeline widget
    (``LinearPipeline``, ``PipelineStep``).

pipeline_widgets
    Qt widget that renders a ``LinearPipeline`` and lets the user reorder
    and toggle steps.

dialogs
    Custom dialogs (prompts, warnings, file drop, landmark selector, resource-type-to-folder
    editor, about box, etc.).

dialog_helpers
    Utility functions for common dialogs (directory picker, progress bars,
    splash screen, popups).

preferences
    Preferences UI and its ``PreferenceUi`` controller.

gui_logging
    ``Printer`` widget for normal and error logging to the in-app text area
    and log files. This is currently used as a hack to track progress.

gui_utils_base
    Low-level Qt helpers (widget replacement, layout utilities, grid
    computation, etc.).

gui_utils_images
    Image conversion utilities for GUI display (numpy → QPixmap, etc.).

style
    The themes (style sheet, message and tooltip colours of each), colour constants and stylesheet fragments used
    throughout the GUI.

qt_binding
    Chooses the Qt binding (``QT_API``) before any Qt import, for qtpy and pyqtgraph.

qt_resources
    Compiles the icon resources with the ``rcc`` of the loaded Qt and registers them.

tab_registry
    ``TabRegistry``: determines which tabs are valid given the current
    sample state and app mode.

widget_monkeypatch_callbacks
    Functions bound at runtime to compound Qt widgets (doublets, triplets,
    etc.) to give them a uniform value-changed interface.

event_bus integration
    The GUI communicates with backend workers through typed events
    on the ``EventBus`` (see :mod:`ClearMap.Utils.event_bus`).


Bugs
====

Please report bugs on the `GitHub issue tracker`_ using the **GUI** label.

.. _GitHub issue tracker: https://github.com/ClearAnatomics/ClearMap/issues