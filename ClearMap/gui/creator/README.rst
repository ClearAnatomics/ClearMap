Designer files
==============

The ``.ui`` files of the GUI, edited with Qt Designer. The compiled resources (``.qrc``) are registered with Qt when
the GUI starts (``ClearMap.gui.qt_resources``).

To see a file in Designer as it looks in the GUI (dark style, with the real colours of the widgets), start Designer
with the style sheet of the Qt binding of the environment. The small icons of qdarkstyle (arrows, check marks) are
Qt resources that only Python registers, so the style sheet is pointed at their files instead::

    python - <<'EOF'
    from pathlib import Path
    import qdarkstyle
    from qtpy import API
    icons_folder = Path(qdarkstyle.__file__).parent / 'dark' / 'rc'
    style_sheet = qdarkstyle.load_stylesheet(qt_api=API).replace(':/qss_icons/dark/rc/', f'{icons_folder}/')
    Path('/tmp/clearmap_dark.qss').write_text(style_sheet)
    EOF
    designer -stylesheet /tmp/clearmap_dark.qss ClearMap/gui/creator/sample_tab.ui

qdarkstyle prints "No QCoreApplication instance found": it only means that its application patches are not applied,
which Designer does not need. The workspace around the form in Designer stays light grey (a rule on ``QMdiArea``
does not change it).

Do not set a ``styleSheet`` on the root widget of a file to get a dark preview: it would apply in the GUI too
(the loader, ``qtpy.uic.loadUi``, sets every property of the root widget).

Compound boxes: singlets, doublets and triplets
-----------------------------------------------

A setting made of one to three numbers (an overlap, a range, a resolution) is a ``QFrame`` which holds the spin boxes
and, optionally, a check box which enables them. In Designer, promote the ``QFrame`` to ``NumberListBox`` (right click,
"Promote to...": base class ``QFrame``, class name ``NumberListBox``, header file ``ClearMap.gui.compound_boxes``), and
name it after its number of spin boxes: ``...Singlet``, ``...Doublet`` or ``...Triplet``. The order of the values is the
number at the end of the names of the spin boxes (``spinBox_1``, ``spinBox_2``; two spin boxes may end in ``min`` and
``max`` instead). A frame which is not promoted has no methods and is refused when it is linked to a setting.
