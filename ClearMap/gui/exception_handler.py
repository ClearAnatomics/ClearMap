"""
exception_handler
=================

Centralised GUI exception handling for ClearMap.

Usage
-----

From any tab or controller::

    from ClearMap.gui.exception_handler import handle_exception

    try:
        worker.run()
    except ClearMapException as exc:
        action = handle_exception(exc, parent=self.main_window)
        if action == 'retry':
            ...
        elif action == 'reset_workspace':
            self.main_window.trigger_workspace_reset()

At application startup::

    from ClearMap.gui.exception_handler import install_global_handler
    install_global_handler(app, parent_getter=lambda: gui.window)
"""
from __future__ import annotations

import sys
import traceback as tb_module
from typing import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QStyle, QTextEdit, QVBoxLayout, QWidget)

from ClearMap.Utils.exceptions import DISMISS, ClearMapException, Severity

# ---------------------------------------------------------------------------
# Severity → visual mapping
# ---------------------------------------------------------------------------

_SEVERITY_ICON = {
    Severity.RECOVERABLE: QStyle.SP_MessageBoxInformation,
    Severity.DEGRADED:    QStyle.SP_MessageBoxWarning,
    Severity.FATAL:       QStyle.SP_MessageBoxCritical,
    Severity.CRITICAL:    QStyle.SP_MessageBoxCritical,
}

_SEVERITY_COLOR = {
    Severity.RECOVERABLE: '#5c9e5c',   # muted green
    Severity.DEGRADED:    '#d4a017',   # amber
    Severity.FATAL:       '#d35400',   # orange
    Severity.CRITICAL:    '#c0392b',   # red
}

_SEVERITY_LABEL = {
    Severity.RECOVERABLE: '',
    Severity.DEGRADED:    ' (degraded)',
    Severity.FATAL:       '',
    Severity.CRITICAL:    ' — Action Required',
}


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ExceptionDialog(QDialog):
    """
    Modal dialog that presents a :class:`ClearMapException` to the user.
    After ``exec_()``, read :attr:`chosen_action` for the action string
    corresponding to the :class:`RecoveryOption` the user clicked.
    """

    def __init__(self, exc: ClearMapException, parent: QWidget | None = None, context: str = ''):
        super().__init__(parent)
        self.exc = exc
        self.chosen_action: str = 'dismiss'
        self._build(context)

    # ------------------------------------------------------------------ #

    def _build(self, context: str) -> None:
        sev = self.exc.severity
        color = _SEVERITY_COLOR.get(sev, 'white')
        suffix = _SEVERITY_LABEL.get(sev, '')

        self.setWindowTitle(f'{sev.value.title()}{suffix} — {self.exc.user_title}')
        self.setMinimumWidth(480)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        root = QVBoxLayout(self)
        root.setSpacing(12)

        # ---- header: icon + title ----
        header = QHBoxLayout()
        icon_lbl = QLabel()
        icon_enum = _SEVERITY_ICON.get(sev, QStyle.SP_MessageBoxWarning)
        icon_lbl.setPixmap(self.style().standardIcon(icon_enum).pixmap(48, 48))
        header.addWidget(icon_lbl)

        title_lbl = QLabel(self.exc.user_title)
        title_lbl.setStyleSheet(f'font-size: 15pt; font-weight: bold; color: {color};')
        title_lbl.setWordWrap(True)
        header.addWidget(title_lbl, stretch=1)
        root.addLayout(header)

        # ---- separator ----
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        root.addWidget(line)

        # ---- message body ----
        body = QLabel(self.exc.user_message)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root.addWidget(body)

        # ---- hint ----
        hint_text = self.exc.user_hint  # may be property
        if hint_text:
            hint = QLabel(f'<b>Hint:</b> {hint_text}')
            hint.setTextFormat(Qt.RichText)
            hint.setWordWrap(True)
            hint.setStyleSheet('color: #74c69d; margin-top: 4px;')
            root.addWidget(hint)

        # ---- CRITICAL banner ----
        if sev is Severity.CRITICAL:
            banner = QLabel('⚠️  The workspace may be in an inconsistent state.\n'
                            'It is strongly recommended to reset and reload.')
            banner.setWordWrap(True)
            banner.setStyleSheet('background-color: rgba(192, 57, 43, 0.15); '
                                 'border: 1px solid #c0392b; border-radius: 4px; '
                                 'padding: 8px; color: #e74c3c; font-weight: bold;')
            root.addWidget(banner)

        # ---- context (small grey) ----
        if context:
            ctx = QLabel(f'Context: {context}')
            ctx.setStyleSheet('color: #888; font-size: 9pt;')
            root.addWidget(ctx)

        # ---- expandable technical details ----
        self._details_text = QTextEdit()
        self._details_text.setReadOnly(True)
        self._details_text.setVisible(False)
        self._details_text.setMaximumHeight(180)
        self._details_text.setStyleSheet('font-family: monospace; font-size: 9pt;')

        tb_str = ''
        if self.exc.__traceback__:
            tb_str = ''.join(tb_module.format_exception(type(self.exc), self.exc, self.exc.__traceback__))
        self._details_text.setPlainText(f'{type(self.exc).__name__}: {self.exc}\n\n{tb_str}')

        toggle_btn = QPushButton('▶  Technical details')
        toggle_btn.setFlat(True)
        toggle_btn.setStyleSheet('text-align: left; color: #888; padding: 0;')
        toggle_btn.clicked.connect(lambda: self._toggle_details(toggle_btn))

        copy_btn = QPushButton('📋 Copy')
        copy_btn.setFixedWidth(70)
        copy_btn.setVisible(False)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self._details_text.toPlainText()))
        self._copy_btn = copy_btn

        details_header = QHBoxLayout()
        details_header.addWidget(toggle_btn)
        details_header.addStretch()
        details_header.addWidget(copy_btn)
        root.addLayout(details_header)
        root.addWidget(self._details_text)

        # ---- action buttons ----
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        options = self.exc.recovery_options or (DISMISS,)
        for opt in options:
            btn = QPushButton(opt.label)
            if opt.tooltip:
                btn.setToolTip(opt.tooltip)
            if opt.is_destructive:
                btn.setStyleSheet('QPushButton { background-color: #c0392b; '
                                  'color: white; font-weight: bold; padding: 6px 16px; }'
                                  'QPushButton:hover { background-color: #e74c3c; }')
            else:
                btn.setStyleSheet('padding: 6px 16px;')
            if opt.is_default:
                btn.setDefault(True)
                btn.setFocus()
            btn.clicked.connect(lambda _=False, a=opt.action: self._choose(a))
            btn_row.addWidget(btn)

        root.addLayout(btn_row)

    def _toggle_details(self, toggle_btn: QPushButton) -> None:
        show = not self._details_text.isVisible()
        self._details_text.setVisible(show)
        self._copy_btn.setVisible(show)
        toggle_btn.setText('▼  Technical details' if show else '▶  Technical details')
        self.adjustSize()

    def _choose(self, action: str) -> None:
        self.chosen_action = action
        self.accept()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def handle_exception(exc: Exception, parent: QWidget | None = None, context: str = '') -> str:
    """
    Show an exception dialog and return the chosen action string.
    For non-ClearMap exceptions a generic FATAL dialog is shown.

    Parameters
    ----------
    exc
        The exception to present.
    parent
        Parent widget for the dialog (centres it on the parent window).
    context
        Optional extra line (e.g. ``"CellCounterTab.detect_cells"``).

    Returns
    -------
    str
        The ``action`` field of the chosen :class:`RecoveryOption`,
        e.g. ``'dismiss'``, ``'retry'``, ``'reset_workspace'``, ``'close'``.
    """
    try:
        if not isinstance(exc, ClearMapException):
            wrapper = ClearMapException(str(exc))
            wrapper.__cause__ = exc
            wrapper.__traceback__ = exc.__traceback__
            wrapper.user_title = type(exc).__name__
            exc = wrapper

        dlg = ExceptionDialog(exc, parent=parent, context=context)
        dlg.exec_()
        return dlg.chosen_action

    except Exception as meta_exc:
        # Last resort: if the dialog itself crashes, print to stderr.
        print(f'CRITICAL: exception handler failed ({meta_exc}); original error: {exc}', file=sys.stderr)
        tb_module.print_exception(type(exc), exc, exc.__traceback__)
        return 'dismiss'


def install_global_handler(app: QApplication, parent_getter: Callable[[], QWidget | None] = lambda: None,
                           on_reset: Callable[[], None] | None = None, on_close: Callable[[], None] | None = None) -> None:
    """Install a ``sys.excepthook`` that shows :func:`handle_exception`.

    Parameters
    ----------
    app
        The running QApplication.
    parent_getter
        Callable returning the current main window (or ``None``).
    on_reset
        Callback for the ``'reset_workspace'`` action.
        When ``None``, falls back to ``app.quit()``.
    on_close
        Callback for the ``'close'`` action.
        When ``None``, calls ``app.quit()``.
    """
    _original = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        tb_module.print_exception(exc_type, exc_value, exc_tb)  # Always log to stderr first.
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            _original(exc_type, exc_value, exc_tb)
            return

        # Prevent double-dialog if wrap_step/wrap_plot already handled it
        if getattr(exc_value, '_gui_handled', False):
            return

        exc_value.__traceback__ = exc_tb
        action = handle_exception(exc_value, parent=parent_getter(), context='Unhandled exception')
        _dispatch_action(action, app, on_reset, on_close)

    sys.excepthook = _hook


def _dispatch_action(action: str, app: QApplication, on_reset: Callable | None, on_close: Callable | None) -> None:
    """Execute the user's chosen recovery action."""
    if action == 'close':
        (on_close or app.quit)()
    elif action == 'reset_workspace':
        if on_reset:
            on_reset()
        else:
            app.quit()
    # 'dismiss', 'skip', 'retry', etc. → caller handles or no-op