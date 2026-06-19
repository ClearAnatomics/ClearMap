from __future__ import annotations
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QPalette

from .pipeline_model import LinearPipeline, StepSpec, BINARIZATION_STEPS

_CONNECTOR_H = 18   # shared by _ConnectorArrow and LinearPipelineWidget

# green / red colors compatible with QDarken style, used for the tick/cross marks on the chips.
_TICK_COLOR  = QColor('#5c9e5c')   # muted sage-green
_CROSS_COLOR = QColor('#b85c5c')   # muted dusty-red


class StepChip(QWidget):
    """Single step tile — click to toggle enabled/disabled."""
    #: Triggered when step is toggled
    toggled = pyqtSignal(str, bool)   # step_id, enabled
    #: Triggered when ``keep`` step is toggled
    keep_changed = pyqtSignal(str, bool)   # step_id, keep_intermediate

    _W, _H = 150, 32
    _ICON_ZONE = 35    # px reserved on each side for icons

    def __init__(self, step, spec: StepSpec, parent=None):
        super().__init__(parent)
        self._step = step
        self._spec = spec
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(spec.description)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        pal   = self.palette()
        group = QPalette.Active if self._step.enabled else QPalette.Disabled

        # ---- colors pulled entirely from the current palette ---------------
        bg = pal.color(group, QPalette.Button)

        if self._step.locked:
            border = pal.color(QPalette.Active, QPalette.Mid)
        elif self._step.enabled:
            border = pal.color(QPalette.Active, QPalette.Highlight)
        else:
            border = pal.color(QPalette.Disabled, QPalette.Mid)

        fg = pal.color(group, QPalette.ButtonText)

        # ---- body ----------------------------------------------------------
        p.setBrush(QBrush(bg))
        p.setPen(QPen(border, 2))
        p.drawRoundedRect(1, 1, self._W - 2, self._H - 2, 7, 7)

        # label — centred in the middle zone between the two icon zones
        p.setPen(QPen(fg))
        p.setFont(QFont('Segoe UI', 9, QFont.Bold))
        p.drawText(self._ICON_ZONE, 0, self._W - 2 * self._ICON_ZONE, self._H,
                   Qt.AlignCenter, self._spec.label)

# ---- enabled / disabled tick or cross ------------------------------
        if self._step.enabled:
            mark_color = _TICK_COLOR
            mark = '✓'
        else:
            # Tone the cross down further when the whole step is disabled,
            # so it doesn't fight with the greyed-out label.
            mark_color = _CROSS_COLOR.darker(130) if self._step.locked  else _CROSS_COLOR
            mark = '✗'

        p.setPen(QPen(mark_color))
        p.setFont(QFont('Segoe UI', 10, QFont.Bold))
        p.drawText(0, 0, self._ICON_ZONE, self._H, Qt.AlignCenter, mark)

        # ── keep_intermediate indicator — right zone ──────────────────────
        keep_color = (pal.color(QPalette.Active, QPalette.Highlight)
                      if self._step.keep_intermediate
                      else pal.color(QPalette.Disabled, QPalette.Mid))
        p.setPen(QPen(keep_color))
        p.setFont(QFont('Segoe UI', 8, QFont.Bold))
        action = 'Keep' if self._step.keep_intermediate else 'del'
        p.drawText(self._W - self._ICON_ZONE, 0, self._ICON_ZONE, self._H, Qt.AlignCenter, action)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.x()
            if x <= self._ICON_ZONE: # left zone → toggle enable
                self._step.enabled = not self._step.enabled
                self.update()
                self.toggled.emit(self._step.step_id, self._step.enabled)
            else: # rest of chip → toggle keep_intermediate
                self._step.keep_intermediate = not self._step.keep_intermediate
                self.update()
                self.keep_changed.emit(self._step.step_id, self._step.keep_intermediate)


class _ConnectorArrow(QWidget):
    """
    Self-painting connector placed between two StepChips.
    Draws a white downward arrow (stem + filled head) centred
    over the chip column — no coordination with the parent needed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(_CONNECTOR_H)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        cx    = StepChip._W // 2   # stay centred over the chip column
        h     = self.height()      # == _CONNECTOR_H == 18
        color = QColor('white')

        # ---- stem: thin vertical line ------------------------------------
        p.setPen(QPen(color, 2))
        p.drawLine(cx, 0, cx, h - 7)

        # ---- arrowhead: filled downward triangle -------------------------
        head = QPainterPath()
        head.moveTo(cx,     h - 1)   # tip  (bottom)
        head.lineTo(cx - 5, h - 8)   # top-left corner
        head.lineTo(cx + 5, h - 8)   # top-right corner
        head.closeSubpath()

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(color))
        p.drawPath(head)


class LinearPipelineWidget(QWidget):
    """
    Vertical sequence of StepChips with connecting arrows.
    Steps can be reordered with ▲/▼ buttons except locked steps (binarize).
    """
    #: Triggered when new pipeline selected
    pipeline_changed = pyqtSignal(list)   # list[str] of enabled step names
    #: Triggered when ``keep`` checkbox is toggled on a step chip
    keep_intermediate_changed = pyqtSignal(str, bool)  # spec_name, keep

    CONNECTOR_H = _CONNECTOR_H

    def __init__(self, pipeline: LinearPipeline,
                 step_registry: dict[str, StepSpec] | None = None,
                 parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._registry = step_registry or BINARIZATION_STEPS
        self._chips: list[StepChip] = []
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(0)
        self._populate()

    # ---- layout management ------------------------------------------------

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        """Remove and schedule-delete all child widgets from layout."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _populate(self) -> None:
        self._chips.clear()
        self._clear_layout(self._main_layout)

        steps = self._pipeline.steps
        n = len(steps)

        for i, step in enumerate(steps):
            spec = self._registry.get(step.spec_name)
            if spec is None:
                continue

            is_last = (i == n - 1)
            can_up  = (not step.locked and i > 0 and not steps[i - 1].locked)
            can_dn  = (not step.locked and not is_last)

            # ---- row: chip  +  ▲/▼ column --------------------------------
            row = QWidget(self)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)

            chip = StepChip(step, spec, parent=row)
            chip.toggled.connect(self._on_toggled)
            chip.keep_changed.connect(self._on_keep_changed)
            self._chips.append(chip)
            rl.addWidget(chip)

            # Right-side controls (buttons or lock badge)
            ctrl = QWidget(row)
            ctrl.setFixedWidth(22)
            cl = QVBoxLayout(ctrl)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)

            if step.locked:
                lbl = QLabel('🔒', ctrl)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setFixedSize(20, self._H_for_ctrl())
                cl.addWidget(lbl)
            else:
                for symbol, enabled, direction in (('▲', can_up, -1),
                                                   ('▼', can_dn, +1)):
                    btn = QPushButton(symbol, ctrl)
                    btn.setFixedSize(20, 20)
                    btn.setEnabled(enabled)
                    btn.setStyleSheet('font-size: 8pt; padding: 0;')
                    btn.clicked.connect(
                        lambda _=False, sid=step.step_id, d=direction:
                        self._on_move(sid, d))
                    cl.addWidget(btn)

            rl.addWidget(ctrl)
            self._main_layout.addWidget(row)

            # ---- connector arrow (self-painting) ---
            if not is_last:
                self._main_layout.addWidget(_ConnectorArrow(self))

        self._main_layout.addStretch()

    def _on_keep_changed(self, _sid: str, _keep: bool) -> None:
        # Model already updated by StepChip.mousePressEvent.
        # Emit so the params layer can persist to config.
        step = next((s for s in self._pipeline.steps if s.step_id == _sid), None)
        if step is not None:
            self.keep_intermediate_changed.emit(step.spec_name, _keep)

    @staticmethod
    def _H_for_ctrl() -> int:
        return StepChip._H

    # ---- slots ------------------------------------------------------------

    def _on_toggled(self, _sid: str, _enabled: bool) -> None:
        # self.update()
        self.pipeline_changed.emit(self._pipeline.enabled_steps)

    def _on_move(self, step_id: str, direction: int) -> None:
        self._pipeline.move(step_id, direction)
        self._populate()
        # self.update()
        self.pipeline_changed.emit(self._pipeline.enabled_steps)

    # ---- painting ---------------------------------------------------------

    # def paintEvent(self, event):
    #     super().paintEvent(event)
    #     if len(self._chips) < 2:
    #         return
    #
    #     p = QPainter(self)
    #     p.setRenderHint(QPainter.Antialiasing)
    #     pal = self.palette()
    #
    #     for i in range(len(self._chips) - 1):
    #         src = self._chips[i]
    #         dst = self._chips[i + 1]
    #         active = src._step.enabled and dst._step.enabled
    #
    #         # connector color follows the same palette logic as the chips
    #         color = (pal.color(QPalette.Active,   QPalette.Highlight)
    #                  if active
    #                  else pal.color(QPalette.Disabled, QPalette.Mid))
    #
    #         src_pt = src.mapTo(self, QPoint(src.width() // 2, src.height()))
    #         # top-centre of dst chip in self's coordinates
    #         dst_pt = dst.mapTo(self, QPoint(dst.width() // 2, 0))
    #
    #         x  = src_pt.x()
    #         y0 = src_pt.y()
    #         y1 = dst_pt.y()
    #
    #         p.setPen(QPen(color, 2))
    #         p.setBrush(QBrush(color))
    #         p.drawLine(x, y0, x, y1 - 7)
    #
    #         # downward arrowhead
    #         arrow = QPainterPath()
    #         arrow.moveTo(x, y1)
    #         arrow.lineTo(x - 4, y1 - 7)
    #         arrow.lineTo(x + 4, y1 - 7)
    #         arrow.closeSubpath()
    #         p.drawPath(arrow)

    @property
    def pipeline(self) -> LinearPipeline:
        return self._pipeline