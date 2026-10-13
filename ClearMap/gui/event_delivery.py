"""
Delivery of the events of the :class:`~ClearMap.Utils.event_bus.EventBus` to the GUI handlers, in the GUI thread.

The orchestrators publish from whichever thread they run in (often a worker thread, see
:meth:`~ClearMap.gui.app.ClearMapApp.wrap_in_thread`). The handlers subscribed with the
:data:`~ClearMap.Utils.event_bus.GUI_THREAD` affinity touch widgets, so they must run in the GUI thread:
called from the GUI thread, they run at once (as before); from another thread, the call is queued to the
GUI event loop (which :meth:`~ClearMap.gui.app.ClearMapApp.wrap_in_thread` keeps processing while it waits).
"""
import threading
from typing import Any, Callable

from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot


class GuiThreadDelivery(QObject):
    """
    A :data:`~ClearMap.Utils.event_bus.Delivery` which runs the handlers in the GUI thread.

    .. warning::
        Must be created in the GUI thread (it lives there, and so do the queued calls).
    """
    _queued_call = pyqtSignal(object, object)  # (handler, event)

    def __init__(self):
        super().__init__()
        self._gui_thread_id = threading.get_ident()
        self._queued_call.connect(self._run, type=Qt.QueuedConnection)

    def __call__(self, handler: Callable[[Any], None], event: Any) -> None:
        if threading.get_ident() == self._gui_thread_id:
            handler(event)
        else:
            self._queued_call.emit(handler, event)

    @pyqtSlot(object, object)  # A slot of this QObject, so that the queued call runs in its (the GUI) thread
    def _run(self, handler: Callable[[Any], None], event: Any) -> None:
        handler(event)
