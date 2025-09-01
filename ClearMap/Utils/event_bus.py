from __future__ import annotations

import threading
import weakref
from typing import Callable, Any, Dict, List, Optional

Subscriber = Callable[[str, Any], None]

class EventBus:
    """Minimal, thread-safe pub/sub with weakrefs."""
    def __init__(self) -> None:
        self._subs: Dict[str, List[weakref.ReferenceType]] = {}
        self._lock = threading.RLock()
        self._qt_bridge: Optional[Callable[[str, Any], None]] = None

    def subscribe(self, topic: str, callback: Subscriber) -> Callable[[], None]:
        ref: weakref.ReferenceType
        ref = weakref.WeakMethod(callback) if hasattr(callback, "__self__") else weakref.ref(callback)  # type: ignore
        with self._lock:
            self._subs.setdefault(topic, []).append(ref)

        def _unsub() -> None:
            with self._lock:
                lst = self._subs.get(topic, [])
                lst[:] = [r for r in lst if r is not ref]
        return _unsub

    def publish(self, topic: str, payload: Any = None) -> None:
        with self._lock:
            refs = list(self._subs.get(topic, []))
        dead: List[weakref.ReferenceType] = []
        for ref in refs:
            cb = ref()
            if cb is None:
                dead.append(ref)
                continue
            try:
                cb(topic, payload)
            except Exception:
                # swallow to keep bus robust (or log)
                pass
        if dead:
            with self._lock:
                lst = self._subs.get(topic, [])
                lst[:] = [r for r in lst if r not in dead]
        if self._qt_bridge:
            try: self._qt_bridge(topic, payload)
            except Exception: pass

    def bridge_to_qt(self, forwarder: Callable[[str, Any], None]) -> None:
        """Optionally forward all events to a Qt slot/signal."""
        self._qt_bridge = forwarder
