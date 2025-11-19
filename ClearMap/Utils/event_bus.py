import functools
import inspect
import threading
import weakref
from types import FunctionType
from typing import Callable, Any, Dict, List, Optional, Type, TypeVar


Evt = TypeVar("Evt")
EventSubscriber = Callable[[Evt], None]


class Publishes:
    """
    Marker interface for classes that publish events.
    This is just a nice helper for documentation and introspection.
    """
    def __init__(self, *events: type):
        self.events = frozenset(events)


class EventBus:
    """Minimal, thread-safe publisher/subscriber for Events."""
    def __init__(self) -> None:
        self._subs: Dict[Type[Any], List[weakref.ReferenceType]] = {}
        self._lock = threading.RLock()
        self._qt_bridge: Optional[Callable[[Type[Any], Any], None]] = None

    def _remove_refs(self, event_type: Type[Evt], refs_to_remove: List[weakref.ReferenceType]) -> None:
        if not refs_to_remove:
            return
        with self._lock:
            lst = self._subs.get(event_type, [])
            if not lst:
                return
            # refset = set(refs_to_remove)
            # lst[:] = [r for r in lst if r not in refset]
            lst[:] = [r for r in lst if r() is not None]

    def subscribe(self, event_type: Type[Evt], callback: EventSubscriber[Evt]) -> Callable[[], None]:
        """
        Subscribe a handler for a given event class.
        Handler signature: def handler(evt: EventType) -> None
        Returns: unsubscribe() callable.
        """
        ref: weakref.ReferenceType
        # Methods use WeakMethod; functions use weakref.ref
        ref = weakref.WeakMethod(callback) if hasattr(callback, "__self__") \
            else weakref.ref(callback)

        with self._lock:
            self._subs.setdefault(event_type, []).append(ref)

        def _unsub() -> None:
            self._remove_refs(event_type, [ref])

        return _unsub

    def publish(self, event: Evt) -> None:
        """
        Publish a concrete event instance. Dispatch is on its exact class.
        (No inheritance walk for determinism/perf.)
        """
        event_type = type(event)
        with self._lock:
            refs = list(self._subs.get(event_type, []))

        dead: List[weakref.ReferenceType] = []
        for ref in refs:
            cb = ref()
            if cb is None:
                dead.append(ref)
                continue
            # try:
            cb(event)
            # except Exception as err:
            #     print(str(err))   # keep bus robust  #  TODO: log

        self._remove_refs(event_type, dead)

        if self._qt_bridge:
            self._qt_bridge(event_type, event)

    def bridge_to_qt(self, forwarder: Callable[[Type[Any], Any], None]) -> None:
        """Optionally forward all events to a Qt slot/signal."""
        self._qt_bridge = forwarder


class BusSubscriberMixin:
    def __init__(self, bus: EventBus):
        self._bus = bus
        self._unsubs: List[Callable[[], None]] = []

    @staticmethod
    def _is_lambda(cb: Any) -> bool:
        return isinstance(cb, FunctionType) and cb.__name__ == "<lambda>"

    @staticmethod
    def _is_local_function(cb: Any) -> bool:
        # e.g. def inside __init__ / method body; fragile like lambda
        return isinstance(cb, FunctionType) and "<locals>" in getattr(cb, "__qualname__", "")

    @staticmethod
    def _is_partial(cb: Any) -> bool:
        return isinstance(cb, functools.partial)

    @classmethod
    def _is_partial_of_lambda_or_local(cls, cb: Any) -> bool:
        return cls._is_partial(cb) and (cls._is_lambda(cb.func) or cls._is_local_function(cb.func))

    @staticmethod
    def _is_bound_method(cb: Any) -> bool:
        # Bound method: weakref.WeakMethod-safe and lifetime tied to self
        return inspect.ismethod(cb) and getattr(cb, "__self__", None) is not None

    @staticmethod
    def _is_module_level_function(cb: Any) -> bool:
        # Functions defined at module top-level have a strong ref via module dict
        return isinstance(cb, FunctionType) and "<locals>" not in getattr(cb, "__qualname__", "")

    def subscribe(self, event_type: Type[Evt], handler: EventSubscriber[Evt], *, allow_unsafe: bool = False) -> None:
        unsafe = (
                self._is_lambda(handler)
                or self._is_local_function(handler)
                or self._is_partial_of_lambda_or_local(handler)
        )

        if unsafe and not allow_unsafe:
            raise TypeError(
                "Refusing to subscribe an unsafe handler (lambda/local function). "
                "These are weak-referenced by the EventBus and can be garbage-collected, "
                "silently breaking your subscription. Use a bound method (e.g. self.on_event) "
                "or a module-level function, or pass allow_unsafe=True if you really know "
                "what you’re doing."
            )

        unsub = self._bus.subscribe(event_type, handler)
        self._unsubs.append(unsub)

    def unsubscribe_all(self) -> None:
        for u in self._unsubs:
            try:
                u()
            except Exception:
                pass  # TODO: log
        self._unsubs.clear()

    def publish(self, event) -> None:
        self._bus.publish(event)
