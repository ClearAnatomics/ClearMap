import functools
import inspect
import threading
import traceback
import warnings
import weakref
from types import FunctionType
from typing import Callable, Any, Dict, List, Optional, Type, TypeVar

from ClearMap.Utils.exceptions import SubscriberFailedWarning


Evt = TypeVar("Evt")
EventSubscriber = Callable[[Evt], None]
Delivery = Callable[[Callable[[Any], None], Any], None]  # delivery(handler, event): runs handler(event) on its thread

GUI_THREAD = 'gui_thread'
"""
Affinity of the handlers which must run in the GUI thread (they touch widgets). The application registers its
delivery (:meth:`EventBus.set_delivery`); headless (e.g. in the tests) nothing is, and they are called directly.
"""


def report_subscriber_failure(handler: Callable[[Any], None], event: Any) -> None:
    """Report the exception being handled, raised by *handler* while it handled *event*"""
    warnings.warn(f'{getattr(handler, "__qualname__", handler)} failed on {type(event).__name__}:\n'
                  f'{traceback.format_exc()}', SubscriberFailedWarning, stacklevel=2)


def deliver_directly(handler: Callable[[Any], None], event: Any) -> None:
    """The default delivery: call the handler now, in the thread of the publisher"""
    handler(event)


class Publishes:
    """
    Marker interface for classes that publish events.
    This is just a nice helper for documentation and introspection.
    """
    def __init__(self, *events: type):
        self.events = frozenset(events)


class _Subscription:
    """One call to :meth:`EventBus.subscribe`: the (weak) handler and where it must run. Compared by identity."""
    __slots__ = ('ref', 'affinity')

    def __init__(self, ref: weakref.ReferenceType, affinity: Optional[str]):
        self.ref = ref
        self.affinity = affinity


class EventBus:
    """
    Minimal, thread-safe publisher/subscriber for Events.

    Events may be published from any thread (e.g. by an orchestrator running in a worker thread).
    Each handler runs according to the *affinity* it was subscribed with: without affinity, it is called
    directly, in the thread of the publisher; otherwise through the delivery registered for its affinity
    (e.g. the :data:`GUI_THREAD` handlers run in the GUI thread). The bus itself knows nothing of Qt.

    A handler that raises is reported (:class:`SubscriberFailedWarning`): the other handlers still receive the
    event, and the publisher does not see the exception (it is the subscriber's, not the publisher's).
    """
    def __init__(self) -> None:
        self._subs: Dict[Type[Any], List[_Subscription]] = {}
        self._deliveries: Dict[str, Delivery] = {}
        self._lock = threading.RLock()
        self._qt_bridge: Optional[Callable[[Type[Any], Any], None]] = None

    def set_delivery(self, affinity: str, delivery: Delivery) -> None:
        """
        Register how the handlers subscribed with *affinity* are run.

        Parameters
        ----------
        affinity: str
            The affinity (e.g. :data:`GUI_THREAD`)
        delivery: Delivery
            Called as delivery(handler, event) in the thread of the publisher. It must call handler(event)
            in the right thread, now or later. A late call is safe: the handler it is given does nothing if
            its subscriber was unsubscribed (or collected) in the meantime.
        """
        with self._lock:
            self._deliveries[affinity] = delivery

    def _delivery_for(self, affinity: Optional[str]) -> Delivery:
        if affinity is None:
            return deliver_directly
        with self._lock:
            return self._deliveries.get(affinity, deliver_directly)  # Headless: nothing registered

    def _remove_subscriptions(self, event_type: Type[Evt], to_remove: List[_Subscription]) -> None:
        """Remove these subscriptions (by identity: one per call to subscribe), and the dead ones"""
        if not to_remove:
            return
        with self._lock:
            lst = self._subs.get(event_type, [])
            if not lst:
                return
            lst[:] = [sub for sub in lst if sub.ref() is not None and not any(sub is gone for gone in to_remove)]

    def subscribe(self, event_type: Type[Evt], callback: EventSubscriber[Evt], *,
                  affinity: Optional[str] = None) -> Callable[[], None]:
        """
        Subscribe a handler for a given event class.
        Handler signature: def handler(evt: EventType) -> None
        affinity: where the handler must run (e.g. :data:`GUI_THREAD`), None for the thread of the publisher.
        Returns: unsubscribe() callable.
        """
        # Methods use WeakMethod; functions use weakref.ref
        ref = weakref.WeakMethod(callback) if hasattr(callback, "__self__") else weakref.ref(callback)
        subscription = _Subscription(ref, affinity)

        with self._lock:
            self._subs.setdefault(event_type, []).append(subscription)

        def _unsub() -> None:
            self._remove_subscriptions(event_type, [subscription])

        return _unsub

    def publish(self, event: Evt) -> None:
        """
        Publish a concrete event instance, from any thread. Dispatch is on its exact class.
        (No inheritance walk for determinism/perf.)
        """
        if isinstance(event, type):
            raise TypeError(f'Publish an instance of {event.__name__}, not the class: the dispatch is on the type '
                            f'of the event, so no subscriber would receive it.')
        event_type = type(event)
        with self._lock:
            subscriptions = list(self._subs.get(event_type, []))

        dead: List[_Subscription] = []
        for subscription in subscriptions:
            if subscription.ref() is None:
                dead.append(subscription)
                continue
            handler = functools.partial(self._call_if_subscribed, event_type, subscription)
            self._delivery_for(subscription.affinity)(handler, event)

        self._remove_subscriptions(event_type, dead)

        if self._qt_bridge:
            self._qt_bridge(event_type, event)

    def _call_if_subscribed(self, event_type: Type[Any], subscription: _Subscription, event: Any) -> None:
        """Run the handler of *subscription*, unless it was unsubscribed (or collected) since the event was published"""
        with self._lock:
            subscribed = any(sub is subscription for sub in self._subs.get(event_type, ()))
        cb = subscription.ref() if subscribed else None
        if cb is None:
            return
        try:
            cb(event)
        except Exception:  # The failure of one subscriber is its own: the others and the publisher go on
            report_subscriber_failure(cb, event)

    def bridge_to_qt(self, forwarder: Callable[[Type[Any], Any], None]) -> None:
        """Optionally forward all events to a Qt slot/signal."""
        self._qt_bridge = forwarder


class BusSubscriberMixin:
    subscriber_affinity: Optional[str] = None
    """Where the handlers of this subscriber run (see :class:`EventBus`); the GUI classes set :data:`GUI_THREAD`"""

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

        unsub = self._bus.subscribe(event_type, handler, affinity=self.subscriber_affinity)
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
