"""The event bus: subscriptions, and where the handlers run (delivery)"""
import importlib
import threading
from dataclasses import dataclass

import pytest

from ClearMap.Utils.event_bus import EventBus, BusSubscriberMixin, GUI_THREAD
from ClearMap.Utils.exceptions import SubscriberFailedWarning


@dataclass(frozen=True)
class Ping:
    value: int = 0


class Recorder:
    def __init__(self):
        self.events = []

    def on_ping(self, event):
        self.events.append(event)


class FailingSubscriber:
    def on_ping(self, event):
        raise RuntimeError(f'cannot handle {event}')

def test_unsubscribe_stops_the_delivery():
    """unsubscribe() used to only purge the dead subscribers: a live one kept receiving the events"""
    bus, recorder = EventBus(), Recorder()
    unsubscribe = bus.subscribe(Ping, recorder.on_ping)
    bus.publish(Ping(1))
    unsubscribe()
    bus.publish(Ping(2))

    assert recorder.events == [Ping(1)]


def test_unsubscribe_only_removes_its_own_subscription():
    bus, first, second = EventBus(), Recorder(), Recorder()
    unsubscribe_first = bus.subscribe(Ping, first.on_ping)
    bus.subscribe(Ping, second.on_ping)
    unsubscribe_first()
    bus.publish(Ping(1))

    assert (first.events, second.events) == ([], [Ping(1)])


def test_the_same_handler_subscribed_twice_is_unsubscribed_once_per_call():
    bus, recorder = EventBus(), Recorder()
    unsubscribe = bus.subscribe(Ping, recorder.on_ping)
    bus.subscribe(Ping, recorder.on_ping)
    unsubscribe()
    bus.publish(Ping(1))

    assert recorder.events == [Ping(1)]


# ----------------------------------------------------------------------------------------------------------------
# Delivery: where the handlers run
# ----------------------------------------------------------------------------------------------------------------

class QueuedDelivery:
    """Stand-in for the GUI delivery: the calls are queued, then run by whoever owns the queue (the 'GUI thread')"""

    def __init__(self):
        self.queue = []

    def __call__(self, handler, event):
        self.queue.append((handler, event))

    def run_pending(self):
        pending, self.queue = self.queue, []
        for handler, event in pending:
            handler(event)


class ThreadRecorder(Recorder):
    def on_ping(self, event):
        super().on_ping(event)
        self.threads = getattr(self, 'threads', []) + [threading.get_ident()]


def _publish_from_a_worker_thread(bus, event):
    worker = threading.Thread(target=bus.publish, args=(event,))
    worker.start()
    worker.join()


def test_handlers_without_affinity_run_at_once_in_the_thread_of_the_publisher():
    bus, recorder = EventBus(), ThreadRecorder()
    bus.subscribe(Ping, recorder.on_ping)
    _publish_from_a_worker_thread(bus, Ping(1))

    assert recorder.events == [Ping(1)]
    assert recorder.threads[0] != threading.get_ident()


def test_without_registered_delivery_the_gui_handlers_are_called_directly():
    """Headless (tests, scripts): nothing is registered for GUI_THREAD, the handlers are called like the others"""
    bus, recorder = EventBus(), Recorder()
    bus.subscribe(Ping, recorder.on_ping, affinity=GUI_THREAD)
    bus.publish(Ping(1))

    assert recorder.events == [Ping(1)]


def test_gui_handlers_published_from_a_worker_run_in_the_gui_thread():
    bus, gui_handler, headless_handler, delivery = EventBus(), ThreadRecorder(), ThreadRecorder(), QueuedDelivery()
    bus.set_delivery(GUI_THREAD, delivery)
    bus.subscribe(Ping, gui_handler.on_ping, affinity=GUI_THREAD)
    bus.subscribe(Ping, headless_handler.on_ping)

    _publish_from_a_worker_thread(bus, Ping(1))
    assert gui_handler.events == [] and headless_handler.events == [Ping(1)]  # Not yet: queued for the GUI thread

    delivery.run_pending()  # The GUI event loop
    assert gui_handler.events == [Ping(1)]
    assert gui_handler.threads == [threading.get_ident()]


def test_a_queued_event_is_not_delivered_to_a_subscriber_unsubscribed_meanwhile():
    """e.g. a tab torn down between the publication in the worker thread and the GUI event loop"""
    bus, recorder, delivery = EventBus(), Recorder(), QueuedDelivery()
    bus.set_delivery(GUI_THREAD, delivery)
    unsubscribe = bus.subscribe(Ping, recorder.on_ping, affinity=GUI_THREAD)
    bus.publish(Ping(1))
    unsubscribe()
    delivery.run_pending()

    assert recorder.events == []


def test_the_subscriber_mixin_subscribes_with_its_affinity():
    class GuiSubscriber(BusSubscriberMixin):
        subscriber_affinity = GUI_THREAD

        def __init__(self, bus):
            super().__init__(bus)
            self.events = []
            self.subscribe(Ping, self.on_ping)

        def on_ping(self, event):
            self.events.append(event)

    bus, delivery = EventBus(), QueuedDelivery()
    bus.set_delivery(GUI_THREAD, delivery)
    subscriber = GuiSubscriber(bus)
    bus.publish(Ping(1))
    assert subscriber.events == []
    delivery.run_pending()
    assert subscriber.events == [Ping(1)]


@pytest.mark.parametrize('class_path', ['ClearMap.gui.params_interfaces.UiParameter',
                                        'ClearMap.gui.params_interfaces.UiParameterCollection',
                                        'ClearMap.gui.tabs_interfaces.GenericTab'])
def test_the_gui_base_classes_run_their_handlers_in_the_gui_thread(class_path):
    module_name, class_name = class_path.rsplit('.', 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    assert cls.subscriber_affinity == GUI_THREAD


def test_publishing_an_event_class_instead_of_an_instance_fails_loudly():
    """`publish(RegistrationStatusChanged)` (no parentheses) reached nobody: the dispatch is on type(event)"""
    bus, recorder = EventBus(), Recorder()
    bus.subscribe(Ping, recorder.on_ping)

    with pytest.raises(TypeError, match='instance of Ping'):
        bus.publish(Ping)


def test_a_failing_subscriber_is_reported_and_neither_stops_the_others_nor_reaches_the_publisher():
    bus, failing, recorder = EventBus(), FailingSubscriber(), Recorder()
    bus.subscribe(Ping, failing.on_ping)
    bus.subscribe(Ping, recorder.on_ping)

    with pytest.warns(SubscriberFailedWarning, match='FailingSubscriber.on_ping failed on Ping'):
        bus.publish(Ping(1))  # Does not raise

    assert recorder.events == [Ping(1)]


def test_a_failing_gui_subscriber_does_not_break_the_delivery_in_the_gui_thread():
    """The queued call runs in a Qt slot: an exception escaping it would abort the application"""
    bus, failing, recorder, delivery = EventBus(), FailingSubscriber(), Recorder(), QueuedDelivery()
    bus.set_delivery(GUI_THREAD, delivery)
    bus.subscribe(Ping, failing.on_ping, affinity=GUI_THREAD)
    bus.subscribe(Ping, recorder.on_ping, affinity=GUI_THREAD)
    _publish_from_a_worker_thread(bus, Ping(1))

    with pytest.warns(SubscriberFailedWarning):
        delivery.run_pending()  # Does not raise

    assert recorder.events == [Ping(1)]
