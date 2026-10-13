"""The event bus: subscriptions"""
from dataclasses import dataclass

from ClearMap.Utils.event_bus import EventBus


@dataclass(frozen=True)
class Ping:
    value: int = 0


class Recorder:
    def __init__(self):
        self.events = []

    def on_ping(self, event):
        self.events.append(event)


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
