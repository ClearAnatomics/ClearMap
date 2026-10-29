"""
P0013: ``OSError: [Errno 9] Bad file descriptor`` at the end of a step of the GUI (cell detection).

Two threads shut down the same executor: the thread of the GUI calls ``stop_process`` (Abort button), which calls
``executor.shutdown(cancel_futures=True)``, while the thread running the step leaves the
``with CancelableProcessPoolExecutor(...)`` of ``BlockProcessing.process`` (the pending futures are cancelled, so
``f.result()`` raises ``CancelledError``) and so calls ``shutdown(wait=True)`` too. ``ProcessPoolExecutor.shutdown``
is not safe against that: both threads close the reader of the result queue, so the second one closes a file descriptor
that is already closed (or, worse, reused).

``CancelableProcessPoolExecutor`` is the one place where all the ClearMap pools are created, so it is where the
shutdown is serialized.
"""
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.connection import Connection

from ClearMap.Utils.utilities import CancelableProcessPoolExecutor

CLOSE_DURATION = 0.3  # s. Widens the window of the race (a close is otherwise too fast to overlap reliably)


def _slow_down_closing_connections(monkeypatch):
    """
    Make a pipe linger between the closing of its file descriptor and the moment it is marked as closed
    (``Connection.close`` sets ``_handle = None`` in a ``finally`` once ``_close`` returns).

    The real close releases the GIL, so another thread can run in that window: it sees the pipe as still open
    and closes the same descriptor again.
    """
    close = Connection._close

    def close_then_linger(self):
        close(self)
        time.sleep(CLOSE_DURATION)

    monkeypatch.setattr(Connection, '_close', close_then_linger)


def _shut_down_from_two_threads(executor, **kwargs_of_the_second_shutdown):
    errors = []

    def shut_down(**kwargs):
        try:
            executor.shutdown(wait=True, **kwargs)
        except BaseException as err:  # noqa: B036 Whatever the thread dies of, the test has to see it
            errors.append(err)

    threads = [threading.Thread(target=shut_down),
               threading.Thread(target=shut_down, kwargs=kwargs_of_the_second_shutdown)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), 'A shutdown did not finish'
    return errors


def _started_executor():
    executor = CancelableProcessPoolExecutor(max_workers=1)
    assert executor.submit(pow, 2, 3).result(timeout=30) == 8  # Starts the worker and the manager thread
    return executor


def test_the_executor_is_a_process_pool_executor():
    with CancelableProcessPoolExecutor(max_workers=1) as executor:
        assert isinstance(executor, ProcessPoolExecutor)
        assert executor.submit(pow, 2, 3).result(timeout=30) == 8


def test_shutdowns_from_two_threads_do_not_close_the_result_queue_twice(monkeypatch):
    executor = _started_executor()
    _slow_down_closing_connections(monkeypatch)

    errors = _shut_down_from_two_threads(executor)

    assert errors == []


def test_cancelling_from_one_thread_while_another_leaves_the_with_does_not_fail(monkeypatch):
    """What ``stop_process`` (cancel_futures=True) and the ``with`` of ``BlockProcessing.process`` do"""
    executor = _started_executor()
    _slow_down_closing_connections(monkeypatch)

    errors = _shut_down_from_two_threads(executor, cancel_futures=True)

    assert errors == []


def test_a_second_shutdown_after_the_first_is_harmless():
    executor = _started_executor()

    executor.shutdown(wait=True)
    executor.shutdown(wait=True, cancel_futures=True)  # Does not raise
