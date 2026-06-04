# -*- coding: utf-8 -*-
"""
Timer
=====

Module provides tools for timing information.

Example
-------

>>> import ClearMap.Utils.Timer as timer
>>> t = timer.Timer()
>>> for i in range(100000000):
>>>     x = 10 + i
>>> t.print_elapsed_time('test')

"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'
import time
from contextlib import contextmanager

import ClearMap.Utils.Sound as snd


class Timer:
    """
    Class to stop time and print results in formatted way

    Attributes
    ----------
    time: float
        The time since the timer was started.
    head: str or None
        Option prefix to the timing string.
    """

    def __init__(self, head = None):
        self.time = None
        self.head = head
        self.start()

    def start(self):
        """Start the timer"""
        self.time = time.time()

    def reset(self):
        """Reset the timer"""
        self.time = time.time()

    def elapsed_time(self, head=None, as_string=True):
        """
        Calculate elapsed time and return as formated string

        Arguments
        ---------
        head: str or None
            Prefix to the timing string.
        as_string: bool
            If True, return as string, else return elapsed time as float.

        Returns
        -------
        time: str or float
            The elapsed time information.
        """

        t = time.time()

        delta_t = t - self.time
        if as_string:  # TODO: add current datetime to string
            delta_t = self.format_time(delta_t)
            if head is None:
                head = self.head
            elif self.head is not None:
                head = f'{self.head}{head}'

            if head is not None:
                return f'{head}: elapsed time: {delta_t}'
            else:
                return f'Elapsed time: {delta_t}'
        else:
            return delta_t

    def print_elapsed_time(self, head=None, beep=False, reset=False):
        """
        Print elapsed time.

        Arguments
        ---------
        head: str or None
            Prefix to the timing string.
        beep: bool
            If True, beep in addition to print the time.
        reset: bool
            If True, reset the timer after printing.
        """
        print(self.elapsed_time(head=head), flush=True)
        if beep:
            snd.beep()
        if reset:
            self.reset()

    def format_time(self, t):
        """Format time to string.

        Arguments
        ---------
        t: float
            Time in seconds to format.

        Returns
        -------
        time: str
            The time as 'hours:minutes:seconds:milliseconds'.
        """
        m, s = divmod(t, 60)
        h, m = divmod(m, 60)

        hours = int(h)
        minutes = int(m)
        seconds = int(s)
        millis = int((s - seconds) * 1000)

        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    def __str__(self):
        return self.elapsed_time()

    def __repr__(self):
        return self.__str__()


def _format_elapsed(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    ms     = (s % 1) * 1000
    return f'{int(h):d}:{int(m):02d}:{int(s):02d}.{int(ms):03d}'


def timeit(method):
    """Decorator — times a method call."""
    def timed_(*args, **kw):
        ts     = time.time()
        result = method(*args, **kw)
        print(f'{method.__name__}: {_format_elapsed(time.time() - ts)}')
        return result
    return timed_


@contextmanager
def timed(label: str = '', logger=None):
    """Context manager — times a code block."""
    ts = time.time()
    try:
        yield
    finally:
        msg = f'{label}: {_format_elapsed(time.time() - ts)}'
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)


def _test():
    import ClearMap.Utils.Timer as timer
    t = timer.Timer(head = 'Testing')
    for i in range(10000):
        x = 10 + i
    t.print_elapsed_time('test')

    print(t)
