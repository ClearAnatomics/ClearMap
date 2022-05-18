import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import psutil

colors = {
    "WHITE": '\033[1;37m',
    "GREEN": '\033[0;32m',
    "YELLOW": '\033[1;33;48m',
    "RED": '\033[1;31;48m',
    "BLINK": '\33[5m',
    "BLINK2": '\33[6m',
    "RESET": '\033[1;37;0m'
}


def colorize(msg, color):
    color = color.upper()
    color = colors[color]
    return "{color}{msg}{reset_color}".format(color=color, msg=msg, reset_color=colors["RESET"])


def runs_on_spyder():
    return any('SPYDER' in name for name in os.environ)


def runs_from_pycharm():
    return "PYCHARM_HOSTED" in os.environ


def get_free_v_ram():
    cmd = 'nvidia-smi --query-gpu=memory.free --format=noheader,csv,nounits'
    result = subprocess.check_output(cmd, shell=True)
    return int(result)


class CancelableProcessPoolExecutor(ProcessPoolExecutor):
    def immediate_shutdown(self):
        with self._shutdown_lock:
            self._shutdown_thread = True
            # statuses = [psutil.Process(_proc.pid).status() for _proc in self._processes.values()]
            terminated_procs = 0
            for proc in self._processes.values():
                status = psutil.Process(proc.pid).status()
                if status == 'sleeping':
                    proc.terminate()
                    terminated_procs += 1
            if not terminated_procs:
                for proc in self._processes.values():
                    proc.terminate()


def is_in_range(src_array, value_range):
    return np.logical_and(src_array >= value_range[0], src_array <= value_range[1])


def is_iterable(obj):
    try:
        iterator = iter(obj)
        return True
    except TypeError:
        return False


def title_to_snake(string):
    return re.sub('(?!^)([A-Z]+)', r'_\1', string).lower()
