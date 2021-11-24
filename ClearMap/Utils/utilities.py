import os

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