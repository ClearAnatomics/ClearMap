# -*- coding: utf-8 -*-
"""
gui_logging
===========

Defines the Printer class used to log to file and to the GUI widgets from simple prints
"""

from datetime import datetime
from io import UnsupportedOperation

import lxml.html
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget
from matplotlib.colors import cnames


__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'
__copyright__ = 'Copyright © 2022 by Charly Rousseau'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


class Printer(QWidget):
    text_updated = QtCore.pyqtSignal(str)

    def __init__(self, log_path=None, color=None, logger_type='info', app=None, open_mode='a', parent=None):
        super().__init__(parent)
        # self.widget = text_widget
        self.file = None
        if log_path is not None:
            self.file = open(log_path, open_mode)
        self.color = color
        self.type = logger_type
        # self.win = self.widget.window()
        if app is None:
            self.app = QApplication.instance()
        self.n_lines = 0

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def set_file(self, log_path, open_mode='a'):
        if self.file is not None:
            self.file.close()
        self.file = open(log_path, open_mode)
        self.n_lines = 0

    def write(self, msg):
        if self.file is not None:
            if self.type in ('error', 'progress'):
                self.file.write(f'{datetime.now().isoformat()}: ')
            self.file.write(msg+'\n')
            self.file.flush()
        self.text_updated.emit(self.colourise(msg))

    def flush(self):
        if self.file is not None:
            self.file.flush()

    def fileno(self):
        if self.file is not None:
            return self.file.fileno()
        else:
            raise UnsupportedOperation('Cannot access fileno without a file object')

    def colourise(self, msg, force=False):
        """
        Convert msg to the self.color colour in html

        Parameters
        ----------
        msg str:
            The message to colourise
        force bool:
            By default, do not colorise html code (to avoid messing up existing colours). This
            forces the colorise nonetheless.

        Returns
        -------

        """
        if self.color is not None:
            try:
                html = lxml.html.fromstring(msg)
                is_html = html.find('.//*') is not None
            except lxml.etree.ParserError:
                is_html = False
            if force or not is_html:
                colour_msg = '<p style="color:{}">{}</p>'.format(cnames[self.color], msg)
            else:
                colour_msg = msg
        else:
            colour_msg = msg
        return colour_msg
