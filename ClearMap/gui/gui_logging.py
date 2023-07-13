# -*- coding: utf-8 -*-
"""
gui_logging
===========

Defines the Printer class used to log to file and to the GUI widgets from simple prints
"""
import sys
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
    original_std_out = sys.stdout
    original_std_err = sys.stderr

    def __init__(self, log_path=None, color=None, logger_type='info', app=None, open_mode='a', redirects=None, parent=None):
        super().__init__(parent)
        self.file = None
        self.n_lines = 0
        self.color = color
        self.type = logger_type
        self.redirects = redirects

        self.set_file(log_path, open_mode)

        if app is None:
            self.app = QApplication.instance()
        else:
            self.app = app

    def __del__(self):
        self.close_file()

    def close_file(self):
        try:
            self.file.close()
        except AttributeError:
            pass
        self.__unset_redirects()

    def set_file(self, log_path, open_mode='a'):
        self.close_file()
        if log_path:
            self.file = open(log_path, open_mode)
            self.n_lines = 0
            self.__set_redirects()

    def __set_redirects(self):
        if self.redirects == 'stdout':
            sys.stdout = self
        elif self.redirects == 'stderr':
            sys.stderr = self

    def __unset_redirects(self):
        if self.redirects == 'stdout':
            sys.stdout = self.original_std_out
        elif self.redirects == 'stderr':
            sys.stderr = self.original_std_err

    def write(self, msg):
        if self.file is not None:
            if self.type in ('error', 'progress'):
                self.file.write(f'{datetime.now().strftime("%y-%m-%d %H:%M:%S")}: ')
            self.file.write(msg+'\n')
            self.file.flush()
        self.text_updated.emit(self.colourise(msg))

    def flush(self):
        try:
            self.file.flush()
        except AttributeError:
            pass

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
                if len(msg.split(':')) > 2 and msg.split(':')[2].endswith('Warning'):
                    html_col = cnames['yellow']
                else:
                    html_col = cnames[self.color]
                colour_msg = f'<p style="color:{html_col}">{msg}</p>'
            else:
                colour_msg = msg
        else:
            colour_msg = msg
        return colour_msg
