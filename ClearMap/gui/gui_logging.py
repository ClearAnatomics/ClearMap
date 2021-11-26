from io import UnsupportedOperation

import lxml.html
from PyQt5.QtWidgets import QApplication
from matplotlib.colors import cnames


class Printer(object):
    def __init__(self, text_widget, log_path=None, color=None, app=None, open_mode='a'):
        super().__init__()
        self.widget = text_widget
        self.file = None
        if log_path is not None:
            self.file = open(log_path, open_mode)
        self.color = color
        self.win = self.widget.window()
        if app is None:
            self.app = QApplication.instance()

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def set_file(self, log_path, open_mode='a'):
        if self.file is not None:
            self.file.close()
        self.file = open(log_path, open_mode)

    def write(self, msg):
        if self.file is not None:
            self.file.write(msg)
        self.widget.append(self._colourise(msg))

    def flush(self):  # To be compatible with ClearMap.ParallelProcessing.ProcessWriting
        pass

    def fileno(self):
        if self.file is not None:
            return self.file.fileno()
        else:
            raise UnsupportedOperation('Cannot access fileno without a file object')

    def _colourise(self, msg):
        if self.color is not None:
            try:
                html = lxml.html.fromstring(msg)
                is_html = html.find('.//*') is not None
            except lxml.etree.ParserError:
                is_html = False
            if not is_html:
                colour_msg = '<p style="color:{}">{}</p>'.format(cnames[self.color], msg)
            else:
                colour_msg = msg
        else:
            colour_msg = msg
        return colour_msg