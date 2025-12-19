import os
import functools

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap, QPainter, QFont, QFontMetrics, QPen
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog, QDialog, QVBoxLayout, QStyle, QLabel, QHBoxLayout, \
    QPushButton, QDialogButtonBox, QSplashScreen, QProgressBar

from ClearMap.Utils.path_utils import clean_path
from ClearMap.config.defaults_provider import get_defaults_provider
from ClearMap.gui.gui_utils_base import create_clearmap_widget, UI_FOLDER

_defaults = get_defaults_provider()
DISPLAY_CONFIG = _defaults.get('display')


def get_directory_dlg(start_folder, title="Choose the source directory"):
    diag = QFileDialog()  # REFACTOR: move to gui_utils
    opt = QFileDialog.Options(QFileDialog.DontUseNativeDialog)
    start_folder = clean_path(start_folder)
    src_folder = diag.getExistingDirectory(parent=diag, caption=title,
                                           directory=start_folder, options=opt)
    diag.close()
    return src_folder


def prompt_dialog(title, msg):
    pressed_btn = QMessageBox.question(None, title, msg)
    return pressed_btn == QMessageBox.Yes


def input_dialog(title, msg, data_type=str):
    dlg = QInputDialog()
    input_modes_map = {
        str: QInputDialog.TextInput,
        int: QInputDialog.IntInput,
        float: QInputDialog.DoubleInput
    }
    dlg.setInputMode(input_modes_map[data_type])
    dlg.setWindowTitle(title)
    dlg.setLabelText(msg)

    if dlg.exec() == QDialog.Accepted:
        output_functions = {
            str: dlg.textValue,
            int: dlg.intValue,
            float: dlg.doubleValue
        }
        return output_functions[data_type]()


def option_dialog(base_msg, msg, options, parent=None):
    index = [None]

    def get_id(lst, id_):
        lst[0] = id_
        return id_

    dlg = QDialog(parent)

    dlg.setWindowTitle('User input required')
    main_layout = QVBoxLayout()
    dlg.setLayout(main_layout)
    pixmapi = QStyle.SP_MessageBoxQuestion
    icon = dlg.style().standardIcon(pixmapi)
    icon_lbl = QLabel()
    icon_lbl.setPixmap(icon.pixmap(icon.actualSize(QSize(32, 32))))
    h1 = QHBoxLayout()
    main_layout.addLayout(h1)
    h1.addWidget(icon_lbl)
    lbl = QLabel(f'<b>{base_msg}</b>')
    h1.addWidget(lbl)
    lbl2 = QLabel(msg)
    main_layout.addWidget(lbl2)

    layout = QHBoxLayout()
    main_layout.addLayout(layout)

    for i, option in enumerate(options):
        btn = QPushButton(option, parent=dlg)
        btn.clicked.connect(functools.partial(get_id, index, i))
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
    dlg.exec()
    return index[0]


def warning_popup(base_msg, msg=''):
    msg = msg or base_msg
    dlg = QMessageBox()
    dlg.setIcon(QMessageBox.Warning)
    dlg.setWindowTitle('Warning')
    dlg.setText(f'<b>{base_msg}</b>')
    dlg.setInformativeText(msg)
    dlg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    dlg.setDefaultButton(QMessageBox.Ok)
    return dlg.exec()


def abort_retry_popup(base_msg, msg):  # REFACTOR: duplicate code
    dlg = QMessageBox()
    dlg.setIcon(QMessageBox.Warning)
    dlg.setWindowTitle('Warning')
    dlg.setText(f'<b>{base_msg}</b>')
    dlg.setInformativeText(msg)
    dlg.setStandardButtons(QMessageBox.Abort | QMessageBox.Retry)
    dlg.setDefaultButton(QMessageBox.Abort)
    return dlg.exec()


def make_nested_progress_dialog(title='Processing', overall_maximum=100, sub_process_name='', abort_callback=None, parent=None):
    dlg = create_clearmap_widget('nested_progress_dialog.ui', patch_parent_class='QDialog', window_title='Progress')

    progress_icon_path = os.path.join(UI_FOLDER, 'creator', 'graphics_resources', 'searching_mouse.png')
    dlg.progressImageLabel.setPixmap(QPixmap(progress_icon_path))  # TODO: why doesn't work with qrc ??
    dlg.setWindowTitle('Clearmap progress')
    dlg.mainLabel.setText(f'{title}, please wait.')

    dlg.mainProgressBar.setRange(1, overall_maximum)
    dlg.subProgressLabel.setText(sub_process_name)
    if abort_callback is not None:
        dlg.buttonBox.button(QDialogButtonBox.Abort).clicked.connect(abort_callback)
    dlg.show()
    return dlg


def make_simple_progress_dialog(title='Processing', overall_maximum=100, sub_process_name='', abort_callback=None, parent=None):
    dlg = make_nested_progress_dialog(title=title, overall_maximum=overall_maximum, sub_process_name=sub_process_name,
                                      abort_callback=abort_callback, parent=parent)
    dlg.mainProgressBar.setVisible(False)

    return dlg


def make_splash(img_source=None, bar_max=100, message=None, res='hd', font_size=60, text_zone=0):
    if img_source is None:
        img_source = os.path.join(UI_FOLDER, 'creator', 'graphics_resources', 'splash.png')
    splash_pix = QPixmap(img_source)  # .scaled(848, 480)

    # Prepare the painter to overlay text and lines onto the image.
    painter = QPainter(splash_pix)
    painter.setRenderHint(QPainter.Antialiasing)

    # Set up the font for the overlay text.
    font = QFont("Arial", font_size, QFont.Bold)  # FIXME: put setting in config
    painter.setFont(font)
    metrics = QFontMetrics(font)

    if message is None:
        from packaging.version import Version
        from importlib_metadata import version
        clearmap_version = Version(version('ClearMap'))
        message = f"ClearMap {clearmap_version.major}.{clearmap_version.minor}"

    # Calculate horizontal centering for the text.
    text_width = metrics.horizontalAdvance(message)
    x_text = (splash_pix.width() - text_width) // 5

    # Define a top margin. The first line will overlap the top of the image.
    top_margin = 10
    pen = QPen(Qt.white, 2)
    painter.setPen(pen)

    # Draw the top horizontal line.
    # Calculate the y coordinate for the top line: it is aligned with 80% of the font’s ascent.
    # line_y_top = top_margin + metrics.ascent() * 0.8
    # painter.drawLine(0, int(line_y_top), splash_pix.width(), int(line_y_top))

    # Draw the text just below the top line.
    # Compute a y coordinate for the text baseline.
    y_text = top_margin + metrics.ascent() + 10
    painter.drawText(x_text, y_text, message)

    painter.end()

    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())

    progress_bar = QProgressBar(splash)
    progress_bar.setTextVisible(True)
    progress_bar.setFont(QFont('Arial', DISPLAY_CONFIG[res]['splash_font_size'], QFont.Bold))
    progress_bar.setFormat("Loading... \t\t%p%")
    progress_bar.setMaximum(bar_max)

    margin = 50
    bar_height = 20

    if text_zone < 0:  # Clip
        text_zone = 0

    # If text_zone > 0, reserve that many pixels at the very bottom of the splash
    # for the semi-transparent "Importing ..." messages; move the bar up accordingly.
    bar_y = splash_pix.height() - margin - text_zone
    # Ensure we don't send the bar off the top if someone sets a silly text_zone.
    bar_y = max(margin, bar_y)

    progress_bar.setGeometry(margin, bar_y, splash_pix.width() - 2 * margin, bar_height)

    return splash, progress_bar


def update_pbar(app, pbar, value):
    pbar.setValue(value)
    app.processEvents()
