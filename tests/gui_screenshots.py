"""Screenshots of the GUI for the documentation, taken offscreen on a fake experiment.

The screenshots of the documentation are of a GUI that no longer exists. This opens the real application without a
display, on an experiment with one channel of each data type that has a tab (see ``SCREENSHOT_CHANNELS``), and saves a
picture of every tab and of every page of the tabs, toolboxes and channel pages it contains. Run it again at each
release, so that the documentation shows the GUI of the release.

Usage, from the root of the repository, in the ClearMap environment::

    python -m tests.gui_screenshots <out_dir> [--work-dir <dir>] [--whole-window] [--size WIDTHxHEIGHT]
    [--panel-widening PERCENT]

The pictures are written to ``<out_dir>/<tab>/<tab>__<page>__<subpage>.png`` and listed, with the titles they were
reached by, in ``<out_dir>/manifest.tsv``.

* The application runs with a home folder of its own (``<work_dir>/home``): it starts like a new installation, so the
  pictures show the defaults of a new user, and neither ``~/.clearmap`` nor the machine config of whoever runs this are
  read or written. ``--use-real-home`` runs it with the real one.
* ``--work-dir`` (default: a temporary folder) is where the fake experiment is created. The path of the sample folder is
  visible in the pictures, so pass a folder that looks like the one of a user, e.g. ``/data/clearmap``.
* Anything that would open a modal dialog fails the run instead of waiting for someone to answer it.

Not covered (yet): the group mode (Group Analysis and Batch Processing tabs), the dialogs and the data viewers.
"""
from __future__ import annotations

import argparse
import contextlib
import faulthandler
import os
import re
import sys
import tempfile
import time
import traceback
import unicodedata
from pathlib import Path

# The channels of the fake experiment, one per data type that has a tab. Two nuclei channels, because the
# colocalization tab only exists for two channels to detect.
SCREENSHOT_CHANNELS = {
    'cfos': 'nuclei',
    'arc': 'nuclei',
    'autofluorescence': 'autofluorescence',
    'lectin': 'vessels',
    'mbp': 'myelin',
}
SCREENSHOT_SAMPLE_ID = 'sample_01'
SCREENSHOT_SAMPLE_FOLDER_NAME = 'sample_01'

FILE_NAME_SEPARATOR = '__'
UNTITLED = 'untitled'
MANIFEST_NAME = 'manifest.tsv'
MANIFEST_TITLE_SEPARATOR = ' > '

HANG_REPORT_SECONDS = 180  # A run that takes longer prints the stack of every thread, and again at each interval
# The yes/no questions the application asks on its own while it loads the experiment, which are answered with "No":
# the data is not imported (the data of the fake experiment is not meant to be processed)
DECLINED_QUESTION_TITLES = ('Channel data import',)

LOAD_SETTLE_SECONDS = 3.0  # The experiment is loaded through events, and the tabs are built when it is
PAGE_SETTLE_SECONDS = 0.3  # The pages are built and painted when they are shown
EVENT_LOOP_STEP_SECONDS = 0.01

DEFAULT_WINDOW_SIZE = '1000x1300'  # Tall: the open page of a toolbox only gets the height the headers leave
DEFAULT_PANEL_WIDENING_PERCENT = 30.0  # The panel of the application is a little too narrow for its own controls
WINDOW_SIZE_PATTERN = re.compile(r'(\d+)x(\d+)')


def report(message: str) -> None:
    """Say where the run is, at once: in a pipe the output is buffered, and a crash of Qt would lose it"""
    print(message, flush=True)


# ################################################ NAMES (no Qt) ###################################################

def slugify(title: str) -> str:
    """A title of a tab or of a page (e.g. ``'&Atlas settings'``) as a part of a file name (``'atlas_settings'``)"""
    ascii_title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    slug = re.sub(r'[^a-z0-9]+', '_', ascii_title.lower()).strip('_')
    return slug or UNTITLED


class ScreenshotNamer:
    """Gives each screenshot its path in the output folder, from the titles it was reached by (tab, page, ...)"""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self._used: set[Path] = set()

    def next_path(self, titles) -> Path:
        """
        The path of the screenshot reached through ``titles`` (the first one is the tab, which names the folder).
        Two pages with the same title (or with none) get different paths: the second one ends with ``_2``, and so on.
        """
        if not titles:
            raise ValueError('A screenshot needs at least the title of its tab')
        slugs = [slugify(title) for title in titles]
        stem = FILE_NAME_SEPARATOR.join(slugs)
        path = self.out_dir / slugs[0] / f'{stem}.png'
        suffix_number = 2
        while path in self._used:
            path = self.out_dir / slugs[0] / f'{stem}_{suffix_number}.png'
            suffix_number += 1
        self._used.add(path)
        return path


def parse_window_size(text: str) -> tuple[int, int]:
    """``'1000x900'`` as ``(1000, 900)``"""
    match = WINDOW_SIZE_PATTERN.fullmatch(text.strip())
    if match is None:
        raise argparse.ArgumentTypeError(f'expected WIDTHxHEIGHT (e.g. {DEFAULT_WINDOW_SIZE}), got "{text}"')
    return int(match.group(1)), int(match.group(2))


def parse_widening_percent(text: str) -> float:
    """The argument of ``--panel-widening``: a number of percent that is not negative"""
    try:
        percent = float(text)
    except ValueError:
        percent = -1.0
    if percent < 0:
        raise argparse.ArgumentTypeError(f'expected a number of percent that is 0 or more, got "{text}"')
    return percent


def widened_dock_width(*, dock_width: int, panel_width: int, widening_percent: float) -> int:
    """The width the dock must be given for the panel next to it to be wider by ``widening_percent``"""
    return max(dock_width - round(panel_width * widening_percent / 100), 0)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m tests.gui_screenshots', description=__doc__.split('\n\n')[0])
    parser.add_argument('out_dir', type=Path, help='the folder where the screenshots are written (created if missing)')
    parser.add_argument('--work-dir', type=Path, default=None,
                        help='where the fake experiment (and the home folder of the application) are created; '
                             'default: a temporary folder, removed at the end')
    parser.add_argument('--whole-window', action='store_true',
                        help='picture the main window instead of its tab area')
    parser.add_argument('--size', type=parse_window_size, default=parse_window_size(DEFAULT_WINDOW_SIZE),
                        help=f'size of the main window in pixels (default {DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--panel-widening', type=parse_widening_percent, default=DEFAULT_PANEL_WIDENING_PERCENT,
                        metavar='PERCENT',
                        help='widen the control panel (the tabs) by this much, as if dragging its right border, '
                             f'before the pictures (default {DEFAULT_PANEL_WIDENING_PERCENT:g}, 0 to keep the width)')
    parser.add_argument('--use-real-home', action='store_true',
                        help='run with the real home folder (and so the real ~/.clearmap) instead of a new one')
    return parser


# ################################################ ENVIRONMENT ######################################################

def configure_environment(work_dir: Path, *, use_real_home: bool) -> None:
    """
    Set what the application reads at import (before ClearMap and Qt are imported): no display, no prompt, and
    a home folder of its own unless asked otherwise.
    """
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('CLEARMAP_HEADLESS', '1')  # No prompt for the temporary folder
    os.environ.setdefault('CLEARMAP_TMP', str(work_dir / 'tmp'))  # Used when the machine config has no temporary folder
    if not use_real_home:
        home = work_dir / 'home'
        home.mkdir(parents=True, exist_ok=True)
        os.environ['HOME'] = str(home)  # ConfigHandler, Settings and Qt find their folders from it at import


def create_experiment(work_dir: Path) -> Path:
    """Create the fake experiment in ``work_dir``, return its sample folder"""
    from tests.fake_experiment import make_fake_experiment

    make_fake_experiment(work_dir, tuple(SCREENSHOT_CHANNELS), data_types=SCREENSHOT_CHANNELS,
                         sample_id=SCREENSHOT_SAMPLE_ID, sample_folder_name=SCREENSHOT_SAMPLE_FOLDER_NAME)
    return work_dir / SCREENSHOT_SAMPLE_FOLDER_NAME


# ################################################ QT (imported lazily) #############################################

def forbid_modal_dialogs() -> None:
    """
    Make every way of opening a modal dialog raise, so that a run that would wait for an answer fails and says where,
    except the questions of ``DECLINED_QUESTION_TITLES``, which are answered "No", and the notices (a warning or an
    information box with an OK button, e.g. "No cells file found" on an experiment that has not been run), which are
    dismissed with OK and logged: a notice is what a new experiment legitimately shows.
    """
    from PyQt5.QtWidgets import QDialog, QFileDialog, QInputDialog, QMessageBox

    def refuse(description):
        def refuse_modal_dialog(*_args, **_kwargs):
            raise RuntimeError(f'{description} would open a modal dialog, which nobody can answer in this run')
        return refuse_modal_dialog

    refuse_question = refuse('QMessageBox.question')

    def answer_no_to_known_questions(parent, title, *args, **kwargs):
        if title not in DECLINED_QUESTION_TITLES:
            return refuse_question(parent, title, *args, **kwargs)
        report(f'Declined the question "{title}"')
        return QMessageBox.No

    refuse_message_box_exec = refuse('QMessageBox.exec')

    def dismiss_notices(message_box, *args, **kwargs):
        is_notice = (message_box.icon() in (QMessageBox.Warning, QMessageBox.Information)
                     and bool(message_box.standardButtons() & QMessageBox.Ok))
        if not is_notice:
            return refuse_message_box_exec(message_box, *args, **kwargs)
        notice = f'{message_box.text()} {message_box.informativeText()}'.strip()
        report(f'Dismissed the notice "{notice}"')
        return QMessageBox.Ok

    for qt_class, static_methods, instance_methods in (
            (QDialog, (), ('exec', 'exec_')),
            (QMessageBox, ('warning', 'information', 'critical'), ('exec', 'exec_')),
            (QInputDialog, ('getText', 'getInt', 'getDouble', 'getItem'), ()),
            (QFileDialog, ('getExistingDirectory', 'getOpenFileName', 'getSaveFileName'), ())):
        for method_name in static_methods:
            setattr(qt_class, method_name, staticmethod(refuse(f'{qt_class.__name__}.{method_name}')))
        for method_name in instance_methods:
            setattr(qt_class, method_name, refuse(f'{qt_class.__name__}.{method_name}'))
    QMessageBox.question = staticmethod(answer_no_to_known_questions)
    QMessageBox.exec = QMessageBox.exec_ = dismiss_notices


def close_other_dialogs(qapp, window) -> None:
    """Close what the first boot left open (the preferences dialog of a new installation): not part of any picture"""
    from PyQt5.QtWidgets import QDialog

    for widget in qapp.topLevelWidgets():
        if widget is not window and isinstance(widget, QDialog) and widget.isVisible():
            widget.close()


def make_excepthook(uncaught_exceptions: list):
    """
    The hook for the exceptions nothing catches: the ones of the slots, which PyQt would otherwise abort the process
    for (the run goes on, and fails at the end), and the one that ends the run. Each is printed at once, as Python does,
    and kept to count the failures.
    """
    def print_and_keep_uncaught_exception(exception_type, exception, exception_traceback):
        description = ''.join(traceback.format_exception(exception_type, exception, exception_traceback))
        uncaught_exceptions.append(description)
        print(description, file=sys.stderr, flush=True)

    return print_and_keep_uncaught_exception


def prepare_output_dir(out_dir: Path) -> Path:
    """
    Create the folder of the pictures and check that a file can be written in it, before the slow part of the run
    (the application takes minutes to start the first time, and a folder that cannot be written discards all of it).
    """
    from ClearMap.Utils.exceptions import ClearMapIoException, ClearMapPermissionError

    advice = 'Pass an out_dir that exists, or that this user can create.'
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=out_dir):
            pass
    except PermissionError as error:
        raise ClearMapPermissionError(f'Cannot write in the output folder "{out_dir}": {error}. {advice}') from error
    except OSError as error:
        raise ClearMapIoException(f'Cannot use "{out_dir}" as the output folder: {error}. {advice}') from error
    return out_dir


def settle(qapp, seconds: float) -> None:
    """Run the event loop for a while: pages are built and painted, and events are delivered, by the event loop"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(EVENT_LOOP_STEP_SECONDS)


def _is_page_container(widget) -> bool:
    from PyQt5.QtWidgets import QTabWidget, QToolBox
    return isinstance(widget, (QTabWidget, QToolBox))


def containers_below(widget) -> list:
    """The tab widgets and toolboxes in ``widget`` that are not in another one (the ones of the next level)"""
    from PyQt5.QtWidgets import QTabWidget, QToolBox

    top_level_containers = []
    for container in widget.findChildren((QTabWidget, QToolBox)):
        ancestor = container.parentWidget()
        while ancestor is not None and ancestor is not widget and not _is_page_container(ancestor):
            ancestor = ancestor.parentWidget()
        if ancestor is widget or ancestor is None:
            top_level_containers.append(container)
    return top_level_containers


def page_title(container, index: int) -> str:
    from PyQt5.QtWidgets import QToolBox
    return container.itemText(index) if isinstance(container, QToolBox) else container.tabText(index)


class ScreenshotSession:
    """Saves the pictures of the pages of a window, and remembers what it saved (and what failed)"""

    def __init__(self, qapp, picture_source, namer: ScreenshotNamer):
        self.qapp = qapp
        self.picture_source = picture_source  # The widget that is pictured each time (the tab area, or the window)
        self.namer = namer
        self.saved: list[tuple[Path, list[str]]] = []
        self.errors: list[str] = []

    def record_error(self, titles: list[str]) -> None:
        """Keep the exception being handled for the count and print it at once (the run is long: nobody waits for it)"""
        description = f'{MANIFEST_TITLE_SEPARATOR.join(titles)}:\n{traceback.format_exc()}'
        self.errors.append(description)
        print(description, file=sys.stderr, flush=True)

    def save(self, titles: list[str], widget=None) -> None:
        path = self.namer.next_path(titles)
        path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = (widget if widget is not None else self.picture_source).grab()
        if not pixmap.save(str(path), 'PNG'):
            raise OSError(f'Could not write the screenshot "{path}"')
        self.saved.append((path, list(titles)))
        report(f'saved {path}')

    def save_pages_of(self, parent_widget, parent_titles: list[str]) -> None:
        """
        Show each page of each container of ``parent_widget``, picture it, and do the same in the page.
        The pages of a container are shown one at a time with the other containers as they are, and the container
        is left on the page it was on.
        """
        for container in containers_below(parent_widget):
            initial_index = container.currentIndex()
            for index in range(container.count()):
                titles = parent_titles + [page_title(container, index)]
                try:
                    container.setCurrentIndex(index)
                    settle(self.qapp, PAGE_SETTLE_SECONDS)
                    self.save(titles)
                    self.save_pages_of(container.widget(index), titles)
                except Exception:  # One page failing must not hide the others
                    self.record_error(titles)
            container.setCurrentIndex(initial_index)

    def write_manifest(self) -> Path:
        manifest_path = self.namer.out_dir / MANIFEST_NAME
        lines = [f'{path.relative_to(self.namer.out_dir).as_posix()}\t{MANIFEST_TITLE_SEPARATOR.join(titles)}'
                 for path, titles in self.saved]
        manifest_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return manifest_path


def widen_control_panel(qapp, window, *, widening_percent: float) -> None:
    """
    Do what dragging the right border of the control panel does: the data viewer dock on its right gets narrower and
    the panel wider. The window keeps its size. Reports the widths, because a dock cannot go below its minimum width.
    """
    from PyQt5.QtCore import Qt

    panel_width_before = window.tabWidget.width()
    dock_width = widened_dock_width(dock_width=window.graphDock.width(), panel_width=panel_width_before,
                                    widening_percent=widening_percent)
    window.resizeDocks([window.graphDock], [dock_width], Qt.Horizontal)
    settle(qapp, PAGE_SETTLE_SECONDS)
    report(f'Control panel widened from {panel_width_before} to {window.tabWidget.width()} pixels '
           f'(asked: {widening_percent:g}%)')


def open_application(sample_dir: Path, window_size: tuple[int, int], *, panel_widening_percent: float):
    """Start the application like ``ClearMap.gui.app.main`` does (without its event loop), open the experiment"""
    import qdarkstyle

    import ClearMap.gui.app as app_module  # WARNING: runs the first boot and shows the splash screen at import
    from ClearMap.Utils.event_bus import EventBus, GUI_THREAD
    from ClearMap.gui.event_delivery import GuiThreadDelivery
    from ClearMap.gui.tab_registry import TabRegistry

    qapp = app_module.app
    forbid_modal_dialogs()  # After the first boot, which may open the preferences dialog (not modal)

    bus = EventBus()
    bus.set_delivery(GUI_THREAD, GuiThreadDelivery())
    experiment, group_controller = app_module.build_business_objects(bus)
    gui = app_module.GuiController(bus, experiment, TabRegistry(), group_controller)

    qapp.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    gui.start(qapp, centered=False)
    window = gui.window
    app_module.splash.finish(window)
    close_other_dialogs(qapp, window)
    window.resize(*window_size)

    report(f'Opening the experiment {sample_dir}')
    window._select_experiment_mode(sample_dir)  # Like the "Experiment" button, then picking the folder
    settle(qapp, LOAD_SETTLE_SECONDS)
    report('Experiment open')
    widen_control_panel(qapp, window, widening_percent=panel_widening_percent)  # After the load: the tabs are built
    return qapp, window


def take_screenshots(qapp, window, namer: ScreenshotNamer, *, whole_window: bool) -> ScreenshotSession:
    """Picture each tab, the pages in each tab and the workspace manager"""
    session = ScreenshotSession(qapp, window if whole_window else window.tabWidget, namer)
    tab_widget = window.tabWidget
    report('Tabs: ' + ', '.join(tab_widget.tabText(index) for index in range(tab_widget.count())))
    for index in range(tab_widget.count()):
        tab_title = tab_widget.tabText(index)
        report(f'Picturing the tab "{tab_title}"')
        try:
            tab_widget.setCurrentIndex(index)
            settle(qapp, PAGE_SETTLE_SECONDS)
            session.save([tab_title])
            session.save_pages_of(tab_widget.widget(index), [tab_title])
        except Exception:
            session.record_error([tab_title])
    tab_widget.setCurrentIndex(0)

    workspace_manager = getattr(window, 'assetsManagerDock', None)
    if workspace_manager is not None and workspace_manager.isVisible():
        try:
            session.save(['workspace manager'], widget=workspace_manager)
        except Exception:
            session.record_error(['workspace manager'])
    return session


# ################################################ MAIN #############################################################

def main(arguments=None) -> int:
    parsed_arguments = build_argument_parser().parse_args(arguments)
    uncaught_exceptions: list[str] = []
    sys.excepthook = make_excepthook(uncaught_exceptions)
    faulthandler.enable()  # The stack of Python at a crash of Qt (a segmentation fault shows nothing otherwise)
    faulthandler.dump_traceback_later(HANG_REPORT_SECONDS, repeat=True)  # ... and where a run that waits is waiting

    with contextlib.ExitStack() as stack:
        if parsed_arguments.work_dir is None:
            work_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='clearmap_screenshots_')))
        else:
            work_dir = parsed_arguments.work_dir.expanduser().resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
        configure_environment(work_dir, use_real_home=parsed_arguments.use_real_home)

        from ClearMap.Utils.exceptions import ClearMapIoException
        from ClearMap.Utils.utilities import ExitCode

        try:
            out_dir = prepare_output_dir(parsed_arguments.out_dir.expanduser().resolve())
        except ClearMapIoException as error:
            print(error, file=sys.stderr)
            return ExitCode.FAILURE

        report(f'Creating the fake experiment in {work_dir}')
        sample_dir = create_experiment(work_dir)
        report('Starting the application (the first start of a new home folder prepares the atlases: this is slow)')
        qapp, window = open_application(sample_dir, parsed_arguments.size,
                                         panel_widening_percent=parsed_arguments.panel_widening)
        namer = ScreenshotNamer(out_dir)
        session = take_screenshots(qapp, window, namer, whole_window=parsed_arguments.whole_window)
        if session.saved:
            manifest_path = session.write_manifest()
            print(f'{len(session.saved)} screenshots in {namer.out_dir} (listed in {manifest_path.name})')

    faulthandler.cancel_dump_traceback_later()
    failures = session.errors + uncaught_exceptions  # All were printed when they happened
    if failures or not session.saved:
        print(f'{len(failures)} failure(s), {len(session.saved)} screenshot(s) saved', file=sys.stderr)
        return ExitCode.FAILURE
    return ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
