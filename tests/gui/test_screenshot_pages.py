"""
The pages the screenshot tool pictures (tests/gui_screenshots.py), with minimal stand-ins for the Qt containers:
this checks the walk (what is shown, in what order, what is restored, what failing does), not what Qt paints.
"""
import sys
import types

import pytest

import tests.gui_screenshots as screenshots


class FakeWidget:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

    def parentWidget(self):
        return self.parent

    def findChildren(self, classes):
        found = []
        for child in self.children:
            if isinstance(child, classes):
                found.append(child)
            found.extend(child.findChildren(classes))
        return found

    def grab(self):
        return FakePixmap()


class FakePixmap:
    def save(self, path, image_format):
        assert image_format == 'PNG'
        with open(path, 'wb') as png:
            png.write(b'png')
        return True


class FakeContainer(FakeWidget):
    """Pages are (title, widget): the widget is below a stack, like the pages of a QTabWidget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stack = FakeWidget(self)
        self.titles = []
        self.pages = []
        self.index = 0
        self.shown_titles = []  # What setCurrentIndex was asked to show, in order

    def add_page(self, title):
        page = FakeWidget(self.stack)
        self.titles.append(title)
        self.pages.append(page)
        return page

    def count(self):
        return len(self.pages)

    def widget(self, index):
        return self.pages[index]

    def currentIndex(self):
        return self.index

    def setCurrentIndex(self, index):
        self.index = index
        self.shown_titles.append(self.titles[index])


class FakeTabWidget(FakeContainer):
    def tabText(self, index):
        return self.titles[index]


class FakeToolBox(FakeContainer):
    def itemText(self, index):
        return self.titles[index]


class FakeApplication:
    def processEvents(self):
        pass


class FakeMessageBox:
    No = 'No'
    Yes = 'Yes'
    Ok = 1
    Cancel = 2
    Warning = 'Warning'
    Information = 'Information'
    Question = 'Question'
    Critical = 'Critical'

    def __init__(self, icon, buttons, text='', informative_text=''):
        self._icon, self._buttons, self._text, self._informative_text = icon, buttons, text, informative_text

    def icon(self):
        return self._icon

    def standardButtons(self):
        return self._buttons

    def text(self):
        return self._text

    def informativeText(self):
        return self._informative_text


@pytest.fixture(autouse=True)
def fake_qt(monkeypatch):
    qt_widgets = types.ModuleType('PyQt5.QtWidgets')
    qt_widgets.QTabWidget = FakeTabWidget
    qt_widgets.QToolBox = FakeToolBox
    for name in ('QDialog', 'QFileDialog', 'QInputDialog'):  # Fresh classes: the tool replaces their methods
        setattr(qt_widgets, name, type(name, (), {}))
    qt_widgets.QMessageBox = type('QMessageBox', (FakeMessageBox,), {})
    monkeypatch.setitem(sys.modules, 'PyQt5', types.ModuleType('PyQt5'))
    monkeypatch.setitem(sys.modules, 'PyQt5.QtWidgets', qt_widgets)
    monkeypatch.setattr(screenshots, 'PAGE_SETTLE_SECONDS', 0)
    return qt_widgets


def make_session(tmp_path, picture_source=None):
    return screenshots.ScreenshotSession(FakeApplication(), picture_source or FakeWidget(),
                                         screenshots.ScreenshotNamer(tmp_path))


def make_stitching_tab():
    """A tab with a channels tab widget, each channel page having a tab widget of its own (Rigid, Wobbly)"""
    tab = FakeWidget()
    channels = FakeTabWidget(tab)
    for channel in ('cfos', 'arc'):
        channel_page = channels.add_page(channel)
        steps = FakeTabWidget(channel_page)
        steps.add_page('Rigid')
        steps.add_page('Wobbly')
    return tab, channels


def test_every_page_is_pictured_and_the_pages_in_a_page_follow_it(tmp_path):
    tab, _ = make_stitching_tab()
    session = make_session(tmp_path)

    session.save_pages_of(tab, ['Stitching'])

    assert [titles for _, titles in session.saved] == [
        ['Stitching', 'cfos'], ['Stitching', 'cfos', 'Rigid'], ['Stitching', 'cfos', 'Wobbly'],
        ['Stitching', 'arc'], ['Stitching', 'arc', 'Rigid'], ['Stitching', 'arc', 'Wobbly'],
    ]
    assert not session.errors
    assert all(path.exists() for path, _ in session.saved)


def test_the_pages_of_a_page_are_not_pictured_at_the_level_of_the_tab(tmp_path):
    """Only the containers of the next level are walked by the tab: the others are reached through their page"""
    tab, channels = make_stitching_tab()

    containers = screenshots.containers_below(tab)

    assert containers == [channels]


def test_a_container_is_left_on_the_page_it_was_on(tmp_path):
    tab, channels = make_stitching_tab()
    channels.index = 1

    make_session(tmp_path).save_pages_of(tab, ['Stitching'])

    assert channels.currentIndex() == 1
    assert channels.pages[0].children[0].currentIndex() == 0  # The nested ones too


def test_the_pages_of_a_toolbox_and_of_a_tab_widget_in_the_same_tab_are_both_walked(tmp_path):
    tab = FakeWidget()
    toolbox = FakeToolBox(tab)
    toolbox.add_page('Source')
    toolbox.add_page('Atlas space info')
    channels = FakeTabWidget(tab)
    channels.add_page('cfos')

    session = make_session(tmp_path)
    session.save_pages_of(tab, ['Sample Info'])

    assert sorted(titles[-1] for _, titles in session.saved) == ['Atlas space info', 'Source', 'cfos']


def test_a_page_that_fails_is_reported_and_the_other_pages_are_still_pictured(tmp_path):
    tab, channels = make_stitching_tab()
    session = make_session(tmp_path)
    save = session.save

    def save_unless_arc(titles, widget=None):
        if titles == ['Stitching', 'arc']:
            raise RuntimeError('this page cannot be pictured')
        save(titles, widget)

    session.save = save_unless_arc
    session.save_pages_of(tab, ['Stitching'])

    assert len(session.errors) == 1
    assert 'Stitching > arc' in session.errors[0] and 'this page cannot be pictured' in session.errors[0]
    assert ['Stitching', 'cfos', 'Wobbly'] in [titles for _, titles in session.saved]
    assert channels.currentIndex() == 0  # Restored although a page failed


def test_a_page_that_fails_is_printed_when_it_fails_not_only_at_the_end(tmp_path, capsys):
    tab, _ = make_stitching_tab()
    session = make_session(tmp_path)

    def refuse_every_page(titles, widget=None):
        raise PermissionError('/data/tmp')

    session.save = refuse_every_page
    session.save_pages_of(tab, ['Stitching'])

    printed = capsys.readouterr().err
    assert len(session.errors) > 0
    assert printed.count("PermissionError: /data/tmp") == len(session.errors)


def test_the_manifest_lists_each_file_with_the_titles_it_was_reached_by(tmp_path):
    tab, _ = make_stitching_tab()
    session = make_session(tmp_path)
    session.save_pages_of(tab, ['Stitching'])

    lines = session.write_manifest().read_text(encoding='utf-8').splitlines()

    assert lines[0] == 'stitching/stitching__cfos.png\tStitching > cfos'
    assert lines[1] == 'stitching/stitching__cfos__rigid.png\tStitching > cfos > Rigid'
    assert len(lines) == 6


def test_the_question_asked_while_the_experiment_loads_is_answered_no(fake_qt, capsys):
    screenshots.forbid_modal_dialogs()

    answer = fake_qt.QMessageBox.question(None, 'Channel data import', 'Import 5 channel(s) into workspace?')

    assert answer == fake_qt.QMessageBox.No
    assert 'Channel data import' in capsys.readouterr().out


def test_a_warning_with_an_ok_button_is_dismissed_with_ok_and_logged(fake_qt, capsys):
    screenshots.forbid_modal_dialogs()
    warning = fake_qt.QMessageBox(fake_qt.QMessageBox.Warning, fake_qt.QMessageBox.Ok | fake_qt.QMessageBox.Cancel,
                                  '<b>Missing configuration file</b>', 'No cells file found, cannot display histograms')

    answer = warning.exec()

    assert answer == fake_qt.QMessageBox.Ok
    assert 'No cells file found' in capsys.readouterr().out


@pytest.mark.parametrize('icon, buttons', [
    ('Critical', 1),  # An error is not a notice
    ('Question', 1 | 2),  # A question needs an answer that is not always OK
    ('Warning', 2),  # No OK to answer
])
def test_a_message_box_that_is_not_a_notice_fails_the_run(fake_qt, icon, buttons):
    screenshots.forbid_modal_dialogs()

    with pytest.raises(RuntimeError, match='modal dialog'):
        fake_qt.QMessageBox(icon, buttons, 'Text').exec()


@pytest.mark.parametrize('open_dialog', [
    lambda qt: qt.QMessageBox.question(None, 'Convert tiles', 'Convert?'),
    lambda qt: qt.QMessageBox.warning(None, 'Title', 'Text'),
    lambda qt: qt.QInputDialog.getText(None, 'Enter the sample ID', 'ID'),
    lambda qt: qt.QFileDialog.getExistingDirectory(None, 'Choose'),
    lambda qt: qt.QDialog().exec_(),
])
def test_any_other_modal_dialog_fails_the_run_instead_of_waiting_for_an_answer(fake_qt, open_dialog):
    screenshots.forbid_modal_dialogs()

    with pytest.raises(RuntimeError, match='modal dialog'):
        open_dialog(fake_qt)


class FakeMainWindow:
    """A window whose dock gives its width to the panel, as the docks of Qt do"""

    def __init__(self, total_width, panel_width):
        self.total_width = total_width
        self.tabWidget = types.SimpleNamespace(width=lambda: panel_width)
        self.dock_width = total_width - panel_width
        self.graphDock = types.SimpleNamespace(width=lambda: self.dock_width)
        self.resized_docks = []

    def resizeDocks(self, docks, widths, orientation):
        self.resized_docks.append((docks, widths, orientation))
        self.dock_width = widths[0]
        self.tabWidget = types.SimpleNamespace(width=lambda: self.total_width - self.dock_width)


def test_the_control_panel_is_widened_by_shrinking_the_dock_on_its_right(monkeypatch, capsys):
    qt_core = types.ModuleType('PyQt5.QtCore')
    qt_core.Qt = types.SimpleNamespace(Horizontal='horizontal')
    monkeypatch.setitem(sys.modules, 'PyQt5.QtCore', qt_core)
    window = FakeMainWindow(total_width=1000, panel_width=200)

    screenshots.widen_control_panel(FakeApplication(), window, widening_percent=30)

    assert window.resized_docks == [([window.graphDock], [740], 'horizontal')]
    assert window.tabWidget.width() == 260
    assert 'from 200 to 260 pixels' in capsys.readouterr().out
