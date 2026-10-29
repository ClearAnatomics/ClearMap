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


@pytest.fixture(autouse=True)
def fake_qt(monkeypatch):
    qt_widgets = types.ModuleType('PyQt5.QtWidgets')
    qt_widgets.QTabWidget = FakeTabWidget
    qt_widgets.QToolBox = FakeToolBox
    monkeypatch.setitem(sys.modules, 'PyQt5', types.ModuleType('PyQt5'))
    monkeypatch.setitem(sys.modules, 'PyQt5.QtWidgets', qt_widgets)
    monkeypatch.setattr(screenshots, 'PAGE_SETTLE_SECONDS', 0)


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


def test_the_manifest_lists_each_file_with_the_titles_it_was_reached_by(tmp_path):
    tab, _ = make_stitching_tab()
    session = make_session(tmp_path)
    session.save_pages_of(tab, ['Stitching'])

    lines = session.write_manifest().read_text(encoding='utf-8').splitlines()

    assert lines[0] == 'stitching/stitching__cfos.png\tStitching > cfos'
    assert lines[1] == 'stitching/stitching__cfos__rigid.png\tStitching > cfos > Rigid'
    assert len(lines) == 6
