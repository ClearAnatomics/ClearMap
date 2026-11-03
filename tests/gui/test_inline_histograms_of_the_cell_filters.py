"""
The histograms of the cell filters are put above the label and controls of the filter they show, found by name:
by position, a change of the order of the widgets in the .ui file put the size histogram over the intensity controls.
"""
from ClearMap.gui.widgets import ensure_inline_histogram


class FakeWidget:
    def __init__(self, name, width=100):
        self.name = name
        self._width = width
        self.maximum_size = None

    def width(self):
        return self._width

    def height(self):
        return 20

    def resize(self, width, height):
        pass

    def setMaximumSize(self, width, height):
        self.maximum_size = (width, height)

    def setMinimumHeight(self, height):
        pass


class FakeContainer(FakeWidget):
    def __init__(self):
        super().__init__('container')

    def parent(self):
        return self


class FakeGrid:
    """A grid layout whose parent chain is as deep as the one of the cell filters page"""
    def __init__(self):
        self.placed = {}
        self._parent = FakeContainer()

    def parent(self):
        return self._parent

    def indexOf(self, widget):
        return list(self.placed).index(widget) if widget in self.placed else -1

    def addWidget(self, widget, row, column, row_span, column_span):
        self.placed[widget] = (row, column, row_span, column_span)


def test_the_histogram_is_put_above_the_label_and_controls_it_is_given():
    size_label, size_controls = FakeWidget('size label'), FakeWidget('size controls')
    grid = FakeGrid()
    histogram = FakeWidget('size histogram')

    ensure_inline_histogram(histogram, label=size_label, controls=size_controls, row=2, layout=grid)

    assert grid.placed == {histogram: (2, 0, 1, 3), size_label: (3, 0, 1, 1), size_controls: (3, 1, 1, 2)}


def test_a_histogram_already_in_the_layout_is_not_put_again():
    label, controls, histogram = FakeWidget('label'), FakeWidget('controls'), FakeWidget('histogram')
    grid = FakeGrid()
    ensure_inline_histogram(histogram, label=label, controls=controls, row=0, layout=grid)
    grid.placed[histogram] = (9, 9, 9, 9)  # moved by the user of the layout since

    ensure_inline_histogram(histogram, label=label, controls=controls, row=0, layout=grid)

    assert grid.placed[histogram] == (9, 9, 9, 9)


def test_each_histogram_has_its_own_rows_when_both_are_added():
    labels = {'size': FakeWidget('size label'), 'source': FakeWidget('intensity label')}
    controls = {'size': FakeWidget('size controls'), 'source': FakeWidget('intensity controls')}
    histograms = {'size': FakeWidget('size histogram'), 'source': FakeWidget('intensity histogram')}
    grid = FakeGrid()

    for position, key in enumerate(('size', 'source')):
        ensure_inline_histogram(histograms[key], label=labels[key], controls=controls[key],
                                row=2 * position, layout=grid)

    rows = {widget.name: placement[0] for widget, placement in grid.placed.items()}
    assert rows == {'size histogram': 0, 'size label': 1, 'size controls': 1,
                    'intensity histogram': 2, 'intensity label': 3, 'intensity controls': 3}
