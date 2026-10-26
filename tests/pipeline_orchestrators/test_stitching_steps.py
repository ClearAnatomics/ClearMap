"""
The steps the stitching runs (steps_of), which size the main progress bar, are the ones it moves it by: a skipped step
is not run (its result of a previous run is used), the other channels only stitch with the layout of the first one.
The GUI used to count them from the skip flags, and to run all the steps whatever they said ("step 5/4").

The tiles are stitched for real, only the graph library of the rigid alignment (graph_tool, which the tests of the
orchestrators do not need) is replaced by plain Python: the layouts of a few tiles are small.
"""
import collections

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from ClearMap.Alignment.Stitching import layout_graph_utils, stitching_rigid
from ClearMap.Utils.exceptions import MissingRequirementException

from .conftest import RAW_FOLDER_NAME

N_TILES = 2
OVERLAP = 8
TILE_SHAPE = (24, 24, 32)  # z, y, x
OME = {'axes': 'ZYX', 'PhysicalSizeX': 1.625, 'PhysicalSizeY': 1.625, 'PhysicalSizeZ': 3.0}


class MainProgressCounter:
    """Stands for the progress watcher: counts how many times the main progress moves"""

    def __init__(self):
        self.main_steps = 0
        self.main_step_name = ''

    def increment_main_progress(self, increment=1):
        self.main_steps += increment

    def prepare_for_substep(self, *args, **kwargs):
        pass

    def increment(self, *args, **kwargs):
        pass


def _connect_sources(alignments, n_sources, source_to_index):
    graph = collections.defaultdict(list)
    for alignment in alignments:
        i, j = source_to_index[alignment.pre], source_to_index[alignment.post]
        graph[i].append((j, np.array(alignment.displacement)))
        graph[j].append((i, -np.array(alignment.displacement)))
    return graph, None


def _positions_from_tree(graph, fixed_source, source_to_index, n_sources, ndim, pos_tree):
    start = source_to_index[fixed_source] if fixed_source is not None else 0
    positions = {start: np.zeros(ndim, int)}
    todo = [start]
    while todo:
        i = todo.pop()
        for j, displacement in graph[i]:
            if j not in positions:
                positions[j] = positions[i] + displacement
                todo.append(j)
    return np.array([positions.get(i, np.zeros(ndim, int)) for i in range(n_sources)])


def _components_of_alignments(alignments, n_sources, source_to_index=None):
    parent = list(range(n_sources))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for alignment in alignments:
        parent[find(source_to_index[alignment.pre])] = find(source_to_index[alignment.post])
    roots = sorted({find(i) for i in range(n_sources)})
    return np.array([roots.index(find(i)) for i in range(n_sources)]), len(roots)


class _Graph:
    def __init__(self, n_vertices):
        self.n = n_vertices
        self.edges = []

    def add_edge(self, i, j):
        self.edges.append((int(i), int(j)))


def _graph_of(n_vertices, directed=False):
    return _Graph(n_vertices)


def _components_of_graph(graph):
    parent = list(range(graph.n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, j in graph.edges:
        parent[find(i)] = find(j)
    roots = sorted({find(i) for i in range(graph.n)})
    return np.array([roots.index(find(i)) for i in range(graph.n)]), len(roots)


@pytest.fixture(autouse=True)
def graph_library_in_python(monkeypatch):
    monkeypatch.setattr(stitching_rigid, 'connect_sources', _connect_sources)
    monkeypatch.setattr(stitching_rigid, 'get_positions_from_tree', _positions_from_tree)
    monkeypatch.setattr(stitching_rigid, 'get_connected_components', _components_of_alignments)
    monkeypatch.setattr(layout_graph_utils, 'get_graph', _graph_of)
    monkeypatch.setattr(layout_graph_utils, 'graph_to_connected_components', _components_of_graph)


def _write_row_of_tiles(folder, prefix):
    rng = np.random.default_rng(0)
    full_shape = (TILE_SHAPE[0], TILE_SHAPE[1], N_TILES * (TILE_SHAPE[2] - OVERLAP) + OVERLAP)
    full = (ndimage.gaussian_filter(rng.random(full_shape), 1.5) * 60000).astype(np.uint16)
    folder.mkdir(parents=True)
    for x in range(N_TILES):
        start = x * (TILE_SHAPE[2] - OVERLAP)
        tifffile.imwrite(folder / f'{prefix}_X{x:02d}.ome.tif', full[:, :, start:start + TILE_SHAPE[2]], metadata=OME)
    return str(folder / f'{prefix}_X<X,2>.ome.tif')


@pytest.fixture()
def stitching(experiment, channel_editor, tmp_path):
    """A stitching worker with 2 channels: 'main' computes its layout, 'derived' is stitched with it"""
    for name in ('main', 'derived'):
        channel_editor.add(name, 'nuclei')
        path = _write_row_of_tiles(tmp_path / RAW_FOLDER_NAME / name, name)
        experiment.apply_ui_patch({'sample': {'channels': {name: {'path': path}}}})
    worker = experiment.get_worker('stitching')
    worker.setup_if_needed()
    rigid = {'overlap_x': OVERLAP, 'overlap_y': OVERLAP,
             'max_shifts_x': [-4, 4], 'max_shifts_y': [-4, 4], 'max_shifts_z': [-2, 2]}
    experiment.apply_ui_patch({'stitching': {'channels': {'main': {'layout_channel': 'main', 'rigid': rigid},
                                                          'derived': {'layout_channel': 'main'}}}})
    worker.progress_watcher = MainProgressCounter()
    return worker


def _set_skip(experiment, rigid=False, wobbly=False):
    experiment.apply_ui_patch({'stitching': {'channels': {
        'main': {'rigid': {'skip': rigid}, 'wobbly': {'skip': wobbly}}}}})


def _stitch_and_count(worker, channels=('main', 'derived')) -> int:
    worker.progress_watcher.main_steps = 0
    for channel in channels:
        worker.stitch_channel(channel)
    return worker.progress_watcher.main_steps


def test_the_steps_of_a_channel_which_computes_its_layout(stitching, experiment):
    assert stitching.steps_of('main') == ['rigid alignment', 'wobbly alignment', 'wobbly placement', 'stitching']

    _set_skip(experiment, rigid=True)
    assert stitching.steps_of('main') == ['wobbly alignment', 'wobbly placement', 'stitching']

    _set_skip(experiment, wobbly=True)
    assert stitching.steps_of('main') == ['rigid alignment', 'stitching']

    _set_skip(experiment, rigid=True, wobbly=True)
    assert stitching.steps_of('main') == ['stitching']


def test_a_channel_stitched_with_another_layout_only_stitches(stitching, experiment):
    assert stitching.steps_of('derived') == ['stitching']
    assert stitching.n_steps_to_run(['main', 'derived']) == 5


def test_the_main_progress_moves_by_the_number_of_steps_of_the_plan(stitching, experiment):
    assert _stitch_and_count(stitching) == stitching.n_steps_to_run(['main', 'derived'])


def test_a_skipped_rigid_alignment_is_not_run_the_result_of_the_previous_run_is_used(stitching, experiment):
    _stitch_and_count(stitching)  # The rigid alignment is done
    _set_skip(experiment, rigid=True)

    assert _stitch_and_count(stitching) == stitching.n_steps_to_run(['main', 'derived']) == 4


def test_a_skipped_rigid_alignment_without_previous_result_is_reported(stitching, experiment):
    _set_skip(experiment, rigid=True)

    with pytest.raises(MissingRequirementException, match='no result of a previous run'):
        stitching.stitch_channel('main')


def test_a_skipped_wobbly_stitching_places_the_rigid_layout_and_stitches(stitching, experiment):
    _set_skip(experiment, wobbly=True)

    assert _stitch_and_count(stitching) == stitching.n_steps_to_run(['main', 'derived']) == 3
    assert stitching.get('stitched', channel='main').exists
    assert stitching.get('stitched', channel='derived').exists


def test_a_channel_with_an_existing_layout_only_stitches(stitching, experiment):
    _stitch_and_count(stitching)
    experiment.apply_ui_patch({'stitching': {'channels': {'main': {'use_existing_layout': True}}}})

    assert stitching.steps_of('main') == ['stitching']
    assert _stitch_and_count(stitching) == stitching.n_steps_to_run(['main', 'derived']) == 2
