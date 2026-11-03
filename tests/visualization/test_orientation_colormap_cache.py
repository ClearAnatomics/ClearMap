"""
The orientation colormap of n colors per axis is an array of shape (n, n, n, 3), indexed by x, y, z, cached in memory
and in a file named after n. Asking again for the same n uses the cache, and a file is never taken for another size.
"""
import numpy as np
import pytest

from ClearMap import Settings as settings
from ClearMap.Utils.exceptions import ClearMapValueError
from ClearMap.Visualization.Color import orientation_colormap as omap

N_COLORS = 3


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """The folder of the program and the cache folder are temporary ones: (program_dir, cache_dir)."""
    program_dir, cache_dir = tmp_path / 'program', tmp_path / 'cache'
    program_dir.mkdir()
    monkeypatch.setattr(omap, 'COLORMAP_FOLDER', program_dir)
    monkeypatch.setattr(settings, 'clearmap_cache_dir', cache_dir)
    monkeypatch.setattr(omap, '_colormap_cache', None)
    return program_dir, cache_dir


@pytest.fixture
def computations(folders, monkeypatch):
    """The calls to the computation are counted."""
    calls = []
    compute = omap.orientation_color

    def counting_orientation_color(*args, **kwargs):
        calls.append(1)
        return compute(*args, **kwargs)

    monkeypatch.setattr(omap, 'orientation_color', counting_orientation_color)
    return calls


def test_the_colormap_has_n_colors_per_axis_and_is_computed_once_for_the_same_n(computations):
    first = omap.orientation_colormap_cache(n=N_COLORS)
    second = omap.orientation_colormap_cache(n=N_COLORS)

    assert first.shape == (N_COLORS,) * 3 + (3,)
    assert second is first
    assert len(computations) == 1


def test_the_colormap_file_of_the_same_size_is_read_not_computed_again(computations, monkeypatch):
    first = omap.orientation_colormap_cache(n=N_COLORS)
    monkeypatch.setattr(omap, '_colormap_cache', None)  # a new process

    second = omap.orientation_colormap_cache(n=N_COLORS)

    assert np.array_equal(second, first)
    assert len(computations) == 1


def test_each_size_has_its_own_file_and_is_computed_once(computations, folders):
    _, cache_dir = folders
    omap.orientation_colormap_cache(n=N_COLORS)
    other = omap.orientation_colormap_cache(n=N_COLORS + 1)
    back = omap.orientation_colormap_cache(n=N_COLORS)

    assert other.shape == (N_COLORS + 1,) * 3 + (3,)
    assert back.shape == (N_COLORS,) * 3 + (3,)
    assert len(computations) == 2
    assert sorted(path.name for path in cache_dir.glob('omap*.npy')) == [f'omap_{N_COLORS}.npy',
                                                                         f'omap_{N_COLORS + 1}.npy']


def test_the_colormap_file_of_the_former_versions_is_not_taken_for_this_one(computations, folders):
    program_dir, _ = folders
    former = np.zeros((N_COLORS + 1,) * 3 + (3,))  # n + 1 colors per axis, and x and y swapped
    np.save(program_dir / 'omap.npy', former)

    colormap = omap.orientation_colormap_cache(n=N_COLORS)

    assert colormap.shape == (N_COLORS,) * 3 + (3,)
    assert len(computations) == 1
    assert np.array_equal(np.load(program_dir / 'omap.npy'), former)  # left alone, never read


def test_a_file_that_does_not_have_the_size_of_its_name_is_refused(folders):
    program_dir, _ = folders
    np.save(program_dir / f'omap_{N_COLORS}.npy', np.zeros((N_COLORS + 1,) * 3 + (3,)))

    with pytest.raises(ClearMapValueError, match=f'omap_{N_COLORS}.npy'):
        omap.orientation_colormap_cache(n=N_COLORS)


def test_the_cached_colors_are_those_of_the_orientation_not_of_the_orientation_with_x_and_y_swapped(folders):
    cache = omap.orientation_colormap_cache(n=41)
    random_generator = np.random.default_rng(0)
    orientations = random_generator.uniform(-1, 1, (500, 3))
    orientations /= np.linalg.norm(orientations, axis=1, keepdims=True)

    cached = omap.orientation_color_cached(orientations, cache=cache)

    exact = omap.orientation_color(orientations, normalize=True)
    assert np.abs(cached - exact).mean() < 0.05  # 0.3 when x and y are swapped


def test_the_colormap_file_is_made_in_the_cache_and_not_next_to_the_module(folders, computations):
    program_dir, cache_dir = folders

    omap.orientation_colormap_cache(n=N_COLORS)

    assert (cache_dir / f'omap_{N_COLORS}.npy').exists()
    assert list(program_dir.iterdir()) == []


def test_a_colormap_file_prepared_next_to_the_module_is_used(folders, computations):
    program_dir, cache_dir = folders
    prepared = np.zeros((N_COLORS,) * 3 + (3,))
    np.save(program_dir / f'omap_{N_COLORS}.npy', prepared)

    colormap = omap.orientation_colormap_cache(n=N_COLORS)

    assert np.array_equal(colormap, prepared)
    assert computations == []
    assert not cache_dir.exists()
