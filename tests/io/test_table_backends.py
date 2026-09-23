"""Tests for TableSource and the CSV / Feather backends."""
import warnings

import numpy as np
import pandas as pd
import pytest

from ClearMap.IO.source.Source import TableSource
from ClearMap.IO.source.protocol import Backend
from ClearMap.IO.source.backends import csv_backend
from ClearMap.Utils.exceptions import ClearMapValueError, ClearMapNotImplementedError, SourceNotFoundError


@pytest.fixture
def frame():
    return pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0], 'y': [5, 6, 7, 8], 'label': list('abcd')})


def _backends():
    backends = [csv_backend]
    try:
        from ClearMap.IO.source.backends import feather_backend
        backends.append(feather_backend)
    except ImportError:  # pyarrow missing
        pass
    return backends


def _path(tmp_path, backend):
    return str(tmp_path / f'table.{backend.SOURCE_CLASS.backend.extensions[0]}')


# ---- contract --------------------------------------------------------------

@pytest.mark.parametrize('backend', _backends())
def test_backend_identity(backend):
    assert issubclass(backend.SOURCE_CLASS, TableSource)
    assert backend.SOURCE_CLASS.backend.module_name == backend.__name__.rsplit('.', 1)[-1]
    for name in ('open_ro', 'read', 'write', 'create', 'edit'):
        assert callable(getattr(backend, name))


# ---- round trip and item access ----------------------------------------------

@pytest.mark.parametrize('backend', _backends())
def test_round_trip(tmp_path, frame, backend):
    path = _path(tmp_path, backend)
    assert backend.write(path, frame) == path
    pd.testing.assert_frame_equal(backend.read(path), frame)


@pytest.mark.parametrize('backend', _backends())
def test_pandas_item_access(tmp_path, frame, backend):
    path = _path(tmp_path, backend)
    backend.write(path, frame)
    source = backend.open_ro(path)
    assert isinstance(source['x'], pd.Series)
    assert list(source[['x', 'label']].columns) == ['x', 'label']
    assert len(source[1:3]) == 2
    assert list(source[source['y'] > 6]['label']) == ['c', 'd']
    assert backend.read(path, slicing='label').tolist() == list('abcd')


@pytest.mark.parametrize('backend', _backends())
def test_geometry_and_lazy_repr(tmp_path, frame, backend):
    path = _path(tmp_path, backend)
    backend.write(path, frame)
    source = backend.open_ro(path)
    assert 'frame' not in source.__dict__ and '(4, 3)' not in str(source)  # repr does no IO
    assert source.shape == (4, 3) and len(source) == 4 and source.columns == ('x', 'y', 'label')
    assert '(4, 3)' in str(source)


def test_read_returns_caller_owned_copy(tmp_path, frame):
    path = str(tmp_path / 't.csv')
    csv_backend.write(path, frame)
    source = csv_backend.open_ro(path)
    data = source.read()
    data.loc[0, 'x'] = -1
    assert source.frame.loc[0, 'x'] == 1.0


def test_write_through_source_invalidates_cache(tmp_path, frame):
    path = str(tmp_path / 't.csv')
    source = csv_backend.create(path, array=frame)
    assert source.n_rows == 4
    source.write(frame.iloc[:2])
    assert source.n_rows == 2


# ---- header detection --------------------------------------------------------

def test_headerless_legacy_file_is_refused(tmp_path):
    path = str(tmp_path / 'legacy.csv')
    np.savetxt(path, np.random.rand(5, 3), delimiter=',', fmt='%.5e')  # old ClearMap writer
    with pytest.raises(ClearMapValueError, match='header'):
        csv_backend.read(path)
    explicit = csv_backend.read(path, header=None, names=['x', 'y', 'z'])
    assert explicit.shape == (5, 3)


def test_half_numeric_header_is_accepted(tmp_path):
    path = str(tmp_path / 't.csv')
    with open(path, 'w') as f:
        f.write('x,2019\n1,2\n')
    assert csv_backend.read(path).columns.tolist() == ['x', '2019']


def test_numeric_looking_names_refused_on_write(tmp_path):
    with pytest.raises(ClearMapValueError):
        csv_backend.write(str(tmp_path / 't.csv'), pd.DataFrame({'1.0': [1], '2.0': [2]}))


# ---- tables are tables -------------------------------------------------------

@pytest.mark.parametrize('data', [np.zeros((3, 2)), pd.DataFrame(np.zeros((3, 2)))])
def test_arrays_are_not_tables(tmp_path, data):
    with pytest.raises(ClearMapValueError):
        csv_backend.write(str(tmp_path / 't.csv'), data)


def test_index_handling(tmp_path, frame):
    path = str(tmp_path / 't.csv')
    csv_backend.write(path, frame.set_index('label'))
    assert csv_backend.read(path).columns.tolist() == ['label', 'x', 'y']  # named index is data
    csv_backend.write(path, frame[frame['y'] > 6])
    assert csv_backend.read(path).columns.tolist() == ['x', 'y', 'label']  # positional residue dropped


# ---- refusals ----------------------------------------------------------------

def test_unsupported_operations(tmp_path, frame):
    path = str(tmp_path / 't.csv')
    source = csv_backend.create(path, array=frame)
    with pytest.raises(ClearMapNotImplementedError):
        source['x'] = 0
    with pytest.raises(ClearMapNotImplementedError):
        csv_backend.write(path, frame, slicing=slice(0, 2))
    with pytest.raises(ClearMapNotImplementedError):
        csv_backend.edit(path)
    with pytest.raises(ClearMapNotImplementedError):
        source.as_virtual()
    with pytest.raises(FileExistsError):
        csv_backend.write(path, frame, overwrite=False)
    with pytest.raises(ClearMapValueError):
        csv_backend.create(str(tmp_path / 'u.csv'), shape=(2, 2), array=frame)
    with pytest.raises(ClearMapNotImplementedError):
        csv_backend.create(str(tmp_path / 'u.csv'), shape=(2, 2))
    with pytest.raises(ClearMapValueError):
        csv_backend.SOURCE_CLASS(path, mode='r+')


def test_missing_file(tmp_path):
    with pytest.raises(SourceNotFoundError):
        csv_backend.read(str(tmp_path / 'nope.csv'))


def test_legacy_as_source_warns(tmp_path, frame):
    path = str(tmp_path / 't.csv')
    csv_backend.write(path, frame)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        source = csv_backend.read(path, as_source=True)
    assert isinstance(source, csv_backend.CSVSource)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


# ---- arrays with explicit column names ---------------------------------------

def test_array_with_columns_is_a_table(tmp_path):
    path = str(tmp_path / 't.csv')
    csv_backend.write(path, np.arange(6.).reshape(3, 2), columns=['x', 'y'])
    assert csv_backend.read(path).columns.tolist() == ['x', 'y']
    source = csv_backend.create(str(tmp_path / 'u.csv'), array=np.zeros((2, 3)), columns=['a', 'b', 'c'])
    assert source.shape == (2, 3)


@pytest.mark.parametrize('data, columns', [
    (np.zeros((3, 2)), ['x']),                                     # wrong number of names
    (pd.DataFrame({'x': [1]}), ['y']),                             # rename or select? refuse to guess
    (np.zeros(3, dtype=[('x', 'f4'), ('y', 'f4')]), ['x', 'y']),   # structured arrays belong in .npy
])
def test_columns_misuse(tmp_path, data, columns):
    with pytest.raises(ClearMapValueError):
        csv_backend.write(str(tmp_path / 't.csv'), data, columns=columns)


# ---- io_ops and dispatch integration -----------------------------------------

def test_io_ops_tables(tmp_path, frame):
    from ClearMap.IO import io_ops
    path = str(tmp_path / 't.csv')
    io_ops.write(path, frame)                                  # routed by extension, DataFrame passed through
    pd.testing.assert_frame_equal(io_ops.read(path), frame)
    source = io_ops.open_ro(path)
    pd.testing.assert_frame_equal(io_ops.read(source), frame)  # Source branch no longer needs .array
    assert io_ops.read(source, slicing='label').tolist() == list('abcd')
    io_ops.write(path, np.ones((2, 2)), columns=['a', 'b'])
    assert io_ops.read(path).columns.tolist() == ['a', 'b']


def test_as_source_refuses_table_slicing(tmp_path, frame):
    from ClearMap.IO import dispatch
    path = str(tmp_path / 't.csv')
    csv_backend.write(path, frame)
    assert isinstance(dispatch.as_source(path), csv_backend.CSVSource)
    with pytest.raises(ClearMapValueError):
        dispatch.as_source(path, slicing='x')