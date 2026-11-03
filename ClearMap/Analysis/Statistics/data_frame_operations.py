import numpy as np
import pandas as pd


def sanitize_df(df, id_col_name='Structure ID'):
    """
    Remove the rows corresponding to the "brain" structure and the rows with invalid ids

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to sanitize
    id_col_name : str
        The name of the column containing the ids

    Returns
    -------
    pd.DataFrame
        The sanitized dataframe
    """
    valid_idx = np.logical_and(df[id_col_name] > 0, df[id_col_name] < 2 ** 16)
    df = df[valid_idx]
    df = df[df[id_col_name] != 997]  # Not "brain"
    return df


def _sanitize_df_column_names(df):
    """
    Sanitize the column names of a dataframe by lowercasing them and
    replacing spaces with underscores

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to sanitize

    Returns
    -------
    pd.DataFrame
        The sanitized dataframe
    """
    columns = {c: c.lower().replace(' ', '_') for c in df.columns}
    return df.rename(columns=columns)


def fix_df_column_names(stats_df):
    df = stats_df.rename(columns={'Structure ID': 's_id',
                                  'Hemisphere': 'hem_id',
                                  'Cell counts': 'cell_counts'},
                         # 'Average cell size': 'average_cell_size'},
                         errors='raise')
    return df


def normalise_df_column_names(df):
    """
    Return same names wether df is a cell stats df or a group stats df
    Parameters
    ----------
    df

    Returns
    -------

    """
    columns = {
        'Structure ID': 'structure_id',
        'id': 'structure_id',
        'Structure order': 'structure_order',
        'Structure name': 'structure_name',
        'name': 'structure_name',
        'Hemisphere': 'hemisphere',
        'volume': 'structure_volume',
        'Structure volume': 'structure_volume',
        'Cell counts': 'cell_counts',
        'Average cell size': 'average_cell_size'
    }
    return df.rename(columns=columns, errors='ignore')

# ## utils for dataframe counting, grouping, collapsing, filtering and normalizing


def count_cells(path: str) -> pd.DataFrame:
    """
    counts cells from one file of type cells.feather
    returns df with columns id, hemisphere, cell_count and one row per structure x hemisphere
    """
    df = pd.read_feather(path)
    df['hemisphere'] = df['hemisphere'].map({0: 'LH', 255: 'RH'})
    counts = (df.groupby(['id', 'hemisphere'], as_index=False)
              .agg(cell_count=('name', 'count'))
              )
    counts = counts.reset_index(drop=True)
    return counts


def group_counts(counts_s, sample_names) -> pd.DataFrame:
    """
    groups several cell_counts together; sample_names are the names of the samples
    returns df with columns id, hemisphere, and one column per sample
    """
    counts_s = [counts.set_index(['id', 'hemisphere']) for counts in counts_s]
    df = pd.concat(counts_s, axis=1).fillna(0)
    df.columns = sample_names
    df = df.reset_index()
    return df


def collapse_structures(df: pd.DataFrame, map_collapse, collapse_hemispheres=False) -> pd.DataFrame:
    """
    collapses structures according to a dict map_collapse (id -> new_id)
    ids not in map_collapse are kept
    """
    df['id'] = df['id'].map(lambda x: map_collapse.get(x, x))
    if not collapse_hemispheres:
        counts = (df.groupby(['id', 'hemisphere'], as_index=False)
                  .sum()
                  )
    else:
        counts = (df.groupby(['id'], as_index=False)
                  .sum()
                  )
    return counts


def filter_df(df: pd.DataFrame, structure_ids,
              hemispheres=['RH', 'LH'], exclude: bool=False) -> pd.DataFrame:
    """
    returns a df that includes only the
    """
    if not exclude:
        if 'hemisphere' in df.columns:
            mask = df["id"].isin(structure_ids) & df["hemisphere"].isin(hemispheres)
        else:
            mask = df["id"].isin(structure_ids)
    else:
        if 'hemisphere' in df.columns:
            mask = ~df["id"].isin(structure_ids) & df["hemisphere"].isin(hemispheres)
        else:
            mask = ~df["id"].isin(structure_ids)
    df = df.loc[mask].reset_index(drop=True)
    return df.copy()


def normalize_df(df: pd.DataFrame, df_normalize: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(['id', 'hemisphere']).copy()
    df_normalize = df_normalize.set_index(['id', 'hemisphere']).copy()
    normalize_100 = df_normalize.sum(axis=0)
    df = df/normalize_100 * 100
    return df.reset_index()
