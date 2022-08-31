import numpy as np


def sanitize_df(df, id_col_name='Structure ID'):
    valid_idx = np.logical_and(df[id_col_name] > 0, df[id_col_name] < 2 ** 16)
    df = df[valid_idx]
    df = df[df[id_col_name] != 997]  # Not "brain"
    return df


def sanitize_df_column_names(df):
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
