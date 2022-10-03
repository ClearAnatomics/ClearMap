import numpy as np
import pandas as pd


def get_all_structs(dfs):
    """
    Get all the structures that are in any of the dataframes

    Parameters
    ----------
    dfs list(pd.DataFrame)

    Returns
    -------

    """
    structs = pd.Series()
    for df in dfs:
        structs = pd.concat((structs, df['id']))
    return np.sort(structs.unique())
