import os
import json
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

########################### Label table creation

### Utility functions for label table creation

def get_flattened_structure(structure):
    """
    flattens the initial nested dict into a list of dicts (one dict per structure), retaining all the information
    """
    children_list = []
    children = structure.get('children')  # can be empty list
    for child in children:
        children_list.append(child)
        children_list.extend(get_flattened_structure(child))  # recursion
    return children_list

def get_direct_children_structures_ids(children):
    """
    children: list of structures
    returns a list of the ids of direct children only
    """
    return [child.get("id") for child in children]


def get_all_children_structures_ids(children):
    """
    children: list of structures
    returns a list of the ids of all children, (direct children, their children and so on)
    """
    list_all_children = children.copy()
    for child in children:
        list_all_children.extend(get_flattened_structure(child))
    return [child.get("id") for child in list_all_children]

def get_structure_path(structure_id, df):
    """
    Parameters
    ----------
    structure_id: int
        id of the structure of interest
    Returns
    -------
    structure_path: list of int
        path from root structure to structure of interest
        example: [997, 8, 343, 313, 348, 165, 100]
    """
    df = df.set_index('id')
    structure_path = [int(structure_id)]
    while structure_id:
        structure_id = df.loc[structure_id, "parent_structure_id"]
        structure_id = 0 if np.isnan(structure_id) else structure_id
        structure_path = [int(structure_id)] + structure_path if structure_id else structure_path
    return structure_path

### Main function for label table creation

def create_label_table(fpath, save=False, from_cached=False):
    """
    Parameters
    ----------
    fpath: str
        Path to a JSON file similar to the one downloaded from 'http://api.brain-map.org/api/v2/structure_graph_download/1.json'

    Returns
    -------
    df: pd.DataFrame
        dataframe holding informations on the labels ()
    """

    assert fpath.endswith('.json')

    if from_cached:
        if os.path.isfile(fpath + 'l'):
            return pd.read_json(fpath + "l", orient="records", lines=True)

    with open(fpath, 'r') as file_in:
        data = json.load(file_in)['msg']
    assert len(data) == 1

    ## flatten the label file
    data.extend(get_flattened_structure(data[0]))
    df = pd.DataFrame(data)

    ## add 3 columns
    df["direct_children_structures_ids"] = df.children.map(get_direct_children_structures_ids)
    df['all_children_structures_ids'] = df.children.map(get_all_children_structures_ids)
    df['structure_path'] = df['id'].map(lambda x: get_structure_path(x, df))

    ## filter columns of interest
    cols_kept = [
        'id',
        'acronym',
        'name',
        'color_hex_triplet',
        'st_level',
        'parent_structure_id',
        'direct_children_structures_ids',
        'all_children_structures_ids',
        'structure_path',
    ]
    df = df[cols_kept].copy()

    ## save to a JSONL file
    if save:
        df.to_json(fpath + 'l', orient="records", lines=True)

    return df