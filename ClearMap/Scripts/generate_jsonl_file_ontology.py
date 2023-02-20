import pandas as pd
import numpy as np
import json
from ClearMap.Settings import atlas_folder
import os

def get_flattened_structure(structure):
        """
        flatten any structure
        structure : dict
        """
        children_list = []
        children = structure.get('children')  # can be empty list
        for child in children:
            children_list.append(child)
            children_list.extend(get_flattened_structure(child))  # recursion
        return children_list

def get_direct_children_structures_ids(children):
  """
  list the ids of direct children only
  """
  return [child.get("id") for child in children]

def get_all_children_structures_ids(children):
    """
    list the ids of direct children, their children and so on
    """
    list_all_children = children.copy()
    for child in children:
        list_all_children.extend(get_flattened_structure(child))
    return [child.get("id") for child in list_all_children]

def get_parent_id(structure_id):
  value = df.loc[df['id'] == structure_id, "parent_structure_id"].values
  if value.size:
    if np.isnan(value[0]) == False:
      return int(value[0])
      
def get_structure_path(structure_id):
  path_structure = [int(structure_id)]
  while structure_id:
    structure_id = get_parent_id(structure_id)
    path_structure = [structure_id] + path_structure if structure_id else path_structure
  return path_structure 

def make_structure_ids_path_string(path_structure):
  str_path_structure = ''
  for i in path_structure:
    str_path_structure = f'{str_path_structure}/{i}'
  return str_path_structure[1:]

def make_structure_acronyms_path_string(path_structure):
  str_path_structure = ''
  for i in path_structure:
    str_path_structure = f'{str_path_structure} > {dict_id_to_acronym[i]}'
  return str_path_structure[3:]


atlas_name = "ABA_annotation_last"
fpath = os.path.join(atlas_folder, f"{atlas_name}.json")
with open(fpath, "r") as file:
    data = json.load(file)["msg"]

assert len(data) == 1

data.extend(get_flattened_structure(data[0]))
df = pd.DataFrame(data)
df["direct_children_structures_ids"] = df.children.map(get_direct_children_structures_ids)
df['all_children_structures_ids'] = df.children.map(get_all_children_structures_ids)
df['structure_path'] = df['id'].map(get_structure_path)
dict_id_to_acronym = dict(zip(df['id'], df['acronym']))
df['structure_ids_path'] = df['structure_path'].map(make_structure_ids_path_string)
df['structure_acronyms_path'] = df['structure_path'].map(make_structure_acronyms_path_string)

cols_of_interest = [
    'id', 
    'acronym', 
    'name', 
    'color_hex_triplet', 
    'st_level', 
    'parent_structure_id', 
    'direct_children_structures_ids',
    'all_children_structures_ids',
    'structure_path',
    'structure_ids_path',
    'structure_acronyms_path'
    ]

df_to_save = df[cols_of_interest].copy()

fpath = os.path.join(atlas_folder, f"{atlas_name}.jsonl")
print(fpath)
df_to_save.to_json(fpath, orient="records", lines=True)