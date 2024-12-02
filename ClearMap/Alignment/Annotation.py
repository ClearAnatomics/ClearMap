# -*- coding: utf-8 -*-
"""
Annotation
==========

Module to handle label and annotation info from Allen Brain Atlas (v2).
 
Notes
-----      
  - The files containing annotation information is assumed to be in the path
    stored in :const:`atlas_path`, which defaults to the resource path 
    :const:`ClearMap.Settings.resources_path`.
    
  - The annotation file is assumed to be 'annotation_25_full.nrrd'
    but can be set via :const:`default_annotation_file`.

  - The mapping between labels and brain area information is assumed to be in 
    the 'annotation.json' file but the location can be set
    via :const:`default_label_file`.     
      
References
----------
  - `Allen Brain Atlas <https://mouse.brain-map.org/static/atlas>`_
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://github.com/ClearAnatomics/ClearMap'


# TODO: inherit Label from dict
# TODO: integrate with atlas meshes / plotting / transforming data / region selection


import os
import collections

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

import ClearMap.Settings
import ClearMap.Settings as settings

import ClearMap.IO.IO as clearmap_io
import ClearMap.IO.FileUtils as fu

from ClearMap.Alignment.Resampling import orientation_to_transposition, format_orientation
import ClearMap.Utils.HierarchicalDict as hdict

from ClearMap.Alignment.utils import create_label_table
from ClearMap.IO import IO as clearmap_io

from ClearMap.Visualization import Color as col
from ClearMap.config.atlas import STRUCTURE_TREE_NAMES_MAP


###############################################################################
# ## Atlas Structures
###############################################################################

def decompress_atlases(atlas_base_name):  # FIXME: add option to move to a different folder if we can't write to the current one
    paths = {}
    atlas_component_names = ('annotation', 'hemispheres', 'reference', 'distance_to_surface')
    for atlas_type in atlas_component_names:
        f_path = os.path.join(settings.atlas_folder, f'{atlas_base_name}_{atlas_type}.tif')
        fu.uncompress(f_path, extension='auto')
        paths[atlas_type] = f_path
    return paths


ATLAS_BASE_NAME = 'ABA_25um'  # warning: change if different atlas


default_annotation_file, default_hemispheres_file,\
default_reference_file, default_distance_to_surface_file = decompress_atlases(ATLAS_BASE_NAME).values()
"""
Note
----
  This file is by default the Allen brain annotated mouse atlas with 25um 
  isotropic resolution.

  Includes the
    - Default volumetric annotated image file.
    - Default hemispheres annotated image file
    - Default reference (grayscale) image file
    - Corresponding distance to surface file

"""

default_label_file = os.path.join(settings.atlas_folder, 'ABA_annotation.json')
"""Default list of labels and region names in the annotated image.

Note
----
  This file is by default the labels for the Allen brain annotated mouse 
  atlas with 25um isotropic resolution.
"""

default_extra_label = [
  (182305696, 453, 'No label', 'NoL'),
  (182305712, 453, 'No label', 'NoL'),
  (312782560, 315, 'No label', 'NoL'),
  (312782592, 453, 'No label', 'NoL'),
  (312782656, 315, 'No label', 'NoL'),
  (526157184, 993, 'No label', 'NoL'),
  (526322272, 500, 'No label', 'NoL'),
  (527696992, 315, 'No label', 'NoL')
]
"""Additional label not in the Allen Brain Atlas label but in the atlas image.

Warning
-------
    This is required for the older version of the Allen Brain Atlas. There
    are no more labels missing in the current version (2017+ with 2022+ json file).
    
Note
----
  The form is a list of tuples, each tuple has the form 
  (atlas id, parent id, name, acronym).
"""


class Label(object):
    """Class holding information of an individual Atlas label."""

    def __init__(self, data, children=None, parent=None, level=0):
        self.data = data
        self.children = children
        self.parent = parent
        self.data['level'] = level

    @property
    def name(self):
        return self.data['name']

    @property
    def id(self):
        return self.data['id']

    @property
    def level(self):
        return self.data['level']

    @property
    def order(self):
        return self.data['order']

    def color(self, *args, **kwargs):
        return col.color(self.data['rgb'], **kwargs)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def parent_list(self, max_depth=None, min_level=None):
        if max_depth is None:
            max_depth = self.level + 1
        if min_level is None:
            min_level = 1
        if max_depth > 0 and self.level > min_level and self.parent is not None:
            return [self] + self.parent.parent_list(max_depth=max_depth-1, min_level=min_level)
        else:
            return []

    def write(self, with_children=True, indent=None):
        return self.__str__(indent=indent, with_children=with_children)

    def info(self, with_children=True, indent=None):
        print(self.write(indent=indent, with_children=with_children))

    def __str__(self, indent=None, with_children=False):
        indent = indent or ''

        s = ''
        if with_children and isinstance(self.children, list):
            for c in self.children:
                s += '\n' + c.__str__(indent=indent + '  ')

        h = f'{indent}{self.name}\n{indent}{"=" * len(self.name)}\n'

        s = h + hdict.write(self.data, head=indent[:-1]) + '\n' + s

        return s

    def __repr__(self, indent=None, with_children=False):
        return self.__str__(indent=indent, with_children=False)


class Annotation(object):
    """Class that holds information of the annotated regions."""

    def __init__(self, atlas_base_name, slicing, orientation, label_source, target_directory=None, extra_label=None):
        """Initialization

        Arguments
        ---------
        atlas_base_name: str
            The prefix of the atlas files. They will be searched for in the ClearMap resources folder.
        slicing : tuple
            The slicing specification after reorienting.
        orientation : tuple, or None.
            The orientation specification. Strings can be 'left' or 'right', for the two hemispheres.
        label_source : str
            File with label information in json format. It can also be the name as defined in the atlas config.
        target_directory : str
            The target directory where to store the resliced/reoriented atlas files. If None, use ClearMap resources folder.
        extra_label : list of tuples
            Additional labels for IDs present in the atlas but not in the label file.
        """

        if not label_source.startswith('/'):
            label_file_path = Path(settings.atlas_folder) / STRUCTURE_TREE_NAMES_MAP[label_source]
        else:
            label_file_path = label_source

        atlas_source_files = decompress_atlases(atlas_base_name)  # just return the paths if already decompressed
        atlas_files = Annotation.prepare_annotation_files(slicing, orientation, directory=target_directory,
                                                          hemispheres_file=atlas_source_files['hemispheres'],
                                                          annotation_file=atlas_source_files['annotation'],
                                                          distance_to_surface_file=atlas_source_files['distance_to_surface'],
                                                          reference_file=atlas_source_files['reference'],
                                                          verbose=True)
        atlas_files = {k: v for k, v in zip(atlas_source_files.keys(), atlas_files) if v}
        atlas_files['hemispheres'] = atlas_files.get('hemispheres', None)

        self.root = None
        self.structures = None
        self.ids = None
        self.names = None
        self.labels = None
        self.acronyms = None
        self.colors_rgb = None
        self.colors_hex = None
        self.df = None

        self.annotation_file = ''
        self.hemispheres_file = ''
        self.distance_to_surface_file = ''
        self.reference_file = ''

        self.label_file = None
        self.extra_label = None

        self.dict_id_to_acronym = {}
        self.dict_id_to_name = {}
        self.dict_id_to_color = {}
        # self.dict_id_to_path = {}
        self.dict_acronym_to_id = {}
        self.dict_name_to_id = {}

        self.initialize(annotation_file_path=atlas_files['annotation'],
                        hemispheres_file_path=atlas_files['hemispheres'],
                        distance_to_surface_file_path=atlas_files['distance_to_surface'],
                        reference_file_path=atlas_files['reference'],
                        label_file_path=label_file_path,
                        extra_label=extra_label)

    def _initialize_dataframe(self):
        df = pd.DataFrame({
            "id": self.ids,
            "name": self.names,
            "acronym": self.acronyms,
            "colors_hex": self.colors_hex,
            #"colors_rgb": self.colors_rgb,
        })
        df["colors_rgb"] = df["colors_hex"].map(lambda x: col.hex_to_rgb(x))
        return df

    def get_atlas_paths(self):
        return {
            'annotation': self.annotation_file,
            'hemispheres': self.hemispheres_file,
            'distance_to_surface': self.distance_to_surface_file,
            'reference': self.reference_file
        }

    def get_dict(self, from_='id', to='acronym'):
        return dict(zip(self.df[from_], self.df[to]))

    def ids_to_acronyms(self, ids):
        return [self.dict_id_to_acronym[id_] for id_ in ids]

    def ids_to_names(self, ids):
        return [self.dict_id_to_name[id_] for id_ in ids]

    def get_colors_rgba(self, alpha=1):
        return self.df["colors_hex"].map(lambda x: col.hex_to_rgb(x, alpha=alpha))

    def set_annotation_file(self, annotation_file_path):
        print(f"Setting annotation file to {annotation_file_path}.\n"
              f"The annotator will be reinitialized.")
        self.initialize(annotation_file_path=annotation_file_path)

    def set_label_file(self, label_file_path):
        print(f"Setting label file to {label_file_path}.\n"
              f"The annotator will be reinitialized.")
        self.initialize(label_file_path=label_file_path)

    def initialize(self, annotation_file_path=None, hemispheres_file_path=None,
                   distance_to_surface_file_path=None, reference_file_path=None,
                   label_file_path=None,
                   extra_label=None):
        if label_file_path is None:
            if self.label_file is None:
                warnings.warn(f'Label file not defined, using default: {default_label_file}.\n'
                              f'Please note this is deprecated and will be removed in future versions.',
                              DeprecationWarning)
                label_file_path = default_label_file
            else:
                label_file_path = self.label_file
        if annotation_file_path is None:
            if self.annotation_file is None:
                warnings.warn(f'Annotation file not defined, using default: {default_annotation_file}.\n'
                              f'Please note this is deprecated and will be removed in future versions.',
                              DeprecationWarning)
                annotation_file_path = default_annotation_file
            else:
                annotation_file_path = self.annotation_file
        if extra_label is None:
            extra_label = self.extra_label or default_extra_label
        if not extra_label:   # add nodes for missing labels
            extra_label = []

        self.label_file = label_file_path
        self.annotation_file = annotation_file_path
        self.hemispheres_file = hemispheres_file_path or self.hemispheres_file
        self.distance_to_surface_file = distance_to_surface_file_path or self.distance_to_surface_file
        self.reference_file = reference_file_path or self.reference_file
        self.extra_label = extra_label

        # initialize label tree
        with open(label_file_path, 'r') as file_in:
            aba = json.load(file_in)

        root = aba['msg'][0]
        self.root = self.initialize_tree(root)

        # maxgraph = max(self.get_list('graph_order'))
        for a in extra_label:
            i, p, n, l = a
            node = self.find(p, key='id')
            data = node.data.copy()
            data['id'] = i
            data['name'] = n
            data['acronym'] = l
            data['parent_structure_id'] = p
            data['graph_order'] = -1
            data['atlas_id'] = -1
            node.children.append(Label(data, parent=node, children=[], level=node.level+1))

        # initialize generic id
        self.add_data('order', range(self.n_structures))

        # initialize color label
        self.colors_hex = self.get_list('color_hex_triplet')
        self.colors_rgb = np.array([col.hex_to_rgb(c) for c in self.colors_hex])
        self.add_data('rgb', self.colors_rgb)
        self.add_data('color_order', range(self.n_structures))

        # initialize lookup tables
        self.structures = self.get_list(None)
        self.ids = self.get_list('id')
        self.labels = self.get_list('atlas_id')
        self.names = self.get_list('name')
        self.acronyms = self.get_list('acronym')

        # build dataframe
        self.df = self._initialize_dataframe()

        # build dictionaries
        self.dict_id_to_acronym = self.get_dict(from_='id', to='acronym')
        self.dict_id_to_name = self.get_dict(from_='id', to='name')
        self.dict_id_to_color = self.get_dict(from_='id', to='colors_hex')
        # self.dict_id_to_path = self.get_dict(from_='id', to='structure_acronym_path') #TODO to be implemented
        self.dict_acronym_to_id = self.get_dict(from_='acronym', to='id')
        self.dict_name_to_id = self.get_dict(from_='name', to='id')

        # import atlas
        self.atlas = clearmap_io.read(self.annotation_file)
        if self.atlas.dtype.kind == 'f':
            self.atlas = self.atlas.astype(int)
        self.children_df = create_label_table(self.label_file, save=False, from_cached=True)

    def initialize_tree(self, root, parent=None, level=0):
        label = Label({k: v for k, v in root.items() if k != "children"}, parent=parent, level=level)
        label.children = [self.initialize_tree(c, parent=label, level=level + 1) for c in root['children']]
        return label

    def get_list(self, key=None, node=None, level=None):
        if node is None:
            node = self.root

        l = []

        if level is not None:
            n = node
            while n.level > level:
                n = n.parent
        else:
            n = node

        if key is not None:
            l.append(n[key])
        else:
            l.append(n)

        for c in node.children:
            l.extend(self.get_list(node=c, key=key, level=level))

        return l

    @property
    def n_structures(self):
        return len(self.get_list())

    @property
    def max_level(self):
        return np.max(self.get_list('level'))

    def get_hierarchical_dictionary(self, node=None):
        if node is None:
            node = self.root

        d = node.data
        d['children'] = [self.get_hierarchical_dictionary(node=n) for n in node.children]

        return d

    def get_dictionary(self, key, value, node=None, level=None, ordered=False,
                       with_parents=False, max_depth=None, min_level=None):
        if node is None:
            node = self.root

        keys = self.get_list(key=key, node=node, level=None)
        values = self.get_list(key=value, node=node, level=level)

        if ordered:
            dictionary = collections.OrderedDict()
        else:
            dictionary = dict()
        for k, v in zip(keys, values):
            dictionary[k] = v

        if with_parents:
            for k, v in dictionary.items():
                node = self.find(k, key=key)
                parent_list = node.parent_list(max_depth=max_depth, min_level=min_level)
                dictionary[k] = tuple(node[value] for node in parent_list)

        return dictionary

    def get_map(self, key, value, node=None, level=None):
        d = self.get_dictionary(key=key, value=value, node=node, level=level)

        n = max(d.keys())
        m = np.zeros(n+1, dtype=int)
        m[list(d.keys())] = list(d.values())

        return m

    def add_data(self, name, data):
        nodes = self.get_list()
        for n, d in zip(nodes, data):
            n.data[name] = d

    def convert_label(self, label, key='id', value='order', level=None, method='map', node=None):
        if value.lower().startswith('rgb') and len(value) <= 4:
            alpha = value.lower().endswith('a')
            as_int = value.startswith('RGB')
            return self.label_to_color(label, key=key, alpha=alpha, as_int=as_int)
        # string keys or values, force dictionary mode
        if key in ('acronym', 'name', 'color_hex_triplet') or value in ('acronym', 'name', 'color_hex_triplet'):
            method = 'dictionary'
        if method == 'map':
            m = self.get_map(key=key, value=value, node=node, level=level)
            return m[label]
        else:
            d = self.get_dictionary(key=key, value=value, node=node, level=level)
            return np.vectorize(d.__getitem__, otypes=[type(list(d.values())[0])])(label)

    def label_to_color(self, label, key='id', level=None, alpha=True, as_int=False, int_type='uint8'):
        cm = self.colors_rgb
        cm = col.color(cm, alpha=alpha, as_int=as_int, int_type=int_type)
        if key != 'order' or level is not None:
            label = self.convert_label(label, key=key, value='order', level=level)
        return cm[label]

    def find(self, label, key='id', value=None, node=None, level=None):
        d = self.get_dictionary(key=key, value=value, node=node, level=level)
        if isinstance(label, list):  # FIXME: iterable
            return [d[l] for l in label]
        else:
            return d[label]

    def parents(self, label, key='id', value=None):
        d = self.get_dictionary(key=key, value=None, node=None, level=None)
        p = d[label]
        l = [p]
        while p.level > 0:
            p = p.parent
            l.append(p)

        if value is None:
            return l
        else:
            return [ll[value] for ll in l]

    def common_parent(self, label, key='id', value=None):
        parents = [self.parents(l, key=key, value=key) for l in label]

        n = min([len(p) for p in parents])
        p0 = parents[0]
        level = -1
        while -level-1 < n and np.all([p[level] == p0[level] for p in parents]):
            level -= 1

        return self.find(p0[level+1], key=key, value=value)

    @property
    def map_volume(self):
        uniques, counts = np.unique(self.atlas, return_counts=True)
        return dict(zip(uniques, counts))

    def get_lateralised_volume_map(self, atlas_scale, hemispheres_file_path=None):
        if hemispheres_file_path is None:
            hemispheres_file_path = self.hemispheres_file
        hemispheres_atlas = clearmap_io.read(hemispheres_file_path)
        scale = np.prod(atlas_scale)
        hem_ids = sorted(np.unique(hemispheres_atlas))
        volumes = {}
        for hem_id in hem_ids:
            unique_ids, counts = np.unique(self.atlas[hemispheres_atlas == hem_id], return_counts=True)
            for id_, count in zip(unique_ids, counts):
                volumes[(id_, hem_id)] = count * scale
        return volumes

    def get_dict_parents_to_children(self, parents_ids=None, including_parents=False):
        map_children = {}
        for parent_id in parents_ids:
            map_children[parent_id] = self.children_df.set_index('id').loc[parent_id, 'all_children_structures_ids'].copy()
        if including_parents:
            for parent in parents_ids:
                map_children[parent].append(parent)
        return map_children

    def get_dict_children_to_parents(self, parents_ids=None, including_parents=False):
        map_children = self.get_dict_parents_to_children(parents_ids=parents_ids, including_parents=including_parents).copy()
        map_parent = {}
        for parent in map_children:
            for child in map_children[parent].copy():
                map_parent[child] = parent
        return map_parent

    def get_map_to_parent(self, parent_ids):
        """
        uses annotation graph to map all possible children structures to its parent in parent_ids
        """
        return self.get_dict_children_to_parents(parent_ids, including_parents=True)

    def get_children(self, structure_ids):
        if isinstance(structure_ids, int):
            structure_ids = [structure_ids]
        map_children = self.get_dict_parents_to_children(parents_ids=structure_ids, including_parents=False)
        children = []
        for parent in map_children:
            children.extend(map_children[parent])
        return children

    def enrich_df(self, df):
        df = df.copy()
        df['name'] = df['id'].map(self.dict_id_to_name)
        df['acronym'] = df['id'].map(self.dict_id_to_acronym)
        return df

    def label_points_hemispheres(self, points, key='id', level=None, invalid=0):
        return self.label_points(points, annotation_file_path=self.hemispheres_file, key=key, level=level, invalid=invalid)

    def label_points(self, points, annotation_file_path=None, key='id', level=None, invalid=0):  # FIXME: document level
        if annotation_file_path:
            atlas = clearmap_io.read(annotation_file_path)
        else:
            atlas = self.atlas

        # Create mask of coordinates within annotation file coordinates (of shape = len(coordinates))  # TODO: check if warn
        xs, ys, zs = points.astype(int).T
        xmax, ymax, zmax = self.atlas.shape
        within_atlas = (xs >= 0) & (xs < xmax) & (ys >= 0) & (ys < ymax) & (zs >= 0) & (zs < zmax)

        # Create alist of labels (shape = len(coordinates)) which defaults to 'invalid' and where
        # coordinates within the atlas have the atlas value
        cell_labels = np.full(xs.shape, invalid, dtype=np.uint64)  # FIXME: dtype=int in original
        cell_labels[within_atlas] = atlas[xs[within_atlas], ys[within_atlas], zs[within_atlas]]

        # convert cell labels to 'key'
        if key != 'id' or level is not None:
            if annotation_file_path:
                if 'hemisphere' in annotation_file_path.lower():
                    raise ValueError(f'Label conversion for {annotation_file_path} is not implemented')
                else:
                    warnings.warn(f'Label conversion for {annotation_file_path} may not be implemented')
            cell_labels[within_atlas] = self.convert_label(cell_labels[within_atlas], key='id', value=key, level=level)

        return cell_labels

    def get_names_map(self):
        return dict(zip(self.ids, self.names))

    def __str__(self):
        return f'Annotation({self.n_structures})[{self.max_level}]{{{self.label_file}}}'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def prepare_annotation_files(slicing, orientation, directory=None, annotation_file=None,
                                 hemispheres_file=None, reference_file=None, distance_to_surface_file=None,
                                 overwrite=False, verbose=False):
        """
        Crop the annotation, reference and distance files to match the data.

        Arguments
        ---------
        slicing : tuple or None
            The slice specification after reorienting.
        orientation : tuple, str or None.
            The orientation specification. Strings can be 'left' or 'right', for the
            two hemispheres.
        directory : str or None
            The target directory. If None, use ClearMap resources folder.
        annotation_file : str or None
            The annotation file to use.
        hemispheres_file : str or None
            The hemispheres annotation file to use. "hemispheres" must be set to True
        reference_file : str or None
            The reference file to use.
        distance_to_surface_file : str or None
            The distance file to use.
        overwrite : bool
            If True, overwrite existing files.
        verbose : bool
            Whether to print verbose output.

        Returns
        -------
        annotation_file : str
            The cropped annotation file.
        reference_file : str
            The cropped reference file.
        distance_to_surface_file : str
            The distance cropped file.
        """
        return_hemispheres = hemispheres_file is not None

        file_paths = {  # must be ordered
            'annotation': annotation_file,
            'hemispheres': hemispheres_file,
            'reference': reference_file,
            'distance_to_surface': distance_to_surface_file
        }
        specified_paths = {k: v for k, v in file_paths.items() if v}
        if not specified_paths:
            file_paths['annotation'] = default_annotation_file
            file_paths['reference'] = default_reference_file
            file_paths['distance_to_surface'] = default_distance_to_surface_file
            if return_hemispheres:
                file_paths['hemispheres'] = default_hemispheres_file
        else:  # infer the other paths
            existing_keys = list(specified_paths.keys())  # TODO: we should check that if > 1 key, they share pattern
            missing_keys = [k for k in file_paths if k not in existing_keys]
            for k in missing_keys:
                file_paths[k] = file_paths[existing_keys[0]].replace(existing_keys[0], k)

        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        results = []
        for f_path in file_paths.values():
            if not f_path:
                continue
            new_file_path = get_atlas_filepath(f_path, directory=directory, orientation=orientation, slicing=slicing)
            if verbose:
                print(f'Preparing: {new_file_path}')

            if not overwrite and os.path.exists(new_file_path):
                if verbose:
                    print('Atlas file exists, skipping')
                results.append(new_file_path)
                continue

            if not os.path.exists(f_path):
                raise FileNotFoundError(f'Cannot find annotation file: {f_path}')

            # Extract array
            src = clearmap_io.as_source(f_path)
            if verbose:
                print(f'Preparing: from source {src}')
            data = np.array(src.array)

            if orientation is not None:
                # transpose data to match orientation
                per = orientation_to_transposition(orientation)
                data = data.transpose(per)

                # reverse axes if orientation[axis] is negative
                slices = [slice(None, None, -1) if ori < 0 else slice(None) for ori in orientation]
                if any([s != slice(None) for s in slices]):
                    data = data[tuple(slices)]
            # Crop
            if slicing is not None:
                data = data[slicing]

            clearmap_io.write(new_file_path, data)
            results.append(new_file_path)

        return results

    def get_columns(self, coordinates_transformed, atlas_resolution, ids=None):  # TODO: rename
        if ids is None:
            ids = self.label_points(coordinates_transformed)
        out = pd.DataFrame({'id': ids})
        out['xt'] = coordinates_transformed[:, 0]
        out['yt'] = coordinates_transformed[:, 1]
        out['zt'] = coordinates_transformed[:, 2]
        if self.hemispheres_file:
            out['hemisphere'] = self.label_points_hemispheres(coordinates_transformed)

        out['name'] = self.convert_label(out['id'], key='id', value='name')

        unique_ids = np.sort(out['id'].unique())

        order_map = {id_: self.find(id_, key='id')['order'] for id_ in unique_ids}
        out['order'] = out['id'].map(order_map)

        color_map = {id_: self.find(id_, key='id')['rgb'] for id_ in unique_ids}  # WARNING RGB upper case should give integer but does not work
        out['color'] = out['id'].map(color_map)

        volumes = self.get_lateralised_volume_map(atlas_resolution)
        if self.hemispheres_file:
            out['volume'] = out.set_index(['id', 'hemisphere']).index.map(volumes.get)
        else:
            out['volume'] = out['id'].map(volumes)
        return out

    def color_map(self, color_ids=None, alpha=True, as_int=False, int_type='uint8'):
        """
        Generates a color map from color ids to rgb

        Arguments
        ---------
        color_ids: list
            The list of ids to generate the color map for.
            e.g. a rgb tuple, color name from matplotlib or vispy, hex code
        alpha : bool
            If True return a color map with alpha values.
        as_int : bool
            If True return a color map with integer values in the range 0-255.
        int_type : str
            The integer type to use for the output, e.g. 'uint8', 'uint16'.

        Returns
        -------
        color_map : array
            An array of rgb colors for each label.
        """
        if color_ids is None:
            color_ids = self.colors_rgb
        return col.color(color_ids, alpha=alpha, as_int=as_int, int_type=int_type)

    def write_color_palette(self, file_path=''):
        """
        Creates a pal or lut file for Imaris or Imagej based on label colors of atlas.

        Arguments
        ---------
        file_path : str | Path
            The name of the color palette file.

        Returns
        -------
        filename : str
            The name of the file to which the color palette was written.
        """

        cm = self.color_map(alpha=False, as_int=True)

        extension = clearmap_io.file_extension(file_path)
        if extension == 'pal':
            col.write_PAL(file_path, cm)
        elif extension == 'lut':
            col.write_LUT(file_path, cm)
        else:
            raise RuntimeError(f'Color palette format: {extension} not lut or pal')

        return file_path

    def create_color_annotation(self, annotation_file_path='', dest_path=''):  # TEST:
        """Creates a rgb image from the atlas color data.

        Arguments
        ---------
        annotation_file_path : str
            File name of the atlas annotation.
        dest_path : str
            The path to the file where the color atlas should be written.
            If empty, the color atlas is only returned as an array.

        Returns
        -------
        filename : str
            The name of the file to which the color atlas was written.
        """
        if annotation_file_path:
            atlas = clearmap_io.read(annotation_file_path)
        else:
            atlas = self.atlas
        atlas = self.convert_label(atlas, key='id', value='order', method='map')  # FIXME: why value='order'?
        # apply color map
        cm = self.color_map(alpha=False, as_int=True)
        atlas = cm[atlas]
        if dest_path:
            clearmap_io.write(dest_path, atlas)
        return atlas


def get_atlas_filepath(filename, directory=None, orientation=None, slicing=None):
    """Formats the annotation filename given orientation and slicing."""

    def substitute_chars(in_str, chars_to_strip=' ',
                           chars_to_substitute='(,)'):  # TODO: check if we add '[]' to chars_to_substitute
        for c in chars_to_strip:
            in_str = in_str.replace(c, '')
        for c in chars_to_substitute:
            in_str = in_str.replace(c, '_')
        return in_str

    x, y, z = format_orientation(orientation, default=(1, 2, 3))
    postfix = substitute_chars(f'{x}_{y}_{z}_{slicing}')

    base, ext = os.path.splitext(filename)
    fn = base + f'_{postfix}{ext}'

    if directory is not None:
        fn = os.path.join(directory, os.path.basename(filename))

    return fn


def annotation_to_distance_file(annotation_file_path):
    brain_mask = (clearmap_io.read(annotation_file_path) > 0).astype(int)
    distance_array = distance_transform_edt(brain_mask)
    return distance_array


###############################################################################
# ## Tests
###############################################################################


def _test1():
    import numpy as np
    import ClearMap.Alignment.Annotation as ano
    from importlib import reload
    reload(ano)
    annotator = ano.Annotation(ATLAS_BASE_NAME, None, None, default_label_file)

    points = np.array([[162, 200, 138], [246, 486, 138], [246, 486, 138]])

    label = ano.label_points(points)
    print(label)

    cnts = annotator.count_points(points)
    print(cnts)

    cnts = annotator.count_points(points, hierarchical=False)
    print(cnts)

    import ClearMap.IO.IO as io
    ano.write_color_annotation('test.tif')
    io.delete_file('test.tif')

    l = annotator.find(247, key='id')
    print(l)
    l.info(with_children=True)
    print(l.level)

    annotator.get_dictionary(key='id', value='acronym', with_parents=True, min_level=3, max_depth=3)


def _test2():
    import ClearMap.Alignment.Annotation as ano
    from importlib import reload
    reload(ano)
    annotation = ano.Annotation(ATLAS_BASE_NAME, None, None,default_label_file)

    assert annotation.df.shape == (1319, 5)
    assert annotation.dict_id_to_acronym[1] == "TMv"
    assert annotation.dict_name_to_id['Interpeduncular nucleus'] == 100
    assert annotation.dict_id_to_name[1000] == 'extrapyramidal fiber systems'
    assert annotation.dict_acronym_to_id['MO'] == 500
    assert annotation.dict_id_to_color[200] == '61E7B7'

    ### annotation_new contains last annotation atlas (2017) and last annotation graph (from Allen, October 2022
    label_fpath = os.path.join(ClearMap.Settings.atlas_folder, "ABA_annotation_last.json")
    annotation_new = Annotation('ABA_25um_2017', None, None, label_source=label_fpath)

    # assert annotation_new.df.shape == (1336, 5)
    # assert annotation_new.dict_id_to_acronym[1] == "TMv"
    # assert annotation_new.dict_name_to_id['Interpeduncular nucleus'] == 100
    # assert annotation_new.dict_id_to_name[1000] == 'extrapyramidal fiber systems'
    # assert annotation_new.dict_acronym_to_id['MO'] == 500
    # assert annotation_new.dict_id_to_color[200] == '61E7B7'
    #
    # assert annotation_new.children_df.shape == (1327, 9)
    # assert annotation_new.children_df.set_index('id').loc[100, 'structure_path'] == [997, 8, 343, 313, 348, 165, 100]
    # assert annotation_new.children_df.set_index('id').loc[997, 'direct_children_structures_ids'] == [8, 1009, 73, 1024, 304325711]
    # assert annotation_new.children_df.set_index('id').loc[65, 'all_children_structures_ids'] == []
    #
    # assert annotation_new.get_dict_children_to_parents([1032]) == {1055: 1032, 1063: 1032, 1071: 1032, 1078: 1032}
    # assert annotation_new.get_dict_parents_to_children([1032]) == {1032: [1055, 1063, 1071, 1078]}
    # assert (annotation_new.enrich_df(pd.DataFrame([{"id": 1111}])).equals(
    #     pd.DataFrame([{'id': 1111, 'name': 'Primary somatosensory area, trunk, layer 5',
    #                    'acronym': 'SSp-tr5'}])))

