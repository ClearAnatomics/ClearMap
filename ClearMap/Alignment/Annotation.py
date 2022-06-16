# -*- coding: utf-8 -*-
"""
Annotation
==========

Module to handle label and annotation info from Allen Brain Atlas (v2).
 
Notes
-----      
  - The files containting annotation information is assumed to be in the path 
    stored in :const:`atlas_path`, which defaults to the resource path 
    :const:`ClearMap.Settings.resources_path`.
    
  - The annotation file is assumed to be 'annotation_25_full.nrrd'
    but can be set via :const:`default_annotation_file`.

  - The mapping between labels and brain area information is assumed to be in 
    the 'annotation.json' file but the location can be set
    via :const:`default_label_file`.     
      
References
----------
  - `Allen Brain Atlas <http://mouse.brain-map.org/static/atlas>`_  
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import os
import collections 

import numpy as np

import json

import ClearMap.Settings as settings

import ClearMap.IO.IO as io
import ClearMap.IO.FileUtils as fu

import ClearMap.Alignment.Resampling as res

import ClearMap.Utils.HierarchicalDict as hdict

import ClearMap.Visualization.Color as col


###############################################################################
### Atlas Structures
###############################################################################

#TODO: move to settings ?
atlas_path = os.path.join(settings.resources_path, 'Atlas')
"""Default path to atlas infomration.
"""

default_annotation_file = os.path.join(atlas_path, 'ABA_25um_annotation.tif')
"""Default volumetric annotated image file.

Note
----
  This file is by default the Allen brain annotated mouse atlas with 25um 
  isotropic resolution.
"""
fu.uncompress(default_annotation_file)


default_hemispheres_file = os.path.join(atlas_path, 'ABA_25um_hemispheres.tif')
fu.uncompress(default_hemispheres_file)

default_reference_file = os.path.join(atlas_path, 'ABA_25um_reference.tif')
"""Default volumetric annotated image file.

Note
----
  This file is by default the Allen brain annotated mouse atlas with 25um 
  isotropic resolution.
"""
fu.uncompress(default_reference_file)


default_distance_to_surface_file = os.path.join(atlas_path, 'ABA_25um_distance_to_surface.tif')
"""Default volumetric annotated image file.

Note
----
  This file is by default the Allen brain annotated mouse atlas with 25um 
  isotropic resolution.
"""
fu.uncompress(default_distance_to_surface_file)


default_label_file = os.path.join(atlas_path, 'ABA_annotation.json')
"""Default list of labels and region names in the annotated image.

Note
----
  This file is by default the labels for the Allen brain annotated mouse 
  atlas with 25um isotropic resolution.
"""

default_extra_label = [(182305696, 453, 'No label', 'NoL'), 
               (182305712, 453, 'No label', 'NoL'), 
               (312782560, 315, 'No label', 'NoL'), 
               (312782592, 453, 'No label', 'NoL'), 
               (312782656, 315, 'No label', 'NoL'),
               (526157184, 993, 'No label', 'NoL'),
               (526322272, 500, 'No label', 'NoL'),
               (527696992, 315, 'No label', 'NoL')];
"""Additional label not in the Allen Brain Atlas label but in the atlas image.

Note
----
  The form is a list of tuples, each tuple has the form 
  (atlas id, parent id, name, acronym).
"""





class Label(object):
  """Class holding information of an individual Atlas label."""
  
  def __init__(self, data, children = None, parent = None, level = 0):
    self.data = data;
    self.children = children;
    self.parent = parent;
    self.data['level'] = level;
  
  @property
  def name(self):
    return self.data['name'];
  
  @property  
  def id(self):
    return self.data['id'];
    
  @property
  def level(self):
    return self.data['level'];
  
  @property
  def order(self):
    return self.data['order'];
  
  def color(self, *args, **kwargs):
    return col.color(self.data['rgb'], **kwargs);
  
  def __getitem__(self, key):
    return self.data[key];
    
  def __setitem__(self, key, value):
    self.data[key] = value;
  
  def write(self, with_children=True, ident=None):
    return self.__str__(ident=ident, with_children=with_children);
  
  def info(self, with_children=True, ident=None):
    print(self.write(ident=ident, with_children=with_children));
  
  def __str__(self, ident = None, with_children = False):
    if ident is None:
      ident = '';
    
    s = '';      
    if with_children and isinstance(self.children, list):
      for c in self.children:
        s = s + '\n' + c.__str__(ident=ident + '  ');

    h = ident + self.name + '\n' + ident + '=' * len(self.name) + '\n';
   
    s = h + hdict.write(self.data, head=ident[:-1]) + '\n' + s;
    
    return s;
  
  def __repr__(self, ident = None, with_children = False):
    return self.__str__(ident=ident, with_children=False);



class Annotation(object):
  """Class that holds information of the annotated regions."""    
     
  def __init__(self, label_file = None, extra_label = None, annotation_file = None):  
    """Initialization
    
    Arguments
    ---------
    label_file : str
      File with label infomration in json format.
    """
    self.initialize(label_file=label_file, extra_label=extra_label, annotation_file=annotation_file);
  
  
  def initialize(self, label_file = None, extra_label = None, annotation_file = None):  
    # read json file 
    if label_file is None:
      label_file = default_label_file;
    if annotation_file is None:
      annotation_file = default_annotation_file;
    if extra_label is None:
      extra_label = default_extra_label;     
    if extra_label in ['None', '', False]:   #add nodes for missing labels
      extra_label = [];
     
    self.label_file = label_file;
    self.annotation_file = annotation_file;
    self.extra_label = extra_label;
    
    #initialze label tree    
    with open(label_file) as dfile:
      aba_json = dfile.read();
      dfile.close();
    aba = json.loads(aba_json)

    root = aba['msg'][0];
    self.root = self.initialze_tree(root);

    #maxgraph = max(self.get_list('graph_order'));
    for a in extra_label:
      i, p, n, l = a;
      node = self.find(p, key = 'id');
      data = node.data.copy();
      data['id'] = i;
      data['name'] = n;
      data['acronym'] = l;
      data['parent_structure_id'] = p;
      data['graph_order'] = -1;
      data['atlas_id'] = -1;
      node.children.append(Label(data, parent = node, children = [], level = node.level+1));
    
    #initialize generic id      
    self.add_data('order', range(self.n_structures));      
    
    #initialize color label
    self.colors_hex = self.get_list('color_hex_triplet');
    self.colors_rgb = np.array([col.hex_to_rgb(c) for c in self.colors_hex]);
    self.add_data('rgb', self.colors_rgb );
    self.add_data('color_order', range(self.n_structures));  
    
    #initialize lookup tables
    self.structures = self.get_list(None);
    self.ids        = self.get_list('id');
    self.labels     = self.get_list('atlas_id');
    self.names      = self.get_list('name');
    self.acronyms   = self.get_list('acronym');
  
  
  def initialze_tree(self, root, parent = None, level = 0):
    label = Label(  { key : root[key] for key in root.keys() if key not in ["children"]}, parent = parent, level = level);
    label.children = [self.initialze_tree(c, parent = label, level = level + 1) for c in root['children']];
    return label;
  
  
  def get_list(self, key = None, node = None, level = None):
    if node is None:
      node = self.root;
    
    l = [];
    
    if level is not None:
      n = node;
      while n.level > level:
        n = n.parent;
    else:
      n = node;
    
    if key is not None:
      l.append(n[key])
    else:
      l.append(n);
    
    for c in node.children:
      l.extend(self.get_list(node = c, key = key, level = level));
    
    return l;
  
  @property
  def n_structures(self):
    return len(self.get_list());
  
  
  @property
  def max_level(self):
    return np.max(self.get_list('level'));
  
  
  def get_hierarchical_dictionary(self, node = None):
    if node is None:
      node = self.root;
    
    d = node.data;
    d['children'] = [self.get_hierarchical_dictionary(node=n) for n in node.children];
 
    return d;
  
  
  def get_dictionary(self, key, value, node = None, level = None, ordered = False):
    if node is None:
      node = self.root;
    
    keys = self.get_list(key = key, node = node, level = None);
    vals = self.get_list(key = value, node = node, level = level);
    
    if ordered:
      d = collections.OrderedDict
    d = { k : v for k,v in zip(keys, vals)};
    
    return d;
    
  
  def get_map(self, key, value, node = None, level = None):
    d = self.get_dictionary(key = key, value = value, node = node, level = level);
    
    n = max(d.keys());
    m = np.zeros(n+1, dtype = int);
    m[list(d.keys())] = list(d.values());
    
    return m;
  
  
  def add_data(self, name, data):
    nodes = self.get_list();
    for n,d in zip(nodes, data):
      n.data[name] = d;
  
  
  def convert_label(self, label, key = 'order', value  = 'graph_order', node = None, level = None, method = 'map'):
    if method in ['map']:
      m = self.get_map(key = key, value = value, node = node, level = level);
      return m[label];       
    else:
      d = self.get_dictionary(key = key, value = value, node = node, level = level);
      return np.vectorize(d.__getitem__, otypes=[type(d[list(d.keys())[0]])])(label);
 
  
  def label_to_color(self, label, key = 'order', level = None, alpha = True, as_int = False, int_type = 'uint8'):
      cm = self.colors_rgb;
      cm = col.color(cm, alpha=alpha,as_int=as_int, int_type=int_type);
      if key != 'order' or level is not None:
        label = self.convert_label(label, key=key, value='order', level=level);
      return cm[label];
  
  def find(self, label, key = 'id', value = None, node = None, level = None):
    d = self.get_dictionary(key = key, value = value, node = node, level = level);
    if isinstance(label, list):
      return [d[l] for l in label];
    else:
      return d[label];
  
  
  def parents(self, label, key = 'id', value = None):
    d = self.get_dictionary(key = key, value = None, node = None, level = None);
    p = d[label];
    l = [p];
    while p.level > 0:
      p = p.parent;
      l.append(p)
    
    if value is None:
      return l;
    else:
      return [ll[value] for ll in l];
   
  
  def common_parent(self, label, key = 'id', value = None):
    parents = [self.parents(l, key = key, value = key) for l in label];
    
    n = min([len(p) for p in parents]);
    p0 = parents[0];
    level = -1;
    while -level-1 < n and np.all([p[level] == p0[level] for p in parents]):
      level -= 1;
    
    return self.find(p0[level+1], key = key, value = value);
   
    
  def __str__(self):
    return 'Annotation(%d)[%d]{%s}' % (self.n_structures, self.max_level, self.label_file);
  
  def __repr__(self):
    return self.__str__();
  
 

annotation = Annotation();
"""Information on the annotated regions"""

n_structures    = annotation.n_structures;
get_dictionary  = annotation.get_dictionary;
get_list        = annotation.get_list;
get_map         = annotation.get_map;
find            = annotation.find;


def initialize(label_file=None, extra_label = None, annotation_file = None):
  global annotation, n_structures, get_dictionary, get_list, get_map, find
  annotation = Annotation(label_file=label_file, extra_label=extra_label, annotation_file=annotation_file);
  
  n_structures    = annotation.n_structures;
  get_dictionary  = annotation.get_dictionary;
  get_list        = annotation.get_list;
  get_map         = annotation.get_map;
  find            = annotation.find;


def set_annotation_file(annotation_file):
  initialize(annotation_file=annotation_file, label_file=annotation.label_file, extra_label=annotation.extra_label);

def set_label_file(label_file, extra_label = None):
  initialize(annotation_file = annotation.annotation_file, label_file=label_file, extra_label=extra_label);


###############################################################################
### Labeling
###############################################################################

#TODO:use parallel arryay processing and lut routines to speed up? 
def label_points(points, annotation_file = None, invalid = 0, key = 'order', level = None):
  """Label points according to the annotation in the labeled image file.
  
  Arguments
  ---------
  points : array 
    Array of nxdim point coordinates to be labeled.
  annotation_file : str
    File name of the atals annotation.
  invalid : int
    Label for invalid points.
  key : str
    The key of the label, by default the order of the labels.
  
  Returns
  -------
  label : array
    Label of the points corresponding to the given key.
  """
  n_points = points.shape[0];  
  n_dim = points.shape[1];
  
  if annotation_file is None:
    annotation_file = annotation.annotation_file;
  
  atlas = io.read(annotation_file);   
  atlas = np.array(atlas, dtype=int);
  atlas_shape = atlas.shape;
 
  label = np.full(n_points, invalid, dtype=int);
  
  points_int = np.asarray(points, dtype=int)
  for d in range(n_dim):
    if d == 0:
      valid = np.logical_and(points_int[:,d] >= 0, points_int[:,d] < atlas_shape[d]);
    else:
      valid = np.logical_and(valid, np.logical_and(points_int[:,d] >= 0, points_int[:,d] < atlas_shape[d]));
  
  indices = [points_int[valid,d] for d in range(n_dim)];
  label[valid] = atlas[indices];
  
  if key != 'id' or level is not None:
    label[valid] = convert_label(label[valid], key='id', value=key, level=level);
  
  return label;


def convert_label(label, key = 'id', value = 'order', level = None, method = None):
  """Convert label using the atlas annotation data.
  
  Arguments
  ---------
  lable : array
    List of labels to convert.
  key : str
    The key corresponding to the label.
  value : str
    The key to ocnvert the lable to.
  level : nt or None
    Convert at this level of the hierarchy. I fNone use full hierarchy.
  method : 'map' or 'dictionary'
    Convert labels using a mapping array or a dictionary. Depending on the keys 
    either can be faster for large data sets.
    
  Returns
  -------
  label : array
    List of converted labels.
  """
  if value in ['rgb', 'rgba', 'RGB', 'RGBA']:
    alpha = value in ['rgba', 'RGBA'];
    as_int = value[:3] == 'RGB';
    return annotation.label_to_color(label, key=key, alpha=alpha, as_int=as_int)
  return annotation.convert_label(label, key=key, value=value, level=level, method=method);


def convert_label_to_color(label, key = 'id', level = None, alpha = True, as_int = False, int_type = 'uint8'):
  """Convert label using the atlas annotation colors.
  
  Arguments
  ---------
  lable : array
    List of labels to convert.
  key : str
    The key corresponding to the label.
  level : nt or None
    Convert at this level of the hierarchy. I fNone use full hierarchy.
  alpha : bool
    If True, return rgba values.
  as_int : bool
    If True, return rgb valueas as int between 0 and 255.
  int_type : dtype
    Type for the color array.
    
  Returns
  -------
  colors : array
    List of colors for each label.
  """
  return annotation.label_to_color(label, key=key, level=level, alpha=alpha, as_int=as_int, int_type=int_type);


def count_label(label, weights = None, key = 'order', hierarchical = True):
  """Counts the label within the various structures, taking into account the sub-structures.
  
  Arguments
  ---------
  label : array
    List of labels.
  weights : array 
    Optional list of weights for each label.
  key : str
    The key the lables are given in.
  hierarchical : bool
    If True, count all sub-labels.
    
  Returns
  -------
  bins : array
    The counts for each label.
  """
  if key != 'order':
    label = convert_label(label, key=key, value='order', invalid=0);
  
  bins = np.bincount(label, weights=weights, minlength=n_structures);
  
  if hierarchical:
    _recursive_count(annotation.root, bins);
  
  return bins;


def _recursive_count(node, counts):
  """Helper for label counting."""
  n = 0;
  for c in node.children:
    n += _recursive_count(c, counts);
  counts[node['order']] += n;
  return counts[node['order']];


def count_points(points, weights = None, annotation_file = None, invalid = 0, hierarchical = True):
  """Counts the points within the various structures, taking into account the sub-structures.
  
  Arguments
  ---------
  points : array
    Array of nxdim points to annotate and count.
  weights : array or None
    Optional intensity values to weight the count.
  annotation_file : str
    File name of the atlas annotation.
  invalid : int
    Label for invalid points.
  hierarchical : bool
    If True, count all sub-labels.
  
  Returns
  -------
  bins : array
    The counts for each label.
  """
  label = label_points(points, annotation_file=annotation_file, invalid=invalid, key='order');
  return count_label(label, weights=weights, key='order', hierarchical=hierarchical);


###############################################################################
### Color maps
###############################################################################

def color_map(alpha = True, as_int = False, int_type = 'uint8'):
  """Generates a color map from color ids to rgb
  
  Arguments
  ---------
  alpha : bool
    If True return a color map with alpha values.
  
  Returns
  -------
  color_map : array
    An array of rgb colors for each label.
  """
  cm = annotation.colors_rgb;  
  return col.color(cm, alpha=alpha,as_int=as_int, int_type=int_type);
 

def write_color_palette(filename = None):
  """Creates a pal or lut file for imaris or imagej based on label colors of atlas.
  
  Arguments
  ---------
  filename : str
    The name of the color palette file.
  
  Returns
  -------
  filename : str
    The name of the file to which the color palette was written.
  """
  
  cm = color_map(alpha=False,as_int=True);
 
  fext = io.file_extension(filename);
  if fext == 'pal':
    col.write_PAL(filename, cm);
  elif fext == 'lut':
    col.write_LUT(filename, cm);
  else:
    raise RuntimeError('color pallete format: %s not lut or pal' % fext);
      
  return filename;


def write_color_annotation(filename, annotation_file = None):
  """Creates a rgb image from the atlas color data.
  
  Arguments
  ---------
  filename : str
    The name of the color palette file.
  annotation_file : str
    File name of the atals annotation. 
  
  Returns
  -------
  filename : str
    The name of the file to which the color atlas was written.
  """
  #load atlas and convert to order
  if annotation_file is None:
    annotation_file = annotation.annotation_file;
  atlas = np.array(io.read(annotation_file), dtype = int);   
  atlas = convert_label(atlas, key='id', value='order', method='map');
  
  #apply color map
  cm = color_map(alpha=False, as_int=True);
  atlas = cm[atlas];
  
  return io.write(filename, atlas);


###############################################################################
### Labeling
###############################################################################

def prepare_annotation_files(slicing=None, orientation=None, directory=None, postfix=None, annotation_file=None,
                             hemispheres_file=None, reference_file=None, distance_to_surface_file=None,
                             hemispheres=False, overwrite=False, verbose=False):
  """Crop the annotation, reference and distance files to match the data.
  
  Arguments
  ---------
  hemispheres
  slicing : tuple or None
    The slice specification after reorienting.
  orientation : tuple, str or None.
    The orientation specification. Strings can be 'left' or 'right', for the
    two hemispheres.
  directory : str or None
    The target directory. If None, use ClearMap resources folder.
  postfix : str or None
    Use this postfix for the cropped annotation file. If None and automatic 
    label is chosen.
  annotation_file : str or None
    The annotation file to use.
  reference_file : str or None
    The reference file to use.
  distance_to_surface_file : str or None
    The distance file to use.
  overwrite : bool
    If True, overwrite exisitng files.
    
  Returns
  -------
  annotation_file : str
    The cropped annotation file.
  reference_file : str
    The cropped reference file.
  distance_to_surface_file : str
    The distance cropped file.
  """
  if annotation_file is None:
    annotation_file = default_annotation_file
  if hemispheres_file is None:
    hemispheres_file = default_hemispheres_file
  if reference_file is None:
    reference_file = default_reference_file
  if distance_to_surface_file is None:
    distance_to_surface_file = default_distance_to_surface_file

  files = [annotation_file, reference_file, distance_to_surface_file]
  if hemispheres:
    files.insert(1, hemispheres_file)
  
  results = []
  for f_path in files:
    if f_path is not None:
      fn = format_annotation_filename(f_path, orientation=orientation, slicing=slicing, postfix=postfix, directory=directory)
      if verbose:
        print('Preparing: %r' % fn)
      
      if not overwrite and io.is_file(fn):
        results.append(fn)
        continue

      if not io.is_file(f_path):
        raise ValueError('Cannot find annotation file: %s' % f_path)
      
      s = io.as_source(f_path)
      if verbose:
        print('Preparing: from source %r' % s)
      
      data = np.array(s.array)
      
      if orientation is not None:
        # permute
        per = res.orientation_to_permuation(orientation)
        data = data.transpose(per)

        # reverse axes
        reslice = False
        sl = [slice(None)] * data.ndim
        for d, o in enumerate(orientation):
          if o < 0:
            sl[d] = slice(None, None, -1)
            reslice = True
        if reslice:
          data = data[tuple(sl)]
      
      if slicing is not None:
        data = data[slicing]
      io.write(fn, data)
      results.append(fn)
    else:
      results.append(None)
      
  return results


def substitute_chars(s):
  chars_to_strip = ' '
  chars_to_substitute = '(,)'  # TODO: check if we add '[]'
  for c in chars_to_strip:
    s = s.replace(c, '')
  for c in chars_to_substitute:
    s = s.replace(c, '_')
  return s
  

def format_annotation_filename(filename, orientation=None, slicing=None, postfix=None, directory=None):
  """Formats the annotation filename given orientation and slicing."""
  
  if postfix is None:
    orientation = res.format_orientation(orientation, default=(1, 2, 3))
    x, y, z = orientation
    postfix = f'{x}_{y}_{z}_{slicing}'
    postfix = substitute_chars(postfix)

  if postfix:
    base, ext = os.path.splitext(filename)
    fn = base + f'_{postfix}{ext}'
  else:
    fn = filename
  if directory is not None:
    fn = os.path.join(directory, os.path.basename(filename))
  
  return fn


###############################################################################
### Tests
###############################################################################

def _test():
  import numpy as np
  import ClearMap.Alignment.Annotation as ano
  #reload(ano)

  points = np.array([[162, 200, 138], [246, 486, 138], [246, 486, 138]]);
  
  label = ano.label_points(points);
  print(label);
  
  cnts = ano.count_points(points);
  print(cnts);

  cnts = ano.count_points(points, hierarchical=False);
  print(cnts);

  import ClearMap.IO.IO as io
  ano.write_color_annotation('test.tif')
  io.delete_file('test.tif')


  l = ano.find(247, key='id')
  print(l)
  l.info(with_children=True)
  print(l.level)


