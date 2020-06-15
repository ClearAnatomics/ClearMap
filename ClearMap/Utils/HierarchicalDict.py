# -*- coding: utf-8 -*-
"""
HierarchicalDict
================

Provides tools to handle / print hierarchical parameter dictionaries.

Example
-------

>>> import ClearMap.Utils.HierarchicalDict as hdict
>>> d = dict(x = 10, y = 100, z = dict(a = 10, b = 20));
>>> print(hdict.get(d, 'z_a'))
10

>>>  hdict.set(d, 'z_c_q', 42)
>>> hdict.pprint(d)
x: 10
y: 100
z: dict
   a: 10
   b: 20
   c: dict
      q: 42


"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


DELIMITER = '_';

from collections import OrderedDict as odict

#class HierarchicalDict(dict):
##  pass

def get(parameter, key, default = None):
    """Gets a parameter from a dict, returns default value if not defined
    
    Arguments
    ---------
    parameter : dict
      Parameter dictionary.
    key : object
      Parameter key
    default : object
      Default value if parameter not defined.
    
    Returns
    -------
    value : object
      Parameter value for key.
    """

    if not isinstance(parameter, dict):
      return default;

    if not isinstance(key, str):
      return parameter.get(key, default);

    p = parameter;    
    for k in key.split(DELIMITER):
      if k in p.keys():
        p = p[k];
      else:
        return default;
    
    return p;

    
def set(parameter, key = None, value = None, **kwargs):
    """Sets a parameter in a hierarchical dicitionary.
    
    Arguments
    ---------
    parameter : dict
      Parameter dictionary.
    key : object
      Key in dictionary.
    value : object
      Value to set.
    kwargs
      Key : value pairs.
    
    Returns
    -------
    parameter : dict 
      Parameter dictionary.
    """
    if key is None or value is None:
      keys = kwargs.keys();
      values = kwargs.values();
    else:
      keys = [key];
      values = [value];
    
    for k,v in zip(keys, values):
      if not isinstance(k, str):
        parameter[k] = v;
      else:
        p = parameter;
        ks = k.split(DELIMITER);
        for l in ks[:-1]:
          if isinstance(p, dict):
            if l in p.keys():
              p = p[l];
            else:
              p[l] = {};
              p = p[l];
          else:
            raise RuntimeError("set: %s is not a dictionary!" % k);
        
        p[ks[-1]] = v;
    
    return parameter;


def write(parameter = None, head = None, **kwargs):
    """Writes parameter settings in a formatted way.
    
    Arguments
    ---------
    parameter : dict
      Parameter dictionary.
    head : str or None
      Optional prefix of each line.
    kwargs
      Additional parameter values as key=value arguments.
    
    Returns
    -------
    string : str
      A formated string with parameter info.
    """
    if head is None:
      head = '';
    elif len(head) > 0:
      head = head + ' ';
    
    if parameter is None:
      parameter = odict();
    
    parameter = join(parameter, kwargs);
        
    keys = parameter.keys();
    vals = parameter.values();
    parsize = max([len(x) for x in keys]);
    
    s = [];
    for k,v in zip(keys, vals):
      if isinstance(v, dict):
        s.append(head + k.ljust(parsize) + ': dict')
        s.append(write(v, head = ' ' * (len(head) + parsize) + ' '));
      else:
        s.append(head + k.ljust(parsize) + ': ' + str(v)); 
    
    return '\n'.join(s)


def pprint(parameter = None, head = None, **args):
    """Prints parameter settings in a formatted way.
    
    Arguments
    ---------
    parameter : dict
      Parameter dictionary.
    head : str or None
      prefix of each line
    args
      Additional parameter values as key=value arguments.
    """
    print(write(parameter = parameter, head = head, **args));

    
def join(*args):
    """Joins dictionaries in a consitent way
    
    Arguments
    ---------
    args : dicts
      The parameter dictonaries to join.
    
    Returns
    -------
    join : dict
        The joined dictionary.
    """
    new = args[0];    
    for add in args[1:]:
      for k,v in add.items():
        new[k] = v;
        
    return new;
   
#    keyList = [x.keys() for x in args];
#    n = len(args);
#    
#    keys = [];
#    values = [];
#    for i in range(n):
#        values = values + [args[i][k] for k in keyList[i] if k not in keys];
#        keys   = keys + [k for k in keyList[i] if k not in keys];
#    
#    return {keys[i] : values[i] for i in range(len(keys))}
    
    
def prepend(parameter, key):
    """Adds a hierarchical key infront of all the parameter keys in a dictionary.

    Arguments
    ---------
    parameter : dict
      Parameter dictonary.
    key : str
      Key to add infronat of the dictionary keys.
    
    Returns
    -------
    prepend : dict
      The dictionary with modified keys.
    """
    
    keys   = parameter.keys() 
    values = parameter.values();
    keys = [key + '.' + k for k in keys];
    
    return {k : v for k,v in zip(keys, values)}
        

###############################################################################
### Tests
###############################################################################

def _test():
  import ClearMap.Utils.HierarchicalDict as hdict
  

  d = dict(x = 10, y = 100, z = dict(a = 10, b = 20));
  print(hdict.get(d, 'z_a'))
  hdict.set(d, 'z_c_q', 42)
  hdict.pprint(d)
