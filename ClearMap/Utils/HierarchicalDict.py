# -*- coding: utf-8 -*-
"""
HierarchicalDict
================

Provides tools to handle / print hierarchical parameter dictionaries.

Example
-------

>>> import ClearMap.Utils.HierarchicalDict as hdict
>>> d = dict(x = 10, y = 100, z = dict(a = 10, b = 20))
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
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'


from collections import OrderedDict as odict


DELIMITER = '_'
# TODO: Could be handled more professionally by Enum Flag to allow for multiple parameter flags if needed
# TODO: use a HierarchicalDict class -> derive ParameterDict for ClearMap use in GUI


def get(parameter, key, default=None):
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
        return default

    if not isinstance(key, str):
        return parameter.get(key, default)

    p = parameter
    for k in key.split(DELIMITER):
        if k in p.keys():
            p = p[k]
        else:
            return default

    return p


def set(parameter, key=None, value=None, **kwargs):
    """Sets a parameter in a hierarchical dictionary.
    
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
        keys = kwargs.keys()
        values = kwargs.values()
    else:
        keys = [key]
        values = [value]

    for k, v in zip(keys, values):
        if not isinstance(k, str):
            parameter[k] = v
        else:
            p = parameter
            ks = k.split(DELIMITER)
            for l in ks[:-1]:
                if isinstance(p, dict):
                    if l in p.keys():
                        p = p[l]
                    else:
                        p[l] = {}
                        p = p[l]
                else:
                    raise RuntimeError(f"set: {k} is not a dictionary!")

            p[ks[-1]] = v

    return parameter


def write(parameter=None, head=None, **kwargs):
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
      A formatted string with parameter info.
    """
    head = head or ''

    if head:
        head += ' '

    if parameter is None:
        parameter = odict()

    parameter = join(parameter, kwargs)

    par_size = max([len(x) for x in parameter.keys()])

    s = []
    for k, v in parameter.items():
        if isinstance(v, dict):
            s.append(f'{head}{k.ljust(par_size)}: dict')
            s.append(write(v, head=' ' * (len(head) + par_size) + ' '))
        else:
            s.append(f'{head}{k.ljust(par_size)}: {v}')

    return '\n'.join(s)


def pprint(parameter=None, head=None, **args):
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
    print(write(parameter=parameter, head=head, **args))


def join(*args):
    """Joins dictionaries in a consistent way
    
    Arguments
    ---------
    args : dicts
      The parameter dictionaries to join.
    
    Returns
    -------
    join : dict
        The joined dictionary.
    """
    new = args[0]
    for add in args[1:]:
        for k, v in add.items():
            new[k] = v

    return new


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
    """Adds a hierarchical key in front of all the parameter keys in a dictionary.

    Arguments
    ---------
    parameter : dict
      Parameter dictionary.
    key : str
      Key to add in front of the dictionary keys.
    
    Returns
    -------
    prepend : dict
      New dictionary with modified keys.
    """
    return {f'{key}{DELIMITER}{k}': v for k, v in parameter.items()}
    
    keys   = parameter.keys() 
    values = parameter.values();
    keys = [key + '.' + k for k in keys];
    
    return {k : v for k,v in zip(keys, values)}
        

###############################################################################
# Tests
###############################################################################

def _test():
    import ClearMap.Utils.HierarchicalDict as hdict
  
    d = dict(x=10, y=100, z=dict(a=10, b=20))
    print(hdict.get(d, 'z_a'))
    hdict.set(d, 'z_c_q', 42)
    hdict.pprint(d)
