# -*- coding: utf-8 -*-
"""
ParameterTools

Provides formatting tools to handle / print parameter dictionaries
organized as key:value pairs.

Hierarchical parameter structures are supported
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'


delimiter = '_';

def getParameter(parameter, key, default = None):
    """Gets a parameter from a dict, returns default value if not defined
    
    Arguments:
      parameter : dict
        parameter dictionary Copyright 2015 by Christoph Kirst, The Rockefeller University, New York City
      key : object
        key (dot separated hierarchical naming allowed)
      default : object
        default return value if parameter not defined
    
    Returns:
      object
        parameter value for key
    """

    if not isinstance(parameter, dict):
      return default;

    if not isinstance(key, str):
      return parameter.get(key, default);

    p = parameter;    
    for k in key.split(delimiter):
      if k in p.keys():
        p = p[k];
      else:
        return default;
    
    return p;

    
def setParameter(parameter, key = None, value = None, **args):
    """Sets a parameter in a dict
    
    Arguments:
      parameter : dict
        parameter dictionary
      key : object
        key (delimiter separated hierarchical naming allowed)
      value : object
        value to set
      **args
        key : value pairs
    
    Returns:
      dict 
        parameter dictionary
    """
    if key is None or value is None:
      keys = args.keys();
      values = args.values();
    else:
      keys = [key];
      values = [value];
    
    
    for k,v in zip(keys, values):
      if not isinstance(k, str):
        parameter[k] = v;
      else:
        p = parameter;
        ks = k.split(delimiter);
        for l in ks[:-1]:
          if isinstance(p, dict):
            if l in p.keys():
              p = p[l];
            else:
              p[l] = {};
              p = p[l];
          else:
            raise RuntimeError("setParameter: %s is not a dictionary!" % k);
        
        p[ks[-1]] = v;
    
    return parameter;


def writeParameter(parameter = None, head = None, **args):
    """Writes parameter settings in a formatted way
    
    Arguments:
      head : str or None
        prefix of each line
      **args
        the parameter values as key=value arguments
    
    Returns:
      str
        a formated string with parameter info
    """
    if head is None:
      head = '';
    else:
      head = head + ' ';
    
    if parameter is None:
      parameter = args;
        
    keys = parameter.keys();
    vals = parameter.values();
    parsize = max([len(x) for x in keys]);
    
    s = [];
    for k,v in zip(keys, vals):
      if isinstance(v, dict):
        s.append(head + k.ljust(parsize) + ': dict')
        s.append(writeParameter(v, head = ' ' * (len(head) + parsize) + ' '));
      else:
        s.append(head + k.ljust(parsize) + ': ' + str(v)); 
    
    return '\n'.join(s)


def printParameter(parameter = None, head = None, **args):
    """Prints parameter settings in a formatted way
    
    Arguments:
      head : str or None
        prefix of each line
      **args
        the parameter values as key=value arguments
    """
    print(writeParameter(parameter = parameter, head = head, **args));

    
def joinParameter(*args):
    """Joins dictionaries in a consitent way
    
    For multiple occurences of a key the value is defined by the last key : value pair.
    
    Arguments:
      *args
        list of parameter dictonaries
    
    Returns:
      dict
        the joined dictionary
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
    
    
def prependParameter(parameter, name):
    """Adds a hierarchical name infront of all the parameter keys in the passed dictionary

    Arguments:
      parameter : dict
        parameter dictonary
      name : str
        name to add infronat of all keys
    
    Returns:
      dict
        the dictionary with modified keys
    """
    
    keys   = parameter.keys() 
    values = parameter.values();
    keys = [name + '.' + k for k in keys];
    
    return {keys[i] : values[i] for i in range(len(keys))}
        


if __name__ == "__main__":
  import ClearMap.Utils.ParameterTools as par

  d = dict(x = 10, y = 100);

  par.setParameter(d, q_r = 2, q_p = 'hello')

  print (d)