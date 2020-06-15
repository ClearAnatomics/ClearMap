# -*- coding: utf-8 -*-
"""
Parameter
=========

Provides GUI widget to parameter dictionaries used in *ClearMap* processing
 routines.

The module effectively implements and interface between hierarchical
dictionaries and the parameter trees of pyqtgraph.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import pyqtgraph
from pyqtgraph import QtGui

import ClearMap.GUI.ParameterItems
from ClearMap.GUI.ParameterItems import filestr, dirstr

import ClearMap.Utils.ParameterTools as pt


typeToType = {'int' : 'int', 'float': 'float', 'bool' : 'bool', 
              'str' : 'str', 'list' : 'list' , 'tuple': 'tuple', 
              'ndarray' : 'ndarray',
              'option' : 'dropdown', 
              'filestr' : 'filestr', 'dirstr' : 'dirstr', 
              'dict' : 'group'};

def valueToType(value):
  """Determines the type of value to use in Parameter
  
  Arguments:
    value (any): parameter value
  
  Returns:
    type (str): the type string to use in Parameter
  """    
  return typeToType.get(value.__class__.__name__, 'any');


def dictToTree(parameter, title = None):
  """Convert a parameter dictionary to a parameter tree dictionary
  
  Arguments:
    parameter (dict): (nested) dictionary of parameter
  
  Returns:
    dict : parameter tree dict for use with ParameterTree
  """
     
  if not isinstance(parameter, dict):
    return [];
  
  tree = [];    
  for key, value in parameter.iteritems():

    ptype = valueToType(value);    
    if ptype == 'group':
      tree.append({'name' : key, 'type' : ptype, 'children' : dictToTree(value)});
    else:
      tree.append({'name' : key, 'type' : ptype, 'value' : value});
    
  if title is not None:
    tree = [{'name': title, 'type': 'group', 'children': tree}];
  
  return tree;


class ParameterTree(pyqtgraph.parametertree.ParameterTree):
  """Parameter Tree Widget for *ClearMap*"""

  def __init__(self, name, parameter, parent=None, showHeader=True):
    """Create a parameter tree object
    
    Arguments:
      name (str): name of the parameter tree
      parameter (dict): parameter dictionary
      
    Returns:
      ParameterTree: the parameter tree object
    """
    
    super(ParameterTree, self).__init__(parent = parent, showHeader = showHeader); 

    params = dictToTree(parameter);

    self.parameterDict = parameter;
    
    p = pyqtgraph.parametertree.Parameter.create(name = name, type = 'group', children = params)
    self.setParameters(p, showTop=False)
    self.paramSet = p

    self.registerCallback(self.updateParameter);
  
  
  def getParameterItem(self, name):
      """Get an item in a parameter tree by name (delimiter separates groups)
      
      Arguments:
        name (str): parameter name (delimiter separated groups)
        
      Returns:
        ParameterItem: the parameter item
      """
      
      items = self.paramSet.children()
      item = None
      while True:
          first, _, name = name.partition(pt.delimiter)
          names = [i.name() for i in items]
          if first in names:
              item = items[names.index(first)]
              items = item.children()
          if not name:
              return item
              
  
  def changesToFullChanges(self, changes):
    """Converts changes to a dictionary with full hierarchical names
    
    Arguments:
      changes (list): the parameter changes
    """
    new_changes = []
    for param, change, data in changes:
        name = [param.name()]
        parent = param.parent()
        while parent:
            name = [parent.name()] + name
            parent = parent.parent()
        name = pt.delimiter.join(name[1:])
        new_changes.append(dict(name=name, param=param, type=change, data=data))
    
    #print(new_changes)
    return new_changes;

          
  def registerCallback(self, cb):
    """Register a change callback to a parameter tree
    
    Arguments:
      tree (ParameterTree): the parameter tree
      cb (function): the callback function
    """
    
    def change(param, changes):
      cb(self.changesToFullChanges(changes));

    self.paramSet.sigTreeStateChanged.connect(change)

   
  def updateParameter(self, changes):
    """Updates the parameter dictionary on GUI input
    
    Arguments:
      changes (list): the parameter changes
    """
    for change in changes:   
      self.parameterDict = pt.setParameter(self.parameterDict, change['name'], change['data']);
      #print('  parameter: %s' % change['name'])
      #print('  change:    %s' % change['type'])
      #print('  data:      %s' % str(change['data']))
      #print('  class:     %s' % change['data'].__class__.__name__)
      #print('  ----------') 
      #print self.parameterDict


  def getParameter(self):
    """Returns the paramter dictionary corresponding to the actual widget inputs"""
    return self.parameterDict;



def test():
  import numpy as np
  import pyqtgraph as pg
  
  import ClearMap.GUI.Parameter as par;
  reload(par)
  
  def show(widget, title='', width=800, height=800):
    widget.setWindowTitle(title)
    widget.show()
    widget.resize(width, height)
    pg.QtGui.QApplication.instance().exec_()  
  
  pg.mkQApp()
  param = {'x' : 10, 'y' : 'abc', 'file' : par.filestr('asd.txt'), 'd' : {'t' : (100,5), 'x' : ['a', 'z'], 'n' : np.array([[5,6],[6,7]])}};
  
  mainParam = {
      "flatfield"  : True,  
      "background" : None,  
      "scaling"    : "Mean",
      "save"       : par.dirstr('/home/ckirst'),  
      "verbose"    : False,  
      
      "additionalParameter" : param
  }  
  
  paramwidget = par.ParameterTree('Test', mainParam);
  show(paramwidget, title = 'Testing');

if __name__ == '__main__':
  test()

  

