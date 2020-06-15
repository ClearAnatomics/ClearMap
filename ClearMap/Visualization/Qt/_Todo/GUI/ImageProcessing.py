# -*- coding: utf-8 -*-
"""
ImageProcessing
===============

Provides GUI to run image processing analysis routines and pipelines of *ClearMap*
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy

import pyqtgraph as pg

from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import WidgetParameterItem, registerParameterType

from pyqtgraph.Qt import QtGui, QtCore

from ClearMap.Utils.ParameterTools import setParameter


# Create additional Parameter Items to deal with lists and tuples and sliders
class SliderParameterItem(WidgetParameterItem):
    """SliderParameterItem provides slide parameter input"""
    
    def makeWidget(self):
        w = QtGui.QSlider(QtCore.Qt.Horizontal, self.parent())
        w.sigChanged = w.valueChanged
        w.sigChanged.connect(self._set_tooltip)
        self.widget = w
        return w

    def _set_tooltip(self):
        self.widget.setToolTip(str(self.widget.value()))


class SliderParameter(Parameter):
    itemClass = SliderParameterItem

registerParameterType('slider', SliderParameter, override=True)


class TupleParameterItem(WidgetParameterItem):
    """TupleParameterItem provides tuple parameter input"""
    
    def getValue(self):      
      text = unicode(self.widget.text());
      try :
        v = eval(text);
      except:
        v = self.param.defaultValue();
      if not isinstance(v, tuple):
        v = self.param.defaultValue();
      return v;
    
    def makeWidget(self):
        w = QtGui.QLineEdit()
        w.sigChanged = w.editingFinished
        w.value = self.getValue
        w.setValue = lambda v: w.setText(unicode(v))
        w.sigChanging = w.textChanged      
        self.widget = w
        return w
        
class TupleParameter(Parameter):
    itemClass = TupleParameterItem   
 
registerParameterType('tuple', TupleParameter, override=True)   
   
        
class ListParameterItem(WidgetParameterItem):
    """ListParameterItem provides list parameter input"""

    def getValue(self):
      text = unicode(self.widget.text());
      try :
        v = eval(text);
      except:
        v = self.param.defaultValue();
      if not isinstance(v, list):
        v = self.param.defaultValue();
      return v;
    
    def makeWidget(self):
        w = QtGui.QLineEdit()
        w.sigChanged = w.editingFinished
        w.value = self.getValue
        w.setValue = lambda v: w.setText(unicode(v))
        w.sigChanging = w.textChanged      
        self.widget = w
        return w  

class ListParameter(Parameter):
    itemClass = ListParameterItem

registerParameterType('list', ListParameter, override=True)   

registerParameterType('dropdown', pg.parametertree.parameterTypes.ListParameter, override=True)



class NumpyParameterItem(WidgetParameterItem):
    """NumpyParameterItem provides parameter input for numpy type"""

    def getValue(self):
      dv = self.param.defaultValue();
      text = unicode(self.widget.text());
      try :
        v = numpy.array(eval(text));
      except:
        v = dv;
      if not isinstance(v, dv.__class__):
        v = dv;
      return v;
    
    def makeWidget(self):
        w = QtGui.QLineEdit()
        w.sigChanged = w.editingFinished
        w.value = self.getValue
        w.setValue = lambda v: w.setText(unicode(v.tolist()))
        w.sigChanging = w.textChanged      
        self.widget = w
        return w  
        
    def valueChanged(self, param, val, force=False):
        ## called when the parameter's value has changed
        #WidgetParameterItem.valueChanged(self, param, val)
        self.widget.sigChanged.disconnect(self.widgetValueChanged)
        try:
            if force or numpy.any(val != self.widget.value()):
                self.widget.setValue(val)
            self.updateDisplayLabel(val)  ## always make sure label is updated, even if values match!
        finally:
            self.widget.sigChanged.connect(self.widgetValueChanged)
        self.updateDefaultBtn()
        
    def updateDisplayLabel(self, value=None):
        if value is None:
            value = self.param.value()
        text = unicode('array(' + unicode(value.tolist()) + ')');
        self.displayLabel.setText(text)


class NumpyParameter(Parameter):
    
    def valueIsDefault(self):
        """Returns True if this parameter's value is equal to the default value."""
        return numpy.all(self.value() == self.defaultValue())
        
    def setDefault(self, val):
        """Set the default value for this parameter."""
        if numpy.all(self.opts['default'] == val):
            return
        self.opts['default'] = val
        self.sigDefaultChanged.emit(self, val)
    
    def setValue(self, value, blockSignal=None):
        """
        Set the value of this Parameter; return the actual value that was set.
        (this may be different from the value that was requested)
        """
        try:
            if blockSignal is not None:
                self.sigValueChanged.disconnect(blockSignal)
            if numpy.all(self.opts['value'] == value):
                return value
            self.opts['value'] = value
            self.sigValueChanged.emit(self, value)
        finally:
            if blockSignal is not None:
                self.sigValueChanged.connect(blockSignal)
            
        return value
    
    itemClass = NumpyParameterItem

registerParameterType('ndarray', NumpyParameter, override=True)   


class AnyParameterItem(WidgetParameterItem):
    """AnyParameterItem provides parameter input for any type"""

    def getValue(self):
      dv = self.param.defaultValue();
      text = unicode(self.widget.text());
      try :
        v = eval(text);
      except:
        v = dv;
      
      if not isinstance(v, dv.__class__):
        v = dv;
      return v;
    
    def makeWidget(self):
        w = QtGui.QLineEdit()
        w.sigChanged = w.editingFinished
        w.value = self.getValue
        w.setValue = lambda v: w.setText(unicode(v))
        w.sigChanging = w.textChanged      
        self.widget = w
        return w  
        
    def valueChanged(self, param, val, force=False):
        ## called when the parameter's value has changed
        #WidgetParameterItem.valueChanged(self, param, val)
        self.widget.sigChanged.disconnect(self.widgetValueChanged)
        try:
            if force or not val is self.widget.value():
                self.widget.setValue(val)
            self.updateDisplayLabel(val)  ## always make sure label is updated, even if values match!
        finally:
            self.widget.sigChanged.connect(self.widgetValueChanged)
        self.updateDefaultBtn()
        
    def updateDisplayLabel(self, value=None):
        if value is None:
            value = self.param.value()
        text = unicode(value);
        self.displayLabel.setText(text)


class AnyParameter(Parameter):
    
    def valueIsDefault(self):
        """Returns True if this parameter's value is equal to the default value."""
        return self.value() is self.defaultValue()  
        
    def setDefault(self, val):
        """Set the default value for this parameter."""
        if self.opts['default'] is val:
            return
        self.opts['default'] = val
        self.sigDefaultChanged.emit(self, val)
    
    def setValue(self, value, blockSignal=None):
        """
        Set the value of this Parameter; return the actual value that was set.
        (this may be different from the value that was requested)
        """
        try:
            if blockSignal is not None:
                self.sigValueChanged.disconnect(blockSignal)
            if self.opts['value'] is value:
                return value
            self.opts['value'] = value
            self.sigValueChanged.emit(self, value)
        finally:
            if blockSignal is not None:
                self.sigValueChanged.connect(blockSignal)
            
        return value
    
    itemClass = AnyParameterItem


registerParameterType('any', AnyParameter, override=True)   


class FileParameterItem(WidgetParameterItem):
    """FileParameterItem provides file name parameter input"""

    def getValue(self):
      text = unicode(self.widget.text());
      try :
        v = eval(text);
      except:
        v = self.param.defaultValue();
      if not isinstance(v, list):
        v = self.param.defaultValue();
      return v;
      
    
    
    def makeWidget(self):
        w = QtGui.QLineEdit()
        w.sigChanged = w.editingFinished
        w.value = self.getValue
        w.setValue = lambda v: w.setText(unicode(v))
        w.sigChanging = w.textChanged      
        self.widget = w
        return w  

class ListParameter(Parameter):
    itemClass = ListParameterItem

registerParameterType('list', ListParameter, override=True)   


# filename parameter item ?



def valueToType(value):
  """Determines the type of value to use in Parameter
  
  Arguments:
    value (any): parameter value
  
  Returns:
    type (str): the type string to use in Parameter
  """
  
  typeToType = {'int' : 'int', 'float': 'float', 'bool' : 'bool', 
                'str' : 'str', 'list' : 'list' , 'tuple': 'tuple', 
                'ndarray' : 'ndarray', 
                'dict' : 'group'};
        
  return typeToType.get(value.__class__.__name__, 'any');


def dictToTree(parameter, title = None):
  """Convert a parameter dictionary to a parameter tree dictionary
  
  Arguments:
    parameter (dict): (nested) dictionary of parameter
  
  Returns:
    dict : parameter tree dict for use with ParameterTree
  """
  
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


def parameterTree(name, parameter):
    """Create a parameter tree object
    
    Arguments:
      name (str): name of the parameter tree
      parameter (dict): parameter dictionary
      
    Returns:
      ParameterTree: the parameter tree object
    """
    
    pg.mkQApp()
    if not isinstance(parameter, list):
        parameter = [parameter]
    
    p = Parameter.create(name=name, type='group', children=parameter)
    t = ParameterTree()
    t.setParameters(p, showTop=False)
    t.paramSet = p
    return t


def getParameterItem(tree, name):
    """Get an item in a parameter tree by name
    
    Arguments:
      tree (ParameterTree): the parameter tree to retrieve item from
      name (str): parameter name (dot separated groups)
      
    Returns:
      ParameterItem: the parameter item
    """
    
    items = tree.paramSet.children()
    item = None
    while True:
        first, _, name = name.partition('.')
        names = [i.name() for i in items]
        if first in names:
            item = items[names.index(first)]
            items = item.children()
        if not name:
            return item


def registerCallback(tree, cb):
    """Register a change callback to a parameter tree
    
    Arguments:
      tree (ParameterTree): the parameter tree
      cb (function): the callback function
    """
    
    def change(param, changes):
        new_changes = []
        for param, change, data in changes:
            name = [param.name()]
            parent = param.parent()
            while parent:
                name = [parent.name()] + name
                parent = parent.parent()
            name = '.'.join(name[1:])
            new_changes.append(dict(name=name, param=param,
                                    type=change, data=data))
        cb(new_changes)

    tree.paramSet.sigTreeStateChanged.connect(change)


class ParameterWidget(ParameterTree):
  """Parameter tree widget handling dictionaries of parameter"""
  
  def __init__(self, name, parameter, *args, **kwargs):
    """Constructor
    
    Arguments:
      parameter (dict): (nested) dictionary of parameter uased with ClearMap routine
      *args, **kwargs: parameter of constructor for ParameterTree
    """
    
    super(ParameterWidget, self).__init__(*args, **kwargs)    
    
    self.parameterDict = parameter;
    
    #generate widget
    params = dictToTree(parameter);
    self.paramSet = Parameter.create(name = name, type = 'group', children = params);
    self.setParameters(self.paramSet, showTop=False);

    registerCallback(self, self.updateParameter);
  

    
  def updateParameter(self, changes):
    """Updates the parameter dictionary on GUI input
    
    Arguments:
      changes (dict): the parameter changes
    """
    for change in changes:   
      #pi = getParameterItem(self, change['name']);
      self.parameterDict = setParameter(self.parameterDict, change['name'], change['data']);
     
      #print('  parameter: %s' % change['name'])
      #print('  change:    %s' % change['type'])
      #print('  data:      %s' % str(change['data']))
      #print('  class:     %s' % change['data'].__class__.__name__)
      #print('  ----------') 
      print(self.parameterDict)



def show(widget, title='', width=800, height=800):
  """Show a simple application around a widget
  """
  widget.setWindowTitle(title)
  widget.show()
  widget.resize(width, height)
  QtGui.QApplication.instance().exec_()

def test():
  pg.mkQApp()
  param = {'x' : 10, 'y' : 'abc', 'd' : {'t' : (100,5), 'x' : ['a', 'z'], 'n' : numpy.array([[5,6],[6,7]])}};
  paramwidget = ParameterWidget('Test', param);
  show(paramwidget, title = 'Testing');

if __name__ == '__main__':
  test()

  

