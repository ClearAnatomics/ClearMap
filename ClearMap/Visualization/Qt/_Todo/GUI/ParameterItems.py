# -*- coding: utf-8 -*-
"""
ParameterItems
==============

Provides *ClearMap*  specific parameter items for use with pyqtgraph.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'

import numpy as np

#import PyQt4
import pyqtgraph as pg

from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import WidgetParameterItem, registerParameterType

from pyqtgraph.Qt import QtGui, QtCore

from .FileDialog import FileDialog


class filestr(str):
  """String class indicating a file location"""
  pass


class dirstr(str):
  """String class indicating a directory location"""
  pass


class option(object):
  """Options class taking one value from a list of options"""
  def __init__(self, value, options):
    self.options = options;
    super(option, self).__init__(value);
    


#class region(Region)
    


#TODO: Region parameter!, bounded parameters ?, option lists

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
    """SliderParameter provides slide parameter input"""
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
    """TupleParameter provides tuple parameter input"""
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
    """ListParameter provides list parameter input"""
    itemClass = ListParameterItem

registerParameterType('list', ListParameter, override=True)   

registerParameterType('dropdown', pg.parametertree.parameterTypes.ListParameter, override=True)


class NumpyParameterItem(WidgetParameterItem):
    """NumpyParameterItem provides parameter input for numpy type"""

    def getValue(self):
      dv = self.param.defaultValue();
      text = unicode(self.widget.text());
      try :
        v = np.array(eval(text));
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
            if force or np.any(val != self.widget.value()):
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
    """NumpyParameter provides parameter input for numpy type"""
    
    def valueIsDefault(self):
        """Returns True if this parameter's value is equal to the default value."""
        return np.all(self.value() == self.defaultValue())
    
    #overwrite comparison routines using numpy    
    def setDefault(self, val):
        """Set the default value for this parameter."""
        if np.all(self.opts['default'] == val):
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
            if np.all(self.opts['value'] == value):
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
    """AnyParameter provides parameter input for any type"""

    #overwrite routines equating values
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



class FileSelectionWidget(QtGui.QWidget):
  """FileSelectionWidget for use in ParameterTree"""
  
  sigChanged = QtCore.pyqtSignal(object);
  
  def __init__(self, *args, **kwargs):
      super(FileSelectionWidget, self).__init__(*args, **kwargs);
      
      #construct file selection widget
      self.checkbox = QtGui.QCheckBox();
      
      self.lineedit = QtGui.QLineEdit();
      
      self.button = QtGui.QPushButton();
      ic = QtGui.QFileIconProvider().icon(5)
      #btn.setFixedSize(ic.actualSize(ic.availableSizes()[0]));
      ics = QtCore.QSize(20,20);
      self.button.setFixedSize(ic.actualSize(ics));
      self.button.setText("");
      self.button.setIcon(ic);
      self.button.setIconSize(ic.actualSize(ics));
      self.button.clicked.connect(self.getFile);      
      
      layout = QtGui.QHBoxLayout()
      layout.setContentsMargins(0,0,0,0)
      layout.setSpacing(0);
      layout.addWidget(self.checkbox);
      layout.addWidget(self.lineedit);
      layout.addWidget(self.button);    
      self.setLayout(layout)

      self.lineedit.textChanged.connect(self.sigChanged.emit);
      self.checkbox.stateChanged.connect(self.sigChanged.emit);
    
  def getFile(self):
      fd = FileDialog(); 
      #fn = fd.getOpenFileName(caption = "Save", directory = led.text()); 
      #fd.setDirectory(led.text());
      if fd.exec_():
        fn = fd.selectedFiles()[0];
      if fn != '':
        self.lineedit.setText(fn);    
    
  def value(self):
      if self.checkbox.isChecked():
        return filestr(unicode(self.lineedit.text()));
      else:
        return None;
        
  def setValue(self, value):
      if value is None:
        self.checkbox.setChecked(False);
        dv = self.param.defaultValue();
        if isinstance(dv, str):
          self.lineedit.setText(dv);
        else:
          self.lineedit.setText('');
      else:
        self.checkbox.setChecked(True);
        self.lineedit.setText(unicode(value));      
      
      

class FileParameterItem(WidgetParameterItem):
    """FileParameterItem provides parameter input for file names"""
    
    def makeWidget(self):
      w = FileSelectionWidget();
      self.widget = w;
      
      return w  

class FileParameter(Parameter):
  """FileParameter provides parameter input for file names"""
  itemClass = FileParameterItem

registerParameterType('filestr', FileParameter, override=True)




class DirectorySelectionWidget(QtGui.QWidget):
  """DirectorySelectionWidget for use in ParameterTree"""
  
  sigChanged = QtCore.pyqtSignal(object);
  
  def __init__(self, *args, **kwargs):
      super(DirectorySelectionWidget, self).__init__(*args, **kwargs);
      
      #construct file selection widget
      self.checkbox = QtGui.QCheckBox();
      self.lineedit = QtGui.QLineEdit();
      self.button = QtGui.QPushButton();
      ic = QtGui.QFileIconProvider().icon(5)
      #btn.setFixedSize(ic.actualSize(ic.availableSizes()[0]));
      ics = QtCore.QSize(20,20);
      self.button.setFixedSize(ic.actualSize(ics));
      self.button.setText("");
      self.button.setIcon(ic);
      self.button.setIconSize(ic.actualSize(ics));
      self.button.clicked.connect(self.getDirectory);      
      
      layout = QtGui.QHBoxLayout()
      layout.setContentsMargins(0,0,0,0)
      layout.setSpacing(0);
      layout.addWidget(self.checkbox);
      layout.addWidget(self.lineedit);
      layout.addWidget(self.button);    
      self.setLayout(layout)

      self.lineedit.textChanged.connect(self.sigChanged.emit);
      self.checkbox.stateChanged.connect(self.sigChanged.emit);
    
  def getDirectory(self):
      fd = QtGui.QFileDialog(); 
      #fn = fd.getOpenFileName(caption = "Save", directory = led.text()); 
      #fd.setDirectory(led.text());
      fn = fd.getExistingDirectory(directory = self.lineedit.text());
      self.setValue(fn);  
    
  def value(self):
      if self.checkbox.isChecked():
        return dirstr(unicode(self.lineedit.text()));
      else:
        return None;
        
  def setValue(self, value):
      if value is None:
        self.checkbox.setChecked(False);
        dv = self.param.defaultValue();
        if isinstance(dv, str):
          self.lineedit.setText(dv);
        else:
          self.lineedit.setText('');
      else:
        self.checkbox.setChecked(True);
        self.lineedit.setText(unicode(value));      
      
      

class DirectoryParameterItem(WidgetParameterItem):
    """FileParameterItem provides parameter input for file names"""
    
    def makeWidget(self):
      w = DirectorySelectionWidget();
      self.widget = w;
      
      return w  


class DirectoryParameter(Parameter):
  """FileParameter provides parameter input for file names"""
  itemClass = DirectoryParameterItem

registerParameterType('dirstr', DirectoryParameter, override=True)





#
#class PreviewWidget(QtGui.QWidget):
#  """PreviewWidget for use in ParameterTree"""
#  
#  sigChanged = QtCore.pyqtSignal(object);
#  
#  def __init__(self, *args, **kwargs):
#      super(FileSelectionWidget, self).__init__(*args, **kwargs);
#      
#      #construct file selection widget
#      self.checkbox = QtGui.QCheckBox();
#      
#      self.lineedit = [];
#      self.lineeditlabel = ['x', 'y', 'z'];
#      for d in range(3):
#        self.lineedit.append(QtGui.QLineEdit());
#        
#      
#      layout = QtGui.QHBoxLayout()
#      layout.setContentsMargins(0,0,0,0)
#      layout.setSpacing(0);
#      layout.addWidget(self.checkbox);
#      for d in range(len(self.lineedit)):
#        layout.addWidget(QtGui.QLabel(self.lineeditlabel[d]));
#        layout.addWidget(self.lineedit[d]); 
#      self.setLayout(layout)
#
#      for d in range(len(self.lineedit)):
#        self.lineedit[d].textChanged.connect(self.sigChanged.emit);
#      self.checkbox.stateChanged.connect(self.sigChanged.emit);
#  
#  def value(self):
#      if self.checkbox.isChecked():
#        
#        text = unicode(self.widget.text());
#      try :
#        v = eval(text);
#      except:
#        v = self.param.defaultValue();
#      if not isinstance(v, tuple):
#        v = self.param.defaultValue();
#      return v;
#
#      #else:
#      #  return None;
#        
#  def setValue(self, value):
#      if value is None:
#        self.checkbox.setChecked(False);
#        dv = self.param.defaultValue();
#        if isinstance(dv, str):
#          self.lineedit.setText(dv);
#        else:
#          self.lineedit.setText('');
#      else:
#        self.checkbox.setChecked(True);
#        self.lineedit.setText(unicode(value));      
#      
#      
#
#class FileParameterItem(WidgetParameterItem):
#    """FileParameterItem provides parameter input for file names"""
#    
#    def makeWidget(self):
#      w = FileSelectionWidget();
#      self.widget = w;
#      
#      return w  
#
#class FileParameter(Parameter):
#  """FileParameter provides parameter input for file names"""
#  itemClass = FileParameterItem
#
#registerParameterType('filestr', FileParameter, override=True)
#











