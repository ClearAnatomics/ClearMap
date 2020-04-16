# -*- coding: utf-8 -*-
"""
Lazy attributes
"""
__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'


class lazyattr(object):
  """Attribute whose value is computed on first access."""
  
  __slots__ = ('func',)

  def __init__(self, func):
    self.func = func
  
  def __get__(self, instance, owner):
    if instance is None:
      return self
    try:
      value = self.func(instance)
    except AttributeError as e:
      raise RuntimeError(e)
    if value is NotImplemented:
      return getattr(super(owner, instance), self.func.__name__)
    setattr(instance, self.func.__name__, value)
    return value
