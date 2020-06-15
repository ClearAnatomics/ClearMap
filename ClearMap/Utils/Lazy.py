# -*- coding: utf-8 -*-
"""
Lazy
====

Lazy evaluation attributes.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


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
