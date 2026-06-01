# -*- coding: utf-8 -*-
"""
Lazy
====

Lazy evaluation attributes.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'https://idisco.info'
__download__  = 'https://www.github.com/ChristophKirst/ClearMap2'

from importlib import import_module


def lazy_mixin(class_path):
    """
    Injects mixin methods on first access, then self-destructs.
    The mixin module (and its deps like matplotlib) is never
    imported until someone actually calls a mixed-in method.

    Arguments
    ---------
    class_path : str
        The dotted path to the mixin class, e.g. "ClearMap.Visualization.Plotting.PlottingMixin".
    """
    module_path, mixin_name = class_path.rsplit('.', 1)

    def decorator(cls):
        def __getattr__(self, name):
            mixin = getattr(import_module(module_path), mixin_name)
            for k, v in vars(mixin).items():
                if not k.startswith('_'):
                    setattr(cls, k, v)           # patch the class, not the instance
            del cls.__getattr__                  # self-destruct
            return getattr(self, name)           # retry (or AttributeError if truly missing)

        cls.__getattr__ = __getattr__
        return cls

    return decorator


class lazyattr:
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
