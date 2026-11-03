# -*- coding: utf-8 -*-
"""
registry
========

Declarative registry of IO backends.

This module only lazily imports backend modules.
It is imported by :mod:`ClearMap.IO.dispatch`.

Backend modules must follow this nomenclature convention:
# FIXME
"""
__author__ = 'Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE.txt)'

from ClearMap.IO.source.protocol import Backend


BY_KEY = {backend.key: backend for backend in Backend}
BY_MODULE_NAME = {backend.module_name: backend for backend in Backend}
BY_EXTENSION = {ext: backend for backend in Backend for ext in backend.extensions}

_extensions = [ext for backend in Backend for ext in backend.extensions]
if len(_extensions) != len(set(_extensions)):
    raise RuntimeError('Duplicate file extensions declared by IO backends.')

SOURCE_EXTENSIONS = tuple(BY_EXTENSION)


def supports_extension(extension: str) -> bool:
    return extension.lstrip('.').lower() in BY_EXTENSION


def backend_from_extension(extension: str) -> Backend | None:
    return BY_EXTENSION.get(extension.lstrip('.').lower())
