# -*- coding: utf-8 -*-
"""
Feather
=======

Backend for Apache Feather tables.
"""

import pandas as pd
import pyarrow  # noqa: F401  # pandas Feather support depends on pyarrow; fail at import if missing

from ClearMap.IO.source.Source import TableSource
from ClearMap.IO.source.protocol import Backend


class FeatherSource(TableSource):
    """Table source backed by a Feather file."""

    backend = Backend.FEATHER
    _name = 'Feather-Source'  # FIXME

    @classmethod
    def _load(cls, location, **kwargs):
        return pd.read_feather(location, **kwargs)

    @classmethod
    def _dump(cls, frame, location, **kwargs):
        frame.to_feather(location, **kwargs)


SOURCE_CLASS = FeatherSource


###############################################################################
### IO Interface
###############################################################################

def open_ro(source_, **kwargs):
    return FeatherSource.open_ro(source_, **kwargs)


def read(source_, slicing=None, **kwargs):
    return FeatherSource.read_table(source_, slicing=slicing, **kwargs)


def write(sink, data=None, slicing=None, overwrite=True, **kwargs):
    return FeatherSource.write_table(sink, data, slicing=slicing, overwrite=overwrite, **kwargs)


def create(location=None, shape=None, dtype=None, order=None,
           mode=None, array=None, as_source=True, **kwargs):
    return FeatherSource.create_table(location, shape=shape, dtype=dtype, order=order,
                                      mode=mode, array=array, as_source=as_source, **kwargs)


def edit(source_, **kwargs):
    return FeatherSource.edit(source_, **kwargs)
