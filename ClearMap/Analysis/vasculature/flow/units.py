from __future__ import division

import quantities as pq
import string

__all__ = ['pq', 'scaling_factor', 'scaling_factor_du']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


ug = pq.UnitQuantity('microgram', pq.g/1e6, symbol='ug')
ng = pq.UnitQuantity('nanogram', pq.g/1e9, symbol='ng')
pg = pq.UnitQuantity('picogram', pq.g/1e12, symbol='pg')
fL = pq.UnitQuantity('femtoliter', pq.liter/1e15, symbol='fL')

def scaling_factor(Q_rescale, Q_compare):
    """Computes the scaling factor to multiply a quantity with, if it is to be
    expressed in units of another quantity.
    INPUT: Q_rescale: The quantity instance or unit-string that is to be
           rescaled.
           Q_compare: The quantity instance or unit-string whose units are
           desired.
    OUTPUT: Factor by which Q_rescale is to be multiplied with, if expressed in
            units of Q_compare
    """
    if not isinstance(Q_rescale, pq.Quantity):
        Q_rescale = pq.Quantity(1.0, Q_rescale)
    return Q_rescale.rescale(Q_compare).magnitude.tolist()


def scaling_factor_du(Q_rescale, defaultUnits):
    """Computes the scaling factor to multiply a quantity with, if it is to be
    expressed in the default units provided.
    INPUT: Q_rescale: The quantity instance or unit-string that is to be
           rescaled.
           defaultUnits: The default units as dictionary, e.g.: {'length': 'm',
                         'mass': 'kg', 'time': 's'}
    OUTPUT: Factor by which Q_rescale is to be multiplied with, if expressed in
            the default units.
    """
    pq.set_default_units(**defaultUnits)
    if not isinstance(Q_rescale, pq.Quantity):
        Q_rescale = pq.Quantity(1.0, Q_rescale)
    return Q_rescale.simplified.magnitude.tolist()
