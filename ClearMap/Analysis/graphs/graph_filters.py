"""
Graph filters for ClearMap

# make some atomic filters
artery = GraphFilter(g, 'vertex', 'is_artery', True)
small  = GraphFilter(g, 'vertex', 'radius',    (0, 4))
vein   = GraphFilter(g, 'vertex', 'is_vein',   True)
some_edge_prop_filter = GraphFilter(g, 'edge', 'some_edge_prop', True)

# Boolean equation:  (artery & small) | ~(vein)
cap_network = (artery & small) | ~vein & some_edge_prop_filter

mask = cap_network.as_mask('vertex')   # one property read per leaf; single traversal
"""


from functools import cached_property

import numpy as np

from ClearMap.Analysis.vasculature.vasc_graph_utils import vertex_filter_to_edge_filter, edge_filter_to_vertex_filter


def convert_property(graph, mask, src_filter_type, dest_filter_type, operator=np.logical_and):
    if src_filter_type == dest_filter_type:
        return mask

    if src_filter_type == 'vertex' and dest_filter_type == 'edge':
        return vertex_filter_to_edge_filter(graph, mask, operator=operator)
    elif src_filter_type == 'edge' and dest_filter_type == 'vertex':
        return edge_filter_to_vertex_filter(graph, mask, operator=operator)
    else:
        raise ValueError(f'Unsupported conversion from {src_filter_type} to {dest_filter_type}.')


def operator_name_to_function(operator):
    if isinstance(operator, str):
        if operator == 'and':
            operator = np.logical_and
        elif operator == 'or':
            operator = np.logical_or
        else:
            raise ValueError('Operator must be "and" or "or".')
    return operator


class BaseFilter:
    """Shared helpers for GraphFilter & CombinedGraphFilter."""

    # NOT  (~filter)
    def __invert__(self):
        return CombinedFilter(self, None, np.logical_not)

    # AND, OR, XOR keep using your existing overloads
    def __and__(self, other):  return CombinedFilter(self, other, np.logical_and)
    def __rand__(self, other): return CombinedFilter(other, self, np.logical_and)
    def __or__(self, other):   return CombinedFilter(self, other, np.logical_or)
    def __ror__(self, other):  return CombinedFilter(other, self, np.logical_or)
    def __xor__(self, other):  return CombinedFilter(self, other, np.logical_xor)
    def __rxor__(self, other): return CombinedFilter(other, self, np.logical_xor)


class GraphFilter(BaseFilter):
    """
    A class to filter vertices or edges of a graph based on a property.
    The filter can be combined with other filters or masks using logical operators to chain multiple filters together.

    If combining with a mask, the filter type is ignored. However, masks must be of the same size
    (which implies the same graph and the same type).
    If combined with another filter object, the right filter will be cast to the type (vertex/edge)
    of the left operand when combined with another filter.

    You can combine filters using the following operators (think of them as set operations):
    - `&` (and): Intersection of two filters
    - `|` (or): Union of two filters
    - `+` (or): Union of two filters (same as `|`)
    """
    def __init__(self, graph, filter_type='', property_name='', property_value=None):
        if not filter_type in ['vertex', 'edge', '']:
            raise ValueError('Filter type must be "vertex" or "edge" or "" (unspecified).')
        self.graph = graph
        self.filter_type = filter_type
        self.property_name = property_name
        self.property_value = property_value

    def combine_with(self, other, operator):
        """
        Combine this filter with another filter or a mask using the specified operator.
        If combined with a mask, return a mask.
        If combined with another filter, return a CombinedGraphFilter object.

        Parameters
        ----------
        other: GraphFilter or np.ndarray
            The other filter or mask to combine with.
        operator: str or callable
            The operator to use for combining the filters or masks. Can be 'and' or 'or'.
            If a callable is provided, it should take two boolean arrays and return a boolean array.

        Returns
        -------
        np.ndarray or CombinedGraphFilter
            If combined with a mask, returns a boolean mask.
            If combined with another filter returns a CombinedGraphFilter object.
        """
        if not isinstance(other, (GraphFilter, np.ndarray)):
            raise TypeError('Can only combine with another GraphFilter object.')

        operator = operator_name_to_function(operator)

        if isinstance(other, np.ndarray):
            mask = self.as_mask()
            other_mask = other
            if mask.size != other_mask.size:
                raise ValueError('Masks must be of the same size.'
                                 'Size difference implies different graphs or different types (vertex vs edge).')
            return operator(mask, other_mask)
        else:
            if not self.is_defined() or not other.is_defined():
                raise ValueError('Cannot combine with another filter if this filter is not defined.')
            return CombinedFilter(self, other, operator)


    def is_defined(self):
        return (self.graph is not None and
                self.filter_type != '' and
                self.property_name != '' and
                self.property_value is not None)

    # def as_type(self, filter_type):
    #     if self.filter_type == filter_type:
    #         return self
    #     elif self.filter_type == 'vertex' and filter_type == 'edge':
    #         return GraphFilter(self.graph, filter_type, self.property_name,
    #                            vertex_filter_to_edge_filter(self.property_value))

    @cached_property
    def _raw_property(self):
        if self.filter_type == 'vertex':
            if self.property_name in ('degree', 'degrees'):
                return self.graph.vertex_degrees()
            return self.graph.vertex_property(self.property_name)
        elif self.filter_type == 'edge':
            return self.graph.edge_property(self.property_name)
        raise RuntimeError

    def as_mask(self, filter_type=None):
        if not self.is_defined():
            return

        if filter_type is None:
            filter_type = self.filter_type

        prop = self._raw_property

        if isinstance(self.property_value, (tuple, list)):
            if len(self.property_value) == 2:  # pair, then min/max
                mask = np.logical_and(prop >= self.property_value[0],
                                      prop <= self.property_value[1])
            else:  # list, then in
                mask = np.isin(prop, self.property_value)
        else:  # scalar, string, bool, etc.
            mask = prop == self.property_value

        if filter_type != self.filter_type:
            mask = convert_property(self.graph, mask, self.filter_type, filter_type)

        return mask


class CombinedFilter(BaseFilter):
    def __init__(self, left, right, op):
        self.left  = left      # GraphFilter | CombinedFilter | None (for NOT)
        self.right = right     # idem
        self.op    = op        # a numpy ufunc (logical_and, logical_not, …)

    # -------- core ------------------------------------------------------------
    def as_mask(self, filter_type):
        """
        Evaluate the whole expression in one pass.

        Parameters
        ----------
        filter_type : 'vertex' | 'edge'
            Type of mask the caller wants back.
        """
        # Post-order traversal: evaluate leaves first, then apply op
        if self.right is None:  # NOT
            return self.op(self.left.as_mask(filter_type))

        m1 = self.left.as_mask(filter_type)
        m2 = self.right.as_mask(filter_type)
        return self.op(m1, m2)
