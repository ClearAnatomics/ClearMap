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


import re
from functools import cached_property

import numpy as np

from ClearMap.Analysis.vasculature.vasc_graph_utils import vertex_filter_to_edge_filter, edge_filter_to_vertex_filter
from ClearMap.Utils.exceptions import ClearMapValueError

COMBINE_OPERATOR_NAMES = ('and', 'or')
FILE_NAME_TOKEN_RE = re.compile(r'[A-Za-z0-9_.+-]+')  # What a filter may contribute to a file name


def convert_property(graph, mask, src_filter_type, dest_filter_type, operator=np.logical_and):
    if src_filter_type == dest_filter_type:
        return mask

    if src_filter_type == 'vertex' and dest_filter_type == 'edge':
        return vertex_filter_to_edge_filter(graph, mask, operator=operator)
    elif src_filter_type == 'edge' and dest_filter_type == 'vertex':
        return edge_filter_to_vertex_filter(graph, mask, operator=operator)
    else:
        raise ValueError(f'Unsupported conversion from {src_filter_type} to {dest_filter_type}.')


def file_name_token(value) -> str:
    """
    The file name safe string of a filter value (see GraphFilter.as_mask for the semantics):
    a pair is a range (``0to5``), another list a set of values (``1+2+3``), anything else is the value.

    Raises
    ------
    ClearMapValueError
        If the value does not make a file name safe string (e.g. empty, spaces or path separators),
        rather than silently mapping two different values to the same file.
    """
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            token = f'{value[0]}to{value[1]}'
        else:
            token = '+'.join([str(v) for v in value])
    else:
        token = str(value)
    if not FILE_NAME_TOKEN_RE.fullmatch(token):
        raise ClearMapValueError(f'Filter value {value!r} cannot be used in a file name '
                                   f'(allowed characters: {FILE_NAME_TOKEN_RE.pattern}).')
    return token


def combine_filters(filters, operators):
    """
    Combine the filters from left to right: ``filters[0] operators[0] filters[1] operators[1] ...``
    is ``((filters[0] operators[0] filters[1]) operators[1] ...)``, i.e. reading order, not boolean precedence.

    Parameters
    ----------
    filters: list[GraphFilter]
    operators: list[str]
        The operator between each pair of consecutive filters ('and' or 'or'), so len(filters) - 1 of them.

    Returns
    -------
    GraphFilter | CombinedFilter
    """
    _check_operators(filters, operators)
    combined = filters[0]
    for operator, graph_filter in zip(operators, filters[1:]):
        combined = combined.combine_with(graph_filter, operator)
    return combined


def combined_filters_name(filters, operators) -> str:
    """
    The file name safe description of combine_filters(filters, operators),
    e.g. ``vertex_radii_0to5_and_vertex_artery_True``
    """
    _check_operators(filters, operators)
    parts = [filters[0].name]
    for operator, graph_filter in zip(operators, filters[1:]):
        parts += [operator, graph_filter.name]
    return '_'.join(parts)


def _check_operators(filters, operators):
    if not filters:
        raise ClearMapValueError('At least one filter is required.')
    if len(operators) != len(filters) - 1:
        raise ClearMapValueError(f'{len(filters)} filters need {len(filters) - 1} operators, got {len(operators)}.')
    for operator in operators:
        if operator not in COMBINE_OPERATOR_NAMES:
            raise ClearMapValueError(f'Operator must be one of {COMBINE_OPERATOR_NAMES}, got {operator!r}.')


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
    """Shared helpers for GraphFilter & CombinedFilter."""

    def is_defined(self) -> bool:
        raise NotImplementedError

    def as_mask(self, filter_type=None) -> np.ndarray:
        raise NotImplementedError

    def combine_with(self, other, operator):
        """
        Combine this filter with another filter or a mask using the specified operator.
        If combined with a mask, return a mask.
        If combined with another filter, return a CombinedFilter object.

        Parameters
        ----------
        other: GraphFilter or np.ndarray
            The other filter or mask to combine with.
        operator: str or callable
            The operator to use for combining the filters or masks. Can be 'and' or 'or'.
            If a callable is provided, it should take two boolean arrays and return a boolean array.

        Returns
        -------
        np.ndarray or CombinedFilter
            If combined with a mask, returns a boolean mask.
            If combined with another filter, returns a CombinedFilter object.
        """
        if not isinstance(other, (BaseFilter, np.ndarray)):
            raise TypeError('Can only combine with another filter or a mask.')

        operator = operator_name_to_function(operator)

        if isinstance(other, np.ndarray):
            mask = self.as_mask()
            other_mask = other
            if mask.size != other_mask.size:
                raise ValueError('Masks must be of the same size.'
                                 'Size difference implies different graphs or different types (vertex vs edge).')
            return operator(mask, other_mask)
        else:
            for operand in (self, other):
                if not operand.is_defined():
                    raise ClearMapValueError(f'Cannot combine an undefined filter: {operand!r}.')
            return CombinedFilter(self, other, operator)

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

    @property
    def name(self) -> str:
        """The file name safe description of the filter: ``<filter_type>_<property_name>_<value>``"""
        if not self.filter_type or not self.property_name or self.property_value is None:
            raise ClearMapValueError(f'Cannot name an incomplete filter ({self.filter_type=}, '
                                       f'{self.property_name=}, {self.property_value=}).')
        if not FILE_NAME_TOKEN_RE.fullmatch(self.property_name):
            raise ClearMapValueError(f'Property name {self.property_name!r} cannot be used in a file name '
                                       f'(allowed characters: {FILE_NAME_TOKEN_RE.pattern}).')
        return f'{self.filter_type}_{self.property_name}_{file_name_token(self.property_value)}'

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

    def __repr__(self):
        return (f'GraphFilter({self.filter_type!r}, {self.property_name!r}, {self.property_value!r}, '
                f'graph={"set" if self.graph is not None else None})')

    def as_mask(self, filter_type=None):
        if not self.is_defined():  # Fail loudly: indexing with a None mask adds an axis instead of selecting
            raise ClearMapValueError(f'Cannot compute the mask of an undefined filter: {self!r}.')

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

    def __repr__(self):
        return f'CombinedFilter({self.left!r}, {self.right!r}, {getattr(self.op, "__name__", self.op)})'

    @property
    def filter_type(self):
        """The right operand is cast to the type of the left one"""
        return self.left.filter_type

    def is_defined(self) -> bool:
        return self.left.is_defined() and (self.right is None or self.right.is_defined())

    # -------- core ------------------------------------------------------------
    def as_mask(self, filter_type=None):
        """
        Evaluate the whole expression in one pass.

        Parameters
        ----------
        filter_type : 'vertex' | 'edge' | None
            Type of mask the caller wants back. None: the type of the left operand.
        """
        if filter_type is None:
            filter_type = self.filter_type
        # Post-order traversal: evaluate leaves first, then apply op
        if self.right is None:  # NOT
            return self.op(self.left.as_mask(filter_type))

        m1 = self.left.as_mask(filter_type)
        m2 = self.right.as_mask(filter_type)
        return self.op(m1, m2)
