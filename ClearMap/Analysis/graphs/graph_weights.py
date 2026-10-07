"""
Weights for the voxelization of graph vertices: one non-negative number per vertex, read from a
numeric vertex property, or from a numeric edge property reduced onto the vertices with the rule
of graph_processing.DEFAULT_EDGE_TO_VERTEX.

    radius = GraphWeight(graph, 'vertex', 'radius_units')
    length = GraphWeight(graph, 'edge', 'length')   # half of each incident edge
    length.name            # 'edge_length', names the weighted density file
    length.vertex_values()

Filters select the vertices (see graph_filters), weights say how much each one counts.
"""

import numpy as np

from ClearMap.Analysis.graphs.graph_filters import graph_property, PROPERTY_TYPES
from ClearMap.Analysis.graphs.graph_processing import DEFAULT_EDGE_TO_VERTEX
from ClearMap.IO.assets_constants import FILE_NAME_TOKEN_RE
from ClearMap.Utils.exceptions import ClearMapValueError

# graph-tool scalar numeric value types (bool is a filter, not a weight)
NUMERIC_VALUE_TYPES = ('uint8_t', 'int16_t', 'short', 'int32_t', 'int', 'int64_t', 'long', 'long long',
                       'float', 'double', 'long double')
DEGREE_WEIGHT_NAME = 'vertex_degree'


class GraphWeight:
    def __init__(self, graph, property_type: str, property_name: str):
        """
        Parameters
        ----------
        graph: Graph | None
            None is enough to name the weight.
        property_type: str
            'vertex' or 'edge'.
        property_name: str
            A numeric property. An edge property needs a rule in DEFAULT_EDGE_TO_VERTEX.
        """
        if property_type not in PROPERTY_TYPES:
            raise ClearMapValueError(f'Property type must be one of {PROPERTY_TYPES}, got {property_type!r}.')
        if property_type == 'edge' and property_name not in DEFAULT_EDGE_TO_VERTEX:
            raise ClearMapValueError(f'No rule to turn the edge property {property_name!r} into vertex weights. '
                                     f'Known edge properties: {list(DEFAULT_EDGE_TO_VERTEX)} '
                                     f'(graph_processing.DEFAULT_EDGE_TO_VERTEX).')
        if not FILE_NAME_TOKEN_RE.fullmatch(property_name):
            raise ClearMapValueError(f'Property name {property_name!r} cannot be used in a file name '
                                     f'(allowed characters: {FILE_NAME_TOKEN_RE.pattern}).')
        self.graph = graph
        self.property_type = property_type
        self.property_name = property_name

    @classmethod
    def from_name(cls, graph, name: str) -> 'GraphWeight':
        """The inverse of GraphWeight.name, e.g. 'vertex_radius_units' or 'edge_length'"""
        property_type, _, property_name = name.partition('_')
        return cls(graph, property_type, property_name)

    @property
    def name(self) -> str:
        """``<property_type>_<property_name>``, which names the weighted density file"""
        return f'{self.property_type}_{self.property_name}'

    def vertex_values(self) -> np.ndarray:
        """
        One weight per vertex of the graph.

        Raises
        ------
        ClearMapValueError
            If the property is not a 1D numeric property with finite non-negative values.
        """
        values = _checked_weights(graph_property(self.graph, self.property_type, self.property_name),
                                  f'{self.property_type} property {self.property_name!r}')
        if self.property_type == 'edge':
            to_vertex = DEFAULT_EDGE_TO_VERTEX[self.property_name]
            values = to_vertex(self.graph.edge_connectivity(order='eid'), values, self.graph.n_vertices)
        return values


def weight_choices(graph) -> list[str]:
    """The names (see GraphWeight.from_name) of the properties the vertices of graph can be weighted by"""
    choices = [f'vertex_{name}' for name in graph.vertex_properties
               if graph.vertex_property_map(name).value_type() in NUMERIC_VALUE_TYPES]
    choices.append(DEGREE_WEIGHT_NAME)
    choices += [f'edge_{name}' for name in graph.edge_properties
                if name in DEFAULT_EDGE_TO_VERTEX
                and graph.edge_property_map(name).value_type() in NUMERIC_VALUE_TYPES]
    return choices


def _checked_weights(values, description: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ClearMapValueError(f'The {description} cannot be used as weights: it has shape {values.shape}, '
                                 f'one value per element is required.')
    if values.dtype == bool or not np.issubdtype(values.dtype, np.number):
        raise ClearMapValueError(f'The {description} cannot be used as weights: it is not numeric ({values.dtype}). '
                                 f'Use a filter to select by a boolean property.')
    if not np.all(np.isfinite(values)):
        raise ClearMapValueError(f'The {description} has non finite values (NaN or inf): it cannot be used as weights.')
    if np.any(values < 0):
        raise ClearMapValueError(f'The {description} has negative values: it cannot be used as weights.')
    return values
