#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Fast graph reduction for degree-2 vertices.

This module provides a Cython implementation for finding chains formed by degree-2 vertices.

The logic is as follows:
- Identify Potential Start Edges: For each edge (v1, v2):
    If exactly one endpoint is degree-2 and the other is not degree-2, this edge can be a starting point for a chain.
    If this edge is not visited, start building a chain from this edge.

- Chain Construction:
    Suppose (v1, v2) is such an edge, and v1 is degree-2, v2 is not degree-2.
    Mark this edge visited.
    Initialize chain_edges = [this_edge] and chain_vertices = [v2, v1].
    Then, while the current vertex has degree-2:
        Find the other edge at current vertex (not back to prev_vertex and not visited).
        Mark that new edge visited, append to chain_edges, and set current_vertex to the other end of that edge.
    Once you exit, you have a maximal chain from v2 to a new non-degree-2 vertex.

"""
import warnings

from cython.parallel import prange, parallel

import numpy as np
cimport numpy as cnp

from libc.stdio cimport printf
from libc.stdint cimport uint32_t, uint8_t

from libcpp.vector cimport vector
from libcpp.pair cimport pair


ctypedef uint32_t index_t
ctypedef vector[index_t] index_vector_t

ctypedef uint32_t vertex_t
ctypedef vector[vertex_t] vertex_vector_t

ctypedef pair[vertex_t, vertex_t] edge_t  # (used by find_endpoint_vertex)
ctypedef vector[edge_t] edge_vector_t

ctypedef pair[index_t, vertex_t] adj_entry_t  # (edge ID, neighbour vertex)

ctypedef uint8_t degree_t
ctypedef vector[degree_t] degree_vector_t

ctypedef fused source_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.float32_t
    cnp.float64_t


ctypedef fused sink_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.float32_t
    cnp.float64_t


ctypedef fused graphtool_scalar_t:
    double  # float64  (and promoted float32)
    long long  # int64 / uint64
    int  # int32
    unsigned char  # bool_


# ctypedef np.uint32_t np_vertex_t
# ctypedef np.uint8_t np_degree_t
# ctypedef np.uint32_t[::1] np_vertex_array_t
# ctypedef np.uint32_t[::1] np_index_array_t
# ctypedef np.uint8_t[::1] np_degree_array_t

# Define a sentinel value for uninitialized degrees
# cdef degree_t UN_INITIALIZED_DEGREE = 0xFF  # Replaces literal (type_max(degree_t) if unsigned, use -1 if signed
cdef index_t INVALID_IDX = <index_t> -1  # cast -1 to uint to wrap around to max  # Replaces literal (type_max(index_t) if unsigned, use -1 if signed
cdef size_t OK = 0, INVALID_EDGE = 1, INVALID_DEGREE_2 = 2

# # Import C++ std::array
# cdef extern from "<array>" namespace "std":
#     cdef cppclass array[T, size_t]:
#         T& operator[](size_t)  # For indexing
#         size_t size()  # To get the size of the array


# FIXME: adjacency is actually build on edge_id, neigbour_vertex pairs
#   adjacency[u].push_back(edge_t(i, w))
#   adjacency[w].push_back(edge_t(i, u))
#   so that for vertex[u] we have the edge_id and the neighbour_vertex w

cdef edge_t find_endpoint_vertex(vertex_t v1, vertex_t v2,
                                 degree_t deg_v1, degree_t deg_v2):
    """
    Given two vertices and their degrees, return (non-deg2, deg2) or (INVALID_IDX, INVALID_IDX).
    """
    if deg_v1 == 2 and deg_v2 != 2:
        return edge_t(v2, v1)
    elif deg_v2 == 2 and deg_v1 != 2:
        return edge_t(v1, v2)
    else:
        return edge_t(INVALID_IDX, INVALID_IDX)


cdef size_t trace_chain(
    index_t start_edge_idx,
    uint32_t[:, :] connectivity,
    uint8_t[:] vertex_degs,
    vector[vector[adj_entry_t]] &adjacency,
    vector[uint8_t] &visited_edges,
    list result
):
    """
    Start from a given edge that connects a degree-2 vertex and a non-degree-2 vertex.
    Follow the chain through degree-2 vertices until reaching another non-degree-2 vertex.
    """
    cdef index_t e = start_edge_idx
    cdef vertex_t v1 = connectivity[e, 1]
    cdef vertex_t v2 = connectivity[e, 2]

    cdef degree_t deg_v1 = vertex_degs[v1]
    cdef degree_t deg_v2 = vertex_degs[v2]

    cdef edge_t ordered_edge = find_endpoint_vertex(v1, v2, deg_v1, deg_v2)
    cdef vertex_t endpoint_vertex = ordered_edge.first
    cdef vertex_t deg2_vertex = ordered_edge.second
    if endpoint_vertex == INVALID_IDX or deg2_vertex == INVALID_IDX:
        # print(f"{start_edge_idx} is not a valid start edge (degrees = {deg_v1}, {deg_v2}\n")
        printf(b"%u: invalid start edge (degrees = %u, %u)\n", start_edge_idx, deg_v1, deg_v2)
        return INVALID_EDGE

    visited_edges[e] = 1

    # Initialize chain with this edge and its vertices
    cdef index_vector_t chain_edges
    cdef vertex_vector_t chain_verts
    cdef uint8_t expected_chain_size = 200
    chain_edges.reserve(expected_chain_size)
    chain_verts.reserve(expected_chain_size + 1)

    chain_edges.push_back(e)
    # Order: endpoint_vertex (non-degree-2), deg2_vertex, then follow chain
    chain_verts.push_back(endpoint_vertex)
    chain_verts.push_back(deg2_vertex)

    cdef vertex_t current_vertex = deg2_vertex
    cdef vertex_t prev_vertex = endpoint_vertex

    cdef adj_entry_t entry
    cdef size_t n_neighbours = 0, i = 0
    while vertex_degs[current_vertex] == 2:
        n_neighbours = adjacency[current_vertex].size()
        if n_neighbours != 2:
            printf(b"ERROR: Vertex %u should be degree 2 but has %zu neighbours\n", current_vertex, n_neighbours)
            return INVALID_DEGREE_2
        entry = adj_entry_t(INVALID_IDX, INVALID_IDX)
        for i in range(n_neighbours):
            entry = adjacency[current_vertex][i]
            if entry.second != prev_vertex and not visited_edges[entry.first]:
                break
        if entry.first == INVALID_IDX or visited_edges[entry.first]:
            break

        visited_edges[entry.first] = True
        chain_edges.push_back(entry.first)
        chain_verts.push_back(entry.second)

        prev_vertex = current_vertex
        current_vertex = entry.second


    # Append the chain to result
    to_python(chain_edges, chain_verts, connectivity[:, 0], result)
    return OK


cdef void to_python(
    index_vector_t& full_edges,
    vertex_vector_t& full_verts,
    uint32_t[:] edge_ids,
    list result
):
    """
    Convert to Python objects, and append to result.
    """

    cdef size_t i = 0
    cdef size_t j = 0

    cdef list py_edges = []
    for i in range(full_edges.size()):
        py_edges.append(edge_ids[full_edges[i]])  # FIXME: unsure about this
        # py_edges.append(full_edges[i])  # TODO: replace by this if pure index

    cdef cnp.ndarray[cnp.uint64_t, ndim=1] py_verts = np.zeros(full_verts.size(), dtype=np.uint64)
    for j in range(full_verts.size()):
        py_verts[j] = full_verts[j]

    cdef tuple chain_info = (py_edges, py_verts)
    result.append(chain_info)


# cdef void draw_progress_bar(int counter, int num_vertices, int max_symbols=30):
#     cdef float percentage = (counter / num_vertices) * 100
#     cdef int num_symbols = (counter / num_vertices) * max_symbols
#     cdef int n_blanks = (max_symbols - num_symbols)
#     # FIXME: use printf
#     print("\r[%-*s%s] %.2f%%", max_symbols, b"#" * num_symbols, b" " * n_blanks, percentage)

# ================================================================
# MAIN FUNCTION

# ================================================================

cdef vertex_t find_max_v_id(uint32_t[:, :] arr):
    cdef vertex_t max_vertex_id = 0
    cdef size_t n_rows = arr.shape[0]

    # Unrolling the loop for columns 1 and 2
    cdef size_t i
    for i in range(n_rows):
        # Compare values in column 1
        if arr[i, 1] > max_vertex_id:
            max_vertex_id = arr[i, 1]
        # Compare values in column 2
        if arr[i, 2] > max_vertex_id:
            max_vertex_id = arr[i, 2]

    return max_vertex_id


cdef uint32_t find_max_eid(uint32_t[:, :] arr):
    """
    Find the maximum edge id in the edges array.
    """
    cdef uint32_t max_eid = 0
    cdef size_t n_rows = arr.shape[0]

    # Unrolling the loop for column 0
    cdef size_t i
    for i in range(n_rows):
        if arr[i, 0] > max_eid:
            max_eid = arr[i, 0]

    return max_eid


cpdef object find_degree2_branches(
    uint32_t[:, :] edges_array,
    uint32_t[:] end_edge_ids,
    uint8_t[:] vertex_degs,
    int print_step=10
):
# cpdef object find_degree2_branches(
#     np_index_array_t  edges_array,
#     np_index_array_t  end_edge_ids,
#     np_degree_array_t vertex_degs,
#     int print_step = 10
# ):
    """
    Find chains formed by degree-2 vertices. Returns a list of tuples:
      (edge_ids_list, vertex_ids_list)

    .. warning::
        We make 3 important assumptions:
        - there are no pure degree2 loops in the graph
        (i.e. the graph is a largest component)
        - the vertices in the graph are monotonic with unit step (from 0)
        - the edge ids are reindexed prior to this function

    Steps:
    - Compress vertex IDs and build adjacency.
    - Start from edges that connect a degree-2 vertex to a non-degree-2 vertex.
    - Follow chains from these edges.
    """
    cdef list result = [], null_result = []
    cdef size_t num_edges = edges_array.shape[0]
    cdef vertex_t max_vertex_id = find_max_v_id(edges_array)  # FIXME: should be vertex_degs.shape[0] - 1
    printf(b"Max vertex id = %u\n", max_vertex_id)

    # Build adjacency
    cdef vector[vector[adj_entry_t]] adjacency
    # noinspection PyUnresolvedReferences
    adjacency.resize(max_vertex_id+1)
    printf(<const char *> b"Adjacency created\n")

    cdef size_t e  # edge counter
    cdef index_t edge_id
    cdef vertex_t v1, v2
    for e in range(num_edges):
        edge_id = edges_array[e, 0]
        v1 = edges_array[e, 1]
        v2 = edges_array[e, 2]

        adjacency[v1].push_back(adj_entry_t(edge_id, v2))
        adjacency[v2].push_back(adj_entry_t(edge_id, v1))

    printf(<const char *> b"Adjacency computed\n")

    cdef vector[uint8_t] visited_edges  # Store as uint8_t to avoid slow down due to c++ bool memory optimization
    cdef uint32_t max_eid = find_max_eid(edges_array)
    if max_eid > num_edges:
        printf(b"WARNING: max edge id %u > num edges %zu\n", max_eid, num_edges)
        return null_result
    # noinspection PyUnresolvedReferences
    visited_edges.resize(num_edges+1, 0)

    # Step 4: Trace chains
    # Start from edges that have exactly one endpoint degree=2 and the other not,
    # and not visited.
    cdef bint is_start_edge
    printf(b"Starting loop on %zu edges\n", num_edges)
    cdef size_t i = 0, ret_code = 0, n_end_edges = len(end_edge_ids)
    for i, edge_id in enumerate(end_edge_ids):  # We only loop over end edges (i.e. mixed deg2, non deg2)
        if (i % print_step) == 0:
            # print(f'\r{i}/{n_end_edges}', end='')
            printf(b"\r%zu/%zu", i+1, n_end_edges)
            # draw_progress_bar(i, num_edges)
        if not visited_edges[edge_id]:
            ret_code = trace_chain(edge_id, edges_array, vertex_degs, adjacency, visited_edges, result)
            if ret_code != 0:
                printf(b"\nError %zu\n", ret_code)
                return null_result
    printf(b"\r%zu/%zu\n", i+1, n_end_edges)  # End progress bar

    return result


cdef enum REDUCER:
    RED_MIN = 0
    RED_MAX = 1
    RED_SUM = 2
    RED_MEAN = 3

ctypedef unsigned long long  uint64 # For offsets, to avoid overflow issues with large arrays

# helper to map Python function or name -> enum
cdef int get_reducer_enum(object reducer_fn):
    cdef str name = reducer_fn.__name__
    if name == 'amin' or name == 'min':
        return RED_MIN
    elif name == 'amax' or name == 'max':
        return RED_MAX
    elif name == 'sum':
        return RED_SUM
    elif name == 'mean':
        return RED_MEAN
    else:
        return -1    # signal “unknown Python reducer”


cdef inline sink_t c_min(source_t[:] src, sink_t[:] sink, uint64[:] idxs, uint64 start, uint64 end)  nogil except +:
    """Compute the minimum of src over indices in idxs[start:end]."""
    cdef Py_ssize_t j
    cdef uint64 idx
    cdef sink_t acc
    # initialize with first element
    idx = idxs[start]
    acc = <sink_t>src[idx]
    for j in range(start + 1, end):
        idx = idxs[j]
        if src[idx] < acc:
            acc = <sink_t>src[idx]
    return <sink_t>acc

cdef inline sink_t c_max(source_t[:] src, sink_t[:] sink, uint64[:] idxs, uint64 start, uint64 end)  nogil except +:
    """Compute the maximum of src over indices in idxs[start:end]."""
    cdef Py_ssize_t j
    cdef uint64 idx
    cdef sink_t acc
    # initialize with first element
    idx = idxs[start]
    acc = <sink_t>src[idx]
    for j in range(start + 1, end):
        idx = idxs[j]
        if src[idx] > acc:
            acc = <sink_t>src[idx]
    return <sink_t>acc

cdef inline double c_sum(source_t[:] src, sink_t[:] sink, uint64[:] idxs, uint64 start, uint64 end)  nogil except +:
    """Compute the sum of src over indices in idxs[start:end]."""
    cdef Py_ssize_t j
    cdef double acc = 0  # FIXME: float and force casting to source_t
    for j in range(start, end):
        acc = acc + <double>src[idxs[j]]
    return acc

cdef inline sink_t c_mean(source_t[:] src, sink_t[:] sink, uint64[:] idxs, uint64 start, uint64 end)  nogil except +:
    """Compute the mean of src over indices in idxs[start:end]."""
    cdef uint64 length = end - start
    if length <= 0:
        return 0
    cdef double total = c_sum(src, sink, idxs, start, end)
    return <sink_t>(total / length)


cpdef bint cy_reduce(source_t[:] source, sink_t[:] sink, uint64[:] idx_stack, uint64[:] offsets,
                            object reducer_fn, int num_threads=10):
    """
    Parallel reducer over ragged slices of *arr*:

    Parameters
    ----------
    source: np.ndarray
        The array to reduce.
    idx_stack: np.ndarray
        The index stack that defines the slices.
    offsets: np.ndarray
        The offsets that define the slices in *idx_stack*.
    reducer_fn: callable
        The function to apply to each slice.
        Must be a numpy function that can release the GIL, e.g. np.sum, np.mean, np.max, np.min.
    num_threads: int
        The number of threads to use for parallel reduction (in the case of standard Cython reducers).


    .. warning::
        if reduction function is sum, sink must be of type float64


    Note
    ----
    slice i  ==  arr[idx_stack[offsets[i]:offsets[i+1]]]

    Returns
    -------
    bool
        True if the reduction was successful, False if an unknown reducer function was provided.
    """
    cdef:
        Py_ssize_t n_chain = offsets.shape[0] - 1
        Py_ssize_t i
        uint64 start, end
        int function_code = get_reducer_enum(reducer_fn)

    # if function_code == RED_SUM:
    #     if sink.format != 'd':  # FIXME: check if this is correct
    #         raise ValueError(f"Sink array must be of type 'd' (float64) for sum reduction, got {sink.format}.")

    if function_code == -1:  # use Python reducer
        warnings.warn(f'Unknown reducer function {reducer_fn}. Using Python fallback.'
                      f'This is typically significantly slower', UserWarning)
        return False
    else:
        with nogil, parallel(num_threads=num_threads):
            for i in prange(n_chain, schedule='static'):
                start = offsets[i]
                end = offsets[i + 1]
                if function_code == RED_SUM:  # Sum needs upcasting to float
                    sink[i] = <sink_t>c_sum(source, sink, idx_stack, start, end)
                elif function_code == RED_MEAN:  # Mean needs upcasting to float
                    sink[i] = c_mean(source, sink, idx_stack, start, end)
                elif function_code == RED_MIN:  # Min needs upcasting to source_t
                    sink[i] = c_min(source, sink, idx_stack, start, end)
                elif function_code == RED_MAX:  # Max needs upcasting to source_t
                    sink[i] = c_max(source, sink, idx_stack, start, end)
                else:  # Unknown function code, use Python fallback
                    return False
        return True
