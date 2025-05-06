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

- From the degree-2 vertex (v1), follow connected edges through other degree-2 vertices:
    Find the next edge out of v1 that isn't visited and leads to another vertex.
    Continue until you reach a vertex with degree ≠ 2 or no unvisited edges remain.

Append the resulting chain to result.
"""
from libc.stdio cimport printf
from libc.stdint cimport uint32_t, uint8_t

from libcpp.vector cimport vector
from libcpp.utility cimport pair

import numpy as np
cimport numpy as cnp

ctypedef uint32_t index_t
ctypedef vector[index_t] index_vector_t

ctypedef uint32_t vertex_t
ctypedef vector[vertex_t] vertex_vector_t

ctypedef pair[vertex_t, vertex_t] edge_t
ctypedef vector[edge_t] edge_vector_t

ctypedef uint8_t degree_t
ctypedef vector[degree_t] degree_vector_t

ctypedef np.uint32_t np_vertex_t
ctypedef np.uint8_t np_degree_t
ctypedef np.uint32_t[::1] np_vertex_array_t
ctypedef np.uint32_t[::1] np_index_array_t
ctypedef np.uint8_t[::1] np_degree_array_t

# Define a sentinel value for uninitialized degrees
cdef degree_t UN_INITIALIZED_DEGREE = 0xFF  # Replaces literal (type_max(degree_t) if unsigned, use -1 if signed
cdef index_t INVALID_IDX = 0xFFFFFFFF  # Replaces literal (type_max(index_t) if unsigned, use -1 if signed



# # Import C++ std::array
# cdef extern from "<array>" namespace "std":
#     cdef cppclass array[T, size_t]:
#         T& operator[](size_t)  # For indexing
#         size_t size()  # To get the size of the array


# FIXME: adjacency is actually build on edge_id, neigbour_vertex pairs
#   adjacency[u].push_back(edge_t(i, w))
#   adjacency[w].push_back(edge_t(i, u))
#   so that for vertex[u] we have the edge_id and the neighbour_vertex w

cdef edge_t find_endpoint_vertex(
    vertex_t v1, vertex_t v2, degree_t deg_v1, degree_t deg_v2
):
    """
    Given two vertices and their degrees, return the endpoint vertex.
    """
    if deg_v1 == 2 and deg_v2 != 2:
        return edge_t(v2, v1)
    elif deg_v2 == 2 and deg_v1 != 2:
        return edge_t(v1, v2)
    else:
        return edge_t(INVALID_IDX, INVALID_IDX)

# A helper function to follow a chain starting from a specific edge that connects
# a degree-2 vertex and a non-degree-2 vertex.
cdef int trace_chain(
    index_t start_edge_idx,
    uint32_t[:, :] connectivity,
    uint8_t[:] vertex_degs,
    vector[vector[pair[index_t, vertex_t]]]& adjacency,
    vector[bint]& visited_edges,
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
    if endpoint_vertex == INVALID_IDX:
        # print(f"{start_edge_idx} is not a valid start edge (degrees = {deg_v1}, {deg_v2}\n")
        printf(b"%d is not a valid start edge (degrees = %d, %d)\n", start_edge_idx, deg_v1, deg_v2)
        return 1  # INVALID edge
    # TODO: assert that neither vertex is INVALID_IDX

    visited_edges[e] = 1

    # Initialize chain with this edge and its vertices
    cdef index_vector_t chain_edges
    cdef vertex_vector_t chain_verts

    chain_edges.push_back(e)
    # Order: endpoint_vertex (non-degree-2), deg2_vertex, then follow chain
    chain_verts.push_back(endpoint_vertex)
    chain_verts.push_back(deg2_vertex)

    cdef vertex_t current_vertex = deg2_vertex
    cdef vertex_t prev_vertex = endpoint_vertex

    # Follow through degree-2 vertices
    cdef size_t n_neighbours = 0
    cdef vertex_t next_vertex = 0, neighbour_vertex = 0
    cdef index_t next_edge_id = 0, eid = 0

    cdef size_t i = 0
    while True:
        if vertex_degs[current_vertex] != 2:
            # Reached a non-degree-2 vertex or can't continue
            # print(f'\tVertex {current_vertex} is not degree-2')
            break

        # Find next edge from current_vertex not visited and not prev_vertex
        n_neighbours = adjacency[current_vertex].size()
        if n_neighbours != 2:
            printf(b"ERROR: Vertex %d should be degree 2 but has %zu neighbours\n",
                   current_vertex, n_neighbours)
            return 2  # invalid degree-2 vertex
        next_edge_id = INVALID_IDX
        next_vertex = INVALID_IDX
        for i in range(n_neighbours):
            eid = adjacency[current_vertex][i].first
            neighbour_vertex = adjacency[current_vertex][i].second
            if neighbour_vertex != prev_vertex and not visited_edges[eid]:
                next_edge_id = eid
                next_vertex = neighbour_vertex
                break
        # print(f'\tneighbour: {neighbour_vertex}')

        if next_edge_id == INVALID_IDX:
            # No way forward
            break

        # Continue the chain
        visited_edges[next_edge_id] = True
        chain_edges.push_back(next_edge_id)
        chain_verts.push_back(next_vertex)

        prev_vertex = current_vertex
        current_vertex = next_vertex

    # Append the chain to result
    to_python(chain_edges, chain_verts, connectivity[:, 0], result)
    return 0


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

    cdef list py_verts = []
    for j in range(full_verts.size()):
        py_verts.append(full_verts[j])

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
    int print_step = 10
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
    cdef vertex_t max_vertex_id = find_max_v_id(edges_array)  # FIXME: should be len(vertex_degs) - 1
    print(f'Max vertex id = {max_vertex_id}')

    cdef vector[vector[pair[index_t, vertex_t]]] adjacency
    adjacency.resize(max_vertex_id + 1)
    print('Adjacency created')

    cdef size_t e  # edge counter
    cdef size_t num_edges = edges_array.shape[0]
    cdef index_t edge_id
    cdef vertex_t v1, v2
    # For each edge in edges_array:
    # edges_array: (edge_id, v1, v2)
    for e in range(num_edges):
        edge_id = edges_array[e, 0]
        v1 = edges_array[e, 1]
        v2 = edges_array[e, 2]

        adjacency[v1].push_back(pair[index_t, vertex_t](edge_id, v2))
        adjacency[v2].push_back(pair[index_t, vertex_t](edge_id, v1))

    print('Adjacency computed')

    cdef vector[uint8_t] visited_edges  # Store as uint8_t to avoid slow down due to c++ bool memory optimization
    cdef uint32_t max_eid = find_max_eid(edges_array)
    if max_eid > num_edges:
        print(f'WARNING: max edge id {max_eid} > num edges {num_edges}')
        return []
    visited_edges.resize(num_edges + 1, 0)

    cdef list result = []

    # Step 4: Trace chains
    # Start from edges that have exactly one endpoint degree=2 and the other not,
    # and not visited.
    cdef bint is_start_edge
    cdef degree_t deg_v1, deg_v2
    cdef size_t i = 0
    print(f'Starting loop on {num_edges} edges')
    cdef size_t n_end_edges = len(end_edge_ids)
    cdef int ret_code = 0
    for i, edge_id in enumerate(end_edge_ids):  # We only loop over end edges (i.e. mixed deg2, non deg2)
        if (i % print_step) == 0:
            # print(f'\r{i}/{n_end_edges}', end='')
            printf(b"\r%d/%d", i, n_end_edges)
            # draw_progress_bar(i, num_edges)
        if not visited_edges[edge_id]:
            ret_code = trace_chain(edge_id, edges_array, vertex_degs, adjacency, visited_edges, result)
            if ret_code != 0:
                return []
    printf(b'\n')  # End progress bar

    return result
