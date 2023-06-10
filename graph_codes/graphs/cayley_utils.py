import numpy
from numba import jit
from numba.core import types
from numba.typed import Dict

edge_id_type = types.int64
array_of_vertices = types.int64[:]
array_of_edge_ids = types.int64[:]

edge_type = types.UniTuple(
    types.int64,
    2,
)
square_label_type = types.int64
vertex = types.int64
counter_type = types.int64
neighbour_number = types.int64


@jit(nopython=True, cache=True)
def get_id(vertices, g):
    """
    Returns the vertex corresponding to group element g.

    Keyword arguments:
    vertices - hash(group element) -> vertex id
    g - group element.
    """
    if hash(g) in vertices:
        return vertices[hash(g)]
    return -1


def create_cayley_graph(group_elements, vertices, gens):
    """
    Returns a graph defined by the following mappings:
        vertex_to_edges - vertex -> an array of edge ids.
        vertex_to_neighbours - vertex -> an array of neighbouring vertices.

    Keyword arguments:
    group_elements - list of group elements.
    vertices - hash(group element) -> vertex id.
    gens - list of generating group elements.
    """
    group_size = len(group_elements)
    vertex_to_edges = Dict.empty(
        key_type=vertex,
        value_type=array_of_edge_ids,
    )
    edges_to_ids = Dict.empty(
        key_type=edge_type,
        value_type=edge_id_type,
    )
    vertex_to_neighbours = Dict.empty(
        key_type=vertex,
        value_type=array_of_vertices,
    )
    # init empty placeholder arrays to be filled
    for v in range(group_size):
        vertex_to_edges[v] = numpy.array([0] * len(gens))
        vertex_to_neighbours[v] = numpy.array([0] * len(gens))
    edge_id = 0
    for g in group_elements:
        v = get_id(vertices, g)
        for k in range(len(gens)):
            (a1, g1,) = get_edge_unique_representation(gens[k], g, vertices)
            edge = (
                gens.index(a1),
                get_id(vertices, g1),
            )
            # check if the edge has already appeared
            if edge in edges_to_ids:
                vertex_to_edges[v][k] = edges_to_ids[edge]
            else:
                edges_to_ids[edge] = edge_id
                vertex_to_edges[v][k] = edge_id
                edge_id += 1
    assert edge_id == len(group_elements) * len(gens) / 2
    return vertex_to_edges, vertex_to_neighbours


def get_edge_unique_representation(a, g, vertices):
    """
    Returns a unique triplet of group elements representing the square (a, g, b).

    Keyword arguments:
    a - group element.
    g - group element.
    b - group element.
    vertices - hash(group element) -> vertex id
    """
    a_inv = a.invert()
    ag = a * g
    v_g = get_id(vertices, g)
    v_ag = get_id(vertices, ag)
    min_vertex_index = min(v_g, v_ag)
    if min_vertex_index == v_g:
        return a, g
    elif min_vertex_index == v_ag:
        return a_inv, ag
