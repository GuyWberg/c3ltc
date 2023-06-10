import random
import numpy
from numba import jit
from numba.core import types
from numba.typed import Dict

@jit(nopython=True, cache=True)
def get_index_of_inverse(gens, i):
    """
    Returns the index of inverse(gens[i]).
    
    Keyword arguments:
    gens - a list of group elements generators.
    i - an index.
    """
    if gens[i].is_non_identity_order_2():
        return i
    
    # this assumes that the gens are ordered such that the generators on each pair of indices (e.g. (0,1),(2,3)..) are inverses
    i_inverse = i + 1
    if i % 2 == 1:
        i_inverse = i - 1
    return i_inverse

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

@jit(nopython=True, cache=True)
def get_square_id(squares, square):
    """
    Returns the id of a given square.
    
    Keyword arguments:
    squares - indices corresponding to squares.
    square - a tuple of four indices of vertices.
    """
    if square in squares:
        return squares[square]
    return -1

square_type = types.UniTuple(types.int64, 3)
square_vertices_type = types.UniTuple(types.int64, 4)
array_of_vertices = types.int64[:]
array_of_edge_ids = types.int64[:]
matrix_of_square_ids = types.int64[:, :]
vertex_pair = types.UniTuple(types.int64, 2)
square_id_type = types.int64
vertex_type = types.int64
counter_type = types.int64
gen_id = types.int64

def create_lr_cayley_graph_new(group_elements, vertices, gens_A, gens_B):
    """
    Returns a c3LTC defined by the following mappings:
        vertex_to_squares - vertex -> matrix of square ids.
        square_counter - the number of squares.
        square_to_vertices - a mapping of square indices to 4 vertices.
        vertex_to_neighbours_B - a mapping of vertex id (label) to an array of neighbours it has by applying B gens.
        vertex_to_neighbours_A - a mapping of vertex id (label) to an array of neighbours it has by applying A gens.
        edges_A - a set of 2-tuples of A edges (an edge is a tuple of 2 vertices).
        edges_B - a set of 2-tuples of B edges (an edge is a tuple of 2 vertices).

    A square is mapped to a triplet of group elements: (a,g,b), such that a is in gens_A, b is in gens_B and the vertices in that square are: (g, ag, agb, gb).
    
    Keyword arguments:
    group_elements - list of group elements.
    vertices - indices corresponding to vertices (group_elements[i] corresponds to vertices[i]).
    gens_A - list of generating elements.
    gens_B - list of generating elements.
    """
    squares = Dict.empty(key_type=square_type, value_type=square_id_type)
    group_size = len(group_elements)
    vertex_to_squares = Dict.empty(key_type=vertex_type, value_type=matrix_of_square_ids)
    vertex_to_neighbours_A = Dict.empty(key_type=vertex_type, value_type=array_of_vertices)
    vertex_to_neighbours_B = Dict.empty(key_type=vertex_type, value_type=array_of_vertices)
    square_to_vertices = Dict.empty(key_type=square_id_type, value_type=array_of_vertices)
    edges_A = Dict.empty(key_type=vertex_pair, value_type=gen_id)
    edges_B = Dict.empty(key_type=vertex_pair, value_type=gen_id)
    
    # init empty placeholder arrays to be filled
    for ka in range(group_size):
        vertex_to_squares[ka] = numpy.array([[0] * len(gens_B)] * len(gens_A))
        vertex_to_neighbours_A[ka] = numpy.array([0] * len(gens_A))
        vertex_to_neighbours_B[ka] = numpy.array([0] * len(gens_B))
    
    for s in range(int(group_size * len(gens_A) * len(gens_B) / 4)):
        square_to_vertices[s] = numpy.array([0] * 4)
    
    square_counter = 0
    print("[*] Start square resolution")
    
    for g in group_elements:
        v = get_id(vertices, g)
        
        # fill neighbouring vertices for every vertex by applying A,B generators
        for ka in range(len(gens_A)):
            ag = gens_A[ka] * g
            av = get_id(vertices, ag)
            if v < av:
                edge = (v, av)
                edges_A[edge] = ka
            vertex_to_neighbours_A[v][ka] = av
        
        for kb in range(len(gens_B)):
            gb = g * gens_B[kb]
            vb = get_id(vertices, gb)
            if v < vb:
                edge = (v, vb)
                edges_B[edge] = kb
            vertex_to_neighbours_B[v][kb] = vb
        
        # resolve squares
        for ka in range(len(gens_A)):
            for kb in range(len(gens_B)):
                # get a unique (a,g,b) triplet for this square
                (a1, g1, b1,) = get_square_unique_representation(gens_A[ka], g, gens_B[kb], vertices)
                v1 = get_id(vertices, g1)
                square = (gens_A.index(a1), v1, gens_B.index(b1))
                
                # calculate the vertices in the square
                a1v1 = get_id(vertices, a1 * g1)
                a1v1b1 = get_id(vertices, a1 * g1 * b1)
                v1b1 = get_id(vertices, g1 * b1)
                square_vertices = (v1, a1v1, a1v1b1, v1b1)
                
                # Proceed checks if:
                # 1) The square is not already in squares.
                # 2) If TNC doesn't hold, it might be that v1 == a1v1b1, so in that case the square looks like S=(g,ag,g,gb).
                #    If it is the case that for some previous generators a',b', we already saw the square S'=(g,a'g,g,gb') = (g,gb,g,ga)
                #    we shouldn't process S, and we should proceed with the index of S'.
                #    Therefore we check if we already saw the square (a',g,b') for a'=gbg^(-1) and b'=g^(-1)ag.
                square_id = get_square_id(squares, square)
                if square_id == -1 and v1 == a1v1b1:
                    S_ = (gens_A.index(g1 * b1 * g1.invert()), v1, gens_B.index(g1.invert() * a1 * g1))  # S'
                    square_id = get_square_id(squares, S_)
                
                if square_id == -1:
                    square_id = square_counter
                    squares[square] = square_id
                    add_square_to_maps(square_vertices, square_id, square_to_vertices)
                    square_counter += 1
                
                assert square_id != -1
                vertex_to_squares[v][ka, kb] = square_id
    
    print("[*] Finished square resolution")
    assert square_counter == len(gens_A) * len(gens_B) * len(group_elements) / 4
    
    return (
        vertex_to_squares,
        square_to_vertices,
        vertex_to_neighbours_B,
        vertex_to_neighbours_A,
        edges_A,
        edges_B,
    )

@jit(nopython=True, cache=True)
def add_square_to_maps(square_vertices, square_id, square_to_vertices):
    """
    Adds the new squares to graph mappings: square_to_vertices.

    Keyword arguments:
    closing_label - vertex index.
    A_neighbour_label - vertex index.
    B_neighbour_label - vertex index.
    square_label - square index.
    square_to_vertices - a mapping of squares indices to 4 vertices.
    vertex_label - vertex index.
    """
    v = square_vertices[0]
    av = square_vertices[1]
    avb = square_vertices[2]
    vb = square_vertices[3]
    square_to_vertices[square_id][0] = v
    square_to_vertices[square_id][1] = av
    square_to_vertices[square_id][2] = avb
    square_to_vertices[square_id][3] = vb


# @jit(nopython=True, cache=True)
def get_square_unique_representation(a, g, b, vertices):
    """
    Returns a unique triplet of group elements representing the square (a,g,b).

    Keyword arguments:
    a - group element.
    g - group element.
    b - group element.
    vertices - hash(group element) -> vertex id
    """
    a_inv = a.invert()
    b_inv = b.invert()
    ag = a * g
    agb = a * g * b
    gb = g * b
    v_g = get_id(vertices, g)
    v_ag = get_id(vertices, ag)
    v_agb = get_id(vertices, agb)
    v_gb = get_id(vertices, gb)
    min_vertex_index = min(v_g, v_ag, v_agb, v_gb)
    if min_vertex_index == v_g:
        return a, g, b
    elif min_vertex_index == v_ag:
        return a_inv, ag, b
    elif min_vertex_index == v_agb:
        return a_inv, agb, b_inv
    elif min_vertex_index == v_gb:
        return a, gb, b_inv
