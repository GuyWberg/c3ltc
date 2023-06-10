import numpy
from numba import jit
from numba.core import types

from local_codes.linear_code_utils import *
from codes.tensor_code_utils import *

values_type = types.int64[:, :]
vertex_label_type = types.int64


#@jit(nopython=True, cache=True)
def get_local_view_matrix_values(vertex_to_squares, vertex, noisy_word):
    """
    Returns the local tensor-word that a vertex sees on the squares around it. 
    
    Keyword arguments:
    vertex_to_squares - vertex -> matrix of square ids.   
    vertex - the id (number) of a group element (vertex) in the graph. 
    noisy_word - a (possibly noisy) string of that assign values to squares according to the square-code construction. 
    """
    square_ids = vertex_to_squares[vertex]
    num_rows, num_cols = len(square_ids), len(square_ids[0])
    M = numpy.zeros((num_rows, num_cols), dtype=int)
    
    for i in range(num_rows):
        for j in range(num_cols):
            square_id = vertex_to_squares[vertex][i][j]
            M[i][j] = noisy_word[square_id]
    
    return M

#@jit(nopython=True, cache=True)
def get_edges_around_square(vertices_around_square, edges_A, edges_B):
    """
    Returns two arrays of tuples of two vertices (an edge) - the first tuple is the A-edges of the square corresponding and the second to the B-edges. 
    
    Keyword arguments:
    vertex_to_squares - A mapping between the id of a vertex to a matrix corresponding to the indices of the squres it sees in its local tensor-code word.   
    edges_to_gens_A - A set of tuples, e=(v1,v2) for the A-edges.
    edges_to_gens_B - A set of tuples, e=(v1,v2) for the B-edges.
    """
    v, va, closing, vb = vertices_around_square
    square_edges_A = [(min(v, va), max(v, va)), (min(vb, closing), max(vb, closing))]
    square_edges_B = [(min(va, closing), max(va, closing)), (min(v, vb), max(v, vb))]
    return square_edges_A, square_edges_B


# #@jit(nopython=True, cache=True)
def decode_along_the_edges(noisy_word, vertex_to_squares, square_to_vertices, edges_A, edges_B,
                           local_generator_A, local_parity_check_A,
                           local_generator_B, local_parity_check_B, prime, syndromes_A=None,
                           syndromes_B=None,distance_A = None, distance_B = None):
    """
    Returns the corrected tensor codeword. 
    
    Keyword arguments:
    noisy_word - a (possibly noisy) string of that assign values to squares according to the square-code construction.
    vertex_to_squares - vertex -> matrix of square ids.   
    square_to_vertices - a mapping of squres indices to 4 vertices. 
    edges_to_gens_A - A set of tuples, e=(v1,v2) for the A-edges.
    edges_to_gens_B - A set of tuples, e=(v1,v2) for the B-edges.
    local_generator_A - a matrix, corresponding to a generator matrix of code_A enforced on the B-edges.
    local_parity_A - a matrix, corresponding to a parity check matrix of code_A enforced on the B-edges.
    local_generator_B - a matrix, corresponding to a generator matrix of code_B enforced on the A-edges.
    local_parity_B - a matrix, corresponding to a parity check matrix of code_B code enforced on the A-edges. 
    syndromes_A - syndrom of local word in code_A -> the corresponding error vector. 
    syndromes_B - syndrom of local word in code_B -> the corresponding error vector. 
    distance_A - distance of code_B. 
    distance_B - distance of code_A. 
    prime - the prime associated with the underlying field.  
    """
    noisy_word = numpy.copy(noisy_word) # copy the original values to avoid changing the input
    past_states = [] # during the algorithm, the value of noisy_word is going to be updated, this array stores a snapshot of the matrix before a decoding iteration, to make sure thw algorithm won't get stuck
    suspect_edges_A = edges_A # the set of A-edges the algorithms is going to decode
    suspect_edges_B = edges_B # the set of B-edges the algorithms is going to decode
    h = hash(str(noisy_word))
    while (len(suspect_edges_A) != 0 or len(suspect_edges_B) != 0) and h not in past_states:
        past_states.append(h)
        new_suspect_edges_A = set([])
        new_suspect_edges_B = set([])
        for e in suspect_edges_A:
            v = e[0]
            k = edges_A[e] # e[0]*A_k = e[1]
            local_square_ids = vertex_to_squares[v][k, :]
            local_square_values = numpy.array([noisy_word[s] for s in local_square_ids])
            corrected_local_square_values = decode(local_generator_A, local_parity_check_A, local_square_values, prime,
                             syndromes_A)
            for i, square_id in enumerate(local_square_ids): # local_square_ids[i] = square_id
                if noisy_word[square_id] != corrected_local_square_values[i]: # if value on that square needs to change after decoding
                    e_A, e_B = square_to_edges(square_to_vertices, edges_A, edges_B, local_square_ids, i)
                    for e_a in e_A: # add the other A edge of this square to the new set of A-edges for iteration
                        if e_a != e:
                            new_suspect_edges_A.add(e_a)
                    for e_b in e_B: # add the two B edges to the new set of B-edges for iteration
                            new_suspect_edges_B.add(e_b)
                noisy_word[square_id] = corrected_local_square_values[i] # set the square value to the decoded value 
        for e in suspect_edges_B:
            v = e[0]
            k = edges_B[e]
            local_square_ids = vertex_to_squares[v][:, k]
            local_square_values = numpy.array([noisy_word[s] for s in local_square_ids])
            corrected_local_square_values = decode(local_generator_B, local_parity_check_B, local_square_values, prime,
                             syndromes_B)
            for i, square_id in enumerate(local_square_ids):
                if noisy_word[square_id] != corrected_local_square_values[i]: # if value on that square needs to change after decoding
                    e_A, e_B = square_to_edges(square_to_vertices, edges_A, edges_B, local_square_ids, i)
                    for e_a in e_A: # add the two A edges to the new set of A-edges for iteration
                            new_suspect_edges_A.add(e_a)
                    for e_b in e_B: # add the other B edge of this square to the new set of B-edges for iteration
                        if e_a != e_b:
                            new_suspect_edges_B.add(e_b)
                noisy_word[square_id] = corrected_local_square_values[i] # set the square value to the decoded value 
        suspect_edges_A = set(new_suspect_edges_A) # replace the old set with the new one
        suspect_edges_B = set(new_suspect_edges_B) # replace the old set with the new one
        h = hash(str(noisy_word))
    return noisy_word

def square_to_edges(square_to_vertices, edges_A, edges_B, local_square_ids, i):
    vertices_around_square = square_to_vertices[local_square_ids[i]] 
    square_edges_A, square_edges_B = get_edges_around_square(vertices_around_square, edges_A, edges_B)
    return square_edges_A,square_edges_B


# #@jit(nopython=True, cache=True)
def decode_along_the_vertices(noisy_word, vertex_to_squares, square_to_vertices,
                              local_generator_left, local_parity_check_left,
                              local_generator_right, local_parity_check_right, prime, syndromes_left=None,
                              syndromes_right=None,distance_left = None, distance_right = None):
    """
    Returns the corrected tensor codeword. 
    
    Below we implement a simple decoding along the vertices: we go over all the verices, and correct their local tensor-code view, adding any affected vertices
    to a list of suspect vertices that need to be checked (and possibly corrected), until this list is empty (or we reached an infinite loop). 
    
    Keyword arguments:
    noisy_word - a (possibly noisy) string of that assign values to squares according to the square-code construction.
    vertex_to_squares - vertex -> matrix of square ids.   
    square_to_vertices - a mapping of squres indices to 4 vertices. 
    edges_to_gens_A - A set of tuples, e=(v1,v2) for the A-edges.
    edges_to_gens_B - A set of tuples, e=(v1,v2) for the B-edges.
    local_generator_A - a matrix, corresponding to a generator matrix of code_A enforced on the B-edges.
    local_parity_A - a matrix, corresponding to a parity check matrix of code_A enforced on the B-edges.
    local_generator_B - a matrix, corresponding to a generator matrix of code_B enforced on the A-edges.
    local_parity_B - a matrix, corresponding to a parity check matrix of code_B code enforced on the A-edges. 
    syndromes_A - syndrom of local word in code_A -> the corresponding error vector. 
    syndromes_B - syndrom of local word in code_B -> the corresponding error vector. 
    distance_A - distance of code_B. 
    distance_B - distance of code_A. 
    prime - the prime associated with the underlying field.  
    """
    noisy_word = numpy.copy(noisy_word)
    past_states = [] # during the algorithm, the value of noisy_word is going to be updated, this array stores a snapshot of the matrix before a decoding iteration, to make sure thw algorithm won't get stuck.
    num_vertices = len(vertex_to_squares)
    suspect_vertices = set([i for i in range(num_vertices)])
    h = hash(str(noisy_word)) # h = hash(noisy_word)
    while len(suspect_vertices) != 0 and h not in past_states:
        print("-- by vertex decoding, NNZ in word = ", numpy.count_nonzero(noisy_word))
        past_states.append(h)
        new_suspect_vertices = {0} # numba trick
        new_suspect_vertices.remove(0) # numba trick
        for v in suspect_vertices:
            local_square_ids = vertex_to_squares[v]
            local_square_values = get_local_view_matrix_values(vertex_to_squares, v, noisy_word)
            corrected_local_square_values = tensor_code_decoding(local_square_values, local_generator_left,
                                                                  local_parity_check_left,
                                                                  local_generator_right,
                                                                  local_parity_check_right, prime,
                                                                  syndromes_left,
                                                                  syndromes_right,distance_left , distance_right)
            for i in range(len(corrected_local_square_values)):
                for j in range(len(corrected_local_square_values[0])):
                    if noisy_word[local_square_ids[i][j]] != corrected_local_square_values[i][j]: # if value on that square needs to change after decoding.
                        for v1 in square_to_vertices[local_square_ids[i][j]]: # add all the vertices that this square touches to the new iterating set. 
                            if v != v1:
                                new_suspect_vertices.add(int(v1))
                    noisy_word[local_square_ids[i][j]] = corrected_local_square_values[i][j]
        suspect_vertices = set(new_suspect_vertices) # replace the old set with the new one
        h = hash(str(noisy_word))
    return noisy_word
