import numpy
from numba import njit, jit
from numba.core import types
from numba.typed import Dict

matrix_cell_index = types.UniTuple(types.int64, 2)
code_alphabet = types.int64

@jit(nopython=True, cache=True)
def embedding_local_parity_constraints_on_squares(vertex_to_squares, edges_to_gens_A, edges_to_gens_B, local_parity_A, local_parity_B):
    """
    Returns a representation of the parity check matrix of c3LTC, and the number of constraints (i.e. the number of rows in the parity check matrix).

    Keyword arguments:
    vertex_to_squares - vertex -> matrix of square ids.
    edges_to_gens_A - A set of tuples, e=(v1,v2) -> k, v1,v2 indices of vertices, A_k * v1 = v2
    edges_to_gens_B - A set of tuples, e=(v1,v2) -> k, v1,v2 indices of vertices, v1 * B_k = v2
    local_parity_A - A matrix, corresponding to the parity check matrix of the local code on the A-side.
    local_parity_B - A matrix, corresponding to the parity check matrix of the local code on the B-side.

    """
    print("[*] Start constraints resolution")
    local_parity_num_rows_A = local_parity_A.shape[0]
    local_parity_num_rows_B = local_parity_B.shape[0]
    global_parity = Dict.empty(key_type=matrix_cell_index, value_type=code_alphabet)
    row_count = 0
    for e in edges_to_gens_A:
        v = e[0]
        k = edges_to_gens_A[e]
        row_of_squares = vertex_to_squares[v][k, :]
        for i in range(local_parity_num_rows_A):
            for j, s in enumerate(row_of_squares):
                global_parity[(row_count, s)] = local_parity_B[i][j]
            row_count += 1
    for e in edges_to_gens_B:
        v = e[0]
        k = edges_to_gens_B[e]
        row_of_squares = vertex_to_squares[v][:, k]
        for i in range(local_parity_num_rows_B):
            for j, s in enumerate(row_of_squares):
                global_parity[(row_count, s)] = local_parity_A[i][j]
            row_count += 1
    print("[*] End constraints resolution")
    assert row_count == len(edges_to_gens_B) * local_parity_num_rows_A + len(edges_to_gens_A) * local_parity_num_rows_B
    return global_parity
