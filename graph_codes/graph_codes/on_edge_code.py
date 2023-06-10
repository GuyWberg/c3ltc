import numpy
from numba import jit
from numba.core import types
from numba.typed import Dict

matrix_cell_index = types.UniTuple(types.int64, 2)
code_alphabet = types.int64

# @jit(nopython=True, cache=True)
def embedding_local_parity_constraints_on_edges(vertex_to_edges, local_parity):
    """
    Returns a representation of the parity check matrix of an expander code, and the number of constraints (i.e. the number of rows in the parity check matrix).

    Keyword arguments:
    vertex_to_edges - A mapping between the id of a vertex to an array of indices of edges.
    local_parity - A matrix, corresponding to the parity check matrix constraints that is enforced around each vertex.
    """
    print("[*] Start constraints resolution")
    local_parity_num_rows = local_parity.shape[0]
    global_parity = Dict.empty(key_type=matrix_cell_index, value_type=code_alphabet)
    row_count = 0
    for v in vertex_to_edges:
        outgoing_edges = vertex_to_edges[v]
        for i in range(local_parity_num_rows):
            for j, edge in enumerate(outgoing_edges):  # outgoing_edges[j] = edge
                global_parity[(row_count, edge)] = local_parity[i][j]
            row_count += 1
    print("[*] End constraints resolution")
    assert row_count == len(vertex_to_edges) * local_parity_num_rows
    return global_parity
