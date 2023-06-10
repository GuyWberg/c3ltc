import numpy
from numba import jit


class Graphs:
    @staticmethod
    def neighbours_list_to_adj_matrix(neighbours_list):
        """
        Converts the mapping vertices->neighbours to adjacency matrix.
        """
        return neighbours_list_to_adj_matrix(neighbours_list)

    @staticmethod
    def get_expansion(adj):
        """
        Returns the expansion of the graph (the value of the largest normalized eigenvalue different than 1).
        """
        sorted_eigenvalues = Graphs.get_eigenvalues(adj)
        assert numpy.isclose(sorted_eigenvalues[-1], 1, rtol=1e-05, atol=1e-08, equal_nan=False)
        gap = max(abs(sorted_eigenvalues[-2]), abs(sorted_eigenvalues[1]))
        if not numpy.isclose(sorted_eigenvalues[0], -1, rtol=1e-05, atol=1e-08, equal_nan=False):
            gap = max(gap, abs(sorted_eigenvalues[0]))
        return gap

    @staticmethod
    def get_eigenvalues(adj):
        """
        Returns normalized eigenvalues of the graph.
        """
        sorted_eigenvalues = numpy.sort(numpy.linalg.eigvals(adj))
        return numpy.real(sorted_eigenvalues / sorted_eigenvalues[adj.shape[0] - 1])


@jit(nopython=True, cache=True)
def neighbours_list_to_adj_matrix(neighbours_list):
    adj = numpy.zeros((len(neighbours_list), len(neighbours_list)))
    for v in neighbours_list:
        for (i, j) in enumerate(neighbours_list[v]):
            adj[v][j] = 1
    return adj
