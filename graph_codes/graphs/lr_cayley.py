import os
import pickle
import galois
import numpy
from graphs.graph_utils import Graphs, neighbours_list_to_adj_matrix
from graphs.lr_cayley_utils import create_lr_cayley_graph_new
from fast_linear_algebra.row_reduce import row_reduce_and_orthogonal
from numba.typed import List


class LeftRightCayleyGraph:
    def __init__(self, group, gens_A, gens_B):
        self.gens_A = gens_A
        self.gens_B = gens_B
        self.group = group
        self.name = (
            "LeftRightCayleyGraph "
            + self.group.name
            + " |A|="
            + str(len(self.gens_A))
            + " |B|="
            + str(len(self.gens_B))
        )
        file_name = "./concrete_graphs/" + self.name
        self.number_of_vertices = self.group.elements
        (
            self.vertex_to_squares,
            self.square_to_vertices,
            self.vertex_to_neighbours_right,
            self.vertex_to_neighbours_left,
            self.edges_left,
            self.edges_right,
        ) = create_lr_cayley_graph_new(
            self.group.elements,
            self.group.labels,
            gens_A,
            gens_B,
        )
        self.number_of_squares = int(
            len(gens_A) * len(gens_B) * len(self.group.elements) / 4
        )
        self.lambda2 = max(
            Graphs.get_expansion(neighbours_list_to_adj_matrix(self.vertex_to_neighbours_right)),
            Graphs.get_expansion(neighbours_list_to_adj_matrix(self.vertex_to_neighbours_left)),
        )

    def get_lambda2(self):
        return self.lambda2

    def get_eigenvalues_A(self):
        return Graphs.get_eigenvalues(
            neighbours_list_to_adj_matrix(self.vertex_to_neighbours_right)
        )

    def get_eigenvalues_B(self):
        return Graphs.get_eigenvalues(neighbours_list_to_adj_matrix(self.vertex_to_neighbours_left))

    def is_bipartite_A(self):
        return numpy.isclose(
            1 - abs(self.get_eigenvalues_A()[0]),
            0,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=False,
        )

    def is_bipartite_B(self):
        return numpy.isclose(
            1 - abs(self.get_eigenvalues_B()[0]),
            0,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=False,
        )

    def save_graph(self, path=""):
        if path != "":
            path = path + "/"
        else:
            path = "./concrete_graphs/"
        if not os.path.isdir(path + "/" + self.name):
            os.mkdir(path + self.name)

        with open(path + self.name + "/vertex_to_squares", "wb") as file:
            pickle.dump(
                dict(self.vertex_to_squares),
                file,
                pickle.HIGHEST_PROTOCOL,
            )
        with open(path + self.name + "/square_to_vertices", "wb") as file:
            pickle.dump(
                dict(self.square_to_vertices),
                file,
                pickle.HIGHEST_PROTOCOL,
            )
        with open(path + self.name + "/vertex_to_neighbours_right", "wb") as file:
            pickle.dump(
                dict(self.vertex_to_neighbours_right),
                file,
                pickle.HIGHEST_PROTOCOL,
            )
        with open(path + self.name + "/edges_left", "wb") as file:
            pickle.dump(
                dict(self.edges_left),
                file,
                pickle.HIGHEST_PROTOCOL,
            )
        with open(path + self.name + "/edges_right", "wb") as file:
            pickle.dump(
                dict(self.edges_right),
                file,
                pickle.HIGHEST_PROTOCOL,
            )
        numpy.savetxt(
            path + self.name + "/generators_A",
            numpy.array([g.value for g in self.gens_A], dtype=int),
            fmt="%i",
            delimiter=",",
        )
        numpy.savetxt(
            path + self.name + "/generators_B",
            numpy.array([g.value for g in self.gens_B], dtype=int),
            fmt="%i",
            delimiter=",",
        )
        numpy.savetxt(
            path + self.name + "/eigenvalues_A",
            self.get_eigenvalues_A(),
            fmt="%f",
            delimiter=",",
        )
        numpy.savetxt(
            path + self.name + "/eigenvalues_B",
            self.get_eigenvalues_B(),
            fmt="%f",
            delimiter=",",
        )