import os
import pickle
import numpy
from numba.core import types
from numba.typed import List, Dict
from graphs.graph_utils import Graphs, neighbours_list_to_adj_matrix
from graphs.cayley_utils import create_cayley_graph

edge_id = types.int64
array_type = types.int64[:]
edge_type = types.UniTuple(types.int64, 2)
square_label_type = types.int64
group_element = types.int64
counter_type = types.int64
neighbour_number = types.int64

class CayleyGraph:
    def __init__(self, group, gens):
        self.gens = gens
        self.group = group
        self.name = "CayleyGraph " + self.group.name + " |generators|=" + str(len(self.gens))
        print("[*] Start generating graph")
        self.vertex_to_edges, self.vertex_to_neighbours = create_cayley_graph(
            self.group.elements,
            self.group.labels,
            gens
        )
        self.number_of_edges = int(len(self.group.elements) * len(gens) / 2)
        self.lambda2 = Graphs.get_expansion(neighbours_list_to_adj_matrix(self.vertex_to_neighbours))
        print("[*] Finished generating graph")

    def _load(self, could_be_file):
        with open(could_be_file, "rb") as file:
            objects = pickle.load(file)
        return objects

    def load_from_file(self):
        vertex_to_edges = Dict.empty(key_type=group_element, value_type=array_type)
        edges_to_labels = Dict.empty(key_type=edge_type, value_type=edge_id)
        labels_to_edges = Dict.empty(key_type=edge_id, value_type=edge_type)
        vertex_to_neighbours = Dict.empty(key_type=group_element, value_type=array_type)
        tmp_load = self._load("./concrete_graphs/" + self.name + "/vertex_to_edges")
        for k, v in tmp_load.items():
            vertex_to_edges[k] = v
        tmp_load = self._load("./concrete_graphs/" + self.name + "/vertex_to_neighbours")
        for k, v in tmp_load.items():
            vertex_to_neighbours[k] = v
        return vertex_to_edges, edges_to_labels, labels_to_edges, vertex_to_neighbours

    def save_graph(self, path=""):
        if path != "":
            path = path + "/"
        else:
            path = "./concrete_graphs/"
        if not os.path.isdir(path + "/" + self.name):
            os.mkdir(path + self.name)
        with open(path + self.name + "/vertex_to_edges", "wb") as file:
            pickle.dump(dict(self.vertex_to_edges), file, pickle.HIGHEST_PROTOCOL)
        with open(path + self.name + "/vertex_to_neighbours", "wb") as file:
            pickle.dump(dict(self.vertex_to_neighbours), file, pickle.HIGHEST_PROTOCOL)
        numpy.savetxt(path + self.name + "/generators", numpy.array([g.value for g in self.gens], dtype=int), fmt="%i", delimiter=",")
        numpy.savetxt(path + self.name + "/eigenvalues", self.get_eigenvalues(), fmt="%f", delimiter=",")

    def get_lambda2(self):
        return self.lambda2

    def is_bipartite(self):
        return numpy.isclose(1 - abs(self.get_eigenvalues()[0]), 0, rtol=1e-05, atol=1e-08, equal_nan=False)

    def get_eigenvalues(self):
        return Graphs.get_eigenvalues(neighbours_list_to_adj_matrix(self.vertex_to_neighbours))
