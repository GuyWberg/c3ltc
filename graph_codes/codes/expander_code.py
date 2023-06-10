import os

import numpy

from local_codes.linear_code import LinearCode
from fast_linear_algebra.row_reduce import row_reduce_and_orthogonal
from graph_codes.on_edge_code import embedding_local_parity_constraints_on_edges
from graphs.cayley import CayleyGraph
import shutil

from datetime import datetime


class ExpanderCode:
    def __init__(self, graph: CayleyGraph, small_code: LinearCode, test=0):
        super().__init__()
        self.graph = graph
        self.small_code = small_code
        self.prime = self.small_code.prime
        self.name = (
            "ExpanderGraph q="
            + str(self.prime)
            + " C_0={n_0="
            + str(small_code.generator.shape[1])
            + " k0="
            + str(small_code.generator.shape[0])
            + "} Graph={"
            + self.graph.name
            + "} lambda2="
            + str(graph.get_lambda2())[:4]
        )
        print("[*] Started generating code")
        matrix_cell_to_value = embedding_local_parity_constraints_on_edges(
            self.graph.vertex_to_edges, self.small_code.parity
        )
        row_counter = len(self.graph.vertex_to_edges) * self.small_code.parity.shape[0]
        if test:
            self.generator_path = row_reduce_and_orthogonal(
                matrix_cell_to_value, self.prime, row_counter, self.graph.number_of_edges, 1
            )
            with open(self.generator_path) as f:
                first_line = f.readline()
            fl_values = first_line.split(" ")
            self.k = int(fl_values[0])
            self.n = int(fl_values[1])
        else:
            self.generator, self.parity = row_reduce_and_orthogonal(
                matrix_cell_to_value, self.prime, row_counter, self.graph.number_of_edges
            )
            self.k = self.generator.shape[0]
            self.n = self.generator.shape[1]
        self.rate = self.k / self.n
        self.name += " k=" + str(self.k) + " n=" + str(self.n) + " rate=" + str(self.rate)[:6]
        self.save_code(test)
        print("[*] Finished generating code")

    def save_code(self, test=0):
        now = datetime.now()
        self.name += " " + str(now)
        if not os.path.isdir("./concrete_codes/" + self.name):
            os.mkdir("./concrete_codes/" + self.name)
            if test:
                shutil.move(
                    self.generator_path, "./concrete_codes/" + self.name + "/generator_matrix.txt"
                )
            else:
                numpy.savetxt(
                    "./concrete_codes/" + self.name + "/parity_check.txt",
                    self.parity,
                    fmt="%i",
                    newline="\n",
                )
                numpy.savetxt(
                    "./concrete_codes/" + self.name + "/generator_matrix.txt",
                    self.generator,
                    fmt="%i",
                    newline="\n",
                )
        self.graph.save_graph("./concrete_codes/" + self.name)
        self.small_code.save_code("./concrete_codes/" + self.name)
        test_type = ""
        if self.small_code.name[0:2] == "RS":
            test_type = "RS"
        else:
            test_type = "CycleCode"
        with open("./concrete_codes/log.txt", "a") as file:
            file.write(
                "ecode"
                + " "
                + str(self.prime)
                + " "
                + str(self.small_code.generator.shape[1])
                + " "
                + str(self.small_code.generator.shape[0])
                + " "
                + self.graph.group.name
                + " "
                + str(self.small_code.generator.shape[1])
                + " "
                + str(self.graph.get_lambda2())
                + " "
                + str(self.k)
                + " "
                + str(self.n)
                + " "
                + str(test_type)
                + " "
                + str(self.rate)
                + " "
                + str(now)
                + "\n"
            )
