import os
import shutil

import numpy

from codes.c3ltc_utils import (
    decode_along_the_vertices,
    decode_along_the_edges,
    get_local_view_matrix_values,
)
from fast_linear_algebra.row_reduce import row_reduce_and_orthogonal
from graph_codes.on_square_code import embedding_local_parity_constraints_on_squares

from datetime import datetime


class c3LTC:
    def __init__(self, lr_cayley, code_A, code_B, test=0):
        super().__init__()
        self.code_A = code_A
        self.code_B = code_B
        self.lr_cayley = lr_cayley
        self.prime = self.code_A.prime
        self.name = (
            "3cLTC q="
            + str(self.prime)
            + " C_A={n_0="
            + str(code_A.generator.shape[1])
            + " k0="
            + str(code_A.generator.shape[0])
            + "} C_B={n_0="
            + str(code_B.generator.shape[1])
            + " k0="
            + str(code_B.generator.shape[0])
            + "} Graph={"
            + self.lr_cayley.name
            + "} lambda2="
            + str(lr_cayley.get_lambda2())[:4]
        )
        print("[*] Started generating code")
        matrix_cell_to_value = embedding_local_parity_constraints_on_squares(
            self.lr_cayley.vertex_to_squares,
            self.lr_cayley.edges_left,
            self.lr_cayley.edges_right,
            self.code_A.parity,
            self.code_B.parity,
        )
        row_counter = (
            len(self.lr_cayley.edges_left) * self.code_B.parity.shape[0]
            + len(self.lr_cayley.edges_right) * self.code_A.parity.shape[0]
        )
        if test:
            self.generator_path = row_reduce_and_orthogonal(
                matrix_cell_to_value, self.prime, row_counter, lr_cayley.number_of_squares, 1
            )
            with open(self.generator_path) as f:
                first_line = f.readline()
            fl_values = first_line.split(" ")
            self.k = int(fl_values[0])
            self.n = int(fl_values[1])
        else:
            self.generator, self.parity = row_reduce_and_orthogonal(
                matrix_cell_to_value, self.prime, row_counter, lr_cayley.number_of_squares
            )
            self.k = self.generator.shape[0]
            self.n = self.generator.shape[1]
        self.rate = self.k / self.n
        self.name += " k=" + str(self.k) + " n=" + str(self.n) + " rate=" + str(self.rate)[:6]
        self.save_code(test)
        print("[*] Finished generating code")

    def decode_along_the_vertices(self, noisy_word):
        return decode_along_the_vertices(
            noisy_word,
            self.lr_cayley.vertex_to_squares,
            self.lr_cayley.square_to_vertices,
            self.code_A.generator,
            self.code_A.parity,
            self.code_A.generator,
            self.code_A.parity,
            self.prime,
            self.code_A.syndromes,
            self.code_B.syndromes,
            self.code_A.distance,
            self.code_B.distance,
        )

    def decode_along_the_edges(self, noisy_word):
        return decode_along_the_edges(
            noisy_word,
            self.lr_cayley.vertex_to_squares,
            self.lr_cayley.square_to_vertices,
            self.lr_cayley.edges_left,
            self.lr_cayley.edges_right,
            self.code_A.generator,
            self.code_A.parity,
            self.code_A.generator,
            self.code_A.parity,
            self.prime,
            self.code_A.syndromes,
            self.code_B.syndromes,
            self.code_A.distance,
            self.code_B.distance,
        )

    def is_word_in_code(self, word):
        return numpy.count_nonzero(self.parity @ word % self.prime) == 0

    def get_local_view_matrix_values(self, v, word):
        return get_local_view_matrix_values(self.lr_cayley.vertex_to_squares, v, word)

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
        self.lr_cayley.save_graph("./concrete_codes/" + self.name)
        self.code_A.save_code("./concrete_codes/" + self.name)
        self.code_B.save_code("./concrete_codes/" + self.name)

        test_type = ""
        if self.code_A.name[0:2] == "RS":
            test_type = "RS"
        else:
            test_type = "CycleCode"

        with open("./concrete_codes/log.txt", "a") as file:
            file.write(
                "c3ltc"
                + " "
                + str(self.prime)
                + " "
                + str(self.code_A.generator.shape[1])
                + " "
                + str(self.code_A.generator.shape[0])
                + " "
                + self.lr_cayley.group.name
                + " "
                + str(self.code_A.generator.shape[1])
                + " "
                + str(self.lr_cayley.get_lambda2())
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
