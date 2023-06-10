import os

import numpy


class LinearCode:
    def __init__(
        self,
        generator,
        parity,
        prime,
        epsilon=None,
        syndromes=None,
        name="",
        distance=None,
        decoding_function=None,
    ):
        self.parity = parity
        self.generator = generator
        self.prime = prime
        self.name = name
        self.syndromes = syndromes
        self.distance = distance
        self.epsilon = epsilon

    def save_code(
        self,
        path,
    ):
        path = path + "/"
        if not os.path.isdir(path + self.name):
            os.mkdir(path + self.name)
            numpy.savetxt(
                path + self.name + "/parity_check.txt",
                self.parity,
                fmt="%i",
                newline="\n",
            )
            numpy.savetxt(
                path + self.name + "/generator_matrix.txt",
                self.generator,
                fmt="%i",
                newline="\n",
            )
