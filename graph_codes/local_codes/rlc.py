import os

import galois
import numpy

from local_codes.linear_code import (
    LinearCode,
)
from local_codes.linear_code_utils import (
    get_min_dist,
    get_syndromes,
)


class RandomLinearCode:
    @staticmethod
    def get_random_linear_code(
        n,
        k,
        prime,
        test = 0
    ):
        prime = prime
        name = "RLC_q_" + str(prime) + "_n_" + str(n) + "_k_" + str(k)
        min_dist = -1
        generator = numpy.array([[0]])
       
        parity = numpy.random.randint(
            0,
            high=prime,
            size=(
                n - k,
                n,
            ),
            dtype=int,
        )
        generator = numpy.array(
            galois.GF(prime)(parity).null_space(),
            dtype=int,
        )
        min_dist = None 

        syndromes = None 

        return LinearCode(
            generator,
            parity,
            prime,
            0,
            syndromes,
            name,
            3,
        )

    @staticmethod
    def save_code(
        parity,
        generator,
        name,
    ):
        if not os.path.isdir("./concrete_codes/" + name):
            os.mkdir("./concrete_codes/" + name)
            numpy.savetxt(
                "./concrete_codes/" + name + "/parity_check.txt",
                parity,
                fmt="%i",
                newline="\n",
            )
            numpy.savetxt(
                "./concrete_codes/" + name + "/generator_matrix.txt",
                generator,
                fmt="%i",
                newline="\n",
            )
