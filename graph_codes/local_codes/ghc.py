import itertools

import galois
import numpy


class GeneralHammingCode:
    @staticmethod
    def get_ghc_generator(
        n,
    ):
        H = numpy.array(
            list(
                itertools.product(
                    range(
                        0,
                        2,
                    ),
                    repeat=n,
                )
            )[1:]
        ).transpose()
        return numpy.array(galois.GF2(H).null_space())
