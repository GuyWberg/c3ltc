import galois
import numpy

from local_codes.linear_code import (
    LinearCode,
)
from local_codes.linear_code_utils import (
    get_syndromes,
)


class ReedSolomonCode:
    @staticmethod
    def get_rs_code(
        n,
        k,
        prime,
    ):
        gen = galois.GF(prime)(
            numpy.array(
                [
                    [
                        int(
                            pow(
                                j,
                                i,
                            )
                        )
                        % prime
                        for j in range(n)
                    ]
                    for i in range(k)
                ],
                dtype=int,
            )
        )
        par = numpy.array(
            gen.null_space(),
            dtype=int,
        )
        gen = numpy.array(
            gen,
            dtype=int,
        )
        # syndromes = get_syndromes(par, gen, n + 1)
        syndromes = None
        name = "RS_q_" + str(prime) + "_n_" + str(n) + "_k_" + str(k)
        return LinearCode(
            gen,
            par,
            prime,
            0,
            syndromes,
            name,
            n - k + 1,
        )
