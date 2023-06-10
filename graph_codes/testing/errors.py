from random import randint
import random
import numpy
from codes.c3ltc import c3LTC
from local_codes.linear_code_utils import get_random_codeword


def get_random_error(c3ltc):
    return numpy.array([randint(0, c3ltc.prime - 1) * randint(0, 1) * randint(0, 1) * randint(0, 1) for _ in range(c3ltc.generator.shape[1])])

def get_noisy_zero_codeword(c3ltc):
    return (numpy.array([0 for _ in range(c3ltc.n)]) - numpy.array([randint(0, 1) * randint(0, 1) * randint(0, 1) for _ in range(c3ltc.n)])) % c3ltc.prime

def get_noisy_codeword(c3ltc):
    w = get_random_codeword(c3ltc.generator, c3ltc.prime)
    return w, (w - numpy.array([randint(0, 1) * randint(0, 1) * randint(0, 1) * randint(0, 1) * randint(0, 1) for _ in range(c3ltc.n)])) % c3ltc.prime
