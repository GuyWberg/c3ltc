from graphs.lr_cayley import LeftRightCayleyGraph
from codes.c3ltc import c3LTC
import sys
import time
import galois
import numpy

from codes.expander_code import ExpanderCode
from local_codes.linear_code import LinearCode
from local_codes.linear_code_utils import get_min_dist, get_max_dist
from local_codes.rlc import RandomLinearCode
from local_codes.rs_code import ReedSolomonCode
from epsilon_biased.epsilon_biased import get_epsilon_biased_space_sampler
from groups.generators import get_AB_with_TNC, get_random_generators
from groups.psl_group import PSL
from graphs.cayley import CayleyGraph
from groups.Fqm import Fqm, FqmElement
from groups.group import Group
from testing.errors import get_noisy_codeword, get_noisy_zero_codeword


def get_code_from_generator(generator, prime, code_name, compute_distance=False):
    parity = numpy.array(galois.GF(prime)(generator).null_space())
    syndromes = None
    if not compute_distance:
        distance = -1
    else:
        distance = get_min_dist(generator, prime)
    name = (
        code_name
        + " q="
        + str(prime)
        + " n_0="
        + str(generator.shape[1])
        + " k_0="
        + str(generator.shape[0])
        + " d_0="
        + str(distance)
    )
    return LinearCode(
        generator, parity, prime, 0, syndromes=syndromes, name=name, distance=distance
    )


def compute_epsilon(generator, prime):
    min_dist = get_min_dist(generator, prime) / generator.shape[1]
    max_dist = get_max_dist(generator, prime) / generator.shape[1]
    return max(0.5 - min_dist, max_dist - 0.5)


def get_non_zero_squares(word):
    x = []
    for i, w in enumerate(word):
        if w != 0:
            x.append(i)
    return x


def matrix_to_list_of_generators(generator, prime, k):
    A0 = [FqmElement(numpy.array(r), prime, k) for r in generator.transpose().tolist()]
    A1 = [FqmElement(-numpy.array(r) % prime, prime, k) for r in generator.transpose().tolist()]
    A = set(A0 + A1)  # set operation permutes generators
    return A


def random_code(n, k, prime):
    code = RandomLinearCode.get_random_linear_code(n, k, prime)
    A = matrix_to_list_of_generators(code.generator, prime, k)
    while len(set(A)) != len(A):
        code = RandomLinearCode.get_random_linear_code(n, k, prime)
        A = matrix_to_list_of_generators(code.generator, prime, k)
    print("[*] Found code")
    return code.generator


def sample_epsilon_biased_code(n, k, epsilon=0.1):
    sampler = get_epsilon_biased_space_sampler(n, epsilon)
    code = []
    for i in range(k):
        code.append(sampler())
    generator = numpy.array(code).transpose()
    zero = FqmElement(numpy.array([0] * n), 2, k)
    A = [FqmElement(numpy.array(r), 2, k) for r in generator.tolist()]
    while len(set(A)) != len(A) or zero in A:
        sampler = get_epsilon_biased_space_sampler(n, epsilon)
        code = []
        for i in range(k):
            code.append(sampler())
        generator = numpy.array(code).transpose()
        A = [FqmElement(numpy.array(r), 2, k) for r in generator.tolist()]
    print("[*] Found code")
    return generator.transpose()


def get_non_zero_squares(word):
    x = []
    for i, w in enumerate(word):
        if w != 0:
            x.append(i)
    return x


#############################
# Code
#############################

prime_code = 7
n, kb = 6, 4

code = ReedSolomonCode.get_rs_code(n, kb, prime_code)
# code = get_code_from_parity(numpy.array([[1]*n],dtype=int),2,"CycleCode")

#############################
# Group
#############################

prime_field = 7
e, vertices, name = PSL.generate_elements(prime_field)
g = Group(e, name, vertices)

# e, labels, name = Fqm.generate_elements(2, n-1)
# g = Group(e, name, labels)

#############################
# Expander Code
#############################

print("generate random generators ---------------------------------")
start = time.time()

gens = get_random_generators(g.elements, n)
# gens = get_random_generators(g.elements, 0,n)

end = time.time()
print("Number of vertices: ", len(e))
print("generate random generators time", end - start)

print("cayley generation ---------------------------------")
start = time.time()
graph = CayleyGraph(g, gens)
end = time.time()
print("cayley generation time", end - start)
print("Expansion", graph.get_lambda2())
assert graph.get_lambda2() <= 0.85
print("generate cayley time", end - start)

print("expander code generation ---------------------------------")
start = time.time()
expander_code = ExpanderCode(graph, code, 1)
end = time.time()
print("Generator dimension", expander_code.k)
print("Parity dimension", expander_code.n - expander_code.k)
print("Rate", expander_code.k / expander_code.n)
print("expander code generation time", end - start)

#############################
# c3LTC
#############################

print("generate random generators ---------------------------------")
start = time.time()

l = get_random_generators(g.elements, n)
r = get_random_generators(g.elements, n)

# l = get_random_generators(g.elements, 0,n)
# r = get_random_generators(g.elements, 0,n)

# l,r = get_AB_with_TNC(g, n)
# l,r = get_AB_with_TNC(g, 0, n)

end = time.time()
print("Number of vertices: ", len(e))
print("generate random generators time", end - start)

print("lr cayley generation ---------------------------------")
start = time.time()
lr_cayley = LeftRightCayleyGraph(g, r, l)
end = time.time()
print("lr generation time", end - start)
print("Squares", lr_cayley.number_of_squares)
print("Expansion", lr_cayley.get_lambda2())
print("generate lr cayley time", end - start)
assert lr_cayley.get_lambda2() <= 0.85

print("c3ltc generation ---------------------------------")
start = time.time()
c3ltc = c3LTC(lr_cayley, code, code, 0)
end = time.time()
print("Generator dimension", c3ltc.k)
print("Parity dimension", c3ltc.n - c3ltc.k)
print("Rate", c3ltc.k / c3ltc.n)
print("c3ltc generation time", end - start)


print("ltc rate", c3ltc.k / c3ltc.n)
print("expander rate ^ 2", expander_code.k /
      expander_code.n * expander_code.k / expander_code.n)
print("rate(expander)^2 = rate(c3ltc) ?", c3ltc.k / c3ltc.n ==
      expander_code.k / expander_code.n * expander_code.k / expander_code.n)


# w, w_  = get_noisy_codeword(c3ltc)
# corrected = c3ltc.decode_along_the_edges(w_)
# print("Non zeros in noisy word", numpy.count_nonzero((w-w_)%prime_code))
# print("Decoded correctly by edges?", numpy.array_equal(corrected,w))
# corrected = c3ltc.decode_along_the_vertices(w_)
# print("Decoded correctly by vertices?", numpy.array_equal(corrected,w))
