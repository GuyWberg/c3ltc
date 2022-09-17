#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
import random

from sage.coding.grs_code import ReedSolomonCode
from sage.rings.finite_rings.finite_field_prime_modn import FiniteField_prime_modn as GF
from sage.groups.perm_gps.permgroup_named import PSL
from sage.matrix.matrix_space import MatrixSpace


def get_square_label(list_G, squares, to_label):
    """ Returns the label of a square - 3 group elements (a,g,b).

    Keyword arguments:
    list_G -- lisf of group elements.
    squares -- list of past squares.
    to_label -- square - 3 group elements (a,g,b) - to label.
    """

    a = to_label[0]
    g = to_label[1]
    b = to_label[2]
    gb = g * b
    ag = a * g
    agb = a * g * b
    if (a, gb, b.inverse()) in squares:
        return (a, gb, b.inverse())
    elif (a.inverse(), ag, b) in squares:
        return (a.inverse(), ag, b)
    elif (a.inverse(), agb, b.inverse()) in squares:
        return (a.inverse(), agb, b.inverse())
    return (a, g, b)


def embedding_local_parity_constraints_on_squares(
    code_a,
    code_b,
    G,
    A,
    B,
    ):
    """ Returns
    1) Sparse representation of the constrsints by amapping of squares (represeted by 3 group elements (a,g,b))
        to dictionary whose keys are rows in which the square has non zero value, and value is the value in the relevant row and column.
    2) Number constraints of rows in the constraint matrix.

    Keyword arguments:
    code_a -- Sage code object.
    code_a -- Sage code object.
    G -- Sage group object.
    A -- list of group elements.
    B -- list of group elements.

    Note:
    1) The length of code_a codewords has to be equal to the number of elements in A.
    2) The length of code_b codewords has to be equal to the number of elements in B.
    3) Both A and B has to be inverse closed (not checked).
    4) code_a and code_b must be defined over the same finite field.
    """

    constraints_by_edge_a = {}
    constraints_by_edge_b = {}
    constraints = {}
    list_G = list(G)
    squares = []
    for g in G:
        for (i, a) in enumerate(A):
            if a.inverse() in A[:i]:
                continue
            assert (a, g) not in constraints_by_edge_a
            constraints_by_edge_a[(a, g)] = {}
            for (j, b) in enumerate(B):
                square = get_square_label(list_G, squares, (a, g, b))
                if square not in squares:
                    squares.append(square)
                constraints_by_edge_a[(a, g)][square] = \
                    code_a.parity_check_matrix()[:, j]
        for (j, b) in enumerate(B):
            if b.inverse() in B[:j]:
                continue
            assert (g, b) not in constraints_by_edge_b
            constraints_by_edge_b[(g, b)] = {}
            for (i, a) in enumerate(A):
                square = get_square_label(list_G, squares, (a, g, b))
                if square not in squares:
                    squares.append(square)
                constraints_by_edge_b[(g, b)][square] = \
                    code_b.parity_check_matrix()[:, i]
    assert len(set(squares)) == len(A) * len(B) * len(list_G) / 4
    assert len(constraints_by_edge_a) == len(list_G) * len(A) / 2
    assert len(constraints_by_edge_b) == len(list_G) * len(B) / 2
    constraint_count = 0
    number_of_rows_in_parity_a = \
        len(code_a.parity_check_matrix().rows())
    for (i, e) in enumerate(constraints_by_edge_a):
        for square in constraints_by_edge_a[e]:
            if square not in constraints:
                constraints[square] = {}
            for (k, values) in \
                enumerate(constraints_by_edge_a[e][square]):
                assert len(constraints_by_edge_a[e][square].rows()) \
                    == number_of_rows_in_parity_a
                constraints[square][constraint_count + k] = \
                    constraints_by_edge_a[e][square][k][0]
        constraint_count += number_of_rows_in_parity_a
    number_of_rows_in_parity_b = \
        len(code_b.parity_check_matrix().rows())
    for (i, e) in enumerate(constraints_by_edge_b):
        for square in constraints_by_edge_b[e]:
            if square not in constraints:
                constraints[square] = {}
            for (k, values) in \
                enumerate(constraints_by_edge_b[e][square]):
                assert len(constraints_by_edge_b[e][square].rows()) \
                    == number_of_rows_in_parity_b
                constraints[square][constraint_count + k] = \
                    constraints_by_edge_b[e][square][k][0]
        constraint_count += number_of_rows_in_parity_b
    assert constraint_count == number_of_rows_in_parity_a \
        * len(constraints_by_edge_a) + number_of_rows_in_parity_b \
        * len(constraints_by_edge_b)
    return (constraints, constraint_count)


def random_generators(G, n):
    """ Returns an inverse closed set of N elements from G.


    Keyword arguments:
    G -- Sage group object.
    n -- number.

    Note:
    1) n has to be smaller than the number of elements in G.
    """

    list_G = list(G)
    assert n < len(list_G)
    gens = []
    i = 0
    while i < n / 2:
        c = random.choice(list_G)
        if c not in gens and c * c != G.identity():
            gens.append(c)
            gens.append(c.inverse())
            i += 1
    return gens


class c3LTC:

    def __init__(
        self,
        code_a,
        code_b,
        G,
        A,
        B,
        ):
        assert len(code_a.generator_matrix().columns()) == len(A)
        assert len(code_b.generator_matrix().columns()) == len(B)
        assert code_a.base_field().characteristic() \
            == code_b.base_field().characteristic()

        (sparse_constraints, count) = \
            embedding_local_parity_constraints_on_squares(code_a,
                code_b, G, A, B)

        # process sparse constraints

        constraints = numpy.zeros((count, len(sparse_constraints)))
        for (i, l) in enumerate(sparse_constraints):
            for k in sparse_constraints[l]:
                constraints[k][i] = sparse_constraints[l][k]

        # additional mappings

        self.__square_to_index = {}
        self.__index_to_square = {}
        self.__squares = []
        self.__list_G = list(G)
        for (i, l) in enumerate(sparse_constraints):
            self.__squares.append(l)
            self.__square_to_index[l] = i
            self.__index_to_square[i] = l
        self.__constraint_matrix = constraints
        self.vertex_to_squares = {}
        for g in G:
            view = numpy.zeros((len(A), len(B)))
            for (i, a) in enumerate(A):
                for (j, b) in enumerate(B):
                    view[i][j] = \
                        self.__squares.index(get_square_label(self.__list_G,
                            self.__squares, (a, g, b)))
            self.vertex_to_squares[self.__list_G.index(g)] = view
        self.vertex_to_neighbours_left = {}
        for g in G:
            view = []
            for (i, a) in enumerate(A):
                view.append(self.__list_G.index(a * g))
            self.vertex_to_neighbours_left[self.__list_G.index(g)] = \
                view
        self.vertex_to_neighbours_right = {}
        for g in G:
            view = []
            for (j, b) in enumerate(B):
                view.append(self.__list_G.index(g * b))
            self.vertex_to_neighbours_right[self.__list_G.index(g)] = \
                view
        self.square_to_vertices = {}
        for (i, s) in enumerate(self.__squares):
            view = []
            a = s[0]
            g = s[1]
            b = s[2]
            view.append(self.__list_G.index(a * g))
            view.append(self.__list_G.index(g * b))
            view.append(self.__list_G.index(a * g * b))
            view.append(self.__list_G.index(g * b))
            self.square_to_vertices[i] = view

        # properties of the code

        self.A = A
        self.B = B
        self.code_a = code_a
        self.code_b = code_b
        self.G = G
        self.base_field = code_a.base_field()
        M = MatrixSpace(self.base_field, constraints.shape[0],
                        constraints.shape[1], sparse=True)
        dual = LinearCode(M(constraints))
        self.generator_matrix = dual.parity_check_matrix()
        self.parity_check_matrix = dual.generator_matrix()
        self.length = dual.length()
        self.dimension = dual.length() - dual.dimension()

    def decode(self, noisy_word):
        squares_to_values = {}
        for (i, v) in enumerate(noisy_word):
            squares_to_values[self.__index_to_square[i]] = v

        for _ in range(3):
            for g in G:
                for (i, a) in enumerate(A):
                    if a.inverse() in A[:i]:
                        continue
                    local_word = vector(self.base_field, [0]
                            * len(self.B))
                    for (j, b) in enumerate(B):
                        square = get_square_label(self.__list_G,
                                self.__squares, (a, g, b))
                        squares_to_values[j] = squares_to_values[square]
                    corrected_localy = \
                        self.code_b.decode_to_code(local_word)
                    for (j, b) in enumerate(B):
                        square = get_square_label(self.__list_G,
                                self.__squares, (a, g, b))
                        noisy_word[self.__square_to_index[square]] = \
                            corrected_localy[j]
                for (j, b) in enumerate(B):
                    if b.inverse() in B[:j]:
                        continue
                    for (i, a) in enumerate(A):
                        square = get_square_label(self.__list_G,
                                self.__squares, (a, g, b))
                        squares_to_values[i] = squares_to_values[square]
                    corrected_localy = \
                        self.code_a.decode_to_code(local_word)
                    for (i, a) in enumerate(A):
                        square = get_square_label(self.__list_G,
                                self.__squares, (a, g, b))
                        noisy_word[self.__square_to_index[square]] = \
                            corrected_localy[i]
        return noisy_word

    def syndrome(self, c):
        return self.parity_check_matrix * c

    def local_codeword_on_vertex(self, vertex, word):
        labels_view = self.vertex_to_squares[vertex]
        rows = len(labels_view)
        cols = len(labels_view[0])
        local_view_values = numpy.array([0] * rows
                * cols).reshape((rows, cols))
        for i in range(rows):
            for j in range(cols):
                local_view_values[i][j] = \
                    int(word[int(self.vertex_to_squares[vertex][i][j])])
        return local_view_values

    def __repr__(self):
        rep = 'c3LTC'
        return rep
