import os
import pickle

import numpy
from numba import (
    jit,
    int64,
)
from numba.core import (
    types,
)
from numba.experimental import (
    jitclass,
)
from numba.typed import (
    Dict,
    List,
)
import itertools


@jit(
    nopython=True,
    cache=True,
)
def to_int(
    x,
    q,
):
    y = 0
    for (
        i,
        j,
    ) in enumerate(x):
        y += j * int(
            pow(
                q,
                i,
            )
        )
    return y


def save_group(
    labels,
    name,
):
    if not os.path.isdir("./concrete_groups/" + name):
        os.mkdir("./concrete_groups/" + name)
    with open(
        "./concrete_groups/" + name + "/labels",
        "wb",
    ) as file:
        pickle.dump(
            list(labels),
            file,
            pickle.HIGHEST_PROTOCOL,
        )


def load_from_file(
    could_be_file,
):
    with open(
        could_be_file,
        "rb",
    ) as file:
        objects = pickle.load(file)
    return objects


class Fqm:
    @staticmethod
    def generate_elements(
        q,
        m,
    ):
        name = "F" + str(q) + "^" + str(m)
        file_name = "./concrete_groups/" + name
        # if os.path.isdir(file_name):
        #     print("[*] Start loading Fq^m")
        #     elements = format_element_list(List(itertools.product(range(0, q), repeat=m)), q, m)
        #     labels = List(load_from_file(file_name + "/labels"))
        #     print("[*] Finished loading Fq^m")
        # else:
        print("[*] Start generating Fq^m")
        (elements, labels,) = generate_elements(
            q,
            m,
        )
        # save_group(labels, name)
        print("[*] Finished generating Fq^m")
        return (
            elements,
            labels,
            name,
        )


def generate_elements(
    q,
    m,
):
    elements = format_element_list(
        List(
            itertools.product(
                range(
                    0,
                    q,
                ),
                repeat=m,
            )
        ),
        q,
        m,
    )
    labels = elements_to_labels(
        elements,
        q,
    )
    return (
        elements,
        labels,
    )


# @jit(nopython=True, cache=True)
def format_element_list(
    element_list,
    q,
    m,
):
    return List(
        [
            FqmElement(
                numpy.array(e),
                q,
                m,
            )
            for e in element_list
        ]
    )


def elements_to_labels(
    elements,
    q,
):
    return List(
        [
            to_int(
                e.value,
                q,
            )
            for e in elements
        ]
    )


spec = [
    (
        "_value",
        int64[:],
    ),
    (
        "_q",
        int64,
    ),
    (
        "_m",
        int64,
    ),
]


@jitclass(spec)
class FqmElement:
    def __init__(
        self,
        value,
        q,
        m,
    ):
        self._value = value
        self._q = q
        self._m = m

    def __mul__(
        self,
        other,
    ):
        res = FqmElement(
            (self.value + other.value) % self._q,
            self._q,
            self._m,
        )
        return res

    @property
    def value(
        self,
    ):
        return self._value

    @property
    def q(
        self,
    ):
        return self._q

    @property
    def m(
        self,
    ):
        return self._m

    def __eq__(
        self,
        other,
    ):
        if isinstance(
            other,
            FqmElement,
        ):
            return numpy.array_equal(
                self._value,
                other.value,
            )
        else:
            return False

    def __hash__(
        self,
    ):
        return to_int(
            self._value,
            self._q,
        )

    def is_non_identity_order_2(
        self,
    ):  # is of order 2 and non-identity
        if self._q == 2:
            return not self.is_identity()
        else:
            return not self.is_identity() and self * self == self

    def invert(
        self,
    ):
        return FqmElement(
            (-self.value) % self._q,
            self._q,
            self._m,
        )

    def is_identity(
        self,
    ):
        return numpy.count_nonzero(self._value) == 0
