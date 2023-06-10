import numpy
from numba import jit, int64
from numba.core import types
from numba.experimental import jitclass
from numba.typed import Dict, List

from groups.group import GroupElement

square_type = types.UniTuple(types.int64, 4)


class PSL:
    @staticmethod
    def generate_elements(q):
        return generate_elements(q)


# @jit(nopython=True, cache=True)
def generate_elements(q):
    print("[*] Start generating PSL")
    psl_elements = List()
    labels = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    number_of_elements = 0
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    det = (a * d - b * c) % q
                    if det == 1 or det == -(q - 1):
                        found = False
                        for k in range(number_of_elements):
                            if hash((a, b, c, d)) == hash(psl_elements[k]) or hash(
                                ((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q)
                            ) == hash(psl_elements[k]):
                                found = True
                                break
                        if not found:
                            element = PSLElement(numpy.array([a, b, c, d]), q)
                            labels[hash((a, b, c, d))] = number_of_elements
                            labels[
                                hash(((q - a) % q, (q - b) % q, (q - c) % q, (q - d) % q))
                            ] = number_of_elements
                            psl_elements.append(element)
                            number_of_elements += 1
    print("[*] Finished generating PSL")
    return psl_elements, labels, "PSL(2," + str(q) + ")"


spec = [("_value", int64[:]), ("_q", int64)]


@jitclass(spec)
class PSLElement:
    def __init__(self, value, q):
        self._value = value
        self._q = q

    def __mul__(self, other):
        a = (self._value[0] * other.value[0] + self._value[1] * other.value[2]) % self._q
        b = (self._value[0] * other.value[1] + self._value[1] * other.value[3]) % self._q
        c = (self._value[2] * other.value[0] + self._value[3] * other.value[2]) % self._q
        d = (self._value[2] * other.value[1] + self._value[3] * other.value[3]) % self._q
        res = PSLElement(numpy.array([a, b, c, d]), self._q)
        return res

    @property
    def value(self):
        return self._value

    @property
    def q(self):
        return self._q

    def minus(self):
        return PSLElement(-self.value % self._q, self._q)

    def __hash__(self):
        return hash((self._value[0], self._value[1], self._value[2], self._value[3]))

    def __eq__(self, other):
        if isinstance(other, PSLElement):
            return (
                numpy.array_equal(self._value, other.value)
                or numpy.array_equal(self.minus().value, other.value)
            ) and self._q == other.q
        else:
            return False

    def is_non_identity_order_2(self):  # is of order 2 and non-identity
        return (self * self).is_identity() and not self.is_identity()

    def invert(self):
        return PSLElement(
            numpy.array(
                [
                    self._value[3],
                    (self._q - self._value[1]) % self._q,
                    (self._q - self._value[2]) % self._q,
                    self._value[0],
                ]
            ),
            self._q,
        )

    def is_identity(self):
        return hash(self) == hash((1, 0, 0, 1)) or hash(self) == hash(
            (self._q - 1, 0, 0, self._q - 1)
        )
