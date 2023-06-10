from abc import ABC, abstractmethod


class Group:
    def __init__(self, elements, name, labels=None):
        self.elements = elements
        self.labels = labels
        self.name = name


class GroupElement(ABC):
    @abstractmethod
    def __init__(self, value, q):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def is_non_identity_order_2(self):  # is of order 2 and non-identity
        pass

    @abstractmethod
    def invert(self):
        pass

    @abstractmethod
    def is_identity(self):
        pass
