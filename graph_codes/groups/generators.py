import numpy


def get_AB_with_TNC(G, n, n_order_2=0, trials=100):
    """
    Returns a set of left-right generators for the Left-Right-Cayley graph, such that the TNC condition holds.
    That is, a * g != g * b for all g in G, a in A and b in B.
    
    Keyword arguments:
    G - a group object. 
    n - number of generators if order != 2. 
    n_order_2 - number of generators if order 2.
    trials - number of trials (if failed to find suitable sets after this many trials, it exits with an error). 
    """
    for _ in range(trials):
        A = get_random_generators(G.elements, n, n_order_2)
        B = get_random_generators(G.elements, n, n_order_2)
        violations = 0
        for a in A:
            for b in B:
                for g in G.elements:
                    if a * g == g * b:
                        violations += 1
                        break
        if violations == 0:
            return A, B
    exit(1)


def get_random_generators(elements, number_of_non_order_2_gens, number_of_order_2_elements=0):
    """
    Returns a set of generators such that, for non-order 2 generators, each generator is followed by its inverse.
    
    Keyword arguments:
    elements - a set of group elements. 
    number_of_non_order_2_gens - number of generators if order != 2. 
    number_of_order_2_elements - number of generators if order 2.
    """
    print("[*] Start generating generators")
    assert number_of_non_order_2_gens % 2 == 0
    generators = []
    generators_hash = []
    for i in range(int(number_of_non_order_2_gens / 2)):
        found = False
        while not found:
            r = numpy.random.randint(0, len(elements) - 1)
            cand = elements[r]
            inv = cand.invert()
            if (
                hash(cand) not in generators_hash
                and not cand.is_non_identity_order_2()
                and not cand.is_identity()
            ):
                generators.append(cand)
                generators_hash.append(hash(cand))
                generators.append(inv)
                generators_hash.append(hash(inv))
                found = True
    for i in range(number_of_order_2_elements):
        found = False
        while not found:
            r = numpy.random.randint(0, len(elements) - 1)
            cand = elements[r]
            inv = cand.invert()
            if (
                hash(cand) not in generators_hash
                and not cand.is_identity()
                and cand.is_non_identity_order_2()
            ):
                generators.append(cand)
                generators_hash.append(hash(cand))
                found = True
    print("[*] Finished getting generators")
    return generators
