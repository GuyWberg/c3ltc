import random
import numpy
from numba import jit, types
from numba.typed.typeddict import Dict
import galois

from local_codes.linear_code import LinearCode


def number_to_base(n, b, pad):
    '''
    Converts the number n to its b-ary decomposition (as an array), and pad the resulted array to size pad.  
    
    Example:
    number_to_base(10,2,10) == [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    '''
    tmp = numpy.array(list(numpy.base_repr(n, b)), dtype=int)
    return numpy.concatenate((numpy.array([0] * (pad - len(tmp)), dtype=int), tmp))[::-1]


def base_to_number(arr, b):
    '''
    Converts an array representing a b-ary decomposition of some number to its decimal value in base b.  
    
    Example:
    base_to_number([0, 1, 0, 1, 0, 0, 0, 0, 0, 0],2) == 10
    '''
    i = len(arr) - 1
    res = arr[i]
    while i != 1:
        res = res * b + arr[i - 1]
        i -= 1
    res += arr[0]
    return res


def encode(generator, message, prime):
    '''
    Encodes a message 
    '''
    return message @ generator % prime


def decode(generator, parity, noisy_word, prime, syndromes_to_error_vectors=None):
    """
    Decode a possibly noisy word:
    1. If the word is in the code, it returns the word. 
    2. If syndromes are provided, perform syndrom decoding - find the syndrom of the word, and subtruct the
       it from the word. 
    3. Else, resort to brute force decoding.    
    
    Keyword arguments:
    generator - generator matrix of the code. 
    parity - parity check matrix of the code. 
    noisy_word - the noisy word to be decoded. 
    prime - prime representing the field Z_p over which the code is defined. 
    syndromes - syndrom -> error vector.  
    """
    if numpy.count_nonzero(numpy.dot(parity, noisy_word) % prime) == 0:
        return noisy_word
    if syndromes_to_error_vectors is not None:
        h = base_to_number(parity.transpose() @ noisy_word % prime, prime)
        if h in syndromes_to_error_vectors:
            return (noisy_word - syndromes_to_error_vectors[h]) % prime
        else:
            return numpy.array([0] * generator.shape[1])
    return brute_force_decoding(generator, noisy_word, prime)


def brute_force_decoding(generator, noisy_word, prime):
    all_messages_size = int(pow(prime, generator.shape[0]))
    k = generator.shape[0]
    n = generator.shape[1]
    min_dist = len(noisy_word)
    closest_word = None
    for m in range(all_messages_size):  # exhaustive search
        message = number_to_base(m, prime, generator.shape[0])
        word = encode(generator, message, prime)
        non_zeros = 0
        for i in range(n):
            non_zeros += word[i] % prime != noisy_word[i] % prime
        if min_dist > non_zeros:
            min_dist = non_zeros
            closest_word = word
    return closest_word


def get_min_dist(generator, prime):
    '''
    Returns the word with the smallest Hamming weight (the minimal distance) in the code defined by the given generator matrix.
    '''
    all_messages_size = int(pow(prime, generator.shape[0]))
    k = generator.shape[0]
    n = generator.shape[1]
    min_dist = n
    for m in range(all_messages_size):
        message = numpy.array([0] * k)
        number_to_base(message, m, prime)
        word = encode(generator, message, prime)
        if numpy.count_nonzero(word) > 0:
            min_dist = min(min_dist, numpy.count_nonzero(word))
    return min_dist


def get_max_dist(generator, prime):
    '''
    Returns the word with the largest Hamming weight in the code defined by the given generator matrix.
    '''
    all_messages_size = int(pow(prime, generator.shape[0]))
    k = generator.shape[0]
    max_dist = 0
    for m in range(all_messages_size):
        message = numpy.array([0] * k)
        number_to_base(message, m, prime)
        word = encode(generator, message, prime)
        if numpy.count_nonzero(word) > 0:
            max_dist = max(max_dist, numpy.count_nonzero(word))
    return max_dist


array_type = types.int64[:]
index_type = types.int64


def get_syndromes(parity, generator, prime):
    '''
    Returns the syndrom decoding mapping of a code given its generator and parity matrices.  
    
    The syndrom decoding mapping is a mapping of a syndrom to its corresponding error vector. 
    That is, for an error vector e, the syndrom s is s = parity * e. 
    
    Using the syndroms decoding mapping, the decoding algorithm for a code could work as follows:
    1. Multiply the given (possibly noisy) word by the parity check, and recover the syndrom. 
    2. Finding the error vector according to the syndrom decoding array mapping.
    3. Subtract the error vector from the noisy word.    
    '''
    min_dist = get_min_dist(generator, prime)
    syndromes = Dict.empty(
        key_type=index_type,  # tuple_type
        value_type=array_type,
    )
    n = generator.shape[1]
    all_error_size = int(pow(prime, n))
    for m in range(all_error_size):
        error = numpy.array([0] * n)
        number_to_base(error, m, prime)
        if numpy.count_nonzero(error) < min_dist / 2:
            h = base_to_number(parity.transpose() @ error % prime, prime)
            syndromes[h] = error
    return syndromes


def get_random_codeword(generator, prime):
    '''
    Get a random codeword by choosing a random message and encoding it.  
    '''
    k = generator.shape[0]
    m = numpy.random.randint(0, high=prime, size=k, dtype=int)
    return numpy.array(generator.transpose() @ m % prime)[0]
