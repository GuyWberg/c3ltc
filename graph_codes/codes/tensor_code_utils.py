import numpy

from local_codes.linear_code_utils import *


# @jit(nopython=True, cache=True)
def is_matrix_in_tensor_code(M, local_parity_A, local_parity_B, prime):
    """
    Checks if a matrix in F_prime is in the tensor code_A tensor code_B.

    Remark: if j = 1..k' is a generating set, it's enough to check all i's and only these j.

    Keyword arguments:
    M - a matrix of size n_A x n_B (where n_A,n_B are the lengths of the codes whose parity matrices are local_parity_A, local_parity_B correspondingly).
    local_parity_A - a matrix, corresponding to a parity check matrix of the code enforced on the columns.
    local_parity_B - a matrix, corresponding to a parity check matrix of the code enforced on the rows.
    prime - the prime associated with the underlying field.
    """
    for i in range(len(M)):  # number of rows
        if numpy.count_nonzero(numpy.dot(local_parity_B, M[i, :]) % prime) != 0:
            return False
    for j in range(len(M[0])):  # number of columns
        if numpy.count_nonzero(numpy.dot(local_parity_A, M[:, j]) % prime) != 0:
            return False
    return True


# @jit(nopython=True, cache=True)
def tensor_code_decoding(noisy_word, local_generator_A, local_parity_A, local_generator_B, local_parity_B, prime, syndromes_A=None, syndromes_B=None, distance_A=None, distance_B=None):
    """
    Returns the corrected tensor codeword.

    Below we implement a simple tensor decoding algorithm. Given a word and two codes $C_A,C_B$, the algorithm alterates between decoding the rows according to $C_B$,
    and all columns according to $C_A$. This algorithm corrects up to $(d_1d_2-1)/4$ errors.

    Here, we adopt a slight improvement of this algorithm, as follows:
    1. Decode along all the rows.
    2. Alternate columns and rows repeatedly: we keep two sets called ``suspect_rows`` and ``suspect_columns``.
    Upon a correction of row $i$ (related to code $C_B$), each entry that was change signifies a column that needs to be checked
    (as it might not be in the code due to the change). Thus we'll add this column to ``suspect_columns``.
    Iterating over ``suspect_columns``, we do the same and update ``suspect_rows``.


    Keyword arguments:
    noisy_word - a matrix of size n_A x n_B (where n_A,n_B are the lengths of the codes whose parity matrices are local_parity_A, local_parity_B correspondingly).
    local_generator_A - a matrix, corresponding to a generator matrix of code_A enforced on the columns.
    local_parity_A - a matrix, corresponding to a parity check matrix of code_A enforced on the columns.
    local_generator_B - a matrix, corresponding to a generator matrix of code_B enforced on the rows.
    local_parity_B - a matrix, corresponding to a parity check matrix of code_B code enforced on the rows.
    syndromes_A - syndrom of local word in code_A -> the corresponding error vector.
    syndromes_B - syndrom of local word in code_B -> the corresponding error vector.
    distance_A - distance of code_B.
    distance_B - distance of code_A.
    prime - the prime associated with the underlying field.
    """
    noisy_word = numpy.copy(noisy_word)  # copy the original values to avoid changing the input
    if is_matrix_in_tensor_code(noisy_word, local_parity_A, local_parity_B, prime):
        return noisy_word % prime
    for i in set([i for i in range(len(noisy_word))]):
        row = noisy_word[i, :]
        decoded_row = decode(local_generator_A, local_parity_A, row, prime, syndromes_A)
        for j in range(len(decoded_row)):
            noisy_word[i][j] = decoded_row[j]

    suspect_rows = set([])  # the set of row indices of the rows the algorithm still needs to check
    suspect_columns = set([j for j in range(len(noisy_word[0]))])  # the set of column indices of the rows the algorithm still needs to check
    past_states = []  # during the algorithm, the value of noisy_word is going to be updated, this array stores a snapshot of the matrix before a decoding iteration, to make sure thw algorithm won't get stuck
    h = hash(str((noisy_word)))  # h = hash(noisy_word)
    while (len(suspect_rows) != 0 or len(suspect_columns) != 0) and h not in past_states:  # while the sets of iterating columns and rows are non empty and the decoding did not entrer a loop.
        past_states.append(h)
        new_suspect_rows = []
        new_suspect_columns = []
        for j in suspect_columns:
            column = noisy_word[:, j]
            decoded_column = decode(local_generator_B, local_parity_B, column, prime, syndromes_B)
            for i in range(len(decoded_column)):
                if noisy_word[i][j] != decoded_column[j]:  # if the entry in a column was changed by the decoding, then the column corresponding to this entry is added to teh iterating set of rows.
                    new_suspect_rows.append(j)
                noisy_word[i][j] = decoded_column[j]  # set the entry value to the decoded value
        suspect_rows = set(new_suspect_rows)  # replace the old set of iterating rows with the new set accumulated in this iteration.
        for i in suspect_rows:
            row = noisy_word[i, :]
            decoded_row = decode(local_generator_A, local_parity_A, row, prime, syndromes_A)
            for j in range(len(decoded_row)):
                if noisy_word[i][j] != decoded_row[j]:  # if the entry in a row was changed by the decoding, then the column corresponding to this entry is added to teh iterating set of columns.
                    new_suspect_columns.append(j)
                noisy_word[i][j] = decoded_row[j]  # set the entry value to the decoded value
        suspect_columns = set(new_suspect_columns)  # replace the old set of iterating columns with the new set accumulated in this iteration.
        h = hash(str((noisy_word)))
    return noisy_word % prime  # reduce the decoded matrix mod the relevant prime
