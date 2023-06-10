import os
import subprocess
import time

import galois
import numpy
from numba import jit
import random

from fast_linear_algebra.utils import convert_sms_to_dense


def row_reduce_and_orthogonal(matrix, prime, cols, rows, test=0):
    print("[*] Start row reduce from c")
    r = random.randint(0, 10000)
    name = "./tmp" + str(r)
    with open(name, "w+") as file:
        file.write("%s %s %s \n" % (rows, cols, "M"))
        content = get_matrix_content(matrix)
        file.write(content)
    path_to_sms_par = "./par" + str(r)
    path_to_sms_gen = "./gen" + str(r)
    args = [
        "requirements/spasm/test/kernel",
        str(prime),
        path_to_sms_gen,
        path_to_sms_par,
        name,
        str(test),
    ]
    start = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    while True:
        line = proc.stdout.readline()
        if "Done" in str(line):
            break
    end = time.time()
    print("[*] Actual time in c", end - start)
    if test:
        gen = path_to_sms_gen
    if not test:
        gen = convert_sms_to_dense(path_to_sms_gen)
        os.remove(path_to_sms_gen)
        par = convert_sms_to_dense(path_to_sms_par)
        os.remove(path_to_sms_par)
    # os.remove(name)
    print("[*] Finished row reduce from c")
    if test:
        return gen
    else:
        return gen, par


def get_matrix_content(sparse_matrix_dict):
    content = ""
    for k in sorted(sparse_matrix_dict.keys()):
        content += "%s %s %s \n" % (k[1] + 1, k[0] + 1, sparse_matrix_dict[k])
    content += "%s %s %s \n" % (0, 0, 0)
    return content


def from_sparse_dict_to_matrix(sparse_dict, rows, cols):
    M = numpy.zeros((rows, cols), dtype=int)
    for k in sparse_dict:
        M[k[0]][k[1]] = sparse_dict[k]
    return M


def from_matrix_to_sparse_dict(M):
    rows, cols = M.shape[0], M.shape[1]
    sparse_dict = {}
    for i in range(len(M)):
        for j in range(len(M[0])):
            if M[i][j] != 0:
                sparse_dict[(i, j)] = M[i][j]
    return sparse_dict, rows, cols


def row_reduce_sparse_constraints(matrix_cell_to_value, inverses, rowCount, columnCount):
    prime = len(inverses)
    mat = dictionary_to_matrix(matrix_cell_to_value, rowCount, columnCount)
    row_reduced = numpy.array(
        galois.GF(prime)(numpy.array(mat, dtype=int)).row_reduce()
    )
    row_reduced = row_reduced[~numpy.all(row_reduced == 0, axis=1)]
    return row_reduced[~numpy.all(row_reduced == 0, axis=1)]


def dictionary_to_matrix(d, rows, columns):
    mat = numpy.zeros((rows, columns))
    for e in d:
        if e[0] < rows:
            mat[e[0]][e[1]] = d[e]
    return mat
