import subprocess
import time
from scipy.sparse import csr_matrix, coo_matrix
import os

def row_reduce_and_orthogonal(sparse_constraints, prime, cols, rows):
    print("[*] Start row reduce from c")
    name = "./tmp"
    with open(name, "w+") as file:
        file.write(
            "%s %s %s \n"
            % (rows, cols, "M")
        )
        content = get_matrix_content(sparse_constraints)
        file.write(content)
    path_to_sms_par = "./par"
    path_to_sms_gen = "./gen"
    args = [
        "./spasm/test/kernel",
        str(prime),
        path_to_sms_gen,
        path_to_sms_par,
        name,
    ]
    start = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    while True:
        line = proc.stdout.readline()
        if "Done" in str(line):
            break
    end = time.time()
    print("[*] Actual time in c", end - start)
    gen = convert_sms_to_dense(path_to_sms_gen, prime)
    par = convert_sms_to_dense(path_to_sms_par, prime)
    os.remove(path_to_sms_par)
    os.remove(path_to_sms_gen)
    os.remove(name)
    print("[*] Finished row reduce from c")
    return gen, par


def get_matrix_content(sparse_constraints):
    content = ""
    for (i, l) in enumerate(sparse_constraints):
        for (k, v) in enumerate(sparse_constraints[l]):
            content += "%s %s %s \n" % (i + 1, v + 1, sparse_constraints[l][v])
    content += "%s %s %s \n" % (0, 0, 0)
    return content


def convert_sms_to_dense(sms_file, prime):
    file = open(sms_file, "r")
    lines = file.readlines()
    data = []
    row = []
    col = []
    for line in lines:
        if "M" in line and line[0] == '0':
            raise "No rate"
        if "M" not in line and "0 0 0" not in line:
            s = line.strip().split(" ")
            r = int(s[0])
            c = int(s[1])
            v = int(s[2])
            if v < 0:
                v = prime + v
            data.append(v)
            row.append(r - 1)
            col.append(c - 1)
    coo = coo_matrix((data, (row, col)))
    file.close()
    mat = csr_matrix(coo).todense()
    return mat