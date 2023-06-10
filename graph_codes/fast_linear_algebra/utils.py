from scipy.sparse import csr_matrix, coo_matrix


def convert_sms_to_dense(sms_file):
    file = open(sms_file, "r")
    lines = file.readlines()
    data = []
    row = []
    col = []
    for line in lines:
        if "M" in line and line[0] == "0":
            raise Exception("No rate")
        if "M" not in line and "0 0 0" not in line:
            s = line.strip().split(" ")
            r = int(s[0])
            c = int(s[1])
            v = int(s[2])
            data.append(v)
            row.append(r - 1)
            col.append(c - 1)
    coo = coo_matrix((data, (row, col)))
    file.close()
    mat = csr_matrix(coo).todense()
    return mat
