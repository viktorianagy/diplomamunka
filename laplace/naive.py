import numpy as np
import math
import networkx as nx
import scipy.sparse.linalg
import scipy.sparse.csr
import itertools
import json
import heapq
import os
import subprocess
import timeit
from operator import itemgetter
from scipy.linalg import logm
import math
import csv


def generate_graph(datasetpath=" "):

    #fh = open(preprocessgraphfile(datasetpath), "r")
    #G = nx.read_edgelist(fh, nodetype=int)
    #G = nx.barabasi_albert_graph(20, 5)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    #fh.close()
    A = nx.linalg.graphmatrix.adjacency_matrix(G)
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    return G, A, L


def preprocessgraphfile(in_name, out_name="tmp_graph"):
    gr = open(in_name, "r")
    tmp_file = open(out_name, "w")

    for line in gr:

        if line.startswith("%"):
            continue


        l = line.split()

        if len(l) == 3:
            print("l:", l)
            del l[-1]

        for i in l:
            i = int(i) - 1
            tmp_file.write(str(i) + " ")

        tmp_file.write("\n")

    gr.close()
    tmp_file.close()

    return out_name





def invPowerMethod(A, x=None, tol=1e-10, maxiter=10000):
    n = A.shape[0]

    if x is None:
        x = np.ones(n)

    err = np.inf
    iterations = 0
    y = np.ones(n)
    alpha0 = 0

    while err > tol and iterations < maxiter:
        x = y / np.linalg.norm(y, 2)
        y = scipy.sparse.linalg.spsolve(A, x)
        alpha1 = np.dot(x, y)

        if iterations > 0:
            err = np.abs(alpha1 - alpha0) / np.abs(alpha0)

        alpha0 = alpha1
        iterations += 1

    minEigenvalue = 1 / alpha1
    v = x

    B = A.toarray()
    # print(np.linalg.eig(B))

    return minEigenvalue, v, err, iterations

# Work only to csr format
def delete_from_csr(mat, row_indices=[], col_indices=[]):
    rows = []
    cols = []

    if row_indices:
        rows = list(row_indices)

    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:, col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat


def generate_list(k, n):
    s = list(range(n))
    return list(itertools.combinations(s, k))

"""
def  gap_enumerate(G, L):
    n = nx.number_of_nodes(G)
    gaps = []
    max_gap = 0
    max_set = None
    max_L_eigs = None

    for k in range(n):
        list_of_sets = generate_list(k, n)

        for s in list_of_sets:
            l = list(s)
            L_red = delete_from_csr(L, l, l)
            L_red_eigs, L_red_ev = scipy.linalg.eigh(L_red.toarray())
            i = 0

            while L_red_eigs[i] < 10**-10 and i < n - k and i < len(L_red_eigs) - 1:
                i+=1

            if L_red_eigs[i] > max_gap:
                max_gap = L_red_eigs[i]
                max_set = s
                max_L_eigs = L_red_eigs
                print("%.15f" % round(max_gap, 15))


    return max_gap, max_set, max_L_eigs
"""

def gap_enum(G, L, k):
    n = nx.number_of_nodes(G)
    max_gap = 0
    max_set = None
    list_of_sets = generate_list(k, n)

    for s in list_of_sets:
        l = list(s)
        L_red = delete_from_csr(L, l, l)
        min_eig, v, err, iterations = invPowerMethod(L_red)
        print(min_eig, s)

        if min_eig > max_gap:
            max_gap = min_eig
            max_set = s

    return max_gap, max_set


def gap_enum_ugly(n, L, k):
    #n = n
    max_gap = 0
    max_set = None
    list_of_sets = generate_list(k, n)

    for s in list_of_sets:
        l = list(s)
        L_red = delete_from_csr(L, l, l)
        min_eig, v, err, iterations = invPowerMethod(L_red)
        print(min_eig)

        if min_eig > max_gap:
            max_gap = min_eig
            max_set = s

    return max_gap, max_set


def naive(L, k):
    start = timeit.default_timer()
    S = []
    lam = 0

    for i in range(k):
        lam_s = 0
        n = L.shape[0]

        for j in range(n):
            J = S.copy()

            if j not in S:
                #print("j: ", j)
                #print("S: ", S)
                J.append(j)
                #print("J:", len(J))
                #print("S:", len(S))
                L1 = delete_from_csr(L, J, J)
                #L1 = delete_from_csr(L1, J, [1])
                #print(L1.toarray())
                #print("shape", L1.shape)
                L1 = L1.asfptype()
                lam_j, v, err, iterations = invPowerMethod(L1)
                lam_s_1 = lam_j - lam
                # print("lam", lam)
                # print("lam_j", lam_j)
                # print("lam_s_1",lam_s_1)

                if lam_s_1 > lam_s:
                    lam_s = lam_s_1
                    s = j
                    #print(s)
                    lam = lam_j
        # print("s: ", s)
        S.append(s)
    stop = timeit.default_timer()
    runtime = stop - start
    return S, lam, runtime



def make_json(A, L, max_gap, max_set, name, time):

    data = {
        'adjacency' : str(A),
        'laplacian': str(L),
        'max': max_gap,
        'max set': max_set,
        'time': time
    }

    with open('{}.json'.format(name), 'w') as outfile:
        json.dump(data, outfile, indent=4)


def gershgorin_circle(M, i):
    rad = abs(M[i, :]).sum() - abs(M[i, i])
    circle_min = M[i, i] - rad
    circle_max = M[i, i] + rad
    return circle_min, circle_max, rad, M


def make_circles(M):
    circles_min = []
    rad = []
    circles_max = []
    indexes = []

    for i in range(len(scipy.sparse.csc_matrix.getnnz(M, 1))):
        cmin, cmax, r, M = gershgorin_circle(M, i)
        circles_min.append(cmin)
        circles_max.append(cmax)
        rad.append(r)
        indexes.append(i)

    #gersh_data = list(zip(indexes, circles_min, circles_max, rad))

    #return gersh_data
    return circles_min, circles_max, rad



def gershgorin_fast(M, k, G, item, minmax):
    start = timeit.default_timer()
    indexes = []
    indexes_original = list(range(len(scipy.sparse.csc_matrix.getnnz(M, 1))))
    circles_min, circles_max, rad = make_circles(M)
    gersh_data = list(zip(indexes_original, circles_min, circles_max, rad))

    for i in range(k):
        #gersh_data = make_circles(M)
        #max_rad_or_ind = max(gersh_data, key=itemgetter(3))[0]
        if minmax == 'min':
            min_rad = min(gersh_data, key=itemgetter(item))
        else:
            min_rad = max(gersh_data, key=itemgetter(item))
        min_rad_or_ind = min_rad[0]
        indexes.append(min_rad_or_ind)
        d_rc = [gersh_data.index(min_rad)]
        M = delete_from_csr(M, d_rc, d_rc)
        # print("M", M.toarray())
        indexes_original.remove(min_rad_or_ind)
        gersh_data.remove(min_rad)
        # gersh_data[gersh_data.index(max_rad)] = (gersh_data[gersh_data.index(max_rad)][0], -1, -1, -1)
        minEigenvalue, v, err, iterations = invPowerMethod(M)

        for j in G.neighbors(min_rad_or_ind):

            if j in indexes_original:
                idx = 0

                for kiscica in gersh_data:

                    if kiscica[0] == j:
                        break

                    idx += 1

                gersh_data[idx] = (gersh_data[idx][0], gersh_data[idx][1] + 1, gersh_data[idx][2] - 1, gersh_data[idx][3] - 1)


    stop = timeit.default_timer()
    runtime = stop - start
    return minEigenvalue, indexes, runtime


def min_deg(M, G, k):
    start = timeit.default_timer()
    degrees = list(G.degree())

    indexes = []
    for i in range(k):
        min_deg = min(degrees, key=itemgetter(1))[0]
        #print(min_deg)
        indexes.append(min_deg)
        degrees = [(key, val) for (key, val) in degrees if key != min_deg]

    M = delete_from_csr(M, indexes, indexes)
    minEigenvalue, v, err, iterations = invPowerMethod(M)
    stop = timeit.default_timer()
    runtime = stop - start
    return minEigenvalue, indexes, runtime


def max_deg(M, G, k):
    start = timeit.default_timer()
    degrees = list(G.degree())

    indexes = []
    for i in range(k):
        max_deg = max(degrees, key=itemgetter(1))[0]
        #print(min_deg)
        indexes.append(max_deg)
        degrees = [(key, val) for (key, val) in degrees if key != max_deg]

    M = delete_from_csr(M, indexes, indexes)
    minEigenvalue, v, err, iterations = invPowerMethod(M)
    stop = timeit.default_timer()
    runtime = stop - start

    return minEigenvalue, indexes, runtime


def make_ampl_data_file(k, name, G):
    f = open('C:\\ampl_mswin64\\models\\{}.dat'.format(name), 'w')
    f.write("param k := {}; \n".format(k))
    f.write("set N := \n")

    for i in range(G.number_of_nodes()):
        f.write("{} \n".format(i))

    f.write("; \n")
    f.write("set E := \n")
    edges = list(G.edges)

    for i in range(G.number_of_edges()):
        f.write("{} {}\n".format(edges[i][0], edges[i][1]))
        f.write("{} {}\n".format(edges[i][1], edges[i][0]))

    f.write(";")
    f.close()


def make_ampl_run_file(name):
    f = open('C:\\ampl_mswin64\\models\\cover2.run', 'w')
    f.write("model models\cover.mod; \n")
    f.write("data models\{}.dat; \n".format(name))
    f.write("param d{N} default 0; \n")
    f.write("for{(i, j) in E}{ \n")
    f.write("let d[i]:= d[i] + 1;\n let d[j]:=d[j]+1; \n };")
    f.write("param L default 0; \n")
    f.write("param maxdeg default 0; \n")
    f.write("for{i in N}{ \n if (d[i] > maxdeg) then { \n let L := i; \n let maxdeg := d[i]; \n}; \n};")
    f.write("fix x[L] := 1; \n")
    f.write("option solver cplex; \n")
    f.write("option solver_msg 0; \n")
    f.write("solve; \n")
    f.write("printf \"\\n\"; \n")
    f.write("printf{i in N: x[i]==1} \"%d\\n\", i; \n")
    f.close()


def make_ampl_run_file2(name):
    f = open('C:\\ampl_mswin64\\models\\cover.run', 'w')
    f.write("model models\cover.mod; \n")
    f.write("data models\{}.dat; \n".format(name))
    f.write("option solver cplex; \n")
    f.write("option solver_msg 0; \n")
    f.write("solve; \n")
    f.write("printf \"\\n\"; \n")
    f.write("printf{i in N: x[i]==1} \"%d\\n\", i; \n")
    f.close()


def ampl_run(name, k, G, M):
    make_ampl_data_file(k, name, G)
    make_ampl_run_file(name)
    start = timeit.default_timer()
    os.chdir(r'C:\ampl_mswin64')
    result = subprocess.run(['ampl', 'models\\cover2.run'], capture_output=True, text=True)
    rows = result.stdout.split("\n", 1)[1]
    rows = rows.strip()
    rows = list(rows.split("\n"))
    rows = list(map(int, rows))
    M = delete_from_csr(M, rows, rows)
    minEigenvalue, v, err, iterations = invPowerMethod(M)
    stop = timeit.default_timer()
    runtime = stop - start
    return rows, minEigenvalue, runtime


def ampl_run2(name, k, G, M):
    make_ampl_data_file(k, name, G)
    make_ampl_run_file2(name)
    start = timeit.default_timer()
    os.chdir(r'C:\ampl_mswin64')
    result = subprocess.run(['ampl', 'models\\cover.run'], capture_output=True, text=True)
    rows = result.stdout.split("\n", 1)[1]
    rows = rows.strip()
    rows = list(rows.split("\n"))
    rows = list(map(int, rows))
    M = delete_from_csr(M, rows, rows)
    minEigenvalue, v, err, iterations = invPowerMethod(M)
    stop = timeit.default_timer()
    runtime = stop - start
    return rows, minEigenvalue, runtime


def importance(j, u, S):
    neighbors = [n for n in G[j]]
    lambdaSj = 0

    for i in neighbors:

        if i not in S:
            lambdaSj = lambdaSj + 2 * u[j] * u[i]



    return lambdaSj


def wokngo(M, G, h, epsilon):
    S = []
    N = G.nodes
    lambdaSj = []
    M = M.asfptype()
    J = []

    for i in range(h):
        w, u = scipy.sparse.linalg.eigsh(M, k=1, which='SM', tol=epsilon)
        #minEigenvalue, u, err, iterations = invPowerMethod(M)
        print("u", u)

        for j in N:

            if j not in S:
                lambdaSj.append(importance(j, u, S))
                J.append(j)
        print("lambdaSj", lambdaSj)
        l_max = max(lambdaSj)
        print("l_max", l_max)
        s = J[lambdaSj.index(l_max)]
        S.append(s)
        print(S)
    M = delete_from_csr(M, S, S)
    minEigenvalue, v, err, iterations = invPowerMethod(M)
    return S, minEigenvalue


def make_csv_file(name1, name2, dataset, k_hatar):

    header = ['k', 'naive', 'max deg', 'min deg', 'G_max_rad', 'G_min_rad', 'max(c max)', 'min(c max)', 'max(c min)', 'min(c min)', 'cover']
    print("h")
    f1 = open("{}.csv".format(name1), 'w')
    print("f1")
    f2 = open("{}.csv".format(name2), 'w')
    print("f2")

    writer1 = csv.writer(f1)
    writer2 = csv.writer(f2)

    G, A, L = generate_graph(dataset)
    print(nx.is_connected(G))
    print(L.toarray())

    G = G.subgraph(max(nx.connected_components(G), key=len))

    print(G.number_of_nodes())
    print(G)
    if k_hatar > 0:
        m = k_hatar
    else:
        m = G.number_of_nodes() / 10

    k = 1
    writer1.writerow(header)
    writer2.writerow(header)
    naive_runtime = 0
    print("valami")

    while k <= m:
        print(k)
        row1 = [k]
        row2 = [k]

        if naive_runtime < 2400:
            S, lam, naive_runtime = naive(L, k)

            row1.append(lam)
            row2.append(naive_runtime)
            print('naive:', naive_runtime)
        else:
            row1.append("null")
            row2.append("null")

        mineig, ind, runtime = max_deg(L, G, k)
        row1.append(mineig)
        row2.append(runtime)
        print('maxdeg:', runtime)

        mineig, ind, runtime = min_deg(L, G, k)
        row1.append(mineig)
        row2.append(runtime)

        minEigenvalue, indexes, runtime = gershgorin_fast(L, k, G, 3, 'max')
        row1.append(minEigenvalue)
        row2.append(runtime)
        print('gfast:', runtime)

        minEigenvalue, indexes, runtime = gershgorin_fast(L, k, G, 3, 'min')
        row1.append(minEigenvalue)
        row2.append(runtime)

        minEigenvalue, indexes, runtime = gershgorin_fast(L, k, G, 2, 'max')
        row1.append(minEigenvalue)
        row2.append(runtime)

        minEigenvalue, indexes, runtime = gershgorin_fast(L, k, G, 2, 'min')
        row1.append(minEigenvalue)
        row2.append(runtime)

        minEigenvalue, indexes, runtime = gershgorin_fast(L, k, G, 1, 'max')
        row1.append(minEigenvalue)
        row2.append(runtime)

        minEigenvalue, indexes, runtime = gershgorin_fast(L, k, G, 1, 'min')
        row1.append(minEigenvalue)
        row2.append(runtime)

        ind, mineig, runtime = ampl_run2('graf', k, G, L)
        row1.append(mineig)
        row2.append(runtime)
        print('cover:', runtime)

        writer1.writerow(row1)
        writer2.writerow(row2)

        k += 1

    f1.close()
    f2.close()

    return 0

k = 1
naive_eig = []
naive_time = []

maxrad_eig = []
maxrad_time = []

minrad_eig = []
minrad_time = []

ampl_eig = []
ampl_time = []

ampl2_eig = []
ampl2_time = []

G, A, L = generate_graph()
print(G)

gap_enum(G, L, 3)

# make_csv_file('proba_eigs', 'proba_time', "C:\\Diplomamunka\\brunson_revolution\\out.brunson_revolution_revolution", 19)
"""
while k <= 15:

    minEigenvalue, indexes, runtime = gershgorin_fast(L, k, G, 3, 'max')
    print("fast: ", minEigenvalue, indexes, runtime)

    ind, mineig, runtime = ampl_run('graf', k, G, L)
    print("ampl: ", mineig, ind, runtime)

    ind, mineig, runtime = ampl_run2('graf', k, G, L)
    print("ampl2: ", mineig, ind, runtime)
    #print(L)
    
    S, lam, runtime = naive(L, k)
    print(k)
    #print("naiveL", L)
    print("naive:", S, lam, runtime)
    mineig, ind, runtime = delete_max_rad(L, k)
    #print(L)

    print("maxrad:", ind, mineig, runtime)
    mineig, ind = delete_min_rad(L, k)
    print("minrad:", ind, mineig)

    mineig, ind = delete_cmax_min(L, k)
    print("min(c max):", ind, mineig)

    mineig, ind = delete_cmax_max(L, k)
    print("max(c max):", ind, mineig)

    mineig, ind = delete_cmin_max(L, k)
    print("max(c min):", ind, mineig)

    mineig, ind = delete_cmin_min(L, k)
    print("min(c min):", ind, mineig)
    A = delete_from_csr(L, ind, ind)
    A = A.toarray()
    n = len(A[0])
    deter = np.linalg.det(A)
    kozel = math.pow((n - 1) / n, (n - 1) / 2) * deter
    print(kozel)

    mineig, ind, runtime = min_deg(L, G, k)
    print("min deg:", mineig, ind)
    mineig, ind = max_deg(L, G, k)
    print("max deg:", mineig, ind, runtime)
    A = delete_from_csr(L, ind, ind)
    A = A.toarray()
    n = len(A[0])
    deter = np.linalg.det(A)
    normik = []
    for i in range(n):
        normik.append(np.linalg.norm(A[i]))
    cmin = min(normik)
    prodi = np.prod(normik)
    kozel = math.pow((n - 1) / n, (n - 1) / 2) * deter * cmin / prodi
    print(kozel)

    ind, mineig, runtime= ampl_run('graf', k, G, L)
    print("ampl: ", mineig, ind, runtime)

    A = delete_from_csr(L, ind, ind)
    A = A.toarray()
    n = len(A[0])
    deter = np.linalg.det(A)
    #print("det:", deter)
    #print("szam:", math.pow((n-1)/n, (n - 1)/2))
    kozel = math.pow((n-1)/n, (n - 1)/2)*deter
    print(kozel)

    ind, mineig, runtime = ampl_run2('graf', k, G, L)
    print("ampl2: ", mineig, ind, runtime)

    #ind, mineig = wokngo(L, G, k, 0.000001)
    #print('wokngo', mineig, ind)

   # mineig, ind = gap_enum(G, L, k)
    #print(mineig, ind)
   
    k += 1
"""
