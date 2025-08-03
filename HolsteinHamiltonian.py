import numpy as np
import math
import sys
import scipy
np.set_printoptions(threshold=sys.maxsize)


#Convention: First level is lower in energy; tensor structure is electronic x lowest level's oscillator x second lowest ..., so the first rows correspond to the ground electronic and all states in the groud state except for the highest energy states which change

class PositionError(Exception):
    "Positions should be from 1 to num, not 0 to num - 1"

def nn_coupling(n, J):
    off_diag = J*np.eye(n - 1)
    off_diag = np.hstack((off_diag, np.zeros((n-1, 1))))
    off_diag = np.vstack((np.zeros(n), off_diag))
    off_diag = off_diag + off_diag.transpose()
    return off_diag

def nn_coupling_sparse(n, J):
    rows = range(n - 1)
    columns = range(1, n)
    data = (n - 1) * [J]
    off_diag = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (n, n)).tocsr()
    off_diag = off_diag + off_diag.transpose()
    return off_diag

def lin_bias(n, delta):
    op = np.zeros((n,n))
    for i in range(n):
        op[i,i] = delta*((i + 1) - (n+1)/2)
    return op

def lin_bias_sparse(n, delta):
    rows = range(n)
    columns = range(n)

    #Normal version of energy labeling
    # data = delta*(np.subtract(list(range(1, n+1)), n * [(n + 1)/2]))

    #Version of energy labeling used in paper
    data = delta*(np.array(list(range(1, n + 1))))
    print("Using alternative energy labeling")


    op = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (n, n)).tocsr()
    return op

def a(n):
    #Note we do not have a sqrt(2) factor in our annihilation operator
    if n == 1:
        return np.array([0])
    else:
        op = np.zeros((n-1,n-1))
        for i in range(n-1):
            op[i,i] = math.sqrt(i+1)
        op = np.hstack((np.zeros((n-1, 1)), op))
        op = np.vstack((op, np.zeros(n)))
        return op

def a_sparse(n):
    if n == 1:
        return scipy.sparse.coo_matrix(([], ([], [])), shape = (1,1)).tocsr()
    else:
        rows = range(n - 1)
        columns = range(1, n)
        data = np.sqrt(list(range(1, n)))
        op = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (n, n)).tocsr()
    return op


def a_dag(n):
    if n == 1:
        return np.array([0])
    return a(n).transpose()

def a_dag_sparse(n):
    return a_sparse(n).transpose()

def num(n):
    if n == 1:
        return np.array([1]) #Should this be a zero?
    else:
        return a_dag(n)@a(n)

def num_sparse(n):
    rows = range(n)
    columns = range(n)
    data = range(n)
    op = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (n, n)).tocsr()
    return op

def tensored_ladder_op(position, n, vib, type):
    if position == 0:
        raise PositionError
#Position ranges from 1 to n
    if type == "a":
        op = a(vib)
    elif type == "a_dag":
        op = a_dag(vib)
    elif type == "num":
        op = num(vib)
    op = np.kron(np.eye(int(n*vib**(position - 1))), op)
    op = np.kron(op, np.eye(int(vib**(n - position))))
    return op

def tensored_ladder_op_sparse(position, n, vib, type):
    if type == "a":
        op = a_sparse(vib)
    elif type == "a_dag":
        op = a_dag_sparse(vib)
    elif type == "num":
        op = num_sparse(vib)
    if op.getformat() != "csr":
        op = op.tocsr()
        # raise errors.CSR_Error_1
    op = scipy.sparse.kron(scipy.sparse.eye(int(n*vib**(position - 1)), format = "csr"), op, format = "csr")
    op = scipy.sparse.kron(op, scipy.sparse.eye(int(vib**(n - position)), format = "csr"), format = "csr")
    if op.getformat() != "csr":
        print(op.getformat())
        op = op.tocsr()
        # raise errors.CSR_Error_2
    return op

def site_projector(position, num, vib):
#Position from 1 to num
    op = np.zeros((num, num))
    op[(position - 1, position - 1)] = 1
    op = np.kron(op, np.eye(vib**num))
    return op

def site_projector_sparse(position, num, vib):
    rows = [position - 1]
    columns = [position - 1]
    data = [1]
    op = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (num, num)).tocsr()
    op = scipy.sparse.kron(op, scipy.sparse.eye(vib**num, format = "csr"), format = "csr")
    if op.getformat() != "csr":
        op = op.tocsr()
        # raise errors.CSR_Error_2
    return op

def HolsteinHam(omega, J, num, vib, delta, lam):
    Ham = np.kron(nn_coupling(num, J), np.eye(vib**num))
    Ham = Ham + np.kron(lin_bias(num, delta), np.eye(vib**num))
    for i in range(num):
        position = i + 1
        #Depending on convention divide by sqrt(2) in the next line
        Ham = Ham + lam*(tensored_ladder_op(position, num, vib, "a") + tensored_ladder_op(position, num, vib, "a_dag"))@site_projector(position, num, vib)
        #Normal
        Ham = Ham + omega*(tensored_ladder_op(position, num, vib, "num") + 1/2*np.eye(num*vib**num))
        #Diff omega
        # if position == 1:
        #     Ham = Ham + (0.8)*omega*(tensored_ladder_op(position, num, vib, "num") + 1/2*np.eye(num*vib**num))
        # if position == 2:
        #     Ham = Ham + (1.2)*omega*(tensored_ladder_op(position, num, vib, "num") + 1/2*np.eye(num*vib**num))
        # print("diff omega")
    return Ham

def HolsteinHam_sparse(omega, J, num, vib, delta, lam, anharm = 0):
    Ham = scipy.sparse.kron(nn_coupling_sparse(num, J), scipy.sparse.eye(vib**num, format = "csr"), format = "csr")
    Ham = Ham + scipy.sparse.kron(lin_bias_sparse(num, delta), scipy.sparse.eye(vib**num, format = "csr"), format = "csr")
    for i in range(num):
        position = i + 1
        Ham = Ham + lam*(tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib, "a_dag"))@site_projector_sparse(position, num, vib)
        Ham = Ham + omega*(tensored_ladder_op_sparse(position, num, vib, "num") + 1/2*scipy.sparse.eye(num*vib**num, format = "csr"))
        # Ham = Ham + anharm*(tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib, "a_dag"))@(tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib, "a_dag"))@(tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib, "a_dag"))@(tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib, "a_dag"))
        # Ham = Ham + anharm * (tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib, "a_dag"))
        # Ham = Ham + anharm * (tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib,"a_dag"))@(tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib,"a_dag"))
        Ham = Ham + anharm * (tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib,"a_dag"))@(tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib,"a_dag"))@(tensored_ladder_op_sparse(position, num, vib, "a") + tensored_ladder_op_sparse(position, num, vib,"a_dag"))/math.sqrt(8)

            #adds an epsilon*(a + a^\dagger)^4 anharmonicity
        if anharm != 0:
            print("anharmonic")
    if Ham.getformat() != "csr":
        Ham = Ham.tocsr()
        # raise errors.CSR_Error_2
    return Ham


def HolsteinHamWGround(omega, J, num, vib, delta, lam):
    Ham = HolsteinHam(omega, J, num, vib, delta, lam)
    Ham = np.pad(Ham, [(1,0), (1,0)])
    # Ham[0,0] = -1*delta*((num+1)/2) + (1/2)*num*omega #Is this the right energy?
    return Ham

def HolsteinHamWGround_sparse(omega, J, num, vib, delta, lam, anharm = 0):
    Ham_coo = HolsteinHam_sparse(omega, J, num, vib, delta, lam, anharm = anharm).tocoo()
    data = Ham_coo.data
    rows = list(Ham_coo.row)
    for i in range(len(rows)):
        rows[i] += 1
    columns = list(Ham_coo.col)
    for i in range(len(columns)):
        columns[i] += 1
    Ham = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (num*vib**num + 1, num*vib**num + 1))
    return Ham

if __name__ == "__main__":
    n = 3
    v = 7
    A = HolsteinHam(1, 0.001, n, v, 0.1, 1)
    B = HolsteinHamWGround(1, 0.001, n, v, 0.1, 0.02)
    vals, vecs = scipy.linalg.eig(B)
    vals = list(vals)
    vals.sort()
    print(vals)
    print(A)
    C = HolsteinHam_sparse(1, 0.001, n, v, 0.1, 1)
    D = HolsteinHamWGround_sparse(1, 0.001, n, v, 0.1, 0.02)
    print(np.allclose(A, C.toarray(), atol = 1e-17))
    print(np.allclose(B, D.toarray(), atol = 1e-17))
