import numpy as np
import math
import scipy
import sparse_matrix_fxns


class SizeError(Exception):
    "Invalid Dimensions"

def supervector(rho):
    # Convention: Column Stacking
    rows, columns = rho.shape
    if rows != columns:
        raise SizeError
    super = np.zeros((rows*columns, 1), dtype = rho.dtype)
    for i in range(columns):
        super[rows*i:rows*(i+1), 0] = rho[:, i]
    return super

def unsup(rho):
    """(Added several months after code was written): I think we need to pass in an n by 1 column vector (not a row vector) or else an error will occur"""
    length = (math.sqrt(rho.size))
    if length % 1 != 0:
        raise SizeError
    length = int(length)
    m = np.zeros((length, length), dtype = rho.dtype)
    for i in range(length):
        m[:, i] = rho[length*i:length*(i+1), 0]
    return m

def left_multiply(op):
    rows, columns = op.shape
    if rows != columns:
        raise SizeError
    return np.kron(np.eye(rows), op)

def left_multiply_sparse(sparse_op, form = "csr"):
    rows, columns = sparse_op.shape
    if rows != columns:
        raise SizeError
    # return scipy.sparse.kron(scipy.sparse.eye(rows, format = form), sparse_op, format = form) #This is still taking up too much memory
    return sparse_matrix_fxns.custom_sparse_kron(scipy.sparse.eye(rows, format = form), sparse_op)


def right_multiply(op):
    rows, columns = op.shape
    if rows != columns:
        raise SizeError
    return np.kron(op.transpose(), np.eye(rows))

def right_multiply_sparse(sparse_op, form = "csr"):
    rows, columns = sparse_op.shape
    if rows != columns:
        raise SizeError
    # return scipy.sparse.kron(sparse_op.transpose(), scipy.sparse.eye(rows, format = form), format = form)
    return sparse_matrix_fxns.custom_sparse_kron(sparse_op.transpose(), scipy.sparse.eye(rows, format = form))


