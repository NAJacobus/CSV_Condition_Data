import HolsteinHamiltonian
import numpy as np
import scipy


#Note that in this convention the lower mode of the spin is first
def sig_z():
    op = np.zeros((2,2))
    op[0,0] = -1
    op[1,1] = 1
    return op

def sig_z_sparse():
    return scipy.sparse.coo_matrix(([-1,1], ([0,1], [0,1]))).tocsr()

def sig_x():
    op = np.zeros((2,2))
    op[1,0] = 1
    op[0,1] = 1
    return op

def sig_x_sparse():
    return scipy.sparse.coo_matrix(([1,1], ([1,0], [0,1]))).tocsr()

def RabiHamiltonian(vib, delta, lam, omega, sym_breaking = 0):
    op = (delta/2)*np.kron(sig_z(), np.eye(vib))
    op += omega*np.kron(np.eye(2), HolsteinHamiltonian.num(vib))
    op += lam*np.kron(sig_x(), HolsteinHamiltonian.a(vib) + HolsteinHamiltonian.a_dag(vib))
    ###Symmetry breaking term
    op += sym_breaking*np.kron(sig_x(), np.eye(vib))

    return op

def RabiHamiltonian_sparse(vib, delta, lam, omega, sym_breaking = 0):
    return scipy.sparse.csr_matrix(RabiHamiltonian(vib, delta, lam, omega, sym_breaking))

def sig_minus():
    op = np.zeros((2,2))
    op[0,1] = 1
    return op

def sig_plus():
    op = np.zeros((2,2))
    op[1,0] = 1
    return op

def sig_plus_sparse():
    op = scipy.sparse.coo_matrix(([1],([1], [0]))).tocsr()
    return op

def sig_minus_sparse():
    op = scipy.sparse.coo_matrix(([1],([0], [1]))).tocsr()
    return op

def JCHamiltonian(vib, delta, lam, omega):
    op = (delta/2)*np.kron(sig_z(), np.eye(vib))
    op += omega*np.kron(np.eye(2), HolsteinHamiltonian.num(vib))
    op += lam*(np.kron(sig_minus(), HolsteinHamiltonian.a_dag(vib)) + np.kron(sig_plus(), HolsteinHamiltonian.a(vib)))
    return op

def JCHamiltonian_sparse(vib, delta, lam, omega):
    return scipy.sparse.csr_matrix(JCHamiltonian(vib, delta, lam, omega))