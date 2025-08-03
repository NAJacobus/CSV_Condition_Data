import numpy as np
import scipy
import math
import get_dens_op
from qutip import Qobj

def get_ee(rho):
    """Calculates the Von Neumann Entropy of density matrix rho (using base 2 for the log), taking the sum over \lambda_i \log_2(\lambda_i) over the positive eigenvalues (note that we ignore the unphysical negative eigenvalues that may arise from Redfield"""
    eigs = scipy.linalg.eigvalsh(rho)
    pos_eigs = np.array([i for i in eigs if i > 0])
    logged = np.log2(pos_eigs)
    return -1*np.dot(pos_eigs, logged)

##NOTE: I do not think this approximation is that useful, at least for the Rabi model, because it is very unstable with respect to numerical errors - see the June 4th notes in the 2024 Summer Research Document
def get_average_mixed_bipartite_ee(full_rho, qutip_dims, trace_index):
    """Determines an entanglement of formation using the entanglement entropies of the eigenvalues of rho;
    however this provides an upper bound to the actual entropy of formation which uses an infimum
    qutip_dims supplies the bipartition while trace_index provides the indices to be kept
    IMPORTANT NOTE: I do not think this approximation is that useful, at least for the Rabi model, because it is very unstable with respect to numerical errors - see the June 4th notes in the 2024 Summer Research Document"""
    weights, vecs = scipy.linalg.eig(full_rho)
    ee_tot = 0
    for j in range(len(weights)):
        if weights[j] > 0:
            state_space_dim = len(weights)
            rho_j = get_dens_op.pure_dens_op(vecs[:,j].reshape(state_space_dim, 1))
            rho_j_red = Qobj(rho_j, dims = [qutip_dims, qutip_dims]).ptrace(trace_index).full()
            ee_j = get_ee(rho_j_red)
            ee_tot += weights[j]*ee_j
    return ee_j

