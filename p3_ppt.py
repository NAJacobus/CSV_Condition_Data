import numpy as np
import qutip
import scipy

def negativity_all_subdivisions(rho, qutip_dims, thres, return_negativity_sum = False) -> bool:
    """Dense operations; returns True if there is a negative eigenvalue of the partial transpose; False otherwise
    If return_negativity_sum = True, also returns the total negativity obtained from going over all the subdivisions (note we don't go over the complement transpose since it should have the same eigenvalues I think)"""
    #Checks all partitions for violations of negativity; as a consequence worse-case time complexity is O(2^(n-1)), where n is the size of the input (though I do not know if this increases the sensitivity; I think we don't have to do the complement of a binary string since the tranpose should have the same eigenvalues - hence we can just do
    n = len(qutip_dims)
    thres_check = False
    neg = 0
    for i in range(2**(n - 1)):
        bipartition = bin(i)
        mask = list(bipartition)[2:] + [0] #Selects among all but the last subsystem for partial transposition in order to check if P3-PPT condition is violated
        for j in range(len(mask)):
            mask[j] = int(mask[j])
        # print(mask)
        partial_transpose = qutip.partial_transpose(qutip.Qobj(rho, dims = [qutip_dims, qutip_dims]), mask).full()
        evals = scipy.linalg.eig(partial_transpose)[0]
        for k in evals:
            if k < -1*thres:
                # print(k)
                thres_check = True
                neg += k
    if return_negativity_sum:
        return thres_check, -1*neg
    else:
        return thres_check



zero_tol_for_counting = 1e-12
def num_negativity(rho_pt):
    #Returns negativity, assuming rho_pt is Hermitian
    evals = scipy.linalg.eigvalsh(rho_pt)
    neg = 0
    neg_count = 0
    for k in evals:
        if k < 0:
            neg -= k
        if k < -1*zero_tol_for_counting:
            neg_count += 1
    # if neg_count >= 2:
    #     print(f"{neg_count} evals > {zero_tol_for_counting}")
    return neg

def num_negativity_sparse(rho_pt, num_k):
    #Calculates an approximation of the negativity using num_k Lanczos eigenvectors
    rho_sparse = scipy.sparse.csr_matrix(rho_pt)
    evals = scipy.sparse.linalg.eigsh(rho_sparse, k = num_k, which = "SA")[0]
    neg = 0
    neg_count = 0
    for k in evals:
        if k < 0:
            neg -= k
        if k < -1 * zero_tol_for_counting:
            neg_count += 1
    if neg_count >= 2:
        print(f"{neg_count} evals > {zero_tol_for_counting}")
    return neg

