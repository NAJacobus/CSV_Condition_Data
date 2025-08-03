import numpy as np
import scipy

def W(L):
    #Takes L expressed in eigenbasis
    dim = np.size(L, 0)
    L_squared = np.square(np.abs(L))
    result = np.square(np.abs(L))
    for i in range(dim):
        result[i,i] -= np.sum(L_squared[:, i])
    return result

def pops_to_full(soln):
    dim = np.size(soln)
    result = np.zeros((dim,dim), dtype = complex)
    for i in range(dim):
        result[i,i] = soln[i]
    return result


def get_full_W_matrix(L_list, cob_eig_to_original):
    """Assumes all operators in L_list are either dense or sparse"""
    cob_original_to_eig = np.transpose(np.conj(cob_eig_to_original)) #Change of basis should be unitary so take the adjoint as the inverse
    # print(cob_original_to_eig@cob_eig_to_original)
    L_cob = []
    if scipy.sparse.issparse(L_list[0]):
        #Uses sparse multiplication
        cob_eig_to_original_sparse = scipy.sparse.csr_matrix(cob_eig_to_original)
        cob_original_to_eig_sparse = scipy.sparse.csr_matrix(cob_original_to_eig)
        for i in L_list:
            L_cob.append((cob_original_to_eig_sparse@i@cob_eig_to_original_sparse).toarray())
            #Converts the Ls back to dense operators at the end
    else:
        for i in L_list:
            # print(i)
            L_cob.append(cob_original_to_eig@i@cob_eig_to_original) # Gives L operators in energy eigenbasis
    dim = np.shape(cob_eig_to_original)[1]
    W_matrix = np.zeros((dim, dim), dtype = complex)
    for i in L_cob:
        W_matrix += W(i)
    return W_matrix


def secular_approx(L_list, H):
    dim = np.size(H,0)
    vals, vecs = np.linalg.eigh(H)
    cob_eig_to_original = vecs
    cob_original_to_eig = np.transpose(np.conj(vecs))
    W_matrix = get_full_W_matrix(L_list, cob_eig_to_original)
    soln = scipy.linalg.null_space(W_matrix)
    if np.size(soln) != dim:
        print("Incorrect dimensions; choosing first column")
        soln = soln[:,0]
    soln = pops_to_full(soln)
    return cob_eig_to_original@soln@cob_original_to_eig # Changes to site basis; this isn't normalized

def secular_approx_sparse(L_list, H_sparse, num_k):
    """Returns normalized energy eigenstate populations, cob_original_to_eig, and cob_original_to_eig; the L list is dense (might be able to optimize by making it sparse)"""
    dim = H_sparse.shape[0]
    vals, vecs = scipy.sparse.linalg.eigs(H_sparse, k = num_k, which = "SR")
    cob_eig_to_original = vecs
    cob_original_to_eig = np.transpose(np.conj(vecs))
    W_matrix = get_full_W_matrix(L_list, cob_eig_to_original)
    soln = scipy.linalg.null_space(W_matrix)
    if np.size(soln) != num_k:
        print("Incorrect dimensions; choosing first column")
        soln = soln[:,0]
    soln = soln/np.sum(soln)
    return soln, vals, cob_original_to_eig

###############################################################
#The following functions are used to compute the partial trace without using the product basis; may be possible to optimize it further

def get_diag_product_entry(op, cob_right, coords: tuple):
    """Assumes op is a vector containing the diagonals of the diagonalized operator and that cob_right is unitary and written in an ON basis; calculates entry of adjoint(cob_right)@op@cob_right; calculating entries individually will ideally save memory"""
    col_coord = coords[1]
    col = cob_right[:, col_coord]
    #Might need to reshape op
    op_mult = op.reshape(np.shape(col))
    col = np.multiply(op_mult,col) #element-wise multiplication
    row_coord = coords[0]
    row = np.conj(cob_right[:, row_coord].transpose())
    return row@col

def sparse_ptrace_matrix(op, cob_right, structure, ptrace_index):
    """Obtains ptraced density operator in a different basis; assumes op is a vector containing the diagonals of the diagonalized operator and that cob_right is unitary and written in an ON basis; ptrace_index starts at 0; structure contains dimensions of subsystems (ex: num, vib, vib, vib, ...)
    Let m = dimension of subsystem and n = dim_total/m; storing all of the necessary coordinates will be O(n) in memory if we use iterators (but we might be able to reduce this using generators if necessary) and the general procedure will be O(n*m^2) in time; if we just take the diagonals it will be O(n*m) = O(dim)"""
    dims = structure[ptrace_index]
    result = np.empty((dims, dims), dtype = complex) #Can we get away with a different datatype?
    for i in range(dims):
        for j in range(dims):
            entry_val = 0
            coord_ptrace = (i,j)
            for k in get_coords_for_ptrace_summation(coord_ptrace, structure, ptrace_index):
                entry_val += get_diag_product_entry(op, cob_right, k)
            result[i,j] = entry_val
    return result


def sparse_ptrace_diag(op, cob_right, structure, ptrace_index):
    """See sparse_ptrace_matrix for more details"""
    dims = structure[ptrace_index]
    result = np.empty((dims), dtype = complex) #Can we get away with reals for the diagonals?
    for i in range(dims):
        entry_val = 0
        coord_ptrace = (i,i)
        for k in get_coords_for_ptrace_summation(coord_ptrace, structure, ptrace_index):
            entry_val += get_diag_product_entry(op, cob_right, k)
        result[i] = entry_val
    return result

def get_coords_for_ptrace_summation(subsystem_coord, structure, ptrace_index):
    """Currently returns a list for iteration; might be able to save on memory if we use a generator instead"""
    result = []
    other_subsys = structure[:]
    del other_subsys[ptrace_index]
    for k in get_cart_prod(other_subsys):
        place_vals_row = list(k[:ptrace_index]) + [subsystem_coord[0]] + list(k[ptrace_index:])
        place_vals_col = list(k[:ptrace_index]) + [subsystem_coord[1]] + list(k[ptrace_index:])
        row_index = convert_place_vals_to_int(place_vals_row, structure)
        col_index = convert_place_vals_to_int(place_vals_col, structure)
        result.append((row_index, col_index))
    return result

def convert_place_vals_to_int(place_vals, place_vals_system):
    """Assumes at least one place val"""
    int_vals = [1]
    for i in range(len(place_vals_system) - 1):
        int_vals.append(int_vals[-1]*place_vals_system[i + 1])
    int_vals.reverse()
    # print(int_vals)
    num = 0
    for i in range(len(int_vals)):
        num += place_vals[i]*int_vals[i]
    return num

def get_cart_prod(lst):
    """Returns tuples of Cartesian products for the sets range(lst[0]), range(lst[1]), ..."""
    """"""
    if len(lst) == 1:
        result = []
        for i in range(lst[0]):
            result.append((i,))
        return result
    else:
        initial_result = get_cart_prod(lst[1:])
        final_result = []
        for i in range(lst[0]):
            for j in initial_result:
                final_result.append((i,) + j)
        return final_result

def get_ptrace_diag_same_basis(pops, structure, ptrace_index):
    """Takes in a vector containing POPULATIONS (i.e. diagonal of the density operator; NOT the coefficients of a vector) written in the site basis; returns diagonal of partial trace
    We could probably accomplish the same effect by using the change of basis ptrace with the identity as the basis change but this might be faster"""
    dims = structure[ptrace_index]
    result = np.zeros(dims, dtype = float)
    for i in range(dims):
        coord = (i,i)
        for j in get_coords_for_ptrace_summation(coord, structure, ptrace_index):
            if j[0] == j[1]:
                result[i] += pops[j[0]]
    return result



# def get_one_ex
##########################################################


def get_tdep_secular(L_list, H, rho_0, t_list):
    vals, vecs = np.linalg.eigh(H)
    cob_eig_to_original = vecs
    cob_original_to_eig = np.transpose(np.conj(vecs))
    W_matrix = get_full_W_matrix(L_list, H)
    rho_0_ee = cob_eig_to_original@rho_0@cob_original_to_eig
    pops_i = np.diag(rho_0_ee)

    rho_t_list = []
    for i in t_list:
        pops_t = scipy.linalg.expm(i*W_matrix)@pops_i
        rho_t_ee = pops_to_full(pops_t)
        rho_t_site = cob_eig_to_original@rho_t_ee@cob_original_to_eig
        rho_t_list.append(rho_t_site)
    return rho_t_list

if __name__ == "__main__":
    from scipy.stats import unitary_group
    # x = unitary_group.rvs(3)
    # x_adj = np.conj(x.transpose())
    # A = np.random.rand(3,1)
    # B = np.diag(A.reshape(3,))
    # C = x_adj@B@x
    # D = np.zeros((3,3), dtype = complex)
    # for i in range(3):
    #     for j in range(3):
    #         D[i,j] = get_diag_product_entry(A, x, (i,j))
    # print(C - D)
    from qutip import Qobj
    E = np.random.rand(4)
    H = np.diag(E)
    x = unitary_group.rvs(18)
    x = x[:4, :]
    x_adj = np.conj(x.transpose())
    pti = 0
    F = Qobj(x_adj@H@x, dims = [[2,3,3], [2,3,3]]).ptrace(pti).full()
    G = sparse_ptrace_matrix(E, x, [2,3,3], pti)
    print(F - G)
    print(np.allclose(F, G))
    J = sparse_ptrace_diag(E, x, [2,3,3], pti)
    print(np.diag(F) - J)
    print(np.allclose(np.diag(F),J))


