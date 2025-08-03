import numpy as np
import scipy
import math
import copy
from matplotlib import pyplot as plt
import cmath


beta_gap_size_cutoff = 200
zero_tol = 1e-15



###Full Redfield; no approximations - scales like N^4
###Tested this on a two-level system; it seems to work since we get detailed balance?
###NOTE: The full and regular Redfield can take in multiple system bath coupling terms, but as implemented right now it assumes the bath cross-correlation functions vanish (e.g. if we have two separate harmonic baths both coupling via the x coordinate) because we don't include the cross terms - more generally you would need this, see eg Nitzan problem 10.17 on page 376
def get_gamma_index(sys_op_ee, sys_evals, spec_fxn, beta, m,n,o,p, atten_fac = 1):
    """Assumes the 0-frequency correlation functions vanish"""
    gap = sys_evals[o] - sys_evals[p]
    if gap > zero_tol:
        if beta*gap >= beta_gap_size_cutoff:
            # return 0
            return sys_op_ee[m, n] * sys_op_ee[o, p] * spec_fxn(gap) * atten_fac * math.exp(-1*beta*gap)
        else:
            return sys_op_ee[m, n] * sys_op_ee[o, p] * spec_fxn(gap) * (atten_fac / (math.exp(beta * gap) - 1))
    elif gap < -1*zero_tol:
        if beta*gap <= -1*beta_gap_size_cutoff:
            return sys_op_ee[m, n] * sys_op_ee[o, p]* atten_fac #is this the correct 0-temp limit, since it is 1 - (1-atten fac), and corresponds to reducing the transition rate by the attenuation factor?
        else:
            return sys_op_ee[m, n] * sys_op_ee[o, p] * spec_fxn(-1 * gap) * (math.exp(-1 * beta * gap)*atten_fac)/ (math.exp(-1 * beta * gap) - 1)
    else:
        return 0

def get_gamma_from_Aijkl(Aijkl_ee, sys_evals, spec_fxn, beta, m,n,o,p, atten_fac = 1):
    """Lets you get the gamma indices if all you have are the A_ijkl instead of the identities of the system coupling operators themselves"""
    gap = sys_evals[o] - sys_evals[p]
    if gap > zero_tol:
        if beta * gap >= beta_gap_size_cutoff:
            # return 0
            return Aijkl_ee[m,n,o,p] * spec_fxn(gap) * atten_fac * math.exp(-1*beta*gap)
        else:
            return Aijkl_ee[m,n,o,p] * spec_fxn(gap) * (atten_fac / (math.exp(beta * gap) - 1))
    elif gap < -1*zero_tol:
        if beta * gap <= -1*beta_gap_size_cutoff:
            return Aijkl_ee[m,n,o,p] * atten_fac
        else:
            return Aijkl_ee[m,n,o,p] * spec_fxn(-1 * gap) * (math.exp(-1 * beta * gap)*atten_fac) / (math.exp(-1 * beta * gap) - 1)
    else:
        return 0


def get_gamma_tensor(sys_op_ee, sys_evals, spec_fxn, beta, atten_fac = 1, type = "op"):
    """System Operator should be given in energy eigenbasis; returns gamma tensor in energy eigenbasis; note that this uses that the mean bath operator vanishes for x coordinate coupling so we don't need to include it"""
    """Use type = "op" if sys_op_ee contains the system operator, use type = "A" if sys_op_ee is the A_ijkl tensor"""
    dim = len(sys_evals)
    gamma = np.zeros((dim,dim,dim,dim), dtype = complex)
    for m in range(len(sys_evals)):
        for n in range((len(sys_evals))):
            for o in range((len(sys_evals))):
                for p in range((len(sys_evals))):
                    if type == "op":
                        gamma[m,n,o,p] = get_gamma_index(sys_op_ee, sys_evals, spec_fxn, beta, m, n, o, p, atten_fac)
                    elif type == "A":
                        gamma[m,n,o,p] = get_gamma_from_Aijkl(sys_op_ee, sys_evals, spec_fxn, beta, m, n, o, p, atten_fac)
                    else:
                        raise Exception("type should be either `op` or `A`")
    return gamma
def get_Redfield_tensor(sys_op_ee, sys_evals, spec_fxn, beta, atten_fac = 1, type = "op"):
    """System Operator should be given in energy eigenbasis; returns Redfield tensor in energy eigenbasis"""
    dim = len(sys_evals)
    R = np.zeros((dim, dim, dim, dim), dtype = complex)
    gamma = get_gamma_tensor(sys_op_ee, sys_evals, spec_fxn, beta, atten_fac, type)
    for m in range(len(sys_evals)):
        for n in range((len(sys_evals))):
            for o in range((len(sys_evals))):
                for p in range((len(sys_evals))):
                    R[m,n,o,p] = gamma[p,n,m,o] + np.conj(gamma[o,m,n,p])
                    if n == p:
                        for q in range(len(sys_evals)):
                            R[m, n, o, p] -= gamma[m,q,q,o]
                    if m == o:
                        for q in range(len(sys_evals)):
                            R[m, n, o, p] -= np.conj(gamma[n,q,q,p])

    return R

def get_relaxation_derivative(sys_op_ee_list, sys_evals, spec_fxn_list, beta_list, atten_fac_list = None, type_list = None):
    """Gives drho/dt in the system energy eigenbasis, where we arrange rho as a supervector as column 1, then column 2, ..."""
    if atten_fac_list is None:
        atten_fac_list = len(sys_op_ee_list)*[1]
    if type_list is None:
        type_list = len(sys_op_ee_list)*["op"]
    dim = len(sys_evals)
    rho_dot = np.zeros((dim**2, dim**2), dtype = complex)
    R = np.zeros((dim, dim, dim, dim), dtype = complex)
    for j in range(len(sys_op_ee_list)):
        R += get_Redfield_tensor(sys_op_ee_list[j], sys_evals, spec_fxn_list[j], beta_list[j], atten_fac_list[j], type_list[j])
    for m in range(len(sys_evals)):
        for n in range((len(sys_evals))):
            for o in range((len(sys_evals))):
                for p in range((len(sys_evals))):
                    row_index = n*dim + m
                    col_index = p*dim + o
                    if row_index == col_index:
                        gap = sys_evals[m] - sys_evals[n]
                        rho_dot[row_index, col_index] = -1j*gap
                    rho_dot[row_index, col_index] += R[m,n,o,p]
    return rho_dot

###Secular Redfield, where we decouple the population and coherence dynamics (should hold best if we assume the difference between any two frequencies omega_ab and omega_cd are large

def get_secular_gamma_matrix(sys_op_ee, sys_evals, spec_fxn, beta, atten_fac = 1):
    """Assembles a matrix whose ab entry is Gamma_{ab,ba}"""
    dim = len(sys_evals)
    mat = np.zeros((dim, dim), dtype = complex)
    for m in range(dim):
        for n in range(dim):
            mat[m,n] = get_gamma_index(sys_op_ee, sys_evals, spec_fxn, beta, m, n, n, m, atten_fac)
    return mat

def get_gamma_index_from_Aij(Aij_ee, sys_evals, spec_fxn, beta, m,n, atten_fac = 1):
    """Instead of using the A_ijkl as the input it uses the dim by dim matrix A_ij with entries A_ijji"""
    gap = sys_evals[n] - sys_evals[m]
    if gap > zero_tol:
        if beta * gap >= beta_gap_size_cutoff:
            # return 0
            return Aij_ee[m,n] * spec_fxn(gap) * atten_fac * math.exp(-1*beta*gap)
        else:
            return Aij_ee[m,n] * spec_fxn(gap) * (atten_fac / (math.exp(beta * gap) - 1))
    elif gap < -1*zero_tol:
        if beta * gap <= -1*beta_gap_size_cutoff:
            return Aij_ee[m,n]* atten_fac
        else:
            return Aij_ee[m,n] * spec_fxn(-1 * gap) * (math.exp(-1 * beta * gap)*atten_fac) / (math.exp(-1 * beta * gap) - 1)
    else:
        return 0

def get_secular_gamma_from_Aij(Aij_ee, sys_evals, spec_fxn, beta, atten_fac = 1):
    """Assembles a matrix whose ab entry is gamma_abba, but now using A_ijkl instead of the system coupling operators themselves; A_ij should be a dim by dim matrix whose ij entry is A_ijji (note in this implementation of full secular we only need the ijji elements of A"""
    dim = len(sys_evals)
    mat = np.zeros((dim, dim), dtype=complex)
    for m in range(dim):
        for n in range(dim):
            mat[m, n] = get_gamma_index_from_Aij(Aij_ee, sys_evals, spec_fxn, beta, m, n, atten_fac)
    return mat
def get_secular_t_dep(sys_op_ee_list, sys_evals, spec_fxn_list, beta_list, atten_fac_list = None, type_list = None):
    """Assumes the 0-frequency correlation function vanishes, so that type t2 relaxation of the coherences is ignored"""
    if atten_fac_list is None:
        atten_fac_list = len(sys_op_ee_list)*[1]
    if type_list is None:
        type_list = len(sys_op_ee_list)*["op"]
    dim = len(sys_evals)
    gammas = np.zeros((dim,dim), dtype = complex)
    for j in range(len(sys_op_ee_list)):
        if type_list[j] == "op":
            gammas += get_secular_gamma_matrix(sys_op_ee_list[j], sys_evals, spec_fxn_list[j], beta_list[j], atten_fac_list[j])
        elif type_list[j] == "A":
            gammas += get_secular_gamma_from_Aij(sys_op_ee_list[j], sys_evals, spec_fxn_list[j], beta_list[j], atten_fac_list[j])
        else:
            raise Exception("type should be either `op` or `A`")
    pop_derivative = np.zeros((dim,dim), dtype = float) #Only need floats because in secular the population dynamics are pure relaxation so all entries are real
    for m in range(dim):
        for n in range(dim):
            # if n >= 14:
            #     print("here")
            pop_derivative[m,n] += 2*np.real(gammas[n,m])
            pop_derivative[m,m] -= 2*np.real(gammas[m,n])
    coh_derivatives = {} #Each entry is the derivative of an individual coherence; Note we are only including the type t1 relaxation, not t2
    for m in range(dim):
        for n in range(m):
            d_mn = -1j*(sys_evals[m] - sys_evals[n])
            for q in range((len(sys_evals))):
                d_mn -= (gammas[m,q] + np.conj(gammas[n,q]))
            index = (m,n)
            coh_derivatives[index] = d_mn
    return pop_derivative, coh_derivatives

def prop_sec_pops(pop_derivative, t_list, p_0):
    pop_list = np.copy(p_0)
    for t in t_list:
        pop_t = scipy.linalg.expm(t*pop_derivative)@p_0
        pop_list = np.concatenate((pop_list, pop_t), axis = 1)
    return pop_list[:, 1:]

def prop_sec_coh(coh_derivatives, t_list, coh_0):
    coh_dictionaries = []
    for t in t_list:
        coh_t = {}
        for key in coh_derivatives:
            coh_t[key] = cmath.exp(coh_derivatives[key]*t)*coh_0[key]
        coh_dictionaries.append(copy.deepcopy(coh_t))
    return coh_dictionaries

def convert_rho_to_p_and_coh(rho):
    """Assumes rho is symmetric in the given basis (i.e. it's an ON basis)"""
    p = np.diag(rho)
    dim = np.size(p)
    p = p.reshape(dim,1)
    coh_dic = {}
    for m in range(np.size(p)):
        for n in range(m):
            coh_dic[(m,n)] = rho[m,n]
    return p, coh_dic

def full_sec_prop(pop_derivative, coh_derivatives, t_list, p_0, coh_0):
    dim = np.size(p_0)
    rho_list = []
    pop_list = prop_sec_pops(pop_derivative, t_list, p_0)
    coh_list = prop_sec_coh(coh_derivatives, t_list, coh_0)
    count = 0
    for t in t_list:
        rho_t = np.diag(pop_list[:,count]).astype("complex")
        for index in coh_derivatives:
            row = index[0]
            col = index[1]
            rho_t[row, col] = coh_list[count][index]
            rho_t[col, row] = np.conj(coh_list[count][index])
        rho_list.append(rho_t)
        count += 1
    return rho_list










if __name__ == "__main__":
    #Test on 2-level system
    eps = 1
    H = np.zeros((2,2))
    H[0,0] = eps
    sig_x = np.zeros((2,2), dtype = complex)
    sig_x[1,0] = 1
    sig_x[0,1] = 1
    beta = 1
    diss = 0.1
    def J(x):
        return diss * x * math.exp(-1 * abs(x) / diss)
    rho_0 = np.array([0.2,50 + 27j, 50 - 27j,0.8])
    rho_0 = rho_0.reshape(4,1)
    sys_evals = [eps, 0]


    ###Full Redfield Test
    rho_dot = get_relaxation_derivative([sig_x], sys_evals, [J], [beta])
    t_list = np.linspace(0,2e6,50)
    top_pops = []
    Re_coh = []
    Im_coh = []
    rho_nonsec_t_list = []
    for t in t_list:
        rho_t = scipy.linalg.expm(rho_dot*t)@rho_0
        top_pops.append(rho_t[0])
        Re_coh.append(np.real(rho_t[1]))
        Im_coh.append(np.imag(rho_t[1]))
        rho_t = np.reshape(rho_t, (2,2)).transpose()
        rho_nonsec_t_list.append(rho_t)
    # fig, ax = plt.subplots()
    # ax.plot(t_list, top_pops, c = "blue", label = r"$\rho_{00}$; Non-Secular")

    # plt.axhline(y= math.exp(-beta*eps), c = "r", label = r"$e^{-\beta \Delta E}$")



    ###Secular Redfield Test
    p_0 = np.array([rho_0[0],rho_0[3]]).reshape((2,1))
    coh_0 = {}
    coh_0[(1,0)] = rho_0[1]
    pop_derivative, coh_derivatives = get_secular_t_dep([sig_x], sys_evals, [J], [beta])
    pop_t = prop_sec_pops(pop_derivative, t_list, p_0)
    coh_t = prop_sec_coh(coh_derivatives, t_list, coh_0)
    Re_coh_sec = []
    Im_coh_sec = []
    for index in range(len(t_list)):
        Re_coh_sec.append(np.real(coh_t[index][(1,0)]))
        Im_coh_sec.append(np.imag(coh_t[index][(1,0)]))
    # ax.plot(t_list, pop_t[0,:], c = "r", label = r"$\rho_{00}$, Secular")
    # # plt.axhline(y= math.exp(-beta*eps), c = "r", label = r"$e^{-\beta \Delta E}$")
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$P_{ex}$")
    # plt.legend()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(t_list, Re_coh, c = "purple", label = r"$\text{Re}(\rho_{10})$; Non-Secular")
    # ax.plot(t_list, Re_coh_sec, c = "g", label = r"$\text{Re}(\rho_{10})$, Secular")
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$Re(\rho_{10})$")
    # plt.legend()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(t_list, Im_coh, c = "cyan", label = r"$\text{Im}(\rho_{10})$; Non-Secular")
    # ax.plot(t_list, Im_coh_sec, c = "black", label = r"$\text{Im}(\rho_{10})$, Secular")
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$Im(\rho_{10})$")
    # plt.legend()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # delta_pops = np.array(top_pops).reshape((50)) - pop_t[0,:]
    # ax.plot(t_list, delta_pops, c = "g", label = r"$\Delta\rho_{00}$")
    # ax.plot(t_list, np.array(Re_coh) - np.array(Re_coh_sec), c = "b", label = r"$\Delta(\text{Re}(\rho_{10}))$")
    # ax.plot(t_list, np.array(Im_coh) - np.array(Im_coh_sec), c = "r", label = r"$\Delta(\text{Im}(\rho_{10}))$")
    # plt.legend()

    rho_t_sec = full_sec_prop(pop_derivative, coh_derivatives, t_list, p_0, coh_0)
    tot_err = 0
    for index in range(len(t_list)):
        tot_err += np.max(np.abs(rho_t_sec[index] - rho_nonsec_t_list[index]))
    print(tot_err)

    for index in range(len(t_list)):
        print(rho_t_sec[index])
        print(convert_rho_to_p_and_coh(rho_t_sec[index]))

    ###Testing the A_ijkl versions
    A_ijkl = np.empty((2,2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    A_ijkl[i,j,k,l] = sig_x[i,j]*sig_x[k,l]
    rho_dot_A = get_relaxation_derivative([A_ijkl], sys_evals, [J], [beta], type_list = ["A"])
    top_pops_A = []
    Re_coh_A = []
    Im_coh_A = []
    rho_nonsec_A_t_list = []
    for t in t_list:
        rho_t_A = scipy.linalg.expm(rho_dot_A * t) @ rho_0
        top_pops_A.append(rho_t_A[0])
        Re_coh_A.append(np.real(rho_t_A[1]))
        Im_coh_A.append(np.imag(rho_t_A[1]))
        rho_t_A = np.reshape(rho_t_A, (2, 2)).transpose()
        rho_nonsec_A_t_list.append(rho_t_A)

    ###Secular Redfield Test, A version
    A_ij_matrix = np.empty((2,2))
    for i in range(2):
        for j in range(2):
            A_ij_matrix[i,j] = sig_x[i,j]*sig_x[j,i]
    pop_derivative_A, coh_derivatives_A = get_secular_t_dep([A_ij_matrix], sys_evals, [J], [beta], type_list = ["A"])
    pop_t_A = prop_sec_pops(pop_derivative_A, t_list, p_0)
    coh_t_A = prop_sec_coh(coh_derivatives_A, t_list, coh_0)
    Re_coh_sec_A = []
    Im_coh_sec_A = []
    for index in range(len(t_list)):
        Re_coh_sec_A.append(np.real(coh_t_A[index][(1, 0)]))
        Im_coh_sec_A.append(np.imag(coh_t_A[index][(1, 0)]))

    fig, ax = plt.subplots()
    ax.plot(t_list, top_pops, c = "blue", label = r"$\rho_{00}$; Non-Secular")
    ax.plot(t_list, top_pops_A, c = "g", label = r"$\rho_{00}$; Non-Secular, A version")
    ax.plot(t_list, pop_t[0, :], c="r", label=r"$\rho_{00}$, Secular")
    ax.plot(t_list, pop_t_A[0,:], c = "purple", label = r"$\rho_{00}$, Secular, A version")
    # plt.axhline(y= math.exp(-beta*eps), c = "r", label = r"$e^{-\beta \Delta E}$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$P_{ex}$")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()

    ax.plot(t_list, Re_coh, c="blue", label=r"$\text{Re}(\rho_{10})$; Non-Secular")
    ax.plot(t_list, Re_coh_A, c="g", label=r"$\text{Re}(\rho_{10})$; Non-Secular, A version")
    ax.plot(t_list, Re_coh_sec, c="r", label=r"$\text{Re}(\rho_{10})$, Secular")
    ax.plot(t_list, Re_coh_sec_A, c="purple", label=r"$\text{Re}(\rho_{10})$, Secular, A version")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$Re(\rho_{10})$")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(t_list, Im_coh, c="blue", label=r"$\text{Im}(\rho_{10})$; Non-Secular")
    ax.plot(t_list, Im_coh_A, c="g", label=r"$\text{Im}(\rho_{10})$; Non-Secular, A version")
    ax.plot(t_list, Im_coh_sec, c="r", label=r"$\text{Im}(\rho_{10})$, Secular")
    ax.plot(t_list, Im_coh_sec, c="purple", label=r"$\text{Im}(\rho_{10})$, Secular, A version")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$Im(\rho_{10})$")
    plt.legend()
    plt.show()

    # fig, ax = plt.subplots()
    # delta_pops = np.array(top_pops).reshape((50)) - pop_t[0, :]
    # ax.plot(t_list, delta_pops, c="g", label=r"$\Delta\rho_{00}$")
    # ax.plot(t_list, np.array(Re_coh) - np.array(Re_coh_sec), c="b", label=r"$\Delta(\text{Re}(\rho_{10}))$")
    # ax.plot(t_list, np.array(Im_coh) - np.array(Im_coh_sec), c="r", label=r"$\Delta(\text{Im}(\rho_{10}))$")
    # plt.legend()
