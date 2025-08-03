import numpy as np
import math
import HolsteinHamiltonian, errors
from supervector import left_multiply, right_multiply, left_multiply_sparse, right_multiply_sparse
import scipy
import RabiHamiltonian


#Each bosonic jump operator a separate dissipator

def jump_op_Lindblad(L):
    L_dag = np.conj(L).transpose()
    return left_multiply(L)@right_multiply(L_dag) - 1/2*(left_multiply(L_dag@L) + right_multiply(L_dag@L))

def jump_op_Lindblad_sparse(L):
    """L should be csr so that conjugate works (I think)"""
    L_dag = L.conjugate(copy = False).transpose()
    return left_multiply_sparse(L)@right_multiply_sparse(L_dag) - 1/2*(left_multiply_sparse(L_dag@L) + right_multiply_sparse(L_dag@L))

def von_neumann_super(H):
    return -1j*(left_multiply(H) - right_multiply(H))

def von_neumann_super_sparse(H_sparse):
    return -1j*(left_multiply_sparse(H_sparse, form = "dia") - right_multiply_sparse(H_sparse, form = "dia")) #Running into memory issues; could look for better formatting based on the form of the matrix

    # #Chern's version
    # dims = H_sparse.shape[0]
    # diags = H_sparse.diagonal()
    # diag_list = []
    # for i in diags:
    #     for j in diags:
    #         diag_list.append(j - i)
    # VN = scipy.sparse.diags(diag_list, shape = (dims**2, dims**2))
    # H_off = H_sparse - scipy.sparse.diags(diags, shape = (dims, dims))
    # VN = VN + scipy.sparse.kron(scipy.sparse.eye(dims), H_off) - scipy.sparse.kron(H_off.transpose(), scipy.sparse.eye(dims))
    # return VN



#This should map the lowest-energy site to the highest
def L_site(num, vib, r_s, version):

    # #Old Version
    # L = np.zeros((num*vib**num, num*vib**num))
    # L[((num - 1)*vib**num, 0)] = 1
    # if version == "up":
    #     print("up r_s bias")
    # if version == "down":
    #     L = L.transpose()
    #     print("down r_s bias")
    # L = math.sqrt(r_s)*L

    #New Version
    J_s = np.eye(num-1)
    J_s = np.pad(J_s, ((1,0), (0,1)))
    if version == "up":
        print("up r_s bias (dense)")
    if version == "down":
        J_s = J_s.transpose()
        print("down r_s bias (dense)")
    L = np.kron(J_s, np.eye(vib**num))
    L = math.sqrt(r_s)*L
    # print(f"L_s = {L}")
    return L

def L_site_sparse(num, vib, r_s, version):

    #Old Version not implemented yet

    #New Version:
    rows = range(num - 1)
    columns = range(1, num)
    data = (num - 1) * [1]
    J_s = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (num, num)).tocsr()
    if version == "up":
        print("up r_s bias (sparse)")
    if version == "down":
        J_s = J_s.transpose()
        print("down r_s bias (sparse)")
    L = scipy.sparse.kron(J_s, scipy.sparse.eye(vib**num, format = "csr"), format = "csr")
    L = math.sqrt(r_s)*L
    return L

def L_site_Lindblad(num, vib, r_s, rs_version):
    return jump_op_Lindblad(L_site(num, vib, r_s, rs_version))

def L_site_Lindblad_sparse(num, vib, r_s, rs_version):
    return jump_op_Lindblad_sparse(L_site_sparse(num, vib, r_s, rs_version))

def single_L_boson(position, num, vib, r_b, version):
    L = math.sqrt(r_b) * HolsteinHamiltonian.tensored_ladder_op(position, num, vib, "a")
    if version == "down":
        print("down r_b bias (dense)")
    elif version == "up":
        L = L.transpose()
        print("up r_b bias (dense)")
    else:
        raise errors.Rb_Version_Error
    return L

def single_L_boson_sparse(position, num, vib, r_b, version):
    L = math.sqrt(r_b) * HolsteinHamiltonian.tensored_ladder_op_sparse(position, num, vib, "a")
    if version == "down":
        print("down r_b bias (sparse)")
    elif version == "up":
        L = L.transpose()
        print("up r_b bias (sparse)")
    else:
        raise errors.Rb_Version_Error
    if L.getformat() != "csr":
        L = L.tocsr()
    return L


def total_L_boson_Lindblad(num, vib, r_b, rb_version):
    dim = (num*vib**num)**2
    L_tot = np.zeros((dim, dim))
    for i in range(num):
        index = i + 1
        #Depending on convention may need to divide by sqrt(2) or 2 in the next line
        L_tot = L_tot + jump_op_Lindblad(single_L_boson(index, num, vib, r_b, rb_version)) #Division by 2 because there was a sqrt(2) in the definition of the annihilation operator
    return L_tot

def total_L_boson_Lindblad_sparse(num, vib, r_b, rb_version):
    L_tot = jump_op_Lindblad_sparse(single_L_boson_sparse(1, num, vib, r_b, 1))
    if L_tot.getformat() != "csr":
        L_tot = L_tot.tocsr()
    for i in range(1, num):
        index = i + 1
        L_tot = L_tot + jump_op_Lindblad_sparse(single_L_boson_sparse(index, num, vib, r_b, 1))
        if L_tot.getformat() != "csr":
            L_tot = L_tot.tocsr()
    return L_tot

def Louivillian(omega, J, num, vib, delta, lam, r_s, r_b, rs_version, rb_version):
    Ham = HolsteinHamiltonian.HolsteinHam(omega, J, num, vib, delta, lam)
    return von_neumann_super(Ham) + L_site_Lindblad(num, vib, r_s, rs_version) + total_L_boson_Lindblad(num, vib, r_b, rb_version)

def Liouvillian_sparse(omega, J, num, vib, delta, lam, r_s, r_b, rs_version, rb_version):
    Ham = HolsteinHamiltonian.HolsteinHam_sparse(omega, J, num, vib, delta, lam)
    return von_neumann_super_sparse(Ham) + L_site_Lindblad_sparse(num, vib, r_s, rs_version) + total_L_boson_Lindblad_sparse(num, vib, r_b, rb_version)

def L_list(r_b, r_sd, num, vib, rs_version, rb_version):
    lst = [L_site(num, vib, r_sd, rs_version)]
    for i in range(num):
        lst.append(single_L_boson(i + 1, num, vib, r_b, rb_version))
    return lst

def L_list_sparse(r_b, r_sd, num, vib, rs_version, rb_version):
    lst = [L_site_sparse(num, vib, r_sd, rs_version)]
    for i in range(num):
        lst.append(single_L_boson_sparse(i + 1, num, vib, r_b, rb_version))
    return lst

def L_rho_matrix_product(rho_ss, L):
    L_dag = np.conj(L).transpose()
    return L@rho_ss@L_dag - (1/2)*(L_dag@L@rho_ss - rho_ss@L_dag@L)

def L_rho_matrix_product_sparse(rho_ss, L):
    """Converts rho_ss to sparse if it is not already"""
    L_dag = L.conjugate(copy = False).transpose()
    if scipy.sparse.issparse(rho_ss):
        return L@rho_ss@L_dag - (1/2)*(L_dag@L@rho_ss - rho_ss@L_dag@L)
    else:
        rho_sparse = scipy.sparse.csr_matrix(rho_ss)
        return L@rho_sparse@L_dag - (1/2)*(L_dag@L@rho_sparse - rho_sparse@L_dag@L)


#Holstein with Ground State Lindblads
#Don't think I need FC factors as long as I don't use the polaron transformation

def single_L_boson_ground(position, num, vib, r_b, version):
    L = single_L_boson(position, num, vib, r_b, version)
    L = np.pad(L, [(1,0), (1,0)])
    return L

def single_L_boson_ground_sparse(position, num, vib, r_b, version):
    L_ng = single_L_boson_sparse(position, num, vib, r_b, version).tocoo()
    data = L_ng.data
    rows = list(L_ng.row)
    for i in range(len(rows)):
        rows[i] += 1
    columns = list(L_ng.col)
    for i in range(len(columns)):
        columns[i] += 1
    L = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (num*vib**num + 1, num*vib**num + 1)).tocsr()
    return L


def total_L_boson_ground(num, vib, r_b, rb_version):
    dim = (num*vib**num + 1)**2
    L_tot = np.zeros((dim, dim))
    for i in range(num):
        index = i + 1
        #Depending on convention may need to divide by sqrt(2) or 2 in the next line
        L_tot = L_tot + jump_op_Lindblad(single_L_boson_ground(index, num, vib, r_b, rb_version)) #Division by 2 because there was a sqrt(2) in the definition of the annihilation operator
    return L_tot

def total_L_boson_ground_sparse(num, vib, r_b, rb_version):
    L_tot = jump_op_Lindblad_sparse(single_L_boson_ground_sparse(1, num, vib, r_b, rb_version))
    if L_tot.getformat() != "csr":
        L_tot = L_tot.tocsr()
    for i in range(1, num):
        index = i + 1
        L_tot = L_tot + jump_op_Lindblad_sparse(single_L_boson_ground_sparse(index, num, vib, r_b, rb_version))
    if L_tot.getformat() != "csr":
        L_tot = L_tot.tocsr()
    return L_tot

def single_L_site_down_ground(position, num, vib, rs_down):
    """Positions from 1 to n"""
    L = np.zeros((num*vib**num + 1, num*vib**num + 1))
    # Version 1:
    for i in range(vib**num):
        L[0, 1 + (position - 1)*vib**num + i] = 1

   # # Version 2:
   #  L[0, 1 + (position - 1)*vib**num] = 1


    L = math.sqrt(rs_down)*L
    return L

def single_L_site_down_ground_sparse(position, num, vib, rs_down):
    """Positions from 1 to n"""
    data = (vib**num) * [1]
    rows = (vib**num) * [0]
    columns = []
    for i in range(vib**num):
        columns.append(1 + (position - 1)*vib**num + i)
    L = scipy.sparse.coo_matrix((data, (rows, columns)), shape = (num*vib**num + 1, num*vib**num + 1)).tocsr()
    L = math.sqrt(rs_down)*L
    return L

def total_L_site_down(num, vib, rs_down):
    dims = (num*vib**num + 1)**2
    L_tot = np.zeros((dims, dims))
    for i in range(num):
        L_tot += jump_op_Lindblad(single_L_site_down_ground(i + 1, num, vib, rs_down))
    return L_tot


def total_L_site_down_sparse(num, vib, rs_down):
    L_tot = jump_op_Lindblad_sparse(single_L_site_down_ground_sparse(1, num, vib, rs_down))
    if L_tot.getformat() != "csr":
        L_tot = L_tot.tocsr()
    for i in range(1, num):
        L_tot += jump_op_Lindblad_sparse(single_L_site_down_ground_sparse(i + 1, num, vib, rs_down))
    if L_tot.getformat() != "csr":
        L_tot = L_tot.tocsr()
    return L_tot


def L_ground_to_top(num, vib, rs_up):
    L = np.zeros((num*vib**num + 1, num*vib**num + 1))
    L[1 + (num - 1)*vib**num, 0] = 1
    L = math.sqrt(rs_up)*L
    return L

def L_ground_to_top_sparse(num, vib, rs_up):
    data = [1]
    rows = [1 + (num - 1)*vib**num]
    columns = [0]
    L = scipy.sparse.coo_matrix((data, (rows,columns)), shape = (num*vib**num + 1, num*vib**num + 1))
    L = math.sqrt(rs_up)*L
    return L


#TODO: Convert Pump_To_All and Pump_To_Outer to sparse formats
def L_Pump_To_All(num, vib, rs_up):
    L = np.zeros((num*vib**num + 1, num*vib**num + 1))
    for i in range(num):
        L[1 + i*vib**num, 0] = 1
    L = math.sqrt(rs_up)*L
    return L

def L_Pump_To_All_sparse(num, vib, rs_up):
    rows = np.arange(1, num*vib**num + 1, vib**num)
    col = np.zeros(num*vib**num)
    data = np.ones(num*vib**num)
    L = scipy.sparse.coo_matrix((data, (rows, col)), shape = (num*vib**num + 1, num*vib**num + 1))
    L = math.sqrt(rs_up)*L
    return L

def L_Pump_To_Outer(num, vib, rs_up):
    L = np.zeros((num*vib**num + 1, num*vib**num + 1))
    L[1,0] = 1
    L[1 + (num - 1)*vib**num, 0] = 1
    L = math.sqrt(rs_up)*L
    return L

def L_Pump_To_Outer_sparse(num, vib, rs_up):
    rows = [1, (num - 1)*vib**num + 1]
    col = [0,0]
    data = [1,1]
    L = scipy.sparse.coo_matrix((data, (rows, col)), shape = (num*vib**num + 1, num*vib**num + 1))
    L = math.sqrt(rs_up)*L
    return L

def Liouvillian_ground(num, vib, omega, J, delta, lam, rs_down, rs_up, rb, Lb_version):
    return von_neumann_super(
        HolsteinHamiltonian.HolsteinHamWGround(omega, J, num, vib, delta, lam)) + total_L_boson_ground(num, vib, rb, Lb_version) + total_L_site_down(num, vib, rs_down) + jump_op_Lindblad(L_ground_to_top(num, vib, rs_up))

def Liouvillian_ground_sparse(num, vib, omega, J, delta, lam, rs_down, rs_up, rb, Lb_version):
    Ham = HolsteinHamiltonian.HolsteinHamWGround_sparse(omega, J, num, vib, delta, lam)
    return von_neumann_super_sparse(Ham) + total_L_boson_ground_sparse(num, vib, rb, Lb_version) + total_L_site_down_sparse(num, vib, rs_down) + jump_op_Lindblad_sparse(L_ground_to_top_sparse(num, vib, rs_up))

#TODO: Add sparse versions of these functions
def L_list_ground(num, vib, rs_down, rs_up, rb, rb_version):
    lst = [L_ground_to_top(num, vib, rs_up)]
    for i in range(num):
        lst.append(single_L_site_down_ground(i + 1, num, vib, rs_down))
        lst.append(single_L_boson_ground(i + 1, num, vib, rb, rb_version))
    return lst

def L_list_ground_sparse(num, vib, rs_down, rs_up, rb, rb_version):
    lst = [L_ground_to_top_sparse(num, vib, rs_up)]
    for i in range(num):
        lst.append(single_L_site_down_ground_sparse(i + 1, num, vib, rs_down))
        lst.append(single_L_boson_ground_sparse(i + 1, num, vib, rb, rb_version))
    return lst

def L_list_uniform_pumping(num, vib, rs_down, rs_up, rb, rb_version):
    lst = [L_Pump_To_All(num, vib, rs_up)]
    for i in range(num):
        lst.append(single_L_site_down_ground(i + 1, num, vib, rs_down))
        lst.append(single_L_boson_ground(i + 1, num, vib, rb, rb_version))
    return lst

def L_list_uniform_pumping_sparse(num, vib, rs_down, rs_up, rb, rb_version):
    lst = [L_Pump_To_All_sparse(num, vib, rs_up)]
    for i in range(num):
        lst.append(single_L_site_down_ground_sparse(i + 1, num, vib, rs_down))
        lst.append(single_L_boson_ground_sparse(i + 1, num, vib, rb, rb_version))
    return lst


def L_list_outer_pumping(num, vib, rs_down, rs_up, rb, rb_version):
    lst = [L_Pump_To_Outer(num, vib, rs_up)]
    for i in range(num):
        lst.append(single_L_site_down_ground(i + 1, num, vib, rs_down))
        lst.append(single_L_boson_ground(i + 1, num, vib, rb, rb_version))
    return lst

def L_list_outer_pumping_sparse(num, vib, rs_down, rs_up, rb, rb_version):
    lst = [L_Pump_To_Outer_sparse(num, vib, rs_up)]
    for i in range(num):
        lst.append(single_L_site_down_ground_sparse(i + 1, num, vib, rs_down))
        lst.append(single_L_boson_ground_sparse(i + 1, num, vib, rb, rb_version))
    return lst

def get_ground_fluxes_sec(ee_pops, cob_original_to_eig, L_list):
    #TODO: Put this into the secular approximation code in order to avoid redoing the matrix multiplication
    """Assumes L_list is given in the following order:
    L_list[0] = Pumping
    L_list[2n - 1] = nth Recombination
    L_list[2n] = nth Boson Dissipation
    Where n is an integer > 0"""
    num_l = len(L_list)
    cob_eig_to_original = np.transpose(np.conj(cob_original_to_eig)) #Change of basis should be unitary so take the adjoint as the inverse
    cob_eig_to_original_sparse = scipy.sparse.csr_matrix(cob_eig_to_original)
    cob_original_to_eig_sparse = scipy.sparse.csr_matrix(cob_original_to_eig)
    L_pump_diag = np.diag((cob_original_to_eig_sparse@L_list[0]@cob_eig_to_original_sparse).toarray())
    L_recomb_diag = np.diag((cob_original_to_eig_sparse@L_list[1]@cob_eig_to_original_sparse).toarray())
    L_boson_diag = np.diag((cob_original_to_eig_sparse@L_list[2]@cob_eig_to_original_sparse).toarray())
    for i in range(2, num_l/2 + 1/2):
        L_recomb_diag += np.diag((cob_original_to_eig_sparse@L_list[2*i - 1]@cob_eig_to_original_sparse).toarray())
        L_boson_diag += np.diag((cob_original_to_eig_sparse@L_list[2*i]@cob_eig_to_original_sparse).toarray())
    pump_flux = np.transpose(L_pump_diag)@ee_pops
    recomb_flux = np.transpose(L_recomb_diag)@ee_pops
    boson_flux = np.transpose(L_boson_diag)@ee_pops
    return pump_flux, recomb_flux, boson_flux


################################## Rabi Lindblads ####################################

def Rabi_Ls(vib, rs):
    return math.sqrt(rs)*np.kron(RabiHamiltonian.sig_minus(), np.eye(vib))

def Rabi_Ls_sparse(vib, rs):
    return math.sqrt(rs)*scipy.sparse.kron(RabiHamiltonian.sig_minus_sparse(), scipy.sparse.eye(vib, format ="csr"), format ="csr")

def Rabi_Lb(vib,rb):
    return math.sqrt(rb)*np.kron(np.eye(2), HolsteinHamiltonian.a(vib))

def Rabi_Lb_sparse(vib, rb):
    return math.sqrt(rb)*scipy.sparse.kron(scipy.sparse.eye(2, format = "csr"), HolsteinHamiltonian.a_sparse(vib), format ="csr")

def Rabi_L_list(vib, rb, rs):
    return [Rabi_Ls(vib, rs), Rabi_Lb(vib, rb)]

def Rabi_L_list_sparse(vib, rb, rs):
    return [Rabi_Ls_sparse(vib, rs), Rabi_Lb_sparse(vib, rb)]

def Rabi_Liouvillian(H, vib, rb, rs):
    return von_neumann_super(H) + jump_op_Lindblad(Rabi_Lb(vib, rb)) + jump_op_Lindblad(Rabi_Ls(vib, rs))





# if __name__ == "__main__":
#     A = L_list_ground(2, 3, 1e-6, 1e-7, 1e-8, "down")
#     B = 5

if __name__ == "__main__":
    num = 3
    vib = 10
    omega = 1
    J = 0.01
    delta = 0.5
    lam = 1.5
    rs_down = 1e-6
    rs_up = 1e-9
    rb = 1e-7
    Lb_version = "down"
    # A = Liouvillian_ground(num, vib, omega, J, delta, lam, rs_down, rs_up, rb, Lb_version)
    B = Liouvillian_ground_sparse(num, vib, omega, J, delta, lam, rs_down, rs_up, rb, Lb_version)
    # print(np.allclose(A, B.toarray(), atol = 1e-17))








