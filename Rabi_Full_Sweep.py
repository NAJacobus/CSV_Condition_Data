import Full_Redfield
import numpy as np
import scipy
import RabiHamiltonian
import entanglement_entropy
import supervector, HolsteinHamiltonian
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math
from qutip import Qobj
import p3_ppt
import qutip

#Run Settings
sec_mode = "sec"
sparse_mode = "dense"
JC = False #Note that for JC we still use the sigma x bath coupling
num_k = 4
vib = 10

#Sweep Parameters
#Omega
num_tested_omega = 1
omega_range = (1,1)
#Beta Spin
num_tested_bs = 10
bs_range = (1e-7, 100)
#Beta Boson
num_tested_bb = 10
bb_range = (1e-7,100)
#Spin Coupling
num_spin_coupling = 1
spin_coupling_range = (1e-5, 1e-5)
#Boson Coupling
num_boson_coupling = 1
boson_coupling_range = (1e-3, 1e-3)
#Delta
num_tested_deltas = 25
delta_range = (0.1, 10)
add_int_deltas = False
#Lambda
num_tested_lam = 25
lam_range = (0.1, 2.5)
#Time
num_tested_t = 1
t_range = (0,100)
t_type = "lin" #lin or log

#Initial Conditions
rho_0_spin = np.zeros((2,2))
rho_0_spin[1,1] = 1
rho_0_boson = np.zeros((vib, vib))
rho_0_boson[0,0] = 1
rho_0_site_basis = np.kron(rho_0_spin, rho_0_boson)

omega_list = np.linspace(omega_range[0], omega_range[1], num_tested_omega)
bs_list = np.linspace(bs_range[0], bs_range[1], num_tested_bs)
bb_list = np.linspace(bb_range[0], bb_range[1], num_tested_bb)
sc_list = np.linspace(spin_coupling_range[0], spin_coupling_range[1], num_spin_coupling)
bc_list = np.linspace(boson_coupling_range[0], boson_coupling_range[1], num_boson_coupling)
if t_type == "log":
    t_list = np.logspace(t_range[0], t_range[1], num_tested_t)
    if t_list[0] == 1:
        t_list[0] = 0
elif t_type == "lin":
    t_list = np.linspace(t_range[0], t_range[1], num_tested_t)
deltas = np.linspace(delta_range[0], delta_range[1], num_tested_deltas + 1)[1:]
if add_int_deltas:
    for j in range(1, 2):
        #Find numerators
        counter = 0
        while counter/j < delta_range[0]:
            counter += 1
        first_num = counter
        while counter/j <= delta_range[1]:
            counter += 1
        last_num = counter #Note that this is one more than the last one so that range gives all desired numerators
        for i in range(first_num, last_num):
            if i/j not in deltas:
                deltas = np.append(deltas, i/j)
            if i/j - 0.005 not in deltas and i/j - 0.005 >= delta_range[0]:
                deltas = np.append(deltas, i/j - 0.005)
            if i/j + 0.005 not in deltas and i/j + 0.005 <= delta_range[1]:
                deltas = np.append(deltas, i/j + 0.005)
deltas = np.sort(deltas)
num_tested_deltas = np.size(deltas)
lam_list = np.linspace(lam_range[0], lam_range[1], num_tested_lam)

dim_tuple = (num_tested_omega, num_tested_bs, num_tested_bb, num_spin_coupling, num_boson_coupling, num_tested_deltas, num_tested_lam, num_tested_t)
spin_down_pops = np.empty(dim_tuple)
purity_data = np.empty(dim_tuple)
average_bi_ee = np.empty(dim_tuple)
numerical_negativity_data = np.empty(dim_tuple)
stored_ee_t_rho = np.empty(dim_tuple, dtype = np.ndarray)
stored_site_t_rho = np.empty(dim_tuple, dtype = np.ndarray)
stored_site_t_pt = np.empty(dim_tuple, dtype = np.ndarray)

om_ind = -1
for omega in omega_list:
    om_ind += 1
    del_ind = -1
    for delta in deltas:
        del_ind += 1
        lam_ind = -1
        for lam in lam_list:
            lam_ind += 1

            if sparse_mode == "dense":
                if JC:
                    Ham = RabiHamiltonian.JCHamiltonian(vib, delta, lam, omega)
                else:
                    Ham = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega)
                evals, evecs = scipy.linalg.eigh(Ham)

            if sparse_mode == "sparse":
                if JC:
                    Ham = RabiHamiltonian.JCHamiltonian_sparse(vib, delta, lam, omega)
                else:
                    Ham = RabiHamiltonian.RabiHamiltonian_sparse(vib, delta, lam, omega)
                evals, evecs = scipy.sparse.linalg.eigs(Ham, k=num_k, which="SR")

            rho_0_energy_eigenbasis = np.transpose(evecs.conj()) @ rho_0_site_basis @ evecs
            if sec_mode == "sec":
                pops_0, coh_0 = Full_Redfield.convert_rho_to_p_and_coh(rho_0_energy_eigenbasis)
            if sec_mode == "non-sec":
                rho_0_super = supervector.supervector(rho_0_energy_eigenbasis) #Within the sparse calculation, we should ensure that the eigenspace truncation does not destroy the positivity of rho_0

            spin_bath_op = np.kron(RabiHamiltonian.sig_x(), np.eye(vib))
            boson_bath_op = np.kron(np.eye(2), HolsteinHamiltonian.a(vib) + HolsteinHamiltonian.a_dag(vib)) #Watch out for factors of \sqrt{2} in this

            spin_bath_op_energy_eigenbasis = np.transpose(evecs.conj())@spin_bath_op@evecs
            boson_bath_op_energy_eigenbasis = np.transpose(evecs.conj())@boson_bath_op@evecs

            bath_coupling_list = [spin_bath_op_energy_eigenbasis, boson_bath_op_energy_eigenbasis]

            spin_cutoff = delta
            boson_cutoff = omega

            sc_ind = -1
            for spin_coupling in sc_list:
                sc_ind += 1
                bc_ind = -1
                for boson_coupling in bc_list:
                    bc_ind += 1
                    def J_spin(x): #Define these to be antisymmetric
                        return (spin_coupling)**2*x*math.exp(-1*abs(x)/spin_cutoff)

                    def J_boson(x): #Define these to be antisymmetric
                        return (boson_coupling)**2*x*math.exp(-1*abs(x)/boson_cutoff)

                    spec_density_list = [J_spin, J_boson]

                    bs_ind = -1
                    for beta_spin in bs_list:
                        bs_ind += 1
                        bb_ind = -1
                        for beta_boson in bb_list:
                            bb_ind += 1
                            if sec_mode == "non-sec":
                                D_super = Full_Redfield.get_relaxation_derivative(bath_coupling_list, evals, spec_density_list, [beta_spin, beta_boson])
                                t_ind = -1
                                for t in t_list:
                                    t_ind += 1
                                    if sparse_mode == "dense":
                                        rho_t = supervector.unsup((scipy.linalg.expm(D_super*t)@rho_0_super).reshape((2*vib)**2, 1)) #I think this gives the density operator in the energy eigenbasis; need to convert back to the site basis
                                    else:
                                        rho_t = supervector.unsup((scipy.linalg.expm(D_super*t)@rho_0_super).reshape(num_k**2, 1)) #I think this gives the density operator in the energy eigenbasis; need to convert back to the site basis
                                    rho_t = rho_t/np.trace(rho_t)
                                    index_tuple = (om_ind, bs_ind, bb_ind, sc_ind, bc_ind, del_ind, lam_ind, t_ind)
                                    print(index_tuple)
                                    rho_t_site = evecs @ rho_t @ np.transpose(evecs.conj())
                                    pt = qutip.partial_transpose(qutip.Qobj(rho_t_site, dims=[[2, vib], [2, vib]]),[1, 0]).full()
                                    spin_down_pops[*index_tuple] = Qobj(rho_t_site, dims=[[2, vib], [2, vib]]).ptrace(0).full()[0, 0]
                                    purity_data[*index_tuple] = np.trace(rho_t_site @ rho_t_site)
                                    numerical_negativity_data[*index_tuple] = p3_ppt.negativity_all_subdivisions(rho_t_site, [2, vib], 1e-8, return_negativity_sum=True)[1]
                                    stored_ee_t_rho[*index_tuple] = rho_t
                                    stored_site_t_rho[*index_tuple] = rho_t_site
                                    stored_site_t_pt[*index_tuple] = pt

                            if sec_mode == "sec":
                                pop_derivative, coh_derivatives = Full_Redfield.get_secular_t_dep(bath_coupling_list, evals, spec_density_list, [beta_spin, beta_boson])
                                t_ind = -1
                                for t in t_list:
                                    t_ind += 1
                                    rho_t = Full_Redfield.full_sec_prop(pop_derivative, coh_derivatives, [t], pops_0, coh_0)[0]
                                    rho_t = rho_t/np.trace(rho_t)
                                    index_tuple = (om_ind, bs_ind, bb_ind, sc_ind, bc_ind, del_ind, lam_ind, t_ind)
                                    print(index_tuple)
                                    rho_t_site = evecs@rho_t@np.transpose(evecs.conj())
                                    pt = qutip.partial_transpose(qutip.Qobj(rho_t_site, dims=[[2, vib], [2, vib]]), [1, 0]).full()
                                    spin_down_pops[*index_tuple] = Qobj(rho_t_site, dims = [[2,vib], [2,vib]]).ptrace(0).full()[0,0]
                                    purity_data[*index_tuple] = np.trace(rho_t_site@rho_t_site)
                                    numerical_negativity_data[*index_tuple] = p3_ppt.negativity_all_subdivisions(rho_t_site, [2,vib], 1e-8, return_negativity_sum = True)[1]
                                    stored_ee_t_rho[*index_tuple] = rho_t
                                    stored_site_t_rho[*index_tuple] = rho_t_site
                                    stored_site_t_pt[*index_tuple] = pt
if JC:
    model = "JC"
else:
    model = "Rabi"

file_name = os.path.join("Rabi_Full_Sweep_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_{model}_{sparse_mode}")
extend = 0
while os.path.exists(file_name + ".npz"):
    extend += 1
    file_name = os.path.join("Rabi_Full_Sweep_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_{model}_{sparse_mode}_{extend}")

np.savez(file_name, model = np.array([model]), spin_down_pops = spin_down_pops, purity_data = purity_data, numerical_negativity_data = numerical_negativity_data, stored_ee_t_rho = stored_ee_t_rho, stored_site_t_rho = stored_site_t_rho, stored_site_t_pt = stored_site_t_pt, omega_list = omega_list, bs_list = bs_list, bb_list = bb_list, sc_list = sc_list, bc_list = bc_list, deltas = deltas, lam_list = lam_list, t_list = t_list, allow_pickle=True)
