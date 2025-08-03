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
from matplotlib.collections import LineCollection


#Run Settings
sec_mode = "sec"
sparse_mode = "dense"

#System Properties
omega = 1
vib = 10
lam = 0.7*omega
delta = 3
spin_coupling = 1e-5
boson_coupling = 1e-3


beta_spin_range = (-5, 5)
beta_boson_range = (-5, 5)
interval_type_s = "log"
interval_type_b = "log"
num_beta_s = 25
num_beta_b = 25
tested_beta_b = 1e-3
tested_beta_b_only = False

if interval_type_s == "log":
    beta_s_list = np.logspace(beta_spin_range[0], beta_spin_range[1], num_beta_s)
else:
    beta_s_list = np.linspace(beta_spin_range[0], beta_spin_range[1], num_beta_s)
if interval_type_b == "log":
    beta_b_list = np.logspace(beta_boson_range[0], beta_boson_range[1], num_beta_b)
else:
    beta_b_list = np.linspace(beta_boson_range[0], beta_boson_range[1], num_beta_b)
if tested_beta_b_only:
    beta_b_list = [tested_beta_b]
    num_beta_b = 1

spin_down_pops = np.empty((num_beta_s, num_beta_b))
purity_data = np.empty((num_beta_s, num_beta_b))
numerical_negativity_data = np.empty((num_beta_s, num_beta_b))
single_bb_spin_down_pops = np.empty((num_beta_s))
single_bb_purity_data = np.empty((num_beta_s))
single_bb_numerical_negativity_data = np.empty((num_beta_s))

if sparse_mode == "dense":
    Ham = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega)
    evals, evecs = scipy.linalg.eigh(Ham)

    spin_bath_op = np.kron(RabiHamiltonian.sig_x(), np.eye(vib))
    boson_bath_op = np.kron(np.eye(2), HolsteinHamiltonian.a(vib) + HolsteinHamiltonian.a_dag(vib))  # Watch out for factors of \sqrt{2} in this
    spin_bath_op_energy_eigenbasis = np.transpose(evecs.conj()) @ spin_bath_op @ evecs
    boson_bath_op_energy_eigenbasis = np.transpose(evecs.conj()) @ boson_bath_op @ evecs
    bath_coupling_list = [spin_bath_op_energy_eigenbasis, boson_bath_op_energy_eigenbasis]
    spin_cutoff = delta
    boson_cutoff = omega
    def J_spin(x):  # Define these to be antisymmetric
        return (spin_coupling)**2 * x * math.exp(-1 * abs(x) / spin_cutoff)
    def J_boson(x):  # Define these to be antisymmetric
        return (boson_coupling)**2 * x * math.exp(-1 * abs(x) / boson_cutoff)
    spec_density_list = [J_spin, J_boson]

bs_count = -1
for beta_s in range(num_beta_s):
    bs_count += 1
    beta_s = beta_s_list[bs_count]
    bb_count = -1
    for beta_b in range(num_beta_b):
        bb_count += 1
        beta_b = beta_b_list[bb_count]
        print(bs_count, bb_count)

        if sec_mode == "non-sec":
            D_super = Full_Redfield.get_relaxation_derivative(bath_coupling_list, evals, spec_density_list, [beta_s, beta_b])


            rho_ss_ee = supervector.unsup(scipy.linalg.null_space(D_super).reshape((2*vib)**2, 1)) #I think this gives the density operator in the energy eigenbasis; need to convert back to the site basis
            rho_ss_ee = rho_ss_ee/np.trace(rho_ss_ee)
            rho_ss_site = evecs@rho_ss_ee@np.transpose(evecs.conj())



        if sec_mode == "sec":
            pop_derivative, coh_derivatives = Full_Redfield.get_secular_t_dep(bath_coupling_list, evals, spec_density_list, [beta_s, beta_b])
            unnormalized_pops = scipy.linalg.null_space(pop_derivative).reshape(2*vib, 1)
            pops = unnormalized_pops/np.sum(unnormalized_pops)
            rho_ss_ee = np.diag(pops.reshape(2*vib))
            for j in coh_derivatives:
                if np.real(coh_derivatives[j]) >= 0:
                    raise Exception("Not All Coherences Decay")
            rho_ss_site = evecs@rho_ss_ee@np.transpose(evecs.conj())


        spin_down_pops[bs_count, bb_count] = Qobj(rho_ss_site, dims=[[2, vib], [2, vib]]).ptrace(0).full()[0, 0]
        purity_data[bs_count, bb_count] = np.trace(rho_ss_site @ rho_ss_site)
        numerical_negativity_data[bs_count, bb_count] = p3_ppt.negativity_all_subdivisions(rho_ss_site, [2, vib], 1e-10, return_negativity_sum=True)[1]
        if beta_b == tested_beta_b or tested_beta_b_only:
            single_bb_spin_down_pops[bs_count] = spin_down_pops[bs_count, bb_count]
            single_bb_purity_data[bs_count] = purity_data[bs_count, bb_count]
            single_bb_numerical_negativity_data[bs_count] = numerical_negativity_data[bs_count, bb_count]


file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_Delta={delta}_lambda={lam}".replace(".",""))
extend = 0
while os.path.exists(file_name + ".npy"):
    extend += 1
    file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_Delta={delta}_lambda={lam}_{extend}".replace(".",""))

np.save(file_name, np.empty(1), allow_pickle=True)

if not tested_beta_b_only:
    fig, ax = plt.subplots()
    f = ax.pcolormesh(beta_b_list, beta_s_list, spin_down_pops)
    ax.set_xscale(interval_type_b)
    ax.set_yscale(interval_type_s)
    fig.colorbar(f)
    plt.xlabel(r"$\beta_b$")
    plt.ylabel(r"$\beta_s$")
    plt.title(r"Bottom Spin Steady State Population, $\Delta$ = " + f"{delta}, " + r"$\lambda$ = " + f"{lam}")
    plt.savefig(f"{file_name}_steady_state_spin_pop_temp_scan", bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    f = ax.pcolormesh(beta_b_list, beta_s_list, purity_data)
    ax.set_xscale(interval_type_b)
    ax.set_yscale(interval_type_s)
    fig.colorbar(f)
    plt.xlabel(r"$\beta_b$")
    plt.ylabel(r"$\beta_s$")
    plt.title(r"Steady State Purity, $\Delta$ = " + f"{delta}, " + r"$\lambda$ = " + f"{lam}")
    plt.savefig(f"{file_name}_steady_state_purity_temp_scan", bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    f = ax.pcolormesh(beta_b_list, beta_s_list, numerical_negativity_data)
    ax.set_xscale(interval_type_b)
    ax.set_yscale(interval_type_s)
    fig.colorbar(f)
    plt.xlabel(r"$\beta_b$")
    plt.ylabel(r"$\beta_s$")
    plt.title(r"Steady State Negativity, $\Delta$ = " + f"{delta}, " + r"$\lambda$ = " + f"{lam}")
    plt.savefig(f"{file_name}_steady_state_negativity_temp_scan", bbox_inches='tight')
    plt.show()

if tested_beta_b_only or tested_beta_b in beta_b_list:
    fig, ax = plt.subplots()
    plt.plot(beta_s_list, single_bb_spin_down_pops)
    ax.set_xscale(interval_type_s)
    plt.xlabel(r"$\beta_s$")
    plt.ylabel("Population")
    plt.title(r"Bottom Spin Steady State Population, $\Delta$ = " + f"{delta}, " + r"$\lambda$ = " + f"{lam}, "+ r"$\beta_b$ = " + f"{tested_beta_b}")
    plt.savefig(f"{file_name}_steady_state_spin_pop_temp_scan_bb={tested_beta_b}".replace(".",""), bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(beta_s_list, single_bb_purity_data)
    ax.set_xscale(interval_type_s)
    plt.xlabel(r"$\beta_s$")
    plt.ylabel("Purity")
    plt.title(r"Steady State Purity, $\Delta$ = " + f"{delta}, " + r"$\lambda$ = " + f"{lam}, "+ r"$\beta_b$ = " + f"{tested_beta_b}")
    plt.savefig(f"{file_name}_steady_state_purity_temp_scan_bb={tested_beta_b}".replace(".",""), bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(beta_s_list, single_bb_numerical_negativity_data)
    ax.set_xscale(interval_type_s)
    plt.xlabel(r"$\beta_s$")
    plt.ylabel("Negativity")
    plt.title(r"Steady State Negativity, $\Delta$ = " + f"{delta}, " + r"$\lambda$ = " + f"{lam}, " + r"$\beta_b$ = " + f"{tested_beta_b}")
    plt.savefig(f"{file_name}_steady_state_negativity_temp_scan_bb={tested_beta_b}".replace(".",""), bbox_inches='tight')
    plt.show()

