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
vib = 20
lam = 0.7*omega

beta_spin = 10
beta_boson = 0.4
spin_coupling = 1e-5
boson_coupling = 1e-3

#del_t_scan
del_t_routine = False
num_tested_deltas = 100
tested_delta = 1.5
tested_delta_only = False
add_int_deltas = True
num_tested_times = 100
delta_range = (0, 3.9*omega)
t_range = (0, 10)
t_type = "log"

#propagation
rho_0_spin = np.zeros((2,2))
rho_0_spin[1,1] = 1
rho_0_boson = np.zeros((vib, vib))
rho_0_boson[0,0] = 1
rho_0_site_basis = np.kron(rho_0_spin, rho_0_boson)

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
if tested_delta_only:
    deltas = [tested_delta]

deltas = np.sort(deltas)
num_tested_deltas = np.size(deltas)

if t_type == "log":
    t_list = np.logspace(t_range[0], t_range[1], num_tested_times)
    # if t_list[0] == 1:
    #     t_list[0] = 0
else:
    t_list = np.linspace(t_range[0], t_range[1], num_tested_times)

spin_down_pops = np.empty((num_tested_deltas, num_tested_times))
purity_data = np.empty((num_tested_deltas, num_tested_times))

average_bi_ee = np.empty((num_tested_deltas, num_tested_times))
single_del_average_bi_ee = np.empty((num_tested_times))
single_del_negativity = np.empty((num_tested_times))
numerical_negativity_data = np.empty((num_tested_deltas, num_tested_times))
single_del_numerical_negativity = np.empty((num_tested_times))
single_del_spin_down_pops = np.empty((num_tested_times))
single_del_purity = np.empty((num_tested_times))

d_count = -1
for d in range(num_tested_deltas):
    delta = deltas[d]
    d_count += 1
    if sparse_mode == "dense":
        Ham = RabiHamiltonian.RabiHamiltonian(vib, delta, lam, omega)
        evals, evecs = scipy.linalg.eigh(Ham)

        spin_bath_op = np.kron(RabiHamiltonian.sig_x(), np.eye(vib))
        boson_bath_op = np.kron(np.eye(2), HolsteinHamiltonian.a(vib) + HolsteinHamiltonian.a_dag(vib)) #Watch out for factors of \sqrt{2} in this

        spin_bath_op_energy_eigenbasis = np.transpose(evecs.conj())@spin_bath_op@evecs
        boson_bath_op_energy_eigenbasis = np.transpose(evecs.conj())@boson_bath_op@evecs
        rho_0_energy_eigenbasis = np.transpose(evecs.conj())@rho_0_site_basis@evecs
        pops_0, coh_0 = Full_Redfield.convert_rho_to_p_and_coh(rho_0_energy_eigenbasis)
        rho_0_super = supervector.supervector(rho_0_energy_eigenbasis)

        bath_coupling_list = [spin_bath_op_energy_eigenbasis, boson_bath_op_energy_eigenbasis]

        spin_cutoff = delta
        boson_cutoff = omega

        def J_spin(x): #Define these to be antisymmetric
            return (spin_coupling)**2*x*math.exp(-1*abs(x)/spin_cutoff)

        def J_boson(x): #Define these to be antisymmetric
            return (boson_coupling)**2*x*math.exp(-1*abs(x)/boson_cutoff)

        spec_density_list = [J_spin, J_boson]

        if sec_mode == "non-sec":
            D_super = Full_Redfield.get_relaxation_derivative(bath_coupling_list, evals, spec_density_list, [beta_spin, beta_boson])

            t_count = -1
            for t in t_list:
                t_count += 1
                rho_t_ee = supervector.unsup((scipy.linalg.expm(D_super*t)@rho_0_super).reshape((2*vib)**2, 1)) #I think this gives the density operator in the energy eigenbasis; need to convert back to the site basis
                rho_t_site = evecs@rho_t_ee@np.transpose(evecs.conj())
                if not tested_delta_only:
                    spin_down_pops[d_count, t_count] = Qobj(rho_t_site, dims=[[2, vib], [2, vib]]).ptrace(0).full()[0, 0]
                    purity_data[d_count, t_count] = np.trace(rho_t_site @ rho_t_site)
                    numerical_negativity_data[d_count, t_count] = p3_ppt.negativity_all_subdivisions(rho_t_site, [2, vib], 1e-10, return_negativity_sum=True)[1]
                if delta == tested_delta or tested_delta_only:
                    # single_del_average_bi_ee[t_count] = entanglement_entropy.get_average_mixed_bipartite_ee(full_rho_t, [2,vib], 0)
                    single_del_negativity[t_count], single_del_numerical_negativity[t_count]  = p3_ppt.negativity_all_subdivisions(rho_t_site, [2,vib], 1e-10, return_negativity_sum = True)
                    single_del_spin_down_pops[t_count] = Qobj(rho_t_site, dims=[[2, vib], [2, vib]]).ptrace(0).full()[0, 0]
                    single_del_purity[t_count] = np.trace(rho_t_site@rho_t_site)

                print(d,t)

        if sec_mode == "sec":
            print(d)
            pop_derivative, coh_derivatives = Full_Redfield.get_secular_t_dep(bath_coupling_list, evals, spec_density_list, [beta_spin, beta_boson])
            rho_t = Full_Redfield.full_sec_prop(pop_derivative, coh_derivatives, t_list, pops_0, coh_0)
            t_count = -1
            for t in t_list:
                t_count += 1
                full_rho_t = evecs@rho_t[t_count]@np.transpose(evecs.conj())
                if not tested_delta_only:
                    spin_down_pops[d_count, t_count] = Qobj(full_rho_t, dims = [[2,vib], [2,vib]]).ptrace(0).full()[0,0]
                    purity_data[d_count, t_count] = np.trace(full_rho_t@full_rho_t)
                    numerical_negativity_data[d_count, t_count] = p3_ppt.negativity_all_subdivisions(full_rho_t, [2,vib], 1e-10, return_negativity_sum = True)[1]
                # average_bi_ee[d_count, t_count] = entanglement_entropy.get_average_mixed_bipartite_ee(full_rho_t, [2,vib], 0)
                if delta == tested_delta or tested_delta_only:
                    # single_del_average_bi_ee[t_count] = entanglement_entropy.get_average_mixed_bipartite_ee(full_rho_t, [2,vib], 0)
                    single_del_negativity[t_count], single_del_numerical_negativity[t_count]  = p3_ppt.negativity_all_subdivisions(full_rho_t, [2,vib], 1e-10, return_negativity_sum = True)
                    single_del_spin_down_pops[t_count] = Qobj(full_rho_t, dims=[[2, vib], [2, vib]]).ptrace(0).full()[0, 0]
                    single_del_purity[t_count] = np.trace(full_rho_t@full_rho_t)

                    print(t_count)


file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_lam={lam}")
extend = 0
while os.path.exists(file_name + ".npy"):
    extend += 1
    file_name = os.path.join("Rabi_Data", f"{datetime.today().strftime('%Y-%m-%d')}_vib={vib}_lam={lam}_{extend}")

np.save(file_name, np.empty(1), allow_pickle=True)

if not tested_delta_only:
    fig, ax = plt.subplots()
    f = ax.pcolormesh(t_list, deltas, spin_down_pops)
    ax.set_xscale(t_type)
    fig.colorbar(f)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\Delta$")
    plt.title(r"Bottom Spin Population, $\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}, "r"$\lambda$ = " + f"{lam}")
    plt.savefig(f"{file_name}_bottom_site_prop".replace(".",""), bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    f = ax.pcolormesh(t_list, deltas, purity_data)
    ax.set_xscale(t_type)
    fig.colorbar(f)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\Delta$")
    plt.title(r"Purity, $\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}, "r"$\lambda$ = " + f"{lam}")
    plt.savefig(f"{file_name}_purity".replace(".",""), bbox_inches='tight')
    plt.show()

    # fig, ax = plt.subplots()
    # f = ax.pcolormesh(t_list, deltas, average_bi_ee)
    # ax.set_xscale(t_type)
    # fig.colorbar(f)
    # plt.xlabel(r"$t$")
    # plt.ylabel(r"$\Delta$")
    # plt.title("(Approximate) Bipartite Entanglement of Formation")
    # plt.savefig(f"{file_name}_del_t_ee", bbox_inches='tight')
    # plt.show()

    fig, ax = plt.subplots()
    f = ax.pcolormesh(t_list, deltas, numerical_negativity_data)
    ax.set_xscale(t_type)
    fig.colorbar(f)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\Delta$")
    plt.title(r"Negativity, $\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}, "r"$\lambda$ = " + f"{lam}")
    plt.savefig(f"{file_name}_del_t_negativity".replace(".",""), bbox_inches='tight')
    plt.show()

if tested_delta in deltas or tested_delta_only:
    # fig, ax = plt.subplots(1)
    # c_ee = ["r" if a else "g" for a in single_del_negativity]
    # lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(t_list[:-1], single_del_average_bi_ee[:-1], t_list[1:], single_del_average_bi_ee[1:])]
    # colored_lines = LineCollection(lines, colors=c_ee, linewidths=(2,))
    # ax.add_collection(colored_lines)
    # plt.axhline(math.log2(2), label = "log2(2) = 1 (Maximum Pure State Entanglement Entropy)")
    # ax.autoscale_view()
    # plt.plot([], [], ' ', c = "r", label= "Red: Negativity Detects Entanglement")
    # plt.plot([], [], ' ', c = "g", label= "Green: Negativity Does Not Detect Entanglement")
    # plt.title(f"Open System Propagation; $\Delta$ = {tested_delta}, $\lambda$ = {lam}")
    # plt.xlabel(f"t")
    # plt.legend()
    # if t_type == "log":
    #     plt.xscale("log")
    # plt.savefig(f"{file_name}_open_sys_single_del_ee_and_negativity.png", bbox_inches='tight', dpi = 300)
    # plt.show()

    fig, ax = plt.subplots()
    f = plt.plot(t_list, single_del_spin_down_pops)
    ax.set_xscale(t_type)
    plt.xlabel(r"$t$")
    plt.ylabel("Population")
    plt.title(r"Spin Down Population; $\Delta$ = " + f"{tested_delta}, " + r"$\lambda$ = " + f"{lam}, " r"$\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}, "r"$\lambda$ = " + f"{lam}")
    plt.savefig(f"{file_name}_single_del_spin_down_pops_del={tested_delta}".replace(".",""), bbox_inches='tight', dpi = 300)
    plt.show()

    fig, ax = plt.subplots()
    f = plt.plot(t_list, single_del_purity)
    ax.set_xscale(t_type)
    plt.xlabel(r"$t$")
    plt.ylabel("Purity")
    plt.title(r"Purity; $\Delta$ = " + f"{tested_delta}, " + r"$\lambda$ = " + f"{lam}, " r"$\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}, "r"$\lambda$ = " + f"{lam}")
    plt.savefig(f"{file_name}_single_del_purity_del={tested_delta}".replace(".",""), bbox_inches='tight', dpi = 300)
    plt.show()

    fig, ax = plt.subplots()
    f = plt.plot(t_list, single_del_numerical_negativity)
    ax.set_xscale(t_type)
    plt.xlabel(r"$t$")
    plt.ylabel("Negativity")
    plt.title(r"Negativity; $\Delta$ = " + f"{tested_delta}, " + r"$\lambda$ = " + f"{lam}, " r"$\beta_s$ = " + f"{beta_spin}, " + r"$\beta_b$ = " + f"{beta_boson}, "r"$\lambda$ = " + f"{lam}")
    plt.savefig(f"{file_name}_single_del_negativity_del={tested_delta}".replace(".",""), bbox_inches='tight', dpi = 300)
    plt.show()

